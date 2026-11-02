#if defined(ANALYSIS) && defined(LYA_STATISTICS) && defined(GPU_LYA_FLUX)

  #include <cmath>
  #include <cstdio>

  #include "../analysis/analysis.h"
  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../utils/gpu.hpp"

  #define TPB_LYA_FLUX 256

// Maximum number of blocks along the y-dimension of a grid. Skewers beyond
// this are processed with a grid-stride loop.
  #define MAX_GRID_Y_LYA_FLUX 65535

/*!
 * \brief Compute the HI and HeII transmitted flux along a set of skewers.
 *
 * Each thread computes the optical depth of one real cell of a skewer by
 * integrating the Voigt (Doppler) profile contributions of every cell of the
 * periodically extended skewer (real cells plus n_ghost cells on each side).
 * The extended skewer is never materialized: ghost cells are read from the
 * real skewer data using periodic indexing. Data of the extended skewer is
 * streamed through shared memory in tiles of blockDim.x cells.
 *
 * The summation order over the extended skewer matches the CPU implementation
 * in Grid3D::Compute_Transmitted_Flux_Skewer.
 */
__global__ void Compute_Transmitted_Flux_Skewers_kernel(int n_skewers, int n_los, int n_ghost, Real dens_factor_HI,
                                                        Real dens_factor_HeII, Real vel_factor, Real temp_factor_HI,
                                                        Real temp_factor_HeII, Real dv_Hubble, Real Lya_sigma_HI,
                                                        Real Lya_sigma_HeII, const Real *skewers_HI_density,
                                                        const Real *skewers_HeII_density, const Real *skewers_velocity,
                                                        const Real *skewers_temperature,
                                                        Real *skewers_transmitted_flux_HI,
                                                        Real *skewers_transmitted_flux_HeII)
{
  __shared__ Real sh_n_HI[TPB_LYA_FLUX];
  __shared__ Real sh_n_HeII[TPB_LYA_FLUX];
  __shared__ Real sh_vel[TPB_LYA_FLUX];
  __shared__ Real sh_b_HI[TPB_LYA_FLUX];
  __shared__ Real sh_b_HeII[TPB_LYA_FLUX];

  int const tid        = threadIdx.x;
  int const los_id     = blockIdx.x * blockDim.x + tid;
  int const n_los_full = n_los + 2 * n_ghost;
  bool const active    = los_id < n_los;

  // Hubble velocity at the center of this (real) cell
  Real const vel_i = (los_id + 0.5) * dv_Hubble;

  // All threads of a block share the same skewer_id, so the __syncthreads()
  // calls below are reached uniformly by the whole block.
  for (int skewer_id = blockIdx.y; skewer_id < n_skewers; skewer_id += gridDim.y) {
    size_t const skewer_offset = static_cast<size_t>(skewer_id) * n_los;

    Real tau_HI   = 0;
    Real tau_HeII = 0;

    for (int tile_start = 0; tile_start < n_los_full; tile_start += blockDim.x) {
      int const j = tile_start + tid;
      if (j < n_los_full) {
        int const src_id       = (((j - n_ghost) % n_los) + n_los) % n_los;
        size_t const id        = skewer_offset + src_id;
        Real const temperature = skewers_temperature[id];
        sh_n_HI[tid]           = skewers_HI_density[id] * dens_factor_HI;
        sh_n_HeII[tid]         = skewers_HeII_density[id] * dens_factor_HeII;
        sh_vel[tid]            = (j - n_ghost + 0.5) * dv_Hubble + (skewers_velocity[id] * vel_factor);
        sh_b_HI[tid]           = sqrt(temp_factor_HI * temperature);
        sh_b_HeII[tid]         = sqrt(temp_factor_HeII * temperature);
      }
      __syncthreads();

      if (active) {
        int const n_tile = min(static_cast<int>(blockDim.x), n_los_full - tile_start);
        for (int k = 0; k < n_tile; k++) {
          Real const vel_j = sh_vel[k];
          Real y_l         = (vel_i - 0.5 * dv_Hubble - vel_j) / sh_b_HI[k];
          Real y_r         = (vel_i + 0.5 * dv_Hubble - vel_j) / sh_b_HI[k];
          tau_HI += sh_n_HI[k] * (erf(y_r) - erf(y_l)) / 2;
          y_l = (vel_i - 0.5 * dv_Hubble - vel_j) / sh_b_HeII[k];
          y_r = (vel_i + 0.5 * dv_Hubble - vel_j) / sh_b_HeII[k];
          tau_HeII += sh_n_HeII[k] * (erf(y_r) - erf(y_l)) / 2;
        }
      }
      __syncthreads();
    }

    if (active) {
      tau_HI *= Lya_sigma_HI;
      tau_HeII *= Lya_sigma_HeII;
      skewers_transmitted_flux_HI[skewer_offset + los_id]   = exp(-tau_HI);
      skewers_transmitted_flux_HeII[skewer_offset + los_id] = exp(-tau_HeII);
    }
  }
}

void Grid3D::Compute_Transmitted_Flux_Skewers_GPU(int axis)
{
  int n_skewers, n_los_total, n_ghost;
  bool am_I_root;
  Real Lbox, delta_x;
  Real *skewers_HI_density_root;
  Real *skewers_HeII_density_root;
  Real *skewers_velocity_root;
  Real *skewers_temperature_root;
  Real *skewers_transmitted_flux_HI;
  Real *skewers_transmitted_flux_HeII;
  Real *dev_skewers_HI_density;
  Real *dev_skewers_HeII_density;
  Real *dev_skewers_velocity;
  Real *dev_skewers_temperature;
  Real *dev_skewers_transmitted_flux_HI;
  Real *dev_skewers_transmitted_flux_HeII;

  n_ghost = Analysis.n_ghost_skewer;

  if (axis == 0) {
    Lbox                              = Analysis.Lbox_x;
    delta_x                           = Analysis.dx;
    am_I_root                         = Analysis.am_I_root_x;
    n_skewers                         = Analysis.n_skewers_local_x;
    n_los_total                       = Analysis.nx_total;
    skewers_HI_density_root           = Analysis.skewers_HI_density_root_x;
    skewers_HeII_density_root         = Analysis.skewers_HeII_density_root_x;
    skewers_velocity_root             = Analysis.skewers_velocity_root_x;
    skewers_temperature_root          = Analysis.skewers_temperature_root_x;
    skewers_transmitted_flux_HI       = Analysis.skewers_transmitted_flux_HI_x;
    skewers_transmitted_flux_HeII     = Analysis.skewers_transmitted_flux_HeII_x;
    dev_skewers_HI_density            = Analysis.dev_skewers_HI_density_x;
    dev_skewers_HeII_density          = Analysis.dev_skewers_HeII_density_x;
    dev_skewers_velocity              = Analysis.dev_skewers_velocity_x;
    dev_skewers_temperature           = Analysis.dev_skewers_temperature_x;
    dev_skewers_transmitted_flux_HI   = Analysis.dev_skewers_transmitted_flux_HI_x;
    dev_skewers_transmitted_flux_HeII = Analysis.dev_skewers_transmitted_flux_HeII_x;
  }

  if (axis == 1) {
    Lbox                              = Analysis.Lbox_y;
    delta_x                           = Analysis.dy;
    am_I_root                         = Analysis.am_I_root_y;
    n_skewers                         = Analysis.n_skewers_local_y;
    n_los_total                       = Analysis.ny_total;
    skewers_HI_density_root           = Analysis.skewers_HI_density_root_y;
    skewers_HeII_density_root         = Analysis.skewers_HeII_density_root_y;
    skewers_velocity_root             = Analysis.skewers_velocity_root_y;
    skewers_temperature_root          = Analysis.skewers_temperature_root_y;
    skewers_transmitted_flux_HI       = Analysis.skewers_transmitted_flux_HI_y;
    skewers_transmitted_flux_HeII     = Analysis.skewers_transmitted_flux_HeII_y;
    dev_skewers_HI_density            = Analysis.dev_skewers_HI_density_y;
    dev_skewers_HeII_density          = Analysis.dev_skewers_HeII_density_y;
    dev_skewers_velocity              = Analysis.dev_skewers_velocity_y;
    dev_skewers_temperature           = Analysis.dev_skewers_temperature_y;
    dev_skewers_transmitted_flux_HI   = Analysis.dev_skewers_transmitted_flux_HI_y;
    dev_skewers_transmitted_flux_HeII = Analysis.dev_skewers_transmitted_flux_HeII_y;
  }

  if (axis == 2) {
    Lbox                              = Analysis.Lbox_z;
    delta_x                           = Analysis.dz;
    am_I_root                         = Analysis.am_I_root_z;
    n_skewers                         = Analysis.n_skewers_local_z;
    n_los_total                       = Analysis.nz_total;
    skewers_HI_density_root           = Analysis.skewers_HI_density_root_z;
    skewers_HeII_density_root         = Analysis.skewers_HeII_density_root_z;
    skewers_velocity_root             = Analysis.skewers_velocity_root_z;
    skewers_temperature_root          = Analysis.skewers_temperature_root_z;
    skewers_transmitted_flux_HI       = Analysis.skewers_transmitted_flux_HI_z;
    skewers_transmitted_flux_HeII     = Analysis.skewers_transmitted_flux_HeII_z;
    dev_skewers_HI_density            = Analysis.dev_skewers_HI_density_z;
    dev_skewers_HeII_density          = Analysis.dev_skewers_HeII_density_z;
    dev_skewers_velocity              = Analysis.dev_skewers_velocity_z;
    dev_skewers_temperature           = Analysis.dev_skewers_temperature_z;
    dev_skewers_transmitted_flux_HI   = Analysis.dev_skewers_transmitted_flux_HI_z;
    dev_skewers_transmitted_flux_HeII = Analysis.dev_skewers_transmitted_flux_HeII_z;
  }

  if (!am_I_root) return;
  if (n_skewers == 0) return;

  std::cout << "computing transmitted flux skewers on GPU  for axis: " << axis << " with n_skewers: " << n_skewers << std::endl;

  // Constants in CGS
  Real Msun, kpc, Mp, kpc3, Me, e_charge, c, Kb;
  Kb       = 1.38064852e-16;  // g (cm/s)^2 K-1
  Msun     = 1.98847e33;      // g
  Mp       = 1.6726219e-24;   // g
  Me       = 9.10938356e-28;  // g
  c        = 2.99792458e10;   // cm/s
  kpc      = 3.0857e21;       // cm
  kpc3     = kpc * kpc * kpc;
  e_charge = 4.8032e-10;  // cm^3/2 g^1/2 s^-1

  // Conversion from comoving density to physical number density in units 1/cm^3
  Real dens_factor, dens_factor_HI, dens_factor_HeII, vel_factor;
  dens_factor      = 1. / (Cosmo.current_a * Cosmo.current_a * Cosmo.current_a) * Cosmo.cosmo_h * Cosmo.cosmo_h;
  dens_factor_HI   = dens_factor * Msun / (kpc3) / Mp;
  dens_factor_HeII = dens_factor * Msun / (kpc3) / (4 * Mp);
  vel_factor       = 1e5;  // cm/s

  // Get Cosmological variables
  Real H, current_a, dx_proper, dv_Hubble;
  Real H_cgs, Lya_lambda_HI, f_12, Lya_sigma_HI;
  Real Lya_lambda_HeII, Lya_sigma_HeII;
  current_a = Cosmo.current_a;
  dx_proper = delta_x * current_a / Cosmo.cosmo_h;
  H         = Cosmo.Get_Hubble_Parameter(current_a);
  dv_Hubble = H * dx_proper * vel_factor;  // cm/s

  Lya_lambda_HI   = 1.21567e-5;          // cm  Rest wave length of the Lyman Alpha Transition Hydrogen
  Lya_lambda_HeII = Lya_lambda_HI / 4;   // cm  Rest wave length of the Lyman Alpha Transition Helium II
  f_12            = 0.416;               // Lya transition Oscillator strength
  H_cgs           = H * 1e5 / kpc;

  Lya_sigma_HI   = M_PI * e_charge * e_charge / Me / c * Lya_lambda_HI * f_12 / H_cgs;
  Lya_sigma_HeII = M_PI * e_charge * e_charge / Me / c * Lya_lambda_HeII * f_12 / H_cgs;

  // Doppler parameter b = sqrt( temp_factor * T )
  Real temp_factor_HI   = 2 * Kb / Mp;
  Real temp_factor_HeII = 2 * Kb / (4 * Mp);

  size_t const field_size = static_cast<size_t>(n_skewers) * n_los_total * sizeof(Real);
  GPU_Error_Check(cudaMemcpy(dev_skewers_HI_density, skewers_HI_density_root, field_size, cudaMemcpyHostToDevice));
  GPU_Error_Check(
      cudaMemcpy(dev_skewers_HeII_density, skewers_HeII_density_root, field_size, cudaMemcpyHostToDevice));
  GPU_Error_Check(cudaMemcpy(dev_skewers_velocity, skewers_velocity_root, field_size, cudaMemcpyHostToDevice));
  GPU_Error_Check(cudaMemcpy(dev_skewers_temperature, skewers_temperature_root, field_size, cudaMemcpyHostToDevice));

  int const n_blocks_per_skewer = (n_los_total - 1) / TPB_LYA_FLUX + 1;
  int const n_blocks_skewers    = n_skewers < MAX_GRID_Y_LYA_FLUX ? n_skewers : MAX_GRID_Y_LYA_FLUX;
  dim3 dim2dGrid(n_blocks_per_skewer, n_blocks_skewers, 1);
  dim3 dim1dBlock(TPB_LYA_FLUX, 1, 1);

  hipLaunchKernelGGL(Compute_Transmitted_Flux_Skewers_kernel, dim2dGrid, dim1dBlock, 0, 0, n_skewers, n_los_total,
                     n_ghost, dens_factor_HI, dens_factor_HeII, vel_factor, temp_factor_HI, temp_factor_HeII,
                     dv_Hubble, Lya_sigma_HI, Lya_sigma_HeII, dev_skewers_HI_density, dev_skewers_HeII_density,
                     dev_skewers_velocity, dev_skewers_temperature, dev_skewers_transmitted_flux_HI,
                     dev_skewers_transmitted_flux_HeII);
  GPU_Error_Check();

  GPU_Error_Check(
      cudaMemcpy(skewers_transmitted_flux_HI, dev_skewers_transmitted_flux_HI, field_size, cudaMemcpyDeviceToHost));
  GPU_Error_Check(cudaMemcpy(skewers_transmitted_flux_HeII, dev_skewers_transmitted_flux_HeII, field_size,
                             cudaMemcpyDeviceToHost));

  for (int skewer_id = 0; skewer_id < n_skewers; skewer_id++) {
    Analysis.Compute_Lya_Mean_Flux_Skewer(skewer_id, axis);
  }
}

void AnalysisModule::Initialize_Lya_Statistics_GPU()
{
  if (am_I_root_x) {
    size_t const field_size_x = static_cast<size_t>(n_skewers_local_x) * nx_total * sizeof(Real);
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_HI_density_x, field_size_x));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_HeII_density_x, field_size_x));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_velocity_x, field_size_x));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_temperature_x, field_size_x));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_transmitted_flux_HI_x, field_size_x));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_transmitted_flux_HeII_x, field_size_x));
  }

  if (am_I_root_y) {
    size_t const field_size_y = static_cast<size_t>(n_skewers_local_y) * ny_total * sizeof(Real);
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_HI_density_y, field_size_y));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_HeII_density_y, field_size_y));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_velocity_y, field_size_y));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_temperature_y, field_size_y));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_transmitted_flux_HI_y, field_size_y));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_transmitted_flux_HeII_y, field_size_y));
  }

  if (am_I_root_z) {
    size_t const field_size_z = static_cast<size_t>(n_skewers_local_z) * nz_total * sizeof(Real);
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_HI_density_z, field_size_z));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_HeII_density_z, field_size_z));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_velocity_z, field_size_z));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_temperature_z, field_size_z));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_transmitted_flux_HI_z, field_size_z));
    GPU_Error_Check(cudaMalloc((void **)&dev_skewers_transmitted_flux_HeII_z, field_size_z));
  }
}

void AnalysisModule::Free_Lya_Statistics_GPU()
{
  if (am_I_root_x) {
    GPU_Error_Check(cudaFree(dev_skewers_HI_density_x));
    GPU_Error_Check(cudaFree(dev_skewers_HeII_density_x));
    GPU_Error_Check(cudaFree(dev_skewers_velocity_x));
    GPU_Error_Check(cudaFree(dev_skewers_temperature_x));
    GPU_Error_Check(cudaFree(dev_skewers_transmitted_flux_HI_x));
    GPU_Error_Check(cudaFree(dev_skewers_transmitted_flux_HeII_x));
  }

  if (am_I_root_y) {
    GPU_Error_Check(cudaFree(dev_skewers_HI_density_y));
    GPU_Error_Check(cudaFree(dev_skewers_HeII_density_y));
    GPU_Error_Check(cudaFree(dev_skewers_velocity_y));
    GPU_Error_Check(cudaFree(dev_skewers_temperature_y));
    GPU_Error_Check(cudaFree(dev_skewers_transmitted_flux_HI_y));
    GPU_Error_Check(cudaFree(dev_skewers_transmitted_flux_HeII_y));
  }

  if (am_I_root_z) {
    GPU_Error_Check(cudaFree(dev_skewers_HI_density_z));
    GPU_Error_Check(cudaFree(dev_skewers_HeII_density_z));
    GPU_Error_Check(cudaFree(dev_skewers_velocity_z));
    GPU_Error_Check(cudaFree(dev_skewers_temperature_z));
    GPU_Error_Check(cudaFree(dev_skewers_transmitted_flux_HI_z));
    GPU_Error_Check(cudaFree(dev_skewers_transmitted_flux_HeII_z));
  }
}

#endif  // ANALYSIS && LYA_STATISTICS && GPU_LYA_FLUX

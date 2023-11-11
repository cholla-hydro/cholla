#if defined(FEEDBACK) && defined(PARTICLES_GPU) && defined(PARTICLE_AGE) && defined(PARTICLE_IDS)

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>

  #include <cstring>
  #include <fstream>
  #include <sstream>
  #include <vector>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../utils/DeviceVector.h"
  #include "feedback.h"

  #define FEED_INFO_N     8
  #define i_RES           1
  #define i_UNRES         2
  #define i_ENERGY        3
  #define i_MOMENTUM      4
  #define i_UNRES_ENERGY  5
  #define i_WIND_MOMENTUM 6
  #define i_WIND_ENERGY   7

  #define TPB_FEEDBACK 128

  #ifndef O_HIP
inline __device__ double atomicMax(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}
  #endif  // O_HIP

/** the prescription for dividing a scalar quantity between 3x3x3 cells is done
   by imagining a 2x2x2 cell volume around the SN.  These fractions, then,
   represent the linear extent of this volume into the cell in question. For i=0
   this should be 1*1/2. For i=-1 this should be (1-dx)*1/2. For i=+1 this
   should be dx*1/2. In the above the 1/2 factor is normalize over 2
   cells/direction.
  */
inline __device__ Real Frac(int i, Real dx) { return (-0.5 * i * i - 0.5 * i + 1 + i * dx) * 0.5; }

inline __device__ Real D_Frac(int i, Real dx)
{
  return (dx > 0.5) * i * (1 - 2 * dx) + ((i + 1) * dx + 0.5 * (i - 1)) - 3 * (i - 1) * (i + 1) * (0.5 - dx);
}

/** This function used for debugging potential race conditions.  Feedback from neighboring
    particles could simultaneously alter one hydro cell's conserved quantities.
 */
inline __device__ bool Particle_Is_Alone(Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev, part_int_t n_local,
                                         int gtid, Real dx)
{
  Real x0 = pos_x_dev[gtid];
  Real y0 = pos_y_dev[gtid];
  Real z0 = pos_z_dev[gtid];
  // Brute force loop to see if particle is alone
  for (int i = 0; i < n_local; i++) {
    if (i == gtid) continue;
    if (abs(x0 - pos_x_dev[i]) > dx) continue;
    if (abs(y0 - pos_y_dev[i]) > dx) continue;
    if (abs(z0 - pos_z_dev[i]) > dx) continue;
    // If we made it here, something is too close.
    return false;
  }
  return true;
}

inline __device__ Real Get_Average_Density(Real* density, int xi, int yi, int zi, int nx_grid, int ny_grid, int n_ghost)
{
  Real d_average = 0.0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        d_average +=
            density[(xi + n_ghost + i) + (yi + n_ghost + j) * nx_grid + (zi + n_ghost + k) * nx_grid * ny_grid];
      }
    }
  }
  return d_average / 27;
}

inline __device__ Real Get_Average_Number_Density_CGS(Real* density, int xi, int yi, int zi, int nx_grid, int ny_grid,
                                                      int n_ghost)
{
  return Get_Average_Density(density, xi, yi, zi, nx_grid, ny_grid, n_ghost) * DENSITY_UNIT / (MU * MP);
}

__device__ void Apply_Resolved_SN(Real pos_x, Real pos_y, Real pos_z, Real xMin, Real yMin, Real zMin, Real dx, Real dy,
                                  Real dz, int nx_g, int ny_g, int n_ghost, int n_cells,
                                  Real* conserved_device, Real feedback_density, Real feedback_energy)
{
  // For 2x2x2, a particle between 0-0.5 injects onto cell - 1
  int indx_x = (int)floor((pos_x - xMin - 0.5 * dx) / dx);
  int indx_y = (int)floor((pos_y - yMin - 0.5 * dy) / dy);
  int indx_z = (int)floor((pos_z - zMin - 0.5 * dz) / dz);

  Real cell_center_x = xMin + indx_x * dx + 0.5 * dx;
  Real cell_center_y = yMin + indx_y * dy + 0.5 * dy;
  Real cell_center_z = zMin + indx_z * dz + 0.5 * dz;

  Real delta_x = 1 - (pos_x - cell_center_x) / dx;
  Real delta_y = 1 - (pos_y - cell_center_y) / dy;
  Real delta_z = 1 - (pos_z - cell_center_z) / dz;

  Real* density    = conserved_device;
  Real* energy     = &conserved_device[n_cells * grid_enum::Energy];
#ifdef DE
  Real* gasEnergy  = &conserved_device[n_cells * grid_enum::GasEnergy];
#endif

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        int indx    = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;
        Real x_frac = i * (1 - delta_x) + (1 - i) * delta_x;
        Real y_frac = j * (1 - delta_y) + (1 - j) * delta_y;
        Real z_frac = k * (1 - delta_z) + (1 - k) * delta_z;

        atomicAdd(&density[indx], x_frac * y_frac * z_frac * feedback_density);
#ifdef DE
        atomicAdd(&gasEnergy[indx], x_frac * y_frac * z_frac * feedback_energy);
#endif
        atomicAdd(&energy[indx], x_frac * y_frac * z_frac * feedback_energy);

      }  // k loop
    }    // j loop
  }      // i loop
}

/* \brief Function used for depositing energy or momentum from an unresolved
 * supernova or from a stellar wind
 *
 * \note
 * Previously there were 2 separate functions defined to perform this operation.
 * They were functionally the same. They only differences were the names of
 * variables.
 *
 * \par
 * There are currently issues with the internals of this function:
 * - this requires the codebase to be compiled with the dual energy formalism
 * - momentum and total energy are not updated self-consistently
 */
__device__ void Apply_Energy_Momentum_Deposition(Real pos_x, Real pos_y, Real pos_z, Real xMin, Real yMin, Real zMin,
                                                 Real dx, Real dy, Real dz, int nx_g, int ny_g, int n_ghost,
                                                 int n_cells, Real* conserved_device,
                                                 Real feedback_density, Real feedback_momentum, Real feedback_energy,
                                                 int indx_x, int indx_y, int indx_z)
{
  Real delta_x = (pos_x - xMin - indx_x * dx) / dx;
  Real delta_y = (pos_y - yMin - indx_y * dy) / dy;
  Real delta_z = (pos_z - zMin - indx_z * dz) / dz;

  Real* density    = conserved_device;
  Real* momentum_x = &conserved_device[n_cells * grid_enum::momentum_x];
  Real* momentum_y = &conserved_device[n_cells * grid_enum::momentum_y];
  Real* momentum_z = &conserved_device[n_cells * grid_enum::momentum_z];
  Real* energy     = &conserved_device[n_cells * grid_enum::Energy];
#ifdef DE
  Real* gas_energy = &conserved_device[n_cells * grid_enum::GasEnergy];
#endif

  // loop over the 27 cells to add up all the allocated feedback
  // momentum magnitudes.  For each cell allocate density and
  // energy based on the ratio of allocated momentum to this overall sum.
  Real mag = 0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        Real x_frac = D_Frac(i, delta_x) * Frac(j, delta_y) * Frac(k, delta_z);
        Real y_frac = Frac(i, delta_x) * D_Frac(j, delta_y) * Frac(k, delta_z);
        Real z_frac = Frac(i, delta_x) * Frac(j, delta_y) * D_Frac(k, delta_z);

        mag += sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac);
      }
    }
  }

  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        // index in array of conserved quantities
        int indx = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;

        Real x_frac = D_Frac(i, delta_x) * Frac(j, delta_y) * Frac(k, delta_z);
        Real y_frac = Frac(i, delta_x) * D_Frac(j, delta_y) * Frac(k, delta_z);
        Real z_frac = Frac(i, delta_x) * Frac(j, delta_y) * D_Frac(k, delta_z);

        Real px       = x_frac * feedback_momentum;
        Real py       = y_frac * feedback_momentum;
        Real pz       = z_frac * feedback_momentum;
        Real f_dens   = sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac) / mag * feedback_density;
        Real f_energy = sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac) / mag * feedback_energy;

        atomicAdd(&density[indx], f_dens);
        atomicAdd(&momentum_x[indx], px);
        atomicAdd(&momentum_y[indx], py);
        atomicAdd(&momentum_z[indx], pz);
        atomicAdd(&energy[indx], f_energy);

#ifdef DE
        gas_energy[indx] = energy[indx] - (momentum_x[indx] * momentum_x[indx] + momentum_y[indx] * momentum_y[indx] +
                                           momentum_z[indx] * momentum_z[indx]) /
                                              (2 * density[indx]);
#endif
        /*
        energy[indx] = ( momentum_x[indx] * momentum_x[indx] +
                         momentum_y[indx] * momentum_y[indx] +
                         momentum_z[indx] * momentum_z[indx] ) /
                       2 / density[indx] + gasEnergy[indx];
        */
      }  // k loop
    }    // j loop
  }      // i loop
}

__device__ void SN_Feedback(Real pos_x, Real pos_y, Real pos_z, Real age, Real* mass_dev, part_int_t* id_dev, Real xMin,
                            Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g,
                            int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt,
                            const feedback::SNRateCalc snr_calc, Real* prev_dens,
                            Real* s_info, Real* conserved_dev, Real gamma, int indx_x, int indx_y, int indx_z)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  Real dV = dx * dy * dz;
  Real feedback_density, feedback_momentum, feedback_energy;
  int n_cells    = nx_g * ny_g * nz_g;

  Real average_num_sn = snr_calc.Get_SN_Rate(age) * mass_dev[gtid] * dt;
  int N               = snr_calc.Get_Number_Of_SNe_In_Cluster(average_num_sn, n_step, id_dev[gtid]);
  /*
  if (gtid == 0) {
    kernel_printf("SNUMBER n_step: %d, id: %lld, N: %d\n", n_step, id_dev[gtid], N);
  }
  */

  // no sense doing anything if there was no SN
  if (N != 0) {
    Real n_0;
    Real* density             = conserved_dev;
    n_0                       = Get_Average_Number_Density_CGS(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
    prev_dens[gtid]           = n_0;
    s_info[FEED_INFO_N * tid] = 1. * N;

    feedback_energy  = N * feedback::ENERGY_PER_SN / dV;
    feedback_density = N * feedback::MASS_PER_SN / dV;

    Real shell_radius = feedback::R_SH * pow(n_0, -0.46) * pow(fabsf(N), 0.29);
  #ifdef ONLY_RESOLVED
    bool is_resolved = true;
  #else
    bool is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius;
  #endif

    if (is_resolved) {
      // inject energy and density
      s_info[FEED_INFO_N * tid + i_RES]    = 1. * N;
      s_info[FEED_INFO_N * tid + i_ENERGY] = feedback_energy * dV;
      Apply_Resolved_SN(pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost, n_cells,
                        conserved_dev, feedback_density, feedback_energy);
    } else {
      // inject momentum and density
      feedback_momentum = feedback::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(fabsf(N), 0.93) / dV / sqrt(3.0);
      s_info[FEED_INFO_N * tid + i_UNRES]        = 1. * N;
      s_info[FEED_INFO_N * tid + i_MOMENTUM]     = feedback_momentum * dV * sqrt(3.0);
      s_info[FEED_INFO_N * tid + i_UNRES_ENERGY] = feedback_energy * dV;
      Apply_Energy_Momentum_Deposition(
          pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost, n_cells, conserved_dev,
          feedback_density, feedback_momentum, feedback_energy, indx_x, indx_y, indx_z);
    }
  }

}

__device__ void Wind_Feedback(Real pos_x, Real pos_y, Real pos_z, Real age, Real* mass_dev, part_int_t* id_dev,
                              Real xMin, Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy,
                              Real dz, int nx_g, int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt,
                              const feedback::SWRateCalc sw_calc, Real* s_info,
                              Real* conserved_dev, Real gamma, int indx_x, int indx_y, int indx_z)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  Real dV = dx * dy * dz;
  int n_cells    = nx_g * ny_g * nz_g;

  if ((age < 0) or not sw_calc.is_active(age)) return;
  Real feedback_momentum = sw_calc.Get_Wind_Flux(age);
  // no sense in proceeding if there is no feedback.
  if (feedback_momentum == 0) return;
  Real feedback_energy  = sw_calc.Get_Wind_Power(age);
  Real feedback_density = sw_calc.Get_Wind_Mass(feedback_momentum, feedback_energy);

  // feedback_momentum now becomes momentum component along one direction.
  feedback_momentum *= mass_dev[gtid] * dt / dV / sqrt(3.0);
  feedback_density *= mass_dev[gtid] * dt / dV;
  feedback_energy *= mass_dev[gtid] * dt / dV;

  /* TODO refactor into separate kernel call
  if (true) {
    mass_dev[gtid]   -= feedback_density * dV;
  }*/

  // we log net momentum, not momentum density, and magnitude (not the
  // component along a direction)
  s_info[FEED_INFO_N * tid + i_WIND_MOMENTUM] = feedback_momentum * dV * sqrt(3.0);
  s_info[FEED_INFO_N * tid + i_WIND_ENERGY]   = feedback_energy * dV;

  Apply_Energy_Momentum_Deposition(pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost,
                                   n_cells, conserved_dev, feedback_density,
                                   feedback_momentum, feedback_energy, indx_x, indx_y, indx_z);
}

__device__ void Cluster_Feedback_Helper(part_int_t n_local, Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev,
                                        Real* age_dev, Real* mass_dev, part_int_t* id_dev, Real xMin, Real yMin,
                                        Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g,
                                        int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt, Real* dti,
                                        const feedback::SNRateCalc snr_calc, Real* prev_dens, const feedback::SWRateCalc sw_calc,
                                        Real* s_info, Real* conserved_dev, Real gamma)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;
  // Bounds check on particle arrays
  if (gtid >= n_local) return;

  Real pos_x    = pos_x_dev[gtid];
  Real pos_y    = pos_y_dev[gtid];
  Real pos_z    = pos_z_dev[gtid];
  bool in_local = (pos_x >= xMin && pos_x < xMax) && (pos_y >= yMin && pos_y < yMax) && (pos_z >= zMin && pos_z < zMax);
  // Particle is outside bounds, exit
  if (!in_local) return;

  int indx_x  = (int)floor((pos_x - xMin) / dx);
  int indx_y  = (int)floor((pos_y - yMin) / dy);
  int indx_z  = (int)floor((pos_z - zMin) / dz);
  bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x >= nx_g - 2 * n_ghost ||
                indx_y >= ny_g - 2 * n_ghost || indx_z >= nz_g - 2 * n_ghost;
  // Ignore this particle, exit
  if (ignore) return;

  // bool is_alone = Particle_Is_Alone(pos_x_dev, pos_y_dev, pos_z_dev, n_local, gtid, 6*dx);
  // if (is_alone) kernel_printf(" particle not alone: step %d, id %ld\n", n_step, id_dev[gtid]);
  // if (!is_alone) return;

  // note age_dev is actually the time of birth
  Real age = t - age_dev[gtid];

  bool is_sn_feedback = false;
  bool is_wd_feedback = false;
  #ifndef NO_SN_FEEDBACK
  is_sn_feedback = true;
  #endif
  #ifndef NO_WIND_FEEDBACK
  is_wd_feedback = true;
  #endif

  if (is_sn_feedback)
    SN_Feedback(pos_x, pos_y, pos_z, age, mass_dev, id_dev, xMin, yMin, zMin, xMax, yMax, zMax, dx, dy, dz, nx_g,
                ny_g, nz_g, n_ghost, n_step, t, dt, snr_calc, prev_dens,
                s_info, conserved_dev, gamma, indx_x, indx_y, indx_z);
  if (is_wd_feedback)
    Wind_Feedback(pos_x, pos_y, pos_z, age, mass_dev, id_dev, xMin, yMin, zMin, xMax, yMax, zMax, dx, dy, dz, nx_g,
                  ny_g, nz_g, n_ghost, n_step, t, dt, sw_calc,
                  s_info, conserved_dev, gamma, indx_x, indx_y, indx_z);

  return;
}

__global__ void Cluster_Feedback_Kernel(part_int_t n_local, part_int_t* id_dev, Real* pos_x_dev, Real* pos_y_dev,
                                        Real* pos_z_dev, Real* mass_dev, Real* age_dev, Real xMin, Real yMin, Real zMin,
                                        Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g, int ny_g,
                                        int nz_g, int n_ghost, Real t, Real dt, Real* dti, Real* info, Real* density,
                                        Real gamma, Real* prev_dens, const feedback::SNRateCalc snr_calc,
                                        const feedback::SWRateCalc sw_calc, int n_step)
{
  int tid = threadIdx.x;

  // for collecting SN feedback information
  __shared__ Real s_info[FEED_INFO_N * TPB_FEEDBACK];
  s_info[FEED_INFO_N * tid]     = 0;  // number of supernovae
  s_info[FEED_INFO_N * tid + 1] = 0;  // number of resolved events
  s_info[FEED_INFO_N * tid + 2] = 0;  // number of unresolved events
  s_info[FEED_INFO_N * tid + 3] = 0;  // resolved energy
  s_info[FEED_INFO_N * tid + 4] = 0;  // unresolved momentum
  s_info[FEED_INFO_N * tid + 5] = 0;  // unresolved KE added via momentum
  s_info[FEED_INFO_N * tid + 6] = 0;  // wind momentum
  s_info[FEED_INFO_N * tid + 7] = 0;  // wind energy added

  Cluster_Feedback_Helper(n_local, pos_x_dev, pos_y_dev, pos_z_dev, age_dev, mass_dev, id_dev, xMin, yMin, zMin, xMax,
                          yMax, zMax, dx, dy, dz, nx_g, ny_g, nz_g, n_ghost, n_step, t, dt, dti, snr_calc, prev_dens,
                          sw_calc, s_info, density, gamma);

  __syncthreads();

  // reduce the info from all the threads in the block
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_info[FEED_INFO_N * tid] += s_info[FEED_INFO_N * (tid + s)];
      s_info[FEED_INFO_N * tid + 1] += s_info[FEED_INFO_N * (tid + s) + 1];
      s_info[FEED_INFO_N * tid + 2] += s_info[FEED_INFO_N * (tid + s) + 2];
      s_info[FEED_INFO_N * tid + 3] += s_info[FEED_INFO_N * (tid + s) + 3];
      s_info[FEED_INFO_N * tid + 4] += s_info[FEED_INFO_N * (tid + s) + 4];
      s_info[FEED_INFO_N * tid + 5] += s_info[FEED_INFO_N * (tid + s) + 5];
      s_info[FEED_INFO_N * tid + 6] += s_info[FEED_INFO_N * (tid + s) + 6];
      s_info[FEED_INFO_N * tid + 7] += s_info[FEED_INFO_N * (tid + s) + 7];
    }
    __syncthreads();
  }

  // atomicAdd reduces across all blocks
  if (tid == 0) {
    atomicAdd(info, s_info[0]);
    atomicAdd(info + 1, s_info[1]);
    atomicAdd(info + 2, s_info[2]);
    atomicAdd(info + 3, s_info[3]);
    atomicAdd(info + 4, s_info[4]);
    atomicAdd(info + 5, s_info[5]);
    atomicAdd(info + 6, s_info[6]);
    atomicAdd(info + 7, s_info[7]);
  }
}

__global__ void Adjust_Cluster_Mass_Kernel(part_int_t n_local, Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev,
                                           Real* age_dev, Real* mass_dev, part_int_t* id_dev, Real xMin, Real yMin,
                                           Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz,
                                           int nx_g, int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt,
                                           const feedback::SNRateCalc snr_calc, const feedback::SWRateCalc sw_calc)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;
  // Bounds check on particle arrays
  if (gtid >= n_local) return;

  Real pos_x    = pos_x_dev[gtid];
  Real pos_y    = pos_y_dev[gtid];
  Real pos_z    = pos_z_dev[gtid];
  bool in_local = (pos_x >= xMin && pos_x < xMax) && (pos_y >= yMin && pos_y < yMax) && (pos_z >= zMin && pos_z < zMax);
  // Particle is outside bounds, exit
  if (!in_local) return;

  int indx_x  = (int)floor((pos_x - xMin) / dx);
  int indx_y  = (int)floor((pos_y - yMin) / dy);
  int indx_z  = (int)floor((pos_z - zMin) / dz);
  bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x >= nx_g - 2 * n_ghost ||
                indx_y >= ny_g - 2 * n_ghost || indx_z >= nz_g - 2 * n_ghost;
  // Ignore this particle, exit
  if (ignore) return;

  // bool is_alone = Particle_Is_Alone(pos_x_dev, pos_y_dev, pos_z_dev, n_local, gtid, 6*dx);
  // if (is_alone) kernel_printf(" particle not alone: step %d, id %ld\n", n_step, id_dev[gtid]);
  // if (!is_alone) return;

  Real age = t - age_dev[gtid];

  #ifndef NO_SN_FEEDBACK
  Real average_num_sn = snr_calc.Get_SN_Rate(age) * mass_dev[gtid] * dt;
  int N               = snr_calc.Get_Number_Of_SNe_In_Cluster(average_num_sn, n_step, id_dev[gtid]);
  mass_dev[gtid] -= N * feedback::MASS_PER_SN;
  #endif

  #ifndef NO_WIND_FEEDBACK
  Real feedback_momentum  = sw_calc.Get_Wind_Flux(age);
  Real feedback_energy    = sw_calc.Get_Wind_Power(age);
  Real feedback_mass_rate = sw_calc.Get_Wind_Mass(feedback_momentum, feedback_energy);

  mass_dev[gtid] -= feedback_mass_rate * dt;
  #endif
}

__device__ void Set_Average_Density(int indx_x, int indx_y, int indx_z, int nx_g, int ny_g, int n_ghost, Real* density,
                                    Real ave_dens)
{
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        int indx = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;

        density[indx] = ave_dens;
      }
    }
  }
}

__global__ void Set_Ave_Density_Kernel(part_int_t n_local, Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev,
                                       Real* mass_dev, Real* age_dev, part_int_t* id_dev, Real xMin, Real yMin,
                                       Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g,
                                       int ny_g, int nz_g, int n_ghost, Real t, Real dt, Real* density,
                                       const feedback::SNRateCalc snr_calc, const feedback::SWRateCalc sw_calc,
                                       int n_step)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;
  // Bounds check on particle arrays
  if (gtid >= n_local) return;

  Real pos_x    = pos_x_dev[gtid];
  Real pos_y    = pos_y_dev[gtid];
  Real pos_z    = pos_z_dev[gtid];
  bool in_local = (pos_x >= xMin && pos_x < xMax) && (pos_y >= yMin && pos_y < yMax) && (pos_z >= zMin && pos_z < zMax);
  // Particle is outside bounds, exit
  if (!in_local) return;

  int indx_x  = (int)floor((pos_x - xMin) / dx);
  int indx_y  = (int)floor((pos_y - yMin) / dy);
  int indx_z  = (int)floor((pos_z - zMin) / dz);
  bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x >= nx_g - 2 * n_ghost ||
                indx_y >= ny_g - 2 * n_ghost || indx_z >= nz_g - 2 * n_ghost;
  // Ignore this particle, exit
  if (ignore) return;

  // bool is_alone = Particle_Is_Alone(pos_x_dev, pos_y_dev, pos_z_dev, n_local, gtid, 6*dx);
  // if (is_alone) kernel_printf(" particle not alone: step %d, id %ld\n", n_step, id_dev[gtid]);
  // if (!is_alone) return;

  bool is_sn_feedback   = false;
  bool is_wind_feedback = false;
  #ifndef NO_SN_FEEDBACK
  is_sn_feedback = true;
  #endif
  #ifndef NO_WIND_FEEDBACK
  is_wind_feedback = true;
  #endif

  Real ave_dens;
  Real age = t - age_dev[gtid];
  if (is_wind_feedback) {
    if (sw_calc.is_active(age)) {
      ave_dens = Get_Average_Density(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
      Set_Average_Density(indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost, density, ave_dens);
      // since we've set the average density, no need to keep
      // checking whether we should do so.
      return;
    }
  }
  if (is_sn_feedback) {
    if (snr_calc.nonzero_sn_probability(age)) {
      Real average_num_sn = snr_calc.Get_SN_Rate(age) * mass_dev[gtid] * dt;
      int N               = snr_calc.Get_Number_Of_SNe_In_Cluster(average_num_sn, n_step, id_dev[gtid]);
      /*
      if (gtid == 0) {
        kernel_printf("AVEDENS n_step: %d, id: %lld, N: %d\n", n_step, id_dev[gtid], N);
      }*/
      Real n_0          = Get_Average_Number_Density_CGS(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
      Real shell_radius = feedback::R_SH * pow(n_0, -0.46) * pow(N, 0.29);
  #ifdef ONLY_RESOLVED
      bool is_resolved = true;
  #else
      bool is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius;
  #endif

      // resolved SN feedback does not average densities.
      if (!is_resolved && N > 0) {
        ave_dens = n_0 * MU * MP / DENSITY_UNIT;
        Set_Average_Density(indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost, density, ave_dens);
      }
    }
  }
}

feedback::ClusterFeedbackMethod::ClusterFeedbackMethod(struct parameters& P, FeedbackAnalysis& analysis)
  : analysis(analysis),
    snr_calc_(P),
    sw_calc_(P)
{ }

/**
 * @brief Stellar feedback function (SNe and stellar winds)
 *
 * @param G
 */
void feedback::ClusterFeedbackMethod::operator()(Grid3D& G)
{
  #ifdef CPU_TIME
  G.Timer.Feedback.Start();
  #endif

  if (G.H.dt == 0) return;

  // h_info is used to store feedback summary info on the host. The following
  // syntax sets all entries to 0 -- important if a process has no particles
  // (this is valid C++ syntax, but historically wasn't valid C syntax)
  Real h_info[FEED_INFO_N] = {};

  // only apply feedback if we have clusters
  if (G.Particles.n_local > 0) {

    // Declare/allocate device buffer for accumulating summary information about feedback
    cuda_utilities::DeviceVector<Real> d_info(FEED_INFO_N, true);  // initialized to 0

    // Declare/allocate device buffers for d_dti and d_prev_dens. These only exist for historical
    // reasons. Previously we added feedback after computing the hydro timestep and we needed to
    // rewind the effect and try again if it made the minimum hydro timestep too large
    // -> d_dti was needed to track hydro-timestep that would be computed after applying feedback
    // -> it's not totally clear what d_prev_dens was for (previously, I thought it was for
    //    recording the average density around each particle, but that's wrong)
    cuda_utilities::DeviceVector<Real> d_dti(1, true);  // initialized to 0
    cuda_utilities::DeviceVector<Real> d_prev_dens(G.Particles.n_local, true);  // initialized to 0

    // I have no idea what ngrid is used for...
    int ngrid = (G.Particles.n_local - 1) / TPB_FEEDBACK + 1;

    // before applying feedback, set gas density around clusters to the
    // average value from the 27 neighboring cells.  We don't want to
    // do this during application of feedback since "undoing it" in the
    // event that the time step is too large becomes difficult.
    hipLaunchKernelGGL(Set_Ave_Density_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local, G.Particles.pos_x_dev,
                       G.Particles.pos_y_dev, G.Particles.pos_z_dev, G.Particles.mass_dev, G.Particles.age_dev,
                       G.Particles.partIDs_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal, G.H.xblocal_max, G.H.yblocal_max,
                       G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny, G.H.nz, G.H.n_ghost, G.H.t, G.H.dt,
                       G.C.d_density, snr_calc_, sw_calc_, G.H.n_step);

    CHECK(cudaDeviceSynchronize());  // probably unnecessary (it replaced a now-unneeded cudaMemset)

    hipLaunchKernelGGL(Cluster_Feedback_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local,
                       G.Particles.partIDs_dev, G.Particles.pos_x_dev, G.Particles.pos_y_dev, G.Particles.pos_z_dev,
                       G.Particles.mass_dev, G.Particles.age_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal,
                       G.H.xblocal_max, G.H.yblocal_max, G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny,
                       G.H.nz, G.H.n_ghost, G.H.t, G.H.dt, d_dti.data(), d_info.data(), G.C.d_density, gama, 
                       d_prev_dens.data(), snr_calc_, sw_calc_, G.H.n_step);

    CHECK(cudaDeviceSynchronize());  // probably unnecessary (it replaced a now-unneeded cudaMemcpy)

    // There was previously a comment here stating "TODO reduce cluster mass",
    // but we're already  doing this here, right?
    hipLaunchKernelGGL(Adjust_Cluster_Mass_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local,
                       G.Particles.pos_x_dev, G.Particles.pos_y_dev, G.Particles.pos_z_dev, G.Particles.age_dev,
                       G.Particles.mass_dev, G.Particles.partIDs_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal,
                       G.H.xblocal_max, G.H.yblocal_max, G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny,
                       G.H.nz, G.H.n_ghost, G.H.n_step, G.H.t, G.H.dt, snr_calc_, sw_calc_);

    // copy summary data back to the host
    CHECK(cudaMemcpy(&h_info, d_info.data(), FEED_INFO_N * sizeof(Real), cudaMemcpyDeviceToHost));
  }

  // now gather the feedback summary info into an array called info.
  #ifdef MPI_CHOLLA
  Real info[FEED_INFO_N];
  MPI_Reduce(&h_info, &info, FEED_INFO_N, MPI_CHREAL, MPI_SUM, root, world);
  #else
  Real* info = h_info;
  #endif

  #ifdef MPI_CHOLLA  // only do stats gathering on root rank
  if (procID == 0) {
  #endif

    analysis.countSN += (long)info[feedback::SN];
    analysis.countResolved += (long)info[feedback::RESOLVED];
    analysis.countUnresolved += (long)info[feedback::NOT_RESOLVED];
    analysis.totalEnergy += info[feedback::ENERGY];
    analysis.totalMomentum += info[feedback::MOMENTUM];
    analysis.totalUnresEnergy += info[feedback::UNRES_ENERGY];
    analysis.totalWindMomentum += info[i_WIND_MOMENTUM];
    analysis.totalWindEnergy += info[i_WIND_ENERGY];

    chprintf("iteration %d, t %.4e, dt %.4e", G.H.n_step, G.H.t, G.H.dt);

  #ifndef NO_SN_FEEDBACK
    Real global_resolved_ratio = 0.0;
    if (analysis.countResolved > 0 || analysis.countUnresolved > 0) {
      global_resolved_ratio = analysis.countResolved / (analysis.countResolved + analysis.countUnresolved);
    }
    chprintf(": number of SN: %d,(R: %d, UR: %d)\n", (int)info[feedback::SN], (long)info[feedback::RESOLVED],
             (long)info[feedback::NOT_RESOLVED]);
    chprintf("    cummulative: #SN: %d, ratio of resolved (R: %d, UR: %d) = %.3e\n", (long)analysis.countSN,
             (long)analysis.countResolved, (long)analysis.countUnresolved, global_resolved_ratio);
    chprintf("    sn  r energy  : %.5e erg, cumulative: %.5e erg\n", info[feedback::ENERGY] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalEnergy * FORCE_UNIT * LENGTH_UNIT);
    chprintf("    sn ur energy  : %.5e erg, cumulative: %.5e erg\n",
             info[feedback::UNRES_ENERGY] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalUnresEnergy * FORCE_UNIT * LENGTH_UNIT);
    chprintf("    sn momentum  : %.5e SM km/s, cumulative: %.5e SM km/s\n",
             info[feedback::MOMENTUM] * VELOCITY_UNIT / 1e5, analysis.totalMomentum * VELOCITY_UNIT / 1e5);
  #endif  // NO_SN_FEEDBACK

  #ifndef NO_WIND_FEEDBACK
    chprintf("    wind momentum: %.5e S.M. km/s,  cumulative: %.5e S.M. km/s\n",
             info[i_WIND_MOMENTUM] * VELOCITY_UNIT / 1e5, analysis.totalWindMomentum * VELOCITY_UNIT / 1e5);
    chprintf("    wind energy  : %.5e erg,  cumulative: %.5e erg\n", info[i_WIND_ENERGY] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalWindEnergy * FORCE_UNIT * LENGTH_UNIT);
  #endif  // NO_WIND_FEEDBACK

  #ifdef MPI_CHOLLA
  }  //   end if procID == 0
  #endif

  #ifdef CPU_TIME
  G.Timer.Feedback.End();
  #endif
}

#endif  // FEEDBACK & PARTICLES_GPU & PARTICLE_IDS & PARTICLE_AGE

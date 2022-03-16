#if defined(COSMOLOGY) && defined(PARTICLES_GPU)


#include "../cosmology/cosmology_functions_gpu.h"


// __device__ Real Get_Hubble_Parameter_dev( Real a, Real H0, Real Omega_M, Real Omega_L, Real Omega_K ){
//   Real a2 = a * a;
//   Real a3 = a2 * a;
//   Real factor = ( Omega_M/a3 + Omega_K/a2 + Omega_L );
//   return H0 * sqrt(factor);
//
// }


void __global__ Change_GAS_Frame_System_kernel( Real dens_factor, Real momentum_factor, Real energy_factor,
          int nx, int ny, int nz, Real *density_d, Real *momentum_x_d, Real *momentum_y_d, Real *momentum_z_d,
          Real *Energy_d, Real *GasEnergy_d ){

  int tid_x, tid_y, tid_z, tid_grid;
  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  tid_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (tid_x >= nx || tid_y >= ny || tid_z >= nz ) return;

  tid_grid = tid_x + tid_y*nx + tid_z*nx*ny;

  density_d[tid_grid]    = density_d[tid_grid]    * dens_factor;
  momentum_x_d[tid_grid] = momentum_x_d[tid_grid] * momentum_factor;
  momentum_y_d[tid_grid] = momentum_y_d[tid_grid] * momentum_factor;
  momentum_z_d[tid_grid] = momentum_z_d[tid_grid] * momentum_factor;
  Energy_d[tid_grid]     = Energy_d[tid_grid]     * energy_factor;
  #ifdef DE
  GasEnergy_d[tid_grid]  = GasEnergy_d[tid_grid]  * energy_factor;
  #endif

  //NOTE If CHEMISTRY_GPU I need to add the conversion for the chemical species here

}


void Grid3D::Change_GAS_Frame_System_GPU( bool forward ){

  Real dens_factor, momentum_factor, energy_factor;
  if ( forward ){
    dens_factor = 1 / Cosmo.rho_0_gas;
    momentum_factor = 1 / Cosmo.rho_0_gas / Cosmo.v_0_gas * Cosmo.current_a;
    energy_factor = 1 / Cosmo.rho_0_gas / Cosmo.v_0_gas / Cosmo.v_0_gas * Cosmo.current_a * Cosmo.current_a;
  }
  else{
    dens_factor = Cosmo.rho_0_gas;
    momentum_factor =  Cosmo.rho_0_gas * Cosmo.v_0_gas / Cosmo.current_a;
    energy_factor =  Cosmo.rho_0_gas * Cosmo.v_0_gas * Cosmo.v_0_gas / Cosmo.current_a / Cosmo.current_a;
  }

  int nx, ny, nz;
  nx = H.nx;
  ny = H.ny;
  nz = H.nz;

  // set values for GPU kernels
  int tpb_x = TPBX_COSMO;
  int tpb_y = TPBY_COSMO;
  int tpb_z = TPBZ_COSMO;
  int ngrid_x = (nx - 1) / tpb_x + 1;
  int ngrid_y = (ny - 1) / tpb_y + 1;
  int ngrid_z = (nz - 1) / tpb_z + 1;
  // number of blocks per 1D grid
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);

  Real *GasEnergy_d;
  #ifdef DE
  GasEnergy_d = C.d_GasEnergy;
  #else
  GasEnergy_d = NULL;
  #endif

  hipLaunchKernelGGL(Change_GAS_Frame_System_kernel, dim3dGrid, dim3dBlock, 0, 0, dens_factor, momentum_factor, energy_factor, nx, ny, nz,
                 C.d_density, C.d_momentum_x, C.d_momentum_y, C.d_momentum_z, C.d_Energy, GasEnergy_d   );

}




#endif //COSMOLOGY

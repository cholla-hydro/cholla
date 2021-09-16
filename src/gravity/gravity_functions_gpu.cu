#if defined(GRAVITY) && defined(GRAVITY_GPU)

#include "../grid/grid3D.h"
#include "../global/global.h"
#include "../io/io.h"
#include "../utils/error_handling.h"
#include <cstring>


void Grav3D::AllocateMemory_GPU(){

  CudaSafeCall( cudaMalloc((void**)&F.density_d,  n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F.potential_d,   n_cells_potential*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F.potential_1_d, n_cells_potential*sizeof(Real)) );

  #ifdef GRAVITY_GPU

  #ifdef GRAV_ISOLATED_BOUNDARY_X
  CudaSafeCall( cudaMalloc((void**)&F.pot_boundary_x0_d, N_GHOST_POTENTIAL*ny_local*nz_local*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F.pot_boundary_x1_d, N_GHOST_POTENTIAL*ny_local*nz_local*sizeof(Real)) );
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
  CudaSafeCall( cudaMalloc((void**)&F.pot_boundary_y0_d, N_GHOST_POTENTIAL*nx_local*nz_local*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F.pot_boundary_y1_d, N_GHOST_POTENTIAL*nx_local*nz_local*sizeof(Real)) );
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  CudaSafeCall( cudaMalloc((void**)&F.pot_boundary_z0_d, N_GHOST_POTENTIAL*nx_local*ny_local*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F.pot_boundary_z1_d, N_GHOST_POTENTIAL*nx_local*ny_local*sizeof(Real)) );
  #endif

  #endif//GRAVITY_GPU

  chprintf( "Allocated Gravity GPU memory \n" );
}


void Grav3D::FreeMemory_GPU(void){

  cudaFree( F.density_d );
  cudaFree( F.potential_d );
  cudaFree( F.potential_1_d );


  #ifdef GRAVITY_GPU

  #ifdef GRAV_ISOLATED_BOUNDARY_X
  cudaFree( F.pot_boundary_x0_d);
  cudaFree( F.pot_boundary_x1_d);
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
  cudaFree( F.pot_boundary_y0_d);
  cudaFree( F.pot_boundary_y1_d);
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  cudaFree( F.pot_boundary_z0_d);
  cudaFree( F.pot_boundary_z1_d);
  #endif

  #endif //GRAVITY_GPU

}

void __global__ Copy_Hydro_Density_to_Gravity_Kernel( Real *src_density_d, Real *dst_density_d, int nx_local, int ny_local, int nz_local, int n_ghost, Real cosmo_rho_0_gas   ){

  int tid_x, tid_y, tid_z, tid_grid, tid_dens;
  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  tid_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (tid_x >= nx_local || tid_y >= ny_local || tid_z >= nz_local ) return;

  tid_dens = tid_x + tid_y*nx_local + tid_z*nx_local*ny_local;

  tid_x += n_ghost;
  tid_y += n_ghost;
  tid_z += n_ghost;

  int nx_grid, ny_grid;
  nx_grid = nx_local + 2*n_ghost;
  ny_grid = ny_local + 2*n_ghost;
  tid_grid = tid_x + tid_y*nx_grid + tid_z*nx_grid*ny_grid;

  Real dens;
  dens = src_density_d[tid_grid];

  #ifdef COSMOLOGY
  dens *= cosmo_rho_0_gas;
  #endif

  #ifdef PARTICLES
  dst_density_d[tid_dens] += dens; //Hydro density is added AFTER partices density
  #else
  dst_density_d[tid_dens]  = dens;
  #endif

}

void Grid3D::Copy_Hydro_Density_to_Gravity_GPU(){

  int nx_local, ny_local, nz_local, n_ghost;
  nx_local = Grav.nx_local;
  ny_local = Grav.ny_local;
  nz_local = Grav.nz_local;
  n_ghost  = H.n_ghost;



  // set values for GPU kernels
  int tpb_x = TPBX_GRAV;
  int tpb_y = TPBY_GRAV;
  int tpb_z = TPBZ_GRAV;
  int ngrid_x = (nx_local - 1) / tpb_x + 1;
  int ngrid_y = (ny_local - 1) / tpb_y + 1;
  int ngrid_z = (nz_local - 1) / tpb_z + 1;
  // number of blocks per 1D grid
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);

  Real cosmo_rho_0_gas;

  #ifdef COSMOLOGY
  cosmo_rho_0_gas = Cosmo.rho_0_gas;
  #else
  cosmo_rho_0_gas = 1.0;
  #endif

  #ifndef HYDRO_GPU
  //Copy the hydro density from host to device
  int n_cells_total = ( nx_local + 2*n_ghost ) * ( ny_local + 2*n_ghost ) * ( nz_local + 2*n_ghost );
  CudaSafeCall( cudaMemcpy(C.d_density, C.density, n_cells_total*sizeof(Real), cudaMemcpyHostToDevice) );
  #endif//HYDRO_GPU

  //Copy the density from the device array to the Poisson input density array
  hipLaunchKernelGGL(Copy_Hydro_Density_to_Gravity_Kernel, dim3dGrid, dim3dBlock, 0, 0,  C.d_density, Grav.F.density_d, nx_local, ny_local, nz_local, n_ghost, cosmo_rho_0_gas);


}

void __global__ Extrapolate_Grav_Potential_Kernel( Real *dst_potential, Real *src_potential_0, Real *src_potential_1,
        int nx_pot, int ny_pot, int nz_pot, int nx_grid, int ny_grid, int nz_grid, int n_offset,
        Real dt_now, Real dt_prev, bool INITIAL,  Real cosmo_factor ){

  int tid_x, tid_y, tid_z, tid_grid, tid_pot;
  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  tid_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (tid_x >= nx_pot || tid_y >= ny_pot || tid_z >= nz_pot ) return;

  tid_pot = tid_x + tid_y*nx_pot + tid_z*nx_pot*ny_pot;

  tid_x += n_offset;
  tid_y += n_offset;
  tid_z += n_offset;

  tid_grid = tid_x + tid_y*nx_grid + tid_z*nx_grid*ny_grid;

  Real pot_now, pot_prev, pot_extrp;
  pot_now = src_potential_0[tid_pot]; //Potential at the n-th timestep
  if ( INITIAL ){
    pot_extrp = pot_now; //The first timestep the extrapolated potential is phi_0
  } else {
    pot_prev = src_potential_1[tid_pot]; //Potential at the (n-1)-th timestep ( previous step )
    //Compute the extrapolated potential from phi_n-1 and phi_n
    pot_extrp = pot_now  + 0.5 * dt_now * ( pot_now - pot_prev  ) / dt_prev;
  }

  #ifdef COSMOLOGY
  //For cosmological simulation the potential is transformrd to 'comuving coordinates'
  pot_extrp *= cosmo_factor;
  #endif

  //Save the extrapolated potential
  dst_potential[tid_grid] = pot_extrp;
  //Set phi_n-1 = phi_n, to use it during the next step
  src_potential_1[tid_pot] = pot_now;
}

void Grid3D::Extrapolate_Grav_Potential_GPU(){

  int nx_pot, ny_pot, nz_pot;
  nx_pot = Grav.nx_local + 2*N_GHOST_POTENTIAL;
  ny_pot = Grav.ny_local + 2*N_GHOST_POTENTIAL;
  nz_pot = Grav.nz_local + 2*N_GHOST_POTENTIAL;

  int n_ghost_grid, nx_grid, ny_grid, nz_grid;
  n_ghost_grid = H.n_ghost;
  nx_grid = Grav.nx_local + 2*n_ghost_grid;
  ny_grid = Grav.ny_local + 2*n_ghost_grid;
  nz_grid = Grav.nz_local + 2*n_ghost_grid;

  int n_offset = n_ghost_grid - N_GHOST_POTENTIAL;


  Real dt_now, dt_prev, cosmo_factor;
  dt_now = Grav.dt_now;
  dt_prev = Grav.dt_prev;

  #ifdef COSMOLOGY
  cosmo_factor = Cosmo.current_a * Cosmo.current_a / Cosmo.phi_0_gas;
  #else
  cosmo_factor = 1.0;
  #endif

  // set values for GPU kernels
  int tpb_x = TPBX_GRAV;
  int tpb_y = TPBY_GRAV;
  int tpb_z = TPBZ_GRAV;
  int ngrid_x = (nx_pot - 1) / tpb_x + 1;
  int ngrid_y = (ny_pot - 1) / tpb_y + 1;
  int ngrid_z = (nz_pot - 1) / tpb_z + 1;
  // number of blocks per 1D grid
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);

  hipLaunchKernelGGL(Extrapolate_Grav_Potential_Kernel, dim3dGrid, dim3dBlock, 0, 0, C.d_Grav_potential, Grav.F.potential_d, Grav.F.potential_1_d, nx_pot, ny_pot, nz_pot, nx_grid, ny_grid, nz_grid, n_offset, dt_now, dt_prev, Grav.INITIAL, cosmo_factor );

}





#endif //GRAVITY

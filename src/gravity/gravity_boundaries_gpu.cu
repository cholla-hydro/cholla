#if defined(GRAVITY) && defined(GRAVITY_GPU)


#include <cmath>
#include "../io/io.h"
#include "../grid/grid3D.h"
#include "../gravity/grav3D.h"


#if defined (GRAV_ISOLATED_BOUNDARY_X) || defined (GRAV_ISOLATED_BOUNDARY_Y) || defined(GRAV_ISOLATED_BOUNDARY_Z)

void __global__ Set_Potential_Boundaries_Isolated_kernel(int direction, int side, int size_buffer, int n_i, int n_j, int nx, int ny, int nz, int n_ghost, Real *potential_d, Real *pot_boundary_d   ){

  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_buffer, tid_pot;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;

  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost ) return;

  tid_buffer = tid_i + tid_j*n_i + tid_k*n_i*n_j;

  if ( direction == 0 ){
    if ( side == 0 ) tid_pot = ( tid_k )                + (tid_i+n_ghost)*nx + (tid_j+n_ghost)*nx*ny;
    if ( side == 1 ) tid_pot = ( nx - n_ghost + tid_k ) + (tid_i+n_ghost)*nx + (tid_j+n_ghost)*nx*ny;
  }
  if ( direction == 1 ){
    if ( side == 0 ) tid_pot = (tid_i+n_ghost) + ( tid_k )*nx                 + (tid_j+n_ghost)*nx*ny;
    if ( side == 1 ) tid_pot = (tid_i+n_ghost) + ( ny - n_ghost + tid_k  )*nx + (tid_j+n_ghost)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_pot = (tid_i+n_ghost) + (tid_j+n_ghost)*nx + ( tid_k )*nx*ny;
    if ( side == 1 ) tid_pot = (tid_i+n_ghost) + (tid_j+n_ghost)*nx + ( nz - n_ghost + tid_k  )*nx*ny;
  }

  potential_d[tid_pot] = pot_boundary_d[tid_buffer];
}

void Grid3D::Set_Potential_Boundaries_Isolated_GPU( int direction, int side, int *flags ){

  int n_i, n_j, n_ghost, size_buffer;
  int nx_g, ny_g, nz_g;
  n_ghost = N_GHOST_POTENTIAL;
  nx_g = Grav.nx_local + 2*n_ghost;
  ny_g = Grav.ny_local + 2*n_ghost;
  nz_g = Grav.nz_local + 2*n_ghost;


  Real *pot_boundary_h, *pot_boundary_d;
  #ifdef GRAV_ISOLATED_BOUNDARY_X
  if ( direction == 0 ){
    n_i = Grav.ny_local;
    n_j = Grav.nz_local;
    if ( side == 0 ) pot_boundary_h = Grav.F.pot_boundary_x0;
    if ( side == 1 ) pot_boundary_h = Grav.F.pot_boundary_x1;
    if ( side == 0 ) pot_boundary_d = Grav.F.pot_boundary_x0_d;
    if ( side == 1 ) pot_boundary_d = Grav.F.pot_boundary_x1_d;
  }
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
  if ( direction == 1 ){
    n_i = Grav.nx_local;
    n_j = Grav.nz_local;
    if ( side == 0 ) pot_boundary_h = Grav.F.pot_boundary_y0;
    if ( side == 1 ) pot_boundary_h = Grav.F.pot_boundary_y1;
    if ( side == 0 ) pot_boundary_d = Grav.F.pot_boundary_y0_d;
    if ( side == 1 ) pot_boundary_d = Grav.F.pot_boundary_y1_d;
  }
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  if ( direction == 2 ){
    n_i = Grav.nx_local;
    n_j = Grav.ny_local;
    if ( side == 0 ) pot_boundary_h = Grav.F.pot_boundary_z0;
    if ( side == 1 ) pot_boundary_h = Grav.F.pot_boundary_z1;
    if ( side == 0 ) pot_boundary_d = Grav.F.pot_boundary_z0_d;
    if ( side == 1 ) pot_boundary_d = Grav.F.pot_boundary_z1_d;
  }
  #endif

  size_buffer = N_GHOST_POTENTIAL * n_i * n_j;

  // set values for GPU kernels
  int ngrid = ( size_buffer - 1 ) / TPB_GRAV + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_GRAV, 1, 1);

  //Copy the boundary array from host to device
  cudaMemcpy( pot_boundary_d, pot_boundary_h, size_buffer*sizeof(Real), cudaMemcpyHostToDevice );
  cudaDeviceSynchronize();

  // Copy the potential boundary from buffer to potential array
  hipLaunchKernelGGL( Set_Potential_Boundaries_Isolated_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i, n_j, nx_g, ny_g, nz_g, n_ghost, Grav.F.potential_d, pot_boundary_d );

}


#endif //GRAV_ISOLATED_BOUNDARY


void __global__ Set_Potential_Boundaries_Periodic_kernel(int direction, int side, int n_i, int n_j, int nx, int ny, int nz, int n_ghost, Real *potential_d ){
  
  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_src, tid_dst;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;
  
  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost ) return;
  
  if ( direction == 0 ){
    if ( side == 0 ) tid_src = ( nx - 2*n_ghost + tid_k )  + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 0 ) tid_dst = ( tid_k )                   + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 1 ) tid_src = ( n_ghost + tid_k  )        + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dst = ( nx - n_ghost + tid_k )    + (tid_i)*nx  + (tid_j)*nx*ny;

  }
  if ( direction == 1 ){
    if ( side == 0 ) tid_src = (tid_i) + ( ny - 2*n_ghost + tid_k  )*nx  + (tid_j)*nx*ny;
    if ( side == 0 ) tid_dst = (tid_i) + ( tid_k )*nx                    + (tid_j)*nx*ny;
    if ( side == 1 ) tid_src = (tid_i) + ( n_ghost + tid_k  )*nx         + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dst = (tid_i) + ( ny - n_ghost + tid_k )*nx     + (tid_j)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_src = (tid_i) + (tid_j)*nx + ( nz - 2*n_ghost + tid_k  )*nx*ny;
    if ( side == 0 ) tid_dst = (tid_i) + (tid_j)*nx + ( tid_k  )*nx*ny;
    if ( side == 1 ) tid_src = (tid_i) + (tid_j)*nx + ( n_ghost + tid_k  )*nx*ny;
    if ( side == 1 ) tid_dst = (tid_i) + (tid_j)*nx + ( nz - n_ghost + tid_k  )*nx*ny;
  }
  
  potential_d[tid_dst] = potential_d[tid_src];
  
}


void Grid3D::Set_Potential_Boundaries_Periodic_GPU( int direction, int side, int *flags ){
  
  int n_i, n_j, n_ghost, size;
  int nx_g, ny_g, nz_g;
  n_ghost = N_GHOST_POTENTIAL;
  nx_g = Grav.nx_local + 2*n_ghost;
  ny_g = Grav.ny_local + 2*n_ghost;
  nz_g = Grav.nz_local + 2*n_ghost;

  if ( direction == 0 ){
    n_i = ny_g;
    n_j = nz_g;
  }
  if ( direction == 1 ){
    n_i = nx_g;
    n_j = nz_g;
  }
  if ( direction == 2 ){
    n_i = nx_g;
    n_j = ny_g;
  }

  size = N_GHOST_POTENTIAL * n_i * n_j;

  // set values for GPU kernels
  int ngrid = ( size - 1 ) / TPB_GRAV + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_GRAV, 1, 1);

  // Copy the potential boundary from buffer to potential array
  hipLaunchKernelGGL( Set_Potential_Boundaries_Periodic_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, n_i, n_j, nx_g, ny_g, nz_g, n_ghost, Grav.F.potential_d );


}


void __global__ Set_Potential_Boundaries_Periodic_kernel(int direction, int side, int n_i, int n_j, int nx, int ny, int nz, int n_ghost, Real *potential_d ){
  
  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_src, tid_dst;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;
  
  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost ) return;
  
  if ( direction == 0 ){
    if ( side == 0 ) tid_src = ( nx - 2*n_ghost + tid_k )  + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 0 ) tid_dst = ( tid_k )                   + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 1 ) tid_src = ( n_ghost + tid_k  )        + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dst = ( nx - n_ghost + tid_k )    + (tid_i)*nx  + (tid_j)*nx*ny;

  }
  if ( direction == 1 ){
    if ( side == 0 ) tid_src = (tid_i) + ( ny - 2*n_ghost + tid_k  )*nx  + (tid_j)*nx*ny;
    if ( side == 0 ) tid_dst = (tid_i) + ( tid_k )*nx                    + (tid_j)*nx*ny;
    if ( side == 1 ) tid_src = (tid_i) + ( n_ghost + tid_k  )*nx         + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dst = (tid_i) + ( ny - n_ghost + tid_k )*nx     + (tid_j)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_src = (tid_i) + (tid_j)*nx + ( nz - 2*n_ghost + tid_k  )*nx*ny;
    if ( side == 0 ) tid_dst = (tid_i) + (tid_j)*nx + ( tid_k  )*nx*ny;
    if ( side == 1 ) tid_src = (tid_i) + (tid_j)*nx + ( n_ghost + tid_k  )*nx*ny;
    if ( side == 1 ) tid_dst = (tid_i) + (tid_j)*nx + ( nz - n_ghost + tid_k  )*nx*ny;
  }
  
  potential_d[tid_dst] = potential_d[tid_src];
  
}


void Grid3D::Set_Potential_Boundaries_Periodic_GPU( int direction, int side, int *flags ){
  
  int n_i, n_j, n_ghost, size;
  int nx_g, ny_g, nz_g;
  n_ghost = N_GHOST_POTENTIAL;
  nx_g = Grav.nx_local + 2*n_ghost;
  ny_g = Grav.ny_local + 2*n_ghost;
  nz_g = Grav.nz_local + 2*n_ghost;

  if ( direction == 0 ){
    n_i = ny_g;
    n_j = nz_g;
  }
  if ( direction == 1 ){
    n_i = nx_g;
    n_j = nz_g;
  }
  if ( direction == 2 ){
    n_i = nx_g;
    n_j = ny_g;
  }

  size = N_GHOST_POTENTIAL * n_i * n_j;

  // set values for GPU kernels
  int ngrid = ( size - 1 ) / TPB_GRAV + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_GRAV, 1, 1);

  // Copy the potential boundary from buffer to potential array
  hipLaunchKernelGGL( Set_Potential_Boundaries_Periodic_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, n_i, n_j, nx_g, ny_g, nz_g, n_ghost, Grav.F.potential_d );


}

__global__ void Load_Transfer_Buffer_GPU_kernel( int direction, int side, int size_buffer, int n_i, int n_j, int nx, int ny, int nz, int n_ghost_transfer, int n_ghost_potential, Real *potential_d, Real *transfer_buffer_d   ){

  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_buffer, tid_pot;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;

  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost_transfer ) return;

  tid_buffer = tid_i + tid_j*n_i + tid_k*n_i*n_j;

  if ( direction == 0 ){
    if ( side == 0 ) tid_pot = ( n_ghost_potential + tid_k  )                        + (tid_i)*nx + (tid_j)*nx*ny;
    if ( side == 1 ) tid_pot = ( nx - n_ghost_potential - n_ghost_transfer + tid_k ) + (tid_i)*nx + (tid_j)*nx*ny;
  }
  if ( direction == 1 ){
    if ( side == 0 ) tid_pot = (tid_i) + ( n_ghost_potential + tid_k  )*nx                         + (tid_j)*nx*ny;
    if ( side == 1 ) tid_pot = (tid_i) + ( ny - n_ghost_potential - n_ghost_transfer + tid_k  )*nx + (tid_j)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_pot = (tid_i) + (tid_j)*nx + ( n_ghost_potential + tid_k  )*nx*ny;
    if ( side == 1 ) tid_pot = (tid_i) + (tid_j)*nx + ( nz - n_ghost_potential - n_ghost_transfer + tid_k  )*nx*ny;
  }
  transfer_buffer_d[tid_buffer] = potential_d[tid_pot];

}

int Grid3D::Load_Gravity_Potential_To_Buffer_GPU( int direction, int side, Real *buffer, int buffer_start  ){

  // printf( "Loading Gravity Buffer: Dir %d  side: %d \n", direction, side );
  int nx_pot, ny_pot, nz_pot, size_buffer, n_ghost_potential, n_ghost_transfer, n_i, n_j, ngrid;;
  n_ghost_potential = N_GHOST_POTENTIAL;
  n_ghost_transfer  = N_GHOST_POTENTIAL;
  nx_pot = Grav.nx_local + 2*n_ghost_potential;
  ny_pot = Grav.ny_local + 2*n_ghost_potential;
  nz_pot = Grav.nz_local + 2*n_ghost_potential;

  if ( direction == 0 ){
    n_i = ny_pot;
    n_j = nz_pot;
  }
  if ( direction == 1 ){
    n_i = nx_pot;
    n_j = nz_pot;
  }
  if ( direction == 2 ){
    n_i = nx_pot;
    n_j = ny_pot;
  }

  size_buffer = n_ghost_transfer * n_i * n_j;

  // set values for GPU kernels
  ngrid = ( size_buffer - 1 ) / TPB_GRAV + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_GRAV, 1, 1);

  Real *potential_d;
  potential_d = (Real *)Grav.F.potential_d;

  Real *send_buffer_d;
  send_buffer_d = buffer;

  hipLaunchKernelGGL( Load_Transfer_Buffer_GPU_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i, n_j,  nx_pot, ny_pot, nz_pot, n_ghost_transfer, n_ghost_potential, potential_d, send_buffer_d  );
  CHECK(cudaDeviceSynchronize());

  return size_buffer;
}

__global__ void Unload_Transfer_Buffer_GPU_kernel( int direction, int side, int size_buffer, int n_i, int n_j, int nx, int ny, int nz, int n_ghost_transfer, int n_ghost_potential, Real *potential_d, Real *transfer_buffer_d   ){

  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_buffer, tid_pot;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;

  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost_transfer ) return;

  tid_buffer = tid_i + tid_j*n_i + tid_k*n_i*n_j;

  if ( direction == 0 ){
    if ( side == 0 ) tid_pot = ( n_ghost_potential - n_ghost_transfer + tid_k  ) + (tid_i)*nx + (tid_j)*nx*ny;
    if ( side == 1 ) tid_pot = ( nx - n_ghost_potential + tid_k )                + (tid_i)*nx + (tid_j)*nx*ny;
  }
  if ( direction == 1 ){
    if ( side == 0 ) tid_pot = (tid_i) + ( n_ghost_potential - n_ghost_transfer + tid_k  )*nx + (tid_j)*nx*ny;
    if ( side == 1 ) tid_pot = (tid_i) + ( ny - n_ghost_potential + tid_k  )*nx               + (tid_j)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_pot = (tid_i) + (tid_j)*nx + ( n_ghost_potential - n_ghost_transfer + tid_k  )*nx*ny;
    if ( side == 1 ) tid_pot = (tid_i) + (tid_j)*nx + ( nz - n_ghost_potential + tid_k  )*nx*ny;
  }
  potential_d[tid_pot] = transfer_buffer_d[tid_buffer];

}


void Grid3D::Unload_Gravity_Potential_from_Buffer_GPU( int direction, int side, Real *buffer, int buffer_start  ){

  // printf( "Loading Gravity Buffer: Dir %d  side: %d \n", direction, side );
  int nx_pot, ny_pot, nz_pot, size_buffer, n_ghost_potential, n_ghost_transfer, n_i, n_j, ngrid;;
  n_ghost_potential = N_GHOST_POTENTIAL;
  n_ghost_transfer  = N_GHOST_POTENTIAL;
  nx_pot = Grav.nx_local + 2*n_ghost_potential;
  ny_pot = Grav.ny_local + 2*n_ghost_potential;
  nz_pot = Grav.nz_local + 2*n_ghost_potential;

  if ( direction == 0 ){
    n_i = ny_pot;
    n_j = nz_pot;
  }
  if ( direction == 1 ){
    n_i = nx_pot;
    n_j = nz_pot;
  }
  if ( direction == 2 ){
    n_i = nx_pot;
    n_j = ny_pot;
  }

  size_buffer = n_ghost_transfer * n_i * n_j;

  // set values for GPU kernels
  ngrid = ( size_buffer - 1 ) / TPB_GRAV + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_GRAV, 1, 1);

  Real *potential_d;
  potential_d = (Real *)Grav.F.potential_d;

  Real *recv_buffer_d;
  recv_buffer_d = buffer;

  hipLaunchKernelGGL( Unload_Transfer_Buffer_GPU_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i, n_j,  nx_pot, ny_pot, nz_pot, n_ghost_transfer, n_ghost_potential, potential_d, recv_buffer_d  );

}


#endif //GRAVITY

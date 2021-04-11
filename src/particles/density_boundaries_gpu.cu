#if defined(PARTICLES_GPU) && defined(GRAVITY_GPU)

#include "../io.h"
#include "../grid3D.h"
#include "particles_3D.h"
#include <iostream>





#ifdef MPI_CHOLLA



__global__ void Load_Particles_Density_Boundary_to_Buffer_kernel( int direction, int side, int n_i, int n_j, int nx, int ny, int nz, int n_ghost, Real *density_d, Real *transfer_buffer_d   ){
  
  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_buffer, tid_dens;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;
  
  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost ) return;
  
  tid_buffer = tid_i + tid_j*n_i + tid_k*n_i*n_j;
  
  if ( direction == 0 ){
    if ( side == 0 ) tid_dens = ( tid_k )                 + (tid_i)*nx + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dens = ( nx - n_ghost + tid_k )  + (tid_i)*nx + (tid_j)*nx*ny;
  }  
  if ( direction == 1 ){
    if ( side == 0 ) tid_dens = (tid_i) + ( tid_k )*nx                 + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dens = (tid_i) + ( ny - n_ghost + tid_k )*nx  + (tid_j)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_dens = (tid_i) + (tid_j)*nx + ( tid_k )*nx*ny;
    if ( side == 1 ) tid_dens = (tid_i) + (tid_j)*nx + ( nz - n_ghost + tid_k  )*nx*ny;  
  }
  transfer_buffer_d[tid_buffer] = density_d[tid_dens];
  
}





int Grid3D::Load_Particles_Density_Boundary_to_Buffer_GPU( int direction, int side, Real *buffer  ){
  
  int n_ghost, nx_g, ny_g, nz_g, size_buffer, ngrid, n_i, n_j;
  n_ghost = Particles.G.n_ghost_particles_grid;
  nx_g = Particles.G.nx_local + 2*n_ghost;
  ny_g = Particles.G.ny_local + 2*n_ghost;
  nz_g = Particles.G.nz_local + 2*n_ghost;
  
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
    
  size_buffer = n_ghost * n_i * n_j;
  
  // set values for GPU kernels
  ngrid = ( size_buffer - 1 ) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  Real *density_d;
  density_d = (Real *)Particles.G.density_dev;
  
  Real *send_buffer_d;
  #ifdef GPU_MPI
  send_buffer_d = buffer;
  #else
  if ( direction == 0 ){
    if ( side == 0 ) send_buffer_d = (Real *)Particles.G.send_buffer_x0_d;
    if ( side == 1 ) send_buffer_d = (Real *)Particles.G.send_buffer_x1_d;
  }
  if ( direction == 1 ){
    if ( side == 0 ) send_buffer_d = (Real *)Particles.G.send_buffer_y0_d;
    if ( side == 1 ) send_buffer_d = (Real *)Particles.G.send_buffer_y1_d;
  }
  if ( direction == 2 ){
    if ( side == 0 ) send_buffer_d = (Real *)Particles.G.send_buffer_z0_d;
    if ( side == 1 ) send_buffer_d = (Real *)Particles.G.send_buffer_z1_d;
  }
  #endif  
  
  hipLaunchKernelGGL( Load_Particles_Density_Boundary_to_Buffer_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, n_i, n_j, nx_g, ny_g, nz_g, n_ghost, density_d, send_buffer_d  );
  
  #ifndef GPU_MPI
  //Copy the device buffer back to the host send buffer
  cudaMemcpy( buffer, send_buffer_d, size_buffer*sizeof(Real), cudaMemcpyDeviceToHost );
  #endif
  cudaDeviceSynchronize();
  
  return size_buffer;
}




__global__ void Unload_Particles_Density_Boundary_to_Buffer_kernel( int direction, int side, int n_i, int n_j, int nx, int ny, int nz, int n_ghost, Real *density_d, Real *transfer_buffer_d   ){
  
  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_buffer, tid_dens;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;
  
  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost ) return;
  
  tid_buffer = tid_i + tid_j*n_i + tid_k*n_i*n_j;
  
  if ( direction == 0 ){
    if ( side == 0 ) tid_dens = ( n_ghost + tid_k )        + (tid_i)*nx + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dens = ( nx - 2*n_ghost + tid_k ) + (tid_i)*nx + (tid_j)*nx*ny;
  }  
  if ( direction == 1 ){
    if ( side == 0 ) tid_dens = (tid_i) + ( n_ghost + tid_k )*nx         + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dens = (tid_i) + ( ny - 2*n_ghost + tid_k )*nx  + (tid_j)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_dens = (tid_i) + (tid_j)*nx + ( n_ghost + tid_k )*nx*ny;
    if ( side == 1 ) tid_dens = (tid_i) + (tid_j)*nx + ( nz - 2* n_ghost + tid_k  )*nx*ny;  
  }
  density_d[tid_dens] += transfer_buffer_d[tid_buffer];
  
}




void Grid3D::Unload_Particles_Density_Boundary_From_Buffer_GPU( int direction, int side, Real *buffer  ){
  
  int n_ghost, nx_g, ny_g, nz_g, size_buffer, ngrid, n_i, n_j;
  n_ghost = Particles.G.n_ghost_particles_grid;
  nx_g = Particles.G.nx_local + 2*n_ghost;
  ny_g = Particles.G.ny_local + 2*n_ghost;
  nz_g = Particles.G.nz_local + 2*n_ghost;
  
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
    
  size_buffer = n_ghost * n_i * n_j;
  
  // set values for GPU kernels
  ngrid = ( size_buffer - 1 ) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  Real *density_d;
  density_d = (Real *)Particles.G.density_dev;
  
  Real *recv_buffer_d;
  #ifdef GPU_MPI
  recv_buffer_d = buffer;
  #else
  if ( direction == 0 ){
    if ( side == 0 ) recv_buffer_d = (Real *)Particles.G.recv_buffer_x0_d;
    if ( side == 1 ) recv_buffer_d = (Real *)Particles.G.recv_buffer_x1_d;
  }
  if ( direction == 1 ){
    if ( side == 0 ) recv_buffer_d = (Real *)Particles.G.recv_buffer_y0_d;
    if ( side == 1 ) recv_buffer_d = (Real *)Particles.G.recv_buffer_y1_d;
  }
  if ( direction == 2 ){
    if ( side == 0 ) recv_buffer_d = (Real *)Particles.G.recv_buffer_z0_d;
    if ( side == 1 ) recv_buffer_d = (Real *)Particles.G.recv_buffer_z1_d;
  }
  #endif
    
  #ifndef GPU_MPI
  //Copy the device buffer back to the host recv buffer
  cudaMemcpy( recv_buffer_d, buffer, size_buffer*sizeof(Real), cudaMemcpyHostToDevice );
  #endif
  cudaDeviceSynchronize();
  
  hipLaunchKernelGGL( Unload_Particles_Density_Boundary_to_Buffer_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, n_i, n_j, nx_g, ny_g, nz_g, n_ghost, density_d, recv_buffer_d  );
  
}



#endif//MPI_CHOLLA

#endif//PARTICLES_GPU
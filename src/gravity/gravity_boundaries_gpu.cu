#if defined(GRAVITY) && defined(GRAVITY_GPU)


#include <cmath>
#include "../io.h"
#include "../grid3D.h"
#include "grav3D.h"

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
  #ifdef GPU_MPI
  send_buffer_d = buffer;
  #else
  if ( direction == 0 ){
    if ( side == 0 ) send_buffer_d = (Real *)Grav.F.send_buffer_potential_x0_d;
    if ( side == 1 ) send_buffer_d = (Real *)Grav.F.send_buffer_potential_x1_d;
  }
  if ( direction == 1 ){
    if ( side == 0 ) send_buffer_d = (Real *)Grav.F.send_buffer_potential_y0_d;
    if ( side == 1 ) send_buffer_d = (Real *)Grav.F.send_buffer_potential_y1_d;
  }
  if ( direction == 2 ){
    if ( side == 0 ) send_buffer_d = (Real *)Grav.F.send_buffer_potential_z0_d;
    if ( side == 1 ) send_buffer_d = (Real *)Grav.F.send_buffer_potential_z1_d;
  }
  #endif  
  
  hipLaunchKernelGGL( Load_Transfer_Buffer_GPU_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i, n_j,  nx_pot, ny_pot, nz_pot, n_ghost_transfer, n_ghost_potential, potential_d, send_buffer_d  );
    
  // NOTE: I couldn't figure out how top do it using OpenMP
  // int i, j, k, indx, indx_buff;
  // if ( direction == 0 ){
  //   #pragma omp target teams distribute parallel for collapse ( 2 ) \
  //           private ( indx, indx_buff ) \
  //           firstprivate ( side ) \
  //           is_device_ptr ( send_buffer_d, potential_d )
  //   for ( k=0; k<nz_g; k++ ){
  //     for ( j=0; j<ny_g; j++ ){
  //       for ( i=0; i<nGHST; i++ ){
  //         if ( side == 0 ) indx = (i+nGHST) + (j)*nx_g + (k)*nx_g*ny_g;
  //         if ( side == 1 ) indx = (nx_g - 2*nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
  //         indx_buff = (j) + (k)*ny_g + i*ny_g*nz_g ;
  //         send_buffer_d[indx_buff] = potential_d[indx];
  //       }
  //     }
  //   }
  // }
  // 
  // 
  // if ( direction == 1 ){
  //   #pragma omp target teams distribute parallel for collapse ( 3 ) \
  //           private ( indx, indx_buff ) \
  //           firstprivate ( side ) \
  //           is_device_ptr ( send_buffer_d, potential_d )
  //   for ( k=0; k<nz_g; k++ ){
  //     for ( j=0; j<nGHST; j++ ){
  //       for ( i=0; i<nx_g; i++ ){
  //         if ( side == 0 ) indx = (i) + (j+nGHST)*nx_g + (k)*nx_g*ny_g;
  //         if ( side == 1 ) indx = (i) + (ny_g - 2*nGHST + j)*nx_g + (k)*nx_g*ny_g;
  //         indx_buff = (i) + (k)*nx_g + j*nx_g*nz_g ;
  //         send_buffer_d[indx_buff] = potential_d[indx];
  //       }
  //     }
  //   }
  // }
  // 
  // if ( direction == 1 ){
  //   for ( k=0; k<nGHST; k++ ){
  //     for ( j=0; j<ny_g; j++ ){
  //       for ( i=0; i<nx_g; i++ ){
  //         if ( side == 0 ) indx = (i) + (j)*nx_g + (k+nGHST)*nx_g*ny_g;
  //         if ( side == 1 ) indx = (i) + (j)*nx_g + (nz_g - 2*nGHST + k)*nx_g*ny_g;
  //         indx_buff = (i) + (j)*nx_g + k*nx_g*ny_g ;
  //         send_buffer_d[indx_buff] = potential_d[indx];
  //       }
  //     }
  //   }
  // }
  // 
   
  #ifndef GPU_MPI
  //Copy the device buffer back to the host send buffer
  cudaMemcpy( buffer, send_buffer_d, size_buffer*sizeof(Real), cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();
  #endif
  
  // printf( "Loaded Gravity Buffer \n" );
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
  #ifdef GPU_MPI
  recv_buffer_d = buffer;
  #else
  if ( direction == 0 ){
    if ( side == 0 ) recv_buffer_d = (Real *)Grav.F.send_buffer_potential_x0_d;
    if ( side == 1 ) recv_buffer_d = (Real *)Grav.F.send_buffer_potential_x1_d;
  }
  if ( direction == 1 ){
    if ( side == 0 ) recv_buffer_d = (Real *)Grav.F.send_buffer_potential_y0_d;
    if ( side == 1 ) recv_buffer_d = (Real *)Grav.F.send_buffer_potential_y1_d;
  }
  if ( direction == 2 ){
    if ( side == 0 ) recv_buffer_d = (Real *)Grav.F.send_buffer_potential_z0_d;
    if ( side == 1 ) recv_buffer_d = (Real *)Grav.F.send_buffer_potential_z1_d;
  }
  #endif  
  
  #ifndef GPU_MPI
  //Copy the host recv buffer to the device recv buffer
  cudaMemcpy( recv_buffer_d, buffer, size_buffer*sizeof(Real), cudaMemcpyHostToDevice );
  cudaDeviceSynchronize();
  #endif

  hipLaunchKernelGGL( Unload_Transfer_Buffer_GPU_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i, n_j,  nx_pot, ny_pot, nz_pot, n_ghost_transfer, n_ghost_potential, potential_d, recv_buffer_d  );
  
}


#endif //GRAVITY

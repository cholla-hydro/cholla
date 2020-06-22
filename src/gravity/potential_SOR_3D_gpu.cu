#if defined(CUDA) && defined(GRAVITY) && defined(SOR)

#include "potential_SOR_3D.h"
#include"../global_cuda.h"
#include "../io.h"


#define TPB_SOR 1024


void Potential_SOR_3D::Allocate_Array_GPU_Real( Real **array_dev, grav_int_t size ){
  cudaMalloc( (void**)array_dev, size*sizeof(Real));
  CudaCheckError();
}

void Potential_SOR_3D::Allocate_Array_GPU_bool( bool **array_dev, grav_int_t size ){
  cudaMalloc( (void**)array_dev, size*sizeof(bool));
  CudaCheckError();
}

void Potential_SOR_3D::Free_Array_GPU_Real( Real *array_dev ){
  cudaFree( array_dev );
  CudaCheckError();
}

void Potential_SOR_3D::Free_Array_GPU_bool( bool *array_dev ){
  cudaFree( array_dev );
  CudaCheckError();
}

__global__ void Copy_Input_Kernel( int n_cells, Real *input_d, Real *density_d, Real Grav_Constant, Real dens_avrg, Real current_a ){
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_cells ) return;

  #ifdef COSMOLOGY
  density_d[tid] = 4 * M_PI * Grav_Constant * ( input_d[tid] - dens_avrg ) / current_a;
  #else
  density_d[tid] = 4 * M_PI * Grav_Constant * input_d[tid];
  #endif
  // if (tid == 0) printf("dens: %f\n", density_d[tid]);
}


void Potential_SOR_3D::Copy_Input( int n_cells, Real *input_d, Real *input_density_h, Real Grav_Constant, Real dens_avrg, Real current_a ){
  cudaMemcpy( input_d, input_density_h, n_cells*sizeof(Real), cudaMemcpyHostToDevice );
  
  // set values for GPU kernels
  int ngrid =  (n_cells_local + TPB_SOR - 1) / TPB_SOR;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_SOR, 1, 1);
  
  Copy_Input_Kernel<<<dim1dGrid,dim1dBlock>>>( n_cells_local, F.input_d, F.density_d,  Grav_Constant, dens_avrg, current_a  );
}

void Grav3D::Copy_Isolated_Boundary_To_GPU_buffer( Real *isolated_boundary_h, Real *isolated_boundary_d, int boundary_size ){
 cudaMemcpy( isolated_boundary_d, isolated_boundary_h, boundary_size*sizeof(Real), cudaMemcpyHostToDevice );  
}

__global__ void Initialize_Potential_Kernel( Real init_val, Real *potential_d, Real *density_d, int nx, int ny, int nz, int n_ghost ){

  int tid_x, tid_y, tid_z, tid_pot;
  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  tid_z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (tid_x >= nx || tid_y >= ny || tid_z >= nz ) return;  
  
  // tid = tid_x + tid_y*nx + tid_z*nx*ny;
    
  tid_x += n_ghost;
  tid_y += n_ghost;
  tid_z += n_ghost;

  int nx_pot, ny_pot;
  nx_pot = nx + 2*n_ghost;
  ny_pot = ny + 2*n_ghost;
  

  tid_pot = tid_x + tid_y*nx_pot + tid_z*nx_pot*ny_pot;
  potential_d[tid_pot] = init_val;
  
  if ( potential_d[tid_pot] !=1 ) printf("Error phi value: %f\n", potential_d[tid_pot] );
  

  // Real dens = density_d[tid];
  // potential_d[tid_pot] = -dens;
  
}



void Potential_SOR_3D::Initialize_Potential( int nx, int ny, int nz, int n_ghost_potential, Real *potential_d, Real *density_d ){
  // set values for GPU kernels
  int tpb_x = 16;
  int tpb_y = 8;
  int tpb_z = 8;
  int ngrid_x =  (nx_local + tpb_x - 1) / tpb_x;
  int ngrid_y =  (ny_local + tpb_y - 1) / tpb_y;
  int ngrid_z =  (nz_local + tpb_z - 1) / tpb_z;
  // number of blocks per 1D grid  
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block   
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);
  
  Initialize_Potential_Kernel<<<dim3dGrid,dim3dBlock>>>( 1, potential_d, density_d, nx, ny, nz, n_ghost_potential );

}


__global__ void Iteration_Step_SOR( int n_cells, Real *density_d, Real *potential_d, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real omega, int parity, Real epsilon,  bool *converged_d ){
  
  int tid_x, tid_y, tid_z, tid, tid_pot;
  tid_x = 2*( blockIdx.x * blockDim.x + threadIdx.x );
  tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  tid_z = blockIdx.z * blockDim.z + threadIdx.z;
  
  // Make a checkboard 3D grid
  if ( tid_y%2 == 0 ){
    if ( tid_z%2 == parity ) tid_x +=1;
  }
  else if ( (tid_z+1)%2 == parity ) tid_x +=1;
  
  if (tid_x >= nx || tid_y >= ny || tid_z >= nz ) return;  
  
  int nx_pot, ny_pot;
  nx_pot = nx + 2*n_ghost;
  ny_pot = ny + 2*n_ghost;
  // nz_pot = nz + 2*n_ghost;
  
  tid = tid_x + tid_y*nx + tid_z*nx*ny;
  
  tid_x += n_ghost;
  tid_y += n_ghost;
  tid_z += n_ghost;
  tid_pot = tid_x + tid_y*nx_pot + tid_z*nx_pot*ny_pot; 
  
  // //Set neighbors ids
  int indx_l, indx_r, indx_d, indx_u, indx_b, indx_t;
  
  indx_l = tid_x-1;  //Left
  indx_r = tid_x+1;  //Right
  indx_d = tid_y-1;  //Down
  indx_u = tid_y+1;  //Up
  indx_b = tid_z-1;  //Bottom
  indx_t = tid_z+1;  //Top
  
  //Boundary Conditions are loaded to the potential array, the natural indices work!
  
  // //Periodic Boundary conditions
  // indx_l = tid_x == n_ghost          ?    nx_pot-n_ghost-1 : tid_x-1;  //Left
  // indx_r = tid_x == nx_pot-n_ghost-1 ?             n_ghost : tid_x+1;  //Right
  // indx_d = tid_y == n_ghost          ?    ny_pot-n_ghost-1 : tid_y-1;  //Down
  // indx_u = tid_y == ny_pot-n_ghost-1 ?             n_ghost : tid_y+1;  //Up
  // indx_b = tid_z == n_ghost          ?    nz_pot-n_ghost-1 : tid_z-1;  //Bottom
  // indx_t = tid_z == nz_pot-n_ghost-1 ?             n_ghost : tid_z+1;  //Top
  // 
  // //Zero Gradient Boundary conditions
  // indx_l = tid_x == n_ghost          ?    tid_x+1 : tid_x-1;  //Left
  // indx_r = tid_x == nx_pot-n_ghost-1 ?    tid_x-1 : tid_x+1;  //Right
  // indx_d = tid_y == n_ghost          ?    tid_y+1 : tid_y-1;  //Down
  // indx_u = tid_y == ny_pot-n_ghost-1 ?    tid_y-1 : tid_y+1;  //Up
  // indx_b = tid_z == n_ghost          ?    tid_z+1 : tid_z-1;  //Bottom
  // indx_t = tid_z == nz_pot-n_ghost-1 ?    tid_z-1 : tid_z+1;  //Top
  
  
  
  Real rho, phi_c, phi_l, phi_r, phi_d, phi_u, phi_b, phi_t, phi_new;
  rho = density_d[tid];
  phi_c = potential_d[tid_pot];
  phi_l = potential_d[ indx_l + tid_y*nx_pot + tid_z*nx_pot*ny_pot ];
  phi_r = potential_d[ indx_r + tid_y*nx_pot + tid_z*nx_pot*ny_pot ];
  phi_d = potential_d[ tid_x + indx_d*nx_pot + tid_z*nx_pot*ny_pot ];
  phi_u = potential_d[ tid_x + indx_u*nx_pot + tid_z*nx_pot*ny_pot ];
  phi_b = potential_d[ tid_x + tid_y*nx_pot + indx_b*nx_pot*ny_pot ];
  phi_t = potential_d[ tid_x + tid_y*nx_pot + indx_t*nx_pot*ny_pot ];
  
  phi_new = (1-omega)*phi_c + omega/6*( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx*dx*rho );
  potential_d[tid_pot] = phi_new;
  
  //Check the residual for the convergence criteria
  if ( ( fabs( ( phi_new - phi_c ) / phi_c ) > epsilon ) ) converged_d[0] = 0;
  // if ( ( fabs( ( phi_new - phi_c ) ) > epsilon ) ) converged_d[0] = 0;
  
  
  
}

void Potential_SOR_3D::Poisson_iteration( int n_cells, int nx, int ny, int nz, int n_ghost_potential, Real dx, Real dy, Real dz, Real omega, Real epsilon, Real *density_d, Real *potential_d, bool *converged_h, bool *converged_d ){
  
  // set values for GPU kernels
  int tpb_x = 16;
  int tpb_y = 8;
  int tpb_z = 8;
  int ngrid_x =  (nx_local + tpb_x - 1) / tpb_x;
  int ngrid_y =  (ny_local + tpb_y - 1) / tpb_y;
  int ngrid_z =  (nz_local + tpb_z - 1) / tpb_z;
  int ngrid_x_half = ( nx_local/2 + tpb_x - 1) / tpb_x;
  // number of blocks per 1D grid  
  dim3 dim3dGrid_half(ngrid_x_half, ngrid_y, ngrid_z);
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block   
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);
  
  cudaMemset( converged_d, 1, sizeof(bool) );
  
  Iteration_Step_SOR<<<dim3dGrid_half,dim3dBlock>>>( n_cells, density_d, potential_d, nx, ny, nz, n_ghost_potential, dx, dy, dz, omega, 0, epsilon, converged_d );
  
  Iteration_Step_SOR<<<dim3dGrid_half,dim3dBlock>>>( n_cells, density_d, potential_d, nx, ny, nz, n_ghost_potential, dx, dy, dz, omega, 1, epsilon, converged_d );
  
  cudaMemcpy( converged_h, converged_d, sizeof(bool), cudaMemcpyDeviceToHost );
  
}


void Potential_SOR_3D::Poisson_iteration_Patial_1( int n_cells, int nx, int ny, int nz, int n_ghost_potential, Real dx, Real dy, Real dz, Real omega, Real epsilon, Real *density_d, Real *potential_d, bool *converged_h, bool *converged_d ){
  
  // set values for GPU kernels
  int tpb_x = 16;
  int tpb_y = 8;
  int tpb_z = 8;
  int ngrid_x =  (nx_local + tpb_x - 1) / tpb_x;
  int ngrid_y =  (ny_local + tpb_y - 1) / tpb_y;
  int ngrid_z =  (nz_local + tpb_z - 1) / tpb_z;
  int ngrid_x_half = ( nx_local/2 + tpb_x - 1) / tpb_x;
  // number of blocks per 1D grid  
  dim3 dim3dGrid_half(ngrid_x_half, ngrid_y, ngrid_z);
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block   
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);
  
  cudaMemset( converged_d, 1, sizeof(bool) );
  
  Iteration_Step_SOR<<<dim3dGrid_half,dim3dBlock>>>( n_cells, density_d, potential_d, nx, ny, nz, n_ghost_potential, dx, dy, dz, omega, 0, epsilon, converged_d );
  
}


void Potential_SOR_3D::Poisson_iteration_Patial_2( int n_cells, int nx, int ny, int nz, int n_ghost_potential, Real dx, Real dy, Real dz, Real omega, Real epsilon, Real *density_d, Real *potential_d, bool *converged_h, bool *converged_d ){
  
  // set values for GPU kernels
  int tpb_x = 16;
  int tpb_y = 8;
  int tpb_z = 8;
  int ngrid_x =  (nx_local + tpb_x - 1) / tpb_x;
  int ngrid_y =  (ny_local + tpb_y - 1) / tpb_y;
  int ngrid_z =  (nz_local + tpb_z - 1) / tpb_z;
  int ngrid_x_half = ( nx_local/2 + tpb_x - 1) / tpb_x;
  // number of blocks per 1D grid  
  dim3 dim3dGrid_half(ngrid_x_half, ngrid_y, ngrid_z);
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block   
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);
  
  Iteration_Step_SOR<<<dim3dGrid_half,dim3dBlock>>>( n_cells, density_d, potential_d, nx, ny, nz, n_ghost_potential, dx, dy, dz, omega, 1, epsilon, converged_d );
  cudaMemcpy( converged_h, converged_d, sizeof(bool), cudaMemcpyDeviceToHost );
  
}


__global__ void Set_Isolated_Boundary_GPU_kernel( int direction, int side, int size_buffer, int n_i, int n_j, int n_ghost, int nx_pot, int ny_pot, int nz_pot,  Real *potential_d, Real *boundary_d   ){
  
  // get a global thread ID
  int nx_local, ny_local, nz_local;
  nx_local = nx_pot - 2*n_ghost;
  ny_local = ny_pot - 2*n_ghost;
  nz_local = nz_pot - 2*n_ghost;
  int tid, tid_i, tid_j, tid_k, tid_buffer, tid_pot;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;
   
  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost ) return;
  
  tid_buffer = tid_i + tid_j*n_i + tid_k*n_i*n_j;
  
  if ( direction == 0 ){
    if ( side == 0 ) tid_pot = (tid_k)                  + (tid_i+n_ghost)*nx_pot + (tid_j+n_ghost)*nx_pot*ny_pot;
    if ( side == 1 ) tid_pot = (tid_k+nx_local+n_ghost) + (tid_i+n_ghost)*nx_pot + (tid_j+n_ghost)*nx_pot*ny_pot;
  }  
  if ( direction == 1 ){
    if ( side == 0 ) tid_pot = (tid_i+n_ghost) + (tid_k)*nx_pot                  + (tid_j+n_ghost)*nx_pot*ny_pot;
    if ( side == 1 ) tid_pot = (tid_i+n_ghost) + (tid_k+ny_local+n_ghost)*nx_pot + (tid_j+n_ghost)*nx_pot*ny_pot;
  }  
  if ( direction == 2 ){
    if ( side == 0 ) tid_pot = (tid_i+n_ghost) + (tid_j+n_ghost)*nx_pot + (tid_k)*nx_pot*ny_pot;
    if ( side == 1 ) tid_pot = (tid_i+n_ghost) + (tid_j+n_ghost)*nx_pot + (tid_k+nz_local+n_ghost)*nx_pot*ny_pot;
  }  
  
  potential_d[tid_pot] = boundary_d[tid_buffer];
  
}

void Potential_SOR_3D::Set_Isolated_Boundary_GPU( int direction, int side,   Real *boundary_d  ){
  
  
  int nx_pot, ny_pot, nz_pot, size_buffer, n_i, n_j, ngrid;
  nx_pot = nx_local + 2*n_ghost;
  ny_pot = ny_local + 2*n_ghost;
  nz_pot = nz_local + 2*n_ghost;
  
  if ( direction == 0 ){
    n_i = ny_local;
    n_j = nz_local;
  }
  if ( direction == 1 ){
    n_i = nx_local;
    n_j = nz_local;
  }
  if ( direction == 2 ){
    n_i = nx_local;
    n_j = ny_local;
  }
  
  size_buffer = n_ghost * n_i * n_j;

  // set values for GPU kernels
  ngrid = ( size_buffer - 1 ) / TPB_SOR + 1;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_SOR, 1, 1);
  
  Set_Isolated_Boundary_GPU_kernel<<<dim1dGrid,dim1dBlock>>>( direction, side, size_buffer, n_i, n_j, n_ghost, nx_pot, ny_pot, nz_pot,  F.potential_d, boundary_d  );

}



void Potential_SOR_3D::Copy_Output( Real *output_potential ){
  cudaMemcpy( output_potential, F.potential_d, n_cells_potential*sizeof(Real), cudaMemcpyDeviceToHost );
}

void Potential_SOR_3D::Copy_Potential_From_Host( Real *output_potential ){
  cudaMemcpy(  F.potential_d, output_potential, n_cells_potential*sizeof(Real), cudaMemcpyHostToDevice );
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



void Potential_SOR_3D::Load_Transfer_Buffer_GPU( int direction, int side, int nx, int ny, int nz, int n_ghost_transfer, int n_ghost_potential, Real *potential_d, Real *transfer_buffer_d  ){
  
  int nx_pot, ny_pot, nz_pot, size_buffer, n_i, n_j, ngrid;
  nx_pot = nx + 2*n_ghost_potential;
  ny_pot = ny + 2*n_ghost_potential;
  nz_pot = nz + 2*n_ghost_potential;
  
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
  ngrid = ( size_buffer - 1 ) / TPB_SOR + 1;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_SOR, 1, 1);
  
  
  Load_Transfer_Buffer_GPU_kernel<<<dim1dGrid,dim1dBlock>>>( direction, side, size_buffer, n_i, n_j,  nx_pot, ny_pot, nz_pot, n_ghost_transfer, n_ghost_potential, potential_d, transfer_buffer_d  );
  
}

void Potential_SOR_3D::Unload_Transfer_Buffer_GPU( int direction, int side, int nx, int ny, int nz, int n_ghost_transfer, int n_ghost_potential, Real *potential_d, Real *transfer_buffer_d  ){
  
  int nx_pot, ny_pot, nz_pot, size_buffer, n_i, n_j, ngrid;
  nx_pot = nx + 2*n_ghost_potential;
  ny_pot = ny + 2*n_ghost_potential;
  nz_pot = nz + 2*n_ghost_potential;
  
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
  ngrid = ( size_buffer - 1 ) / TPB_SOR + 1;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_SOR, 1, 1);
  
  
  Unload_Transfer_Buffer_GPU_kernel<<<dim1dGrid,dim1dBlock>>>( direction, side, size_buffer, n_i, n_j, nx_pot, ny_pot, nz_pot, n_ghost_transfer, n_ghost_potential, potential_d, transfer_buffer_d  );

  
}

void Potential_SOR_3D::Copy_Transfer_Buffer_To_Host( int size_buffer, Real *transfer_buffer_h, Real *transfer_buffer_d ){
  CudaSafeCall( cudaMemcpy(transfer_buffer_h, transfer_buffer_d, size_buffer*sizeof(Real), cudaMemcpyDeviceToHost ) );
}


void Potential_SOR_3D::Copy_Transfer_Buffer_To_Device( int size_buffer, Real *transfer_buffer_h, Real *transfer_buffer_d ){
  CudaSafeCall( cudaMemcpy(transfer_buffer_d, transfer_buffer_h, size_buffer*sizeof(Real), cudaMemcpyHostToDevice ) );
}


#endif //GRAVITY







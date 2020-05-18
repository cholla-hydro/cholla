#if defined(GRAVITY) && defined(SOR)

#include "potential_SOR_3D.h"
#include"../global_cuda.h"
#include "../io.h"
#include<iostream>

Potential_SOR_3D::Potential_SOR_3D( void ){}

__global__ void Initialize_Potential_Kernel( Real init_val, Real *potential_d, Real *density_d, int nx, int ny, int nz, int n_ghost );

__global__ void Iteration_Step_SOR( int n_cells, Real *density_d, Real *potential_d, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real omega, int parity, Real epsilon, bool *converged_d );


void Potential_SOR_3D::Initialize( Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx_real, Real dy_real, Real dz_real){
 
  Lbox_x = Lx;
  Lbox_y = Ly;
  Lbox_z = Lz;

  nx_total = nx;
  ny_total = ny;
  nz_total = nz;

  nx_local = nx_real;
  ny_local = ny_real;
  nz_local = nz_real;

  dx = dx_real;
  dy = dy_real;
  dz = dz_real;

  n_ghost = N_GHOST_POTENTIAL;

  nx_pot = nx_local + 2*n_ghost;
  ny_pot = ny_local + 2*n_ghost;
  nz_pot = nz_local + 2*n_ghost;
  
  n_cells_local = nx_local*ny_local*nz_local;
  n_cells_potential = nx_pot*ny_pot*nz_pot;
  n_cells_total = nx_total*ny_total*nz_total;
  chprintf( " Using Poisson Solver: SOR\n");
  chprintf( "  SOR: L[ %f %f %f ] N[ %d %d %d ] dx[ %f %f %f ]\n", Lbox_x, Lbox_y, Lbox_z, nx_local, ny_local, nz_local, dx, dy, dz );

  chprintf( "  SOR: Allocating memory...\n");
  AllocateMemory_CPU();
  AllocateMemory_GPU();
  
  bool potential_initialized = false;
  
}


void Potential_SOR_3D::AllocateMemory_CPU( void ){
  F.output_h = (Real *) malloc(n_cells_local*sizeof(Real));
}


void Potential_SOR_3D::AllocateMemory_GPU( void ){
  cudaMalloc( (void**)&F.input_d, n_cells_local*sizeof(Real));
  cudaMalloc( (void**)&F.density_d, n_cells_local*sizeof(Real));
  cudaMalloc( (void**)&F.potential_d, n_cells_potential*sizeof(Real));
  cudaMalloc( (void**)&F.converged_d, sizeof(bool));
  // cudaMalloc( (void**)&F.output_d, n_cells_local*sizeof(Real));
  CudaCheckError();
}

void Potential_SOR_3D::FreeMemory_GPU( void ){
  cudaFree( F.input_d );
  cudaFree( F.density_d );
  cudaFree( F.potential_d );
  cudaFree( F.converged_d );
  // cudaFree( F.output_d );
  CudaCheckError();
}


void Potential_SOR_3D::Reset( void ){
  free( F.output_h );
  FreeMemory_GPU();
}



Real Potential_SOR_3D::Get_Potential( Real *input_density,  Real *output_potential, Real Grav_Constant, Real dens_avrg, Real current_a ){
  
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
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
  
  
  if ( !potential_initialized ){
    Initialize_Potential_Kernel<<<dim3dGrid,dim3dBlock>>>( 0, F.potential_d, F.density_d, nx_local, ny_local, nz_local, n_ghost );
    potential_initialized = true;
    chprintf( "SOR: Potential Initialized \n");
  }

  Copy_Input( input_density, Grav_Constant, dens_avrg, current_a );
  
  int ngrid_x_half = ( nx_local/2 + tpb_x - 1) / tpb_x;
  dim3 dim3dGrid_half(ngrid_x_half, ngrid_y, ngrid_z);
  
  Real epsilon = 1e-6;
  bool converged_h = 0;
  int max_iter = 10000000;
  int n_iter = 0;
  
  // For Diriclet Boudaries
  Real omega = 2. / ( 1 + M_PI / nx_total  );
  
  // For Periodic Boudaries
  // Real omega = 2. / ( 1 + 2*M_PI / nx_total  );
  // chprintf("Omega: %f \n", omega);
  
  // Iterate to solve Poisson equation
  while (converged_h == 0 ) {
    cudaMemset( F.converged_d, 1, sizeof(bool) );
    
    Iteration_Step_SOR<<<dim3dGrid_half,dim3dBlock>>>( n_cells_local, F.density_d, F.potential_d, nx_local, ny_local, nz_local, n_ghost, dx, dy, dz, omega, 0, epsilon, F.converged_d );
    
    Iteration_Step_SOR<<<dim3dGrid_half,dim3dBlock>>>( n_cells_local, F.density_d, F.potential_d, nx_local, ny_local, nz_local, n_ghost, dx, dy, dz, omega, 1, epsilon, F.converged_d );
    
    cudaMemcpy( &converged_h, F.converged_d, sizeof(bool), cudaMemcpyDeviceToHost );
    n_iter += 1;
    if ( n_iter == max_iter ) break;
  }
  
  if ( n_iter == max_iter ) chprintf(" SOR: No convergence in %d iterations \n", n_iter);
  else chprintf(" SOR: Converged in %d iterations \n", n_iter);
  
  
  
  Copy_Output( output_potential );
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  chprintf( " SOR: Potential Time = %f   msecs\n", milliseconds);
  


  return 0;

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
  
  int nx_pot, ny_pot, nz_pot;
  nx_pot = nx + 2*n_ghost;
  ny_pot = ny + 2*n_ghost;
  nz_pot = nz + 2*n_ghost;
  
  tid = tid_x + tid_y*nx + tid_z*nx*ny;
  
  tid_x += n_ghost;
  tid_y += n_ghost;
  tid_z += n_ghost;
  tid_pot = tid_x + tid_y*nx_pot + tid_z*nx_pot*ny_pot; 
  
  // //Set neighbors ids
  int indx_l, indx_r, indx_d, indx_u, indx_b, indx_t;
  indx_l = tid_x == n_ghost          ?    nx_pot-n_ghost-1 : tid_x-1;  //Left
  indx_r = tid_x == nx_pot-n_ghost-1 ?             n_ghost : tid_x+1;  //Right
  indx_d = tid_y == n_ghost          ?    ny_pot-n_ghost-1 : tid_y-1;  //Down
  indx_u = tid_y == ny_pot-n_ghost-1 ?             n_ghost : tid_y+1;  //Up
  indx_b = tid_z == n_ghost          ?    nz_pot-n_ghost-1 : tid_z-1;  //Bottom
  indx_t = tid_z == nz_pot-n_ghost-1 ?             n_ghost : tid_z+1;  //Top
  
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

  if ( ( fabs( ( phi_new - phi_c ) / phi_c ) > epsilon ) ) converged_d[0] = 0;
  // if ( ( fabs( ( phi_new - phi_c ) ) > epsilon ) ) converged_d[0] = 0;
  
  
  potential_d[tid_pot] = phi_new;
  
  
}


__global__ void Copy_Input_Kernel( int n_cells, Real *input_d, Real *density_d, Real Grav_Constant, Real dens_avrg, Real current_a ){
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_cells ) return;

  #ifdef COSMOLOGY
  density_d[tid] = 4 * M_PI * Grav_Constant * ( input_d[tid] - dens_avrg ) / current_a;
  #else
  density_d[tid] = 4 * M_PI * Grav_Constant * input_d[tid];
  #endif
}

__global__ void Initialize_Potential_Kernel( Real init_val, Real *potential_d, Real *density_d, int nx, int ny, int nz, int n_ghost ){

  int tid_x, tid_y, tid_z, tid, tid_pot;
  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  tid_z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (tid_x >= nx || tid_y >= ny || tid_z >= nz ) return;  
  
  tid = tid_x + tid_y*nx + tid_z*nx*ny;
  Real dens = density_d[tid];
  
  
  tid_x += n_ghost;
  tid_y += n_ghost;
  tid_z += n_ghost;

  int nx_pot, ny_pot;
  nx_pot = nx + 2*n_ghost;
  ny_pot = ny + 2*n_ghost;
  

  tid_pot = tid_x + tid_y*nx_pot + tid_z*nx_pot*ny_pot;
  potential_d[tid_pot] = init_val;
  // potential_d[tid_pot] = dens;
  
}

void Potential_SOR_3D::Copy_Input( Real *input_density, Real Grav_Constant, Real dens_avrg, Real current_a ){
  cudaMemcpy( F.input_d, input_density, n_cells_local*sizeof(Real), cudaMemcpyHostToDevice );
  
  // set values for GPU kernels
  int ngrid =  (n_cells_local + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  Copy_Input_Kernel<<<dim1dGrid,dim1dBlock>>>( n_cells_local, F.input_d, F.density_d,  Grav_Constant, dens_avrg, current_a  );
}


void Potential_SOR_3D::Copy_Output( Real *output_potential ){
  cudaMemcpy( output_potential, F.potential_d, n_cells_potential*sizeof(Real), cudaMemcpyDeviceToHost );
}






#endif //GRAVITY
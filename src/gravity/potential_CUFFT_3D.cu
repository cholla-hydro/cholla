#ifdef GRAVITY
#ifdef POTENTIAL_CUFFT

#include "potential_CUFFT_3D.h"


Potential_CUFFT_3D::Potential_CUFFT_3D( void ){}

void Potential_CUFFT_3D::Initialize( Grav3D Grav){

  Lbox_x = Grav.Lbox_x;
  Lbox_y = Grav.Lbox_y;
  Lbox_z = Grav.Lbox_z;

  nx_total = Grav.nx_total;
  ny_total = Grav.ny_total;
  nz_total = Grav.nz_total;

  nx_local = Grav.nx_local;
  ny_local = Grav.ny_local;
  nz_local = Grav.nz_local;

  dx = Grav.dx;
  dy = Grav.dy;
  dz = Grav.dz;

  n_cells_local = nx_local*ny_local*nz_local;
  n_cells_total = nx_total*ny_total*nz_total;
  chprintf( " Using Poisson Solver: CUFFT\n");
  chprintf( "  CUFFT: L[ %f %f %f ] N[ %d %d %d ] dx[ %f %f %f ]\n", Lbox_x, Lbox_y, Lbox_z, nx_local, ny_local, nz_local, dx, dy, dz );

  AllocateMemory_CPU();

  chprintf( "  CUFFT: Creating FFT plan...\n");
  cufftPlan3d( &plan_cufft_fwd,  nz_local, ny_local,  nx_local, CUFFT_Z2Z);
  cufftPlan3d( &plan_cufft_bwd,  nz_local, ny_local,  nx_local, CUFFT_Z2Z);

  chprintf( "  CUFFT: Computing K for Gravity Green Funtion\n");
  cudaMalloc( (void**)&F.G_d, n_cells_local*sizeof(Real));
  Get_K_for_Green_function();
  threads_per_block = 1024;
  blocks_per_grid = (( n_cells_local - 1 ) / threads_per_block) + 1;
  chprintf( "  CUFFT: Using %d threads and %d blocks for applying G funtion: %d \n", threads_per_block, blocks_per_grid, threads_per_block*blocks_per_grid);

}

void Potential_CUFFT_3D::AllocateMemory_CPU( void ){
  F.output_h = (Complex_cufft *) malloc(n_cells_local*sizeof(Complex_cufft));
  F.G_h = (Real *) malloc(n_cells_local*sizeof(Real));
}

void Potential_CUFFT_3D::AllocateMemory_GPU( void ){
  cudaMalloc( (void**)&F.input_real_d, n_cells_local*sizeof(Real_cufft));
  cudaMalloc( (void**)&F.input_d, n_cells_local*sizeof(Complex_cufft));
  cudaMalloc( (void**)&F.transform_d, n_cells_local*sizeof(Complex_cufft));
  cudaMalloc( (void**)&F.output_d, n_cells_local*sizeof(Complex_cufft));
}

void Potential_CUFFT_3D::FreeMemory_GPU( void ){
  cudaFree( F.input_real_d );
  cudaFree( F.input_d );
  cudaFree( F.output_d );
  cudaFree( F.transform_d );
}

void Potential_CUFFT_3D::Reset( void ){
  free( F.output_h );
  free( F.G_h );
  cudaFree( F.G_d );
}


void Potential_CUFFT_3D::Get_K_for_Green_function( void){
  Real kx, ky, kz, Gx, Gy, Gz, G;
  int id;
  for (int k=0; k<nz_local; k++){
    kz =  2*M_PI*k/nz_local;
    Gz = sin( kz/2 );
    for (int j=0; j<ny_local; j++){
      ky =  2*M_PI*j/ny_local;
      Gy = sin( ky/2 );
      for ( int i=0; i<nx_local; i++){
        id = i + j*nx_local + k*nx_local*ny_local;
        kx =  2*M_PI*i/nx_local;
        Gx = sin( kx/2 );
        G = -1 / ( Gx*Gx + Gy*Gy + Gz*Gz ) * dx * dx / 4 ;
        if ( id == 0 ) G = 1;
        F.G_h[id] = G;
        // F.G_h[id] = 0.1;
      }
    }
  }
  cudaMemcpy( F.G_d, F.G_h, n_cells_local*sizeof(Real), cudaMemcpyHostToDevice );
}

__global__
void Copy_Input_Kernel( int n_cells, Real *input_h, Complex_cufft *input_d ){
  int t_id = threadIdx.x + blockIdx.x*blockDim.x;
  if ( t_id < n_cells ){
    input_d[t_id].x = input_h[t_id];
    input_d[t_id].y = 0.0;
  }
}

void Potential_CUFFT_3D::Copy_Output( Grav3D &Grav ){

  cudaMemcpy( F.output_h, F.output_d, n_cells_local*sizeof(Complex_cufft), cudaMemcpyDeviceToHost );


  int id, id_pot;
  int i, k, j;
  for (k=0; k<nz_local; k++) {
    for (j=0; j<ny_local; j++) {
      for (i=0; i<nx_local; i++) {
        id = i + j*nx_local + k*nx_local*ny_local;
        id_pot = (i+N_GHOST_POTENTIAL) + (j+N_GHOST_POTENTIAL)*(nx_local+2*N_GHOST_POTENTIAL) + (k+N_GHOST_POTENTIAL)*(nx_local+2*N_GHOST_POTENTIAL)*(ny_local+2*N_GHOST_POTENTIAL);
        Grav.F.potential_h[id_pot] = F.output_h[id].x / n_cells_local;
        // Grav.F.density_h[id] = F.G_h[id];
      }
    }
  }
}

void Potential_CUFFT_3D::Copy_Input( Grav3D &Grav ){
  cudaMemcpy( F.input_real_d, Grav.F.density_h, n_cells_local*sizeof(Real_cufft), cudaMemcpyHostToDevice );
  Copy_Input_Kernel<<<blocks_per_grid, threads_per_block>>>( n_cells_local, F.input_real_d, F.input_d );

}


__global__
void Apply_G_Funtion( int n_cells, Complex_cufft *transform, Real *G ){
  int t_id = threadIdx.x + blockIdx.x*blockDim.x;
  Real G_val;
  if ( t_id < n_cells ){
    G_val = G[t_id];
    if ( t_id == 0 ) G_val = 1.0;
    transform[t_id].x *= G_val;
    transform[t_id].y *= G_val;
    if ( t_id == 0 ){
      transform[t_id].x = 0;
      transform[t_id].y = 0;
    }
  }
}


Real Potential_CUFFT_3D::Get_Potential( Grav3D &Grav ){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  AllocateMemory_GPU();
  Copy_Input( Grav );

  cufftExecZ2Z( plan_cufft_fwd, F.input_d, F.transform_d, CUFFT_FORWARD );
  Apply_G_Funtion<<<blocks_per_grid, threads_per_block>>>( n_cells_local, F.transform_d, F.G_d );
  cufftExecZ2Z( plan_cufft_bwd, F.transform_d, F.output_d, CUFFT_INVERSE );
  Copy_Output( Grav );

  FreeMemory_GPU();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // chprintf( " CUFFT: Potential Time = %f   msecs\n", milliseconds);
  return (Real) milliseconds;

}

#endif //POTENTIAL_CUFFT
#endif //GRAVITY

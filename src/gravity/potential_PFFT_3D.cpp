#ifdef GRAVITY
#ifdef PFFT

#include "potential_PFFT_3D.h"
#include<iostream>
#include "../io.h"

Potential_PFFT_3D::Potential_PFFT_3D( void ){}

void Potential_PFFT_3D::Initialize( Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx_real, Real dy_real, Real dz_real){

  //Set Box Size
  Lbox_x = Lx;
  Lbox_y = Ly;
  Lbox_z = Lz;

  //Set Box Left boundary positions
  xMin = x_min;
  yMin = y_min;
  zMin = z_min;
  
  //Set uniform ( dx, dy, dz )
  dx = dx_real;
  dy = dy_real;
  dz = dz_real;

  //Set Box Total number of cells
  nx_total = nx;
  ny_total = ny;
  nz_total = nz;

  //Set local domain number of cells
  nx_local = nx_real;
  ny_local = ny_real;
  nz_local = nz_real;

  //Index where k=0, this will be computed afterwards 
  index_0 = -1;

  //Total and local number of cells
  n_cells_local = nx_local*ny_local*nz_local;
  n_cells_total = nx_total*ny_total*nz_total;
  
  chprintf( " Using Poisson Solver: PFFT\n");

  //Set the number of MPI processes
  nproc_pfft = nproc;

  chprintf(" Initializing PFFT:  %d processes\n", nproc_pfft );
  pfft_init();

  chprintf( "  PFFT: L[ %f %f %f ] N_local[ %d %d %d ] dx[ %f %f %f ]\n", Lbox_x, Lbox_y, Lbox_z, nx_local, ny_local, nz_local, dx, dy, dz );
  
  //Set the MPI processes grid
  nprocs_grid_pfft[0] = nproc_z;
  nprocs_grid_pfft[1] = nproc_y;
  nprocs_grid_pfft[2] = nproc_x;

  //Set the Box global cells size for each direction
  n_pfft[0] = nz_total;
  n_pfft[1] = ny_total;
  n_pfft[2] = nx_total;

  // /* Create 3-dimensional process grid of size np_pfft[0] x np_pfft[1] x np_pfft[2] */
  pfft_create_procmesh(3, MPI_COMM_WORLD, nprocs_grid_pfft, &comm_pfft);
  MPI_Comm_rank( comm_pfft, &procID_pfft );
  MPI_Cart_coords( comm_pfft, procID_pfft, 3, pcoords_pfft);
  
  /* Get parameters of data distribution  for a Real-To-Complex transform*/
  //local_n_in_pfft: local size for the input array
  //local_in_start_pfft: offset for computing the global index for the input array
  //local_n_transform_fwd_pfft: local size for the forward transform array
  //local_transform_fwd_start_pfft: offset for computing the global index for the forward transform array
  alloc_local_fwd = pfft_local_size_dft_r2c_3d(
    n_pfft, comm_pfft, PFFT_TRANSPOSED_OUT ,
    local_n_in_pfft, local_in_start_pfft, local_n_transform_fwd_pfft, local_transform_fwd_start_pfft);
    
  /* Get parameters of data distribution  for a Complex-To-Real transform*/
  //local_n_transform_bwd_pfft: local size for the backward transform array
  //local_transform_bwd_start_pfft: offset for computing the global index for the backward transform array
  //local_n_out_pfft: local size for the output array
  //local_out_start_pfft: offset for computing the global index for the output array
  alloc_local_bwd = pfft_local_size_dft_c2r_3d(
      n_pfft, comm_pfft, PFFT_TRANSPOSED_IN ,
      local_n_transform_bwd_pfft, local_transform_bwd_start_pfft, local_n_out_pfft, local_out_start_pfft);
      
  //Check the dimensions of the input and the transform, make sure that dimensions and offsets  are the same
  bool domain_error = false;
  if ( alloc_local_fwd != alloc_local_bwd ) domain_error = true;
  for (int i=0; i<3; i++ ){
   if ( local_n_in_pfft[i] != local_n_out_pfft[i]) domain_error = true;
   if ( local_in_start_pfft[i] != local_out_start_pfft[i]) domain_error = true;
   if ( local_n_transform_fwd_pfft[i] != local_n_transform_bwd_pfft[i]) domain_error = true;
   if ( local_transform_fwd_start_pfft[i] != local_transform_bwd_start_pfft[i]) domain_error = true;
  }
  //Exit if the dimensions and offsets are not the same
  if ( domain_error ){
   printf("PFFT: FFT doamin error \n" );
   exit(-1);
  }

  chprintf("  PFFT process: nx:%d ny:%d nz:%d \n", nprocs_grid_pfft[2], nprocs_grid_pfft[1], nprocs_grid_pfft[0]);
  chprintf("  PFFT cells:   nx:%d ny:%d nz:%d \n", n_pfft[2], n_pfft[1], n_pfft[0]);
  chprintf("  PFFT local:   nx:%d ny:%d nz:%d \n", local_n_in_pfft[2], local_n_in_pfft[1], local_n_in_pfft[0] );
  
  //Check that size of the local PFFT domain is the same as the local Cholla doamin
  if ( local_n_in_pfft[2] != nx_local ||  local_n_in_pfft[1] != ny_local ||  local_n_in_pfft[0] != nz_local ){
    std::cout << " ERROR: PFFT domain is not the same as Cholla domain\n" << std::endl;
    std::cout << " PFFT   Domain: [ " << local_n_in_pfft[2] << " , " << local_n_in_pfft[1] << " , " << local_n_in_pfft[0] << " ]" << std::endl;
    std::cout << " Cholla Domain: [ " << nx_local << " , " << ny_local << " , " << nz_local << " ]" << std::endl;  
    exit(-1);
  }
  
  
  
  AllocateMemory_CPU();
  
  chprintf( "  PFFT: Creating FFT plan...\n");
  //  Plan parallel  FFTs (forward) 
  plan_fwd = pfft_plan_dft_r2c_3d(n_pfft, F.input, F.transform, comm_pfft,
      PFFT_FORWARD, PFFT_TRANSPOSED_OUT | PFFT_MEASURE);

  // Plan parallel  FFTs (backward) 
  plan_bwd = pfft_plan_dft_c2r_3d(n_pfft, F.transform, F.output, comm_pfft,
      PFFT_BACKWARD, PFFT_TRANSPOSED_IN | PFFT_MEASURE);
      
  //Compute the Greens Function factor to obtain the potential in Fourier space    
  chprintf( "  PFFT: Computing K for Gravity Green Funtion\n");
  Get_K_for_Green_function();
  
  #ifdef GRAVITY_LONG_INTS
  chprintf( "  PFFT: Using Long ints for potential calculation.\n");
  #endif

  MPI_Barrier( world );
  chprintf( " PFFT Initialized Successfully. \n");
}


void Potential_PFFT_3D::AllocateMemory_CPU( void ){
  /* Allocate memory */
  F.input  = pfft_alloc_real(2*alloc_local_fwd);
  F.transform = pfft_alloc_complex(alloc_local_fwd);
  F.output = pfft_alloc_real(2*alloc_local_bwd);
  F.G = (Real *) malloc(n_cells_local*sizeof(Real));
}

void Potential_PFFT_3D::Reset( void ){
  free( F.input );
  free( F.output );
  free( F.transform );
  free( F.G );
}

//Copy the density to the input_array of the poisson solver
void Potential_PFFT_3D::Copy_Input( Real *input_density, Real Grav_Constant, Real dens_avrg, Real current_a ){
  int i, k, j, id;
  
  // #ifdef PARALLEL_OMP
  // #pragma omp parallel for num_threads(N_OMP_THREADS)
  // #endif
  for (k=0; k<nz_local; k++) {
    for (j=0; j<ny_local; j++) {
      for (i=0; i<nx_local; i++) {
        id = i + j*nx_local + k*nx_local*ny_local;
        #ifdef COSMOLOGY
        //input = 4*pi*G*(rho - rho_avrg)/a
        F.input[id] = 4 * M_PI * Grav_Constant * (input_density[id] - dens_avrg ) / current_a ;
        #else
        //input = 4*pi*G*rho
        F.input[id] = 4 * M_PI * Grav_Constant * input_density[id] ;
        #endif //COSMOLOGY
      }
    }
  }
}

//Copy the resulting potential to the output_potential array.
void Potential_PFFT_3D::Copy_Output( Real *output_potential ){
  int id, id_pot;
  int i, k, j;
  
  // #ifdef PARALLEL_OMP
  // #pragma omp parallel for num_threads(N_OMP_THREADS)
  // #endif
  for (k=0; k<nz_local; k++) {
    for (j=0; j<ny_local; j++) {
      for (i=0; i<nx_local; i++) {
        id = i + j*nx_local + k*nx_local*ny_local;
        id_pot = (i+N_GHOST_POTENTIAL) + (j+N_GHOST_POTENTIAL)*(nx_local+2*N_GHOST_POTENTIAL) + (k+N_GHOST_POTENTIAL)*(nx_local+2*N_GHOST_POTENTIAL)*(ny_local+2*N_GHOST_POTENTIAL);
        output_potential[id_pot] = F.output[id] / n_cells_total; //The result from the inverse transform has to be normalized by the total number of cells
      }
    }
  }
}

//Compute the Greens Function to obtain the potential in Fourier space
void Potential_PFFT_3D::Get_K_for_Green_function( void){
  grav_int_t m = 0;
  grav_int_t k_0, k_1, k_2;
  Real k_x, k_y, k_z, G_x, G_y, G_z, G;

  double ksqrd;
  //Get the global index in the box to compute the Greens Function factor 
  for(ptrdiff_t k1=local_transform_fwd_start_pfft[1]; k1<local_transform_fwd_start_pfft[1]+local_n_transform_fwd_pfft[1]; k1++){
    for(ptrdiff_t k2=local_transform_fwd_start_pfft[2]; k2<local_transform_fwd_start_pfft[2]+local_n_transform_fwd_pfft[2]; k2++){
      for(ptrdiff_t k0=local_transform_fwd_start_pfft[0]; k0<local_transform_fwd_start_pfft[0]+local_n_transform_fwd_pfft[0]; k0++){
        //  printf("out[%td, %td, %td] = %.2f + I * %.2f\n", k0, k1, k2, transf[m][0],transf[m][1]);
        k_0 = k0;
        k_1 = k1;
        k_2 = k2;
        if ( k_2 >= nx_total/2) k_2 -= nx_total;
        if ( k_0 >= nz_total/2) k_0 -= nz_total;
        if ( k_1 >= ny_total/2) k_1 -= ny_total;
        k_x = 2*M_PI*k_2/nx_total;
        k_z = 2*M_PI*k_0/nz_total;
        k_y = 2*M_PI*k_1/nx_total;
        G_x = sin( k_x/2  );
        G_y = sin( k_y/2 );
        G_z = sin( k_z/2 );
        if ( k_0==0 && k_1==0 && k_2 == 0 ){
          index_0 = m ;
          G = 1;
          std::cout << "  K=0   index: " << index_0 << std::endl;
        }
        else{
          G = -1 / ( G_x*G_x + G_y*G_y + G_z*G_z ) * dx * dx /4 ;
        }
        F.G[m] = G;
        m += 1;
      }
    }
  }
}

//Multiply the FFT transform by the Greens function to obtain the potential in Fourier space
void Potential_PFFT_3D::Apply_G_Funtion( void ){
  Real G_val;
  int i;
  
  // #ifdef PARALLEL_OMP
  // #pragma omp parallel for num_threads(N_OMP_THREADS)
  // #endif
  for ( i=0; i<n_cells_local; i++ ){
    G_val = F.G[i];
    F.transform[i][0] *= G_val;
    F.transform[i][1] *= G_val;
  }
  if (index_0 > -1){
    F.transform[index_0][0] = 0;
    F.transform[index_0][1] = 0;
  }
}

//Get the global cell index
void Potential_PFFT_3D::Get_Index_Global(int i, int j, int k, int *i_global, int *j_global, int *k_global){

  int i_g, j_g, k_g;
  i_g = xMin/dx + i;
  j_g = yMin/dy + j;
  k_g = zMin/dz + k;

  *i_global = i_g;
  *j_global = j_g;
  *k_global = k_g;
}

//NOT USED: Multiply the FFT transform by -1/K^2  to obtain the potential in Fourier space
void Potential_PFFT_3D::Apply_K2_Funtion( void ){
  Real kx, ky, kz, k2;
  int i_g, j_g, k_g;
  int id;
  int k, j, i;
  
  // #ifdef PARALLEL_OMP
  // #pragma omp parallel for num_threads(N_OMP_THREADS)
  // #endif
  for (k=0; k<nz_local; k++){
    for (j=0; j<ny_local; j++){
      for (i=0; i<nx_local; i++){
        id = i + j*nx_local + k*nx_local*ny_local;
        Get_Index_Global( i, j, k, &i_g, &j_g, &k_g );
        if ( i_g >= nx_total/2) i_g -= nz_total;
        if ( j_g >= ny_total/2) j_g -= ny_total;
        if ( k_g >= nz_total/2) k_g -= nz_total;
        kx =  2 * M_PI * i_g / Lbox_x;
        ky =  2 * M_PI * j_g / Lbox_y;
        kz =  2 * M_PI * k_g / Lbox_z;
        if ( i_g==0 && j_g==0 && k_g== 0){
          k2=1;
        }
        else{
          k2  =  kx*kx + ky*ky + kz*kz ;
        }

        F.transform[id][0] *= -1/k2;
        F.transform[id][1] *= -1/k2;
      }
    }
  }
  if (index_0 > -1){
    F.transform[index_0][0] = 0;
    F.transform[index_0][1] = 0;
  }
}

//Compute the Potential from the input_density and save it in output_potential 
Real Potential_PFFT_3D::Get_Potential( Real *input_density,  Real *output_potential, Real Grav_Constant, Real dens_avrg, Real current_a ){

  //Copy the input density
  //input = 4*pi*G*rho
  //if cosmology: input = 4*pi*G*(rho - rho_avrg)/a
  Copy_Input( input_density, Grav_Constant, dens_avrg, current_a );

  //Compute the FFT forward transform
  pfft_execute( plan_fwd );
  
  //Multiply the FFT transform by the Greens function to obtain the potential in Fourier space
  Apply_G_Funtion();
  // Apply_K2_Funtion();
  
  //Compute the FFT backward (inverse) transform 
  pfft_execute( plan_bwd );
  
  //Copy the result to output_potential
  Copy_Output( output_potential );

  return 0;
}










#endif //PFFT
#endif //GRAVITY
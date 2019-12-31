#ifdef PARTICLES

#include <unistd.h>
#include "../io.h"
#include "../grid3D.h"
#include"../random_functions.h"
#include "particles_3D.h"

#ifdef MPI_CHOLLA
#include"../mpi_routines.h"
#endif

#ifdef PARALLEL_OMP
#include "../parallel_omp.h"
#endif

Particles_3D::Particles_3D( void ){}

void Grid3D::Initialize_Particles( struct parameters *P ){
  
  chprintf( "\nInitializing Particles...\n");
  
  Particles.Initialize( P, Grav, H.xbound, H.ybound, H.zbound, H.xdglobal, H.ydglobal, H.zdglobal );
  
  if (strcmp(P->init, "Uniform")==0)  Initialize_Uniform_Particles();  
  
  #ifdef MPI_CHOLLA
  MPI_Barrier( world );
  #endif
  chprintf( "Particles Initialized Successfully. \n\n");
  
  
} 

void Particles_3D::Initialize( struct parameters *P, Grav3D &Grav,  Real xbound, Real ybound, Real zbound, Real xdglobal, Real ydglobal, Real zdglobal){

  //Initialize local and total number of particles to 0
  n_local = 0;
  n_total = 0;
  n_total_initial = 0;
  
  //Initialize the simulation time and delta_t to 0
  dt = 0.0;
  t = 0.0;
  //Set the maximum delta_t for particles, this can be changed depending on the problem.
  max_dt = 10000; 

  //Courant CFL condition factor for particles
  C_cfl = 0.3;

  #ifdef PARTICLES_CPU
  //Vectors for positions, velocities and accelerations
  real_vector_t pos_x;
  real_vector_t pos_y;
  real_vector_t pos_z;
  real_vector_t vel_x;
  real_vector_t vel_y;
  real_vector_t vel_z;
  real_vector_t grav_x;
  real_vector_t grav_y;
  real_vector_t grav_z;

  #ifndef SINGLE_PARTICLE_MASS
  //Vector for masses
  real_vector_t mass;
  #endif
  #ifdef PARTICLE_IDS
  //Vector for particle IDs
  int_vector_t partIDs;
  #endif

  #ifdef MPI_CHOLLA
  //Vectors for the indices of the particles that need to be transfered via MPI
  int_vector_t out_indxs_vec_x0;
  int_vector_t out_indxs_vec_x1;
  int_vector_t out_indxs_vec_y0;
  int_vector_t out_indxs_vec_y1;
  int_vector_t out_indxs_vec_z0;
  int_vector_t out_indxs_vec_z1;
  #endif
  
  #endif //PARTICLES_CPU

  //Initialize Grid Values
  //Local and total number of cells
  G.nx_local = Grav.nx_local;
  G.ny_local = Grav.ny_local;
  G.nz_local = Grav.nz_local;
  G.nx_total = Grav.nx_total;
  G.ny_total = Grav.ny_total;
  G.nz_total = Grav.nz_total;

  //Uniform (dx, dy, dz)
  G.dx = Grav.dx;
  G.dy = Grav.dy;
  G.dz = Grav.dz;

  //Left boundaries of the local domain
  G.xMin = Grav.xMin;
  G.yMin = Grav.yMin;
  G.zMin = Grav.zMin;

  //Right boundaries of the local domain
  G.xMax = G.xMin + G.nx_local*G.dx;
  G.yMax = G.yMin + G.ny_local*G.dy;
  G.zMax = G.zMin + G.nz_local*G.dz;

  //Left boundaries of the global domain
  G.domainMin_x = xbound;
  G.domainMin_y = ybound;
  G.domainMin_z = zbound;
  
  //Right boundaries of the global domain
  G.domainMax_x = xbound + xdglobal;
  G.domainMax_y = ybound + ydglobal;
  G.domainMax_z = zbound + zdglobal;

  //Number of ghost cells for the particles grid. For CIC one ghost cell is needed
  G.n_ghost_particles_grid = 1;
  
  //Number of cells for the particles grid including ghost cells
  G.n_cells = (G.nx_local+2*G.n_ghost_particles_grid) * (G.ny_local+2*G.n_ghost_particles_grid) * (G.nz_local+2*G.n_ghost_particles_grid);

  #ifdef PARTICLES_GPU
  //Factor to allocate the particles data arrays on the GPU.
  //When using MPI particles will be transfered to other GPU, for that reason we need extra memory allocated
  #ifdef MPI_CHOLLA
  G.allocation_factor = 1.5;
  #else
  G.allocation_factor = 1.0;
  #endif

  G.size_blocks_array = 1024*128;
  G.n_cells_potential = ( G.nx_local + 2*N_GHOST_POTENTIAL ) * ( G.ny_local + 2*N_GHOST_POTENTIAL ) * ( G.nz_local + 2*N_GHOST_POTENTIAL );
  #endif

  // Flags for Initial and tranfer the particles and density
  INITIAL = true;
  TRANSFER_DENSITY_BOUNDARIES = false;
  TRANSFER_PARTICLES_BOUNDARIES = false;
  
  Allocate_Memory();

  //Initialize the particles density and gravitational field to 0.
  Initialize_Grid_Values();
  
  // Initialize Particles
  if (strcmp(P->init, "Spherical_Overdensity_3D")==0) Initialize_Sphere();
  
  if (strcmp(P->init, "Zeldovich_Pancake")==0) Initialize_Zeldovich_Pancake( P );
  
  if (strcmp(P->init, "Read_Grid")==0)  Load_Particles_Data(  P );
  
  #ifdef MPI_CHOLLA
  n_total_initial = ReducePartIntSum(n_local);
  #else
  n_total_initial = n_local;
  #endif
  
  chprintf("Particles Initialized: \n n_local: %lu \n", n_local );
  chprintf(" n_total: %lu \n", n_total_initial );
  chprintf(" xDomain_local:  [%.4f %.4f ] [%.4f %.4f ] [%.4f %.4f ]\n", G.xMin, G.xMax, G.yMin, G.yMax, G.zMin, G.zMax );
  chprintf(" xDomain_global: [%.4f %.4f ] [%.4f %.4f ] [%.4f %.4f ]\n", G.domainMin_x, G.domainMax_x, G.domainMin_y, G.domainMax_y, G.domainMin_z, G.domainMax_z);
  chprintf(" dx: %f  %f  %f\n", G.dx, G.dy, G.dz );
  
  #ifdef PARTICLE_IDS
  chprintf(" Tracking particle IDs\n");
  #endif
  
  #ifdef PRINT_DOMAIN
  for (int n=0; n<nproc; n++){
    if (procID == n ) std::cout << procID << " x["<< G.xMin << "," << G.xMax << "] "  << " y["<< G.yMin << "," << G.yMax << "] "  << " z["<< G.zMin << "," << G.zMax << "] " << std::endl;
    usleep( 100 );
  }
  #endif
  
  #if defined(PARALLEL_OMP) && defined(PARTICLES_CPU)
  chprintf(" Using OMP for particles calculations\n");
  int n_omp_max = omp_get_max_threads();
  chprintf("  MAX OMP Threads: %d\n", n_omp_max);
  chprintf("  N OMP Threads per MPI process: %d\n", N_OMP_THREADS);
  #ifdef PRINT_OMP_DOMAIN
  // Print omp domain for each omp thread
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t omp_pIndx_start, omp_pIndx_end;
    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    #pragma omp barrier
    Get_OMP_Particles_Indxs( n_local, n_omp_procs, omp_id, &omp_pIndx_start, &omp_pIndx_end );
  
    for (int omp_indx = 0; omp_indx<n_omp_procs; omp_indx++){
      if (omp_id == omp_indx) chprintf( "  omp_id:%d  p_start:%ld  p_end:%ld  \n", omp_id, omp_pIndx_start, omp_pIndx_end );
    }
  }
  #endif//PRINT_OMP_DOMAIN
  #endif//PARALLEL_OMP
  
  
  #ifdef MPI_CHOLLA
  chprintf( " N_Particles Boundaries Buffer Size: %d\n", N_PARTICLES_TRANSFER);
  chprintf( " N_Data per Particle Transfer: %d\n", N_DATA_PER_PARTICLE_TRANSFER);
  
  #ifdef PARTICLES_GPU
  Allocate_Memory_GPU_MPI();
  #endif//PARTICLES_GPU
  #endif//MPI_CHOLLA
  


  
}

void Particles_3D::Allocate_Memory( void ){
  
  //Allocate arrays for density and gravitational field
  
  G.density   = (Real *) malloc(G.n_cells*sizeof(Real));
  #ifdef PARTICLES_CPU
  G.gravity_x = (Real *) malloc(G.n_cells*sizeof(Real));
  G.gravity_y = (Real *) malloc(G.n_cells*sizeof(Real));
  G.gravity_z = (Real *) malloc(G.n_cells*sizeof(Real));
  #endif
  
  #ifdef PARTICLES_GPU
  Allocate_Memory_GPU();
  G.dti_array_host = (Real *) malloc(G.size_blocks_array*sizeof(Real));
  #endif
}


#ifdef PARTICLES_GPU
void Particles_3D::Allocate_Memory_GPU(){
  
  //Allocate arrays for density and gravitational field on the GPU
  
  Allocate_Particles_Grid_Field_Real( &G.density_dev, G.n_cells);
  Allocate_Particles_Grid_Field_Real( &G.gravity_x_dev, G.n_cells);
  Allocate_Particles_Grid_Field_Real( &G.gravity_y_dev, G.n_cells);
  Allocate_Particles_Grid_Field_Real( &G.gravity_z_dev, G.n_cells);
  Allocate_Particles_Grid_Field_Real( &G.potential_dev, G.n_cells_potential);
  Allocate_Particles_Grid_Field_Real( &G.dti_array_dev, G.size_blocks_array);  
  chprintf( " Allocated GPU memory.\n");  
}



#ifdef MPI_CHOLLA
void Particles_3D::Allocate_Memory_GPU_MPI(){
  
  //Allocate memory for the the particles MPI transfers
  
  part_int_t size, size_blocks;
  
  size = (part_int_t) n_local;
  size_blocks = G.size_blocks_array;
  
  Allocate_Particles_GPU_Array_bool( &G.transfer_particles_flags_d, size );
  Allocate_Particles_GPU_Array_int( &G.transfer_particles_indxs_d, size);
  Allocate_Particles_GPU_Array_int( &G.transfer_particles_partial_sum_d, size);
  Allocate_Particles_GPU_Array_int( &G.transfer_particles_sum_d, size_blocks);
  Allocate_Particles_GPU_Array_int( &G.n_transfer_d, 1);
  
  G.n_transfer_h = (int *) malloc(sizeof(int));
  chprintf( " Allocated GPU memory for MPI trandfers.\n"); 
}
#endif //MPI_CHOLLA

void Particles_3D::Free_Memory_GPU(){
  
  Free_GPU_Array_Real(G.density_dev);
  Free_GPU_Array_Real(G.potential_dev);
  Free_GPU_Array_Real(G.gravity_x_dev);
  Free_GPU_Array_Real(G.gravity_y_dev);
  Free_GPU_Array_Real(G.gravity_z_dev);
  Free_GPU_Array_Real(G.dti_array_dev);
  
  Free_GPU_Array_Real(pos_x_dev);
  Free_GPU_Array_Real(pos_y_dev);
  Free_GPU_Array_Real(pos_z_dev);
  Free_GPU_Array_Real(vel_x_dev);
  Free_GPU_Array_Real(vel_y_dev);
  Free_GPU_Array_Real(vel_z_dev);
  Free_GPU_Array_Real(grav_x_dev);
  Free_GPU_Array_Real(grav_y_dev);
  Free_GPU_Array_Real(grav_z_dev);
  
  #ifdef MPI_CHOLLA
  Free_GPU_Array_bool(G.transfer_particles_flags_d);
  Free_GPU_Array_int(G.transfer_particles_sum_d);
  Free_GPU_Array_int(G.transfer_particles_partial_sum_d);
  Free_GPU_Array_int(G.transfer_particles_indxs_d);
  Free_GPU_Array_int(G.n_transfer_d);
  free(G.n_transfer_h);
  #endif
}


#endif //PARTICLES_GPU


void Particles_3D::Initialize_Grid_Values( void ){
  
  //Initialize density and gravitaional field to 0.
  
  int id;
  for( id=0; id<G.n_cells; id++ ){
    G.density[id] = 0;
    #ifdef PARTICLES_CPU
    G.gravity_x[id] = 0;
    G.gravity_y[id] = 0;
    G.gravity_z[id] = 0;
    #endif
  }
}

void Particles_3D::Initialize_Sphere( void ){
  
  //Initialize Random positions for sphere of quasi-uniform density
  chprintf( " Initializing Particles Uniform Sphere\n");
  
  int i, j, k, id;
  Real center_x, center_y, center_z, radius, sphereR;
  center_x = 0.5;
  center_y = 0.5;
  center_z = 0.5;;
  sphereR = 0.2;

  //Set the number of particles equal to the number of grid cells
  part_int_t n_particles_local = G.nx_local*G.ny_local*G.nz_local;
  part_int_t n_particles_total = G.nx_total*G.ny_total*G.nz_total;

  //Set the initial density for the particles
  Real rho_start = 1;
  Real M_sphere = 4./3 * M_PI* rho_start * sphereR*sphereR*sphereR;
  Real Mparticle = M_sphere / n_particles_total;

  #ifdef SINGLE_PARTICLE_MASS
  particle_mass = Mparticle;
  #endif
  
  #ifdef PARTICLES_GPU
  // Alocate memory in GPU for particle data
  // particles_buffer_size = n_particles_local * allocation_factor;
  particles_buffer_size = n_particles_local;
  Allocate_Particles_GPU_Array_Real( &pos_x_dev, particles_buffer_size);
  Allocate_Particles_GPU_Array_Real( &pos_y_dev, particles_buffer_size);
  Allocate_Particles_GPU_Array_Real( &pos_z_dev, particles_buffer_size);
  Allocate_Particles_GPU_Array_Real( &vel_x_dev, particles_buffer_size);
  Allocate_Particles_GPU_Array_Real( &vel_y_dev, particles_buffer_size);
  Allocate_Particles_GPU_Array_Real( &vel_z_dev, particles_buffer_size);
  Allocate_Particles_GPU_Array_Real( &grav_x_dev, particles_buffer_size);
  Allocate_Particles_GPU_Array_Real( &grav_y_dev, particles_buffer_size);
  Allocate_Particles_GPU_Array_Real( &grav_z_dev, particles_buffer_size);
  // #ifndef SINGLE_PARTICLE_MASS
  
  n_local = n_particles_local;
  
  //Allocate temporal Host arrays for the particles data
  Real *temp_pos_x  = (Real *) malloc(particles_buffer_size*sizeof(Real));
  Real *temp_pos_y  = (Real *) malloc(particles_buffer_size*sizeof(Real));
  Real *temp_pos_z  = (Real *) malloc(particles_buffer_size*sizeof(Real));
  Real *temp_vel_x  = (Real *) malloc(particles_buffer_size*sizeof(Real));
  Real *temp_vel_y  = (Real *) malloc(particles_buffer_size*sizeof(Real));
  Real *temp_vel_z  = (Real *) malloc(particles_buffer_size*sizeof(Real));
  
  chprintf( " Allocated GPU memory for particle data\n");
  #endif //PARTICLES_GPU


  chprintf( " Initializing Random Positions\n");

  part_int_t pID = 0;
  Real pPos_x, pPos_y, pPos_z, r;
  while ( pID < n_particles_local ){
    pPos_x = Rand_Real( G.xMin, G.xMax );
    pPos_y = Rand_Real( G.yMin, G.yMax );
    pPos_z = Rand_Real( G.zMin, G.zMax );

    r = sqrt( (pPos_x-center_x)*(pPos_x-center_x) + (pPos_y-center_y)*(pPos_y-center_y) + (pPos_z-center_z)*(pPos_z-center_z) );
    if ( r > sphereR ) continue;
    
    #ifdef PARTICLES_CPU
    //Copy the particle data to the particles vectors
    pos_x.push_back( pPos_x );
    pos_y.push_back( pPos_y );
    pos_z.push_back( pPos_z);
    vel_x.push_back( 0.0 );
    vel_y.push_back( 0.0 );
    vel_z.push_back( 0.0 );
    grav_x.push_back( 0.0 );
    grav_y.push_back( 0.0 );
    grav_z.push_back( 0.0 );
    #ifdef PARTICLE_IDS
    partIDs.push_back( pID );
    #endif
    #ifndef SINGLE_PARTICLE_MASS
    mass.push_back( Mparticle );
    #endif
    #endif //PARTICLES_CPU
    
    #ifdef PARTICLES_GPU
    // Copy the particle data to the temporal Host Buffers
    temp_pos_x[pID]  = pPos_x;
    temp_pos_y[pID]  = pPos_y;
    temp_pos_z[pID]  = pPos_z;
    temp_vel_x[pID]  = 0.0;
    temp_vel_y[pID]  = 0.0;
    temp_vel_z[pID]  = 0.0;
    #endif //PARTICLES_GPU
    
    pID += 1;
  }
  
  #ifdef PARTICLES_CPU
  n_local = pos_x.size();
  #endif //PARTICLES_CPU
  
  #ifdef PARTICLES_GPU
  //Copyt the particle data from tepmpotal Host buffer to GPU memory
  Copy_Particles_Array_Real_Host_to_Device( temp_pos_x, pos_x_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device( temp_pos_y, pos_y_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device( temp_pos_z, pos_z_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device( temp_vel_x, vel_x_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device( temp_vel_y, vel_y_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device( temp_vel_z, vel_z_dev, n_local);  
  
  //Free the temporal host buffers
  free( temp_pos_x );
  free( temp_pos_y );
  free( temp_pos_z );
  free( temp_vel_x );
  free( temp_vel_y );
  free( temp_vel_z );
  #endif //PARTICLES_GPU
  
  chprintf( " Particles Uniform Sphere Initialized, n_local: %lu\n", n_local);
  
}


void Particles_3D::Initialize_Zeldovich_Pancake( struct parameters *P ){
  
  //No partidcles for the Zeldovich Pancake problem. n_local=0
  
  chprintf("Setting Zeldovich Pancake initial conditions...\n");
  
  // n_local = pos_x.size();
  n_local = 0;

  chprintf( " Particles Zeldovich Pancake Initialized, n_local: %lu\n", n_local);

}



void Grid3D::Initialize_Uniform_Particles(){
  
  //Initialize positions asigning one particle at each cell in a uniform grid
  
  int i, j, k, id;
  Real x_pos, y_pos, z_pos;
  
  Real dVol, Mparticle;
  dVol = H.dx * H.dy * H.dz;
  Mparticle = dVol;

  #ifdef SINGLE_PARTICLE_MASS
  Particles.particle_mass = Mparticle;
  #endif
  
  part_int_t pID = 0;
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;

        // // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        #ifdef PARTICLES_CPU
        Particles.pos_x.push_back( x_pos - 0.25*H.dx );
        Particles.pos_y.push_back( y_pos - 0.25*H.dy );
        Particles.pos_z.push_back( z_pos - 0.25*H.dz );
        Particles.vel_x.push_back( 0.0 );
        Particles.vel_y.push_back( 0.0 );
        Particles.vel_z.push_back( 0.0 );
        Particles.grav_x.push_back( 0.0 );
        Particles.grav_y.push_back( 0.0 );
        Particles.grav_z.push_back( 0.0 );
        #ifdef PARTICLE_IDS
        Particles.partIDs.push_back( pID );
        #endif
        #ifndef SINGLE_PARTICLE_MASS
        Particles.mass.push_back( Mparticle );
        #endif
        #endif //PARTICLES_CPU
        
        pID += 1;        
      }
    }
  }
  
  #ifdef PARTICLES_CPU
  Particles.n_local = Particles.pos_x.size();
  #endif
  
  
  #ifdef MPI_CHOLLA
  Particles.n_total_initial = ReducePartIntSum(Particles.n_local);
  #else
  Particles.n_total_initial = Particles.n_local;
  #endif

  chprintf( " Particles Uniform Grid Initialized, n_local: %lu, n_total: %lu\n", Particles.n_local, Particles.n_total_initial );
}


void Particles_3D::Free_Memory(void){
  
  //Free the particles arrays
  free(G.density);

  #ifdef PARTICLES_CPU
  free(G.gravity_x);
  free(G.gravity_y);
  free(G.gravity_z);

  //Free the particles vectors
  pos_x.clear();
  pos_y.clear();
  pos_z.clear();
  vel_x.clear();
  vel_y.clear();
  vel_z.clear();
  grav_x.clear();
  grav_y.clear();
  grav_z.clear();

  #ifdef PARTICLE_IDS
  partIDs.clear();
  #endif

  #ifndef SINGLE_PARTICLE_MASS
  mass.clear();
  #endif

  #endif //PARTICLES_CPU

  #ifdef MPI_CHOLLA
  //Free the particles arrays for MPI transfers
  free(send_buffer_x0_particles);
  free(send_buffer_x1_particles);
  free(recv_buffer_x0_particles);
  free(recv_buffer_x1_particles);
  free(send_buffer_y0_particles);
  free(send_buffer_y1_particles);
  free(recv_buffer_y0_particles);
  free(recv_buffer_y1_particles);
  free(send_buffer_z0_particles);
  free(send_buffer_z1_particles);
  free(recv_buffer_z0_particles);
  free(recv_buffer_z1_particles);
  #endif
}

void Particles_3D::Reset( void ){
  Free_Memory();
  
  #ifdef PARTICLES_GPU
  free(G.dti_array_host);
  Free_Memory_GPU();
  #endif
}

















#endif//PARTICLES
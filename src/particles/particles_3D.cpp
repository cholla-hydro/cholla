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

  n_local = 0;
  n_total = 0;
  n_total_initial = 0;
  
  dt = 0.0;
  t = 0.0;
  max_dt = 1e-3;

  C_cfl = 0.3;

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
  real_vector_t mass;
  #endif
  #ifdef PARTICLE_IDS
  int_vector_t partIDs;
  #endif

  #ifdef MPI_CHOLLA
  int_vector_t out_indxs_vec_x0;
  int_vector_t out_indxs_vec_x1;
  int_vector_t out_indxs_vec_y0;
  int_vector_t out_indxs_vec_y1;
  int_vector_t out_indxs_vec_z0;
  int_vector_t out_indxs_vec_z1;
  int_vector_t transfered_x0;
  int_vector_t transfered_x1;
  int_vector_t transfered_y0;
  int_vector_t transfered_y1;
  int_vector_t transfered_z0;
  int_vector_t transfered_z1;
  #endif

  //Initialize Grid Values
  G.nx_local = Grav.nx_local;
  G.ny_local = Grav.ny_local;
  G.nz_local = Grav.nz_local;
  G.nx_total = Grav.nx_total;
  G.ny_total = Grav.ny_total;
  G.nz_total = Grav.nz_total;

  G.dx = Grav.dx;
  G.dy = Grav.dy;
  G.dz = Grav.dz;

  G.xMin = Grav.xMin;
  G.yMin = Grav.yMin;
  G.zMin = Grav.zMin;

  G.xMax = G.xMin + G.nx_local*G.dx;
  G.yMax = G.yMin + G.ny_local*G.dy;
  G.zMax = G.zMin + G.nz_local*G.dz;

  G.domainMin_x = xbound;
  G.domainMax_x = xbound + xdglobal;
  G.domainMin_y = ybound;
  G.domainMax_y = ybound + ydglobal;
  G.domainMin_z = zbound;
  G.domainMax_z = zbound + zdglobal;
  
  G.n_ghost_particles_grid = 1;
  G.n_cells = (G.nx_local+2*G.n_ghost_particles_grid) * (G.ny_local+2*G.n_ghost_particles_grid) * (G.nz_local+2*G.n_ghost_particles_grid);

  INITIAL = true;
  TRANSFER_DENSITY_BOUNDARIES = false;
  TRANSFER_PARTICLES_BOUNDARIES = false;
  
  Allocate_Memory();

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
  
  #ifdef PARALLEL_OMP
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
  #endif
  


  
}

void Particles_3D::Allocate_Memory( void ){
  G.density   = (Real *) malloc(G.n_cells*sizeof(Real));
  G.gravity_x = (Real *) malloc(G.n_cells*sizeof(Real));
  G.gravity_y = (Real *) malloc(G.n_cells*sizeof(Real));
  G.gravity_z = (Real *) malloc(G.n_cells*sizeof(Real));
}


void Particles_3D::Initialize_Grid_Values( void ){
  int id;
  
  for( id=0; id<G.n_cells; id++ ){
    G.density[id] = 0;
    G.gravity_x[id] = 0;
    G.gravity_y[id] = 0;
    G.gravity_z[id] = 0;
  }
}

void Particles_3D::Initialize_Sphere( void ){
  int i, j, k, id;
  Real center_x, center_y, center_z, radius, sphereR;
  center_x = 0.5;
  center_y = 0.5;
  center_z = 0.5;;
  sphereR = 0.2;

  part_int_t n_particles_local = G.nx_local*G.ny_local*G.nz_local;
  part_int_t n_particles_total = G.nx_total*G.ny_total*G.nz_total;

  Real rho_start = 1;
  Real M_sphere = 4./3 * M_PI* rho_start * sphereR*sphereR*sphereR;
  Real Mparticle = M_sphere / n_particles_total;

  #ifdef SINGLE_PARTICLE_MASS
  particle_mass = Mparticle;
  #endif


  part_int_t pID = 0;
  Real pPos_x, pPos_y, pPos_z, r;
  while ( pID < n_particles_local ){
    pPos_x = Rand_Real( G.xMin, G.xMax );
    pPos_y = Rand_Real( G.yMin, G.yMax );
    pPos_z = Rand_Real( G.zMin, G.zMax );

    r = sqrt( (pPos_x-center_x)*(pPos_x-center_x) + (pPos_y-center_y)*(pPos_y-center_y) + (pPos_z-center_z)*(pPos_z-center_z) );
    if ( r > sphereR ) continue;
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
    pID += 1;
  }

  n_local = pos_x.size();

  chprintf( " Particles Uniform Sphere Initialized, n_local: %lu\n", n_local);

}


void Particles_3D::Initialize_Zeldovich_Pancake( struct parameters *P ){
  
  chprintf("Setting Zeldovich Pancake initial conditions...\n");
  
  n_local = pos_x.size();

  chprintf( " Particles Zeldovich Pancake Initialized, n_local: %lu\n", n_local);

}



void Grid3D::Initialize_Uniform_Particles(){
  
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
        pID += 1;        
      }
    }
  }
  
  Particles.n_local = Particles.pos_x.size();
  
  #ifdef MPI_CHOLLA
  Particles.n_total_initial = ReducePartIntSum(Particles.n_local);
  #else
  Particles.n_total_initial = Particles.n_local;
  #endif

  chprintf( " Particles Uniform Grid Initialized, n_local: %lu, n_total: %lu\n", Particles.n_local, Particles.n_total_initial );
}


void Particles_3D::Free_Memory(void){
 free(G.density);
 free(G.gravity_x);
 free(G.gravity_y);
 free(G.gravity_z);

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

 #ifdef MPI_CHOLLA
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
}

















#endif//PARTICLES
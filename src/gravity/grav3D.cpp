#ifdef GRAVITY

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../global/global.h"
#include "../io/io.h"

#include "../gravity/grav3D.h"

#ifdef PARALLEL_OMP
#include "../utils/parallel_omp.h"
#endif



Grav3D::Grav3D( void ){}

void Grav3D::Initialize( Real x_min, Real y_min, Real z_min, Real Lx, Real Ly, Real Lz, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx_real, Real dy_real, Real dz_real, int n_ghost_pot_offset, struct parameters *P )
{

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

  //Set Box local domain number of cells
  nx_local = nx_real;
  ny_local = ny_real;
  nz_local = nz_real;

  //Local n_cells without ghost cells
  n_cells = nx_local*ny_local*nz_local;
  //Local n_cells including ghost cells for the potential array
  n_cells_potential = ( nx_local + 2*N_GHOST_POTENTIAL ) * ( ny_local + 2*N_GHOST_POTENTIAL ) * ( nz_local + 2*N_GHOST_POTENTIAL );

  //Set Initial and dt used for the extrapolation of the potential;
  //The first timestep the potetential in not extrapolated ( INITIAL = TRUE )
  INITIAL = true;
  dt_prev = 0;
  dt_now = 0;

  #ifdef COSMOLOGY
  //Set the scale factor for cosmological simulations to 1,
  //This will be changed to the proper value when cosmology is initialized
  current_a = 1;
  #endif

  //Set the average density=0 ( Not Used )
  dens_avrg = 0;

  //Set the Gravitational Constant ( units must be consistent )
  Gconst = GN;
  if (strcmp(P->init, "Spherical_Overdensity_3D")==0){
    Gconst = 1;
    chprintf(" WARNING: Using Gravitational Constant G=1.\n");
  }

  //Flag to transfer the Potential boundaries
  TRANSFER_POTENTIAL_BOUNDARIES = false;

  // Flag to set the gravity boundariy flags
  BC_FLAGS_SET = false;

  AllocateMemory_CPU();

  #ifdef GRAVITY_GPU
  AllocateMemory_GPU();
  #endif

  Initialize_values_CPU();

  chprintf( "Gravity Initialized: \n Lbox: %0.2f %0.2f %0.2f \n Local: %d %d %d \n Global: %d %d %d \n",
      Lbox_x, Lbox_y, Lbox_z, nx_local, ny_local, nz_local,   nx_total, ny_total, nz_total );

  chprintf( " dx:%f  dy:%f  dz:%f\n", dx, dy, dz );
  chprintf( " N ghost potential: %d\n", N_GHOST_POTENTIAL);
  chprintf( " N ghost offset: %d\n", n_ghost_pot_offset);

  #ifdef PARALLEL_OMP
  chprintf(" Using OMP for gravity calculations\n");
  int n_omp_max = omp_get_max_threads();
  chprintf("  MAX OMP Threads: %d\n", n_omp_max);
  chprintf("  N OMP Threads per MPI process: %d\n", N_OMP_THREADS);
  #endif

  Poisson_solver.Initialize( Lbox_x, Lbox_y, Lbox_z, xMin, yMin, zMin, nx_total, ny_total, nz_total, nx_local, ny_local, nz_local, dx, dy, dz );
  #if defined(PARIS_TEST) || defined(PARIS_GALACTIC_TEST)
  Poisson_solver_test.Initialize( Lbox_x, Lbox_y, Lbox_z, xMin, yMin, zMin, nx_total, ny_total, nz_total, nx_local, ny_local, nz_local, dx, dy, dz );
  #endif
}

void Grav3D::AllocateMemory_CPU(void)
{
  // allocate memory for the density and potential arrays
  F.density_h    = (Real *) malloc(n_cells*sizeof(Real)); //array for the density
  F.potential_h  = (Real *) malloc(n_cells_potential*sizeof(Real));   //array for the potential at the n-th timestep
  F.potential_1_h  = (Real *) malloc(n_cells_potential*sizeof(Real)); //array for the potential at the (n-1)-th timestep
  boundary_flags = (int *) malloc(6*sizeof(int)); // array for the gtravity boundary flags

  #ifdef GRAV_ISOLATED_BOUNDARY_X
  F.pot_boundary_x0  = (Real *) malloc(N_GHOST_POTENTIAL*ny_local*nz_local*sizeof(Real)); //array for the potential isolated boundary
  F.pot_boundary_x1  = (Real *) malloc(N_GHOST_POTENTIAL*ny_local*nz_local*sizeof(Real));
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
  F.pot_boundary_y0  = (Real *) malloc(N_GHOST_POTENTIAL*nx_local*nz_local*sizeof(Real)); //array for the potential isolated boundary
  F.pot_boundary_y1  = (Real *) malloc(N_GHOST_POTENTIAL*nx_local*nz_local*sizeof(Real));
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  F.pot_boundary_z0  = (Real *) malloc(N_GHOST_POTENTIAL*nx_local*ny_local*sizeof(Real)); //array for the potential isolated boundary
  F.pot_boundary_z1  = (Real *) malloc(N_GHOST_POTENTIAL*nx_local*ny_local*sizeof(Real));
  #endif
}

void Grav3D::Set_Boundary_Flags( int *flags ){
  for (int i=0; i<6; i++) boundary_flags[i] = flags[i];
}

void Grav3D::Initialize_values_CPU(void){

  //Set initial values to 0.
  for (int id=0; id<n_cells; id++){
    F.density_h[id] = 0;
  }

  for (int id_pot=0; id_pot<n_cells_potential; id_pot++){
    F.potential_h[id_pot] = 0;
    F.potential_1_h[id_pot] = 0;
  }
}

void Grav3D::FreeMemory_CPU(void)
{
  free(F.density_h);
  free(F.potential_h);
  free(F.potential_1_h);
  free( boundary_flags );

  #ifdef GRAV_ISOLATED_BOUNDARY_X
  free(F.pot_boundary_x0);
  free(F.pot_boundary_x1);
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
  free(F.pot_boundary_y0);
  free(F.pot_boundary_y1);
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  free(F.pot_boundary_z0);
  free(F.pot_boundary_z1);
  #endif

  Poisson_solver.Reset();
  #if defined(PARIS_TEST) || defined(PARIS_GALACTIC_TEST)
  Poisson_solver_test.Reset();
  #endif
}

#endif //GRAVITY

#ifdef GRAVITY

#include"../grid3D.h"
#include"../global.h"
#include "../io.h"
#include "../error_handling.h"

#ifdef CUDA
#include "../cuda_mpi_routines.h"
#endif

#ifdef PARALLEL_OMP
#include"../parallel_omp.h"
#endif


void Grid3D::set_dt_Gravity(){
  
  Real dt_hydro = H.dt;
  
  #ifdef PARTICLES
  Real dt_particles, dt_min;
  
  #ifdef COSMOLOGY
  chprintf( " Current_z: %f \n", Cosmo.current_z );
  Real da_particles, da_min, dt_physical;
  #ifndef AVERAGE_SLOW_CELLS
  Particles.dt = Calc_Particles_dt_Cosmo();
  #endif
  dt_particles = Particles.dt;
  da_particles = Cosmo.Get_da_from_dt( dt_particles );
  da_particles = fmin( da_particles, 1.0 ); 
  
  #ifdef ONLY_PARTICLES
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif
  Particles.dt = Calc_Particles_dt_Cosmo();
  da_particles = Cosmo.Get_da_from_dt( Particles.dt );
  #ifdef CPU_TIME
  Timer.End_and_Record_Time(0);
  #endif
  da_min = da_particles;
  chprintf( " Delta_a_particles: %f \n", da_particles );
  #else
  Real da_hydro;
  da_hydro = Cosmo.Get_da_from_dt( dt_hydro ) * Cosmo.current_a * Cosmo.current_a / Cosmo.H0;
  da_min = fmin( da_hydro, da_particles );
  chprintf( " Delta_a_particles: %f      Delta_a_gas: %f   \n", da_particles, da_hydro );
  #endif//ONLY_PARTICLES
  
  
  Cosmo.max_delta_a = fmin( MAX_EXPANSION_RATE * Cosmo.current_a, MAX_DELTA_A );
  if( da_min > Cosmo.max_delta_a){
    da_min = Cosmo.max_delta_a;
    chprintf( " Seting max delta_a: %f\n", da_min );
  }
  
  #ifdef COOLING_GRACKLE
  if ( fabs(Cosmo.current_a + da_min - Cool.scale_factor_UVB_on) < 0.002 ){
    da_min /= 5;
    chprintf( " Starting UVB. Limiting delta_a:  %f \n", da_min);
  }
  #endif
  
  if ( (Cosmo.current_a + da_min) >  Cosmo.next_output ){
    da_min = Cosmo.next_output - Cosmo.current_a;
    H.Output_Now = true;
  }
  
  
  Cosmo.delta_a = da_min;
  dt_min = Cosmo.Get_dt_from_da( Cosmo.delta_a ) * Cosmo.H0 / ( Cosmo.current_a * Cosmo.current_a );
  H.dt = dt_min; 
  chprintf( " Current_a: %f    delta_a: %f     dt:  %f\n", Cosmo.current_a, Cosmo.delta_a, H.dt  );
  
  dt_physical = Cosmo.Get_dt_from_da( Cosmo.delta_a );
  Cosmo.dt_secs = dt_physical * Cosmo.time_conversion;
  Cosmo.t_secs += Cosmo.dt_secs;
  chprintf( " t_physical: %f Myr   dt_physical: %f Myr\n", Cosmo.t_secs/MYR, Cosmo.dt_secs/MYR );  
  Particles.dt = dt_physical; 
  
  #else // Not Cosmology
  dt_particles = Calc_Particles_dt();
  dt_particles = fmin( dt_particles, Particles.max_dt);
  #ifdef ONLY_PARTICLES
  dt_min = dt_particles;
  chprintf( " dt_particles: %f \n", dt_particles );
  #else
  chprintf( " dt_hydro: %f   dt_particles: %f \n", dt_hydro, dt_particles );
  dt_min = fmin( dt_hydro, dt_particles );
  #endif//ONLY_PARTICLES
  H.dt = dt_min;
  Particles.dt = H.dt;
  #endif//COSMOLOGY
  #endif//PARTICLES
  
  // Set times for potential extrapolation
  if ( Grav.INITIAL ){
    Grav.dt_prev = H.dt;
    Grav.dt_now = H.dt;
  }else{
    Grav.dt_prev = Grav.dt_now;
    Grav.dt_now = H.dt;
  }
}


Real Grav3D::Get_Average_Density(){
  
  Real dens_sum, dens_mean;

  #ifndef PARALLEL_OMP
  dens_sum = Get_Average_Density_function( 0, nz_local );
  #else
  dens_sum = 0;
  Real dens_sum_all[N_OMP_THREADS];
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs( nz_local, n_omp_procs, omp_id, &g_start, &g_end  );
    dens_sum_all[omp_id] = Get_Average_Density_function(  g_start, g_end );

  }
  for ( int i=0; i<N_OMP_THREADS; i++ ){
    dens_sum += dens_sum_all[i];
  }
  #endif

  dens_mean = dens_sum /  ( nx_local * ny_local * nz_local);

  Real dens_avrg_all;
  #ifdef MPI_CHOLLA
  dens_avrg_all = ReduceRealAvg( dens_mean );
  #else
  dens_avrg_all = dens_mean;
  #endif

  dens_avrg = dens_avrg_all;
  
  return dens_avrg_all;

}

Real Grav3D::Get_Average_Density_function( int g_start, int g_end){
  
  int nx = nx_local;
  int ny = ny_local;
  int nz = nz_local;
  int k, j, i, id;
  Real dens_sum=0;
  for( k=g_start; k<g_end; k++){
    for( j=0; j<ny; j++){
      for( i=0; i<nx; i++){
        id = (i) + (j)*nx + (k)*nx*ny;
        dens_sum += F.density_h[id];
      }
    }
  }
  return dens_sum;
}
  




void Grid3D::Initialize_Gravity( struct parameters *P ){
  chprintf( "\nInitializing Gravity... \n");
  Grav.Initialize( H.xblocal, H.yblocal, H.zblocal, H.xdglobal, H.ydglobal, H.zdglobal, P->nx, P->ny, P->nz, H.nx_real, H.ny_real, H.nz_real, H.dx, H.dy, H.dz, H.n_ghost_potential_offset  );
  chprintf( "Gravity Successfully Initialized. \n\n");
  
}

void Grid3D::Compute_Gravitational_Potential( struct parameters *P ){
  
  Real Grav_Constant = Grav.Gconst;
  
  Real dens_avrg, current_a;
  
  dens_avrg = 0;
  current_a = 1;
  
  
  #ifdef PARTICLES
  Copy_Particles_Density_to_Gravity( *P);
  #endif
  
  
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif
  
  #ifndef ONLY_PARTICLES
  Copy_Hydro_Density_to_Gravity();
  #endif

  #ifdef COSMOLOGY
  Grav_Constant = Cosmo.cosmo_G;
  current_a = Cosmo.current_a;
  dens_avrg = Cosmo.rho_0_gas;
  #endif
  
  Grav.Poisson_solver.Get_Potential( Grav.F.density_h, Grav.F.potential_h, Grav_Constant, dens_avrg, current_a);
  
    
  #ifdef CPU_TIME
  Timer.End_and_Record_Time( 3 );
  #endif
  
}

void Grid3D::Copy_Hydro_Density_to_Gravity_Function( int g_start, int g_end){
  
  // Copy the density array from hydro conserved to gravity density array
  Real dens;
  int i, j, k, id, id_grav;
  for (k=g_start; k<g_end; k++) {
    for (j=0; j<Grav.ny_local; j++) {
      for (i=0; i<Grav.nx_local; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        id_grav = (i) + (j)*Grav.nx_local + (k)*Grav.nx_local*Grav.ny_local;

        dens = C.density[id];
        
        #ifdef COSMOLOGY
        dens *= Cosmo.rho_0_gas;
        #endif

        #ifdef PARTICLES
        Grav.F.density_h[id_grav] += dens; //Hydro density is added AFTER partices density
        #else
        Grav.F.density_h[id_grav] = dens;
        #endif
        
      }
    }
  }
}

void Grid3D::Copy_Hydro_Density_to_Gravity(){
  
  #ifndef PARALLEL_OMP
  Copy_Hydro_Density_to_Gravity_Function( 0, Grav.nz_local );
  #else
  
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs( Grav.nz_local, n_omp_procs, omp_id, &g_start, &g_end  );

    Copy_Hydro_Density_to_Gravity_Function(g_start, g_end );
  }
  #endif
  
}



void Grid3D::Extrapolate_Grav_Potential_Function( int g_start, int g_end ){
  
  int nx_pot = Grav.nx_local + 2*N_GHOST_POTENTIAL;
  int ny_pot = Grav.ny_local + 2*N_GHOST_POTENTIAL;
  int nz_pot = Grav.nz_local + 2*N_GHOST_POTENTIAL;
  
  int n_ghost_grid, nx_grid, ny_grid, nz_grid;
  Real *potential_in, *potential_out;
  
  potential_in = Grav.F.potential_h; 
  
  potential_out = C.Grav_potential;
  n_ghost_grid = H.n_ghost;
  
  nx_grid = Grav.nx_local + 2*n_ghost_grid;
  ny_grid = Grav.ny_local + 2*n_ghost_grid;
  nz_grid = Grav.nz_local + 2*n_ghost_grid;
  
  int nGHST = n_ghost_grid - N_GHOST_POTENTIAL;
  Real pot_now, pot_prev, pot_extrp;
  int k, j, i, id_pot, id_grid;
  for ( k=g_start; k<g_end; k++ ){
    for ( j=0; j<ny_pot; j++ ){
      for ( i=0; i<nx_pot; i++ ){
        id_pot = i + j*nx_pot + k*nx_pot*ny_pot;
        id_grid = (i+nGHST) + (j+nGHST)*nx_grid + (k+nGHST)*nx_grid*ny_grid;
        pot_now = potential_in[id_pot]  ;
        if ( Grav.INITIAL ){
          pot_extrp = pot_now;
        } else {
          pot_prev = Grav.F.potential_1_h[id_pot];
          pot_extrp = pot_now  + 0.5 * Grav.dt_now * ( pot_now - pot_prev  ) / Grav.dt_prev;
        }
        
        #ifdef COSMOLOGY
        pot_extrp *= Cosmo.current_a * Cosmo.current_a / Cosmo.phi_0_gas;
        #endif
        
        potential_out[id_grid] = pot_extrp;
        Grav.F.potential_1_h[id_pot] = pot_now;
      }
    }
  }
}

void Grid3D::Extrapolate_Grav_Potential(){
  
  #ifndef PARALLEL_OMP
  Extrapolate_Grav_Potential_Function( 0, Grav.nz_local + 2*N_GHOST_POTENTIAL );
  #else
  
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;
    
    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs( Grav.nz_local + 2*N_GHOST_POTENTIAL, n_omp_procs, omp_id,  &g_start, &g_end  );
    
    Extrapolate_Grav_Potential_Function( g_start, g_end );
  }
  #endif
  
  Grav.INITIAL = false;  
}


#endif //GRAVITY
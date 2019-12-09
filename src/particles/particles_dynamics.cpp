#ifdef PARTICLES


#include<stdio.h>
#include<stdlib.h>
#include"math.h"
#include <iostream>
#include"../global.h"
#include"../grid3D.h"
#include"particles_3D.h"
#include"../io.h"

#ifdef PARALLEL_OMP
#include"../parallel_omp.h"
#endif


//Compute the delta_t for the particles 
Real Grid3D::Calc_Particles_dt( ){
  
  Real dt_particles;
  #ifndef PARALLEL_OMP
  dt_particles = Calc_Particles_dt_function( 0, Particles.n_local );
  #else 
  dt_particles = 1e100;
  Real dt_particles_all[N_OMP_THREADS];
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;
    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );
    dt_particles_all[omp_id] = Calc_Particles_dt_function( p_start, p_end );
  }
  
  for ( int i=0; i<N_OMP_THREADS; i++ ){
    dt_particles = fmin( dt_particles, dt_particles_all[i]);
  }
  #endif 
  
  Real dt_particles_global;
  #ifdef MPI_CHOLLA
  dt_particles_global = ReduceRealMin( dt_particles );
  #else
  dt_particles_global = dt_particles;
  #endif
  
  return dt_particles_global;  
}


//Loop over the particles anf compute dt_min
Real Grid3D::Calc_Particles_dt_function( part_int_t p_start, part_int_t p_end ){
  part_int_t pID;
  Real dt, dt_min, vel;
  dt_min = 1e100;

  for ( pID=p_start; pID<p_end; pID++ ){
    vel = fabs(Particles.vel_x[pID]);
    if ( vel > 0){
      dt = Particles.G.dx / vel;
      dt_min = std::min( dt_min, dt);
    }
    vel = fabs(Particles.vel_y[pID]);
    if ( vel > 0){
      dt = Particles.G.dy / vel;
      dt_min = std::min( dt_min, dt);
    }
    vel = fabs(Particles.vel_z[pID]);
    if ( vel > 0){
      dt = Particles.G.dz / vel;
      dt_min = std::min( dt_min, dt);
    }
  }
  return Particles.C_cfl * dt_min;  
}

//Updated the particles positions and velocities
void Grid3D::Advance_Particles( int N_step ){
  
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif
  
  #ifdef PARTICLES_KDK
  //Updated the velocities by 0.5*delta_t and update the positions by delta_t
  if ( N_step == 1 ) Advance_Particles_KDK_Step1();
  #endif
  
  if ( N_step == 2 ){
    //Compute the particles accelerations at the new positions
    Get_Particles_Acceleration();
    
    #ifdef PARTICLES_KDK
    //Advance the particles velocities by the remaining 0.5*delta_t
    Advance_Particles_KDK_Step2();
    #endif

  }
  
  #ifdef CPU_TIME
  if ( N_step == 1) Timer.End_and_Record_Time(6);
  if ( N_step == 2) Timer.End_and_Record_Time(7);
  #endif
    
}

void Grid3D::Get_Particles_Acceleration(){
  // Get the accteleration for all the particles
  
  //First compute the gravitational field at the center of the grid cells
  Get_Gravity_Field_Particles();
  
  //Then Interpolate the gravitational field from the centers of the cells to the positions of the particles 
  Get_Gravity_CIC();  
}

//Update positions and velocities (step 1 of KDK scheme )
void Grid3D::Advance_Particles_KDK_Step1( ){
  
  #ifndef PARALLEL_OMP
  #ifdef COSMOLOGY
  Advance_Particles_KDK_Cosmo_Step1_function( 0, Particles.n_local );  
  #else
  Advance_Particles_KDK_Step1_function( 0, Particles.n_local );
  #endif//COSMOLOGY
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;
    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );
    #ifdef COSMOLOGY
    Advance_Particles_KDK_Cosmo_Step1_function( p_start, p_end );
    #else
    Advance_Particles_KDK_Step1_function( p_start, p_end );
    #endif//COSMOLOGY
  }
  #endif  
}


//Update velocities (step 2 of KDK scheme )
void Grid3D::Advance_Particles_KDK_Step2( ){
  
  #ifndef PARALLEL_OMP
  #ifdef COSMOLOGY
  Advance_Particles_KDK_Cosmo_Step2_function( 0, Particles.n_local );  
  #else
  Advance_Particles_KDK_Step2_function( 0, Particles.n_local );
  #endif//COSMOLOGY
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;
    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );
    #ifdef COSMOLOGY
    Advance_Particles_KDK_Cosmo_Step2_function( p_start, p_end );
    #else
    Advance_Particles_KDK_Step2_function( p_start, p_end );
    #endif//COSMOLOGY
  }
  #endif  
}


//Update positions and velocities (step 1 of KDK scheme )
void Grid3D::Advance_Particles_KDK_Step1_function( part_int_t p_start, part_int_t p_end ){
  
  part_int_t pID;
  Real dt = Particles.dt;
  // Advance velocities by half a step
  for ( pID=p_start; pID<p_end; pID++ ){
    Particles.vel_x[pID] += 0.5 * dt * Particles.grav_x[pID];
    Particles.vel_y[pID] += 0.5 * dt * Particles.grav_y[pID];
    Particles.vel_z[pID] += 0.5 * dt * Particles.grav_z[pID];
  }

  //Advance Posiotions using advanced velocities
  for ( pID=p_start; pID<p_end; pID++ ){
    Particles.pos_x[pID] += dt * Particles.vel_x[pID];
    Particles.pos_y[pID] += dt * Particles.vel_y[pID];
    Particles.pos_z[pID] += dt * Particles.vel_z[pID];
  }
}


//Update  velocities (step 2 of KDK scheme )
void Grid3D::Advance_Particles_KDK_Step2_function( part_int_t p_start, part_int_t p_end ){
  
  part_int_t pID;
  Real dt = Particles.dt;
  // Advance velocities by the second half a step
  for ( pID=p_start; pID<p_end; pID++ ){
    Particles.vel_x[pID] += 0.5 * dt * Particles.grav_x[pID];
    Particles.vel_y[pID] += 0.5 * dt * Particles.grav_y[pID];
    Particles.vel_z[pID] += 0.5 * dt * Particles.grav_z[pID];
  }
}

#ifdef COSMOLOGY
//Compute the delta_t for the particles  COSMOLOGICAL SIMULATION
Real Grid3D::Calc_Particles_dt_Cosmo(){
  
  Real dt_particles;
  #ifndef PARALLEL_OMP
  dt_particles = Calc_Particles_dt_Cosmo_function( 0, Particles.n_local );
  #else 
  dt_particles = 1e100;
  Real dt_particles_all[N_OMP_THREADS];
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;
    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );
    dt_particles_all[omp_id] = Calc_Particles_dt_Cosmo_function( p_start, p_end );
  }
  
  for ( int i=0; i<N_OMP_THREADS; i++ ){
    dt_particles = fmin( dt_particles, dt_particles_all[i]);
  }
  #endif 
  
  Real dt_particles_global;
  #ifdef MPI_CHOLLA
  dt_particles_global = ReduceRealMin( dt_particles );
  #else
  dt_particles_global = dt_particles;
  #endif
  
  return dt_particles_global;   
}

Real Grid3D::Calc_Particles_dt_Cosmo_function( part_int_t p_start, part_int_t p_end ){

  part_int_t pID;
  Real da, da_min, vel, dt_min;
  da_min = 1e100;
  Real scale_factor = 1 / ( Cosmo.current_a * Cosmo.Get_Hubble_Parameter( Cosmo.current_a) ) * Cosmo.cosmo_h;
  Real a2 = ( Cosmo.current_a )*( Cosmo.current_a  );
  
  Real vel_factor;  
  vel_factor = Cosmo.current_a / scale_factor;
  
  Real vx_max, vy_max, vz_max;
  vx_max = 0;
  vy_max = 0;
  vz_max = 0;

  for ( pID=p_start; pID<p_end; pID++ ){
    vx_max = fmax( vx_max,  fabs(Particles.vel_x[pID]) );
    vy_max = fmax( vy_max,  fabs(Particles.vel_y[pID]) );
    vz_max = fmax( vz_max,  fabs(Particles.vel_z[pID]) );  
  } 

  da_min = fmin( Particles.G.dx / vx_max, Particles.G.dy / vy_max  );
  da_min = fmin( Particles.G.dz / vz_max, da_min  );
  da_min *= vel_factor;
  dt_min = Cosmo.Get_dt_from_da( da_min );
  return Particles.C_cfl * dt_min;
}




//Update positions and velocities (step 1 of KDK scheme ) COSMOLOGICAL SIMULATION
void Grid3D::Advance_Particles_KDK_Cosmo_Step1_function( part_int_t p_start, part_int_t p_end ){
  
  Real dt, dt_half;
  part_int_t pIndx;
  Real a = Cosmo.current_a;
  Real da = Cosmo.delta_a;
  Real da_half = da/2;
  Real a_half = a + da_half;
  
  Real H, H_half;
  H = Cosmo.Get_Hubble_Parameter( a );
  H_half = Cosmo.Get_Hubble_Parameter( a_half );
  
  dt = da / ( a * H ) * Cosmo.cosmo_h;
  dt_half = da / ( a_half * H_half ) * Cosmo.cosmo_h / ( a_half );

  // Advance velocities by half a step
  Real pos_x, vel_x, grav_x;
  Real pos_y, vel_y, grav_y;
  Real pos_z, vel_z, grav_z;
  for ( pIndx=p_start; pIndx<p_end; pIndx++ ){
    pos_x = Particles.pos_x[pIndx];
    pos_y = Particles.pos_y[pIndx];
    pos_z = Particles.pos_z[pIndx];
    vel_x = Particles.vel_x[pIndx];
    vel_y = Particles.vel_y[pIndx];
    vel_z = Particles.vel_z[pIndx];
    grav_x = Particles.grav_x[pIndx];
    grav_y = Particles.grav_y[pIndx];
    grav_z = Particles.grav_z[pIndx];
    
    vel_x = ( a*vel_x + 0.5*dt*grav_x ) / a_half;
    vel_y = ( a*vel_y + 0.5*dt*grav_y ) / a_half;
    vel_z = ( a*vel_z + 0.5*dt*grav_z ) / a_half;
    pos_x += dt_half * vel_x;
    pos_y += dt_half * vel_y;
    pos_z += dt_half * vel_z;    
    
    Particles.pos_x[pIndx] = pos_x;
    Particles.pos_y[pIndx] = pos_y;
    Particles.pos_z[pIndx] = pos_z;

    Particles.vel_x[pIndx] = vel_x;
    Particles.vel_y[pIndx] = vel_y;
    Particles.vel_z[pIndx] = vel_z;
  }
}


//Update velocities (step 2 of KDK scheme ) COSMOLOGICAL SIMULATION
void Grid3D::Advance_Particles_KDK_Cosmo_Step2_function( part_int_t p_start, part_int_t p_end ){
  Real dt;
  part_int_t pIndx;
  Real a = Cosmo.current_a;
  Real da = Cosmo.delta_a;
  Real da_half = da / 2;
  Real a_half = a - da + da_half;
  
  dt = da / ( a * Cosmo.Get_Hubble_Parameter( a ) ) * Cosmo.cosmo_h;

  // Advance velocities by half a step
  Real grav_x, grav_y, grav_z;
  Real vel_x, vel_y, vel_z;
  for ( pIndx=p_start; pIndx<p_end; pIndx++ ){
    grav_x = Particles.grav_x[pIndx];
    grav_y = Particles.grav_y[pIndx];
    grav_z = Particles.grav_z[pIndx];
    
    vel_x = Particles.vel_x[pIndx];
    vel_y = Particles.vel_y[pIndx];
    vel_z = Particles.vel_z[pIndx];
    Particles.vel_x[pIndx] = ( a_half*vel_x + 0.5*dt*grav_x ) / a;
    Particles.vel_y[pIndx] = ( a_half*vel_y + 0.5*dt*grav_y ) / a;
    Particles.vel_z[pIndx] = ( a_half*vel_z + 0.5*dt*grav_z ) / a;
  
  }
}






#endif












#endif//PARTICLES
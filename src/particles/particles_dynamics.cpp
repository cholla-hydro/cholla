#ifdef PARTICLES


#include<stdio.h>
#include<stdlib.h>
#include"math.h"
#include <iostream>
#include"../global.h"
#include"../grid3D.h"
#include"particles_3D.h"

#ifdef PARALLEL_OMP
#include"../parallel_omp.h"
#endif


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


void Grid3D::Advance_Particles( int N_KDK_step ){
  
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif
  
  if ( N_KDK_step == 1 ) Advance_Particles_KDK_Step1();
  
  if ( N_KDK_step == 2 ){
    Get_Particles_Accelration();
    Advance_Particles_KDK_Step2();
  }
  
  #ifdef CPU_TIME
  if ( N_KDK_step == 1) Timer.End_and_Record_Time(6);
  if ( N_KDK_step == 2) Timer.End_and_Record_Time(7);
  #endif
    
}

void Grid3D::Get_Particles_Accelration(){
  // Get the accteleration for all the particles
  Get_Gravity_Field_Particles();
  Get_Gravity_CIC();  
}

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
  Real scale_factor = Cosmo.Scale_Function( Cosmo.current_a , Cosmo.Omega_M, Cosmo.Omega_L, Cosmo.Omega_K  ) / Cosmo.H0 * Cosmo.cosmo_h;
  Real a2 = ( Cosmo.current_a )*( Cosmo.current_a  );
  Real vel_factor = a2 / scale_factor;


  for ( pID=0; pID<Particles.n_local; pID++ ){
    vel = fabs(Particles.vel_x[pID]);
    if ( vel > 0){
      da = Particles.G.dx * vel_factor / vel;
      da_min = std::min( da_min, da);
    }
    vel = fabs(Particles.vel_y[pID]);
    if ( vel > 0){
      da = Particles.G.dy * vel_factor / vel;
      da_min = std::min( da_min, da);
    }
    vel = fabs(Particles.vel_z[pID]);
    if ( vel > 0){
      da = Particles.G.dz * vel_factor / vel;
      da_min = std::min( da_min, da);
    }
  } 
  dt_min = Cosmo.Get_dt_from_da( da_min );
  return Particles.C_cfl * dt_min;
}
  
  




void Grid3D::Advance_Particles_KDK_Cosmo_Step1_function( part_int_t p_start, part_int_t p_end ){
  
  part_int_t pIndx;
  Real da = Cosmo.delta_a;
  Real current_a = Cosmo.current_a;

  Real scale_factor = Cosmo.Scale_Function( current_a, Cosmo.Omega_M, Cosmo.Omega_L, Cosmo.Omega_K ) / Cosmo.H0 * Cosmo.cosmo_h;
  Real scale_factor_1 = Cosmo.Scale_Function( current_a + 0.5*da, Cosmo.Omega_M, Cosmo.Omega_L, Cosmo.Omega_K  ) / Cosmo.H0 * Cosmo.cosmo_h;
  Real a2_inv = 1./( ( current_a + 0.5*da )*( current_a + 0.5*da ));
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

    vel_x += scale_factor * 0.5 * da * grav_x;
    vel_y += scale_factor * 0.5 * da * grav_y;
    vel_z += scale_factor * 0.5 * da * grav_z;

    pos_x += a2_inv * scale_factor_1 * da * vel_x;
    pos_y += a2_inv * scale_factor_1 * da * vel_y;
    pos_z += a2_inv * scale_factor_1 * da * vel_z;
    
    if ( pos_x < Particles.G.xMin || pos_x >= Particles.G.xMax ){
      pos_x = Particles.pos_x[pIndx];
      vel_x = Particles.vel_x[pIndx];
    }
    if ( pos_y < Particles.G.yMin || pos_y >= Particles.G.yMax ){
      pos_y = Particles.pos_y[pIndx];
      vel_y = Particles.vel_y[pIndx];
    }
    if ( pos_z < Particles.G.zMin || pos_z >= Particles.G.zMax ){
      pos_z = Particles.pos_z[pIndx];
      vel_z = Particles.vel_z[pIndx];
    }
      
      
    // Particles.pos_x[pIndx] = pos_x;
    // Particles.pos_y[pIndx] = pos_y;
    // Particles.pos_z[pIndx] = pos_z;
    // 
    // Particles.vel_x[pIndx] = vel_x;
    // Particles.vel_y[pIndx] = vel_y;
    // Particles.vel_z[pIndx] = vel_z;
  }
}

void Grid3D::Advance_Particles_KDK_Cosmo_Step2_function( part_int_t p_start, part_int_t p_end ){
  part_int_t pIndx;
  Real da = Cosmo.delta_a;
  Real current_a = Cosmo.current_a;

  Real scale_factor = Cosmo.Scale_Function( current_a , Cosmo.Omega_M, Cosmo.Omega_L, Cosmo.Omega_K ) / Cosmo.H0 * Cosmo.cosmo_h;
  // Advance velocities by half a step
  Real grav_x;
  Real grav_y;
  Real grav_z;
  // for ( pIndx=p_start; pIndx<p_end; pIndx++ ){
  //   grav_x = Particles.grav_x[pIndx];
  //   grav_y = Particles.grav_y[pIndx];
  //   grav_z = Particles.grav_z[pIndx];
  //   Particles.vel_x[pIndx] += scale_factor * 0.5 * da * grav_x;
  //   Particles.vel_y[pIndx] += scale_factor * 0.5 * da * grav_y;
  //   Particles.vel_z[pIndx] += scale_factor * 0.5 * da * grav_z;
  // }
}






#endif












#endif//PARTICLES
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
  Advance_Particles_KDK_Step1_function( 0, Particles.n_local );
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;
    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );
    Advance_Particles_KDK_Step1_function( p_start, p_end );
  }
  #endif  
}

void Grid3D::Advance_Particles_KDK_Step2( ){
  
  #ifndef PARALLEL_OMP
  Advance_Particles_KDK_Step2_function( 0, Particles.n_local );
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;
    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );
    Advance_Particles_KDK_Step2_function( p_start, p_end );
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












#endif//PARTICLES
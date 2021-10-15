#ifdef PARTICLES


#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <iostream>
#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../particles/particles_3D.h"
#include "../io/io.h"

#ifdef PARALLEL_OMP
#include "../utils/parallel_omp.h"
#endif


//Compute the delta_t for the particles
Real Grid3D::Calc_Particles_dt( ){

  Real dt_particles;

  #ifdef PARTICLES_CPU

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
  #endif //PARALLEL_OMP
  #endif //PARTICLES_CPU


  #ifdef PARTICLES_GPU
  dt_particles = Calc_Particles_dt_GPU();
  #endif//PARTICLES_GPU



  Real dt_particles_global;
  #ifdef MPI_CHOLLA
  dt_particles_global = ReduceRealMin( dt_particles );
  #else
  dt_particles_global = dt_particles;
  #endif

  return dt_particles_global;
}


#ifdef PARTICLES_GPU

//Go over all the particles and find dt_min in the GPU
Real Grid3D::Calc_Particles_dt_GPU(){

  // set values for GPU kernels
  int ngrid =  (Particles.n_local + TPB_PARTICLES - 1) / TPB_PARTICLES;


  if ( ngrid > Particles.G.size_blocks_array ) chprintf(" Error: particles dt_array too small\n");


  Real max_dti;
  max_dti = Particles.Calc_Particles_dt_GPU_function( ngrid, Particles.n_local, Particles.G.dx, Particles.G.dy, Particles.G.dz, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.G.dti_array_host, Particles.G.dti_array_dev );

  Real dt_min;

  #ifdef COSMOLOGY
  Real scale_factor, vel_factor, da_min;
  scale_factor = 1 / ( Cosmo.current_a * Cosmo.Get_Hubble_Parameter( Cosmo.current_a) ) * Cosmo.cosmo_h;
  vel_factor = Cosmo.current_a / scale_factor;
  da_min = vel_factor / max_dti;
  dt_min = Cosmo.Get_dt_from_da( da_min );
  #else
  dt_min = 1 / max_dti;
  #endif

  return Particles.C_cfl*dt_min;

}

//Update positions and velocities (step 1 of KDK scheme ) in the GPU
void Grid3D::Advance_Particles_KDK_Step1_GPU(){

  #ifdef COSMOLOGY
  Particles.Advance_Particles_KDK_Step1_Cosmo_GPU_function( Particles.n_local, Cosmo.delta_a, Particles.pos_x_dev, Particles.pos_y_dev, Particles.pos_z_dev, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.grav_x_dev, Particles.grav_y_dev, Particles.grav_z_dev, Cosmo.current_a, Cosmo.H0, Cosmo.cosmo_h, Cosmo.Omega_M, Cosmo.Omega_L, Cosmo.Omega_K );
  #else
  Particles.Advance_Particles_KDK_Step1_GPU_function( Particles.n_local, Particles.dt, Particles.pos_x_dev, Particles.pos_y_dev, Particles.pos_z_dev, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.grav_x_dev, Particles.grav_y_dev, Particles.grav_z_dev );
  #endif


}

//Update velocities (step 2 of KDK scheme ) in the GPU
void Grid3D::Advance_Particles_KDK_Step2_GPU(){

  #ifdef COSMOLOGY
  Particles.Advance_Particles_KDK_Step2_Cosmo_GPU_function( Particles.n_local, Cosmo.delta_a, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.grav_x_dev, Particles.grav_y_dev, Particles.grav_z_dev, Cosmo.current_a, Cosmo.H0, Cosmo.cosmo_h, Cosmo.Omega_M, Cosmo.Omega_L, Cosmo.Omega_K );
  #else
  Particles.Advance_Particles_KDK_Step2_GPU_function( Particles.n_local, Particles.dt, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.grav_x_dev, Particles.grav_y_dev, Particles.grav_z_dev );
  #endif


}


#endif //PARTICLES_GPU




#ifdef PARTICLES_CPU

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
#endif //PARTICLES_CPU

//Update the particles positions and velocities
void Grid3D::Advance_Particles( int N_step ){

  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif

  #ifdef PARTICLES_KDK
  //Update the velocities by 0.5*delta_t and update the positions by delta_t
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

// Get the accteleration for all the particles
void Grid3D::Get_Particles_Acceleration(){

  //First compute the gravitational field at the center of the grid cells
  Get_Gravity_Field_Particles();

  //Then Interpolate the gravitational field from the centers of the cells to the positions of the particles
  Get_Gravity_CIC();
}

//Update positions and velocities (step 1 of KDK scheme )
void Grid3D::Advance_Particles_KDK_Step1( ){

  #ifdef PARTICLES_CPU

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
  #endif //PARALLEL_OMP
  #endif //PARTICLES_CPU

  #ifdef PARTICLES_GPU
  Advance_Particles_KDK_Step1_GPU();
  #endif //PARTICLES_GPU

}

//Update velocities (step 2 of KDK scheme )
void Grid3D::Advance_Particles_KDK_Step2( ){

  #ifdef PARTICLES_CPU

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
  #endif //PARALLEL_OMP
  #endif //PARTICLES_CPU

  #ifdef PARTICLES_GPU
  Advance_Particles_KDK_Step2_GPU();
  #endif //PARTICLES_GPU

}

#ifdef PARTICLES_CPU
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

  //Advance Positions by delta_t using the updated velocities
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
#endif //PARTICLES_CPU

#ifdef COSMOLOGY

//Compute the delta_t for the particles  COSMOLOGICAL SIMULATION
Real Grid3D::Calc_Particles_dt_Cosmo(){

  Real dt_particles;

  #ifdef PARTICLES_CPU

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
  #endif //PARALLEL_OMP
  #endif //PARTICLES_CPU

  #ifdef PARTICLES_GPU
  dt_particles = Calc_Particles_dt_GPU();
  #endif//PARTICLES_GPU

  Real dt_particles_global;
  #ifdef MPI_CHOLLA
  dt_particles_global = ReduceRealMin( dt_particles );
  #else
  dt_particles_global = dt_particles;
  #endif

  return dt_particles_global;
}


#ifdef PARTICLES_CPU
//Loop over the particles anf compute dt_min for a cosmological simulation
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

    // Advance velocities by half a step
    vel_x = ( a*vel_x + 0.5*dt*grav_x ) / a_half;
    vel_y = ( a*vel_y + 0.5*dt*grav_y ) / a_half;
    vel_z = ( a*vel_z + 0.5*dt*grav_z ) / a_half;

    //Advance the positions by delta_t using the updated velocities
    pos_x += dt_half * vel_x;
    pos_y += dt_half * vel_y;
    pos_z += dt_half * vel_z;


    //Save the updated positions and velocities
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

  Real grav_x, grav_y, grav_z;
  Real vel_x, vel_y, vel_z;
  for ( pIndx=p_start; pIndx<p_end; pIndx++ ){
    grav_x = Particles.grav_x[pIndx];
    grav_y = Particles.grav_y[pIndx];
    grav_z = Particles.grav_z[pIndx];

    vel_x = Particles.vel_x[pIndx];
    vel_y = Particles.vel_y[pIndx];
    vel_z = Particles.vel_z[pIndx];

    // Advance velocities by half a step
    Particles.vel_x[pIndx] = ( a_half*vel_x + 0.5*dt*grav_x ) / a;
    Particles.vel_y[pIndx] = ( a_half*vel_y + 0.5*dt*grav_y ) / a;
    Particles.vel_z[pIndx] = ( a_half*vel_z + 0.5*dt*grav_z ) / a;

  }
}


#endif //PARTICLES_CPU



#endif //COSMOLOGY












#endif//PARTICLES


#ifdef CPU_TIME

#ifndef TIMING_FUNCTIONS_H
#define TIMING_FUNCTIONS_H

#include "../global/global.h"


class Time
{
public:

  int n_steps;

  // Real start_step;
  // Real end_step;
  // Real time_step;

  Real time_start;
  Real time_end;
  Real time;


  Real time_hydro_min;
  Real time_hydro_max;
  Real time_hydro_mean;
  Real time_hydro_all;

  Real time_bound_min;
  Real time_bound_max;
  Real time_bound_mean;
  Real time_bound_all;

  #ifdef GRAVITY
  Real time_potential_min;
  Real time_potential_max;
  Real time_potential_mean;
  Real time_potential_all;

  Real time_dt_min;
  Real time_dt_max;
  Real time_dt_mean;
  Real time_dt_all;

  Real time_bound_pot_min;
  Real time_bound_pot_max;
  Real time_bound_pot_mean;
  Real time_bound_pot_all;

  #endif//GRAVITY

  #ifdef PARTICLES

  Real time_part_dens_min;
  Real time_part_dens_max;
  Real time_part_dens_mean;
  Real time_part_dens_all;

  Real time_part_dens_transf_min;
  Real time_part_dens_transf_max;
  Real time_part_dens_transf_mean;
  Real time_part_dens_transf_all;

  Real time_part_tranf_min;
  Real time_part_tranf_max;
  Real time_part_tranf_mean;
  Real time_part_tranf_all;


  Real time_advance_particles_1_min;
  Real time_advance_particles_1_max;
  Real time_advance_particles_1_mean;
  Real time_advance_particles_1_all;

  Real time_advance_particles_2_min;
  Real time_advance_particles_2_max;
  Real time_advance_particles_2_mean;
  Real time_advance_particles_2_all;
  #endif

  #ifdef COOLING_GRACKLE
  Real time_cooling_min;
  Real time_cooling_max;
  Real time_cooling_mean;
  Real time_cooling_all;
  #endif
  
  #ifdef CHEMISTRY_GPU
  Real time_start_chemistry;
  Real time_end_chemistry;
  Real time_chemistry_min;
  Real time_chemistry_max;
  Real time_chemistry_mean;
  Real time_chemistry_all;
  #endif

  Time();
  void Initialize();
  void Start_Timer();
  void End_and_Record_Time( int time_var );
  void Print_Times();
  void Get_Average_Times();
  void Print_Average_Times( struct parameters P );
  void Substract_Time_From_Timer( Real time_to_substract );
  
  #ifdef COOLING_GRACKLE
  void Print_Cooling_Time();
  #endif

  #ifdef CHEMISTRY_GPU
  void Record_Time_Chemistry( Real time_chemistry );
  #endif
  
};


#endif
#endif






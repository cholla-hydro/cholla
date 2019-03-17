
#ifdef CPU_TIME

#ifndef TIMING_FUNCTIONS_H
#define TIMING_FUNCTIONS_H

#include "global.h"


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

  #ifdef GRAVITY_COUPLE_CPU
  Real time_dt_min;
  Real time_dt_max;
  Real time_dt_mean;
  Real time_dt_all;
  
  Real time_bound_pot_min;
  Real time_bound_pot_max;
  Real time_bound_pot_mean;
  Real time_bound_pot_all;
  #endif//GRAVITY_COUPLE_CPU
  
  #endif//GRAVITY

  Time();
  void Initialize();
  void Start_Timer();
  void End_and_Record_Time( int time_var );
  void Print_Times();
  void Get_Average_Times();
  void Print_Average_Times( struct parameters P );


};


#endif
#endif







#ifdef CPU_TIME

#include "timing_functions.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <string>


using namespace std;

#ifdef MPI_CHOLLA
#include "mpi_routines.h"
#endif


Time::Time( void ){}


void Time::Initialize(){

  n_steps = 0;


  time_hydro_all = 0;
  time_bound_all = 0;

  #ifdef GRAVITY
  time_potential_all = 0;
  #ifdef GRAVITY_COUPLE_CPU
  time_dt_all = 0;
  time_bound_pot_all = 0;
  #endif
  #endif

  chprintf( "\nTiming Functions is ON \n");
}


void Time::Start_Timer(){
  time_start = get_time();
}

void Time::End_and_Record_Time( int time_var ){

  time_end = get_time();
  time = (time_end - time_start)*1000;

  Real t_min, t_max, t_avg;

  #ifdef MPI_CHOLLA
  t_min = ReduceRealMin(time);
  t_max = ReduceRealMax(time);
  t_avg = ReduceRealAvg(time);
  #endif



  if( time_var == 1 ){
    time_hydro_min = t_min;
    time_hydro_max = t_max;
    time_hydro_mean = t_avg;
    if (n_steps > 0) time_hydro_all += t_max;
  }
  if( time_var == 2 ){
    time_bound_min = t_min;
    time_bound_max = t_max;
    time_bound_mean = t_avg;
    if (n_steps > 0) time_bound_all += t_max;
  }
  #ifdef GRAVITY
  if( time_var == 3 ){
    time_potential_min = t_min;
    time_potential_max = t_max;
    time_potential_mean = t_avg;
    if (n_steps > 0) time_potential_all += t_max;
  }

  #ifdef GRAVITY_COUPLE_CPU
  if( time_var == 0 ){
    time_dt_min = t_min;
    time_dt_max = t_max;
    time_dt_mean = t_avg;
    if (n_steps > 0) time_dt_all += t_max;
  }
  if( time_var == 9 ){
    time_bound_pot_min = t_min;
    time_bound_pot_max = t_max;
    time_bound_pot_mean = t_avg;
    if (n_steps > 0) time_bound_pot_all += t_max;
  }
  #endif
  #endif

}


void Time::Print_Times(){
  #ifdef GRAVITY_COUPLE_CPU
  chprintf(" Time Calc dt           min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_dt_min, time_dt_max, time_dt_mean);
  #endif
  chprintf(" Time Hydro             min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_hydro_min, time_hydro_max, time_hydro_mean);
  chprintf(" Time Boundaries        min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_bound_min, time_bound_max, time_bound_mean);
  #ifdef GRAVITY
  chprintf(" Time Grav Potential    min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_potential_min, time_potential_max, time_potential_mean);
  #ifdef GRAVITY_COUPLE_CPU
  chprintf(" Time Pot Boundaries    min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_bound_pot_min, time_bound_pot_max, time_bound_pot_mean);
  #endif
  #endif


}















#endif
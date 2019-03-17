
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

  chprintf( "\nTiming Functions is ON \n\n");


}
















#endif
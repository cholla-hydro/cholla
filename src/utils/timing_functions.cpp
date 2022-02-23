
#ifdef CPU_TIME

#include "../utils/timing_functions.h"
#include "../io/io.h"
#include <iostream>
#include <fstream>
#include <string>

#ifdef MPI_CHOLLA
#include "../mpi/mpi_routines.h"
#endif

void OneTime::Start(){
  if (inactive) return;
  time_start = get_time();
}

void OneTime::Subtract(Real time_to_subtract){
  // Add the time_to_substract to the start time, that way the time_end - time_start is reduced by time_to_substract
  time_start += time_to_subtract;
}

void OneTime::End(){
  if (inactive) return;
  Real time_end = get_time();
  Real time = (time_end - time_start)*1000;

#ifdef MPI_CHOLLA
  t_min = ReduceRealMin(time);
  t_max = ReduceRealMax(time);
  t_avg = ReduceRealAvg(time);
#else
  t_min = time;
  t_max = time;
  t_avg = time;
#endif
  if (n_steps > 0) t_all += t_max;
  n_steps++;
}

void OneTime::PrintStep(){
  chprintf(" Time %-19s min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", name, t_min, t_max, t_avg);
}

void OneTime::PrintAverage(){
  if (n_steps > 1) chprintf(" Time %-19s avg: %9.4f   ms\n", name, t_all/(n_steps-1));
}

void OneTime::PrintAll(){
  chprintf(" Time %-19s all: %9.4f   ms\n", name, t_all);
}

Time::Time( void ){}

void Time::Initialize(){

  n_steps = 0;

  // Add or remove timers by editing this list.
  // add NAME to timing_functions.h
  // add Timer.NAME.Start() and Timer.NAME.End() where appropriate.
  
  onetimes = {
    #ifdef PARTICLES
    &(Calc_dt = OneTime("Calc_dt")),
    #endif
    &(Hydro = OneTime("Hydro")),
    &(Boundaries = OneTime("Boundaries")),
    #ifdef GRAVITY
    &(Grav_Potential = OneTime("Grav_Potential")),
    &(Pot_Boundaries = OneTime("Pot_Boundaries")),
    #endif
    #ifdef PARTICLES
    &(Part_Density = OneTime("Part_Density")),
    &(Part_Boundaries = OneTime("Part_Boundaries")),
    &(Part_Dens_Transf = OneTime("Part_Dens_Transf")),
    &(Advance_Part_1 = OneTime("Advance_Part_1")),
    &(Advance_Part_2 = OneTime("Advance_Part_2")),
    #endif
    #ifdef COOLING_GRACKLE
    &(Cooling = OneTime("Cooling")),
    #endif
    #ifdef CHEMISTRY_GPU
    &(Chemistry = OneTime("Chemistry"));
    #endif
  };
  

  chprintf( "\nTiming Functions is ON \n");

}

void Time::Print_Times(){
  for (OneTime* x : onetimes){
    x->PrintStep();
  }
}

// once at end of run in main.cpp
void Time::Print_Average_Times( struct parameters P ){

  Real time_total=0;

  chprintf("\nAverage Times      n_steps:%d\n", n_steps);

  for (OneTime* x : onetimes){
    time_total += x->t_all;
    x->PrintAverage();
  }
  time_total /= n_steps;

  chprintf(" Time %-19s avg: %9.4f   ms\n", "Total", time_total);


  std::string file_name ( "run_timing.log" );
  std::string header;

  chprintf( "Writing timing values to file: %s  \n", file_name.c_str());

  header = "#n_proc  nx  ny  nz  n_omp  n_steps  ";

  for (OneTime* x : onetimes){
    header += x->name;
    header += "  ";
  }

  header += "total  ";
  header += " \n";

  bool file_exists = false;
  if (FILE *file = fopen(file_name.c_str(), "r")){
    file_exists = true;
    chprintf( " File exists, appending values: %s \n", file_name.c_str() );
    fclose( file );
  } else{
    chprintf( " Creating File: %s \n", file_name.c_str() );
  }

  #ifdef MPI_CHOLLA
  if ( procID != 0 ) return;
  #endif

  std::ofstream out_file;

// Output timing values
  out_file.open(file_name.c_str(), std::ios::app);
  if ( !file_exists ) out_file << header;
  #ifdef MPI_CHOLLA
  out_file << nproc << " ";
  #else
  out_file << 1 << " ";
  #endif
  out_file << P.nx << " " << P.ny << " " << P.nz << " ";
  #ifdef PARALLEL_OMP
  out_file << N_OMP_THREADS << " ";
  #else
  out_file << 0 << " ";
  #endif
  out_file << n_steps << " ";

  for (OneTime* x : onetimes){
    out_file << x->t_all << " ";
  }

  out_file << time_total << " ";

  out_file << "\n";
  out_file.close();

  chprintf( "Saved Timing: %s \n\n", file_name.c_str() );

}

#endif

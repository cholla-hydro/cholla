
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
  
  #ifdef PARTICLES
  time_part_dens_all = 0;
  time_part_dens_transf_all = 0;
  time_part_tranf_all = 0;
  time_advance_particles_1_all = 0;
  time_advance_particles_2_all = 0;
  #endif
  
  #ifdef COOLING_GRACKLE
  time_cooling_all = 0;
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
  
  #ifdef PARTICLES
  if( time_var == 4 ){
    time_part_dens_min = t_min;
    time_part_dens_max = t_max;
    time_part_dens_mean = t_avg;
    if (n_steps > 0) time_part_dens_all += t_max;
  }

  if( time_var == 5 ){
    time_part_dens_transf_min = t_min;
    time_part_dens_transf_max = t_max;
    time_part_dens_transf_mean = t_avg;
    if (n_steps > 0) time_part_dens_transf_all += t_max;
  }

  if( time_var == 6 ){
    time_advance_particles_1_min = t_min;
    time_advance_particles_1_max = t_max;
    time_advance_particles_1_mean = t_avg;
    if (n_steps > 0) time_advance_particles_1_all += t_max;
  }

  if( time_var == 7 ){
    time_advance_particles_2_min = t_min;
    time_advance_particles_2_max = t_max;
    time_advance_particles_2_mean = t_avg;
    if (n_steps > 0) time_advance_particles_2_all += t_max;
  }

  if( time_var == 8 ){
    time_part_tranf_min = t_min;
    time_part_tranf_max = t_max;
    time_part_tranf_mean = t_avg;
    if (n_steps > 0) time_part_tranf_all += t_max;
  }
  #endif

  #ifdef COOLING_GRACKLE
  if( time_var == 10 ){
    time_cooling_min = t_min;
    time_cooling_max = t_max;
    time_cooling_mean = t_avg;
    if (n_steps > 0) time_cooling_all += t_max;
  }
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
  #ifdef PARTICLES
  chprintf(" Time Part Density      min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_part_dens_min, time_part_dens_max, time_part_dens_mean);
  chprintf(" Time Part Boundaries   min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_part_tranf_min, time_part_tranf_max, time_part_tranf_mean);
  chprintf(" Time Part Dens Transf  min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_part_dens_transf_min, time_part_dens_transf_max, time_part_dens_transf_mean);
  chprintf(" Time Advance Part 1    min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_advance_particles_1_min, time_advance_particles_1_max, time_advance_particles_1_mean);
  chprintf(" Time Advance Part 2    min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_advance_particles_2_min, time_advance_particles_2_max, time_advance_particles_2_mean);
  #endif
  
  #ifdef COOLING_GRACKLE
  chprintf(" Time Cooling           min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", time_cooling_min, time_cooling_max, time_cooling_mean);
  #endif


}


void Time::Get_Average_Times(){

  n_steps -= 1; //Ignore the first timestep

  time_hydro_all /= n_steps;
  time_bound_all /= n_steps;

  #ifdef GRAVITY_CPU
  time_dt_all /= n_steps;
  time_bound_pot_all /= n_steps;
  #endif

  #ifdef GRAVITY
  time_potential_all /= n_steps;
  #ifdef PARTICLES
  time_part_dens_all  /= n_steps;
  time_part_tranf_all /= n_steps;
  time_part_dens_transf_all /= n_steps;
  time_advance_particles_1_all /= n_steps;
  time_advance_particles_2_all /= n_steps;
  #endif
  #endif

  #ifdef COOLING_GRACKLE
  time_cooling_all /= n_steps;
  #endif

}

void Time::Print_Average_Times( struct parameters P ){

  Real time_total;
  time_total = time_hydro_all + time_bound_all;

  #ifdef GRAVITY_CPU
  time_total += time_dt_all;
  time_total += time_bound_pot_all;
  #endif

  #ifdef GRAVITY
  time_total += time_potential_all;
  #ifdef PARTICLES
  time_total += time_part_dens_all;
  time_total += time_part_tranf_all;
  time_total += time_part_dens_transf_all;
  time_total += time_advance_particles_1_all;
  time_total += time_advance_particles_2_all;
  #endif
  #endif

  #ifdef COOLING_GRACKLE
  time_total += time_cooling_all;
  #endif

  chprintf("\nAverage Times      n_steps:%d\n", n_steps);
  #ifdef GRAVITY_CPU
  chprintf(" Time Calc dt           avg: %9.4f   ms\n", time_dt_all);
  #endif
  chprintf(" Time Hydro             avg: %9.4f   ms\n", time_hydro_all);
  chprintf(" Time Boundaries        avg: %9.4f   ms\n", time_bound_all);
  #ifdef GRAVITY
  chprintf(" Time Grav Potential    avg: %9.4f   ms\n", time_potential_all);
  #ifdef GRAVITY_CPU
  chprintf(" Time Pot Boundaries    avg: %9.4f   ms\n", time_bound_pot_all);
  #endif
  #ifdef PARTICLES
  chprintf(" Time Part Density      avg: %9.4f   ms\n", time_part_dens_all);
  chprintf(" Time Part Boundaries   avg: %9.4f   ms\n", time_part_tranf_all);
  chprintf(" Time Part Dens Transf  avg: %9.4f   ms\n", time_part_dens_transf_all);
  chprintf(" Time Advance Part 1    avg: %9.4f   ms\n", time_advance_particles_1_all);
  chprintf(" Time Advance Part 2    avg: %9.4f   ms\n", time_advance_particles_2_all);
  #endif
  #endif

  #ifdef COOLING_GRACKLE
  chprintf(" Time Cooling           avg: %9.4f   ms\n", time_cooling_all);
  #endif

  chprintf(" Time Total             avg: %9.4f   ms\n\n", time_total);

  string file_name ( "run_timing.log" );
  string header;


  chprintf( "Writing timming values to file: %s  \n", file_name.c_str());

  header = "# nz ny nx n_proc n_omp n_steps ";



  #ifdef GRAVITY_CPU
  header += "dt ";
  #endif
  header += "hydo ";
  header += "bound ";
  #ifdef GRAVITY
  header += "grav_pot ";
  #ifdef GRAVITY_CPU
  header += "pot_bound ";
  #endif
  #endif
  #ifdef PARTICLES
  header += "part_dens ";
  header += "part_bound ";
  header += "part_dens_boud ";
  header += "part_adv_1 ";
  header += "part_adv_2 ";
  #endif
  #ifdef COOLING_GRACKLE
  header += "cool ";
  #endif
  header += "total ";
  header += " \n";



  bool file_exists = false;
  if (FILE *file = fopen(file_name.c_str(), "r")){
    file_exists = true;
    chprintf( " File exists, appending values: %s \n\n", file_name.c_str() );
    fclose( file );
  } else{
    chprintf( " Creating File: %s \n\n", file_name.c_str() );
  }



  // Output timing values
  ofstream out_file;
  if ( procID == 0 ){
    out_file.open(file_name.c_str(), ios::app);
    if ( !file_exists ) out_file << header;
    out_file << P.nz << " " << P.ny << " " << P.nx << " ";
    out_file << nproc << " ";
    #ifdef PARALLEL_OMP
    out_file << N_OMP_THREADS << " ";
    #else
    out_file << 0 << " ";
    #endif
    out_file << n_steps << " ";
    #ifdef GRAVITY_CPU
    out_file << time_dt_all << " ";
    #endif
    out_file << time_hydro_all << " ";
    out_file << time_bound_all << " ";
    #ifdef GRAVITY
    out_file << time_potential_all << " ";
    #ifdef GRAVITY_CPU
    out_file << time_bound_pot_all << " ";
    #endif
    #endif
    #ifdef PARTICLES
    out_file << time_part_dens_all << " ";
    out_file << time_part_tranf_all << " ";
    out_file << time_part_dens_transf_all << " ";
    out_file << time_advance_particles_1_all << " ";
    out_file << time_advance_particles_2_all << " ";
    #endif
    #ifdef COOLING_GRACKLE
    out_file << time_cooling_all << " ";
    #endif
    out_file << time_total << " ";

    out_file << "\n";
    out_file.close();
  }
}














#endif
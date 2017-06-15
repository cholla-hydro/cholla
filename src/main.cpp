/*! \file main.cpp
 *  \brief Program to run the grid code. */

#ifdef MPI_CHOLLA
#include <mpi.h>
#include "mpi_routines.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "global.h"
#include "grid3D.h"
#include "io.h"
#include "error_handling.h"

#define OUTPUT
//#define CPU_TIME

int main(int argc, char *argv[])
{
  // timing variables
  double start_total, stop_total, start_step, stop_step;
  #ifdef CPU_TIME
  double stop_init, init_min, init_max, init_avg;
  double start_bound, stop_bound, bound_min, bound_max, bound_avg;
  double start_hydro, stop_hydro, hydro_min, hydro_max, hydro_avg;
  double init, bound, hydro;
  init = bound = hydro = 0;
  #endif //CPU_TIME

  // start the total time
  start_total = get_time();

  /* Initialize MPI communication */
  #ifdef MPI_CHOLLA
  InitializeChollaMPI(&argc, &argv);
  #endif /*MPI_CHOLLA*/

  // declare Cfl coefficient and initial inverse timestep
  Real C_cfl = 0.4; // CFL coefficient 0 < C_cfl < 0.5 
  Real dti = 0; // inverse time step, 1.0 / dt

  // input parameter variables
  char *param_file;
  struct parameters P;
  int nfile = 0; // number of output files
  Real outtime = 0; // current output time

  // read in command line arguments
  if (argc != 2)
  {
    chprintf("usage: %s <parameter_file>\n", argv[0]);
    chexit(0);
  } else {
    param_file = argv[1];
  }

  // create the grid
  Grid3D G;

  // read in the parameters
  parse_params (param_file, &P);
  // and output to screen
  chprintf ("Parameter values:  nx = %d, ny = %d, nz = %d, tout = %f, init = %s, boundaries = %d %d %d %d %d %d\n", 
    P.nx, P.ny, P.nz, P.tout, P.init, P.xl_bcnd, P.xu_bcnd, P.yl_bcnd, P.yu_bcnd, P.zl_bcnd, P.zu_bcnd);
  chprintf ("Output directory:  %s\n", P.outdir);


  // initialize the grid
  G.Initialize(&P);
  chprintf("Local number of grid cells: %d %d %d %d\n", G.H.nx_real, G.H.ny_real, G.H.nz_real, G.H.n_cells);
  //printf("procID %d Local number of grid cells: %d %d %d %d\n", procID, G.H.nx_real, G.H.ny_real, G.H.nz_real, G.H.n_cells);



  // Set initial conditions and calculate first dt
  chprintf("Setting initial conditions...\n");
  G.Set_Initial_Conditions(P, C_cfl);

//#ifdef MPI_CHOLLA
  //MPI_Finalize();
  //exit(0);
//#endif //MPI_CHOLLA

  // set boundary conditions (assign appropriate values to ghost cells)
  chprintf("Setting boundary conditions...\n");
  G.Set_Boundary_Conditions(P);
  chprintf("Boundary conditions set.\n");  
  // set main variables for Read_Grid inital conditions
  if (strcmp(P.init, "Read_Grid") == 0) {
    outtime += G.H.t;
    nfile = P.nfile;
    dti = G.calc_dti_CPU(C_cfl);
    G.H.dt = 1.0 / dti;
  }  
  chprintf("Dimensions of each cell: dx = %f dy = %f dz = %f\n", G.H.dx, G.H.dy, G.H.dz);
  chprintf("Ratio of specific heats gamma = %f\n",gama);
  chprintf("Nstep = %d  Timestep = %f  Simulation time = %f\n", G.H.n_step, G.H.dt, G.H.t);


  #ifdef OUTPUT
  // write the initial conditions to file
  chprintf("Writing initial conditions to file...\n");
  WriteData(G, P, nfile);
  // add one to the output file count
  nfile++;
  #endif //OUTPUT
  // increment the next output time
  outtime += P.outstep;

  #ifdef CPU_TIME
  stop_init = get_time();
  init = stop_init - start_total;
  #ifdef MPI_CHOLLA
  init_min = ReduceRealMin(init);
  init_max = ReduceRealMax(init);
  init_avg = ReduceRealAvg(init);
  chprintf("Init  min: %9.4f  max: %9.4f  avg: %9.4f\n", init_min, init_max, init_avg);
  #else
  printf("Init %9.4f\n", init);
  #endif //MPI_CHOLLA
  #endif //CPU_TIME


  // Evolve the grid, one timestep at a time
  chprintf("Starting calculations.\n");
  while (G.H.t < P.tout)
  //while (G.H.n_step < 1)
  {
    // get the start time
    start_step = get_time();

    // Add supernovae
    Real sn_dti;
    sn_dti = G.Add_Supernovae();
    dti = fmax(dti, sn_dti);

    // calculate the timestep
    G.set_dt(C_cfl, dti);
    
    if (G.H.t + G.H.dt > outtime) 
    {
      G.H.dt = outtime - G.H.t;
    }


    // Advance the grid by one timestep
    #ifdef CPU_TIME
    start_hydro = get_time();
    #endif //CPU_TIME
    dti = G.Update_Grid();
    #ifdef CPU_TIME
    stop_hydro = get_time();
    hydro = stop_hydro - start_hydro;
    #ifdef MPI_CHOLLA
    hydro_min = ReduceRealMin(hydro);
    hydro_max = ReduceRealMax(hydro);
    hydro_avg = ReduceRealAvg(hydro);
    #endif //MPI_CHOLLA
    #endif //CPU_TIME

    // update the time
    G.H.t += G.H.dt;

    // add one to the timestep count
    G.H.n_step++;

    // set boundary conditions for next time step 
    #ifdef CPU_TIME
    start_bound = get_time();
    #endif //CPU_TIME
    G.Set_Boundary_Conditions(P);
    #ifdef CPU_TIME
    stop_bound = get_time();
    bound = stop_bound - start_bound;
    #ifdef MPI_CHOLLA
    bound_min = ReduceRealMin(bound);
    bound_max = ReduceRealMax(bound);
    bound_avg = ReduceRealAvg(bound);
    #endif //MPI_CHOLLA
    #endif //CPU_TIME

    #ifdef CPU_TIME
    #ifdef MPI_CHOLLA
    chprintf("hydro min: %9.4f  max: %9.4f  avg: %9.4f\n", hydro_min, hydro_max, hydro_avg);
    chprintf("bound min: %9.4f  max: %9.4f  avg: %9.4f\n", bound_min, bound_max, bound_avg);
    #endif //MPI_CHOLLA
    #endif //CPU_TIME


    // get the time to compute the total timestep
    stop_step = get_time();
    stop_total = get_time();
    G.H.t_wall = stop_total-start_total;
    #ifdef MPI_CHOLLA
    G.H.t_wall = ReduceRealMax(G.H.t_wall);
    #endif 
    chprintf("n_step: %d   sim time: %10.7f   sim timestep: %7.4e  timestep time = %9.3f ms   total time = %9.4f s\n", 
      G.H.n_step, G.H.t, G.H.dt, (stop_step-start_step)*1000, G.H.t_wall);

    if (G.H.t == outtime)
    {
      #ifdef OUTPUT
      /*output the grid data*/
      WriteData(G, P, nfile);
      // add one to the output file count
      nfile++;
      #endif //OUTPUT
      // update to the next output time
      outtime += P.outstep;      
    }


    for (int i=0; i<G.H.nx; i++) {
      for (int j=0; j<G.H.ny; j++) {
        for (int k=0; k<G.H.nz; k++) {
          int id = i + j*G.H.nx + k*G.H.nx*G.H.ny;
          if (G.C.density[id] < 0.0 || G.C.density[id] != G.C.density[id]) {
            printf("Failure in cell %d %d %d. Density %e\n", i, j, k, G.C.density[id]);
            MPI_Finalize();
            exit(0);
          }
        }
      }
    }

    

  } /*end loop over timesteps*/

  // free the grid
  G.Reset();

  #ifdef MPI_CHOLLA
  MPI_Finalize();
  #endif /*MPI_CHOLLA*/

  return 0;

}

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

//#define OUTPUT

int main(int argc, char *argv[])
{
  // timing variables
  double start_total, stop_total, start_step, stop_step;

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

  // Set initial conditions and calculate first dt
  chprintf("Setting initial conditions...\n");
  G.Set_Initial_Conditions(P, C_cfl);
  chprintf("Dimensions of each cell: dx = %f dy = %f dz = %f\n", G.H.dx, G.H.dy, G.H.dz);
  chprintf("Ratio of specific heats gamma = %f\n",gama);
  chprintf("Nstep = %d  Timestep = %f  Simulation time = %f\n", G.H.n_step, G.H.dt, G.H.t);
  // set main variables for Read_Grid inital conditions
  if (strcmp(P.init, "Read_Grid") == 0) {
    outtime += G.H.t;
    nfile = P.nfile;
    dti = 1.0 / G.H.dt;
  }

  // set boundary conditions (assign appropriate values to ghost cells)
  chprintf("Setting boundary conditions...\n");
  G.Set_Boundary_Conditions(P);
  chprintf("Boundary conditions set.\n");


  #ifdef OUTPUT
  // write the initial conditions to file
  chprintf("Writing initial conditions to file...\n");
  WriteData(G, P, nfile);
  // add one to the output file count
  nfile++;
  #endif //OUTPUT
  // increment the next output time
  outtime += P.outstep;



  // Evolve the grid, one timestep at a time
  chprintf("Starting caclulations.\n");
  while (G.H.t < P.tout)
  //while (G.H.n_step < 1)
  {
    // get the start time
    start_step = get_time();

    // calculate the timestep
    G.set_dt(C_cfl, dti);
    
    if (G.H.t + G.H.dt > outtime) 
    {
      G.H.dt = outtime - G.H.t;
    }

    // Advance the grid by one timestep
    dti = G.Update_Grid();

    // update the time
    G.H.t += G.H.dt;

    // add one to the timestep count
    G.H.n_step++;

    // set boundary conditions for next time step 
    G.Set_Boundary_Conditions(P);


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

  } /*end loop over timesteps*/

  // free the grid
  G.Reset();

  #ifdef MPI_CHOLLA
  MPI_Finalize();
  #endif /*MPI_CHOLLA*/

  return 0;

}

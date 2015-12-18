/*! \file main.cpp
 *  \brief Program to run the grid code. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "global.h"
#include "grid3D.h"
#include "io.h"
#include "error_handling.h"
#ifdef MPI_CHOLLA
#include <mpi.h>
#include "mpi_routines.h"
#endif

//#define CPU_TIME
#define OUTPUT
//#define TURBULENCE

int main(int argc, char *argv[])
{
  // timing variables
  double start_total, stop_total, start_step, stop_step;
  double stop_init, init_min, init_max, init_avg;
  double start_calct, stop_calct;
  double start_CTU, stop_CTU, CTU_min, CTU_max, CTU_avg;
  double comm_min, comm_max, comm_avg;
  double other_min, other_max, other_avg;
  double start_bound, stop_bound, bound_min, bound_max, bound_avg;
  double init, calc_dt, bound, CTU_t, bound_total, CTU_total;
  init = calc_dt = bound = CTU_t = bound_total = CTU_total = 0;
  #ifdef TURBULENCE
  Real M = 2.0;
  Real force = 0.1 / M;
  #endif

  // start the total time
  start_total = get_time();

  /* Initialize MPI communication */
  #ifdef MPI_CHOLLA
  InitializeChollaMPI(&argc, &argv);
  #endif /*MPI_CHOLLA*/

  // declare variables
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
  //chexit(0);

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

  #ifdef CPU_TIME
  stop_init = get_time();
  init = stop_init - start_total;
  #ifdef MPI_CHOLLA
  init_min = ReduceRealMin(init);
  init_max = ReduceRealMax(init);
  init_avg = ReduceRealAvg(init);  
  chprintf("Init   min: %9.4f  max: %9.4f  avg: %9.4f\n", init_min, init_max, init_avg);
  #else
  printf("Init %9.4f\n", init);
  #endif //MPI_CHOLLA
  #endif //CPU_TIME

  

  


  // Evolve the grid, one timestep at a time
  chprintf("Starting caclulations.\n");
  while (G.H.t < P.tout)
  //while (G.H.n_step < 1)
  {
    // get the start time
    start_step = get_time();

    // calculate the timestep
    #ifdef CPU_TIME
    start_calct = get_time();
    #endif 
    G.set_dt(C_cfl, dti);
    #ifdef CPU_TIME
    stop_calct = get_time();
    calc_dt += stop_calct-start_calct;
    #endif 
    
    if (G.H.t + G.H.dt > outtime) 
    {
      G.H.dt = outtime - G.H.t;
    }
/*
    for (int i=G.H.n_ghost; i<G.H.nx-G.H.n_ghost; i++) {
      for (int j=G.H.n_ghost; j<G.H.ny-G.H.n_ghost; j++) {
        int id1 = i + G.H.nx*j;
        int id2 = j + G.H.nx*i;
        if (G.C.momentum_x[id1] != G.C.momentum_y[id2]) printf("%3d %3d\n", i, j);
      }
    }    
*/
    // Use either the CTU or VL integrator to advance the grid
    #ifdef CPU_TIME
    start_CTU = get_time();
    #endif
    #ifdef CTU
    dti = G.Update_Grid_CTU();
    #endif
    #ifdef VL
    dti = G.Update_Grid_VL();
    #endif
    #ifdef CPU_TIME
    stop_CTU = get_time();
    CTU_t = stop_CTU - start_CTU;
    #ifdef MPI_CHOLLA
    CTU_min = ReduceRealMin(CTU_t);
    CTU_max = ReduceRealMax(CTU_t);
    CTU_avg = ReduceRealAvg(CTU_t);  
    if (G.H.n_step > 0) CTU_total += CTU_avg;
    #else
    if (G.H.n_step > 0) CTU_total += CTU_t;
    #endif //MPI_CHOLLA
    #endif


    for (int i=G.H.n_ghost; i<G.H.nx-G.H.n_ghost; i++) {
      for (int j=G.H.n_ghost; j<G.H.ny-G.H.n_ghost; j++) {
        for (int k=G.H.n_ghost; k<G.H.nz-G.H.n_ghost; k++) {
        int id = i + G.H.nx*j + G.H.nx*G.H.ny*k;
        if (G.C.density[id] != G.C.density[id]) {
          printf("%d %3d %3d\n", i, j, k);
          exit(0);
        }
        }
      }    
    }
  /*
  for (int j=0; j<G.H.ny; j++) {
    printf("%03d ", j);
    for (int i=0; i<G.H.nx; i++) {
      printf("%f ", G.C.density[i + j*G.H.nx]);
    }
    printf("\n");
  }
  */

    // update the time
    G.H.t += G.H.dt;

    // add one to the timestep count
    G.H.n_step++;

    // apply turbulent forcing
    #ifdef TURBULENCE
    if (G.H.t+0.000001 > force) {
      G.Apply_Forcing();
      force += 0.1 / M;
    }
    #endif

    // set boundary conditions for next time step 
    #ifdef CPU_TIME
    start_bound = get_time();
    #endif
    G.Set_Boundary_Conditions(P);
    #ifdef CPU_TIME
    stop_bound = get_time();
    bound = stop_bound - start_bound;
    #ifdef MPI_CHOLLA
    bound_min = ReduceRealMin(bound);
    bound_max = ReduceRealMax(bound);
    bound_avg = ReduceRealAvg(bound);
    comm_min = ReduceRealMin(t_comm);
    comm_max = ReduceRealMax(t_comm);
    comm_avg = ReduceRealAvg(t_comm);
    other_min = ReduceRealMin(t_other);
    other_max = ReduceRealMax(t_other);
    other_avg = ReduceRealAvg(t_other);
    if (G.H.n_step > 1) bound_total += bound_avg;
    #else
    if (G.H.n_step > 1) bound_total += bound;
    #endif //MPI_CHOLLA
    #endif

    #ifdef CPU_TIME
    #ifdef MPI_CHOLLA
    chprintf("CTU    min: %9.4f  max: %9.4f  avg: %9.4f\n", CTU_min, CTU_max, CTU_avg);
    chprintf("bound  min: %9.4f  max: %9.4f  avg: %9.4f\n", bound_min, bound_max, bound_avg);
    chprintf("comm   min: %9.4f  max: %9.4f  avg: %9.4f\n", comm_min, comm_max, comm_avg);
    chprintf("other  min: %9.4f  max: %9.4f  avg: %9.4f\n", other_min, other_max, other_avg);
    chprintf("bound total : %9.4f  CTU total : %9.4f\n", bound_total, CTU_total);
    #else
    printf("CTU    %9.4f\n", CTU_t);
    printf("bound  %9.4f\n", bound);
    #endif //MPI_CHOLLA
    #endif

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

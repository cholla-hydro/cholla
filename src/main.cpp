/*! \file main.cpp
 *  \brief Program to run the grid code. */

#ifdef MPI_CHOLLA
#include <mpi.h>
#include "mpi/mpi_routines.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cluster/supernova.h" // move this stuff into grid3D eventually
#include "global/global.h"
#include "grid/grid3D.h"
#include "io/io.h"
#include "utils/error_handling.h"


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

  Real dti = 0; // inverse time step, 1.0 / dt

  // input parameter variables
  char *param_file;
  struct parameters P;
  int nfile = 0; // number of output files
  Real outtime = 0; // current output time


  // read in command line arguments
  if (argc < 2)
  {
    chprintf("usage: %s <parameter_file>\n", argv[0]);
    chexit(-1);
  } else {
    param_file = argv[1];
  }

  // create the grid
  Grid3D G;

  // read in the parameters
  parse_params (param_file, &P, argc, argv);
  // and output to screen
  chprintf ("Parameter values:  nx = %d, ny = %d, nz = %d, tout = %f, init = %s, boundaries = %d %d %d %d %d %d\n",
    P.nx, P.ny, P.nz, P.tout, P.init, P.xl_bcnd, P.xu_bcnd, P.yl_bcnd, P.yu_bcnd, P.zl_bcnd, P.zu_bcnd);
  if (strcmp(P.init, "Read_Grid") == 0  ) chprintf ("Input directory:  %s\n", P.indir);
  chprintf ("Output directory:  %s\n", P.outdir);

  //Create a Log file to output run-time messages
  Create_Log_File(P);

  // initialize the grid
  G.Initialize(&P);
  chprintf("Local number of grid cells: %d %d %d %d\n", G.H.nx_real, G.H.ny_real, G.H.nz_real, G.H.n_cells);

  char *message = (char*)malloc(50 * sizeof(char));
  sprintf(message, "Initializing Simulation" );
  Write_Message_To_Log_File( message );

  // Set initial conditions and calculate first dt
  chprintf("Setting initial conditions...\n");
  G.Set_Initial_Conditions(P);
  chprintf("Initial conditions set.\n");
  // set main variables for Read_Grid initial conditions
  if (strcmp(P.init, "Read_Grid") == 0) {
    dti = C_cfl / G.H.dt;
    outtime += G.H.t;
    nfile = P.nfile;
  }

  #ifdef DE
  chprintf("\nUsing Dual Energy Formalism:\n eta_1: %0.3f   eta_2: %0.4f\n", DE_ETA_1, DE_ETA_2 );
  sprintf(message, " eta_1: %0.3f   eta_2: %0.3f  ", DE_ETA_1, DE_ETA_2 );
  Write_Message_To_Log_File( message );
  #endif


  #ifdef CPU_TIME
  G.Timer.Initialize();
  #endif

  #ifdef GRAVITY
  G.Initialize_Gravity(&P);
  #endif

  #ifdef PARTICLES
  G.Initialize_Particles(&P);
  #endif

  #ifdef COSMOLOGY
  G.Initialize_Cosmology(&P);
  #endif

  #ifdef COOLING_GRACKLE
  G.Initialize_Grackle(&P);
  #endif

  #ifdef ANALYSIS
  G.Initialize_Analysis_Module(&P);
  if ( G.Analysis.Output_Now ) G.Compute_and_Output_Analysis(&P);
  #endif

  #ifdef GRAVITY
  // Get the gravitational potential for the first timestep
  G.Compute_Gravitational_Potential( &P);
  #endif

  // Set boundary conditions (assign appropriate values to ghost cells) for hydro and potential
  chprintf("Setting boundary conditions...\n");
  G.Set_Boundary_Conditions_Grid(P);
  chprintf("Boundary conditions set.\n");

  #ifdef GRAVITY_ANALYTIC_COMP
  // add analytic component to gravity potential.
  G.Add_Analytic_Potential(&P);
  #endif

  #ifdef PARTICLES
  // Get the particles acceleration for the first timestep
  G.Get_Particles_Acceleration();
  #endif

  #ifdef SUPERNOVA
  Supernova::Initialize(G, &P);
  #endif

  chprintf("Dimensions of each cell: dx = %f dy = %f dz = %f\n", G.H.dx, G.H.dy, G.H.dz);
  chprintf("Ratio of specific heats gamma = %f\n",gama);
  chprintf("Nstep = %d  Timestep = %f  Simulation time = %f\n", G.H.n_step, G.H.dt, G.H.t);


  #ifdef OUTPUT
  if (strcmp(P.init, "Read_Grid") != 0 || G.H.Output_Now ) {
    // write the initial conditions to file
    chprintf("Writing initial conditions to file...\n");
    #ifdef HYDRO_GPU
    cudaMemcpy(G.C.density, G.C.device,
             G.H.n_fields*G.H.n_cells*sizeof(Real), cudaMemcpyDeviceToHost);
    #endif
    WriteData(G, P, nfile);
  }
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
  sprintf(message, "Starting calculations." );
  Write_Message_To_Log_File( message );
  while (G.H.t < P.tout)
  {
    // get the start time
    start_step = get_time();

    /*
    printf("Supernova Update_Grid Finished\n");
    fflush(stdout);

    */
    // calculate the timestep
    G.set_dt(dti);
    /*
    printf("G.set_dt Finished\n");
    fflush(stdout);
    MPI_Barrier(world);
    */
    if (G.H.t + G.H.dt > outtime) G.H.dt = outtime - G.H.t;

    #ifdef SUPERNOVA
    Supernova::Update_Grid(G, dti);
    #endif



    #ifdef PARTICLES
    //Advance the particles KDK( first step ): Velocities are updated by 0.5*dt and positions are updated by dt
    G.Advance_Particles( 1 );
    //Transfer the particles that moved outside the local domain
    G.Transfer_Particles_Boundaries(P);
    #endif

    // Advance the grid by one timestep
    dti = G.Update_Hydro_Grid();

    // update the simulation time ( t += dt )
    G.Update_Time();


    #ifdef GRAVITY
    //Compute Gravitational potential for next step
    G.Compute_Gravitational_Potential( &P);
    #endif

    // add one to the timestep count
    G.H.n_step++;

    //Set the Grid boundary conditions for next time step
    G.Set_Boundary_Conditions_Grid(P);

    #ifdef GRAVITY_ANALYTIC_COMP
    // add analytic component to gravity potential.
    G.Add_Analytic_Potential(&P);
    #endif

    #ifdef PARTICLES
    ///Advance the particles KDK( second step ): Velocities are updated by 0.5*dt using the Accelerations at the new positions
    G.Advance_Particles( 2 );
    #endif

    #ifdef PARTICLE_AGE
    //G.Cluster_Feedback();
    #endif

    #ifdef CPU_TIME
    G.Timer.Print_Times();
    #endif

    // get the time to compute the total timestep
    stop_step = get_time();
    stop_total = get_time();
    G.H.t_wall = stop_total-start_total;
    #ifdef MPI_CHOLLA
    G.H.t_wall = ReduceRealMax(G.H.t_wall);
    #endif
    chprintf("n_step: %d   sim time: %10.7f   sim timestep: %7.4e  timestep time = %9.3f ms   total time = %9.4f s\n\n",
      G.H.n_step, G.H.t, G.H.dt, (stop_step-start_step)*1000, G.H.t_wall);

    #ifdef OUTPUT_ALWAYS
    G.H.Output_Now = true;
    #endif

    #ifdef ANALYSIS
    if ( G.Analysis.Output_Now ) G.Compute_and_Output_Analysis(&P);
    #endif

    // if ( P.n_steps_output > 0 && G.H.n_step % P.n_steps_output == 0) G.H.Output_Now = true;

    if (G.H.t == outtime || G.H.Output_Now )
    {
      #ifdef OUTPUT
      /*output the grid data*/
      #ifdef HYDRO_GPU
      cudaMemcpy(G.C.density, G.C.device,
                 G.H.n_fields*G.H.n_cells*sizeof(Real), cudaMemcpyDeviceToHost);
      #endif
      WriteData(G, P, nfile);
      // add one to the output file count
      nfile++;
      #endif //OUTPUT
      // update to the next output time
      outtime += P.outstep;
    }

    #ifdef CPU_TIME
    G.Timer.n_steps += 1;
    #endif

    #ifdef N_STEPS_LIMIT
    // Exit the loop when reached the limit number of steps (optional)
    if ( G.H.n_step == N_STEPS_LIMIT) {
      #ifdef HYDRO_GPU
      cudaMemcpy(G.C.density, G.C.device,
                 G.H.n_fields*G.H.n_cells*sizeof(Real), cudaMemcpyDeviceToHost);
      #endif
      WriteData(G, P, nfile);
      break;
    }
    #endif


    #ifdef COSMOLOGY
    // Exit the loop when reached the last scale_factor output
    if ( G.Cosmo.exit_now ) {
      chprintf( "\nReached Last Cosmological Output: Ending Simulation\n");
      break;
    }
    #endif

  } /*end loop over timesteps*/


  #ifdef CPU_TIME
  // Print timing statistics
  G.Timer.Get_Average_Times();
  G.Timer.Print_Average_Times( P );
  #endif

  sprintf(message, "Simulation completed successfully." );
  Write_Message_To_Log_File( message );

  // free the grid
  G.Reset();

  #ifdef MPI_CHOLLA
  MPI_Finalize();
  #endif /*MPI_CHOLLA*/

  return 0;

}

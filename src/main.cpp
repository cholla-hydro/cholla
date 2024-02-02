/*! \file main.cpp
 *  \brief Program to run the grid code. */

#ifdef MPI_CHOLLA
  #include <mpi.h>

  #include "mpi/mpi_routines.h"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "global/global.h"
#include "grid/grid3D.h"
#include "io/io.h"
#include "utils/cuda_utilities.h"
#include "utils/error_handling.h"

#ifdef SUPERNOVA
  #include "particles/supernova.h"
  #ifdef ANALYSIS
    #include "analysis/feedback_analysis.h"
  #endif
#endif  // SUPERNOVA
#ifdef STAR_FORMATION
  #include "particles/star_formation.h"
#endif
#ifdef MHD
  #include "mhd/magnetic_divergence.h"
#endif  // MHD

#include "grid/grid_enum.h"

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
#endif  // CPU_TIME

  // start the total time
  start_total = Get_Time();

#ifdef MPI_CHOLLA
  /* Initialize MPI communication */
  InitializeChollaMPI(&argc, &argv);
#else
  // Initialize subset of global parallelism variables usually managed by MPI
  Init_Global_Parallel_Vars_No_MPI();
#endif /*MPI_CHOLLA*/

  Real dti = 0;  // inverse time step, 1.0 / dt

  // input parameter variables
  char *param_file;
  struct Parameters P;
  int nfile    = 0;  // number of output files
  Real outtime = 0;  // current output time

  // read in command line arguments
  if (argc < 2) {
    chprintf("usage: %s <parameter_file>\n", argv[0]);
    chprintf("Git Commit Hash = %s\n", GIT_HASH);
    chprintf("Macro Flags     = %s\n", MACRO_FLAGS);
    chexit(-1);
  } else {
    param_file = argv[1];
  }

  // create the grid
  Grid3D G;

  // read in the parameters
  Parse_Params(param_file, &P, argc, argv);
  // and output to screen
  chprintf("Git Commit Hash = %s\n", GIT_HASH);
  chprintf("Macro Flags     = %s\n", MACRO_FLAGS);
  chprintf(
      "Parameter values:  nx = %d, ny = %d, nz = %d, tout = %f, init = %s, "
      "boundaries = %d %d %d %d %d %d\n",
      P.nx, P.ny, P.nz, P.tout, P.init, P.xl_bcnd, P.xu_bcnd, P.yl_bcnd, P.yu_bcnd, P.zl_bcnd, P.zu_bcnd);

  bool is_restart = false;
  if (strcmp(P.init, "Read_Grid") == 0) {
    is_restart = true;
  }
  if (strcmp(P.init, "Read_Grid_Cat") == 0) {
    is_restart = true;
  }

  if (is_restart) {
    chprintf("Input directory:  %s\n", P.indir);
  }
  chprintf("Output directory:  %s\n", P.outdir);

  // Check the configuration
  Check_Configuration(P);

  // Create a Log file to output run-time messages and output the git hash and
  // macro flags used
  Create_Log_File(P);
  std::string message = "Git Commit Hash = " + std::string(GIT_HASH);
  Write_Message_To_Log_File(message.c_str());
  message = "Macro Flags     = " + std::string(MACRO_FLAGS);
  Write_Message_To_Log_File(message.c_str());

  // initialize the grid
  G.Initialize(&P);
  chprintf("Local number of grid cells: %d %d %d %d\n", G.H.nx_real, G.H.ny_real, G.H.nz_real, G.H.n_cells);

  message = "Initializing Simulation";
  Write_Message_To_Log_File(message.c_str());

  // Set initial conditions
  chprintf("Setting initial conditions...\n");
  G.Set_Initial_Conditions(P);
  chprintf("Initial conditions set.\n");
  // set main variables for Read_Grid and Read_Grid_Cat initial conditions
  if (is_restart) {
    outtime += G.H.t;
    nfile = P.nfile;
  }

#ifdef DE
  chprintf("\nUsing Dual Energy Formalism:\n eta_1: %0.3f   eta_2: %0.4f\n", DE_ETA_1, DE_ETA_2);
  message = " eta_1: " + std::to_string(DE_ETA_1) + "   eta_2: " + std::to_string(DE_ETA_2);
  Write_Message_To_Log_File(message.c_str());
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

#ifdef CHEMISTRY_GPU
  G.Initialize_Chemistry(&P);
#endif

#ifdef ANALYSIS
  G.Initialize_AnalysisModule(&P);
  if (G.Analysis.Output_Now) {
    G.Compute_and_Output_Analysis(&P);
  }
#endif

#if defined(SUPERNOVA) && defined(PARTICLE_AGE)
  FeedbackAnalysis sn_analysis(G);
  #ifdef MPI_CHOLLA
  supernova::initState(&P, G.Particles.n_total_initial);
  #else
  supernova::initState(&P, G.Particles.n_local);
  #endif  // MPI_CHOLLA
#endif    // SUPERNOVA && PARTICLE_AGE

#ifdef STAR_FORMATION
  star_formation::Initialize(G);
#endif

#ifdef GRAVITY_ANALYTIC_COMP
  G.Setup_Analytic_Potential(&P);
#endif

#ifdef GRAVITY
  // Get the gravitational potential for the first timestep
  G.Compute_Gravitational_Potential(&P);
#endif

  // Set boundary conditions (assign appropriate values to ghost cells) for
  // hydro and potential
  chprintf("Setting boundary conditions...\n");
  G.Set_Boundary_Conditions_Grid(P);
  chprintf("Boundary conditions set.\n");

#ifdef GRAVITY_ANALYTIC_COMP
  G.Add_Analytic_Potential();
#endif

#ifdef PARTICLES
  // Get the particles acceleration for the first timestep
  G.Get_Particles_Acceleration();
#endif

  chprintf("Dimensions of each cell: dx = %f dy = %f dz = %f\n", G.H.dx, G.H.dy, G.H.dz);
  chprintf("Ratio of specific heats gamma = %f\n", gama);
  chprintf("Nstep = %d  Simulation time = %f\n", G.H.n_step, G.H.t);

#ifdef OUTPUT
  if (!is_restart || G.H.Output_Now) {
    // write the initial conditions to file
    chprintf("Writing initial conditions to file...\n");
    Write_Data(G, P, nfile);
  }
  // add one to the output file count
  nfile++;
#endif  // OUTPUT

#ifdef MHD
  // Check that the initial magnetic field has zero divergence
  mhd::checkMagneticDivergence(G);
#endif  // MHD

  // increment the next output time
  outtime += P.outstep;

#ifdef CPU_TIME
  stop_init = Get_Time();
  init      = stop_init - start_total;
  #ifdef MPI_CHOLLA
  init_min = ReduceRealMin(init);
  init_max = ReduceRealMax(init);
  init_avg = ReduceRealAvg(init);
  chprintf("Init  min: %9.4f  max: %9.4f  avg: %9.4f\n", init_min, init_max, init_avg);
  #else
  printf("Init %9.4f\n", init);
  #endif  // MPI_CHOLLA
#endif    // CPU_TIME

  // Evolve the grid, one timestep at a time
  chprintf("Starting calculations.\n");
  message = "Starting calculations.";
  Write_Message_To_Log_File(message.c_str());

  // Compute inverse timestep for the first time
  dti = G.Calc_Inverse_Timestep();

  while (G.H.t < P.tout) {
// get the start time
#ifdef CPU_TIME
    G.Timer.Total.Start();
#endif  // CPU_TIME
    start_step = Get_Time();

    // calculate the timestep by calling MPI_Allreduce
    G.set_dt(dti);

    // adjust timestep based on the next available scheduled time
    const Real next_scheduled_time = fmin(outtime, P.tout);
    if (G.H.t + G.H.dt > next_scheduled_time) {
      G.H.dt = next_scheduled_time - G.H.t;
    }

#if defined(SUPERNOVA) && defined(PARTICLE_AGE)
    supernova::Cluster_Feedback(G, sn_analysis);
#endif  // SUPERNOVA && PARTICLE_AGE

#ifdef PARTICLES
    // Advance the particles KDK( first step ): Velocities are updated by 0.5*dt
    // and positions are updated by dt
    G.Advance_Particles(1);
    // Transfer the particles that moved outside the local domain
    G.Transfer_Particles_Boundaries(P);
#endif

    // Advance the grid by one timestep
    dti = G.Update_Hydro_Grid();

    // update the simulation time ( t += dt )
    G.Update_Time();

#ifdef GRAVITY
    // Compute Gravitational potential for next step
    G.Compute_Gravitational_Potential(&P);
#endif

    // add one to the timestep count
    G.H.n_step++;

    // Set the Grid boundary conditions for next time step
    G.Set_Boundary_Conditions_Grid(P);

#ifdef GRAVITY_ANALYTIC_COMP
    G.Add_Analytic_Potential();
#endif

#ifdef PARTICLES
    /// Advance the particles KDK( second step ): Velocities are updated by
    /// 0.5*dt using the Accelerations at the new positions
    G.Advance_Particles(2);
#endif

#ifdef STAR_FORMATION
    star_formation::Star_Formation(G);
#endif

#ifdef CPU_TIME
    cuda_utilities::Print_GPU_Memory_Usage();
    G.Timer.Total.End();
#endif  // CPU_TIME

#ifdef CPU_TIME
    G.Timer.Print_Times();
#endif

    // get the time to compute the total timestep
    stop_step  = Get_Time();
    stop_total = Get_Time();
    G.H.t_wall = stop_total - start_total;
#ifdef MPI_CHOLLA
    G.H.t_wall = ReduceRealMax(G.H.t_wall);
#endif
    chprintf(
        "n_step: %d   sim time: %10.7f   sim timestep: %7.4e  timestep time = "
        "%9.3f ms   total time = %9.4f s\n\n",
        G.H.n_step, G.H.t, G.H.dt, (stop_step - start_step) * 1000, G.H.t_wall);

    if (P.output_always) G.H.Output_Now = true;

#ifdef ANALYSIS
    if (G.Analysis.Output_Now) {
      G.Compute_and_Output_Analysis(&P);
    }
  #if defined(SUPERNOVA) && defined(PARTICLE_AGE)
    sn_analysis.Compute_Gas_Velocity_Dispersion(G);
  #endif
#endif

    // if ( P.n_steps_output > 0 && G.H.n_step % P.n_steps_output == 0)
    // G.H.Output_Now = true;

    if (G.H.t == outtime || G.H.Output_Now) {
#ifdef OUTPUT
      /*output the grid data*/
      Write_Data(G, P, nfile);
      // add one to the output file count
      nfile++;
#endif  // OUTPUT
      if (G.H.t == outtime) {
        outtime += P.outstep;  // update to the next output time
      }
    }

#ifdef CPU_TIME
    G.Timer.n_steps += 1;
#endif

#ifdef N_STEPS_LIMIT
    // Exit the loop when reached the limit number of steps (optional)
    if (G.H.n_step == N_STEPS_LIMIT) {
  #ifdef OUTPUT
      Write_Data(G, P, nfile);
  #endif  // OUTPUT
      break;
    }
#endif

#ifdef COSMOLOGY
    // Exit the loop when reached the last scale_factor output
    if (G.Cosmo.exit_now) {
      chprintf("\nReached Last Cosmological Output: Ending Simulation\n");
      break;
    }
#endif

#ifdef MHD
    // Check that the magnetic field has zero divergence
    mhd::checkMagneticDivergence(G);
#endif  // MHD
  }     /*end loop over timesteps*/

#ifdef CPU_TIME
  // Print timing statistics
  G.Timer.Print_Average_Times(P);
#endif

  message = "Simulation completed successfully.";
  Write_Message_To_Log_File(message.c_str());

  // free the grid
  G.Reset();

#ifdef MPI_CHOLLA
  MPI_Finalize();
#endif /*MPI_CHOLLA*/

  return 0;
}

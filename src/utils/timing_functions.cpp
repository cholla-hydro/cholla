#include "../utils/timing_functions.h"
#ifdef CPU_TIME

  #include <algorithm>
  #include <fstream>
  #include <iostream>
  #include <string>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../io/io.h"

  #ifdef MPI_CHOLLA
    #include "../mpi/mpi_routines.h"
  #endif

void OneTime::Start()
{
  cudaDeviceSynchronize();
  if (inactive) {
    return;
  }
  time_start = Get_Time();
}

void OneTime::Subtract(Real time_to_subtract)
{
  // Add the time_to_substract to the start time, that way the time_end -
  // time_start is reduced by time_to_substract
  time_start += time_to_subtract;
}

void OneTime::End(bool const print_high_values)
{
  cudaDeviceSynchronize();
  if (inactive) {
    return;
  }
  Real time_end = Get_Time();
  Real time     = (time_end - time_start) * 1000;

  #ifdef MPI_CHOLLA
  t_min = ReduceRealMin(time);
  t_max = ReduceRealMax(time);
  t_avg = ReduceRealAvg(time);
  #else
  t_min = time;
  t_max = time;
  t_avg = time;
  #endif
  if (n_steps > 0) {
    t_all += t_max;
  }

  #ifdef MPI_CHOLLA
  // Print out information if the process is unusually slow
  if ((time >= 1.1 * t_avg) and (n_steps > 0) and print_high_values) {
    // Get node ID
    std::string node_id(MPI_MAX_PROCESSOR_NAME, ' ');
    int length;
    MPI_Get_processor_name(node_id.data(), &length);
    node_id.resize(length);

    // Get GPU ID
    std::string gpu_id(MPI_MAX_PROCESSOR_NAME, ' ');
    int device;
    GPU_Error_Check(cudaGetDevice(&device));
    GPU_Error_Check(cudaDeviceGetPCIBusId(gpu_id.data(), gpu_id.size(), device));
    gpu_id.erase(
        std::find_if(gpu_id.rbegin(), gpu_id.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
        gpu_id.end());

    std::cerr << "WARNING: Rank took longer than expected to execute." << std::endl
              << "         Node Time: " << time << std::endl
              << "         Avg Time: " << t_avg << std::endl
              << "         Node ID: " << node_id << std::endl
              << "         GPU PCI Bus ID: " << gpu_id << std::endl;
  }
  #endif  // MPI_CHOLLA

  // Increment n_steps
  n_steps++;
}

void OneTime::RecordTime(Real time)
{
  time *= 1000;  // Convert from secs to ms
  #ifdef MPI_CHOLLA
  t_min = ReduceRealMin(time);
  t_max = ReduceRealMax(time);
  t_avg = ReduceRealAvg(time);
  #else
  t_min = time;
  t_max = time;
  t_avg = time;
  #endif
  if (n_steps > 0) {
    t_all += t_max;
  }
  n_steps++;
}

void OneTime::PrintStep()
{
  chprintf(" Time %-19s min: %9.4f  max: %9.4f  avg: %9.4f   ms\n", name, t_min, t_max, t_avg);
}

void OneTime::PrintAverage()
{
  if (n_steps > 1) {
    chprintf(" Time %-19s avg: %9.4f   ms\n", name, t_all / (n_steps - 1));
  }
}

void OneTime::PrintAll() { chprintf(" Time %-19s all: %9.4f   ms\n", name, t_all); }

Time::Time(void) {}

void Time::Initialize()
{
  n_steps = 0;

  // Add or remove timers by editing this list, keep TOTAL at the end
  // add NAME to timing_functions.h
  // add Timer.NAME.Start() and Timer.NAME.End() where appropriate.

  onetimes = {
  #ifdef PARTICLES
      &(Calc_dt = OneTime("Calc_dt")),
  #endif
      &(Hydro_Integrator = OneTime("Hydro_Integrator")),
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
  #ifdef COOLING_GPU
      &(Cooling_GPU = OneTime("Cooling_GPU")),
  #endif
  #ifdef COOLING_GRACKLE
      &(Cooling_Grackle = OneTime("Cooling_Grackle")),
  #endif
  #ifdef CHEMISTRY_GPU
      &(Chemistry = OneTime("Chemistry")),
  #endif
  #ifdef FEEDBACK
      &(Feedback = OneTime("Feedback")),
    #ifdef ANALYSIS
      &(FeedbackAnalysis = OneTime("FeedbackAnalysis")),
    #endif
  #endif  // FEEDBACK
      &(Total = OneTime("Total")),
  };

  chprintf("\nTiming Functions is ON \n");
}

void Time::Print_Times()
{
  for (OneTime* x : onetimes) {
    x->PrintStep();
  }
}

// once at end of run in main.cpp
void Time::Print_Average_Times(struct Parameters P)
{
  chprintf("\nAverage Times      n_steps:%d\n", n_steps);

  for (OneTime* x : onetimes) {
    x->PrintAverage();
  }

  std::string file_name("run_timing.log");

  chprintf("Writing timing values to file: %s  \n", file_name.c_str());

  std::string header = "Git Commit Hash = " + std::string(GIT_HASH) + std::string("\n");
  header += "Macro Flags     = " + std::string(MACRO_FLAGS) + std::string("\n");
  header += "Note that the timers all skip the first time step since it always takes longer." + std::string("\n") +
            "To find the average time divide the time shown by n_steps-1" + std::string("\n");

  header += std::string("\n") + "#n_proc  nx  ny  nz  n_omp  n_steps  ";

  for (OneTime* x : onetimes) {
    header += x->name;
    header += "  ";
  }

  header += " \n";

  bool file_exists = false;
  if (FILE* file = fopen(file_name.c_str(), "r")) {
    file_exists = true;
    chprintf(" File exists, appending values: %s \n", file_name.c_str());
    fclose(file);
  } else {
    chprintf(" Creating File: %s \n", file_name.c_str());
  }

  #ifdef MPI_CHOLLA
  if (procID != 0) {
    return;
  }
  #endif

  std::ofstream out_file;

  // Output timing values
  out_file.open(file_name.c_str(), std::ios::app);
  if (!file_exists) {
    out_file << header;
  }
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

  for (OneTime* x : onetimes) {
    out_file << x->t_all << " ";
  }

  out_file << "\n";
  out_file.close();

  chprintf("Saved Timing: %s \n\n", file_name.c_str());
}

#endif  // CPU_TIME

ScopedTimer::ScopedTimer(const char* input_name)
{
#ifdef CPU_TIME
  name       = input_name;
  time_start = Get_Time();
#endif
}

ScopedTimer::~ScopedTimer(void)
{
#ifdef CPU_TIME
  double time_elapsed_ms = (Get_Time() - time_start) * 1000;

  #ifdef MPI_CHOLLA
  double t_min = ReduceRealMin(time_elapsed_ms);
  double t_max = ReduceRealMax(time_elapsed_ms);
  double t_avg = ReduceRealAvg(time_elapsed_ms);
  #else
  double t_min = time_elapsed_ms;
  double t_max = time_elapsed_ms;
  double t_avg = time_elapsed_ms;
  #endif  // MPI_CHOLLA
  chprintf("ScopedTimer Min: %9.4f ms Max: %9.4f ms Avg: %9.4f ms %s \n", t_min, t_max, t_avg, name);
#endif  // CPU_TIME
}

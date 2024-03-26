/*!
 * \file cuda_utilities.cpp
 * \brief Implementation file for cuda_utilities.h
 *
 */
#include "../utils/cuda_utilities.h"

#include <iomanip>
#include <sstream>

#include "../io/io.h"
#include "../mpi/mpi_routines.h"

namespace cuda_utilities
{
void Print_GPU_Memory_Usage(std::string const &additional_text)
{
  // Get the memory usage
  size_t gpu_free_memory, gpu_total_memory;
  GPU_Error_Check(cudaMemGetInfo(&gpu_free_memory, &gpu_total_memory));

  // Assuming that all GPUs in the system have the same amount of memory
  size_t const gpu_used_memory = Reduce_size_t_Max(gpu_total_memory - gpu_free_memory);

  Real const percent_used = 100.0 * (static_cast<Real>(gpu_used_memory) / static_cast<Real>(gpu_total_memory));

  // Prep the message to print
  std::stringstream output_message_stream;
  output_message_stream << std::fixed << std::setprecision(2);
  output_message_stream << "Percentage of GPU memory used: " << percent_used << "%. GPU memory used "
                        << std::to_string(gpu_used_memory) << ", GPU total memory " << std::to_string(gpu_total_memory)
                        << additional_text << std::endl;
  std::string output_message = output_message_stream.str();

  chprintf(output_message.c_str());
}
}  // end namespace cuda_utilities

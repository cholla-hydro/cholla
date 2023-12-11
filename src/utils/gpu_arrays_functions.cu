#include <iostream>

#include "../global/global_cuda.h"
#include "../utils/error_handling.h"
#include "../utils/gpu.hpp"
#include "../utils/gpu_arrays_functions.h"

void Extend_GPU_Array_Real(Real **current_array_d, int current_size, int new_size, bool print_out)
{
  if (new_size <= current_size) {
    return;
  }
  if (print_out) {
    std::cout << " Extending GPU Array, size: " << current_size << "  new_size: " << new_size << std::endl;
  }

  size_t global_free, global_total;
  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
  cudaDeviceSynchronize();
#ifdef PRINT_GPU_MEMORY
  printf("ReAllocating GPU Memory:  %d  MB free \n", (int)global_free / 1000000);
#endif

  if (global_free < new_size * sizeof(Real)) {
    printf("ERROR: Not enough global device memory \n");
    printf(" Available Memory: %d  MB \n", (int)(global_free / 1000000));
    printf(" Requested Memory: %d  MB \n", (int)(new_size * sizeof(Real) / 1000000));
    // exit(-1);
  }

  Real *new_array_d;
  GPU_Error_Check(cudaMalloc((void **)&new_array_d, new_size * sizeof(Real)));
  cudaDeviceSynchronize();
  GPU_Error_Check();
  if (new_array_d == NULL) {
    std::cout << " Error When Allocating New GPU Array" << std::endl;
    chexit(-1);
  }

  // Copy the content of the original array to the new array
  GPU_Error_Check(cudaMemcpy(new_array_d, *current_array_d, current_size * sizeof(Real), cudaMemcpyDeviceToDevice));
  cudaDeviceSynchronize();
  GPU_Error_Check();

  // size_t global_free_before, global_free_after;
  // GPU_Error_Check( cudaMemGetInfo( &global_free_before, &global_total ) );
  // cudaDeviceSynchronize();

  // Free the original array
  cudaFree(*current_array_d);
  cudaDeviceSynchronize();
  GPU_Error_Check();

  // GPU_Error_Check( cudaMemGetInfo( &global_free_after, &global_total ) );
  // cudaDeviceSynchronize();
  //
  // printf("Freed Memory: %d MB\n", (int) (global_free_after -
  // global_free_before)/1000000 );

  // Replace the pointer of the original array with the new one
  *current_array_d = new_array_d;
}

#ifndef GPU_ARRAY_FUNCTIONS_H
#define GPU_ARRAY_FUNCTIONS_H

#include <iostream>

#include "../global/global_cuda.h"
#include "../utils/error_handling.h"
#include "../utils/gpu.hpp"
#include "../utils/gpu_arrays_functions.h"

template <typename T>
void Extend_GPU_Array(T **current_array_d, int current_size, int new_size, bool print_out)
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
  printf("ReAllocating GPU Memory:  %ld  MB free \n", global_free / 1000000);
#endif

  if (global_free < new_size * sizeof(T)) {
    printf("ERROR: Not enough global device memory \n");
    printf(" Available Memory: %ld  MB \n", global_free / 1000000);
    printf(" Requested Memory: %ld  MB \n", new_size * sizeof(T) / 1000000);
    exit(-1);
  }

  T *new_array_d;
  GPU_Error_Check(cudaMalloc((void **)&new_array_d, new_size * sizeof(T)));
  cudaDeviceSynchronize();
  GPU_Error_Check();
  if (new_array_d == NULL) {
    std::cout << " Error When Allocating New GPU Array" << std::endl;
    chexit(-1);
  }

  // Copy the content of the original array to the new array
  GPU_Error_Check(cudaMemcpy(new_array_d, *current_array_d, current_size * sizeof(T), cudaMemcpyDeviceToDevice));
  cudaDeviceSynchronize();
  GPU_Error_Check();

  // Free the original array
  cudaFree(*current_array_d);
  cudaDeviceSynchronize();
  GPU_Error_Check();

  // Replace the pointer of the original array with the new one
  *current_array_d = new_array_d;
}

#endif

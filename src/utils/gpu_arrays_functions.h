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
    std::cout << " procID " << procID << " Extending GPU Array (T), current_size: " << current_size << "  new_size: " << new_size << std::endl;
  }

  /*if (print_out) {
    printf("procID %d About to check memory..\n",procID);
    fflush(stdout);
  }*/


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
  /*if (print_out) {
    printf("procID %d ReAllocating GPU Memory:  %ld  MB free \n", procID, global_free / 1000000);
    fflush(stdout);
  }*/

  T *new_array_d;
  GPU_Error_Check(cudaMalloc((void **)&new_array_d, new_size * sizeof(T)));
  cudaDeviceSynchronize();
  GPU_Error_Check();
  if (new_array_d == NULL) {
    std::cout << " Error When Allocating New GPU Array" << std::endl;
    chexit(-1);
  }

  /*if (print_out) {
    std::cout << "procID " << procID << " New GPU Array buffer successfully allocated." << std::endl;
    fflush(stdout);
  }*/
  // Copy the content of the original array to the new array
  GPU_Error_Check(cudaMemcpy(new_array_d, *current_array_d, current_size * sizeof(T), cudaMemcpyDeviceToDevice));
  cudaDeviceSynchronize();
  GPU_Error_Check();
  /*if (print_out) {
    std::cout << "procID " << procID << " GPU Array buffer successfully copied." << std::endl;
    fflush(stdout);
  }*/

  // Free the original array
  cudaFree(*current_array_d);
  cudaDeviceSynchronize();
  GPU_Error_Check();
  /*if (print_out) {
    std::cout << "procID " << procID << " GPU Array buffer freed." << std::endl;
    fflush(stdout);
  }*/

  // Replace the pointer of the original array with the new one
  *current_array_d = new_array_d; // will this work?
  
/*
  // BRANT changes 6/22/2026
  // reallocate the current array
  GPU_Error_Check(cudaMalloc((void **)&(*current_array_d), new_size * sizeof(T)));
  cudaDeviceSynchronize();
  if (print_out) {
    std::cout << "procID " << procID << " GPU Array buffer re-allocated." << std::endl;
    fflush(stdout);
  }

  // copy from existing buffer array back from new array to current array
  GPU_Error_Check(cudaMemcpy(*current_array_d, new_array_d, current_size * sizeof(T), cudaMemcpyDeviceToDevice));
  cudaDeviceSynchronize();
  GPU_Error_Check();


  if (print_out) {
    std::cout << "procID " << procID << " GPU Array buffer transferred." << std::endl;
    fflush(stdout);
  }

  // free new array
  GPU_Error_Check(cudaFree(new_array_d));
  */

  /*if (print_out) {
    std::cout << "procID " << procID << " GPU array of size  " << new_size << " successfully allocated." << std::endl;
    //std::cout << "procID " << procID << " Exiting Extend_GPU_Array()." << std::endl;
    fflush(stdout);
  }*/
}

#endif

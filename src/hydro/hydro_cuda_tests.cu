/*!
* \file hydro_cuda_tests.cu
* \author Evan Schneider (evs34@pitt.edu)
* \brief Test the code units within hydro_cuda.cu
*
*/

// STL Includes
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"
#include "../utils/testing_utilities.h"
#include "../hydro/hydro_cuda.h"   // Include code to test

#if defined(CUDA)


TEST(tHYDROCalcDt3D, CorrectInputExpectCorrectOutput)
{

  Real* testDt;
  cudaHostAlloc(&testDt, sizeof(Real), cudaHostAllocDefault);

  // Call the function we are testing
  int num_blocks = 1;
  dim3 dim1dGrid(num_blocks, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  int const nx = 1;
  int const ny = 1;
  int const nz = 1;
  int const n_fields   = 5;  // Total number of conserved fields  
  int const  n_ghost = 0;
  Real dx = 1.0;
  Real dy = 1.0;
  Real dz = 1.0;
  Real *host_conserved;
  Real *dev_conserved;
  Real *dev_dti_array;
  Real gamma = 5.0/3.0;
  Real max_dti_slow = 1e10;

  // Allocate host and device arrays and copy data
  cudaHostAlloc(&host_conserved, n_fields*sizeof(Real), cudaHostAllocDefault);
  CudaSafeCall(cudaMalloc(&dev_conserved, n_fields*sizeof(Real)));  
  CudaSafeCall(cudaMalloc(&dev_dti_array, sizeof(Real)));  

  // Set values of conserved variables for input (host)
  host_conserved[0] = 1.0; // density
  host_conserved[1] = 0.0; // x momentum
  host_conserved[2] = 0.0; // y momentum
  host_conserved[3] = 0.0; // z momentum
  host_conserved[4] = 1.0; // Energy

  // Copy host data to device arrray
  CudaSafeCall(cudaMemcpy(dev_conserved, host_conserved, n_fields*sizeof(Real), cudaMemcpyHostToDevice));  

  // Run the kernel
  hipLaunchKernelGGL(Calc_dt_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, dx, dy, dz, dev_dti_array, gamma, max_dti_slow);
  CudaCheckError();

  // Copy the dt value back from the GPU
  CudaSafeCall(cudaMemcpy(testDt, dev_dti_array, sizeof(Real), cudaMemcpyDeviceToHost));  

  // Compare results
  // Check for equality and if not equal return difference
  double fiducialDt = 1.0540925533894598;
  double testData = testDt[0];
  double absoluteDiff;
  int64_t ulpsDiff;
  bool areEqual; 
  areEqual = testingUtilities::nearlyEqualDbl(fiducialDt, testData, absoluteDiff, ulpsDiff);	
  EXPECT_TRUE(areEqual)
    << "The fiducial value is:       " << fiducialDt << std::endl
    << "The test value is:           " << testData     << std::endl
    << "The absolute difference is:  " << absoluteDiff << std::endl
    << "The ULP difference is:       " << ulpsDiff     << std::endl;  
  
}


#endif

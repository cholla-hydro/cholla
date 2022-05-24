/*!
 * \file reduction_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the contents of reduction_utilities.h and reduction_utilities.cpp
 *
 */

// STL Includes
#include <vector>
#include <string>
#include <iostream>
#include <random>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../utils/testing_utilities.h"
#include "../utils/reduction_utilities.h"
#include "../global/global.h"



// =============================================================================
// Tests for divergence max reduction
// =============================================================================
TEST(tALLKernelReduceMax, CorrectInputExpectCorrectOutput)
{
    // Launch parameters
    // =================
    uint numBlocks, threadsPerBlock;
    reduction_utilities::reductionLaunchParams(numBlocks, threadsPerBlock);

    // Grid Parameters & testing parameters
    // ====================================
    size_t const gridSize = 64;
    size_t const size     = std::pow(gridSize, 3);;
    Real   const maxValue = 4;
    std::vector<Real> host_grid(size);
    Real host_max = -DBL_MAX;

    // Fill grid with random values and assign maximum value
    std::mt19937 prng(1);
    std::uniform_real_distribution<double> doubleRand(-std::abs(maxValue)-1, std::abs(maxValue) - 1);
    std::uniform_int_distribution<int> intRand(0, host_grid.size()-1);
    for (size_t i = 0; i < host_grid.size(); i++)
    {
        host_grid.at(i) = doubleRand(prng);
    }
    host_grid.at(intRand(prng)) = maxValue;


    // Allocating and copying to device
    // ================================
    Real *dev_grid;
    CudaSafeCall(cudaMalloc(&dev_grid, host_grid.size() * sizeof(Real)));
    CudaSafeCall(cudaMemcpy(dev_grid, host_grid.data(), host_grid.size() * sizeof(Real), cudaMemcpyHostToDevice));

    Real *dev_max_array;
    CudaSafeCall(cudaMalloc(&dev_max_array, numBlocks*sizeof(Real)));
    cudaMemset(dev_max_array,-DBL_MAX,numBlocks*sizeof(Real));
    
    Real host_max_array[numBlocks];
    //Real *host_max_array = (Real *) malloc(numBlocks*sizeof(Real));
    //CudaSafeCall( cudaHostAlloc(&host_max_array, numBlocks*sizeof(Real), cudaHostAllocDefault) );


    // Do the reduction
    // ================
    hipLaunchKernelGGL(reduction_utilities::kernelReduceMax, numBlocks, threadsPerBlock, 0, 0, dev_grid, dev_max_array, host_grid.size());
    CudaCheckError();


    // Copy back and sync
    // ==================
    CudaSafeCall(cudaMemcpy(&host_max_array, dev_max_array, numBlocks*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    for (int i = 0; i < numBlocks; i++)
    {
        host_max = fmax(host_max,host_max_array[i]);
    }

    //free(host_max_array);

    cudaFree(dev_max_array);

    cudaFree(dev_grid);

    // Perform comparison
    testingUtilities::checkResults(maxValue, host_max, "maximum value found");
}
// =============================================================================
// Tests for divergence max reduction
// =============================================================================

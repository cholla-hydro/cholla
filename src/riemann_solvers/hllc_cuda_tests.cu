/*!
* \file hllc_cuda-tests.cpp
* \author Robert 'Bob' Caddy (rvc@pitt.edu)
* \brief Test the code units within hllc_cuda.cu
*
*/

// STL Includes
#include <iostream>
#include <vector>
#include <string>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"
#include "../utils/testing_utilities.h"
#include "../riemann_solvers/hllc_cuda.h"   // Include code to test


// =============================================================================
#if defined(HYDRO_GPU) \
    && defined(CUDA) \
    && defined(MPI_CHOLLA) \
    && defined(BLOCK) \
    && PRECISION==2 \
    && defined(PPMP) \
    && defined(HLLC) \
// Testing Calculate_HLLC_Fluxes_CUDA
TEST(tHYDROCalculateHLLCFluxesCUDA,  // Test suite name
     LeftSideExpectCorrectOutput)  // Test name
{
    // Physical Values
    Real const density   = 1.0;
    Real const pressure  = 1.0;
    Real const velocityX = 0.0;
    Real const velocityY = 0.0;
    Real const velocityZ = 0.0;
    Real const momentumX = density * velocityX;
    Real const momentumY = density * velocityY;
    Real const momentumZ = density * velocityZ;
    Real const gamma     = 1.4;
    Real const energy    = (pressure/(gamma - 1)) + 0.5 * density * (velocityX*velocityX + velocityY*velocityY + velocityZ*velocityZ);

    // Simulation Paramters
    int const nx        = 1;  // Number of cells in the x-direction?
    int const ny        = 1;  // Number of cells in the y-direction?
    int const nz        = 1;  // Number of cells in the z-direction?
    int const n_ghost   = 0;  // Isn't actually used it appears
    int const direction = 0;  // Which direction, 0=x, 1=y, 2=z
    int const n_fields  = 5;  // Total number of conserved fields

    // Launch Parameters
    dim3 const dimGrid (1,1,1);  // How many blocks in the grid
    dim3 const dimBlock(1,1,1);  // How many threads per block

    // Create arrays like the kernel expects
    Real *conserved = new Real[n_fields] {density, momentumX, momentumY, momentumZ, energy};
    Real *testFlux  = new Real[n_fields];
    Real *dev_conserved;
    Real *dev_testFlux;

    // Fiducial values and field names
    std::vector<Real> const fiducialFlux{0, 1, 0, 0, 0};
    std::vector<std::string> const fieldNames {"Densities",
                                               "X Momentum",
                                               "Y Momentum",
                                               "Z Momentum",
                                               "Energies"};

    CudaSafeCall( cudaMalloc(&dev_conserved, n_fields*sizeof(Real)) );
    CudaSafeCall( cudaMalloc(&dev_testFlux,  n_fields*sizeof(Real)) );
    CudaSafeCall( cudaMemcpy(dev_conserved, conserved, n_fields*sizeof(Real), cudaMemcpyHostToDevice) );
    // Run kernel
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA,
                       dimGrid,
                       dimBlock,
                       0,
                       0,
                       dev_conserved,  // the "left" interface
                       dev_conserved,  // the "right" interface
                       dev_testFlux,
                       nx,
                       ny,
                       nz,
                       n_ghost,
                       gamma,
                       direction,
                       n_fields);

    CudaCheckError();
    CudaSafeCall( cudaMemcpy(testFlux, dev_testFlux, n_fields*sizeof(Real), cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();  // Make sure to sync with the device so we have the results
    CudaCheckError();

    // Check for equality
    for (size_t i = 0; i < n_fields; i++)
    {
        // Check for equality and iff not equal return difference
        double absoluteDiff;
        int64_t ulpsDiff;
        bool areEqual = testingUtilities::nearlyEqualDbl(fiducialFlux[i],
                                                         testFlux[i],
                                                         absoluteDiff,
                                                         ulpsDiff);
        EXPECT_TRUE(areEqual)
            << std::endl
            << "Difference in "                << fieldNames[i]   << std::endl
            << "The fiducial value is:       " << fiducialFlux[i] << std::endl
            << "The test value is:           " << testFlux[i]     << std::endl
            << "The absolute difference is:  " << absoluteDiff    << std::endl
            << "The ULP difference is:       " << ulpsDiff        << std::endl;
    }
}
#endif
// =============================================================================
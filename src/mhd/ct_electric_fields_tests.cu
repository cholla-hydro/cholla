/*!
 * \file ct_electric_fields_tests.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the CT electric fields
 *
 */

// STL Includes
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <cmath>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../utils/testing_utilities.h"
#include "../mhd/ct_electric_fields.h"
#include "../global/global.h"

// =============================================================================
// Tests for the mhd::Calculate_CT_Electric_Fields kernel
// =============================================================================

// =============================================================================
/*!
 * \brief Test fixture for tMHDCalculateCTElectricFields test suite
 *
 */
class tMHDCalculateCTElectricFields : public ::testing::Test
{
public:

    /*!
     * \brief Initialize and allocate all the various required variables and
    * arrays
     *
     */
    tMHDCalculateCTElectricFields()
        :
        nx(3),
        ny(nx),
        nz(nx),
        n_cells(nx*ny*nz),
        fluxX(n_cells * (7+NSCALARS)),
        fluxY(n_cells * (7+NSCALARS)),
        fluxZ(n_cells * (7+NSCALARS)),
        grid (n_cells * (8+NSCALARS)),
        testCTElectricFields(n_cells * 3, -999.),
        fiducialData(n_cells * 3, -999.),
        dimGrid((n_cells + TPB - 1),1,1),
        dimBlock(TPB,1,1)
    {
        // Allocate device arrays
        CudaSafeCall ( cudaMalloc(&dev_fluxX, fluxX.size()*sizeof(double)) );
        CudaSafeCall ( cudaMalloc(&dev_fluxY, fluxY.size()*sizeof(double)) );
        CudaSafeCall ( cudaMalloc(&dev_fluxZ, fluxZ.size()*sizeof(double)) );
        CudaSafeCall ( cudaMalloc(&dev_grid,   grid.size()*sizeof(double)) );
        CudaSafeCall ( cudaMalloc(&dev_testCTElectricFields, testCTElectricFields.size()*sizeof(double)) );

        // Populate the grids with values where vector.at(i) = double(i). The
        // values chosen aren't that important, just that every cell has a unique
        // value
        std::iota(std::begin(fluxX), std::end(fluxX), 0.);
        std::iota(std::begin(fluxY), std::end(fluxY), fluxX.back() + 1);
        std::iota(std::begin(fluxZ), std::end(fluxZ), fluxY.back() + 1);
        std::iota(std::begin(grid),  std::end(grid),  fluxZ.back() + 1);
    }
    ~tMHDCalculateCTElectricFields() = default;
protected:
    // Initialize the test grid and other state variables
    size_t const nx, ny, nz;
    size_t const n_cells;

    // Launch Parameters
    dim3 const dimGrid;  // How many blocks in the grid
    dim3 const dimBlock;  // How many threads per block

    // Make sure the vector is large enough that the locations where the
    // magnetic field would be in the real grid are filled
    std::vector<double> fluxX;
    std::vector<double> fluxY;
    std::vector<double> fluxZ;
    std::vector<double> grid;
    std::vector<double> testCTElectricFields;
    std::vector<double> fiducialData;

    // device pointers
    double *dev_fluxX, *dev_fluxY, *dev_fluxZ, *dev_grid, *dev_testCTElectricFields;

    /*!
     * \brief Launch the kernel and check results
     *
     */
    void runTest()
    {
        // Copy values to GPU
        CudaSafeCall( cudaMemcpy(dev_fluxX, fluxX.data(), fluxX.size()*sizeof(Real), cudaMemcpyHostToDevice) );
        CudaSafeCall( cudaMemcpy(dev_fluxY, fluxY.data(), fluxY.size()*sizeof(Real), cudaMemcpyHostToDevice) );
        CudaSafeCall( cudaMemcpy(dev_fluxZ, fluxZ.data(), fluxZ.size()*sizeof(Real), cudaMemcpyHostToDevice) );
        CudaSafeCall( cudaMemcpy(dev_grid,  grid.data(),   grid.size()*sizeof(Real), cudaMemcpyHostToDevice) );
        CudaSafeCall( cudaMemcpy(dev_testCTElectricFields,
                                 testCTElectricFields.data(),
                                 testCTElectricFields.size()*sizeof(Real),
                                 cudaMemcpyHostToDevice) );

        // Call the kernel to test
        hipLaunchKernelGGL(mhd::Calculate_CT_Electric_Fields,
                           dimGrid,
                           dimBlock,
                           0,
                           0,
                           dev_fluxX,
                           dev_fluxY,
                           dev_fluxZ,
                           dev_grid,
                           dev_testCTElectricFields,
                           nx,
                           ny,
                           nz,
                           n_cells);
        CudaCheckError();

        // Copy test data back
        CudaSafeCall( cudaMemcpy(testCTElectricFields.data(),
                                 dev_testCTElectricFields,
                                 testCTElectricFields.size()*sizeof(Real),
                                 cudaMemcpyDeviceToHost) );
        cudaDeviceSynchronize();

        // Check the results
        for (size_t i = 0; i < fiducialData.size(); i++)
        {
            int xid, yid, zid;
            cuda_utilities::compute3DIndices(i, nx, ny, xid, yid, zid);
            testingUtilities::checkResults(fiducialData.at(i),
                                           testCTElectricFields.at(i),
                                           "value at i = " + std::to_string(i)
                                           + ", xid  = " + std::to_string(xid)
                                           + ", yid  = " + std::to_string(yid)
                                           + ", zid  = " + std::to_string(zid));
        }
    }
};
// =============================================================================

// =============================================================================
TEST_F(tMHDCalculateCTElectricFields,
       PositiveVelocityExpectCorrectOutput)
{
    // Fiducial values
    fiducialData.at(26) =  206.29859653255295;
    fiducialData.at(53) = -334.90052254763339;
    fiducialData.at(80) =  209.53472440298236;

    // Launch kernel and check results
    runTest();
}
// =============================================================================

// =============================================================================
TEST_F(tMHDCalculateCTElectricFields,
       NegativeVelocityExpectCorrectOutput)
{
    // Fiducial values
    fiducialData.at(26) =  203.35149422304994;
    fiducialData.at(53) = -330.9860399765279;
    fiducialData.at(80) =  208.55149905461991;

    // Set the density fluxes to be negative to indicate a negative velocity
    // across the face
    for (size_t i = 0; i < n_cells; i++)
    {
        fluxX.at(i) = -fluxX.at(i);
        fluxY.at(i) = -fluxY.at(i);
        fluxZ.at(i) = -fluxZ.at(i);
    }

    // Launch kernel and check results
    runTest();
}
// =============================================================================

// =============================================================================
TEST_F(tMHDCalculateCTElectricFields,
       ZeroVelocityExpectCorrectOutput)
{
    // Fiducial values
    fiducialData.at(26) =  204.82504537780144;
    fiducialData.at(53) = -332.94328126208063;
    fiducialData.at(80) =  209.04311172880114;

    // Set the density fluxes to be negative to indicate a negative velocity
    // across the face
    for (size_t i = 0; i < n_cells; i++)
    {
        fluxX.at(i) = 0.0;
        fluxY.at(i) = 0.0;
        fluxZ.at(i) = 0.0;
    }

    // Launch kernel and check results
    runTest();
}
// =============================================================================

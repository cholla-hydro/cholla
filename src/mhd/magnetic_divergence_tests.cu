/*!
 * \file magnetic_divergence_tests.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the magnetic divergence code
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
#include "../mhd/magnetic_divergence.h"
#include "../utils/DeviceVector.h"
#include "../global/global.h"

// =============================================================================
// Tests for the magnetic field divergence functions
// =============================================================================
TEST(tMHDLaunchCalculateMagneticDivergence, CorrectInputExpectCorrectOutput)
{
    // Grid Parameters & testing parameters
    size_t const gridSize = 96; // Needs to be at least 64 so that each thread has a value
    size_t const n_ghost  = 4;
    size_t const nx       = gridSize+2*n_ghost, ny = nx, nz = nx;
    size_t const n_cells  = nx*ny*nz;
    size_t const n_fields = 8;
    Real   const dx       = 3, dy = dx, dz = dx;
    std::vector<Real> host_grid(n_cells*n_fields);

    // Fill grid with random values and randomly assign maximum value
    std::mt19937 prng(1);
    std::uniform_real_distribution<double> doubleRand(1, 5);
    for (size_t i = 0; i < host_grid.size(); i++)
    {
        host_grid.at(i) = doubleRand(prng);
    }

    // Allocating and copying to device
    cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
    dev_grid.cpyHostToDevice(host_grid);

    // Get test data
    Real testDivergence = mhd::launchCalculateMagneticDivergence(dev_grid.data(), dx, dy, dz, nx, ny, nz, n_cells);

    // Perform Comparison
    Real const fiducialDivergence = 3.6318132783263106;
    testingUtilities::checkResults(fiducialDivergence, testDivergence, "maximum divergence");
}
// =============================================================================
// End of tests for the magnetic field divergence functions
// =============================================================================

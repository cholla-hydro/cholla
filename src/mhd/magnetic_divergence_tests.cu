/*!
 * \file magnetic_divergence_tests.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the magnetic divergence code
 *
 */

// STL Includes
#include <iostream>
#include <random>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../mhd/magnetic_divergence.h"
#include "../utils/DeviceVector.h"
#include "../utils/testing_utilities.h"

#ifdef MHD
// =============================================================================
// Tests for the magnetic field divergence functions
// =============================================================================
TEST(tMHDGrid3DcheckMagneticDivergence, CorrectInputExpectCorrectOutput)
{
  // Grid Parameters & testing parameters
  size_t const gridSize = 96;  // Needs to be at least 64 so that each thread has a value
  size_t const n_ghost  = 4;

  Real dx      = 3;
  Real dy      = dx;
  Real dz      = dx;
  int nx       = static_cast<int>(gridSize + 2 * n_ghost);
  int ny       = nx;
  int nz       = nx;
  int n_cells  = nx * ny * nz;
  int n_fields = 8;

  // Setup host grid. Fill host grid with random values and randomly assign
  // maximum value
  std::vector<Real> host_grid(n_cells * n_fields);
  std::mt19937 prng(1);
  std::uniform_real_distribution<double> doubleRand(1, 5);
  for (double& host_data : host_grid) {
    host_data = doubleRand(prng) / 1E15;
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

  // Perform test
  InitializeChollaMPI(NULL, NULL);
  double max_magnetic_divergence = mhd::checkMagneticDivergence(dev_grid.data(), dx, dy, dz, nx, ny, nz, n_cells);
  MPI_Finalize();
  // Perform Comparison
  Real const fiducialDivergence = 3.6318132783263106 / 1E15;
  testing_utilities::Check_Results(fiducialDivergence, max_magnetic_divergence, "maximum divergence");
}
// =============================================================================
// End of tests for the magnetic field divergence functions
// =============================================================================
#endif  // MHD

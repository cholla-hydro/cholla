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

  // Instantiate Grid3D object
  Grid3D G;
  G.H.dx       = 3;
  G.H.dy       = G.H.dx;
  G.H.dz       = G.H.dx;
  G.H.nx       = gridSize + 2 * n_ghost;
  G.H.ny       = G.H.nx;
  G.H.nz       = G.H.nx;
  G.H.n_cells  = G.H.nx * G.H.ny * G.H.nz;
  G.H.n_fields = 8;

  // Setup host grid. Fill host grid with random values and randomly assign
  // maximum value
  std::vector<Real> host_grid(G.H.n_cells * G.H.n_fields);
  std::mt19937 prng(1);
  std::uniform_real_distribution<double> doubleRand(1, 5);
  for (size_t i = 0; i < host_grid.size(); i++) {
    host_grid.at(i) = doubleRand(prng) / 1E15;
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  G.C.device = dev_grid.data();
  dev_grid.cpyHostToDevice(host_grid);

  // Perform test
  InitializeChollaMPI(NULL, NULL);
  double max_magnetic_divergence = mhd::checkMagneticDivergence(G);
  MPI_Finalize();
  // Perform Comparison
  Real const fiducialDivergence = 3.6318132783263106 / 1E15;
  testingUtilities::checkResults(fiducialDivergence, max_magnetic_divergence, "maximum divergence");
}
// =============================================================================
// End of tests for the magnetic field divergence functions
// =============================================================================
#endif  // MHD

/*!
 * \file pcm_cuda_tests.cu
 * \brief Contains the tests for the code in pcm_cuda.h and pcm_cuda.cu
 */

// STL Includes
#include <random>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/pcm_cuda.h"
#include "../utils/basic_structs.h"
#include "../utils/testing_utilities.h"

void Check_Interface(reconstruction::InterfaceState const &test_data,
                     reconstruction::InterfaceState const &fiducial_data, size_t const direction)
{
  std::string const message = "Direction " + std::to_string(direction);

  testing_utilities::Check_Results(test_data.density, fiducial_data.density, "density " + message);
  testing_utilities::Check_Results(test_data.energy, fiducial_data.energy, "energy " + message);
  testing_utilities::Check_Results(test_data.pressure, fiducial_data.pressure, "pressure " + message);
  testing_utilities::Check_Results(test_data.velocity.x, fiducial_data.velocity.x, "velocity.x " + message);
  testing_utilities::Check_Results(test_data.velocity.y, fiducial_data.velocity.y, "velocity.y " + message);
  testing_utilities::Check_Results(test_data.velocity.z, fiducial_data.velocity.z, "velocity.z " + message);
  testing_utilities::Check_Results(test_data.momentum.x, fiducial_data.momentum.x, "momentum.x " + message);
  testing_utilities::Check_Results(test_data.momentum.y, fiducial_data.momentum.y, "momentum.y " + message);
  testing_utilities::Check_Results(test_data.momentum.z, fiducial_data.momentum.z, "momentum.z " + message);

#ifdef MHD
  testing_utilities::Check_Results(test_data.total_pressure, fiducial_data.total_pressure, "total_pressure" + message);
  testing_utilities::Check_Results(test_data.magnetic.x, fiducial_data.magnetic.x, "magnetic.x " + message);
  testing_utilities::Check_Results(test_data.magnetic.y, fiducial_data.magnetic.y, "magnetic.y " + message);
  testing_utilities::Check_Results(test_data.magnetic.z, fiducial_data.magnetic.z, "magnetic.z " + message);
#endif  // MHD
}

TEST(tAllReconstructionPCM, CorrectInputExpectCorrectOutput)
{
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx      = 3;
  size_t const ny      = 3;
  size_t const nz      = 3;
  size_t const xid     = 1;
  size_t const yid     = 1;
  size_t const zid     = 1;
  size_t const n_cells = nx * ny * nz;
  double const dx      = doubleRand(prng);
  double const dt      = doubleRand(prng);
  double const gamma   = 5.0 / 3.0;

  // Setup host grid. Fill host grid with random values and randomly assign values
  std::vector<double> host_grid(n_cells * grid_enum::num_fields);
  for (Real &val : host_grid) {
    val = doubleRand(prng);
  }

  // Test each direction
  reconstruction::InterfaceState test_interface_0 =
      reconstruction::PCM_Reconstruction<0>(host_grid.data(), xid, yid, zid, nx, ny, n_cells, gamma);
  reconstruction::InterfaceState test_interface_1 =
      reconstruction::PCM_Reconstruction<1>(host_grid.data(), xid, yid, zid, nx, ny, n_cells, gamma);
  reconstruction::InterfaceState test_interface_2 =
      reconstruction::PCM_Reconstruction<2>(host_grid.data(), xid, yid, zid, nx, ny, n_cells, gamma);

  // Fiducial values
  reconstruction::InterfaceState fiducial_interface_0{
      4.7339225843521469,     {0.57689124878761044, 0.81171783817468879, 0.78775336609408397}, 2.140751210734309,
      9.9999999999999995e-21, {1.9592441096173099, 0.9613575629289659, 3.6497594046359176},    9.0417947782995878},
      fiducial_interface_1{
          4.7339225843521469,     {0.81171783817468879, 0.78775336609408397, 0.57689124878761044}, 2.140751210734309,
          9.9999999999999995e-21, {0.9613575629289659, 3.6497594046359176, 1.9592441096173099},    9.041794778299586},
      fiducial_interface_2{
          4.7339225843521469,     {0.78775336609408397, 0.57689124878761044, 0.81171783817468879}, 2.140751210734309,
          9.9999999999999995e-21, {3.6497594046359176, 1.9592441096173099, 0.9613575629289659},    9.041794778299586};

  // Check correctness
  Check_Interface(test_interface_0, fiducial_interface_0, 0);
  Check_Interface(test_interface_1, fiducial_interface_1, 1);
  Check_Interface(test_interface_2, fiducial_interface_2, 2);
}

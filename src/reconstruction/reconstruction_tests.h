/*!
 * \file reconstruction_tests.cu
 * \brief Tests for the contents of reconstruction.h
 *
 */

// STL Includes
#include <string>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/reconstruction.h"
#include "../utils/DeviceVector.h"
#include "../utils/gpu.hpp"
#include "../utils/testing_utilities.h"

#ifdef MHD
namespace
{
__global__ void test_prim_2_char(reconstruction::Primitive const primitive,
                                 reconstruction::Primitive const primitive_slope, Real const gamma,
                                 Real const sound_speed, Real const sound_speed_squared,
                                 reconstruction::Characteristic *characteristic_slope)
{
  *characteristic_slope =
      reconstruction::Primitive_To_Characteristic(primitive, primitive_slope, sound_speed, sound_speed_squared, gamma);
}

__global__ void test_char_2_prim(reconstruction::Primitive const primitive,
                                 reconstruction::Characteristic const characteristic_slope, Real const gamma,
                                 Real const sound_speed, Real const sound_speed_squared,
                                 reconstruction::Primitive *primitive_slope)
{
  reconstruction::Characteristic_To_Primitive(primitive, characteristic_slope, sound_speed, sound_speed_squared, gamma,
                                              *primitive_slope);
}
}  // namespace

TEST(tMHDReconstructionPrimitive2Characteristic, CorrectInputExpectCorrectOutput)
{
  // Test parameters
  Real const &gamma = 5. / 3.;
  reconstruction::Primitive const primitive{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Primitive const primitive_slope{9, 10, 11, 12, 13, 14, 15, 16};
  Real const sound_speed         = hydro_utilities::Calc_Sound_Speed(primitive.pressure, primitive.density, gamma);
  Real const sound_speed_squared = sound_speed * sound_speed;

  // Run test
  cuda_utilities::DeviceVector<reconstruction::Characteristic> dev_results(1);
  hipLaunchKernelGGL(test_prim_2_char, 1, 1, 0, 0, primitive, primitive_slope, gamma, sound_speed, sound_speed_squared,
                     dev_results.data());
  CudaCheckError();
  cudaDeviceSynchronize();
  reconstruction::Characteristic const host_results = dev_results.at(0);

  // Check results
  reconstruction::Characteristic const fiducial_results{
      3.67609032478613384e+00, -5.64432521030159506e-01, -3.31429408151064075e+00, 7.44000000000000039e+00,
      3.29052143725318791e+00, -1.88144173676719539e-01, 4.07536568422372625e+00};
  testingUtilities::checkResults(fiducial_results.a0, host_results.a0, "a0");
  testingUtilities::checkResults(fiducial_results.a1, host_results.a1, "a1");
  testingUtilities::checkResults(fiducial_results.a2, host_results.a2, "a2");
  testingUtilities::checkResults(fiducial_results.a3, host_results.a3, "a3");
  testingUtilities::checkResults(fiducial_results.a4, host_results.a4, "a4");
  testingUtilities::checkResults(fiducial_results.a5, host_results.a5, "a5");
  testingUtilities::checkResults(fiducial_results.a6, host_results.a6, "a6");
}

TEST(tMHDReconstructionCharacteristic2Primitive, CorrectInputExpectCorrectOutput)
{
  // Test parameters
  Real const &gamma = 5. / 3.;
  reconstruction::Primitive const primitive{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Characteristic const characteristic_slope{17, 18, 19, 20, 21, 22, 23};
  Real const sound_speed         = hydro_utilities::Calc_Sound_Speed(primitive.pressure, primitive.density, gamma);
  Real const sound_speed_squared = sound_speed * sound_speed;

  // Run test
  cuda_utilities::DeviceVector<reconstruction::Primitive> dev_results(1);
  hipLaunchKernelGGL(test_char_2_prim, 1, 1, 0, 0, primitive, characteristic_slope, gamma, sound_speed,
                     sound_speed_squared, dev_results.data());
  CudaCheckError();
  cudaDeviceSynchronize();
  reconstruction::Primitive const host_results = dev_results.at(0);

  // Check results
  reconstruction::Primitive const fiducial_results{
      6.73268997307368267e+01, 1.79977606552837130e+01,  9.89872908629502835e-01, -4.94308571170036792e+00,
      3.94390831089473579e+02, -9.99000000000000000e+02, 2.88004228079705342e+01, 9.36584592818786064e+01};
  testingUtilities::checkResults(fiducial_results.density, host_results.density, "density");
  testingUtilities::checkResults(fiducial_results.velocity_x, host_results.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_results.velocity_y, host_results.velocity_y, "velocity_y", 1.34E-14);
  testingUtilities::checkResults(fiducial_results.velocity_z, host_results.velocity_z, "velocity_z", 1.6E-14);
  testingUtilities::checkResults(fiducial_results.pressure, host_results.pressure, "pressure");
  testingUtilities::checkResults(fiducial_results.magnetic_y, host_results.magnetic_y, "magnetic_y");
  testingUtilities::checkResults(fiducial_results.magnetic_z, host_results.magnetic_z, "magnetic_z");
}
#endif  // MHD
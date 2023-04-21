/*!
 * \file reconstruction_tests.cu
 * \brief Tests for the contents of reconstruction.h
 *
 */

// STL Includes
#include <algorithm>
#include <string>
#include <vector>

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

TEST(tALLReconstructionLoadData, CorrectInputExpectCorrectOutput)
{
  // Set up test and mock up grid
  size_t const nx = 3, ny = 3, nz = 3;
  size_t const n_cells = nx * ny * nz;
  size_t const xid = 1, yid = 1, zid = 1;
  size_t const o1 = grid_enum::momentum_x, o2 = grid_enum::momentum_y, o3 = grid_enum::momentum_z;
  Real const gamma = 5. / 3.;

  std::vector<Real> conserved(n_cells * grid_enum::num_fields);
  std::iota(conserved.begin(), conserved.end(), 0.0);

  // Up the energy part of the grid to avoid negative pressure
  for (size_t i = grid_enum::Energy * n_cells; i < (grid_enum::Energy + 1) * n_cells; i++) {
    conserved.at(i) *= 5.0E2;
  }

  // Get test data
  auto const test_data = reconstruction::Load_Data(conserved.data(), xid, yid, zid, nx, ny, n_cells, o1, o2, o3, gamma);

// Check results
#ifdef MHD
  reconstruction::Primitive const fiducial_data{
      13, 3.0769230769230771, 5.1538461538461542, 7.2307692307692308, 9662.3910256410272, 147.5, 173.5, 197.5};
  testingUtilities::checkResults(fiducial_data.density, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, test_data.pressure, "pressure");
  testingUtilities::checkResults(fiducial_data.magnetic_x, test_data.magnetic_x, "magnetic_x");
  testingUtilities::checkResults(fiducial_data.magnetic_y, test_data.magnetic_y, "magnetic_y");
  testingUtilities::checkResults(fiducial_data.magnetic_z, test_data.magnetic_z, "magnetic_z");
#else  // MHD
  reconstruction::Primitive fiducial_data{13, 3.0769230769230771, 5.1538461538461542, 7.2307692307692308,
                                          39950.641025641031};
  #ifdef DE
  fiducial_data.pressure = 34274.282506448195;
  #endif  // DE
  testingUtilities::checkResults(fiducial_data.density, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, test_data.pressure, "pressure");
#endif    // MHD
}

TEST(tALLReconstructionComputeSlope, CorrectInputExpectCorrectOutput)
{
// Setup input data
#ifdef MHD
  reconstruction::Primitive left{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Primitive right{6, 7, 8, 9, 10, 11, 12, 13};
#else   // MHD
  reconstruction::Primitive left{1, 2, 3, 4, 5};
  reconstruction::Primitive right{6, 7, 8, 9, 10};
#endif  // MHD
  Real const coef = 0.5;

  // Get test data
  auto test_data = reconstruction::Compute_Slope(left, right, coef);

  // Check results
#ifdef MHD
  Real const fiducial_data = -2.5;
  testingUtilities::checkResults(fiducial_data, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data, test_data.pressure, "pressure");
  testingUtilities::checkResults(fiducial_data, test_data.magnetic_y, "magnetic_y");
  testingUtilities::checkResults(fiducial_data, test_data.magnetic_z, "magnetic_z");
#else   // MHD
  Real const fiducial_data = -2.5;
  testingUtilities::checkResults(fiducial_data, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data, test_data.pressure, "pressure");
#endif  // MHD
}

TEST(tALLReconstructionVanLeerSlope, CorrectInputExpectCorrectOutput)
{
// Setup input data
#ifdef MHD
  reconstruction::Primitive left{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Primitive right{6, 7, 8, 9, 10, 11, 12, 13};
#else   // MHD
  reconstruction::Primitive left{1, 2, 3, 4, 5};
  reconstruction::Primitive right{6, 7, 8, 9, 10};
#endif  // MHD

  // Get test data
  auto test_data = reconstruction::Van_Leer_Slope(left, right);

  // Check results
#ifdef MHD
  reconstruction::Primitive const fiducial_data{1.7142857142857142, 3.1111111111111112, 4.3636363636363633,
                                                5.5384615384615383, 6.666666666666667,  0,
                                                8.8421052631578956, 9.9047619047619051};
  testingUtilities::checkResults(fiducial_data.density, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, test_data.pressure, "pressure");
  testingUtilities::checkResults(fiducial_data.magnetic_y, test_data.magnetic_y, "magnetic_y");
  testingUtilities::checkResults(fiducial_data.magnetic_z, test_data.magnetic_z, "magnetic_z");
#else   // MHD
  reconstruction::Primitive const fiducial_data{1.7142857142857142, 3.1111111111111112, 4.3636363636363633,
                                                5.5384615384615383, 6.666666666666667};
  testingUtilities::checkResults(fiducial_data.density, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, test_data.pressure, "pressure");
#endif  // MHD
}

__global__ void test_monotize_characteristic_return_primitive(
    reconstruction::Primitive const primitive, reconstruction::Primitive const del_L,
    reconstruction::Primitive const del_R, reconstruction::Primitive const del_C, reconstruction::Primitive const del_G,
    reconstruction::Characteristic const del_a_L, reconstruction::Characteristic const del_a_R,
    reconstruction::Characteristic const del_a_C, reconstruction::Characteristic const del_a_G, Real const sound_speed,
    Real const sound_speed_squared, Real const gamma, reconstruction::Primitive *monotonized_slope)
{
  *monotonized_slope = reconstruction::Monotonize_Characteristic_Return_Primitive(
      primitive, del_L, del_R, del_C, del_G, del_a_L, del_a_R, del_a_C, del_a_G, sound_speed, sound_speed_squared,
      gamma);
}

TEST(tALLReconstructionMonotonizeCharacteristicReturnPrimitive, CorrectInputExpectCorrectOutput)
{
#ifdef MHD
  reconstruction::Primitive const primitive{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Primitive const del_L{9, 10, 11, 12, 13, 14, 15, 16};
  reconstruction::Primitive const del_R{17, 18, 19, 20, 21, 22, 23, 24};
  reconstruction::Primitive const del_C{25, 26, 27, 28, 29, 30, 31, 32};
  reconstruction::Primitive const del_G{33, 34, 35, 36, 37, 38, 39, 40};
  reconstruction::Characteristic const del_a_L{41, 42, 43, 44, 45, 46, 47};
  reconstruction::Characteristic const del_a_R{48, 49, 50, 51, 52, 53, 54};
  reconstruction::Characteristic const del_a_C{55, 56, 57, 58, 59, 60, 61};
  reconstruction::Characteristic const del_a_G{62, 64, 65, 66, 67, 68, 69};
#else   // MHD
  reconstruction::Primitive const primitive{1, 2, 3, 4, 5};
  reconstruction::Primitive const del_L{9, 10, 11, 12, 13};
  reconstruction::Primitive const del_R{17, 18, 19, 20, 21};
  reconstruction::Primitive const del_C{25, 26, 27, 28, 29};
  reconstruction::Primitive const del_G{33, 34, 35, 36, 37};
  reconstruction::Characteristic const del_a_L{41, 42, 43, 44, 45};
  reconstruction::Characteristic const del_a_R{48, 49, 50, 51, 52};
  reconstruction::Characteristic const del_a_C{55, 56, 57, 58, 59};
  reconstruction::Characteristic const del_a_G{62, 64, 65, 66, 67};
#endif  // MHD
  Real const sound_speed = 17.0, sound_speed_squared = sound_speed * sound_speed;
  Real const gamma = 5. / 3.;

  // Get test data
  cuda_utilities::DeviceVector<reconstruction::Primitive> dev_results(1);
  hipLaunchKernelGGL(test_monotize_characteristic_return_primitive, 1, 1, 0, 0, primitive, del_L, del_R, del_C, del_G,
                     del_a_L, del_a_R, del_a_C, del_a_G, sound_speed, sound_speed_squared, gamma, dev_results.data());
  CudaCheckError();
  cudaDeviceSynchronize();
  reconstruction::Primitive const host_results = dev_results.at(0);

  // Check results
#ifdef MHD
  reconstruction::Primitive const fiducial_data{174, 74.796411763317991,  19.428234044886157, 16.129327015450095, 33524,
                                                0,   -1385.8699833027156, -1407.694707449215};
  testingUtilities::checkResults(fiducial_data.density, host_results.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, host_results.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, host_results.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, host_results.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, host_results.pressure, "pressure");
  testingUtilities::checkResults(fiducial_data.magnetic_y, host_results.magnetic_y, "magnetic_y");
  testingUtilities::checkResults(fiducial_data.magnetic_z, host_results.magnetic_z, "magnetic_z");
#else   // MHD
  reconstruction::Primitive const fiducial_data{170, 68, 57, 58, 32946};
  testingUtilities::checkResults(fiducial_data.density, host_results.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, host_results.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, host_results.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, host_results.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, host_results.pressure, "pressure");
#endif  // MHD
}

TEST(tALLReconstructionMonotizeParabolicInterface, CorrectInputExpectCorrectOutput)
{
// Input Data
#ifdef MHD
  reconstruction::Primitive const cell_i{1.4708046701, 9.5021020181, 3.7123503442, 4.6476103466,
                                         3.7096802847, 8.9692274397, 9.3416846121, 2.7707989229};
  reconstruction::Primitive const cell_im1{3.9547588941, 3.1552319951, 3.0209247624, 9.5841013261,
                                           2.2945188332, 8.2028929443, 1.6941969156, 8.9424967039};
  reconstruction::Primitive const cell_ip1{5.1973323534, 6.9132613767, 1.8397298636, 5.341960387,
                                           9.093498542,  3.6911762486, 7.3777130085, 3.6711825219};
  reconstruction::Primitive interface_L_iph{6.7787324804, 9.5389820358, 9.8522754567, 7.8305142852,
                                            2.450533435,  9.4782390708, 5.6820584385, 4.7115587023};
  reconstruction::Primitive interface_R_imh{4.8015193892, 5.9124263972, 8.7513040382, 8.3659359773,
                                            1.339777121,  4.5589857979, 1.4398647311, 8.8727778983};
#else   // not MHD
  reconstruction::Primitive const cell_i{1.4708046701, 9.5021020181, 3.7123503442, 4.6476103466, 3.7096802847};
  reconstruction::Primitive const cell_im1{3.9547588941, 3.1552319951, 3.0209247624, 9.5841013261, 2.2945188332};
  reconstruction::Primitive const cell_ip1{5.1973323534, 6.9132613767, 1.8397298636, 5.341960387, 9.093498542};
  reconstruction::Primitive interface_L_iph{6.7787324804, 9.5389820358, 9.8522754567, 7.8305142852, 2.450533435};
  reconstruction::Primitive interface_R_imh{4.8015193892, 5.9124263972, 8.7513040382, 8.3659359773, 1.339777121};
#endif  // MHD

  // Get test data
  reconstruction::Monotonize_Parabolic_Interface(cell_i, cell_im1, cell_ip1, interface_L_iph, interface_R_imh);

// Check results
#ifdef MHD
  reconstruction::Primitive const fiducial_interface_L{1.4708046700999999, 9.5021020181000004, 3.7123503441999999,
                                                       4.6476103465999996, 3.7096802847000001, 0 < 9.3416846120999999,
                                                       2.7707989229000001};
  reconstruction::Primitive const fiducial_interface_R{1.4708046700999999, 9.428341982700001,  3.7123503441999999,
                                                       4.6476103465999996, 3.7096802847000001, 0 < 9.3416846120999999,
                                                       2.7707989229000001};
  testingUtilities::checkResults(fiducial_interface_L.density, interface_L_iph.density, "density");
  testingUtilities::checkResults(fiducial_interface_L.velocity_x, interface_L_iph.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_interface_L.velocity_y, interface_L_iph.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_interface_L.velocity_z, interface_L_iph.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_interface_L.pressure, interface_L_iph.pressure, "pressure");

  testingUtilities::checkResults(fiducial_interface_R.density, interface_R_imh.density, "density");
  testingUtilities::checkResults(fiducial_interface_R.velocity_x, interface_R_imh.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_interface_R.velocity_y, interface_R_imh.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_interface_R.velocity_z, interface_R_imh.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_interface_R.pressure, interface_R_imh.pressure, "pressure");
#else   // MHD
  reconstruction::Primitive const fiducial_interface_L{1.4708046700999999, 9.5021020181000004, 3.7123503441999999,
                                                       4.6476103465999996, 3.7096802847000001};
  reconstruction::Primitive const fiducial_interface_R{1.4708046700999999, 9.428341982700001, 3.7123503441999999,
                                                       4.6476103465999996, 3.7096802847000001};
  testingUtilities::checkResults(fiducial_interface_L.density, interface_L_iph.density, "density");
  testingUtilities::checkResults(fiducial_interface_L.velocity_x, interface_L_iph.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_interface_L.velocity_y, interface_L_iph.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_interface_L.velocity_z, interface_L_iph.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_interface_L.pressure, interface_L_iph.pressure, "pressure");

  testingUtilities::checkResults(fiducial_interface_R.density, interface_R_imh.density, "density");
  testingUtilities::checkResults(fiducial_interface_R.velocity_x, interface_R_imh.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_interface_R.velocity_y, interface_R_imh.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_interface_R.velocity_z, interface_R_imh.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_interface_R.pressure, interface_R_imh.pressure, "pressure");
#endif  // MHD
}

TEST(tALLReconstructionCalcInterfaceLinear, CorrectInputExpectCorrectOutput)
{
  // Setup input data
#ifdef MHD
  reconstruction::Primitive left{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Primitive right{6, 7, 8, 9, 10, 11, 12, 13};
#else   // MHD
  reconstruction::Primitive left{1, 2, 3, 4, 5};
  reconstruction::Primitive right{6, 7, 8, 9, 10};
#endif  // MHD
  Real const coef = 0.5;

  // Get test data
  auto test_data = reconstruction::Calc_Interface_Linear(left, right, coef);

  // Check results
#ifdef MHD
  reconstruction::Primitive const fiducial_data{2.5, 3.75, 5, 6.25, 7.5, 0, 10, 11.25};
  testingUtilities::checkResults(fiducial_data.density, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, test_data.pressure, "pressure");
  testingUtilities::checkResults(fiducial_data.magnetic_y, test_data.magnetic_y, "magnetic_y");
  testingUtilities::checkResults(fiducial_data.magnetic_z, test_data.magnetic_z, "magnetic_z");
#else   // MHD
  reconstruction::Primitive const fiducial_data{2.5, 3.75, 5, 6.25, 7.5};
  testingUtilities::checkResults(fiducial_data.density, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, test_data.pressure, "pressure");
#endif  // MHD
}

TEST(tALLReconstructionCalcInterfaceParabolic, CorrectInputExpectCorrectOutput)
{
  // Setup input data
#ifdef MHD
  reconstruction::Primitive cell_i{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Primitive cell_im1{6, 7, 8, 9, 10, 11, 12, 13};
  reconstruction::Primitive slopes_i{14, 15, 16, 17, 18, 19, 20, 21};
  reconstruction::Primitive slopes_im1{22, 23, 24, 25, 26, 27, 28, 29};
#else   // MHD
  reconstruction::Primitive cell_i{1, 2, 3, 4, 5};
  reconstruction::Primitive cell_im1{6, 7, 8, 9, 10};
  reconstruction::Primitive slopes_i{14, 15, 16, 17, 18};
  reconstruction::Primitive slopes_im1{22, 23, 24, 25, 26};
#endif  // MHD

  // Get test data
  auto test_data = reconstruction::Calc_Interface_Parabolic(cell_i, cell_im1, slopes_i, slopes_im1);

  // Check results
#ifdef MHD
  reconstruction::Primitive const fiducial_data{4.833333333333333,  5.833333333333333,  6.833333333333333,
                                                7.833333333333333,  8.8333333333333339, 0.0,
                                                10.833333333333334, 11.833333333333334};
  testingUtilities::checkResults(fiducial_data.density, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, test_data.pressure, "pressure");
  testingUtilities::checkResults(fiducial_data.magnetic_y, test_data.magnetic_y, "magnetic_y");
  testingUtilities::checkResults(fiducial_data.magnetic_z, test_data.magnetic_z, "magnetic_z");
#else   // MHD
  reconstruction::Primitive const fiducial_data{4.833333333333333, 5.833333333333333, 6.833333333333333,
                                                7.833333333333333, 8.8333333333333339};
  testingUtilities::checkResults(fiducial_data.density, test_data.density, "density");
  testingUtilities::checkResults(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testingUtilities::checkResults(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testingUtilities::checkResults(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testingUtilities::checkResults(fiducial_data.pressure, test_data.pressure, "pressure");
#endif  // MHD
}

TEST(tALLReconstructionWriteData, CorrectInputExpectCorrectOutput)
{
  // Set up test and mock up grid
#ifdef MHD
  reconstruction::Primitive interface {
    1, 2, 3, 4, 5, 6, 7, 8
  };
#else   // MHD
  reconstruction::Primitive interface {
    6, 7, 8, 9, 10
  };
#endif  // MHD
  size_t const nx = 3, ny = 3, nz = 3;
  size_t const n_cells = nx * ny * nz;
  size_t const xid = 1, yid = 1, zid = 1;
  size_t const id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
  size_t const o1 = grid_enum::momentum_x, o2 = grid_enum::momentum_y, o3 = grid_enum::momentum_z;
  Real const gamma = 5. / 3.;

  std::vector<Real> conserved(n_cells * grid_enum::num_fields);
  std::vector<Real> interface_arr(n_cells * grid_enum::num_fields);

  // Get test data
  reconstruction::Write_Data(interface, interface_arr.data(), conserved.data(), id, n_cells, o1, o2, o3, gamma);

// Fiducial Data
#ifdef MHD
  std::unordered_map<int, double> fiducial_interface = {{13, 1},     {40, 2},  {67, 3}, {94, 4},
                                                        {121, 78.5}, {148, 7}, {175, 8}};
#else   // MHD
  std::unordered_map<int, double> fiducial_interface = {{13, 6}, {40, 42}, {67, 48}, {94, 54}, {121, 597}};
#endif  // MHD

  // Perform Comparison
  for (size_t i = 0; i < interface_arr.size(); i++) {
    // Check the interface
    double test_val     = interface_arr.at(i);
    double fiducial_val = (fiducial_interface.find(i) == fiducial_interface.end()) ? 0.0 : fiducial_interface[i];

    testingUtilities::checkResults(fiducial_val, test_val, "Interface at i=" + std::to_string(i));
  }
}

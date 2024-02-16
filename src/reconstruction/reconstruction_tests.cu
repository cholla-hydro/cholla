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
#include "../io/io.h"
#include "../reconstruction/reconstruction.h"
#include "../utils/DeviceVector.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "../utils/testing_utilities.h"

#ifdef MHD
__global__ void Test_Prim_2_Char(reconstruction::Primitive const primitive,
                                 reconstruction::Primitive const primitive_slope,
                                 reconstruction::EigenVecs const eigenvectors, Real const gamma, Real const sound_speed,
                                 Real const sound_speed_squared, reconstruction::Characteristic *characteristic_slope)
{
  *characteristic_slope = reconstruction::Primitive_To_Characteristic(primitive, primitive_slope, eigenvectors,
                                                                      sound_speed, sound_speed_squared, gamma);
}

__global__ void Test_Char_2_Prim(reconstruction::Primitive const primitive,
                                 reconstruction::Characteristic const characteristic_slope,
                                 reconstruction::EigenVecs const eigenvectors, Real const gamma, Real const sound_speed,
                                 Real const sound_speed_squared, reconstruction::Primitive *primitive_slope)
{
  *primitive_slope = reconstruction::Characteristic_To_Primitive(primitive, characteristic_slope, eigenvectors,
                                                                 sound_speed, sound_speed_squared, gamma);
}

__global__ void Test_Compute_Eigenvectors(reconstruction::Primitive const primitive, Real const sound_speed,
                                          Real const sound_speed_squared, Real const gamma,
                                          reconstruction::EigenVecs *eigenvectors)
{
  *eigenvectors = reconstruction::Compute_Eigenvectors(primitive, sound_speed, sound_speed_squared, gamma);
}

TEST(tMHDReconstructionPrimitive2Characteristic, CorrectInputExpectCorrectOutput)
{
  // Test parameters
  Real const &gamma = 5. / 3.;
  reconstruction::Primitive const primitive{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Primitive const primitive_slope{9, 10, 11, 12, 13, 14, 15, 16};
  reconstruction::EigenVecs const eigenvectors{
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
  };
  Real const sound_speed         = hydro_utilities::Calc_Sound_Speed(primitive.pressure, primitive.density, gamma);
  Real const sound_speed_squared = sound_speed * sound_speed;

  // Run test
  cuda_utilities::DeviceVector<reconstruction::Characteristic> dev_results(1);
  hipLaunchKernelGGL(Test_Prim_2_Char, 1, 1, 0, 0, primitive, primitive_slope, eigenvectors, gamma, sound_speed,
                     sound_speed_squared, dev_results.data());
  GPU_Error_Check();
  cudaDeviceSynchronize();
  reconstruction::Characteristic const host_results = dev_results.at(0);

  // Check results
  reconstruction::Characteristic const fiducial_results{-40327, 110, -132678, 7.4400000000000004, 98864, 98, 103549};
  testing_utilities::Check_Results(fiducial_results.a0, host_results.a0, "a0");
  testing_utilities::Check_Results(fiducial_results.a1, host_results.a1, "a1");
  testing_utilities::Check_Results(fiducial_results.a2, host_results.a2, "a2");
  testing_utilities::Check_Results(fiducial_results.a3, host_results.a3, "a3");
  testing_utilities::Check_Results(fiducial_results.a4, host_results.a4, "a4");
  testing_utilities::Check_Results(fiducial_results.a5, host_results.a5, "a5");
  testing_utilities::Check_Results(fiducial_results.a6, host_results.a6, "a6");
}

TEST(tMHDReconstructionCharacteristic2Primitive, CorrectInputExpectCorrectOutput)
{
  // Test parameters
  Real const &gamma = 5. / 3.;
  reconstruction::Primitive const primitive{1, 2, 3, 4, 5, 6, 7, 8};
  reconstruction::Characteristic const characteristic_slope{17, 18, 19, 20, 21, 22, 23};
  reconstruction::EigenVecs const eigenvectors{
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
  };
  Real const sound_speed         = hydro_utilities::Calc_Sound_Speed(primitive.pressure, primitive.density, gamma);
  Real const sound_speed_squared = sound_speed * sound_speed;

  // Run test
  cuda_utilities::DeviceVector<reconstruction::Primitive> dev_results(1);
  hipLaunchKernelGGL(Test_Char_2_Prim, 1, 1, 0, 0, primitive, characteristic_slope, eigenvectors, gamma, sound_speed,
                     sound_speed_squared, dev_results.data());
  GPU_Error_Check();
  cudaDeviceSynchronize();
  reconstruction::Primitive const host_results = dev_results.at(0);

  // Check results
  reconstruction::Primitive const fiducial_results{1740, 2934, -2526, -2828, 14333.333333333338, 0.0, -24040, 24880};
  testing_utilities::Check_Results(fiducial_results.density, host_results.density, "density");
  testing_utilities::Check_Results(fiducial_results.velocity_x, host_results.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_results.velocity_y, host_results.velocity_y, "velocity_y", 1.34E-14);
  testing_utilities::Check_Results(fiducial_results.velocity_z, host_results.velocity_z, "velocity_z", 1.6E-14);
  testing_utilities::Check_Results(fiducial_results.pressure, host_results.pressure, "pressure");
  testing_utilities::Check_Results(fiducial_results.magnetic_y, host_results.magnetic_y, "magnetic_y");
  testing_utilities::Check_Results(fiducial_results.magnetic_z, host_results.magnetic_z, "magnetic_z");
}

TEST(tMHDReconstructionComputeEigenvectors, CorrectInputExpectCorrectOutput)
{
  // Test parameters
  Real const &gamma = 5. / 3.;
  reconstruction::Primitive const primitive{1, 2, 3, 4, 5, 6, 7, 8};
  Real const sound_speed         = hydro_utilities::Calc_Sound_Speed(primitive.pressure, primitive.density, gamma);
  Real const sound_speed_squared = sound_speed * sound_speed;

  // Run test
  cuda_utilities::DeviceVector<reconstruction::EigenVecs> dev_results(1);
  hipLaunchKernelGGL(Test_Compute_Eigenvectors, 1, 1, 0, 0, primitive, sound_speed, sound_speed_squared, gamma,
                     dev_results.data());
  GPU_Error_Check();
  cudaDeviceSynchronize();
  reconstruction::EigenVecs const host_results = dev_results.at(0);
  // std::cout << to_string_exact(host_results.magnetosonic_speed_fast) << ",";
  // std::cout << to_string_exact(host_results.magnetosonic_speed_slow) << ",";
  // std::cout << to_string_exact(host_results.magnetosonic_speed_fast_squared) << ",";
  // std::cout << to_string_exact(host_results.magnetosonic_speed_slow_squared) << ",";
  // std::cout << to_string_exact(host_results.alpha_fast) << ",";
  // std::cout << to_string_exact(host_results.alpha_slow) << ",";
  // std::cout << to_string_exact(host_results.beta_y) << ",";
  // std::cout << to_string_exact(host_results.beta_z) << ",";
  // std::cout << to_string_exact(host_results.n_fs) << ",";
  // std::cout << to_string_exact(host_results.sign) << ",";
  // std::cout << to_string_exact(host_results.q_fast) << ",";
  // std::cout << to_string_exact(host_results.q_slow) << ",";
  // std::cout << to_string_exact(host_results.a_fast) << ",";
  // std::cout << to_string_exact(host_results.a_slow) << ",";
  // std::cout << to_string_exact(host_results.q_prime_fast) << ",";
  // std::cout << to_string_exact(host_results.q_prime_slow) << ",";
  // std::cout << to_string_exact(host_results.a_prime_fast) << ",";
  // std::cout << to_string_exact(host_results.a_prime_slow) << "," << std::endl;
  // Check results
  reconstruction::EigenVecs const fiducial_results{
      12.466068627219666,   1.3894122191714398,  155.40286701855041,  1.9304663147829049,   0.20425471836256681,
      0.97891777490585408,  0.65850460786851805, 0.75257669470687782, 0.059999999999999984, 1,
      2.546253336541183,    1.3601203180183106,  0.58963258314939582, 2.825892204282022,    0.15277520019247093,
      0.081607219081098623, 0.03537795498896374, 0.1695535322569213};
  testing_utilities::Check_Results(fiducial_results.magnetosonic_speed_fast, host_results.magnetosonic_speed_fast,
                                   "magnetosonic_speed_fast");
  testing_utilities::Check_Results(fiducial_results.magnetosonic_speed_slow, host_results.magnetosonic_speed_slow,
                                   "magnetosonic_speed_slow");
  testing_utilities::Check_Results(fiducial_results.magnetosonic_speed_fast_squared,
                                   host_results.magnetosonic_speed_fast_squared, "magnetosonic_speed_fast_squared");
  testing_utilities::Check_Results(fiducial_results.magnetosonic_speed_slow_squared,
                                   host_results.magnetosonic_speed_slow_squared, "magnetosonic_speed_slow_squared");
  testing_utilities::Check_Results(fiducial_results.alpha_fast, host_results.alpha_fast, "alpha_fast");
  testing_utilities::Check_Results(fiducial_results.alpha_slow, host_results.alpha_slow, "alpha_slow");
  testing_utilities::Check_Results(fiducial_results.beta_y, host_results.beta_y, "beta_y");
  testing_utilities::Check_Results(fiducial_results.beta_z, host_results.beta_z, "beta_z");
  testing_utilities::Check_Results(fiducial_results.n_fs, host_results.n_fs, "n_fs");
  testing_utilities::Check_Results(fiducial_results.sign, host_results.sign, "sign");
  testing_utilities::Check_Results(fiducial_results.q_fast, host_results.q_fast, "q_fast");
  testing_utilities::Check_Results(fiducial_results.q_slow, host_results.q_slow, "q_slow");
  testing_utilities::Check_Results(fiducial_results.a_fast, host_results.a_fast, "a_fast");
  testing_utilities::Check_Results(fiducial_results.a_slow, host_results.a_slow, "a_slow");
  testing_utilities::Check_Results(fiducial_results.q_prime_fast, host_results.q_prime_fast, "q_prime_fast");
  testing_utilities::Check_Results(fiducial_results.q_prime_slow, host_results.q_prime_slow, "q_prime_slow");
  testing_utilities::Check_Results(fiducial_results.a_prime_fast, host_results.a_prime_fast, "a_prime_fast");
  testing_utilities::Check_Results(fiducial_results.a_prime_slow, host_results.a_prime_slow, "a_prime_slow");
}
#endif  // MHD

TEST(tALLReconstructionThreadGuard, CorrectInputExpectCorrectOutput)
{
  // Test parameters
  int const order = 3;
  int const nx    = 6;
  int const ny    = 6;
  int const nz    = 6;

  // fiducial data
  std::vector<int> fiducial_vals(nx * ny * nz, 1);
  fiducial_vals.at(86) = 0;

  // loop through all values of the indices and check them
  for (int xid = 0; xid < nx; xid++) {
    for (int yid = 0; yid < ny; yid++) {
      for (int zid = 0; zid < nz; zid++) {
        // Get the test value
        bool test_val = reconstruction::Thread_Guard<order>(nx, ny, nz, xid, yid, zid);

        // Compare
        int id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
        ASSERT_EQ(test_val, fiducial_vals.at(id))
            << "Test value not equal to fiducial value at id = " << id << std::endl;
      }
    }
  }
}

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
  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
  testing_utilities::Check_Results(fiducial_data.magnetic_x, test_data.magnetic_x, "magnetic_x");
  testing_utilities::Check_Results(fiducial_data.magnetic_y, test_data.magnetic_y, "magnetic_y");
  testing_utilities::Check_Results(fiducial_data.magnetic_z, test_data.magnetic_z, "magnetic_z");
#else  // MHD
  reconstruction::Primitive fiducial_data{13, 3.0769230769230771, 5.1538461538461542, 7.2307692307692308,
                                          39950.641025641031};
  #ifdef DE
  fiducial_data.pressure = 39950.641025641031;
  #endif  // DE
  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
#endif    // MHD
}

TEST(tALLReconstructionComputeSlope, CorrectInputExpectCorrectOutput)
{
// Setup input data
#ifdef MHD
  reconstruction::Primitive left{6, 7, 8, 9, 10, 11, 12, 13};
  reconstruction::Primitive right{1, 2, 3, 4, 5, 6, 7, 8};
#else   // MHD
  reconstruction::Primitive left{6, 7, 8, 9, 10};
  reconstruction::Primitive right{1, 2, 3, 4, 5};
#endif  // MHD
  Real const coef = 0.5;

  // Get test data
  auto test_data = reconstruction::Compute_Slope(left, right, coef);

  // Check results
#ifdef MHD
  Real const fiducial_data = -2.5;
  testing_utilities::Check_Results(fiducial_data, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data, test_data.pressure, "pressure");
  testing_utilities::Check_Results(fiducial_data, test_data.magnetic_y, "magnetic_y");
  testing_utilities::Check_Results(fiducial_data, test_data.magnetic_z, "magnetic_z");
#else   // MHD
  Real const fiducial_data = -2.5;
  testing_utilities::Check_Results(fiducial_data, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data, test_data.pressure, "pressure");
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
  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
  testing_utilities::Check_Results(fiducial_data.magnetic_y, test_data.magnetic_y, "magnetic_y");
  testing_utilities::Check_Results(fiducial_data.magnetic_z, test_data.magnetic_z, "magnetic_z");
#else   // MHD
  reconstruction::Primitive const fiducial_data{1.7142857142857142, 3.1111111111111112, 4.3636363636363633,
                                                5.5384615384615383, 6.666666666666667};
  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
#endif  // MHD
}

__global__ void Test_Monotize_Characteristic_Return_Primitive(
    reconstruction::Primitive const primitive, reconstruction::Primitive const del_L,
    reconstruction::Primitive const del_R, reconstruction::Primitive const del_C, reconstruction::Primitive const del_G,
    reconstruction::Characteristic const del_a_L, reconstruction::Characteristic const del_a_R,
    reconstruction::Characteristic const del_a_C, reconstruction::Characteristic const del_a_G,
    reconstruction::EigenVecs const eigenvectors, Real const sound_speed, Real const sound_speed_squared,
    Real const gamma, reconstruction::Primitive *monotonized_slope)
{
  *monotonized_slope = reconstruction::Monotonize_Characteristic_Return_Primitive(
      primitive, del_L, del_R, del_C, del_G, del_a_L, del_a_R, del_a_C, del_a_G, eigenvectors, sound_speed,
      sound_speed_squared, gamma);
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
  reconstruction::EigenVecs const eigenvectors{
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
  };

  // Get test data
  cuda_utilities::DeviceVector<reconstruction::Primitive> dev_results(1);
  hipLaunchKernelGGL(Test_Monotize_Characteristic_Return_Primitive, 1, 1, 0, 0, primitive, del_L, del_R, del_C, del_G,
                     del_a_L, del_a_R, del_a_C, del_a_G, eigenvectors, sound_speed, sound_speed_squared, gamma,
                     dev_results.data());
  GPU_Error_Check();
  cudaDeviceSynchronize();
  reconstruction::Primitive const host_results = dev_results.at(0);

  // Check results
#ifdef MHD
  reconstruction::Primitive const fiducial_data{5046, 2934, -2526, -2828, 1441532, 0.0, -69716, 72152};
  testing_utilities::Check_Results(fiducial_data.density, host_results.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, host_results.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, host_results.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, host_results.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, host_results.pressure, "pressure");
  testing_utilities::Check_Results(fiducial_data.magnetic_y, host_results.magnetic_y, "magnetic_y");
  testing_utilities::Check_Results(fiducial_data.magnetic_z, host_results.magnetic_z, "magnetic_z");
#else   // MHD
  reconstruction::Primitive const fiducial_data{170, 68, 57, 58, 32946};
  testing_utilities::Check_Results(fiducial_data.density, host_results.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, host_results.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, host_results.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, host_results.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, host_results.pressure, "pressure");
#endif  // MHD
}

TEST(tHYDROReconstructionMonotizeParabolicInterface, CorrectInputExpectCorrectOutput)
{
  // Input Data

  reconstruction::Primitive const cell_i{1.4708046701, 9.5021020181, 3.7123503442, 4.6476103466, 3.7096802847};
  reconstruction::Primitive const cell_im1{3.9547588941, 3.1552319951, 3.0209247624, 9.5841013261, 2.2945188332};
  reconstruction::Primitive const cell_ip1{5.1973323534, 6.9132613767, 1.8397298636, 5.341960387, 9.093498542};
  reconstruction::Primitive interface_L_iph{6.7787324804, 9.5389820358, 9.8522754567, 7.8305142852, 2.450533435};
  reconstruction::Primitive interface_R_imh{4.8015193892, 5.9124263972, 8.7513040382, 8.3659359773, 1.339777121};

  // Get test data
  reconstruction::Monotonize_Parabolic_Interface(cell_i, cell_im1, cell_ip1, interface_L_iph, interface_R_imh);

  // Check results
  reconstruction::Primitive const fiducial_interface_L{1.4708046700999999, 9.5021020181000004, 3.7123503441999999,
                                                       4.6476103465999996, 3.7096802847000001};
  reconstruction::Primitive const fiducial_interface_R{1.4708046700999999, 9.428341982700001, 3.7123503441999999,
                                                       4.6476103465999996, 3.7096802847000001};
  testing_utilities::Check_Results(fiducial_interface_L.density, interface_L_iph.density, "density");
  testing_utilities::Check_Results(fiducial_interface_L.velocity_x, interface_L_iph.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_interface_L.velocity_y, interface_L_iph.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_interface_L.velocity_z, interface_L_iph.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_interface_L.pressure, interface_L_iph.pressure, "pressure");

  testing_utilities::Check_Results(fiducial_interface_R.density, interface_R_imh.density, "density");
  testing_utilities::Check_Results(fiducial_interface_R.velocity_x, interface_R_imh.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_interface_R.velocity_y, interface_R_imh.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_interface_R.velocity_z, interface_R_imh.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_interface_R.pressure, interface_R_imh.pressure, "pressure");
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
  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
  testing_utilities::Check_Results(fiducial_data.magnetic_y, test_data.magnetic_y, "magnetic_y");
  testing_utilities::Check_Results(fiducial_data.magnetic_z, test_data.magnetic_z, "magnetic_z");
#else   // MHD
  reconstruction::Primitive const fiducial_data{2.5, 3.75, 5, 6.25, 7.5};
  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
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
  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
  testing_utilities::Check_Results(fiducial_data.magnetic_y, test_data.magnetic_y, "magnetic_y");
  testing_utilities::Check_Results(fiducial_data.magnetic_z, test_data.magnetic_z, "magnetic_z");
#else   // MHD
  reconstruction::Primitive const fiducial_data{4.833333333333333, 5.833333333333333, 6.833333333333333,
                                                7.833333333333333, 8.8333333333333339};
  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity_x, test_data.velocity_x, "velocity_x");
  testing_utilities::Check_Results(fiducial_data.velocity_y, test_data.velocity_y, "velocity_y");
  testing_utilities::Check_Results(fiducial_data.velocity_z, test_data.velocity_z, "velocity_z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
#endif  // MHD
}

TEST(tALLReconstructionPPMSingleVariable, CorrectInputExpectCorrectOutput)
{
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(-100, 100);

  // Set up testing parameters
  size_t const n_tests = 100;
  std::vector<double> input_data(n_tests * 5);
  for (double &val : input_data) {
    val = doubleRand(prng);
  }

  std::vector<double> fiducial_left_interface{
      50.429040149605328,  -40.625142952817804, 37.054257344499717,  -55.796322960572695, -14.949021655598202,
      -10.760611497035882, 71.107183338735751,  -29.453314279116661, 7.38606168778702,    -23.210826670297152,
      -85.15197822983292,  18.98804944849401,   64.754272117396766,  4.5584678980835918,  45.81912726561103,
      58.769584663215738,  47.626531326553447,  9.3792919223901166,  47.06767164062336,   -53.975231802858218,
      -81.51278133300454,  -74.554960772880221, 96.420244795844823,  37.498528618937456,  -41.370881014041672,
      -41.817524439980467, 58.391560533135817,  -85.991024651293131, -12.674113472365306, 30.421304081280084,
      43.700175645941769,  58.342347077360131,  -31.574197692184548, 98.151410701129635,  -9.4994975790183389,
      -87.49117921577357,  -94.449608348937488, 79.849643090061676,  93.096197902468759,  -64.374502025066192,
      82.037247010307937,  -60.629868182203786, -41.343090531127039, -75.449850543801574, -82.52313028208863,
      19.871484181185011,  -22.253989777496159, 86.943333900988137,  -83.887344220269938, 73.270857190511975,
      84.784625452008811,  -27.929776508530765, -9.6992610428405612, -65.233676045197072, -88.498474065470134,
      47.637114710282589,  -69.50911815749248,  -69.848254012650372, -7.4520009269431711, 90.887158278825865,
      -50.671539065300863, 13.424189957034622,  80.237684918029572,  32.454734198410179,  66.84741286999801,
      24.53669768915492,   -67.195147776790975, 72.277527112459907,  -46.094192444366435, -99.915875366345205,
      32.244024128018054,  -95.648868731550635, 17.922876720365402,  -86.334093878928797, -16.580223524066724,
      39.48244113577249,   64.203567686297504,  23.62791013796798,   59.620571575902432,  41.0983082454959,
      -30.533954819557593, -23.149979553301478, -54.098849622102691, -45.577469823900444, 33.284499908516068,
      -39.186662569988762, 76.266375356625161,  -51.650172854435624, -68.894636301310584, 98.410134045837452,
      -49.167117951549066, 78.440749922366507,  51.390453104722326,  3.1993391287610393,  43.749856317813453,
      -81.399433434996496, 88.385686355761862,  78.242223440453444,  27.539590130937498,  -6.9781781598207147,
  };
  std::vector<double> fiducial_right_interface{
      50.429040149605328,  4.4043935241855703,  37.054257344499717,  23.707343328192596,  -14.949021655598202,
      -10.760611497035882, 8.367260859616664,   8.5357943668839624,  7.38606168778702,    -23.210826670297152,
      -85.15197822983292,  18.98804944849401,   64.754272117396766,  4.5584678980835918,  45.81912726561103,
      58.769584663215738,  47.626531326553447,  23.370742401854159,  47.06767164062336,   -53.975231802858218,
      -81.51278133300454,  -74.554960772880221, 75.572387546643355,  61.339053128914685,  -41.370881014041672,
      -41.817524439980467, 58.391560533135817,  -85.991024651293131, -36.626332669233776, 30.421304081280084,
      20.637382412674096,  58.342347077360131,  -79.757902483702381, 98.151410701129635,  -9.4994975790183389,
      -87.49117921577357,  -39.384192078363533, 79.849643090061676,  93.096197902468759,  -64.374502025066192,
      82.037247010307937,  -20.951323678824952, 46.927431599533087,  -75.449850543801574, -54.603894223278004,
      -59.419110050353098, -22.253989777496159, 86.943333900988137,  -83.887344220269938, 73.270857190511975,
      84.784625452008811,  -27.929776508530765, -9.6992610428405612, -65.233676045197072, -88.498474065470134,
      47.637114710282589,  -69.50911815749248,  -69.848254012650372, -7.4520009269431711, 90.887158278825865,
      -79.086012597191512, -45.713537271527976, 80.237684918029572,  -60.666381661910016, 68.727158732184449,
      24.53669768915492,   -67.195147776790975, 72.610434112023597,  54.910597945673814,  -19.862686571231023,
      32.244024128018054,  -95.648868731550635, -34.761757909478987, -86.334093878928797, -16.580223524066724,
      39.48244113577249,   64.203567686297504,  0.77846541072490538, 59.620571575902432,  41.0983082454959,
      -2.6491435658297036, -23.149979553301478, -54.098849622102691, -45.577469823900444, 33.284499908516068,
      -39.186662569988762, 76.266375356625161,  -51.650172854435624, -68.894636301310584, 98.410134045837452,
      30.9954824410611,    78.440749922366507,  51.390453104722326,  70.625792807373429,  43.749856317813453,
      -81.399433434996496, 88.385686355761862,  78.242223440453444,  27.539590130937498,  -6.9781781598207147,
  };

  // Run n_tests iterations of the loop choosing random numbers to put into the interface state computation and checking
  // the results
  for (size_t i = 0; i < n_tests; i++) {
    // Run the function
    double test_left_interface, test_right_interface;
    size_t const idx = 5 * i;
    reconstruction::PPM_Single_Variable(input_data[idx], input_data[idx + 1], input_data[idx + 2], input_data[idx + 3],
                                        input_data[idx + 4], test_left_interface, test_right_interface);

    // Compare results
    testing_utilities::Check_Results(fiducial_left_interface.at(i), test_left_interface, "left i+1/2 interface");
    testing_utilities::Check_Results(fiducial_right_interface.at(i), test_right_interface, "right i-1/2 interface");
  }
}

TEST(tALLReconstructionWriteData, CorrectInputExpectCorrectOutput)
{
  // Set up test and mock up grid
#ifdef MHD
  reconstruction::Primitive interface{1, 2, 3, 4, 5, 6, 7, 8};
#else   // MHD
  reconstruction::Primitive interface{6, 7, 8, 9, 10};
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

    testing_utilities::Check_Results(fiducial_val, test_val, "Interface at i=" + std::to_string(i));
  }
}

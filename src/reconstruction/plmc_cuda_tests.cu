/*!
 * \file plmc_cuda_tests.cu
 * \brief Tests for the contents of plmc_cuda.h
 *
 */

// STL Includes
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../reconstruction/plmc_cuda.h"
#include "../utils/DeviceVector.h"
#include "../utils/hydro_utilities.h"
#include "../utils/testing_utilities.h"

__global__ void Plmc_Runner(hydro_utilities::Primitive const cell_im1, hydro_utilities::Primitive const cell_i,
                            hydro_utilities::Primitive const cell_ip1, Real const dt, Real const dx, Real const gamma,
                            Real const sign, hydro_utilities::Primitive *interface)
{
  interface[0] = reconstruction::PLMC_Reconstruction(cell_im1, cell_i, cell_ip1, dt, dx, gamma, sign);
}

TEST(tAllReconstructionPLMC, CorrectInputExpectCorrectOutput)
{
  // Mock up needed information
  double const dx    = 20.2288558964;
  double const dt    = 28.5895937563;
  double const gamma = 5.0 / 3.0;
  hydro_utilities::Primitive const cell_im1{94.0737573854,
                                            {86.1240771558, 38.3149551956, 80.7871904777},
                                            75.914350552,
                                            {65.05895878, 41.8340074107, 37.5939259052}};
  hydro_utilities::Primitive const cell_i{96.0525043314,
                                          {21.5684784546, 19.3096825095, 79.6815953751},
                                          91.852075682,
                                          {26.3770673291, 89.8731274153, 93.0662874044}};
  hydro_utilities::Primitive const cell_ip1{39.2963436066,
                                            {15.3963384821, 52.4727085915, 38.7033417224},
                                            35.2447511462,
                                            {91.2885507593, 13.4899966998, 80.3737860571}};

  // Test each sign
  cuda_utilities::DeviceVector<hydro_utilities::Primitive> dev_interface(2);
  hipLaunchKernelGGL(Plmc_Runner, 1, 1, 0, 0, cell_im1, cell_i, cell_ip1, dt, dx, gamma, 1.0, dev_interface.data());
  hipLaunchKernelGGL(Plmc_Runner, 1, 1, 0, 0, cell_im1, cell_i, cell_ip1, dt, dx, gamma, -1.0,
                     dev_interface.data() + 1);

// Fiducial values
#ifdef MHD
  hydro_utilities::Primitive const fiducial_interface_1{77.373002976265923,
                                                        {18.565998641863278, 18.402359257184891, 78.742035252990831},
                                                        62.081014807242752,
                                                        {-36.395878840497765, 71.675187995051118, 74.221781722221834}};
  hydro_utilities::Primitive const fiducial_interface_m1{114.73200568653408,
                                                         {24.570958267336724, 20.217005761815109, 80.621155497209173},
                                                         121.62313655675726,
                                                         {-36.395878840497765, 108.07106683554889, 111.91079308657817}};
#else
  hydro_utilities::Primitive const fiducial_interface_1{
      96.052504331400002, {15.93495759469292, 19.3096825095, 78.605045632348165}, 91.852075682000006};
  hydro_utilities::Primitive const fiducial_interface_m1{
      96.052504331400002, {27.201999314507084, 19.3096825095, 80.758145117851839}, 91.852075682000006};
#endif  // MHD
  // Check correctness
  testing_utilities::Check_Primitive(dev_interface[0], fiducial_interface_1);
  testing_utilities::Check_Primitive(dev_interface[1], fiducial_interface_m1);
}

TEST(tAllReconstructionPLMCCharacteristicEvolution, CorrectInputExpectCorrectOutput)
{
  // Mock up needed information
  double const dx          = 20.2288558964;
  double const dt          = 28.5895937563;
  double const sound_speed = 1.26245095747556224;
  double const gamma       = 5.0 / 3.0;
  hydro_utilities::Primitive const cell_i{96.0525043314,
                                          {21.5684784546, 19.3096825095, 79.6815953751},
                                          91.852075682,
                                          {26.3770673291, 89.8731274153, 93.0662874044}};
  hydro_utilities::Primitive const del_m{3.2963436066,
                                         {1.53963384821, 5.24727085915, 3.87033417224},
                                         3.52447511462,
                                         {9.12885507593, 1.34899966998, 8.03737860571}};

  hydro_utilities::Primitive test_interface_1{77.373002976265923,
                                              {18.565998641863278, 18.402359257184891, 78.742035252990831},
                                              62.081014807242752,
                                              {-36.395878840497765, 71.675187995051118, 74.221781722221834}};
  hydro_utilities::Primitive test_interface_m1 = test_interface_1;

  // Test each sign
  reconstruction::PLMC_Characteristic_Evolution(cell_i, del_m, dt, dx, sound_speed, 1.0, test_interface_1);
  reconstruction::PLMC_Characteristic_Evolution(cell_i, del_m, dt, dx, sound_speed, -1.0, test_interface_m1);

  // Fiducial values
  hydro_utilities::Primitive fiducial_interface_1{-77.372015909163821,
                                                  {-4.9261771938467085, -61.573637147740598, 19.752545258363881},
                                                  -158.19368148106042,
                                                  {-36.395878840497765, 71.675187995051118, 74.221781722221834}};
  hydro_utilities::Primitive fiducial_interface_m1{77.373002976265923,
                                                   {18.565998641863278, 18.402359257184891, 78.742035252990831},
                                                   62.081014807242752,
                                                   {-36.395878840497765, 71.675187995051118, 74.221781722221834}};

  // Check correctness
  testing_utilities::Check_Primitive(test_interface_1, fiducial_interface_1);
  testing_utilities::Check_Primitive(test_interface_m1, fiducial_interface_m1);
}

/*!
 * \file plm_cuda_tests.cu
 * \brief Tests for the contents of plm_cuda.h and plm_cuda.cu
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
#include <algorithm>

#include "../global/global.h"
#include "../io/io.h"
#include "../reconstruction/plm_cuda.h"
#include "../utils/DeviceVector.h"
#include "../utils/hydro_utilities.h"
#include "../utils/testing_utilities.h"

TEST(tHYDROPlmReconstructor, CorrectInputExpectCorrectOutput)
{
#ifndef VL
  std::cerr << "Warning: The tHYDROPlmReconstructor.CorrectInputExpectCorrectOutput test only supports the Van Leer "
               "(VL) integrator"
            << std::endl;
  return;
#endif  // VL
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx       = 5;
  size_t const ny       = 4;
  size_t const nz       = 4;
  size_t const n_fields = 5;
  double const dx       = doubleRand(prng);
  double const dt       = doubleRand(prng);
  double const gamma    = 5.0 / 3.0;

  // Setup host grid. Fill host grid with random values and randomly assign maximum value
  std::vector<double> host_grid(nx * ny * nz * n_fields);
  for (Real &val : host_grid) {
    val = doubleRand(prng);
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

  // Fiducial Data
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{26, 3.8877922383184833},
                                                                           {27, 0.70033864721549188},
                                                                           {106, 5.6625525038177784},
                                                                           {107, 3.0633780053857027},
                                                                           {186, 4.0069556576401011},
                                                                           {187, 2.1015872413794123},
                                                                           {266, 5.1729859852329314},
                                                                           {267, 3.9675148506537838},
                                                                           {346, 9.6301414677176531},
                                                                           {347, 21.091316282933843}},
                                                                          {{21, 0.74780807318015607},
                                                                           {37, 0.19457128219588618},
                                                                           {101, 5.6515522777659895},
                                                                           {117, 4.4286255636679313},
                                                                           {181, 0.13115998072061905},
                                                                           {197, 2.2851440769830953},
                                                                           {261, 1.5834637771067519},
                                                                           {277, 2.697375839048191},
                                                                           {341, 23.043749364531674},
                                                                           {357, 82.515887983144168}},
                                                                          {{25, 2.2863650183226212},
                                                                           {29, 1.686415421301841},
                                                                           {105, 0.72340346106443465},
                                                                           {109, 5.9563546443402542},
                                                                           {185, 3.6128571662018358},
                                                                           {189, 5.3735653401079038},
                                                                           {265, 0.95177493689267167},
                                                                           {269, 0.46056494878491938},
                                                                           {345, 3.1670194578067843},
                                                                           {349, 19.142817472509272}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{25, 3.8877922383184833},
                                                                            {26, 0.70033864721549188},
                                                                            {105, 1.594778794367564},
                                                                            {106, 3.0633780053857027},
                                                                            {185, 4.0069556576401011},
                                                                            {186, 2.1015872413794123},
                                                                            {265, 1.7883678016935782},
                                                                            {266, 3.9675148506537838},
                                                                            {345, 2.8032969746372531},
                                                                            {346, 21.091316282933843}},
                                                                           {{17, 0.43265217076853835},
                                                                            {33, 0.19457128219588618},
                                                                            {97, 3.2697645945288754},
                                                                            {113, 4.4286255636679313},
                                                                            {177, 0.07588397666718491},
                                                                            {193, 2.2851440769830953},
                                                                            {257, 0.91612950577699748},
                                                                            {273, 2.697375839048191},
                                                                            {337, 13.332201861384396},
                                                                            {353, 82.515887983144168}},
                                                                           {{5, 2.2863650183226212},
                                                                            {9, 1.686415421301841},
                                                                            {85, 0.72340346106443465},
                                                                            {89, 1.77925054463361},
                                                                            {165, 5.3997753452111859},
                                                                            {169, 1.4379190463124141},
                                                                            {245, 0.95177493689267167},
                                                                            {249, 0.46056494878491938},
                                                                            {325, 6.6889498465051398},
                                                                            {329, 1.6145084086614285}}};

#ifdef PLMP
  // Change the fiducial data for the PLMP version of this test. Not all elements need to be changed
  fiducial_interface_left.at(2)[29]  = 1.1600386335016277;
  fiducial_interface_left.at(2)[109] = 4.0972120006692201;
  fiducial_interface_left.at(2)[189] = 3.6963273197291162;
  fiducial_interface_left.at(2)[269] = 0.31680991947688009;
  fiducial_interface_left.at(2)[349] = 13.167815913968777;

  fiducial_interface_right.at(2)[9]   = 2.2127922091020542;
  fiducial_interface_right.at(2)[89]  = 2.3346037361106178;
  fiducial_interface_right.at(2)[169] = 1.8867332584893817;
  fiducial_interface_right.at(2)[249] = 0.60431997809295868;
  fiducial_interface_right.at(2)[329] = 2.1184410336202282;
#endif  // PLMP

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Assign the shape
    size_t nx_rot, ny_rot, nz_rot;
    switch (direction) {
      case 0:
        nx_rot = nx;
        ny_rot = ny;
        nz_rot = nz;
        break;
      case 1:
        nx_rot = ny;
        ny_rot = nz;
        nz_rot = nx;
        break;
      case 2:
        nx_rot = nz;
        ny_rot = nx;
        nz_rot = ny;
        break;
    }

    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(host_grid.size(), true);
    cuda_utilities::DeviceVector<double> dev_interface_right(host_grid.size(), true);

    // Launch kernel
    switch (direction) {
      case 0:
        hipLaunchKernelGGL(PLM_cuda<0>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx_rot, ny_rot, nz_rot, dx, dt, gamma);
        break;
      case 1:
        hipLaunchKernelGGL(PLM_cuda<1>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx_rot, ny_rot, nz_rot, dx, dt, gamma);
        break;
      case 2:
        hipLaunchKernelGGL(PLM_cuda<2>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx_rot, ny_rot, nz_rot, dx, dt, gamma);
        break;
    }
    GPU_Error_Check();
    GPU_Error_Check(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < host_grid.size(); i++) {
      // Check the left interface
      double test_val = dev_interface_left.at(i);
      double fiducial_val =
          (fiducial_interface_left.at(direction).find(i) == fiducial_interface_left.at(direction).end())
              ? 0.0
              : fiducial_interface_left.at(direction)[i];

      testing_utilities::Check_Results(
          fiducial_val, test_val,
          "left interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));

      // Check the right interface
      test_val     = dev_interface_right.at(i);
      fiducial_val = (fiducial_interface_right.at(direction).find(i) == fiducial_interface_right.at(direction).end())
                         ? 0.0
                         : fiducial_interface_right.at(direction)[i];

      // if (test_val != 0.0) std::cout << "{" << i << ", " << to_string_exact(test_val) << "}," << std::endl;

      testing_utilities::Check_Results(
          fiducial_val, test_val,
          "right interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));
    }
  }
}

TEST(tMHDPlmReconstructor, CorrectInputExpectCorrectOutput)
{
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx = 4, ny = nx, nz = nx;
  size_t const n_fields          = 8;
  size_t const n_cells_grid      = nx * ny * nz * n_fields;
  size_t const n_cells_interface = nx * ny * nz * (n_fields - 1);
  double const dx                = doubleRand(prng);
  double const dt                = doubleRand(prng);
  double const gamma             = 5.0 / 3.0;

  // Setup host grid. Fill host grid with random values and randomly assign maximum value
  std::vector<double> host_grid(n_cells_grid);
  for (Real &val : host_grid) {
    val = doubleRand(prng);
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

  // Fiducial Data
  std::vector<std::unordered_map<int, double>> fiducial_interface_left  = {{{21, 0.59023012197434721},
                                                                            {85, 3.0043379408547275},
                                                                            {149, 2.6320759184913625},
                                                                            {213, 0.9487867623146744},
                                                                            {277, 18.551193003661723},
                                                                            {341, 1.8587936590169301},
                                                                            {405, 2.1583975283044725}},
                                                                           {{21, 0.73640639402573249},
                                                                            {85, 3.3462413154443715},
                                                                            {149, 2.1945584994458125},
                                                                            {213, 0.67418839414138987},
                                                                            {277, 16.909618487528142},
                                                                            {341, 2.1533768050263267},
                                                                            {405, 1.6994195863331925}},
                                                                           {{21, 0.25340904981266843},
                                                                            {85, 2.0441984720128734},
                                                                            {149, 1.9959059157695584},
                                                                            {213, 0.45377591914009824},
                                                                            {277, 23.677832869261188},
                                                                            {341, 1.5437923271692418},
                                                                            {405, 1.8141353672443383}}};
  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{20, 0.59023012197434721},
                                                                            {84, 3.0043379408547275},
                                                                            {148, 2.6320759184913625},
                                                                            {212, 0.9487867623146744},
                                                                            {276, 22.111134849009044},
                                                                            {340, 1.8587936590169301},
                                                                            {404, 2.1583975283044725}},
                                                                           {
                                                                               {17, 0.44405384992296193},
                                                                               {81, 2.5027813113931279},
                                                                               {145, 2.6371119205792346},
                                                                               {209, 1.0210845222961809},
                                                                               {273, 21.360010722689488},
                                                                               {337, 2.1634182515826184},
                                                                               {401, 1.7073441775673177},
                                                                           },
                                                                           {
                                                                               {5, 0.92705119413602599},
                                                                               {69, 1.9592598982258778},
                                                                               {133, 0.96653490574340428},
                                                                               {197, 1.3203867992383289},
                                                                               {261, 8.0057564947791793},
                                                                               {325, 1.8629714367312684},
                                                                               {389, 1.9034519507895218},
                                                                           }};

#ifdef PLMP
  // Change the fiducial data for the PLMP version of this test. Not all elements need to be changed
  fiducial_interface_left.at(1)[21]  = 0.74780807318015607;
  fiducial_interface_left.at(1)[85]  = 2.7423587304689869;
  fiducial_interface_left.at(1)[149] = 2.0740041194407692;
  fiducial_interface_left.at(1)[213] = 1.2020911406758266;
  fiducial_interface_left.at(1)[277] = 14.779549963507595;
  fiducial_interface_left.at(1)[341] = 2.1583975283044725;
  fiducial_interface_left.at(1)[405] = 1.7965456420437496;
  fiducial_interface_left.at(2)[21]  = 0.2849255850177943;
  fiducial_interface_left.at(2)[85]  = 2.2425026758932325;
  fiducial_interface_left.at(2)[149] = 2.2953938046062077;
  fiducial_interface_left.at(2)[213] = 0.57593655962650969;
  fiducial_interface_left.at(2)[277] = 26.253543495427323;
  fiducial_interface_left.at(2)[341] = 1.7033818819502551;
  fiducial_interface_left.at(2)[405] = 1.9082633079592932;

  fiducial_interface_right.at(1)[17]  = 0.43265217076853835;
  fiducial_interface_right.at(1)[81]  = 2.8178764819733342;
  fiducial_interface_right.at(1)[145] = 2.6588068794446604;
  fiducial_interface_right.at(1)[209] = 0.69548238395352202;
  fiducial_interface_right.at(1)[273] = 23.035559922784643;
  fiducial_interface_right.at(1)[337] = 2.1583975283044725;
  fiducial_interface_right.at(1)[401] = 1.6102181218567606;
  fiducial_interface_right.at(2)[5]   = 0.89553465893090012;
  fiducial_interface_right.at(2)[69]  = 2.068452470340659;
  fiducial_interface_right.at(2)[133] = 0.77257393218171133;
  fiducial_interface_right.at(2)[197] = 1.0689231838445987;
  fiducial_interface_right.at(2)[261] = 7.3918889587333432;
  fiducial_interface_right.at(2)[325] = 1.7033818819502551;
  fiducial_interface_right.at(2)[389] = 1.8093240100745669;
#endif  // PLMP

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(n_cells_interface, true);
    cuda_utilities::DeviceVector<double> dev_interface_right(n_cells_interface, true);

    // Launch kernel
    switch (direction) {
      case 0:
        hipLaunchKernelGGL(PLM_cuda<0>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, dx, dt, gamma);
        break;
      case 1:
        hipLaunchKernelGGL(PLM_cuda<1>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, dx, dt, gamma);
        break;
      case 2:
        hipLaunchKernelGGL(PLM_cuda<2>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, dx, dt, gamma);
        break;
    }
    GPU_Error_Check();
    GPU_Error_Check(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < dev_interface_right.size(); i++) {
      // Check the left interface
      double test_val = dev_interface_left.at(i);
      double fiducial_val =
          (fiducial_interface_left.at(direction).find(i) == fiducial_interface_left.at(direction).end())
              ? 0.0
              : fiducial_interface_left.at(direction)[i];

      testing_utilities::Check_Results(
          fiducial_val, test_val,
          "left interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));

      // Check the right interface
      test_val     = dev_interface_right.at(i);
      fiducial_val = (fiducial_interface_right.at(direction).find(i) == fiducial_interface_right.at(direction).end())
                         ? 0.0
                         : fiducial_interface_right.at(direction)[i];

      testing_utilities::Check_Results(
          fiducial_val, test_val,
          "right interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));
    }
  }
}

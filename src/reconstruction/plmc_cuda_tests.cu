/*!
 * \file plmc_cuda_tests.cu
 * \brief Tests for the contents of plmc_cuda.h and plmc_cuda.cu
 *
 */

// STL Includes
#include <random>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include <algorithm>

#include "../global/global.h"
#include "../io/io.h"
#include "../reconstruction/plmc_cuda.h"
#include "../utils/DeviceVector.h"
#include "../utils/testing_utilities.h"

TEST(tHYDROPlmcReconstructor, CorrectInputExpectCorrectOutput)
{
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
  for (size_t i = 0; i < host_grid.size(); i++) {
    host_grid.at(i) = doubleRand(prng);
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

  // Fiducial Data
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {
      {{26, 2.1584359129984056},
       {27, 0.70033864721549188},
       {106, 2.2476363309467553},
       {107, 3.0633780053857027},
       {186, 2.2245934101106259},
       {187, 2.1015872413794123},
       {266, 2.1263341057778309},
       {267, 3.9675148506537838},
       {346, 3.3640057502842691},
       {347, 21.091316282933843}},
      {{21, 0.72430827309279655},  {26, 2.1584359129984056},  {27, 0.70033864721549188}, {37, 0.19457128219588618},
       {101, 5.4739527659741896},  {106, 2.2476363309467553}, {107, 3.0633780053857027}, {117, 4.4286255636679313},
       {181, 0.12703829036056602}, {186, 2.2245934101106259}, {187, 2.1015872413794123}, {197, 2.2851440769830953},
       {261, 1.5337035731959561},  {266, 2.1263341057778309}, {267, 3.9675148506537838}, {277, 2.697375839048191},
       {341, 22.319601655044117},  {346, 3.3640057502842691}, {347, 21.091316282933843}, {357, 82.515887983144168}},
      {{21, 0.72430827309279655},  {25, 2.2863650183226212},  {26, 2.1584359129984056},   {27, 0.70033864721549188},
       {29, 1.686415421301841},    {37, 0.19457128219588618}, {101, 5.4739527659741896},  {105, 0.72340346106443465},
       {106, 2.2476363309467553},  {107, 3.0633780053857027}, {109, 5.4713687086831388},  {117, 4.4286255636679313},
       {181, 0.12703829036056602}, {185, 3.929100145230096},  {186, 2.2245934101106259},  {187, 2.1015872413794123},
       {189, 4.9166140516911483},  {197, 2.2851440769830953}, {261, 1.5337035731959561},  {265, 0.95177493689267167},
       {266, 2.1263341057778309},  {267, 3.9675148506537838}, {269, 0.46056494878491938}, {277, 2.697375839048191},
       {341, 22.319601655044117},  {345, 3.6886096301452787}, {346, 3.3640057502842691},  {347, 21.091316282933843},
       {349, 16.105488797582133},  {357, 82.515887983144168}}};
  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {
      {{25, 3.8877922383184833},
       {26, 0.70033864721549188},
       {105, 1.5947787943675635},
       {106, 3.0633780053857027},
       {185, 4.0069556576401011},
       {186, 2.1015872413794123},
       {265, 1.7883678016935785},
       {266, 3.9675148506537838},
       {345, 2.8032969746372527},
       {346, 21.091316282933843}},
      {{17, 0.43265217076853835},  {25, 3.8877922383184833},  {26, 0.70033864721549188}, {33, 0.19457128219588618},
       {97, 3.2697645945288754},   {105, 1.5947787943675635}, {106, 3.0633780053857027}, {113, 4.4286255636679313},
       {177, 0.07588397666718491}, {185, 4.0069556576401011}, {186, 2.1015872413794123}, {193, 2.2851440769830953},
       {257, 0.91612950577699748}, {265, 1.7883678016935785}, {266, 3.9675148506537838}, {273, 2.697375839048191},
       {337, 13.332201861384396},  {345, 2.8032969746372527}, {346, 21.091316282933843}, {353, 82.515887983144168}},
      {{5, 2.2863650183226212},    {9, 1.686415421301841},    {17, 0.43265217076853835},  {25, 3.8877922383184833},
       {26, 0.70033864721549188},  {33, 0.19457128219588618}, {85, 0.72340346106443465},  {89, 1.7792505446336098},
       {97, 3.2697645945288754},   {105, 1.5947787943675635}, {106, 3.0633780053857027},  {113, 4.4286255636679313},
       {165, 5.3997753452111859},  {169, 1.4379190463124139}, {177, 0.07588397666718491}, {185, 4.0069556576401011},
       {186, 2.1015872413794123},  {193, 2.2851440769830953}, {245, 0.95177493689267167}, {249, 0.46056494878491938},
       {257, 0.91612950577699748}, {265, 1.7883678016935785}, {266, 3.9675148506537838},  {273, 2.697375839048191},
       {325, 6.6889498465051407},  {329, 1.6145084086614281}, {337, 13.332201861384396},  {345, 2.8032969746372527},
       {346, 21.091316282933843},  {353, 82.515887983144168}}};

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
    cuda_utilities::DeviceVector<double> dev_interface_left(host_grid.size());
    cuda_utilities::DeviceVector<double> dev_interface_right(host_grid.size());

    // Launch kernel
    hipLaunchKernelGGL(PLMC_cuda, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                       dev_interface_right.data(), nx_rot, ny_rot, nz_rot, dx, dt, gamma, direction, n_fields);
    CudaCheckError();
    CHECK(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < host_grid.size(); i++) {
      // Check the left interface
      double test_val = dev_interface_left.at(i);
      double fiducial_val =
          (fiducial_interface_left.at(direction).find(i) == fiducial_interface_left.at(direction).end())
              ? 0.0
              : fiducial_interface_left.at(direction)[i];

      testingUtilities::checkResults(
          fiducial_val, test_val,
          "left interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));

      // Check the right interface
      test_val     = dev_interface_right.at(i);
      fiducial_val = (fiducial_interface_right.at(direction).find(i) == fiducial_interface_right.at(direction).end())
                         ? 0.0
                         : fiducial_interface_right.at(direction)[i];

      testingUtilities::checkResults(
          fiducial_val, test_val,
          "right interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));
    }
  }
}

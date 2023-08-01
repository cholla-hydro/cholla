/*!
 * \file ppmc_cuda_tests.cu
 * \brief Tests for the contents of ppmc_cuda.h and ppmc_cuda.cu
 *
 */

// STL Includes
#include <algorithm>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../io/io.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../utils/DeviceVector.h"
#include "../utils/hydro_utilities.h"
#include "../utils/testing_utilities.h"

TEST(tHYDROPpmcCTUReconstructor, CorrectInputExpectCorrectOutput)
{
  // Alwin: skip until this has been fixed
  GTEST_SKIP();
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx       = 7;
  size_t const ny       = 7;
  size_t const nz       = 7;
  size_t const n_fields = 5;
  double const dx       = doubleRand(prng);
  double const dt       = doubleRand(prng);
  double const gamma    = 5.0 / 3.0;

  // Setup host grid. Fill host grid with random values and randomly assign maximum value
  std::vector<double> host_grid(nx * ny * nz * n_fields);
  for (double &val : host_grid) {
    val = doubleRand(prng);
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

  // Fiducial Data
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{171, 1.7598055553475744},
                                                                           {514, 3.3921082637175894},
                                                                           {857, 3.5866056366266772},
                                                                           {1200, 3.4794572581328902},
                                                                           {1543, 10.363861270296034}},
                                                                          {{171, 1.6206985712721598},
                                                                           {514, 3.123972986618837},
                                                                           {857, 3.30309596610488},
                                                                           {1200, 3.204417323222251},
                                                                           {1543, 9.544631281899882}},
                                                                          {{171, 1.6206985712721595},
                                                                           {514, 5.0316428671215876},
                                                                           {857, 2.3915465711497186},
                                                                           {1200, 3.2044173232222506},
                                                                           {1543, 12.74302824034023}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{170, 1.7857012385420896},
                                                                            {513, 3.4420234152477129},
                                                                            {856, 3.6393828329638049},
                                                                            {1199, 3.5306577572855762},
                                                                            {1542, 10.516366339570284}},
                                                                           {{164, 1.6206985712721595},
                                                                            {507, 3.1239729866188366},
                                                                            {850, 3.3030959661048795},
                                                                            {1193, 3.2044173232222506},
                                                                            {1536, 9.5446312818998802}},
                                                                           {{122, 1.6206985712721595},
                                                                            {465, 5.4375307473677061},
                                                                            {808, 2.2442413290889327},
                                                                            {1151, 3.2044173232222506},
                                                                            {1494, 13.843305272338561}}};

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(host_grid.size(), true);
    cuda_utilities::DeviceVector<double> dev_interface_right(host_grid.size(), true);

    // Launch kernel
    hipLaunchKernelGGL(PPMC_CTU, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                       dev_interface_right.data(), nx, ny, nz, dx, dt, gamma, direction);
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

TEST(tALLPpmcVLReconstructor, CorrectInputExpectCorrectOutput)
{
#ifdef DE
  /// This test doesn't support Dual Energy. It wouldn't be that hard to add support for DE but the DE parts of the
  /// reconstructor (loading and PPM_Single_Variable) are well tested elsewhere so there's no need to add the extra
  /// complexity here.
  GTEST_SKIP();
#endif  // DE

  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx    = 7;
  size_t const ny    = 7;
  size_t const nz    = 7;
  double const gamma = 5.0 / 3.0;
#ifdef MHD
  size_t const n_fields = 8;
#else   // not MHD
  size_t const n_fields                                                = 5;
#endif  // MHD

  // Setup host grid. Fill host grid with random values and randomly assign maximum value
  std::vector<double> host_grid(nx * ny * nz * n_fields);
  for (double &val : host_grid) {
    val = doubleRand(prng);
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

// Fiducial Data
#ifdef MHD
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{171, 1.5556846217288991},
                                                                           {514, 1.7422005905354798},
                                                                           {857, 3.6289199464135558},
                                                                           {1200, 2.1487031353407438},
                                                                           {1543, 22.988345461909127},
                                                                           {1886, 3.1027541330860546},
                                                                           {2229, 3.2554981416903335}},
                                                                          {{171, 1.7167767631895592},
                                                                           {514, 1.8447385381907686},
                                                                           {857, 2.9211469103910663},
                                                                           {1200, 2.626030390823102},
                                                                           {1543, 28.84165870179233},
                                                                           {1886, 3.8209152940021962},
                                                                           {2229, 2.7248523895714203}},
                                                                          {{171, 1.421933695280897},
                                                                           {514, 1.2318388818745061},
                                                                           {857, 2.8667822907691818},
                                                                           {1200, 2.1256773710028964},
                                                                           {1543, 15.684026541123352},
                                                                           {1886, 2.3642698195433232},
                                                                           {2229, 2.9207483994866617}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{170, 1.4838721492695441},
                                                                            {513, 1.3797509020377114},
                                                                            {856, 3.223172223924883},
                                                                            {1199, 2.2593969253004111},
                                                                            {1542, 15.634488002075017},
                                                                            {1885, 2.7494588681249819},
                                                                            {2228, 3.2540533219925698}},
                                                                           {{164, 1.4075989434297753},
                                                                            {507, 1.34947711631431},
                                                                            {850, 3.605198021293794},
                                                                            {1193, 1.9244827470895529},
                                                                            {1536, 13.52285212927548},
                                                                            {1879, 2.9568307038177966},
                                                                            {2222, 2.1086380065800636}},
                                                                           {{122, 1.9532382085816002},
                                                                            {465, 2.6860067041011249},
                                                                            {808, 5.1657781029381917},
                                                                            {1151, 2.7811084475444732},
                                                                            {1494, 24.999993264381686},
                                                                            {1837, 2.3090650532529238},
                                                                            {2180, 2.8525500781893642}}};
#else   // not MHD
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {
      {{171, 1.5239648818969727}, {514, 1.658831367400063}, {857, 3.3918153400617137}, {1200, 2.4096936604224304}},
      {{171, 1.5239639282226562}, {514, 1.6246850138898132}, {857, 3.391813217514656}, {1200, 2.3220060950058032}},
      {{171, 1.7062816619873047}, {514, 1.3300289077249516}, {857, 3.5599794228554593}, {1200, 2.5175993972231074}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{135, 6.5824208227997447},
                                                                            {170, 1.5239620208740234},
                                                                            {513, 1.5386557138925041},
                                                                            {856, 3.3918089724205411},
                                                                            {1199, 1.9263881802230425}},
                                                                           {{135, 6.4095055796015963},
                                                                            {164, 1.5239639282226562},
                                                                            {507, 1.5544994569400168},
                                                                            {850, 3.391813217514656},
                                                                            {1193, 2.1017627061702138}},
                                                                           {{122, 1.3893871307373047},
                                                                            {135, 6.0894802934332555},
                                                                            {465, 2.1518846449159135},
                                                                            {808, 3.4792525252435533},
                                                                            {1151, 2.0500250813102903}}};
#endif  // MHD

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(nx * ny * nz * (n_fields - 1), true);
    cuda_utilities::DeviceVector<double> dev_interface_right(nx * ny * nz * (n_fields - 1), true);

    // Launch kernel
    hipLaunchKernelGGL(PPMC_VL, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                       dev_interface_right.data(), nx, ny, nz, gamma, direction);
    CudaCheckError();
    CHECK(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < dev_interface_left.size(); i++) {
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

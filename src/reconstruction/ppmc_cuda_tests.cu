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
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx       = 6;
  size_t const ny       = 6;
  size_t const nz       = 6;
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
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{86, 2.6558981128823214},
                                                                           {302, 0.84399195916314151},
                                                                           {518, 2.2002498722761787},
                                                                           {734, 1.764334292986655},
                                                                           {950, 3.3600925565746804}},
                                                                          {{86, 2.4950488327292639},
                                                                           {302, 0.79287723513518138},
                                                                           {518, 1.7614576990062414},
                                                                           {734, 1.8238574169157304},
                                                                           {950, 3.14294317122161}},
                                                                          {{86, 2.6558981128823214},
                                                                           {302, 0.84399195916314151},
                                                                           {518, 2.0109603398129137},
                                                                           {734, 1.764334292986655},
                                                                           {950, 3.2100231679403066}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{85, 2.6558981128823214},
                                                                            {301, 0.84399195916314151},
                                                                            {517, 1.8381070277226794},
                                                                            {733, 1.764334292986655},
                                                                            {949, 3.0847691079841209}},
                                                                           {{80, 3.1281603739188069},
                                                                            {296, 0.99406757727427164},
                                                                            {512, 1.8732124042412865},
                                                                            {728, 1.6489758692176784},
                                                                            {944, 2.8820015278590443}},
                                                                           {{50, 2.6558981128823214},
                                                                            {266, 0.84399195916314151},
                                                                            {482, 2.0109603398129137},
                                                                            {698, 1.764334292986655},
                                                                            {914, 3.2100231679403066}}};

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(host_grid.size(), true);
    cuda_utilities::DeviceVector<double> dev_interface_right(host_grid.size(), true);

    // Launch kernel
    switch (direction) {
      case 0:
        hipLaunchKernelGGL(PPMC_CTU<0>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, dx, dt, gamma);
        break;
      case 1:
        hipLaunchKernelGGL(PPMC_CTU<1>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, dx, dt, gamma);
        break;
      case 2:
        hipLaunchKernelGGL(PPMC_CTU<2>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, dx, dt, gamma);
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

      testing_utilities::Check_Results(
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
  size_t const nx    = 6;
  size_t const ny    = 6;
  size_t const nz    = 6;
  double const gamma = 5.0 / 3.0;
#ifdef MHD
  size_t const n_fields = 8;
#else   // not MHD
  size_t const n_fields = 5;
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
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{86, 3.6926886385390683},
                                                                           {302, 2.3022467009220993},
                                                                           {518, 2.3207781368125389},
                                                                           {734, 2.6544338753333747},
                                                                           {950, 11.430630157120799},
                                                                           {1166, 0.6428577630032507},
                                                                           {1382, 4.1406925096276597}},
                                                                          {{86, 3.811691682348938},
                                                                           {302, 1.4827993897794758},
                                                                           {518, 2.3955690789476871},
                                                                           {734, 4.06241130448349},
                                                                           {950, 10.552876853630949},
                                                                           {1166, 3.5147238706385471},
                                                                           {1382, 1.2344879085821312}},
                                                                          {{86, 3.1608655959160155},
                                                                           {302, 1.5377824007725194},
                                                                           {518, 0.41798730655927896},
                                                                           {734, 2.2721408530383784},
                                                                           {950, 5.6329522765789646},
                                                                           {1166, 0.84450832590555991},
                                                                           {1382, 1.4279317910797107}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{85, 2.8949509658187838},
                                                                            {301, 0.25766140043685887},
                                                                            {517, 1.8194165731976308},
                                                                            {733, 2.0809921071868756},
                                                                            {949, 8.1315538869542046},
                                                                            {1165, 0.49708185787322312},
                                                                            {1381, 3.2017395511439881}},
                                                                           {{80, 2.8600082827930269},
                                                                            {296, 0.37343415089084014},
                                                                            {512, 1.7974558224423689},
                                                                            {728, 0.94369445956099784},
                                                                            {944, 7.7011501503138504},
                                                                            {1160, 3.5147238706385471},
                                                                            {1376, 1.2344879085821312}},
                                                                           {{50, 3.1608655959160155},
                                                                            {266, 0.32035830490636008},
                                                                            {482, 3.1721881746709815},
                                                                            {698, 2.2721408530383784},
                                                                            {914, 14.017699282483312},
                                                                            {1130, 1.5292690020097823},
                                                                            {1346, -0.12121484974901264}}};
#else   // not MHD
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {
      {{86, 4.155160222900312}, {302, 1.1624633361407897}, {518, 1.6379195998743412}, {734, 2.9868746414179093}},
      {{86, 4.1795874335665655}, {302, 2.1094239978455054}, {518, 2.6811988240843849}, {734, 4.2540957888954054}},
      {{86, 2.1772852940944429}, {302, 0.58167501916840214}, {518, 1.3683785996473696}, {734, 0.40276763592716164}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{54, 3.8655260187947502},
                                                                            {85, 2.6637168309565289},
                                                                            {301, 0.69483650107094164},
                                                                            {517, 2.7558388224532218},
                                                                            {733, 1.9147729154830744}},
                                                                           {{54, 5.7556871317935459},
                                                                            {80, 2.6515032256234021},
                                                                            {296, 0.39344537106429511},
                                                                            {512, 1.6491544916805785},
                                                                            {728, 0.85830485311660487}},
                                                                           {{50, 2.8254070932730269},
                                                                            {54, 2.1884721760267873},
                                                                            {266, 0.75482470285166003},
                                                                            {482, 1.7757096932649317},
                                                                            {698, 3.6101832818706452}}};
#endif  // MHD

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(nx * ny * nz * (n_fields - 1), true);
    cuda_utilities::DeviceVector<double> dev_interface_right(nx * ny * nz * (n_fields - 1), true);

    // Launch kernel
    switch (direction) {
      case 0:
        hipLaunchKernelGGL(PPMC_VL<0>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, gamma);
        break;
      case 1:
        hipLaunchKernelGGL(PPMC_VL<1>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, gamma);
        break;
      case 2:
        hipLaunchKernelGGL(PPMC_VL<2>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, gamma);
        break;
    }
    GPU_Error_Check();
    GPU_Error_Check(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < dev_interface_left.size(); i++) {
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

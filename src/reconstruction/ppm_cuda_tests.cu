/*!
 * \file ppm_cuda_tests.cu
 * \brief Tests for the contents of ppm_cuda.h and ppm_cuda.cu
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
#include "../reconstruction/ppm_cuda.h"
#include "../utils/DeviceVector.h"
#include "../utils/hydro_utilities.h"
#include "../utils/testing_utilities.h"

TEST(tALLPpmReconstructor, CorrectInputExpectCorrectOutput)
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
  #ifdef PPMC
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
  #else   // PPMC
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{86, 3.1608646282711232},
                                                                           {302, 0.84444422521258167},
                                                                           {518, 1.2459789393105685},
                                                                           {734, 2.2721401574613527},
                                                                           {950, 7.7508629541568022},
                                                                           {1166, 0.54567382624989913},
                                                                           {1382, 3.5147238706385462}},
                                                                          {{86, 3.6292858956631076},
                                                                           {302, 1.8316886259802778},
                                                                           {518, 2.2809308293670103},
                                                                           {734, 3.6939841768696002},
                                                                           {950, 10.405768833830281},
                                                                           {1166, 3.5147238706385462},
                                                                           {1382, 1.234487908582131}},
                                                                          {{86, 3.1608646282711232},
                                                                           {302, 0.84444422521258167},
                                                                           {518, 1.9865377887960551},
                                                                           {734, 1.1540870822905045},
                                                                           {950, 4.8971025794015812},
                                                                           {1166, 1.234487908582131},
                                                                           {1382, 0.54567382624989913}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{301, 0.84444422521258167},
                                                                            {85, 3.1608646282711232},
                                                                            {733, 2.2721401574613527},
                                                                            {517, 3.2701799807980008},
                                                                            {949, 10.497902459040514},
                                                                            {1165, 0.54567382624989913},
                                                                            {1381, 3.5147238706385462}},
                                                                           {{80, 2.245959460360242},
                                                                            {296, 0.33326844362749702},
                                                                            {512, 1.4115388872411132},
                                                                            {728, 0.72702830835784316},
                                                                            {944, 7.5422056995631559},
                                                                            {1160, 3.5147238706385462},
                                                                            {1376, 1.234487908582131}},
                                                                           {{50, 3.1608646282711232},
                                                                            {266, 0.84444422521258167},
                                                                            {482, 1.9865377887960551},
                                                                            {698, 4.1768690252280765},
                                                                            {914, 14.823997016980297},
                                                                            {1130, 1.234487908582131},
                                                                            {1346, 0.54567382624989913}}};
  #endif  // PPMC
#else     // not MHD
  #ifdef PPMC
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
  #else   // PPMC
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {
      {{86, 3.1608646282711232}, {302, 0.84444422521258167}, {518, 1.2459789393105685}, {734, 2.2721401574613527}},
      {{86, 3.6292858956631076}, {302, 1.8316886259802778}, {518, 2.2809308293670103}, {734, 3.6939841768696002}},
      {{86, 3.1608646282711232}, {302, 0.84444422521258167}, {518, 1.9865377887960551}, {734, 1.1540870822905045}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{54, 3.4283787020401455},
                                                                            {85, 3.1608646282711232},
                                                                            {301, 0.84444422521258167},
                                                                            {517, 3.2701799807980008},
                                                                            {733, 2.2721401574613527}},
                                                                           {{54, 5.3122571267813665},
                                                                            {80, 2.245959460360242},
                                                                            {296, 0.33326844362749702},
                                                                            {512, 1.4115388872411132},
                                                                            {728, 0.72702830835784316}},
                                                                           {{50, 3.1608646282711232},
                                                                            {54, 3.2010935757366896},
                                                                            {266, 0.84444422521258167},
                                                                            {482, 1.9865377887960551},
                                                                            {698, 4.1768690252280765}}};
  #endif  // PPMC
#endif    // MHD

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(nx * ny * nz * (n_fields - 1), true);
    cuda_utilities::DeviceVector<double> dev_interface_right(nx * ny * nz * (n_fields - 1), true);

    // Launch kernel
    switch (direction) {
      case 0:
        hipLaunchKernelGGL(PPM_cuda<0>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, 0, 0, gamma);
        break;
      case 1:
        hipLaunchKernelGGL(PPM_cuda<1>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, 0, 0, gamma);
        break;
      case 2:
        hipLaunchKernelGGL(PPM_cuda<2>, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                           dev_interface_right.data(), nx, ny, nz, 0, 0, gamma);
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

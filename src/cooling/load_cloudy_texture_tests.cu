/*!
 *  \file
 *  \brief Tests logic pertaining to loading the cloudy table as a texture
 */

#include <filesystem>
#include <string>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

#include "../utils/testing_utilities.h"
#include "cooling_cuda.h"  // coolTexObj, heatTexObj
#include "load_cloudy_texture.h"
#include "texture_utilities.h"

/*! This is a reusable fixture for getting the path to the root of the Cholla directory
 *
 *  The basic idea is to use this fixture when you want to write a test that needs a
 *  file within the cholla directory. The fixture is designed so that if the path to
 *  the cholla directory can't be inferred, the associated test will be skipped.
 *
 *  At the time of writing, the tests will all be skipped if the path to the Cholla
 *  directory isn't explicitly provided on the command line. This fixture is being
 *  created so we can consider relaxing that requirement in the future.
 *
 *  \note Right now, we are just using this for a single test. But, this will start
 *        to grow in the future.
 *
 *  \warning A lot of care needs to be taken if you want to subclass this fixture
 *           (i.e. you need to explicitly call this fixture's SetUp method from the
 *           subclass's SetUp method)
 */
class ChollaRootFixture : public testing::Test
{
 protected:
  std::filesystem::path cholla_root_path_;

  void SetUp() override
  {
    // we choose to set up the fixture in a SetUp method, rather than a constructor
    // purely so that we can skip tests when the path isn't provided on the command line
    std::string tmp = globalChollaRoot.getString();
    if (tmp.empty()) {
      GTEST_SKIP() << "path to the root of the cholla repository is not known";
    }
    cholla_root_path_ = tmp;
  }

  /*! returns the path to the root of the cholla repository */
  const std::filesystem::path& getRootPath() const { return cholla_root_path_; }
};

// define an alias of ChollaRootFixture that is named for the test suite
using tALLLoadCloudyTexture = ChollaRootFixture;

/* Consider this function only to be used at the end of Load_Cuda_Textures when
 * testing Evaluate texture on grid of size num_n num_T for variables n,T */
__global__ void Test_Cloudy_Textures_Kernel(int num_n, int num_T, cudaTextureObject_t coolTexObj,
                                            cudaTextureObject_t heatTexObj)
{
  int id, id_n, id_T;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  // Calculate log_T and log_n based on id
  id_T = id / num_n;
  id_n = id % num_n;

  float grid_offset = 0.1 / 512.0;
  // Min value, but include id=-1 as an outside value to check clamping. Use dx
  // = 0.05 instead of 0.1 to check interpolation
  float log_T = 1.0 + (id_T - 1) * 0.05 + grid_offset;
  float log_n = -6.0 + (id_n - 1) * 0.05 + grid_offset;

  // Remap for texture with normalized coords
  // float rlog_T = (log_T - 1.0) / 8.1;
  // float rlog_n = (log_n + 6.0) / 12.1;

  // Remap for texture without normalized coords
  float rlog_T = (log_T - 1.0) * 10;
  float rlog_n = (log_n + 6.0) * 10;

  // Evaluate
  float lambda = Bilinear_Texture(coolTexObj, rlog_T, rlog_n);  // tex2D<float>(coolTexObj, rlog_T, rlog_n);
  float heat   = Bilinear_Texture(heatTexObj, rlog_T, rlog_n);  // tex2D<float>(heatTexObj, rlog_T, rlog_n);

  // Hackfully print it out for processing for correctness
  printf("TEST_Cloudy: %.17e %.17e %.17e %.17e \n", log_T, log_n, lambda, heat);
}

void Test_Cloudy_Textures()
{
  int num_n = 1 + 2 * 121;
  int num_T = 1 + 2 * 81;
  dim3 dim1dGrid((num_n * num_T + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Test_Cloudy_Textures_Kernel, dim1dGrid, dim1dBlock, 0, 0, num_n, num_T, coolTexObj, heatTexObj);
  GPU_Error_Check(cudaDeviceSynchronize());
}

TEST_F(tALLLoadCloudyTexture, LegacySimple)
{
  std::filesystem::path path = getRootPath() / "src/cooling/cloudy_coolingcurve.txt";
  Load_Cuda_Textures(std::string(path));
  Test_Cloudy_Textures();
  Free_Cuda_Textures();
}

/* Consider this function only to be used at the end of Load_Cuda_Textures when
 * testing Evaluate texture on grid of size num_n num_T for variables n,T */
__global__ void Test_Cloudy_Speed_Kernel(int num_n, int num_T, cudaTextureObject_t coolTexObj,
                                         cudaTextureObject_t heatTexObj)
{
  int id, id_n, id_T;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  // Calculate log_T and log_n based on id
  id_T = id / num_n;
  id_n = id % num_n;

  // Min value, but include id=-1 as an outside value to check clamping. Use dx
  // = 0.05 instead of 0.1 to check interpolation float log_T = 1.0  +
  // (id_T-1)*0.05;
  //  float log_n = -6.0 + (id_n-1)*0.05;

  // Remap for texture with normalized coords
  // float rlog_T = (log_T - 1.0) / 8.1;
  // float rlog_n = (log_n + 6.0) / 12.1;

  // Remap for texture without normalized coords
  // float rlog_T = (log_T - 1.0) * 10;
  // float rlog_n = (log_n + 6.0) * 10;

  float rlog_T = (id_T - 1) * 0.0125;
  float rlog_n = (id_n - 1) * 0.0125;

  // Evaluate
  float lambda = Bilinear_Texture(coolTexObj, rlog_T, rlog_n);  // tex2D<float>(coolTexObj, rlog_T, rlog_n);
  float heat   = Bilinear_Texture(heatTexObj, rlog_T, rlog_n);  // tex2D<float>(heatTexObj, rlog_T, rlog_n);

  // Hackfully print it out for processing for correctness
  // printf("TEST_Cloudy: %.17e %.17e %.17e %.17e \n",log_T, log_n, lambda,
  // heat);
}

void Test_Cloudy_Speed()
{
  int num_n = 1 + 80 * 121;
  int num_T = 1 + 80 * 81;
  dim3 dim1dGrid((num_n * num_T + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  GPU_Error_Check(cudaDeviceSynchronize());
  Real time_start = Get_Time();
  for (int i = 0; i < 100; i++) {
    hipLaunchKernelGGL(Test_Cloudy_Speed_Kernel, dim1dGrid, dim1dBlock, 0, 0, num_n, num_T, coolTexObj, heatTexObj);
  }
  GPU_Error_Check(cudaDeviceSynchronize());
  Real time_end = Get_Time();
  printf(" Cloudy Test Time %9.4f micro-s \n", (time_end - time_start));
}

TEST_F(tALLLoadCloudyTexture, LegacySpeed)
{
  std::filesystem::path path = getRootPath() / "src/cooling/cloudy_coolingcurve.txt";
  Load_Cuda_Textures(std::string(path));
  Test_Cloudy_Speed();
  Free_Cuda_Textures();
}
/*!
 *  \file
 *  \brief Tests logic pertaining to loading the cloudy table as a texture
 */

#include <algorithm>  //std::max
#include <filesystem>
#include <string>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

#include "../utils/error_handling.h"
#include "../utils/gpu.hpp"
#include "../utils/shared.h"
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

/*! A grid of n and T values*/
struct EvaluationGrid {
  int num_n;
  int num_T;
  SharedDevPtr<Real> n_vals;
  SharedDevPtr<Real> T_vals;

  __host__ static EvaluationGrid allocate(int num_n, int num_T)
  {
    CHOLLA_ASSERT(num_n > 0, "num_n must be positive");
    CHOLLA_ASSERT(num_T > 0, "num_T must be positive");

    // define a callback function for deallocating pointers
    auto deleter = [](Real* dev_ptr) {
      GPU_Error_Check(cudaDeviceSynchronize());  // <- this is just to be safe
      GPU_Error_Check(cudaFree(dev_ptr));
    };

    Real* n_ptr;
    GPU_Error_Check(cudaMalloc(&n_ptr, num_n * sizeof(Real)));
    Real* T_ptr;
    GPU_Error_Check(cudaMalloc(&T_ptr, num_T * sizeof(Real)));

    SharedDevPtr<Real> n_vals(n_ptr, deleter);
    SharedDevPtr<Real> T_vals(T_ptr, deleter);
    return {num_n, num_T, std::move(n_vals), std::move(T_vals)};
  }
};

EvaluationGrid setup_grid(bool speed_mode)
{
  if (speed_mode) {
    int num_n                = 1 + 80 * 121;
    int num_T                = 1 + 80 * 81;
    EvaluationGrid eval_grid = EvaluationGrid::allocate(num_n, num_T);
    auto fn                  = [eval_grid] __device__(int i) {
      if (i < eval_grid.num_n) eval_grid.n_vals[i] = pow(10.0, -6.0 + (i - 1) * 0.0125);
      if (i < eval_grid.num_T) eval_grid.T_vals[i] = pow(10.0, 1.0 + (i - 1) * 0.0125);
    };
    gpuFor(std::max(num_n, num_T), fn);
    return eval_grid;
  } else {
    int num_n                = 1 + 2 * 121;
    int num_T                = 1 + 2 * 81;
    EvaluationGrid eval_grid = EvaluationGrid::allocate(num_n, num_T);
    auto fn                  = [eval_grid] __device__(int i) {
      float grid_offset = 0.1 / 512.0;
      // Min value, but include id=-1 as an outside value to check clamping. Use dx
      // = 0.05 instead of 0.1 to check interpolation
      if (i < eval_grid.num_n) eval_grid.n_vals[i] = pow(10.0, -6.0 + (i - 1) * 0.05 + grid_offset);
      if (i < eval_grid.num_T) eval_grid.T_vals[i] = pow(10.0, 1.0 + (i - 1) * 0.05 + grid_offset);
    };
    gpuFor(std::max(num_n, num_T), fn);
    return eval_grid;
  }
}

// define an alias of ChollaRootFixture that is named for the test suite
using tALLLoadCloudyTexture = ChollaRootFixture;

/* Consider this function only to be used at the end of Load_Cuda_Textures when
 * testing Evaluate texture on grid of size num_n num_T for variables n,T */
template <bool SPEED_MODE>
__global__ void Test_Cloudy_Textures_Kernel(EvaluationGrid eval_grid, cudaTextureObject_t coolTexObj,
                                            cudaTextureObject_t heatTexObj)
{
  int id, id_n, id_T;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  // Calculate log_T and log_n based on id
  id_T = id / eval_grid.num_n;
  id_n = id % eval_grid.num_n;

  float log10_T = static_cast<float>(log10(eval_grid.T_vals[id_T]));
  float log10_n = static_cast<float>(log10(eval_grid.n_vals[id_n]));

  // Remap for texture without normalized coords
  float rlog_T = (log10_T - 1.0) * 10;
  float rlog_n = (log10_n + 6.0) * 10;

  // Evaluate
  float lambda = Bilinear_Texture(coolTexObj, rlog_T, rlog_n);  // tex2D<float>(coolTexObj, rlog_T, rlog_n);
  float heat   = Bilinear_Texture(heatTexObj, rlog_T, rlog_n);  // tex2D<float>(heatTexObj, rlog_T, rlog_n);

  if constexpr (not SPEED_MODE) {  // Hackfully print it out for processing for correctness
    printf("TEST_Cloudy: %.17e %.17e %.17e %.17e \n", log10_T, log10_n, lambda, heat);
  }
}

TEST_F(tALLLoadCloudyTexture, LegacySimple)
{
  std::filesystem::path path = getRootPath() / "src/cooling/cloudy_coolingcurve.txt";
  Load_Cuda_Textures(std::string(path));

  // actually run the test calculation
  constexpr bool SPEED_MODE = false;
  EvaluationGrid eval_grid  = setup_grid(SPEED_MODE);
  dim3 dim1dGrid((eval_grid.num_n * eval_grid.num_T + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Test_Cloudy_Textures_Kernel<SPEED_MODE>, dim1dGrid, dim1dBlock, 0, 0, eval_grid, coolTexObj,
                     heatTexObj);
  GPU_Error_Check(cudaDeviceSynchronize());

  // perform cleanup
  Free_Cuda_Textures();
}

TEST_F(tALLLoadCloudyTexture, LegacySpeed)
{
  std::filesystem::path path = getRootPath() / "src/cooling/cloudy_coolingcurve.txt";
  Load_Cuda_Textures(std::string(path));

  // actually run the test calculation
  constexpr bool SPEED_MODE = true;
  EvaluationGrid eval_grid  = setup_grid(SPEED_MODE);
  dim3 dim1dGrid((eval_grid.num_n * eval_grid.num_T + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  GPU_Error_Check(cudaDeviceSynchronize());
  Real time_start = Get_Time();
  for (int i = 0; i < 100; i++) {
    hipLaunchKernelGGL(Test_Cloudy_Textures_Kernel<SPEED_MODE>, dim1dGrid, dim1dBlock, 0, 0, eval_grid, coolTexObj,
                       heatTexObj);
  }
  GPU_Error_Check(cudaDeviceSynchronize());
  Real time_end = Get_Time();
  printf(" Cloudy Test Time %9.4f micro-s \n", (time_end - time_start));

  // perform cleanup
  Free_Cuda_Textures();
}
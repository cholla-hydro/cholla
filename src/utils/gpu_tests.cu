/*! \brief
 *  Tests the \ref gpuFor function
 */

#include <gtest/gtest.h>

#include <algorithm>  // max
#include <vector>

#include "DeviceVector.h"
#include "gpu.hpp"

/*! \brief A callable type that sets each visited location to hold a value of 1
 *
 *  For more context, a "callable" object is sometimes called a "functor." Essentially
 *  a "callable" object carries around state and can be called like a function.
 */
struct MaskFiller {
  /// this is a pointer to an int, rather than a pointer to a bool to make it possible
  /// to manage the memory with a std::vector
  int* mask_ptr;

  /// should be 0, 1, or 2
  int choice;

  __host__ __device__ void operator()(int argA, int argB, int argC) const
  {
    // printf("{%d, %d, %d}, {%d, %d, %d}\n", threadIdx.x, threadIdx.y, threadIdx.z, argA, argB, argC);
    switch (choice) {
      case 0:
        if (argB == 0 and argC == 0) {
          // printf("recording argA=%d\n", argA);
          mask_ptr[argA] = 1;
        }
        break;
      case 1:
        if (argA == 0 and argC == 0) {
          // printf("recording argB=%d\n", argB);
          mask_ptr[argB] = 1;
        }
        break;
      case 2:
        if (argA == 0 and argB == 0) {
          // printf("recording argC=%d\n", argC);
          mask_ptr[argC] = 1;
        }
        break;
    }
  }
};

/*! Return the max index of vec that is nonzero */
static int Max_Index_With_Non_Zero_Val(const std::vector<int>& vec)
{
  int size = vec.size();
  int out  = -1;

  // printf("inferring the max val from {");
  // for (int i = 0; i < size; i++) {
  //   if (i > 0) {
  //     printf(", ");
  //   }
  //   printf("%d", vec[i]);
  // }
  // printf("}\n");
  // fflush(stdout);

  for (int i = 0; i < size; i++) {
    // out = std::max(out, i) could be simplified, but that's ok for us
    if (vec[i] != 0) out = std::max(out, i);
  }
  return out;
}

/*! Returns the max value that is received by a callable passed to gpuFor */
static std::vector<int> Get_Max_Indices(int n0, int n1, int n2)
{
  std::vector<int> out;
  int max_dim = std::max({n0, n1, n2});

  // loop over argument identifiers.
  // -> imagine that MaskFiller::operator()(...)'s arguments got converted to a 3
  //    element array called `argArray`
  // -> each pass through this loop infers the max value passed to `argArray[choice]`
  for (int choice = 0; choice < 3; choice++) {
    // initialize mask so that it holds 0s
    std::vector<int> host_mask(max_dim, 0);  // <- initialize with values of 0
    // use host_mask to set up dev_mask
    cuda_utilities::DeviceVector<int> dev_mask(max_dim);
    dev_mask.cpyHostToDevice(host_mask);
    cudaDeviceSynchronize();

    // perform the operation
    MaskFiller mask_filler{dev_mask.data(), choice};
    gpuFor(n0, n1, n2, mask_filler);
    cudaDeviceSynchronize();

    // copy the values back
    dev_mask.cpyDeviceToHost(host_mask);
    cudaDeviceSynchronize();

    // infer and record max argument
    out.push_back(Max_Index_With_Non_Zero_Val(host_mask));
  }
  return out;
}

TEST(tALLGpuFor, MaxIndexArgsVersionA)
{
  int n0 = 1;
  int n1 = 2;
  int n2 = 3;

  std::vector<int> max_args = Get_Max_Indices(n0, n1, n2);
  EXPECT_EQ(n0, max_args[0] + 1);
  EXPECT_EQ(n1, max_args[1] + 1);
  EXPECT_EQ(n2, max_args[2] + 1);
}

TEST(tALLGpuFor, MaxIndexArgsVersionB)
{
  int n0 = 3;
  int n1 = 2;
  int n2 = 1;

  std::vector<int> max_args = Get_Max_Indices(n0, n1, n2);
  EXPECT_EQ(n0, max_args[0] + 1);
  EXPECT_EQ(n1, max_args[1] + 1);
  EXPECT_EQ(n2, max_args[2] + 1);
}
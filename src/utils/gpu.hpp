#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <experimental/source_location>
#include <iostream>

#include "../utils/error_handling.h"

#ifdef O_HIP

  #include <hip/hip_runtime.h>

  #if defined(PARIS) || defined(PARIS_GALACTIC)

    #include <hipfft.h>

  #endif  // CUFFT PARIS PARIS_GALACTIC

  #define WARPSIZE 64
static constexpr int maxWarpsPerBlock = 1024 / WARPSIZE;

  #define CUFFT_D2Z     HIPFFT_D2Z
  #define CUFFT_FORWARD HIPFFT_FORWARD
  #define CUFFT_INVERSE HIPFFT_BACKWARD
  #define CUFFT_Z2D     HIPFFT_Z2D
  #define CUFFT_Z2Z     HIPFFT_Z2Z
  #define CUFFT_SUCCESS HIPFFT_SUCCESS

  #define cudaDeviceSynchronize              hipDeviceSynchronize
  #define cudaError                          hipError_t
  #define cudaError_t                        hipError_t
  #define cudaErrorInsufficientDriver        hipErrorInsufficientDriver
  #define cudaErrorNoDevice                  hipErrorNoDevice
  #define cudaEvent_t                        hipEvent_t
  #define cudaEventCreate                    hipEventCreate
  #define cudaEventElapsedTime               hipEventElapsedTime
  #define cudaEventRecord                    hipEventRecord
  #define cudaEventSynchronize               hipEventSynchronize
  #define cudaFree                           hipFree
  #define cudaFreeHost                       hipHostFree
  #define cudaGetDevice                      hipGetDevice
  #define cudaGetDeviceCount                 hipGetDeviceCount
  #define cudaGetErrorString                 hipGetErrorString
  #define cudaGetLastError                   hipGetLastError
  #define cudaHostAlloc                      hipHostMalloc
  #define cudaHostAllocDefault               hipHostMallocDefault
  #define cudaMalloc                         hipMalloc
  #define cudaMemcpy                         hipMemcpy
  #define cudaMemcpyAsync                    hipMemcpyAsync
  #define cudaMemcpyPeer                     hipMemcpyPeer
  #define cudaMemcpyDeviceToHost             hipMemcpyDeviceToHost
  #define cudaMemcpyDeviceToDevice           hipMemcpyDeviceToDevice
  #define cudaMemcpyHostToDevice             hipMemcpyHostToDevice
  #define cudaMemGetInfo                     hipMemGetInfo
  #define cudaMemset                         hipMemset
  #define cudaReadModeElementType            hipReadModeElementType
  #define cudaSetDevice                      hipSetDevice
  #define cudaSuccess                        hipSuccess
  #define cudaDeviceProp                     hipDeviceProp_t
  #define cudaGetDeviceProperties            hipGetDeviceProperties
  #define cudaPointerAttributes              hipPointerAttribute_t
  #define cudaPointerGetAttributes           hipPointerGetAttributes
  #define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
  #define cudaMemGetInfo                     hipMemGetInfo
  #define cudaDeviceGetPCIBusId              hipDeviceGetPCIBusId

  // Texture definitions
  #define cudaArray           hipArray
  #define cudaMallocArray     hipMallocArray
  #define cudaFreeArray       hipFreeArray
  #define cudaMemcpyToArray   hipMemcpyToArray
  #define cudaMemcpy2DToArray hipMemcpy2DToArray

  #define cudaTextureObject_t      hipTextureObject_t
  #define cudaCreateTextureObject  hipCreateTextureObject
  #define cudaDestroyTextureObject hipDestroyTextureObject

  #define cudaChannelFormatDesc      hipChannelFormatDesc
  #define cudaCreateChannelDesc      hipCreateChannelDesc
  #define cudaChannelFormatKindFloat hipChannelFormatKindFloat

  #define cudaResourceDesc      hipResourceDesc
  #define cudaResourceTypeArray hipResourceTypeArray
  #define cudaTextureDesc       hipTextureDesc
  #define cudaAddressModeClamp  hipAddressModeClamp
  #define cudaFilterModeLinear  hipFilterModeLinear
  #define cudaFilterModePoint   hipFilterModePoint
  // Texture Definitions
  #define cudaPointerAttributes    hipPointerAttribute_t
  #define cudaPointerGetAttributes hipPointerGetAttributes

  // FFT definitions
  #define cufftDestroy       hipfftDestroy
  #define cufftDoubleComplex hipfftDoubleComplex
  #define cufftDoubleReal    hipfftDoubleReal
  #define cufftExecD2Z       hipfftExecD2Z
  #define cufftExecZ2D       hipfftExecZ2D
  #define cufftExecZ2Z       hipfftExecZ2Z
  #define cufftHandle        hipfftHandle
  #define cufftPlan3d        hipfftPlan3d
  #define cufftPlanMany      hipfftPlanMany

  #define curandStateMRG32k3a_t hiprandStateMRG32k3a_t
  #define curand_init           hiprand_init
  #define curand                hiprand
  #define curand_poisson        hiprand_poisson

#else  // not O_HIP

  #include <cuda_runtime.h>

  #if defined(PARIS) || defined(PARIS_GALACTIC)

    #include <cufft.h>

  #endif  // defined(PARIS) || defined(PARIS_GALACTIC)

  #define WARPSIZE                               32
static constexpr int maxWarpsPerBlock = 1024 / WARPSIZE;
  #define hipLaunchKernelGGL(F, G, B, M, S, ...) F<<<G, B, M, S>>>(__VA_ARGS__)
  #define __shfl_down(...)                       __shfl_down_sync(0xFFFFFFFF, __VA_ARGS__)

#endif  // O_HIP

#define GPU_MAX_THREADS 256

/*!
 * \brief Check for CUDA/HIP error codes. Can be called wrapping a GPU function that returns a value or with no
 * arguments and it will get the latest error code.
 *
 * \param[in] code The code to check. Defaults to the last error code
 * \param[in] abort Whether or not to abort if an error is encountered. Defaults to True
 * \param[in] location The location of the call. This should be left as the default value.
 */
inline void GPU_Error_Check(cudaError_t code = cudaPeekAtLastError(), bool abort = true,
                            std::experimental::source_location location = std::experimental::source_location::current())
{
#ifndef DISABLE_GPU_ERROR_CHECKING
  code = cudaDeviceSynchronize();

  // Check the code
  if (code != cudaSuccess) {
    std::cout << "GPU_Error_Check: Failed at "
              << "Line: " << location.line() << ", File: " << location.file_name()
              << ", Function: " << location.function_name() << ", with code: " << cudaGetErrorString(code) << std::endl;
    if (abort) {
      chexit(code);
    }
  }
#endif  // DISABLE_GPU_ERROR_CHECKING
}

#if defined(PARIS) || defined(PARIS_GALACTIC)
/*!
 * \brief Check for CUFFT/HIPFFT error codes. Can be called wrapping a FFT function that returns a value
 *
 * \param[in] code The code to check
 * \param[in] abort Whether or not to abort if an error is encountered. Defaults to True
 * \param[in] location The location of the call. This should be left as the default value.
 */
inline void GPU_Error_Check(cufftResult_t code, bool abort = true,
                            std::experimental::source_location location = std::experimental::source_location::current())
{
  #ifndef DISABLE_GPU_ERROR_CHECKING
  // Check the code
  if (code != CUFFT_SUCCESS) {
    std::cout << "GPU_Error_Check: Failed at "
              << "Line: " << location.line() << ", File: " << location.file_name()
              << ", Function: " << location.function_name() << ", with FFT code: " << code << std::endl;
    if (abort) {
      chexit(code);
    }
  }
  #endif  // DISABLE_GPU_ERROR_CHECKING
}
#endif  // defined(PARIS) || defined(PARIS_GALACTIC)

#if defined(__CUDACC__) || defined(__HIPCC__)

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun0(const int n0, const F f)
{
  const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i0 < n0) {
    f(i0);
  }
}

template <typename F>
void gpuFor(const int n0, const F f)
{
  if (n0 <= 0) {
    return;
  }
  const int b0 = (n0 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
  const int t0 = (n0 + b0 - 1) / b0;
  gpuRun0<<<b0, t0>>>(n0, f);
  GPU_Error_Check();
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun0x2(const F f)
{
  const int i0 = threadIdx.y;
  const int i1 = threadIdx.x;
  f(i0, i1);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun1x1(const F f)
{
  const int i0 = blockIdx.x;
  const int i1 = threadIdx.x;
  f(i0, i1);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun2x0(const int n1, const F f)
{
  const int i0 = blockIdx.y;
  const int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i1 < n1) {
    f(i0, i1);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const F f)
{
  if ((n0 <= 0) || (n1 <= 0)) {
    return;
  }
  const long nl01 = long(n0) * long(n1);
  assert(nl01 < long(INT_MAX));

  if (n1 > GPU_MAX_THREADS) {
    const int b1 = (n1 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
    const int t1 = (n1 + b1 - 1) / b1;
    gpuRun2x0<<<dim3(b1, n0), dim3(t1)>>>(n1, f);
    GPU_Error_Check();
  } else if (nl01 > GPU_MAX_THREADS) {
    gpuRun1x1<<<n0, n1>>>(f);
    GPU_Error_Check();
  } else {
    gpuRun0x2<<<1, dim3(n1, n0)>>>(f);
    GPU_Error_Check();
  }
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun0x3(const F f)
{
  const int i0 = threadIdx.z;
  const int i1 = threadIdx.y;
  const int i2 = threadIdx.x;
  f(i0, i1, i2);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun1x2(const F f)
{
  const int i0 = blockIdx.x;
  const int i1 = threadIdx.y;
  const int i2 = threadIdx.x;
  f(i0, i1, i2);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun2x1(const F f)
{
  const int i0 = blockIdx.y;
  const int i1 = blockIdx.x;
  const int i2 = threadIdx.x;
  f(i0, i1, i2);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun3x0(const int n2, const F f)
{
  const int i0 = blockIdx.z;
  const int i1 = blockIdx.y;
  const int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i2 < n2) {
    f(i0, i1, i2);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, const F f)
{
  if ((n0 <= 0) || (n1 <= 0) || (n2 <= 0)) {
    return;
  }
  const long nl12  = long(n1) * long(n2);
  const long nl012 = long(n0) * nl12;
  assert(nl012 < long(INT_MAX));

  if (n2 > GPU_MAX_THREADS) {
    const int b2 = (n2 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
    const int t2 = (n2 + b2 - 1) / b2;
    gpuRun3x0<<<dim3(b2, n1, n0), t2>>>(n2, f);
    GPU_Error_Check();
  } else if (nl12 > GPU_MAX_THREADS) {
    gpuRun2x1<<<dim3(n1, n0), n2>>>(f);
    GPU_Error_Check();
  } else if (nl012 > GPU_MAX_THREADS) {
    gpuRun1x2<<<n0, dim3(n2, n1)>>>(f);
    GPU_Error_Check();
  } else {
    gpuRun0x3<<<1, dim3(n2, n1, n0)>>>(f);
    GPU_Error_Check();
  }
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun1x3(const F f)
{
  const int i0 = blockIdx.x;
  const int i1 = threadIdx.z;
  const int i2 = threadIdx.y;
  const int i3 = threadIdx.x;
  f(i0, i1, i2, i3);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun2x2(const F f)
{
  const int i0 = blockIdx.y;
  const int i1 = blockIdx.x;
  const int i2 = threadIdx.y;
  const int i3 = threadIdx.x;
  f(i0, i1, i2, i3);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun3x1(const F f)
{
  const int i0 = blockIdx.z;
  const int i1 = blockIdx.y;
  const int i2 = blockIdx.x;
  const int i3 = threadIdx.x;
  f(i0, i1, i2, i3);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun4x0(const int n23, const int n3, const F f)
{
  const int i23 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i23 < n23) {
    const int i0 = blockIdx.z;
    const int i1 = blockIdx.y;
    const int i2 = i23 / n3;
    const int i3 = i23 % n3;
    f(i0, i1, i2, i3);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, const int n3, const F f)
{
  if ((n0 <= 0) || (n1 <= 0) || (n2 <= 0) || (n3 <= 0)) {
    return;
  }
  const long n23_long  = long(n2) * long(n3);
  const long n123_long = long(n1) * n23_long;
  assert(long(n0) * n123_long < long(INT_MAX));

  const int n23  = int(n23_long);
  const int n123 = int(n123_long);
  if (n3 > GPU_MAX_THREADS) {
    const int b23 = (n23 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
    const int t23 = (n23 + b23 - 1) / b23;
    gpuRun4x0<<<dim3(b23, n1, n0), t23>>>(n23, n3, f);
    GPU_Error_Check();
  } else if (n23 > GPU_MAX_THREADS) {
    gpuRun3x1<<<dim3(n2, n1, n0), n3>>>(f);
    GPU_Error_Check();
  } else if (n123 > GPU_MAX_THREADS) {
    gpuRun2x2<<<dim3(n1, n0), dim3(n3, n2)>>>(f);
    GPU_Error_Check();
  } else {
    gpuRun1x3<<<n0, dim3(n3, n2, n1)>>>(f);
    GPU_Error_Check();
  }
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun2x3(const F f)
{
  const int i0 = blockIdx.y;
  const int i1 = blockIdx.x;
  const int i2 = threadIdx.z;
  const int i3 = threadIdx.y;
  const int i4 = threadIdx.x;
  f(i0, i1, i2, i3, i4);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun3x2(const F f)
{
  const int i0 = blockIdx.z;
  const int i1 = blockIdx.y;
  const int i2 = blockIdx.x;
  const int i3 = threadIdx.y;
  const int i4 = threadIdx.x;
  f(i0, i1, i2, i3, i4);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun4x1(const int n1, const F f)
{
  const int i01 = blockIdx.z;
  const int i0  = i01 / n1;
  const int i1  = i01 % n1;
  const int i2  = blockIdx.y;
  const int i3  = blockIdx.x;
  const int i4  = threadIdx.x;
  f(i0, i1, i2, i3, i4);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun5x0(const int n1, const int n34, const int n4, const F f)
{
  const int i34 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i34 < n34) {
    const int i01 = blockIdx.z;
    const int i0  = i01 / n1;
    const int i1  = i01 % n1;
    const int i2  = blockIdx.y;
    const int i3  = i34 / n4;
    const int i4  = i34 % n4;
    f(i0, i1, i2, i3, i4);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, const int n3, const int n4, const F f)
{
  if ((n0 <= 0) || (n1 <= 0) || (n2 <= 0) || (n3 <= 0) || (n4 <= 0)) {
    return;
  }
  const long nl01 = long(n0) * long(n1);
  const long nl34 = long(n3) * long(n4);
  assert(nl01 * long(n2) * nl34 < long(INT_MAX));

  const int n34 = int(nl34);
  if (n4 > GPU_MAX_THREADS) {
    const int n01 = int(nl01);
    const int b34 = (n34 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
    const int t34 = (n34 + b34 - 1) / b34;
    gpuRun5x0<<<dim3(b34, n2, n01), t34>>>(n1, n34, n4, f);
    GPU_Error_Check();
  } else if (n34 > GPU_MAX_THREADS) {
    const int n01 = n0 * n1;
    gpuRun4x1<<<dim3(n3, n2, n01), n4>>>(n1, f);
    GPU_Error_Check();
  } else if (n2 * n34 > GPU_MAX_THREADS) {
    gpuRun3x2<<<dim3(n2, n1, n0), dim3(n4, n3)>>>(f);
    GPU_Error_Check();
  } else {
    gpuRun2x3<<<dim3(n1, n0), dim3(n4, n3, n2)>>>(f);
    GPU_Error_Check();
  }
}

  #define GPU_LAMBDA [=] __device__

#endif

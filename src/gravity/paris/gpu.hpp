#pragma once

#include <cstdio>

static void check(const cufftResult err, const char *const file, const int line)
{
  if (err == CUFFT_SUCCESS) return;
  fprintf(stderr,"CUFFT ERROR AT LINE %d OF FILE '%s': %d\n",line,file,err);
  fflush(stderr);
  exit(err);
}

static void check(const cudaError_t err, const char *const file, const int line)
{
  if (err == cudaSuccess) return;
  fprintf(stderr,"CUDA ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,cudaGetErrorName(err),cudaGetErrorString(err));
  fflush(stderr);
  exit(err);
}

#define CHECK(X) check(X,__FILE__,__LINE__)

constexpr long GPU_THREADS = 256;

template <typename F>
__global__ void gpuRun(F f, const long n0)
{
  const long i0 = threadIdx.x+long(blockDim.x)*blockIdx.x;
  if (i0 < n0) f(i0);
}

template <typename F>
void gpuFor(const long n0, F f)
{
  const long nb = (n0+GPU_THREADS-1)/GPU_THREADS;
  gpuRun<<<dim3(nb),dim3(GPU_THREADS)>>>(f,n0);
  CHECK(cudaGetLastError());
}

template <typename F>
__global__ void gpuRun(F f, const long n01, const long n1)
{
  const long i01 = threadIdx.x+long(blockDim.x)*blockIdx.x;
  if (i01 < n01) {
    const long i0 = i01/n1;
    const long i1 = i01-i0*n1;
    f(i01,i0,i1);
  }
}

template <typename F>
void gpuFor(const long n0, const long n1, F f)
{
  const long n01 = n0*n1;
  const long nb = (n01+GPU_THREADS-1)/GPU_THREADS;
  gpuRun<<<dim3(nb),dim3(GPU_THREADS)>>>(f,n01,n1);
  CHECK(cudaGetLastError());
}

template <typename F>
__global__ void gpuRun(F f, const long n012, const long n12, const int n2)
{
  const long i012 = threadIdx.x+long(blockDim.x)*blockIdx.x;
  if (i012 < n012) {
    long i = i012;
    const int i0 = i/n12;
    i -= i0*n12;
    const int i1 = i/n2;
    const int i2 = i-i1*n2;
    f(i012,i0,i1,i2);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, F f)
{
  const long n12 = long(n1)*n2;
  const long n012 = n0*n12;
  const long nb = (n012+GPU_THREADS-1)/GPU_THREADS;
  gpuRun<<<dim3(nb),dim3(GPU_THREADS)>>>(f,n012,n12,n2);
  CHECK(cudaGetLastError());
}

template <typename F>
__global__ void gpuRun(F f, const long n0123, const long n123, const long n23, const int n3)
{
  const long i0123 = threadIdx.x+long(blockDim.x)*blockIdx.x;
  if (i0123 < n0123) {
    long i = i0123;
    const int i0 = i/n123;
    i -= i0*n123;
    const int i1 = i/n23;
    i -= i1*n23;
    const int i2 = i/n3;
    const int i3 = i-i2*n3;
    f(i0123,i0,i1,i2,i3);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, const int n3, F f)
{
  const long n23 = long(n2)*n3;
  const long n123 = n1*n23;
  const long n0123 = n0*n123;
  const long nb = (n0123+GPU_THREADS-1)/GPU_THREADS;
  gpuRun<<<dim3(nb),dim3(GPU_THREADS)>>>(f,n0123,n123,n23,n3);
  CHECK(cudaGetLastError());
}

template <typename F>
__global__ void gpuRun(F f, const long n01234, const long n1234, const long n234, const long n34, const int n4)
{
  const long i01234 = threadIdx.x+long(blockDim.x)*blockIdx.x;
  if (i01234 < n01234) {
    long i = i01234;
    const int i0 = i/n1234;
    i -= i0*n1234;
    const int i1 = i/n234;
    i -= i1*n234;
    const int i2 = i/n34;
    i -= i2*n34;
    const int i3 = i/n4;
    const int i4 = i-i3*n4;
    f(i01234,i0,i1,i2,i3,i4);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, const int n3, const int n4, F f)
{
  const long n34 = long(n3)*n4;
  const long n234 = n2*n34;
  const long n1234 = n1*n234;
  const long n01234 = n0*n1234;
  const long nb = (n01234+GPU_THREADS-1)/GPU_THREADS;
  gpuRun<<<dim3(nb),dim3(GPU_THREADS)>>>(f,n01234,n1234,n234,n34,n4);
  CHECK(cudaGetLastError());
}

#define GPU_LAMBDA [=] __device__

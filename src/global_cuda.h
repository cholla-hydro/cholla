/*! /file global_cuda.h
 *  /brief Declarations of global variables and functions for the cuda kernels. */

#ifdef CUDA

#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include"global.h"


#ifndef GLOBAL_CUDA_H
#define GLOBAL_CUDA_H

#define TPB 128 // threads per block
//#define TPB 64


// Define this to turn on error checking
// (defined in the makefile)
//#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



/*! \fn Real minof3(Real a, Real b, Real c)
 *  \brief Returns the minimum of three floating point numbers. */
__device__ inline Real minof3(Real a, Real b, Real c)
{
  return fmin(a, fmin(b,c));
}



/*! \fn int sgn_CUDA
 *  \brief Mathematical sign function. Returns sign of x. */
__device__ inline int sgn_CUDA(Real x)
{
  if (x < 0) return -1;
  else return 1;
}


__global__ void test_function();



#endif //GLOBAL_CUDA_H

#endif //CUDA

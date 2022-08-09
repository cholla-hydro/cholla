/*! /file global_cuda.h
 *  /brief Declarations of global variables and functions for the cuda kernels. */

#ifdef CUDA

#include <stdlib.h>
#include <stdio.h>
#include "../utils/gpu.hpp"
#include <math.h>
#include "../global/global.h"


#ifndef GLOBAL_CUDA_H
#define GLOBAL_CUDA_H

#define TPB 256 // threads per block
//#define TPB 64


extern bool memory_allocated; // Flag becomes true after allocating the memory on the first timestep

// Arrays are global so that they can be allocated only once.
// Not all arrays will be allocated for every integrator
// GPU arrays
// conserved variables
extern Real *dev_conserved, *dev_conserved_half;
// input states and associated interface fluxes (Q* and F* from Stone, 2008)
extern Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;

// Scalar for storing device side hydro/MHD time steps
extern Real *dev_dti;

// array of inverse timesteps for dt calculation (brought back by Alwin May 24 2022)
extern Real *host_dti_array;
extern Real *dev_dti_array;

//Arrays for potential in GPU: Will be set to NULL if not using GRAVITY
extern Real *dev_grav_potential;
extern Real *temp_potential;
extern Real *buffer_potential;

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


/*! \fn int sgn_CUDA
 *  \brief Mathematical sign function. Returns sign of x. */
__device__ inline int sgn_CUDA(Real x)
{
  if (x < 0) return -1;
  else return 1;
}


#endif //GLOBAL_CUDA_H

#endif //CUDA

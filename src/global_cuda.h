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

#define TPB 256 // threads per block
//#define TPB 64


extern bool memory_allocated; // Flag becomes true after allocating the memory on the first timestep
extern bool block_size; // Flag becomes true after determining subgrid block size on the first timestep

// Arrays are global so that they can be allocated only once.
// Not all arrays will be allocated for every integrator
// GPU arrays
// conserved variables
extern Real *dev_conserved, *dev_conserved_half;
// input states and associated interface fluxes (Q* and F* from Stone, 2008)
extern Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;
// array of inverse timesteps for dt calculation
extern Real *dev_dti_array;
#ifdef COOLING_GPU
// array of timesteps for dt calculation (cooling restriction)
extern Real *dev_dt_array;
#endif  
// Array on the CPU to hold max_dti returned from each thread block
extern Real *host_dti_array;
#ifdef COOLING_GPU
extern Real *host_dt_array;
#endif
// Buffer to copy conserved variable blocks to/from
extern Real *buffer;
// Pointers for the location to copy from and to
extern Real *tmp1;
extern Real *tmp2;

// Similarly, sizes of subgrid blocks and kernel dimensions are global variables
// so subgrid splitting function is only called once
// dimensions of subgrid blocks
extern int nx_s, ny_s, nz_s; 
// x, y, and z offsets for subgrid blocks
extern int x_off_s, y_off_s, z_off_s;
// total number of subgrid blocks needed
extern int block_tot;
// number of subgrid blocks needed in each direction
extern int block1_tot, block2_tot, block3_tot;
// modulus of number of cells after block subdivision in each direction
extern int remainder1, remainder2, remainder3;
// number of cells in one subgrid block
extern int BLOCK_VOL;
// dimensions for the GPU grid
extern int ngrid;


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

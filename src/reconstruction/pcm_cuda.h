/*! \file pcm_cuda.h
 *  \brief Declarations of the cuda pcm kernels */

#ifdef CUDA

  #ifndef PCM_CUDA_H
    #define PCM_CUDA_H

__global__ void PCM_Reconstruction_1D(Real *dev_conserved, Real *dev_bounds_L,
                                      Real *dev_bounds_R, int n_cells,
                                      int n_ghost, Real gamma, int n_fields);

__global__ void PCM_Reconstruction_2D(Real *dev_conserved, Real *dev_bounds_Lx,
                                      Real *dev_bounds_Rx, Real *dev_bounds_Ly,
                                      Real *dev_bounds_Ry, int nx, int ny,
                                      int n_ghost, Real gamma, int n_fields);

__global__ void PCM_Reconstruction_3D(Real *dev_conserved, Real *dev_bounds_Lx,
                                      Real *dev_bounds_Rx, Real *dev_bounds_Ly,
                                      Real *dev_bounds_Ry, Real *dev_bounds_Lz,
                                      Real *dev_bounds_Rz, int nx, int ny,
                                      int nz, int n_ghost, Real gamma,
                                      int n_fields);

  #endif  // PCM_CUDA_H
#endif    // CUDA

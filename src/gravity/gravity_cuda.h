/*! \file gravity_cuda.h
 *  \brief Declarations of functions used to calculate gravitational accelerations. */

#ifdef CUDA
#ifndef GRAVITY_CUDA_H
#define GRAVITY_CUDA_H

#include "../global/global.h"


__device__ void calc_g_1D(int xid, int x_off, int n_ghost, Real dx, Real xbound, Real *gx);

__device__ void calc_g_2D(int xid, int yid, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real *gx, Real *gy);

__device__ void calc_g_3D(int xid, int yid, int zid, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real *gx, Real *gy, Real *gz);

#endif // GRAVITY_CUDA_H
#endif // CUDA

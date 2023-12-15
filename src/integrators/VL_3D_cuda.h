/*! \file VL_3D_cuda.h
 *  \brief Declarations for the cuda version of the 3D VL algorithm. */

#ifndef VL_3D_CUDA_H
#define VL_3D_CUDA_H

#include "../global/global.h"

void VL_Algorithm_3D_CUDA(Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
                          int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound,
                          Real dt, int n_fields, int custom_grav, Real density_floor, Real U_floor,
                          Real *host_grav_potential);

void Free_Memory_VL_3D();

#endif  // VL_3D_CUDA_H

/*! \file CTU_3D_cuda.h
 *  \brief Declarations for the cuda version of the 3D CTU algorithm. */

#ifdef CUDA

#ifndef CTU_3D_CUDA_H
#define CTU_3D_CUDA_H

#include "../global/global.h"

void CTU_Algorithm_3D_CUDA(Real *d_conserved, int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt, int n_fields, Real density_floor, Real U_floor, Real *host_grav_potential  );

void Free_Memory_CTU_3D();

#endif //CTU_3D_CUDA_H
#endif //CUDA

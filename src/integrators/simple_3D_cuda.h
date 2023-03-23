/*! \file simple_3D_cuda.h
 *  \brief Declarations for the cuda version of the 3D simple algorithm. */

#ifdef CUDA

  #ifndef SIMPLE_3D_CUDA_H
    #define SIMPLE_3D_CUDA_H

    #include "../chemistry_gpu/chemistry_gpu.h"
    #include "../global/global.h"

void Simple_Algorithm_3D_CUDA(Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
                              int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound,
                              Real dt, int n_fields, Real density_floor, Real U_floor, Real *host_grav_potential);

void Free_Memory_Simple_3D();

  #endif  // SIMPLE_3D_CUDA_H
#endif    // CUDA

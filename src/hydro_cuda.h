/*! \file hydro_cuda.h
 *  \brief Declarations of functions used in all cuda integration algorithms. */

#ifdef CUDA
#ifndef HYDRO_CUDA_H
#define HYDRO_CUDA_H

#include"global.h"


__global__ void Update_Conserved_Variables_1D(Real *dev_conserved, Real *dev_F, int n_cells, int x_off, int n_ghost, 
                                              Real dx, Real dt, Real gamma);

__global__ void Update_Conserved_Variables_1D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F, 
                                                   int n_cells, int n_ghost, Real dx, Real dt, Real gamma);


__global__ void Update_Conserved_Variables_2D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny, 
                                              int x_off, int y_off, int n_ghost, Real dx, Real dy, Real dt, Real gamma);

__global__ void Update_Conserved_Variables_2D_half(Real *dev_conserved, Real *dev_conserved_half, 
                                                   Real *dev_F_x, Real *dev_F_y, int nx, int ny,
                                                   int n_ghost, Real dx, Real dy, Real dt, Real gamma);



__global__ void Update_Conserved_Variables_3D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int x_off, int y_off, int z_off,
                                              int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma);


__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma);




__global__ void Calc_dt_1D(Real *dev_conserved, int n_cells, int n_ghost, Real dx, Real *dti_array, Real gamma);


__global__ void Calc_dt_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real dx, Real dy, Real *dti_array, Real gamma);


__global__ void Calc_dt_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real *dti_array, Real gamma);


__global__ void Sync_Energies_1D(Real *dev_conserved, int n_cells, int n_ghost, Real gamma);


__global__ void Sync_Energies_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real gamma);


__global__ void Sync_Energies_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real gamma);

__device__ void calc_g_2D(int xid, int yid, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real *gx, Real *gy);


#endif //HYDRO_CUDA_H
#endif //CUDA

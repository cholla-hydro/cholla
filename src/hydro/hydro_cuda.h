/*! \file hydro_cuda.h
 *  \brief Declarations of functions used in all cuda integration algorithms. */

#ifdef CUDA
#ifndef HYDRO_CUDA_H
#define HYDRO_CUDA_H

#include "../global/global.h"


__global__ void Update_Conserved_Variables_1D(Real *dev_conserved, Real *dev_F, int n_cells, int x_off, int n_ghost, Real dx, Real xbound, Real dt, Real gamma, int n_fields);


__global__ void Update_Conserved_Variables_2D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real dt, Real gamma, int n_fields);


__global__ void Update_Conserved_Variables_3D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly, Real *Q_Ry, Real *Q_Lz, Real *Q_Rz, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z, int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt, Real gamma, int n_fields, Real density_floor, Real *dev_potential );


__global__ void Calc_dt_1D(Real *dev_conserved, int n_cells, int n_ghost, Real dx, Real *dti_array, Real gamma);


__global__ void Calc_dt_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real dx, Real dy, Real *dti_array, Real gamma);


__global__ void Calc_dt_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real *dti_array, Real gamma,  Real max_dti_slow);


__global__ void Sync_Energies_1D(Real *dev_conserved, int nx, int n_ghost, Real gamma, int n_fields);


__global__ void Sync_Energies_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real gamma, int n_fields);


__global__ void Sync_Energies_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real gamma, int n_fields);

#ifdef TEMPERATURE_FLOOR
__global__ void Apply_Temperature_Floor(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,  Real U_floor );
#endif

static inline __host__ __device__ Real Get_Pressure_From_DE( Real E, Real U_total, Real U_advected, Real gamma ){

  Real U, P;
  Real eta = DE_ETA_1;

  // Apply same condition as Byan+2013 to select the internal energy from which compute pressure.
  if( U_total / E > eta ) U = U_total;
  else U = U_advected;

  P = U * (gamma - 1.0);
  return P;
}


__global__ void Partial_Update_Advected_Internal_Energy_1D( Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, int nx, int n_ghost, Real dx, Real dt, Real gamma, int n_fields );

__global__ void Partial_Update_Advected_Internal_Energy_2D( Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly, Real *Q_Ry, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, Real gamma, int n_fields );

__global__ void Partial_Update_Advected_Internal_Energy_3D( Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly, Real *Q_Ry, Real *Q_Lz, Real *Q_Rz, int nx, int ny, int nz,  int n_ghost, Real dx, Real dy, Real dz,  Real dt, Real gamma, int n_fields );

__global__ void Select_Internal_Energy_1D( Real *dev_conserved, int nx, int n_ghost, int n_fields );

__global__ void Select_Internal_Energy_2D( Real *dev_conserved, int nx, int ny, int n_ghost, int n_fields );

__global__ void Select_Internal_Energy_3D( Real *dev_conserved, int nx, int ny, int nz,  int n_ghost, int n_fields );

__device__ void Average_Cell_All_Fields( int i, int j, int k, int nx, int ny, int nz, int ncells, Real *conserved );

__device__ Real Average_Cell_Single_Field( int field_indx, int i, int j, int k, int nx, int ny, int nz, int ncells, Real *conserved );


#endif //HYDRO_CUDA_H
#endif //CUDA

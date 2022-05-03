#ifndef DUST_CUDA_H
#define DUST_CUDA_H

#include<math.h>

void Dust_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, Real *params_dev);

__global__ void Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, Real *params_dev);

void Conserved_Init(Real *host_conserved, Real rho, Real vx, Real vy, Real vz, Real P, Real rho_dust, Real gamma, int n_cells, int nx, int ny, int nz, int n_ghost, int n_fields);

__device__ Real calc_tau_sp(Real n, Real T);

__device__ Real calc_dd_dt(Real d_dust, Real tau_sp);

#endif // DUST_CUDA_H
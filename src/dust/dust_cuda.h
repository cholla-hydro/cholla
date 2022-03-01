#ifdef CUDA
#ifdef DUST_GPU

#ifndef DUST_CUDA_H
#define DUST_CUDA_H

#include"gpu.hpp"
#include<math.h>
#include"global.h"

__global__ void dust_kernel(Real *dev_conserved, int nx, int ny, int nz, 
  int n_ghost, int n_fields, Real dt, Real gamma, Real *dt_array);

__device__ Real d_gas_accretion(Real T, Real d_gas, Real d_dust, 
  Real d_metal);

__device__ Real thermal_sputtering(Real T, Real d_dust);

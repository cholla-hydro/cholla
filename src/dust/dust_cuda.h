/*! \file dust_cuda.h
 *  \brief Declarations of dust functions. */

#ifdef CUDA
#ifdef DUST_GPU

#ifndef DUST_CUDA_H
#define DUST_CUDA_H

#include"gpu.hpp"
#include<math.h>
#include"global.h"

__global__ void dust_kernel(Real *dev_conserved, int nx, int ny, int nz, 
int n_ghost, int n_fields, Real dt, Real gamma, Real *dt_array);

// general purpose functions:
__device__ void Get_Indices(int n_ghost, int nx, int ny, int nz, int &is, int &ie, int &js, int &je, int &ks, int &ke);

__device__ void Get_GTID(int &id, int &xid, int &yid, int &zid, int &tid, int nx, int ny, int nz);

__device__ Real Calc_Pressure(Real E, Real d_gas, Real vx, Real vy, Real vz, 
Real gamma);

__device__ Real Calc_Temp(Real p, Real n);

#ifdef DE
__device__ Real Calc_Temp_DE(Real d_gas, Real ge, Real gamma, Real n);
#endif // DE

class Dust: {

  public:
    Real T, n, dt, d_gas, d_dust;
    Real tau_sp;
    Dust(Real T_in, Real n_in, Real dt_in, Real d_gas_in, Real d_dust_in) {
      T = T_in;
      n = n_in;
      dt = dt_in;
      d_gas = d_gas_in;
      d_dust = d_dust_in;
    }
    void calc_tau_sp();
    Real calc_dd_dt();

  private:
    Real MP = 1.6726*pow(10,-24); // proton mass in g
    Real YR_IN_S = 3.154*pow(10,7); // one year in s

};

#endif // DUST_CUDA_H
#endif // DUST_GPU
#endif // CUDA

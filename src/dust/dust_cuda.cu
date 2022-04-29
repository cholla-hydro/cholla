#ifdef CUDA
#ifdef SCALAR

#include "dust_model.h"

#include <cstdio>
#include<stdio.h>
#include <fstream>

#include <vector>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"
#include "../utils/cuda_utilities.h"
#include "../grid/grid3D.h"

int main() {
  Conserved_Init(host_conserved, rho, vx, vy, vz, P, rho_d, gamma, k_n_cells, k_nx, k_ny, k_nz, k_n_ghost, k_n_fields);
}
  

void Dust_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma) {
    dim3 dim1dGrid(k_ngrid, 1, 1);
    dim3 dim1dBlock(TPB, 1, 1);
    hipLaunchKernelGGL(Dust_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dt, gamma, params_dev);
    CudaCheckError();  
}

__global__ void Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma) {
    //__shared__ Real min_dt[TPB];
    // get grid indices
    Real const K = 1e30;
    int n_cells = nx * ny * nz;
    int is, ie, js, je, ks, ke;
    cuda_utilities::Get_Real_Indices(n_ghost, nx, ny, nz, is, ie, js, je, ks, ke);
    // get a global thread ID
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int id = threadIdx.x + blockId * blockDim.x;
    int zid = id / (nx * ny);
    int yid = (id - zid * nx * ny) / nx;
    int xid = id - zid * nx * ny - yid * nx;
    // add a thread id within the block 

    // define physics variables
    Real d_gas, d_dust; // fluid mass densities
    Real n = 1; // gas number density
    Real T, E, P; // temperature, energy, pressure
    Real vx, vy, vz; // velocities
    #ifdef DE
    Real ge;
    #endif // DE

    dt *= 3.154e7; // in seconds

    // define integration variables
    Real dd_dt; // instantaneous rate of change in dust density
    Real dd; // change in dust density at current time-step
    Real dd_max = 0.01; // allowable percentage of dust density increase
    Real dt_sub; //refined timestep

    if (xid >= is && xid < ie && yid >= js && yid < je && zid >= ks && zid < ke) {
        // get quantities from dev_conserved
        d_gas = dev_conserved[id];
        //d_dust = dev_conserved[5*n_cells + id];
        d_dust = dev_conserved[5*n_cells + id];
        E = dev_conserved[4*n_cells + id];
        //printf("kernel: %7.4e\n", d_dust);
        // make sure thread hasn't crashed

        // multiply small values by arbitrary constant to preserve precision
        d_gas *= K;
        d_dust *= K;

        if (E < 0.0 || E != E) return;
        
        vx = dev_conserved[1*n_cells + id] / d_gas;
        vy = dev_conserved[2*n_cells + id] / d_gas;
        vz = dev_conserved[3*n_cells + id] / d_gas;

        #ifdef DE
        ge = dev_conserved[(n_fields-1)*n_cells + id] / d_gas;
        ge = fmax(ge, (Real) TINY_NUMBER);
        #endif // DE

        // calculate physical quantities
        P = hydro_utilities::Calc_Pressure_Primitive(E, d_gas, vx, vy, vz, gamma);

        Real T_init;
        T_init = hydro_utilities::Calc_Temp(P, n);

        #ifdef DE
        T_init = hydro_utilities::Calc_Temp_DE(d_gas, ge, gamma, n);
        #endif // DE

        T = T_init;

        Real tau_sp = calc_tau_sp(n, T);

        dd_dt = calc_dd_dt(d_dust, tau_sp);
        dd = dd_dt * dt;

        params_dev[0] = T;
        params_dev[1] = n;
        params_dev[2] = tau_sp/3.154e7;
        params_dev[3] = dd_dt;
        params_dev[4] = dd; 

        // ensure that dust density is not changing too rapidly
        bool time_refine = false;
        while (dd/d_dust > dd_max) {
            time_refine = true;
            dt_sub = dd_max * d_dust / dd_dt;
            d_dust += dt_sub * dd_dt;
            dt -= dt_sub;
            dd_dt = calc_dd_dt(d_dust, tau_sp);
            dd = dt * dd_dt;
        }

        params_dev[5] = time_refine;

        // update dust density
        d_dust += dd;

        // remove scaling constant
        d_gas /= K;
        d_dust /= K;
        dev_conserved[5*n_cells + id] = d_dust;
        
        #ifdef DE
        dev_conserved[(n_fields-1)*n_cells + id] = d*ge;
        #endif
    }
}

__device__ Real calc_tau_sp(Real n, Real T) {
  Real YR_IN_S = 3.154e7;
  Real a1 = 1; // dust grain size in units of 0.1 micrometers
  Real d0 = n / (6e-4); // gas density in units of 10^-27 g/cm^3
  Real T_0 = 2e6; // K
  Real omega = 2.5;
  Real A = 0.17e9 * YR_IN_S; // 0.17 Gyr in s

  return A * (a1/d0) * (pow(T_0/T, omega) + 1); // s
}

__device__ Real calc_dd_dt(Real d_dust, Real tau_sp) {
    return -d_dust / (tau_sp/3);
}

// function to initialize conserved variable array, similar to Grid3D::Constant in grid/initial_conditions.cpp 
void Conserved_Init(Real *host_conserved, Real rho, Real vx, Real vy, Real vz, Real P, Real rho_d, Real gamma, int n_cells, int nx, int ny, int nz, int n_ghost, int n_fields)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;

  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny-H.n_ghost;
  }
  else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for(k=kstart-1; k<kend; k++) {
    for(j=jstart-1; j<jend; j++) {
      for(i=istart-1; i<iend; i++) {

        //get cell index
        id = i + j*nx + k*nx*ny;

        // Exclude the rightmost ghost cell on the "left" side
        if ((k >= kstart) and (j >= jstart) and (i >= istart))
        {
          // set constant initial states
          host_conserved[id] = rho;
          host_conserved[1*n_cells+id] = rho*vx;
          host_conserved[2*n_cells+id] = rho*vy;
          host_conserved[3*n_cells+id] = rho*vz;
          host_conserved[4*n_cells+id] = P/(gamma-1.0) + 0.5*rho*(vx*vx + vy*vy + vz*vz);
          #ifdef DE
          host_conserved[(n_fields-1)*n_cells+id] = P/(gamma-1.0);
          #endif  // DE
          #ifdef SCALAR
          host_conserved[5*n_cells+id] = rho_d;
          #endif // SCALAR
        }
      }
    }
  }
}


#endif // SCALAR
#endif // CUDA
/*!
 * \file dust_cuda.cu
 * \author Helena Richie (helenarichie@gmail.com)
 * \brief Contains code that updates the dust density scalar field. The dust_kernel function determines the rate of 
 * change of dust density, which is controlled by the sputtering timescale. The sputtering timescale is from the 
 * McKinnon et al. (2017) model of dust sputtering, which depends on the cell's gas density and temperature.
 */

#ifdef CUDA
  #ifdef DUST

    // STL includes
    #include <stdio.h>

    // External includes
    #include <cstdio>
    #include <fstream>
    #include <vector>

    // Local includes
    #include "../global/global.h"
    #include "../global/global_cuda.h"
    #include "../grid/grid3D.h"
    #include "../grid/grid_enum.h"
    #include "../utils/cuda_utilities.h"
    #include "../utils/gpu.hpp"
    #include "../utils/hydro_utilities.h"
    #include "dust_cuda.h"

void Dust_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma)
{
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Dust_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dt, gamma);
  CudaCheckError();
}

__global__ void Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma)
{
  // get grid indices
  int n_cells = nx * ny * nz;
  int is, ie, js, je, ks, ke;
  cuda_utilities::Get_Real_Indices(n_ghost, nx, ny, nz, is, ie, js, je, ks, ke);
  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int id      = threadIdx.x + blockId * blockDim.x;
  int zid     = id / (nx * ny);
  int yid     = (id - zid * nx * ny) / nx;
  int xid     = id - zid * nx * ny - yid * nx;

  // define physics variables
  Real d_gas, d_dust;  // fluid mass densities
  Real n;              // gas number density
  Real mu = 0.6;       // mean molecular weight
  Real T, E, P;        // temperature, energy, pressure
  Real vx, vy, vz;     // velocities
    #ifdef DE
  Real ge;
    #endif  // DE

  // define integration variables
  Real dd_dt;          // instantaneous rate of change in dust density
  Real dd;             // change in dust density at current timestep
  Real dd_max = 0.01;  // allowable percentage of dust density increase
  Real dt_sub;         // refined timestep

  if (xid >= is && xid < ie && yid >= js && yid < je && zid >= ks && zid < ke) {
    // get conserved quanitites
    d_gas  = dev_conserved[id + n_cells * grid_enum::density];
    d_dust = dev_conserved[id + n_cells * grid_enum::dust_density];
    E      = dev_conserved[id + n_cells * grid_enum::Energy];

    // convert mass density to number density
    n = d_gas * DENSITY_UNIT / (mu * MP);

    if (E < 0.0 || E != E) {
      return;
    }

    // get conserved quanitites
    vx = dev_conserved[id + n_cells * grid_enum::momentum_x] / d_gas;
    vy = dev_conserved[id + n_cells * grid_enum::momentum_y] / d_gas;
    vz = dev_conserved[id + n_cells * grid_enum::momentum_z] / d_gas;
    #ifdef DE
    ge = dev_conserved[id + n_cells * grid_enum::GasEnergy] / d_gas;
    ge = fmax(ge, (Real)TINY_NUMBER);
    #endif  // DE

    // calculate physical quantities
    P = hydro_utilities::Calc_Pressure_Primitive(E, d_gas, vx, vy, vz, gamma);

    Real T_init;
    T_init = hydro_utilities::Calc_Temp(P, n);

    #ifdef DE
    T_init = hydro_utilities::Calc_Temp_DE(d_gas, ge, gamma, n);
    #endif  // DE

    // if dual energy is turned on use temp from total internal energy
    T = T_init;

    Real tau_sp = calc_tau_sp(n, T) / TIME_UNIT;  // sputtering timescale, kyr (sim units)

    dd_dt = calc_dd_dt(d_dust, tau_sp);  // rate of change in dust density at current timestep
    dd    = dd_dt * dt;  // change in dust density at current timestep

    // ensure that dust density is not changing too rapidly
    while (dd / d_dust > dd_max) {
      dt_sub = dd_max * d_dust / dd_dt;
      d_dust += dt_sub * dd_dt;
      dt -= dt_sub;
      dd_dt = calc_dd_dt(d_dust, tau_sp);
      dd    = dt * dd_dt;
    }

    // update dust density
    d_dust += dd;

    dev_conserved[id + n_cells * grid_enum::dust_density] = d_dust;

    #ifdef DE
    dev_conserved[id + n_cells * grid_enum::GasEnergy] = d_dust * ge;
    #endif
  }
}

// McKinnon et al. (2017)
__device__ __host__ Real calc_tau_sp(Real n, Real T)
{
  Real YR_IN_S = 3.154e7;
  Real a1      = 1;           // dust grain size in units of 0.1 micrometers
  Real d0      = n / (6e-4);  // gas density in units of 10^-27 g/cm^3
  Real T_0     = 2e6;         // K
  Real omega   = 2.5;
  Real A       = 0.17e9 * YR_IN_S;  // 0.17 Gyr in s

  Real tau_sp = A * (a1 / d0) * (pow(T_0 / T, omega) + 1);  // sputtering timescale, s

  return tau_sp;
}

// McKinnon et al. (2017)
__device__ __host__ Real calc_dd_dt(Real d_dust, Real tau_sp) { return -d_dust / (tau_sp / 3); }

  #endif  // DUST
#endif    // CUDA

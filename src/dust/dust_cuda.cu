/*!
 * \file dust_cuda.cu
 * \author Helena Richie (helenarichie@gmail.com)
 * \brief Contains code that updates the dust density scalar field. The dust_kernel function determines the rate of
 * change of dust density, which is controlled by the sputtering timescale. The sputtering timescale is from the
 * McKinnon et al. (2017) model of dust sputtering, which depends on the cell's gas density and temperature.
 */

#ifdef DUST

  // STL includes
  #include <stdio.h>

  #include <cstdio>
  #include <fstream>
  #include <vector>

  // Local includes
  #include "../dust/dust_cuda.h"
  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../utils/cuda_utilities.h"
  #include "../utils/gpu.hpp"
  #include "../utils/hydro_utilities.h"

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
  int id_z    = id / (nx * ny);
  int id_y    = (id - id_z * nx * ny) / nx;
  int id_x    = id - id_z * nx * ny - id_y * nx;

  // define physics variables
  Real density_gas, density_dust;           // fluid mass densities
  Real number_density;                      // gas number density
  Real mu = 0.6;                            // mean molecular weight
  Real temperature, energy, pressure;       // temperature, energy, pressure
  Real velocity_x, velocity_y, velocity_z;  // velocities
  #ifdef DE
  Real energy_gas;
  #endif  // DE

  // define integration variables
  Real dd_dt;          // instantaneous rate of change in dust density
  Real dd;             // change in dust density at current timestep
  Real dd_max = 0.01;  // allowable percentage of dust density increase
  Real dt_sub;         // refined timestep

  if (id_x >= is && id_x < ie && id_y >= js && id_y < je && id_z >= ks && id_z < ke) {
    // get conserved quanitites
    density_gas  = dev_conserved[id + n_cells * grid_enum::density];
    density_dust = dev_conserved[id + n_cells * grid_enum::dust_density];
    energy       = dev_conserved[id + n_cells * grid_enum::Energy];

    // convert mass density to number density
    number_density = density_gas * DENSITY_UNIT / (mu * MP);

    if (energy < 0.0 || energy != energy) {
      return;
    }

    // get conserved quanitites
    velocity_x = dev_conserved[id + n_cells * grid_enum::momentum_x] / density_gas;
    velocity_y = dev_conserved[id + n_cells * grid_enum::momentum_y] / density_gas;
    velocity_z = dev_conserved[id + n_cells * grid_enum::momentum_z] / density_gas;
  #ifdef DE
    energy_gas = dev_conserved[id + n_cells * grid_enum::GasEnergy] / density_gas;
    energy_gas = fmax(ge, (Real)TINY_NUMBER);
  #endif  // DE

    // calculate physical quantities
    pressure = hydro_utilities::Calc_Pressure_Primitive(energy, density_gas, velocity_x, velocity_y, velocity_z, gamma);

    Real temperature_init;
    temperature_init = hydro_utilities::Calc_Temp(pressure, number_density);

  #ifdef DE
    temperature_init = hydro_utilities::Calc_Temp_DE(density_gas, energy_gas, gamma, number_density);
  #endif  // DE

    // if dual energy is turned on use temp from total internal energy
    temperature = temperature_init;

    Real tau_sp = Calc_Sputtering_Timescale(number_density, temperature) / TIME_UNIT;  // sputtering timescale, kyr (sim units)

    dd_dt = Calc_dd_dt(density_dust, tau_sp);  // rate of change in dust density at current timestep
    dd    = dd_dt * dt;                        // change in dust density at current timestep

    // ensure that dust density is not changing too rapidly
    while (dd / density_dust > dd_max) {
      dt_sub = dd_max * density_dust / dd_dt;
      density_dust += dt_sub * dd_dt;
      dt -= dt_sub;
      dd_dt = Calc_dd_dt(density_dust, tau_sp);
      dd    = dt * dd_dt;
    }

    // update dust density
    density_dust += dd;

    dev_conserved[id + n_cells * grid_enum::dust_density] = density_dust;

  #ifdef DE
    dev_conserved[id + n_cells * grid_enum::GasEnergy] = density_dust * energy_gas;
  #endif
  }
}

// McKinnon et al. (2017) sputtering timescale
__device__ __host__ Real Calc_Sputtering_Timescale(Real number_density, Real temperature)
{
  Real grain_radius  = 1;          // dust grain size in units of 0.1 micrometers
  Real temperature_0 = 2e6;        // temp above which the sputtering rate is ~constant in K
  Real omega         = 2.5;        // controls the low-temperature scaling of the sputtering rate
  Real A             = 5.3618e15;  // 0.17 Gyr in s

  number_density /= (6e-4);  // gas number density in units of 10^-27 g/cm^3

  // sputtering timescale, s
  Real tau_sp = A * (grain_radius / number_density) * (pow(temperature_0 / temperature, omega) + 1);

  return tau_sp;
}

// McKinnon et al. (2017) sputtering model
__device__ __host__ Real Calc_dd_dt(Real density_dust, Real tau_sp) { return -density_dust / (tau_sp / 3); }

#endif  // DUST

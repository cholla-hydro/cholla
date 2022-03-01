#ifdef CUDA
#ifdef DUST_GPU

#include"dust_cuda.h"
#include<math.h>
#include<vector>
#include"global.h"
#include"global_cuda.h"
#include"gpu.hpp"

__global__ void dust_kernel(Real *dev_conserved, int nx, int ny, int nz,
  int n_ghost, int n_fields, Real dt, Real gamma, Real *dt_array) {
  __shared__ Real min_dt[TPB];  // TPB = threads per block

  int n_cells = nx * ny * nz;
  int is, ie, js, je, ks, ke;
  is = n_ghost;
  ie = nx - n_ghost;
  if (ny == 1) {
    js = 0;
    je = 1;
  } else {
    js = n_ghost;
    je = ny - n_ghost;
  }
  if (nz == 1) {
    ks = 0;
    ke = 1;
  } else {
    ks = n_ghost;
    ke = nz - n_ghost;
  }

  Real d_gas, E;  // gas density, energy
  Real n, T;  // number density, temperature, initial temperature
  // dust density rate of change, change in dust density, refined timestep
  Real dd_dt, dd, dt_sub;
  Real mu;  // mean molecular weight
  Real d_dust;  // dust density
  Real d_metal; // metal density
  Real vx, vy, vz, p;
  #ifdef DE
  Real ge;
  #endif

  mu = 0.6;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int zid = id / (nx * ny);
  int yid = (id - zid * nx * ny) / nx;
  int xid = id - zid * nx * ny - yid * nx;
  // add a thread id within the block
  int tid = threadIdx.x;

  _syncthreads();

  // only threads corresponding to real cells do the calculation
  if (xid >= is && xid < ie && yid >= js && yid < je && zid >= ks && zid < ke) {
    d_gas = dev_conserved[id];
    E = dev_conserved[4*n_cells + id];
    // make sure thread hasn't crashed
    if (E < 0.0 || E != E) return;

    vx =  dev_conserved[1*n_cells + id] / d_gas;
    vy =  dev_conserved[2*n_cells + id] / d_gas;
    vz =  dev_conserved[3*n_cells + id] / d_gas;

    p  = (E - 0.5 * d_gas * (vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    p  = fmax(p, (Real) TINY_NUMBER);

    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id] / d_gas;
    ge = fmax(ge, (Real) TINY_NUMBER);
    #endif

    n = d_gas * DENSITY_UNIT / (mu * MP);  // number density of gas (in cgs)

    // calculate the temperature of the gas
    T_init = p * PRESSURE_UNIT / (n * KB);

    #ifdef DE
    T_init = d_gas * ge * (gamma - 1.0) * PRESSURE_UNIT / (n * KB);
    #endif

    T = T_init;

    // dust density
    d_dust = dev_conserved[5*n_cells + id];

    // dust mass rate of change
    dd_dt = d_gas_accretion(T, d_gas, d_dust, d_metal) +
      d_thermal_sputtering(T, d_gas, d_dust);

    // Calculate change in dust density during simulation timestep
    dd = dt * dd_dt;

    // if change in dust density is greater than 1% then refine timestep
    while (dd/d_dust > 0.01) {
      // what dt gives dd = 0.01*d_dust?
      dt_sub = 0.01 * d_dust / dd_dt;
      // use dt_sub in forward Euler update
      d_dust += dd_dt * dt_sub;

      // how much time is left from the original timestep?
      dt -= dt_sub;

      // update dust density rate of change
      dd_dt = gas_accretion(T, d_gas, d_dust, d_metal) +
        thermal_sputtering(T, d_gas, d_dust);

      /* calculate new change in density at this rate and repeat if greater
         than 1% change */
      dd = dt * dd_dt;
    }

    d_dust += dt * dd_dt;

    dev_conserved[5*n_cells + id] = d_dust;

    if (n > 0 && T > 0 && dd_dt > 0.0) {
      // limit the timestep such that delta_T is 10% 
      min_dt[tid] = 0.01 * d_dust / dd_dt;
    }
  }
  __syncthreads()
  
  // do the reduction in shared memory (find the min timestep in the block)
  for (unsigned int s=1; s<blockDim.x; s*=2) {
    if (tid % (2*s) == 0) {
      min_dt[tid] = fmin(min_dt[tid], min_dt[tid + s]);
    }
    __syncthreads();
  }
  
  // write the result for this block to global memory
  if (tid == 0) dt_array[blockIdx.x] = min_dt[0];
}



class DustIntegrator:
{
  public:
    Real T;
    Real n;
    Real dt;
    Real tmax;
    Real int_type;
    Real d0_gas, d0_metal, d0_dust = initialize_densities();
    Real d_gas, d_metal, d_dust = initialize_densities();
    Real tau_g = calc_tau_g();
    Real tau_sp = calc_tau_sp();
  
  protected:
    Real MP = 1.6726*pow(10,-24); // proton mass in g
    Real YR_IN_S = 3.154*pow(10,7); // one year in s
    // solar abundances (percentage of total mass)
    // O, C, N, Si, Mg, Ne, Fe, S
    vector<Real> metals = {0.97, 0.40, 0.096, 0.099, 0.079, 0.058, 0.14, 0.040};
    Real metallicity
    std::for_each(metals.begin(), metals.end(), [&] (int n) {
     metallicity += n;
    });

  Real initialize_densities() {
    Real d0_gas = MP * n; // g/cm^3
    Real d0_metal = metallicity * d0_gas;
    Real d0_dust = d0_gas / 100 // assume 1% dust-to-gas fraction

    return d0_gas, d0_metal, d0_dust;
  }

  Real calc_tau_g() {
    Real tau_g_ref = 0.2*pow(10, 9); // 0.2 Gyr in s
    Real d_ref = MP; // 1 H atom per cubic centimeter
    Real T_ref = 20.0; // 20 K
    Real tau_g;
    tau_g = tau_g_ref * (d_ref/d0_gas)) * pow(T_ref/T,1/2);
    
    return tau_g;
  }

  Real calc_tau_sp() {
    Real a1 = 1; // dust grain size in units of 0.1 micrometers
    Real d0 = n/(6*pow(10,-4)); // gas density in units of 10^-27 g/cm^3
    Real T_0 = 2*pow(10,6); // K
    Real omega = 2.5;
    Real A = 0.17*pow(10,9) * YR_IN_S; // 0.17 Gyr in s

    return  A * (a1/d0) * (pow(T_0/self.T, omega) + 1); // s
  }

};
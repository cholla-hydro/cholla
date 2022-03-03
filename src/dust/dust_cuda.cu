#ifdef CUDA
#ifdef DUST_GPU

#include"dust_cuda_updated.h"
#include<math.h>
#include<vector.h>
#include"global.h"
#include"global_cuda.h"
#include"gpu.hpp"

__global__ void dust_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, 
int n_fields, Real dt, Real gamma, Real *dt_array) {
    __shared__ Real min_dt[TPB]; // TPB = threads per block

    // get grid inidices
    int n_cells = nx * ny * nz;
    int is, ie, js, je, ks, ke;
    Get_Indices(nx, ny, nz, is, ie, js, je, ks, ke);

    // get a global thread ID
    int id;
    int xid, yid, zid;
    int tid;
    Get_GTID(id, xid, yid, zid, tid);

    // define physics variables
    Real d_gas, d_dust; // fluid mass densities
    Real n; // gas number density
    Real T, E, p; // temperature, energy, pressure
    Real mu = 0.6; // mean molecular weight
    Real vx, vy, vz; // velocities
    #ifdef DE
    Real ge;
    #endif // DE

    // define integration variables
    Real dd_dt; // instantaneous rate of change in dust density
    Real dd; // change in dust density at current time-step
    Real dd_max = 0.01; // allowable percentage of dust density increase

    _syncthreads();
    
    if (xid >= is && xid < ie && yid >= js && yid < je && zid >= ks && zid < ke) {
        // get quantities from dev_conserved
        d_gas = dev_conserved[id];
        d_dust = dev_conserved[5*n_cells + id];
        E = dev_conserved[4*n_cells + id];
        // make sure thread hasn't crashed
        if (E < 0.0 || E != E) return;

        vx =  dev_conserved[1*n_cells + id] / d_gas;
        vy =  dev_conserved[2*n_cells + id] / d_gas;
        vz =  dev_conserved[3*n_cells + id] / d_gas;

        #ifdef DE
        ge = dev_conserved[(n_fields-1)*n_cells + id] / d_gas;
        ge = fmax(ge, (Real) TINY_NUMBER);
        #endif // DE

        // calculate physical quantities
        p = Calc_Pressure(E, d_gas, vx, vy, vz, gamma);

        Real T_init;
        T_init = Calc_Temp(p, n);

        #ifdef DE
        T_init = Calc_Temp_DE(d_gas, ge, gamma, n);
        #endif // DE

        T = T_init;

        // calculate change in dust density
        Dust dustObj(T, n, dt, d_gas, d_dust);
        dustObj.calc_tau_sp();

        dd_dt = dustObj.calc_dd_dt();
        dd = dd_dt * dt;

        // ensure that dust density is not changing too rapidly
        while (d_dust/dd > dd_max) {
            dt_sub = dd_max * d_dust / dd_dt;
            dustObj.d_dust += dt_sub * dd_dt;
            dustObj.dt -= dt_sub;
            dt = dustObj.dt;
            dd_dt = dustObj.calc_dd_dt();
            dd = dt * dd_dt;
        }

        // update dust and gas densities
        dev_conserved[5*n_cells + id] = dustObj.d_dust;
        dev_conserved[id] += dd;
    }
    __syncthreads();
    
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

void Dust::calc_tau_sp() {
  Real a1 = 1; // dust grain size in units of 0.1 micrometers
  Real d0 = n/(6*pow(10,-4)); // gas density in units of 10^-27 g/cm^3
  Real T_0 = 2*pow(10,6); // K
  Real omega = 2.5;
  Real A = 0.17*pow(10,9) * YR_IN_S; // 0.17 Gyr in s

  tau_sp = A * (a1/d0) * (pow(T_0/T, omega) + 1); // s
}

Real Dust::calc_dd_dt() {
    return -d_dust / (tau_sp/3);
}

// forward-Euler methods:

__device__ void Get_Indices(int nx, int ny, int nz, int is, int ie, int js, int je, int ks, int ke) {
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
}

__device__ void Get_GTID(int id, int xid, int yid, int zid, int tid) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int id = threadIdx.x + blockId * blockDim.x;
    int zid = id / (nx * ny);
    int yid = (id - zid * nx * ny) / nx;
    int xid = id - zid * nx * ny - yid * nx;
    // add a thread id within the block
    int tid = threadIdx.x;
}

__device__ Real Calc_Pressure(Real E, Real d_gas, Real vx, Real vy, Real vz, Real gamma) {
    Real p;
    p  = (E - 0.5 * d_gas * (vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    p  = fmax(p, (Real) TINY_NUMBER);
    return p;
}

__device__ Real Calc_Temp(Real p, Real n) {
    Real T = p * PRESSURE_UNIT / (n * KB);
    return T;
}

__device__ Real Calc_Temp_DE(Real d_gas, Real ge, Real gamma, Real n) {
    Real T =  d_gas * ge * (gamma - 1.0) * PRESSURE_UNIT / (n * KB);
    return T;
}

#endif // DUST_GPU
#endif // CUDA
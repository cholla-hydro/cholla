/*! \file hllc_cuda.cu
 *  \brief Function definitions for the cuda HLLC Riemann solver.*/

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/reconstruction.h"
#include "../riemann_solvers/hll_cuda.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"

template <int reconstruction, uint direction>
__global__ void Calculate_HLL_Fluxes_CUDA(Real const *dev_conserved, Real const *dev_bounds_L, Real const *dev_bounds_R,
                                          Real *dev_flux, int const nx, int const ny, int const nz, int const n_cells,
                                          Real const gamma, Real const dx, Real const dt, int const n_fields)
{
  // get a thread index
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int tid     = threadIdx.x + blockId * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(tid, nx, ny, xid, yid, zid);

  // Note that in only this riemann solver neither the scalar nor the dual energy that are carried through are NOT the
  // specific versions despite the name in the struct
  reconstruction::InterfaceState left_state, right_state;

  Real f_d_l, f_mx_l, f_my_l, f_mz_l, f_E_l;
  Real f_d_r, f_mx_r, f_my_r, f_mz_r, f_E_r;
  Real f_d, f_mx, f_my, f_mz, f_E;
  Real Sl, Sr, cfl, cfr;
#ifdef SCALAR
  Real f_sc_l[NSCALARS], f_sc_r[NSCALARS], f_sc[NSCALARS];
#endif

  // Real etah = 0;

  int o1, o2, o3;
  if constexpr (direction == 0) {
    o1 = 1;
    o2 = 2;
    o3 = 3;
  }
  if constexpr (direction == 1) {
    o1 = 2;
    o2 = 3;
    o3 = 1;
  }
  if constexpr (direction == 2) {
    o1 = 3;
    o2 = 1;
    o3 = 2;
  }

  // Thread guard to avoid overrun
  if (not reconstruction::Riemann_Thread_Guard<reconstruction>(nx, ny, nz, xid, yid, zid)) {
    // =========================
    // Load the interface states
    // =========================

    // Check if the reconstruction chosen is implemented as a device function yet
    if constexpr (reconstruction == reconstruction::Kind::pcm) {
      reconstruction::Reconstruct_Interface_States<reconstruction, direction>(
          dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma, dx, dt, left_state, right_state);
    } else {
      // retrieve conserved variables
      left_state.density    = dev_bounds_L[tid];
      left_state.momentum.x = dev_bounds_L[o1 * n_cells + tid];
      left_state.momentum.y = dev_bounds_L[o2 * n_cells + tid];
      left_state.momentum.z = dev_bounds_L[o3 * n_cells + tid];
      left_state.energy     = dev_bounds_L[4 * n_cells + tid];
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        left_state.scalar_specific[i] = dev_bounds_L[(5 + i) * n_cells + tid];
      }
#endif
#ifdef DE
      left_state.gas_energy_specific = dev_bounds_L[(n_fields - 1) * n_cells + tid];
#endif

      right_state.density    = dev_bounds_R[tid];
      right_state.momentum.x = dev_bounds_R[o1 * n_cells + tid];
      right_state.momentum.y = dev_bounds_R[o2 * n_cells + tid];
      right_state.momentum.z = dev_bounds_R[o3 * n_cells + tid];
      right_state.energy     = dev_bounds_R[4 * n_cells + tid];
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        right_state.scalar_specific[i] = dev_bounds_R[(5 + i) * n_cells + tid];
      }
#endif
#ifdef DE
      right_state.gas_energy_specific = dev_bounds_R[(n_fields - 1) * n_cells + tid];
#endif

      // calculate primitive variables
      left_state.velocity.x = left_state.momentum.x / left_state.density;
      left_state.velocity.y = left_state.momentum.y / left_state.density;
      left_state.velocity.z = left_state.momentum.z / left_state.density;
#ifdef DE  // PRESSURE_DE
      Real E_kin = 0.5 * left_state.density *
                   (left_state.velocity.x * left_state.velocity.x + left_state.velocity.y * left_state.velocity.y +
                    left_state.velocity.z * left_state.velocity.z);
      left_state.pressure = hydro_utilities::Get_Pressure_From_DE(left_state.energy, left_state.energy - E_kin,
                                                                  left_state.gas_energy_specific, gamma);
#else
      left_state.pressure = (left_state.energy - 0.5 * left_state.density *
                                                     (left_state.velocity.x * left_state.velocity.x +
                                                      left_state.velocity.y * left_state.velocity.y +
                                                      left_state.velocity.z * left_state.velocity.z)) *
                            (gamma - 1.0);
#endif  // DE
      left_state.pressure = fmax(left_state.pressure, (Real)TINY_NUMBER);
      // #ifdef SCALAR
      // for (int i=0; i<NSCALARS; i++) {
      //   scl[i] = left_state.scalar_specific[i] / left_state.density;
      // }
      // #endif
      // #ifdef DE
      // gel = left_state.gas_energy_specific / left_state.density;
      // #endif
      right_state.velocity.x = right_state.momentum.x / right_state.density;
      right_state.velocity.y = right_state.momentum.y / right_state.density;
      right_state.velocity.z = right_state.momentum.z / right_state.density;
#ifdef DE  // PRESSURE_DE
      E_kin = 0.5 * right_state.density *
              (right_state.velocity.x * right_state.velocity.x + right_state.velocity.y * right_state.velocity.y +
               right_state.velocity.z * right_state.velocity.z);
      right_state.pressure = hydro_utilities::Get_Pressure_From_DE(right_state.energy, right_state.energy - E_kin,
                                                                   right_state.gas_energy_specific, gamma);
#else
      right_state.pressure = (right_state.energy - 0.5 * right_state.density *
                                                       (right_state.velocity.x * right_state.velocity.x +
                                                        right_state.velocity.y * right_state.velocity.y +
                                                        right_state.velocity.z * right_state.velocity.z)) *
                             (gamma - 1.0);
#endif  // DE
      right_state.pressure = fmax(right_state.pressure, (Real)TINY_NUMBER);
    }
    // #ifdef SCALAR
    // for (int i=0; i<NSCALARS; i++) {
    //   scr[i] = right_state.scalar_specific[i] / right_state.density;
    // }
    // #endif
    // #ifdef DE
    // ger = right_state.gas_energy_specific / right_state.density;
    // #endif

    // calculate the enthalpy in each cell
    // Hl = (left_state.energy + left_state.pressure) / left_state.density;
    // Hr = (right_state.energy + right_state.pressure) / right_state.density;

    // calculate averages of the variables needed for the Roe Jacobian
    // (see Stone et al., 2008, Eqn 65, or Toro 2009, 11.118)
    // sqrtdl = sqrt(dl);
    // sqrtdr = sqrt(right_state.density);
    // vx = (sqrtdl*left_state.velocity.x + sqrtdr*right_state.velocity.x) / (sqrtdl + sqrtdr);
    // vy = (sqrtdl*left_state.velocity.y + sqrtdr*right_state.velocity.y) / (sqrtdl + sqrtdr);
    // vz = (sqrtdl*left_state.velocity.z + sqrtdr*right_state.velocity.z) / (sqrtdl + sqrtdr);
    // H  = (sqrtdl*Hl  + sqrtdr*Hr)  / (sqrtdl + sqrtdr);

    // calculate the sound speed squared (Stone B2)
    // vsq = (vx*vx + vy*vy + vz*vz);
    // asq = g1*(H - 0.5*vsq);
    // a = sqrt(asq);

    // calculate the averaged eigenvectors of the Roe matrix (Stone Eqn B2,
    // Toro 11.107) lambda_m = vx - a; lambda_p = vx + a;

    // compute max and min wave speeds
    cfl = sqrt(gamma * left_state.pressure / left_state.density);    // sound speed in left state
    cfr = sqrt(gamma * right_state.pressure / right_state.density);  // sound speed in right state

    // for signal speeds, take max/min of Roe eigenvalues and left and right
    // sound speeds Batten eqn. 48 Sl = fmin(lambda_m, left_state.velocity.x - cfl); Sr =
    // fmax(lambda_p, right_state.velocity.x + cfr);

    // if the H-correction is turned on, add cross-flux dissipation
    // Sl = sgn_CUDA(Sl)*fmax(fabs(Sl), etah);
    // Sr = sgn_CUDA(Sr)*fmax(fabs(Sr), etah);
    Sl = fmin(right_state.velocity.x - cfr, left_state.velocity.x - cfl);
    Sr = fmax(left_state.velocity.x + cfl, right_state.velocity.x + cfr);

    // left and right fluxes
    f_d_l  = left_state.momentum.x;
    f_mx_l = left_state.momentum.x * left_state.velocity.x + left_state.pressure;
    f_my_l = left_state.momentum.y * left_state.velocity.x;
    f_mz_l = left_state.momentum.z * left_state.velocity.x;
    f_E_l  = (left_state.energy + left_state.pressure) * left_state.velocity.x;
#ifdef DE
    Real f_ge_l = left_state.gas_energy_specific * left_state.velocity.x;
#endif
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      f_sc_l[i] = left_state.scalar_specific[i] * left_state.velocity.x;
    }
#endif

    f_d_r  = right_state.momentum.x;
    f_mx_r = right_state.momentum.x * right_state.velocity.x + right_state.pressure;
    f_my_r = right_state.momentum.y * right_state.velocity.x;
    f_mz_r = right_state.momentum.z * right_state.velocity.x;
    f_E_r  = (right_state.energy + right_state.pressure) * right_state.velocity.x;
#ifdef DE
    Real f_ge_r = right_state.gas_energy_specific * right_state.velocity.x;
#endif
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      f_sc_r[i] = right_state.scalar_specific[i] * right_state.velocity.x;
    }
#endif

    // return upwind flux if flow is supersonic
    if (Sl > 0.0) {
      dev_flux[tid]                = f_d_l;
      dev_flux[o1 * n_cells + tid] = f_mx_l;
      dev_flux[o2 * n_cells + tid] = f_my_l;
      dev_flux[o3 * n_cells + tid] = f_mz_l;
      dev_flux[4 * n_cells + tid]  = f_E_l;
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        dev_flux[(5 + i) * n_cells + tid] = f_sc_l[i];
      }
#endif
#ifdef DE
      dev_flux[(n_fields - 1) * n_cells + tid] = f_ge_l;
#endif
      return;
    } else if (Sr < 0.0) {
      dev_flux[tid]                = f_d_r;
      dev_flux[o1 * n_cells + tid] = f_mx_r;
      dev_flux[o2 * n_cells + tid] = f_my_r;
      dev_flux[o3 * n_cells + tid] = f_mz_r;
      dev_flux[4 * n_cells + tid]  = f_E_r;
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        dev_flux[(5 + i) * n_cells + tid] = f_sc_r[i];
      }
#endif
#ifdef DE
      dev_flux[(n_fields - 1) * n_cells + tid] = f_ge_r;
#endif
      return;
    }
    // otherwise compute subsonic flux
    else {
      f_d  = ((Sr * f_d_l) - (Sl * f_d_r) + Sl * Sr * (right_state.density - left_state.density)) / (Sr - Sl);
      f_mx = ((Sr * f_mx_l) - (Sl * f_mx_r) + Sl * Sr * (right_state.momentum.x - left_state.momentum.x)) / (Sr - Sl);
      f_my = ((Sr * f_my_l) - (Sl * f_my_r) + Sl * Sr * (right_state.momentum.y - left_state.momentum.y)) / (Sr - Sl);
      f_mz = ((Sr * f_mz_l) - (Sl * f_mz_r) + Sl * Sr * (right_state.momentum.z - left_state.momentum.z)) / (Sr - Sl);
      f_E  = ((Sr * f_E_l) - (Sl * f_E_r) + Sl * Sr * (right_state.energy - left_state.energy)) / (Sr - Sl);
#ifdef DE
      Real f_ge = ((Sr * f_ge_l) - (Sl * f_ge_r) +
                   Sl * Sr * (right_state.gas_energy_specific - left_state.gas_energy_specific)) /
                  (Sr - Sl);
#endif
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        f_sc[i] = ((Sr * f_sc_l[i]) - (Sl * f_sc_r[i]) +
                   Sl * Sr * (right_state.scalar_specific[i] - left_state.scalar_specific[i])) /
                  (Sr - Sl);
      }
#endif

      // return the hllc fluxes
      dev_flux[tid]                = f_d;
      dev_flux[o1 * n_cells + tid] = f_mx;
      dev_flux[o2 * n_cells + tid] = f_my;
      dev_flux[o3 * n_cells + tid] = f_mz;
      dev_flux[4 * n_cells + tid]  = f_E;
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        dev_flux[(5 + i) * n_cells + tid] = f_sc[i];
      }
#endif
#ifdef DE
      dev_flux[(n_fields - 1) * n_cells + tid] = f_ge;
#endif
    }
  }
}

// Instantiate the templates we need
int const threads_per_block = 256;

template __global__ __launch_bounds__(threads_per_block) void Calculate_HLL_Fluxes_CUDA<reconstruction::Kind::pcm, 0>(
    Real const *dev_conserved, Real const *dev_bounds_L, Real const *dev_bounds_R, Real *dev_flux, int const nx,
    int const ny, int const nz, int const n_cells, Real const gamma, Real const dx, Real const dt, int const n_fields);
template __global__ __launch_bounds__(threads_per_block) void Calculate_HLL_Fluxes_CUDA<reconstruction::Kind::pcm, 1>(
    Real const *dev_conserved, Real const *dev_bounds_L, Real const *dev_bounds_R, Real *dev_flux, int const nx,
    int const ny, int const nz, int const n_cells, Real const gamma, Real const dx, Real const dt, int const n_fields);
template __global__ __launch_bounds__(threads_per_block) void Calculate_HLL_Fluxes_CUDA<reconstruction::Kind::pcm, 2>(
    Real const *dev_conserved, Real const *dev_bounds_L, Real const *dev_bounds_R, Real *dev_flux, int const nx,
    int const ny, int const nz, int const n_cells, Real const gamma, Real const dx, Real const dt, int const n_fields);

#ifndef PCM
template __global__ __launch_bounds__(threads_per_block) void Calculate_HLL_Fluxes_CUDA<
    reconstruction::Kind::chosen, 0>(Real const *dev_conserved, Real const *dev_bounds_L, Real const *dev_bounds_R,
                                     Real *dev_flux, int const nx, int const ny, int const nz, int const n_cells,
                                     Real const gamma, Real const dx, Real const dt, int const n_fields);
template __global__ __launch_bounds__(threads_per_block) void Calculate_HLL_Fluxes_CUDA<
    reconstruction::Kind::chosen, 1>(Real const *dev_conserved, Real const *dev_bounds_L, Real const *dev_bounds_R,
                                     Real *dev_flux, int const nx, int const ny, int const nz, int const n_cells,
                                     Real const gamma, Real const dx, Real const dt, int const n_fields);
template __global__ __launch_bounds__(threads_per_block) void Calculate_HLL_Fluxes_CUDA<
    reconstruction::Kind::chosen, 2>(Real const *dev_conserved, Real const *dev_bounds_L, Real const *dev_bounds_R,
                                     Real *dev_flux, int const nx, int const ny, int const nz, int const n_cells,
                                     Real const gamma, Real const dx, Real const dt, int const n_fields);
#endif  // PCM

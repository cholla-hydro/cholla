/*! \file roe_cuda.cu
 *  \brief Function definitions for the cuda Roe Riemann solver.*/

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/reconstruction.h"
#include "../riemann_solvers/roe_cuda.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"

/*! \fn Calculate_Roe_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
 * *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, Real *dev_etah,
 * int dir, int n_fields) \brief Roe Riemann solver based on the version
 * described in Stone et al, 2008. */
__global__ void Calculate_Roe_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
                                          int nz, int n_ghost, Real gamma, int dir, int n_fields)
{
  // get a thread index
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int tid     = threadIdx.x + blockId * blockDim.x;
  int zid     = tid / (nx * ny);
  int yid     = (tid - zid * nx * ny) / nx;
  int xid     = tid - zid * nx * ny - yid * nx;

  int n_cells = nx * ny * nz;

  reconstruction::InterfaceState left_state, right_state;

  Real etah = 0.0;
  Real g1   = gamma - 1.0;
  Real Hl, Hr;
  Real sqrtdl, sqrtdr, vx, vy, vz, H;
  Real vsq, asq, a;
  Real lambda_m, lambda_0, lambda_p;
  Real f_d_l, f_mx_l, f_my_l, f_mz_l, f_E_l;
  Real f_d_r, f_mx_r, f_my_r, f_mz_r, f_E_r;
  Real del_d, del_mx, del_my, del_mz, del_E;
  Real a0, a1, a2, a3, a4;
  a0 = a1 = a2 = a3 = a4 = 0.0;
  Real sum_0, sum_1, sum_2, sum_3, sum_4;
  sum_0 = sum_1 = sum_2 = sum_3 = sum_4 = 0.0;
  Real test0, test1, test2, test3, test4;
  int hlle_flag = 0;
#ifdef DE
  Real dgel, dger, f_ge_l, f_ge_r, E_kin;
#endif
#ifdef SCALAR
  Real dscalarl[NSCALARS], dscalarr[NSCALARS], f_scalar_l[NSCALARS], f_scalar_r[NSCALARS];
#endif

  int o1, o2, o3;
  if (dir == 0) {
    o1 = 1;
    o2 = 2;
    o3 = 3;
  }
  if (dir == 1) {
    o1 = 2;
    o2 = 3;
    o3 = 1;
  }
  if (dir == 2) {
    o1 = 3;
    o2 = 1;
    o3 = 2;
  }

  // Each thread executes the solver independently
  if (xid < nx && yid < ny && zid < nz) {
    // retrieve conserved variables
    left_state.density    = dev_bounds_L[tid];
    left_state.momentum.x = dev_bounds_L[o1 * n_cells + tid];
    left_state.momentum.y = dev_bounds_L[o2 * n_cells + tid];
    left_state.momentum.z = dev_bounds_L[o3 * n_cells + tid];
    left_state.energy     = dev_bounds_L[4 * n_cells + tid];
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      dscalarl[i] = dev_bounds_L[(5 + i) * n_cells + tid];
    }
#endif
#ifdef DE
    dgel = dev_bounds_L[(n_fields - 1) * n_cells + tid];
#endif

    right_state.density    = dev_bounds_R[tid];
    right_state.momentum.x = dev_bounds_R[o1 * n_cells + tid];
    right_state.momentum.y = dev_bounds_R[o2 * n_cells + tid];
    right_state.momentum.z = dev_bounds_R[o3 * n_cells + tid];
    right_state.energy     = dev_bounds_R[4 * n_cells + tid];
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      dscalarr[i] = dev_bounds_R[(5 + i) * n_cells + tid];
    }
#endif
#ifdef DE
    dger = dev_bounds_R[(n_fields - 1) * n_cells + tid];
#endif

    // calculate primitive variables
    left_state.velocity.x = left_state.momentum.x / left_state.density;
    left_state.velocity.y = left_state.momentum.y / left_state.density;
    left_state.velocity.z = left_state.momentum.z / left_state.density;
#ifdef DE  // PRESSURE_DE
    E_kin = 0.5 * left_state.density *
            (left_state.velocity.x * left_state.velocity.x + left_state.velocity.y * left_state.velocity.y +
             left_state.velocity.z * left_state.velocity.z);
    left_state.pressure =
        hydro_utilities::Get_Pressure_From_DE(left_state.energy, left_state.energy - E_kin, dgel, gamma);
#else
    left_state.pressure = (left_state.energy - 0.5 * left_state.density *
                                                   (left_state.velocity.x * left_state.velocity.x +
                                                    left_state.velocity.y * left_state.velocity.y +
                                                    left_state.velocity.z * left_state.velocity.z)) *
                          (gamma - 1.0);
#endif  // PRESSURE_DE
    left_state.pressure = fmax(left_state.pressure, (Real)TINY_NUMBER);
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      left_state.scalar_specific[i] = dscalarl[i] / left_state.density;
    }
#endif
#ifdef DE
    left_state.gas_energy_specific = dgel / left_state.density;
#endif
    right_state.velocity.x = right_state.momentum.x / right_state.density;
    right_state.velocity.y = right_state.momentum.y / right_state.density;
    right_state.velocity.z = right_state.momentum.z / right_state.density;
#ifdef DE  // PRESSURE_DE
    E_kin = 0.5 * right_state.density *
            (right_state.velocity.x * right_state.velocity.x + right_state.velocity.y * right_state.velocity.y +
             right_state.velocity.z * right_state.velocity.z);
    right_state.pressure =
        hydro_utilities::Get_Pressure_From_DE(right_state.energy, right_state.energy - E_kin, dger, gamma);
#else
    right_state.pressure = (right_state.energy - 0.5 * right_state.density *
                                                     (right_state.velocity.x * right_state.velocity.x +
                                                      right_state.velocity.y * right_state.velocity.y +
                                                      right_state.velocity.z * right_state.velocity.z)) *
                           (gamma - 1.0);
#endif  // PRESSURE_DE
    right_state.pressure = fmax(right_state.pressure, (Real)TINY_NUMBER);
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      right_state.scalar_specific[i] = dscalarr[i] / right_state.density;
    }
#endif
#ifdef DE
    right_state.gas_energy_specific = dger / right_state.density;
#endif

    // calculate the enthalpy in each cell
    Hl = (left_state.energy + left_state.pressure) / left_state.density;
    Hr = (right_state.energy + right_state.pressure) / right_state.density;

    // calculate averages of the variables needed for the Roe Jacobian
    // (see Stone et al., 2008, Eqn 65, or Toro 2009, 11.118)
    sqrtdl = sqrt(left_state.density);
    sqrtdr = sqrt(right_state.density);
    vx     = (sqrtdl * left_state.velocity.x + sqrtdr * right_state.velocity.x) / (sqrtdl + sqrtdr);
    vy     = (sqrtdl * left_state.velocity.y + sqrtdr * right_state.velocity.y) / (sqrtdl + sqrtdr);
    vz     = (sqrtdl * left_state.velocity.z + sqrtdr * right_state.velocity.z) / (sqrtdl + sqrtdr);
    H      = (sqrtdl * Hl + sqrtdr * Hr) / (sqrtdl + sqrtdr);

    // calculate the sound speed squared (Stone B2)
    vsq = (vx * vx + vy * vy + vz * vz);
    asq = g1 * fmax((H - 0.5 * vsq), TINY_NUMBER);
    a   = sqrt(asq);

    // calculate the averaged eigenvectors of the Roe matrix (Stone Eqn B2,
    // Toro 11.107)
    lambda_m = vx - a;
    lambda_0 = vx;
    lambda_p = vx + a;

    // calculate the fluxes for the left and right input states,
    // based on the average values in either cell
    f_d_l  = left_state.momentum.x;
    f_mx_l = left_state.momentum.x * left_state.velocity.x + left_state.pressure;
    f_my_l = left_state.momentum.x * left_state.velocity.y;
    f_mz_l = left_state.momentum.x * left_state.velocity.z;
    f_E_l  = (left_state.energy + left_state.pressure) * left_state.velocity.x;
#ifdef DE
    f_ge_l = left_state.momentum.x * left_state.gas_energy_specific;
#endif
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      f_scalar_l[i] = left_state.momentum.x * left_state.scalar_specific[i];
    }
#endif

    f_d_r  = right_state.momentum.x;
    f_mx_r = right_state.momentum.x * right_state.velocity.x + right_state.pressure;
    f_my_r = right_state.momentum.x * right_state.velocity.y;
    f_mz_r = right_state.momentum.x * right_state.velocity.z;
    f_E_r  = (right_state.energy + right_state.pressure) * right_state.velocity.x;
#ifdef DE
    f_ge_r = right_state.momentum.x * right_state.gas_energy_specific;
#endif
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      f_scalar_r[i] = right_state.momentum.x * right_state.scalar_specific[i];
    }
#endif

    // return upwind flux if flow is supersonic
    if (lambda_m >= 0.0) {
      dev_flux[tid]                = f_d_l;
      dev_flux[o1 * n_cells + tid] = f_mx_l;
      dev_flux[o2 * n_cells + tid] = f_my_l;
      dev_flux[o3 * n_cells + tid] = f_mz_l;
      dev_flux[4 * n_cells + tid]  = f_E_l;
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        dev_flux[(5 + i) * n_cells + tid] = f_scalar_l[i];
      }
#endif
#ifdef DE
      dev_flux[(n_fields - 1) * n_cells + tid] = f_ge_l;
#endif
      return;
    } else if (lambda_p <= 0.0) {
      dev_flux[tid]                = f_d_r;
      dev_flux[o1 * n_cells + tid] = f_mx_r;
      dev_flux[o2 * n_cells + tid] = f_my_r;
      dev_flux[o3 * n_cells + tid] = f_mz_r;
      dev_flux[4 * n_cells + tid]  = f_E_r;
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        dev_flux[(5 + i) * n_cells + tid] = f_scalar_r[i];
      }
#endif
#ifdef DE
      dev_flux[(n_fields - 1) * n_cells + tid] = f_ge_r;
#endif
      return;
    }
    // otherwise calculate the Roe fluxes
    else {
      // calculate the difference in conserved variables across the cell
      // interface Stone Eqn 68
      del_d  = right_state.density - left_state.density;
      del_mx = right_state.momentum.x - left_state.momentum.x;
      del_my = right_state.momentum.y - left_state.momentum.y;
      del_mz = right_state.momentum.z - left_state.momentum.z;
      del_E  = right_state.energy - left_state.energy;

      // evaluate the flux function (Stone Eqn 66 & 67, Toro Eqn 11.29)

      Real Na    = 0.5 / asq;
      Real coeff = 0.0;

      // left eigenvector [0] * del_q
      a0 = del_d * Na * (0.5 * g1 * vsq + vx * a) - del_mx * Na * (g1 * vx + a) - del_my * Na * g1 * vy -
           del_mz * Na * g1 * vz + del_E * Na * g1;
      coeff = a0 * fmax(fabs(lambda_m), etah);
      sum_0 += coeff;
      sum_1 += coeff * (vx - a);
      sum_2 += coeff * vy;
      sum_3 += coeff * vz;
      sum_4 += coeff * (H - vx * a);
      // left eigenvector [1] * del_q
      a1    = -del_d * vy + del_my;
      coeff = a1 * fmax(fabs(lambda_0), etah);
      sum_2 += coeff;
      sum_4 += coeff * vy;
      // left eigenvector [2] * del_q
      a2    = -del_d * vz + del_mz;
      coeff = a2 * fmax(fabs(lambda_0), etah);
      sum_3 += coeff;
      sum_4 += coeff * vz;
      // left eigenvector [3] * del_q
      a3 = del_d * (1.0 - Na * g1 * vsq) + del_mx * g1 * vx / asq + del_my * g1 * vy / asq + del_mz * g1 * vz / asq -
           del_E * g1 / asq;
      coeff = a3 * fmax(fabs(lambda_0), etah);
      sum_0 += coeff;
      sum_1 += coeff * vx;
      sum_2 += coeff * vy;
      sum_3 += coeff * vz;
      sum_4 += coeff * 0.5 * vsq;
      // left eigenvector [4] * del_q
      a4 = del_d * Na * (0.5 * g1 * vsq - vx * a) - del_mx * Na * (g1 * vx - a) - del_my * Na * g1 * vy -
           del_mz * Na * g1 * vz + del_E * Na * g1;
      coeff = a4 * fmax(fabs(lambda_p), etah);
      sum_0 += coeff;
      sum_1 += coeff * (vx + a);
      sum_2 += coeff * vy;
      sum_3 += coeff * vz;
      sum_4 += coeff * (H + vx * a);

      // if density or pressure is negative, compute the HLLE fluxes
      // test intermediate states
      test0 = left_state.density + a0;
      test1 = left_state.momentum.x + a0 * (vx - a);
      test2 = left_state.momentum.y + a0 * vy;
      test3 = left_state.momentum.z + a0 * vz;
      test4 = left_state.energy + a0 * (H - vx * a);

      if (lambda_0 > lambda_m) {
        if (test0 <= 0.0) {
          hlle_flag = 1;
        }
        if (test4 - 0.5 * (test1 * test1 + test2 * test2 + test3 * test3) / test0 < 0.0) {
          hlle_flag = 2;
        }
      }

      test0 += a3 + a4;
      test1 += a3 * vx;
      test2 += a1 + a3 * vy;
      test3 += a2 + a3 * vz;
      test4 += a1 * vy + a2 * vz + a3 * 0.5 * vsq;

      if (lambda_p > lambda_0) {
        if (test0 <= 0.0) {
          hlle_flag = 1;
        }
        if (test4 - 0.5 * (test1 * test1 + test2 * test2 + test3 * test3) / test0 < 0.0) {
          hlle_flag = 2;
        }
      }

      // if pressure or density is negative, and we have not already returned
      // the supersonic fluxes, return the HLLE fluxes
      if (hlle_flag != 0) {
        Real cfl, cfr, bm, bp, tmp;

        // compute max and fmin wave speeds
        cfl = sqrt(gamma * left_state.pressure / left_state.density);    // sound speed in left state
        cfr = sqrt(gamma * right_state.pressure / right_state.density);  // sound speed in right state

        // take max/fmin of Roe eigenvalues and left and right sound speeds
        bm = fmin(fmin(lambda_m, left_state.velocity.x - cfl), (Real)0.0);
        bp = fmax(fmax(lambda_p, right_state.velocity.x + cfr), (Real)0.0);

        // compute left and right fluxes
        f_d_l = left_state.momentum.x - bm * left_state.density;
        f_d_r = right_state.momentum.x - bp * right_state.density;

        f_mx_l = left_state.momentum.x * (left_state.velocity.x - bm) + left_state.pressure;
        f_mx_r = right_state.momentum.x * (right_state.velocity.x - bp) + right_state.pressure;

        f_my_l = left_state.momentum.y * (left_state.velocity.x - bm);
        f_my_r = right_state.momentum.y * (right_state.velocity.x - bp);

        f_mz_l = left_state.momentum.z * (left_state.velocity.x - bm);
        f_mz_r = right_state.momentum.z * (right_state.velocity.x - bp);

        f_E_l = left_state.energy * (left_state.velocity.x - bm) + left_state.pressure * left_state.velocity.x;
        f_E_r = right_state.energy * (right_state.velocity.x - bp) + right_state.pressure * right_state.velocity.x;

#ifdef DE
        f_ge_l = dgel * (left_state.velocity.x - bm);
        f_ge_r = dger * (right_state.velocity.x - bp);
#endif

#ifdef SCALAR
        for (int i = 0; i < NSCALARS; i++) {
          f_scalar_l[i] = dscalarl[i] * (left_state.velocity.x - bm);
          f_scalar_r[i] = dscalarr[i] * (right_state.velocity.x - bp);
        }
#endif

        // compute the HLLE flux at the interface
        tmp = 0.5 * (bp + bm) / (bp - bm);

        dev_flux[tid]                = 0.5 * (f_d_l + f_d_r) + (f_d_l - f_d_r) * tmp;
        dev_flux[o1 * n_cells + tid] = 0.5 * (f_mx_l + f_mx_r) + (f_mx_l - f_mx_r) * tmp;
        dev_flux[o2 * n_cells + tid] = 0.5 * (f_my_l + f_my_r) + (f_my_l - f_my_r) * tmp;
        dev_flux[o3 * n_cells + tid] = 0.5 * (f_mz_l + f_mz_r) + (f_mz_l - f_mz_r) * tmp;
        dev_flux[4 * n_cells + tid]  = 0.5 * (f_E_l + f_E_r) + (f_E_l - f_E_r) * tmp;
#ifdef SCALAR
        for (int i = 0; i < NSCALARS; i++) {
          dev_flux[(5 + i) * n_cells + tid] =
              0.5 * (f_scalar_l[i] + f_scalar_r[i]) + (f_scalar_l[i] - f_scalar_r[i]) * tmp;
        }
#endif
#ifdef DE
        dev_flux[(n_fields - 1) * n_cells + tid] = 0.5 * (f_ge_l + f_ge_r) + (f_ge_l - f_ge_r) * tmp;
#endif
        return;
      }
      // otherwise return the roe fluxes
      else {
        dev_flux[tid]                = 0.5 * (f_d_l + f_d_r - sum_0);
        dev_flux[o1 * n_cells + tid] = 0.5 * (f_mx_l + f_mx_r - sum_1);
        dev_flux[o2 * n_cells + tid] = 0.5 * (f_my_l + f_my_r - sum_2);
        dev_flux[o3 * n_cells + tid] = 0.5 * (f_mz_l + f_mz_r - sum_3);
        dev_flux[4 * n_cells + tid]  = 0.5 * (f_E_l + f_E_r - sum_4);
#ifdef SCALAR
        for (int i = 0; i < NSCALARS; i++) {
          if (dev_flux[tid] >= 0.0) {
            dev_flux[(5 + i) * n_cells + tid] = dev_flux[tid] * left_state.scalar_specific[i];
          } else {
            dev_flux[(5 + i) * n_cells + tid] = dev_flux[tid] * right_state.scalar_specific[i];
          }
        }
#endif
#ifdef DE
        if (dev_flux[tid] >= 0.0) {
          dev_flux[(n_fields - 1) * n_cells + tid] = dev_flux[tid] * left_state.gas_energy_specific;
        } else {
          dev_flux[(n_fields - 1) * n_cells + tid] = dev_flux[tid] * right_state.gas_energy_specific;
        }
#endif
      }
    }
  }
}

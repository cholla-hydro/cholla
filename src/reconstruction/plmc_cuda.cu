/*! \file plmc_cuda.cu
 *  \brief Definitions of the piecewise linear reconstruction functions with
           limiting applied in the characteristic variables, as described
           in Stone et al., 2008. */

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif  // DE

/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
 *dev_bounds_R, int nx, int ny, int nz, Real dx, Real dt, Real
 gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using plm. */
__global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                          Real dt, Real gamma, int dir, int n_fields)
{
  // Compute the total number of cells
  int const n_cells = nx * ny * nz;

  // Set the field indices for the various directions
  int o1, o2, o3;
  switch (dir) {
    case 0:
      o1 = grid_enum::momentum_x;
      o2 = grid_enum::momentum_y;
      o3 = grid_enum::momentum_z;
      break;
    case 1:
      o1 = grid_enum::momentum_y;
      o2 = grid_enum::momentum_z;
      o3 = grid_enum::momentum_x;
      break;
    case 2:
      o1 = grid_enum::momentum_z;
      o2 = grid_enum::momentum_x;
      o3 = grid_enum::momentum_y;
      break;
  }

  // declare other variables to be used
  Real a_i;
  Real del_d_L, del_vx_L, del_vy_L, del_vz_L, del_p_L;
  Real del_d_R, del_vx_R, del_vy_R, del_vz_R, del_p_R;
  Real del_d_C, del_vx_C, del_vy_C, del_vz_C, del_p_C;
  Real del_d_G, del_vx_G, del_vy_G, del_vz_G, del_p_G;
  Real del_a_0_L, del_a_1_L, del_a_2_L, del_a_3_L, del_a_4_L;
  Real del_a_0_R, del_a_1_R, del_a_2_R, del_a_3_R, del_a_4_R;
  Real del_a_0_C, del_a_1_C, del_a_2_C, del_a_3_C, del_a_4_C;
  Real del_a_0_G, del_a_1_G, del_a_2_G, del_a_3_G, del_a_4_G;
  Real del_a_0_m, del_a_1_m, del_a_2_m, del_a_3_m, del_a_4_m;  // _m means monotized slope
  Real lim_slope_a, lim_slope_b;
  Real del_d_m_i, del_vx_m_i, del_vy_m_i, del_vz_m_i, del_p_m_i;
  Real d_L_iph, vx_L_iph, vy_L_iph, vz_L_iph, p_L_iph;
  Real d_R_imh, vx_R_imh, vy_R_imh, vz_R_imh, p_R_imh;
  Real C;
#ifndef VL
  Real dtodx = dt / dx;
  Real lambda_m, lambda_0, lambda_p;
  Real qx;
  Real lamdiff;
  Real sum_0, sum_1, sum_2, sum_3, sum_4;
#endif  // not VL
#ifdef DE
  Real ge_i, ge_imo, ge_ipo;
  Real del_ge_L, del_ge_R, del_ge_C, del_ge_G;
  Real del_ge_m_i;
  Real ge_L_iph, ge_R_imh;
  Real E, E_kin, dge;
  #ifndef VL
  Real sum_ge;
  #endif  // CTU
#endif    // DE
#ifdef SCALAR
  Real scalar_i[NSCALARS], scalar_imo[NSCALARS], scalar_ipo[NSCALARS];
  Real del_scalar_L[NSCALARS], del_scalar_R[NSCALARS], del_scalar_C[NSCALARS], del_scalar_G[NSCALARS];
  Real del_scalar_m_i[NSCALARS];
  Real scalar_L_iph[NSCALARS], scalar_R_imh[NSCALARS];
  #ifndef VL
  Real sum_scalar[NSCALARS];
  #endif  // CTU
#endif    // SCALAR

  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Thread guard to prevent overrun
  if (xid < 1 or xid >= nx - 2 or yid < 1 or yid >= ny - 2 or zid < 1 or zid >= nz - 2) {
    return;
  }
  // load the 3-cell stencil into registers
  // cell i
  plmc_utils::PlmcPrimitive const cell_i =
      plmc_utils::Load_Data(dev_conserved, xid, yid, zid, nx, ny, n_cells, o1, o2, o3, gamma);

  // cell i-1. The equality checks check the direction and subtract one from the direction
  plmc_utils::PlmcPrimitive const cell_imo = plmc_utils::Load_Data(
      dev_conserved, xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny, n_cells, o1, o2, o3, gamma);

  // cell i+1. The equality checks check the direction and add one to the direction
  plmc_utils::PlmcPrimitive const cell_ipo = plmc_utils::Load_Data(
      dev_conserved, xid + int(dir == 0), yid + int(dir == 1), zid + int(dir == 2), nx, ny, n_cells, o1, o2, o3, gamma);

  // calculate the adiabatic sound speed in cell i
  a_i = sqrt(gamma * cell_i.pressure / cell_i.density);

// Compute the eigenvalues of the linearized equations in the
// primitive variables using the cell-centered primitive variables
#ifndef VL
  lambda_m = cell_i.velocity_x - a_i;
  lambda_0 = cell_i.velocity_x;
  lambda_p = cell_i.velocity_x + a_i;
#endif  // VL

  // Compute the left, right, centered, and van Leer differences of the
  // primitive variables Note that here L and R refer to locations relative to
  // the cell center

  // left
  del_d_L  = cell_i.density - cell_imo.density;
  del_vx_L = cell_i.velocity_x - cell_imo.velocity_x;
  del_vy_L = cell_i.velocity_y - cell_imo.velocity_y;
  del_vz_L = cell_i.velocity_z - cell_imo.velocity_z;
  del_p_L  = cell_i.pressure - cell_imo.pressure;

  // right
  del_d_R  = cell_ipo.density - cell_i.density;
  del_vx_R = cell_ipo.velocity_x - cell_i.velocity_x;
  del_vy_R = cell_ipo.velocity_y - cell_i.velocity_y;
  del_vz_R = cell_ipo.velocity_z - cell_i.velocity_z;
  del_p_R  = cell_ipo.pressure - cell_i.pressure;

  // centered
  del_d_C  = 0.5 * (cell_ipo.density - cell_imo.density);
  del_vx_C = 0.5 * (cell_ipo.velocity_x - cell_imo.velocity_x);
  del_vy_C = 0.5 * (cell_ipo.velocity_y - cell_imo.velocity_y);
  del_vz_C = 0.5 * (cell_ipo.velocity_z - cell_imo.velocity_z);
  del_p_C  = 0.5 * (cell_ipo.pressure - cell_imo.pressure);

  // Van Leer
  if (del_d_L * del_d_R > 0.0) {
    del_d_G = 2.0 * del_d_L * del_d_R / (del_d_L + del_d_R);
  } else {
    del_d_G = 0.0;
  }
  if (del_vx_L * del_vx_R > 0.0) {
    del_vx_G = 2.0 * del_vx_L * del_vx_R / (del_vx_L + del_vx_R);
  } else {
    del_vx_G = 0.0;
  }
  if (del_vy_L * del_vy_R > 0.0) {
    del_vy_G = 2.0 * del_vy_L * del_vy_R / (del_vy_L + del_vy_R);
  } else {
    del_vy_G = 0.0;
  }
  if (del_vz_L * del_vz_R > 0.0) {
    del_vz_G = 2.0 * del_vz_L * del_vz_R / (del_vz_L + del_vz_R);
  } else {
    del_vz_G = 0.0;
  }
  if (del_p_L * del_p_R > 0.0) {
    del_p_G = 2.0 * del_p_L * del_p_R / (del_p_L + del_p_R);
  } else {
    del_p_G = 0.0;
  }

#ifdef DE
  del_ge_L = ge_i - ge_imo;
  del_ge_R = ge_ipo - ge_i;
  del_ge_C = 0.5 * (ge_ipo - ge_imo);
  if (del_ge_L * del_ge_R > 0.0) {
    del_ge_G = 2.0 * del_ge_L * del_ge_R / (del_ge_L + del_ge_R);
  } else {
    del_ge_G = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_scalar_L[i] = scalar_i[i] - scalar_imo[i];
    del_scalar_R[i] = scalar_ipo[i] - scalar_i[i];
    del_scalar_C[i] = 0.5 * (scalar_ipo[i] - scalar_imo[i]);
    if (del_scalar_L[i] * del_scalar_R[i] > 0.0) {
      del_scalar_G[i] = 2.0 * del_scalar_L[i] * del_scalar_R[i] / (del_scalar_L[i] + del_scalar_R[i]);
    } else {
      del_scalar_G[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  del_a_0_L = -cell_i.density * del_vx_L / (2 * a_i) + del_p_L / (2 * a_i * a_i);
  del_a_1_L = del_d_L - del_p_L / (a_i * a_i);
  del_a_2_L = del_vy_L;
  del_a_3_L = del_vz_L;
  del_a_4_L = cell_i.density * del_vx_L / (2 * a_i) + del_p_L / (2 * a_i * a_i);

  del_a_0_R = -cell_i.density * del_vx_R / (2 * a_i) + del_p_R / (2 * a_i * a_i);
  del_a_1_R = del_d_R - del_p_R / (a_i * a_i);
  del_a_2_R = del_vy_R;
  del_a_3_R = del_vz_R;
  del_a_4_R = cell_i.density * del_vx_R / (2 * a_i) + del_p_R / (2 * a_i * a_i);

  del_a_0_C = -cell_i.density * del_vx_C / (2 * a_i) + del_p_C / (2 * a_i * a_i);
  del_a_1_C = del_d_C - del_p_C / (a_i * a_i);
  del_a_2_C = del_vy_C;
  del_a_3_C = del_vz_C;
  del_a_4_C = cell_i.density * del_vx_C / (2 * a_i) + del_p_C / (2 * a_i * a_i);

  del_a_0_G = -cell_i.density * del_vx_G / (2 * a_i) + del_p_G / (2 * a_i * a_i);
  del_a_1_G = del_d_G - del_p_G / (a_i * a_i);
  del_a_2_G = del_vy_G;
  del_a_3_G = del_vz_G;
  del_a_4_G = cell_i.density * del_vx_G / (2 * a_i) + del_p_G / (2 * a_i * a_i);

  // Apply monotonicity constraints to the differences in the characteristic
  // variables

  del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;  // This should be in the declaration

  if (del_a_0_L * del_a_0_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
    lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
    del_a_0_m   = sgn_CUDA(del_a_0_C) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_1_L * del_a_1_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
    lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
    del_a_1_m   = sgn_CUDA(del_a_1_C) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_2_L * del_a_2_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
    lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
    del_a_2_m   = sgn_CUDA(del_a_2_C) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_3_L * del_a_3_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
    lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
    del_a_3_m   = sgn_CUDA(del_a_3_C) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_4_L * del_a_4_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
    lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
    del_a_4_m   = sgn_CUDA(del_a_4_C) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
#ifdef DE
  del_ge_m_i = 0.0;
  if (del_ge_L * del_ge_R > 0.0) {
    lim_slope_a = fmin(fabs(del_ge_L), fabs(del_ge_R));
    lim_slope_b = fmin(fabs(del_ge_C), fabs(del_ge_G));
    del_ge_m_i  = sgn_CUDA(del_ge_C) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_scalar_m_i[i] = 0.0;
    if (del_scalar_L[i] * del_scalar_R[i] > 0.0) {
      lim_slope_a       = fmin(fabs(del_scalar_L[i]), fabs(del_scalar_R[i]));
      lim_slope_b       = fmin(fabs(del_scalar_C[i]), fabs(del_scalar_G[i]));
      del_scalar_m_i[i] = sgn_CUDA(del_scalar_C[i]) * fmin(2.0 * lim_slope_a, lim_slope_b);
    }
  }
#endif  // SCALAR

  // Project the monotonized difference in the characteristic variables back
  // onto the primitive variables Stone Eqn 39
  del_d_m_i  = del_a_0_m + del_a_1_m + del_a_4_m;
  del_vx_m_i = -a_i * del_a_0_m / cell_i.density + a_i * del_a_4_m / cell_i.density;
  del_vy_m_i = del_a_2_m;
  del_vz_m_i = del_a_3_m;
  del_p_m_i  = a_i * a_i * del_a_0_m + a_i * a_i * del_a_4_m;

  // Compute the left and right interface values using the monotonized
  // difference in the primitive variables

  d_R_imh  = cell_i.density - 0.5 * del_d_m_i;
  vx_R_imh = cell_i.velocity_x - 0.5 * del_vx_m_i;
  vy_R_imh = cell_i.velocity_y - 0.5 * del_vy_m_i;
  vz_R_imh = cell_i.velocity_z - 0.5 * del_vz_m_i;
  p_R_imh  = cell_i.pressure - 0.5 * del_p_m_i;

  d_L_iph  = cell_i.density + 0.5 * del_d_m_i;
  vx_L_iph = cell_i.velocity_x + 0.5 * del_vx_m_i;
  vy_L_iph = cell_i.velocity_y + 0.5 * del_vy_m_i;
  vz_L_iph = cell_i.velocity_z + 0.5 * del_vz_m_i;
  p_L_iph  = cell_i.pressure + 0.5 * del_p_m_i;

#ifdef DE
  ge_R_imh = ge_i - 0.5 * del_ge_m_i;
  ge_L_iph = ge_i + 0.5 * del_ge_m_i;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    scalar_R_imh[i] = scalar_i[i] - 0.5 * del_scalar_m_i[i];
    scalar_L_iph[i] = scalar_i[i] + 0.5 * del_scalar_m_i[i];
  }
#endif  // SCALAR

  // try removing this on shock tubes
  C       = d_R_imh + d_L_iph;
  d_R_imh = fmax(fmin(cell_i.density, cell_imo.density), d_R_imh);
  d_R_imh = fmin(fmax(cell_i.density, cell_imo.density), d_R_imh);
  d_L_iph = C - d_R_imh;
  d_L_iph = fmax(fmin(cell_i.density, cell_ipo.density), d_L_iph);
  d_L_iph = fmin(fmax(cell_i.density, cell_ipo.density), d_L_iph);
  d_R_imh = C - d_L_iph;

  C        = vx_R_imh + vx_L_iph;
  vx_R_imh = fmax(fmin(cell_i.velocity_x, cell_imo.velocity_x), vx_R_imh);
  vx_R_imh = fmin(fmax(cell_i.velocity_x, cell_imo.velocity_x), vx_R_imh);
  vx_L_iph = C - vx_R_imh;
  vx_L_iph = fmax(fmin(cell_i.velocity_x, cell_ipo.velocity_x), vx_L_iph);
  vx_L_iph = fmin(fmax(cell_i.velocity_x, cell_ipo.velocity_x), vx_L_iph);
  vx_R_imh = C - vx_L_iph;

  C        = vy_R_imh + vy_L_iph;
  vy_R_imh = fmax(fmin(cell_i.velocity_y, cell_imo.velocity_y), vy_R_imh);
  vy_R_imh = fmin(fmax(cell_i.velocity_y, cell_imo.velocity_y), vy_R_imh);
  vy_L_iph = C - vy_R_imh;
  vy_L_iph = fmax(fmin(cell_i.velocity_y, cell_ipo.velocity_y), vy_L_iph);
  vy_L_iph = fmin(fmax(cell_i.velocity_y, cell_ipo.velocity_y), vy_L_iph);
  vy_R_imh = C - vy_L_iph;

  C        = vz_R_imh + vz_L_iph;
  vz_R_imh = fmax(fmin(cell_i.velocity_z, cell_imo.velocity_z), vz_R_imh);
  vz_R_imh = fmin(fmax(cell_i.velocity_z, cell_imo.velocity_z), vz_R_imh);
  vz_L_iph = C - vz_R_imh;
  vz_L_iph = fmax(fmin(cell_i.velocity_z, cell_ipo.velocity_z), vz_L_iph);
  vz_L_iph = fmin(fmax(cell_i.velocity_z, cell_ipo.velocity_z), vz_L_iph);
  vz_R_imh = C - vz_L_iph;

  C       = p_R_imh + p_L_iph;
  p_R_imh = fmax(fmin(cell_i.pressure, cell_imo.pressure), p_R_imh);
  p_R_imh = fmin(fmax(cell_i.pressure, cell_imo.pressure), p_R_imh);
  p_L_iph = C - p_R_imh;
  p_L_iph = fmax(fmin(cell_i.pressure, cell_ipo.pressure), p_L_iph);
  p_L_iph = fmin(fmax(cell_i.pressure, cell_ipo.pressure), p_L_iph);
  p_R_imh = C - p_L_iph;

  del_d_m_i  = d_L_iph - d_R_imh;
  del_vx_m_i = vx_L_iph - vx_R_imh;
  del_vy_m_i = vy_L_iph - vy_R_imh;
  del_vz_m_i = vz_L_iph - vz_R_imh;
  del_p_m_i  = p_L_iph - p_R_imh;

#ifdef DE
  C          = ge_R_imh + ge_L_iph;
  ge_R_imh   = fmax(fmin(ge_i, ge_imo), ge_R_imh);
  ge_R_imh   = fmin(fmax(ge_i, ge_imo), ge_R_imh);
  ge_L_iph   = C - ge_R_imh;
  ge_L_iph   = fmax(fmin(ge_i, ge_ipo), ge_L_iph);
  ge_L_iph   = fmin(fmax(ge_i, ge_ipo), ge_L_iph);
  ge_R_imh   = C - ge_L_iph;
  del_ge_m_i = ge_L_iph - ge_R_imh;
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    C                 = scalar_R_imh[i] + scalar_L_iph[i];
    scalar_R_imh[i]   = fmax(fmin(scalar_i[i], scalar_imo[i]), scalar_R_imh[i]);
    scalar_R_imh[i]   = fmin(fmax(scalar_i[i], scalar_imo[i]), scalar_R_imh[i]);
    scalar_L_iph[i]   = C - scalar_R_imh[i];
    scalar_L_iph[i]   = fmax(fmin(scalar_i[i], scalar_ipo[i]), scalar_L_iph[i]);
    scalar_L_iph[i]   = fmin(fmax(scalar_i[i], scalar_ipo[i]), scalar_L_iph[i]);
    scalar_R_imh[i]   = C - scalar_L_iph[i];
    del_scalar_m_i[i] = scalar_L_iph[i] - scalar_R_imh[i];
  }
#endif  // SCALAR

#ifndef VL
  // Integrate linear interpolation function over domain of dependence
  // defined by max(min) eigenvalue
  qx       = -0.5 * fmin(lambda_m, 0.0) * dtodx;
  d_R_imh  = d_R_imh + qx * del_d_m_i;
  vx_R_imh = vx_R_imh + qx * del_vx_m_i;
  vy_R_imh = vy_R_imh + qx * del_vy_m_i;
  vz_R_imh = vz_R_imh + qx * del_vz_m_i;
  p_R_imh  = p_R_imh + qx * del_p_m_i;

  qx       = 0.5 * fmax(lambda_p, 0.0) * dtodx;
  d_L_iph  = d_L_iph - qx * del_d_m_i;
  vx_L_iph = vx_L_iph - qx * del_vx_m_i;
  vy_L_iph = vy_L_iph - qx * del_vy_m_i;
  vz_L_iph = vz_L_iph - qx * del_vz_m_i;
  p_L_iph  = p_L_iph - qx * del_p_m_i;

  #ifdef DE
  ge_R_imh = ge_R_imh + qx * del_ge_m_i;
  ge_L_iph = ge_L_iph - qx * del_ge_m_i;
  #endif  // DE

  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    scalar_R_imh[i] = scalar_R_imh[i] + qx * del_scalar_m_i[i];
    scalar_L_iph[i] = scalar_L_iph[i] - qx * del_scalar_m_i[i];
  }
  #endif  // SCALAR

  // Perform the characteristic tracing
  // Stone Eqns 42 & 43

  // left-hand interface value, i+1/2
  sum_0 = sum_1 = sum_2 = sum_3 = sum_4 = 0;
  #ifdef DE
  sum_ge = 0;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    sum_scalar[i] = 0.0;
  }
  #endif  // SCALAR
  if (lambda_m >= 0) {
    lamdiff = lambda_p - lambda_m;

    sum_0 += lamdiff * (-cell_i.density * del_vx_m_i / (2 * a_i) + del_p_m_i / (2 * a_i * a_i));
    sum_1 += lamdiff * (del_vx_m_i / 2.0 - del_p_m_i / (2 * a_i * cell_i.density));
    sum_4 += lamdiff * (-cell_i.density * del_vx_m_i * a_i / 2.0 + del_p_m_i / 2.0);
  }
  if (lambda_0 >= 0) {
    lamdiff = lambda_p - lambda_0;

    sum_0 += lamdiff * (del_d_m_i - del_p_m_i / (a_i * a_i));
    sum_2 += lamdiff * del_vy_m_i;
    sum_3 += lamdiff * del_vz_m_i;
  #ifdef DE
    sum_ge += lamdiff * del_ge_m_i;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += lamdiff * del_scalar_m_i[i];
    }
  #endif  // SCALAR
  }
  if (lambda_p >= 0) {
    lamdiff = lambda_p - lambda_p;

    sum_0 += lamdiff * (cell_i.density * del_vx_m_i / (2 * a_i) + del_p_m_i / (2 * a_i * a_i));
    sum_1 += lamdiff * (del_vx_m_i / 2.0 + del_p_m_i / (2 * a_i * cell_i.density));
    sum_4 += lamdiff * (cell_i.density * del_vx_m_i * a_i / 2.0 + del_p_m_i / 2.0);
  }

  // add the corrections to the initial guesses for the interface values
  d_L_iph += 0.5 * dtodx * sum_0;
  vx_L_iph += 0.5 * dtodx * sum_1;
  vy_L_iph += 0.5 * dtodx * sum_2;
  vz_L_iph += 0.5 * dtodx * sum_3;
  p_L_iph += 0.5 * dtodx * sum_4;
  #ifdef DE
  ge_L_iph += 0.5 * dtodx * sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    scalar_L_iph[i] += 0.5 * dtodx * sum_scalar[i];
  }
  #endif  // SCALAR

  // right-hand interface value, i-1/2
  sum_0 = sum_1 = sum_2 = sum_3 = sum_4 = 0;
  #ifdef DE
  sum_ge = 0;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    sum_scalar[i] = 0;
  }
  #endif  // SCALAR
  if (lambda_m <= 0) {
    lamdiff = lambda_m - lambda_m;

    sum_0 += lamdiff * (-cell_i.density * del_vx_m_i / (2 * a_i) + del_p_m_i / (2 * a_i * a_i));
    sum_1 += lamdiff * (del_vx_m_i / 2.0 - del_p_m_i / (2 * a_i * cell_i.density));
    sum_4 += lamdiff * (-cell_i.density * del_vx_m_i * a_i / 2.0 + del_p_m_i / 2.0);
  }
  if (lambda_0 <= 0) {
    lamdiff = lambda_m - lambda_0;

    sum_0 += lamdiff * (del_d_m_i - del_p_m_i / (a_i * a_i));
    sum_2 += lamdiff * del_vy_m_i;
    sum_3 += lamdiff * del_vz_m_i;
  #ifdef DE
    sum_ge += lamdiff * del_ge_m_i;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += lamdiff * del_scalar_m_i[i];
    }
  #endif  // SCALAR
  }
  if (lambda_p <= 0) {
    lamdiff = lambda_m - lambda_p;

    sum_0 += lamdiff * (cell_i.density * del_vx_m_i / (2 * a_i) + del_p_m_i / (2 * a_i * a_i));
    sum_1 += lamdiff * (del_vx_m_i / 2.0 + del_p_m_i / (2 * a_i * cell_i.density));
    sum_4 += lamdiff * (cell_i.density * del_vx_m_i * a_i / 2.0 + del_p_m_i / 2.0);
  }

  // add the corrections
  d_R_imh += 0.5 * dtodx * sum_0;
  vx_R_imh += 0.5 * dtodx * sum_1;
  vy_R_imh += 0.5 * dtodx * sum_2;
  vz_R_imh += 0.5 * dtodx * sum_3;
  p_R_imh += 0.5 * dtodx * sum_4;
  #ifdef DE
  ge_R_imh += 0.5 * dtodx * sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    scalar_R_imh[i] += 0.5 * dtodx * sum_scalar[i];
  }
  #endif  // SCALAR
#endif    // CTU

  // apply minimum constraints
  d_R_imh = fmax(d_R_imh, (Real)TINY_NUMBER);
  d_L_iph = fmax(d_L_iph, (Real)TINY_NUMBER);
  p_R_imh = fmax(p_R_imh, (Real)TINY_NUMBER);
  p_L_iph = fmax(p_L_iph, (Real)TINY_NUMBER);

  // Convert the left and right states in the primitive to the conserved
  // variables send final values back from kernel bounds_R refers to the right
  // side of the i-1/2 interface
  int id;
  switch (dir) {
    case 0:
      id = xid - 1 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid - 1) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid - 1) * nx * ny;
      break;
  }

  dev_bounds_R[id]                = d_R_imh;
  dev_bounds_R[o1 * n_cells + id] = d_R_imh * vx_R_imh;
  dev_bounds_R[o2 * n_cells + id] = d_R_imh * vy_R_imh;
  dev_bounds_R[o3 * n_cells + id] = d_R_imh * vz_R_imh;
  dev_bounds_R[4 * n_cells + id] =
      (p_R_imh / (gamma - 1.0)) + 0.5 * d_R_imh * (vx_R_imh * vx_R_imh + vy_R_imh * vy_R_imh + vz_R_imh * vz_R_imh);
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_bounds_R[(5 + i) * n_cells + id] = d_R_imh * scalar_R_imh[i];
  }
#endif  // SCALAR
#ifdef DE
  dev_bounds_R[(n_fields - 1) * n_cells + id] = d_R_imh * ge_R_imh;
#endif  // DE
  // bounds_L refers to the left side of the i+1/2 interface
  id                              = xid + yid * nx + zid * nx * ny;
  dev_bounds_L[id]                = d_L_iph;
  dev_bounds_L[o1 * n_cells + id] = d_L_iph * vx_L_iph;
  dev_bounds_L[o2 * n_cells + id] = d_L_iph * vy_L_iph;
  dev_bounds_L[o3 * n_cells + id] = d_L_iph * vz_L_iph;
  dev_bounds_L[4 * n_cells + id] =
      (p_L_iph / (gamma - 1.0)) + 0.5 * d_L_iph * (vx_L_iph * vx_L_iph + vy_L_iph * vy_L_iph + vz_L_iph * vz_L_iph);
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_bounds_L[(5 + i) * n_cells + id] = d_L_iph * scalar_L_iph[i];
  }
#endif  // SCALAR
#ifdef DE
  dev_bounds_L[(n_fields - 1) * n_cells + id] = d_L_iph * ge_L_iph;
#endif  // DE
}

// =============================================================================
plmc_utils::PlmcPrimitive __device__ __host__ plmc_utils::Load_Data(
    Real const *dev_conserved, size_t const &xid, size_t const &yid, size_t const &zid, size_t const &nx,
    size_t const &ny, size_t const &n_cells, size_t const &o1, size_t const &o2, size_t const &o3, Real const &gamma)
{
  // Compute index
  size_t const id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);

  // Declare the variable we will return
  PlmcPrimitive loaded_data;

  // Load hydro variables except pressure
  loaded_data.density    = dev_conserved[grid_enum::density * n_cells + id];
  loaded_data.velocity_x = dev_conserved[o1 * n_cells + id] / loaded_data.density;
  loaded_data.velocity_y = dev_conserved[o2 * n_cells + id] / loaded_data.density;
  loaded_data.velocity_z = dev_conserved[o3 * n_cells + id] / loaded_data.density;

  // Load MHD variables. Note that I only need the centered values for the transverse fields except for the initial
  // computation of the primitive variables
#ifdef MHD
  auto magnetic_centered = mhd::utils::cellCenteredMagneticFields(dev_conserved, id, xid, yid, zid, n_cells, nx, ny);
  switch (o1) {
    case grid_enum::momentum_x:
      loaded_data.magnetic_y = magnetic_centered.y;
      loaded_data.magnetic_z = magnetic_centered.z;
      break;
    case grid_enum::momentum_y:
      loaded_data.magnetic_y = magnetic_centered.z;
      loaded_data.magnetic_z = magnetic_centered.x;
      break;
    case grid_enum::momentum_z:
      loaded_data.magnetic_y = magnetic_centered.x;
      loaded_data.magnetic_z = magnetic_centered.y;
      break;
  }
#endif  // MHD

// Load pressure accounting for duel energy if enabled
#ifdef DE  // DE
  Real const E          = dev_conserved[grid_enum::Energy * n_cells + id];
  Real const gas_energy = dev_conserved[grid_enum::GasEnergy * n_cells + id];

  Real E_non_thermal = hydro_utilities::Calc_Kinetic_Energy_From_Velocity(
      loaded_data.density, loaded_data.velocity_x, loaded_data.velocity_y, loaded_data.velocity_z);

  #ifdef MHD
  E_non_thermal += mhd::utils::computeMagneticEnergy(magnetic_centered.x, magnetic_centered.y, magnetic_centered.z);
  #endif  // MHD

  loaded_data.pressure   = hydro_utilities::Get_Pressure_From_DE(E, E - E_non_thermal, gas_energy, gamma);
  loaded_data.gas_energy = gas_energy / loaded_data.density;
#else  // not DE
  #ifdef MHD
  loaded_data.pressure =
      hydro_utilities::Calc_Pressure_Primitive(dev_conserved[grid_enum::Energy * n_cells + id], loaded_data.density,
                                               loaded_data.velocity_x, loaded_data.velocity_y, loaded_data.velocity_z,
                                               gamma, magnetic_centered.x, magnetic_centered.y, magnetic_centered.z);
  #else   // not MHD
  loaded_data.pressure = hydro_utilities::Calc_Pressure_Primitive(
      dev_conserved[grid_enum::Energy * n_cells + id], loaded_data.density, loaded_data.velocity_x,
      loaded_data.velocity_y, loaded_data.velocity_z, gamma);
  #endif  // MHD
#endif    // DE

#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    loaded_data.scalar[i] = dev_conserved[(grid_enum::scalar + i) * n_cells + id] / loaded_data.density;
  }
#endif  // SCALAR

  return loaded_data;
}
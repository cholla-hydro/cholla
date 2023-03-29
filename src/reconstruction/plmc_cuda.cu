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
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Thread guard to prevent overrun
  if (xid < 1 or xid >= nx - 2 or yid < 1 or yid >= ny - 2 or zid < 1 or zid >= nz - 2) {
    return;
  }

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
  Real const sound_speed         = hydro_utilities::Calc_Sound_Speed(cell_i.pressure, cell_i.density, gamma);
  Real const sound_speed_squared = sound_speed * sound_speed;

  // Compute the left, right, centered, and van Leer differences of the
  // primitive variables Note that here L and R refer to locations relative to
  // the cell center

  // left
  plmc_utils::PlmcPrimitive const del_L = plmc_utils::Compute_Slope(cell_i, cell_imo);

  // right
  plmc_utils::PlmcPrimitive const del_R = plmc_utils::Compute_Slope(cell_ipo, cell_i);

  // centered
  plmc_utils::PlmcPrimitive const del_C = plmc_utils::Compute_Slope(cell_ipo, cell_imo, 0.5);

  // Van Leer
  plmc_utils::PlmcPrimitive const del_G = plmc_utils::Van_Leer_Slope(del_L, del_R);

  // Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  plmc_utils::PlmcCharacteristic del_a_L, del_a_R, del_a_C, del_a_G, del_a_m;
  del_a_L.a0 =
      -cell_i.density * del_L.velocity_x / (2 * sound_speed) + del_L.pressure / (2 * sound_speed * sound_speed);
  del_a_L.a1 = del_L.density - del_L.pressure / (sound_speed_squared);
  del_a_L.a2 = del_L.velocity_y;
  del_a_G.a3 = del_L.velocity_z;
  del_a_L.a4 = cell_i.density * del_L.velocity_x / (2 * sound_speed) + del_L.pressure / (2 * sound_speed_squared);

  del_a_R.a0 = -cell_i.density * del_R.velocity_x / (2 * sound_speed) + del_R.pressure / (2 * sound_speed_squared);
  del_a_R.a1 = del_R.density - del_R.pressure / (sound_speed_squared);
  del_a_R.a2 = del_R.velocity_y;
  del_a_R.a3 = del_R.velocity_z;
  del_a_R.a4 = cell_i.density * del_R.velocity_x / (2 * sound_speed) + del_R.pressure / (2 * sound_speed_squared);

  del_a_C.a0 = -cell_i.density * del_C.velocity_x / (2 * sound_speed) + del_C.pressure / (2 * sound_speed_squared);
  del_a_C.a1 = del_C.density - del_C.pressure / (sound_speed_squared);
  del_a_C.a2 = del_C.velocity_y;
  del_a_C.a3 = del_C.velocity_z;
  del_a_C.a4 = cell_i.density * del_C.velocity_x / (2 * sound_speed) + del_C.pressure / (2 * sound_speed_squared);

  del_a_G.a0 = -cell_i.density * del_G.velocity_x / (2 * sound_speed) + del_G.pressure / (2 * sound_speed_squared);
  del_a_G.a1 = del_G.density - del_G.pressure / (sound_speed_squared);
  del_a_G.a2 = del_G.velocity_y;
  del_a_G.a3 = del_G.velocity_z;
  del_a_G.a4 = cell_i.density * del_G.velocity_x / (2 * sound_speed) + del_G.pressure / (2 * sound_speed_squared);

  // Apply monotonicity constraints to the differences in the characteristic
  // variables

  del_a_m.a0 = del_a_m.a1 = del_a_m.a2 = del_a_m.a3 = del_a_m.a4 = 0.0;  // This should be in the declaration
  plmc_utils::PlmcPrimitive del_m_i;
  if (del_a_L.a0 * del_a_R.a0 > 0.0) {
    Real const lim_slope_a = fmin(fabs(del_a_L.a0), fabs(del_a_R.a0));
    Real const lim_slope_b = fmin(fabs(del_a_C.a0), fabs(del_a_G.a0));
    del_a_m.a0             = sgn_CUDA(del_a_C.a0) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a1 * del_a_R.a1 > 0.0) {
    Real const lim_slope_a = fmin(fabs(del_a_L.a1), fabs(del_a_R.a1));
    Real const lim_slope_b = fmin(fabs(del_a_C.a1), fabs(del_a_G.a1));
    del_a_m.a1             = sgn_CUDA(del_a_C.a1) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a2 * del_a_R.a2 > 0.0) {
    Real const lim_slope_a = fmin(fabs(del_a_L.a2), fabs(del_a_R.a2));
    Real const lim_slope_b = fmin(fabs(del_a_C.a2), fabs(del_a_G.a2));
    del_a_m.a2             = sgn_CUDA(del_a_C.a2) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_G.a3 * del_a_R.a3 > 0.0) {
    Real const lim_slope_a = fmin(fabs(del_a_G.a3), fabs(del_a_R.a3));
    Real const lim_slope_b = fmin(fabs(del_a_C.a3), fabs(del_a_G.a3));
    del_a_m.a3             = sgn_CUDA(del_a_C.a3) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a4 * del_a_R.a4 > 0.0) {
    Real const lim_slope_a = fmin(fabs(del_a_L.a4), fabs(del_a_R.a4));
    Real const lim_slope_b = fmin(fabs(del_a_C.a4), fabs(del_a_G.a4));
    del_a_m.a4             = sgn_CUDA(del_a_C.a4) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
#ifdef DE
  del_m_i.gas_energy = 0.0;
  if (del_L.gas_energy * del_R.gas_energy > 0.0) {
    Real const lim_slope_a = fmin(fabs(del_L.gas_energy), fabs(del_R.gas_energy));
    Real const lim_slope_b = fmin(fabs(del_C.gas_energy), fabs(del_G.gas_energy));
    del_m_i.gas_energy     = sgn_CUDA(del_C.gas_energy) * fmin(2.0 * lim_slope_a, lim_slope_b);
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_m_i.scalar[i] = 0.0;
    if (del_L.scalar[i] * del_R.scalar[i] > 0.0) {
      Real const lim_slope_a = fmin(fabs(del_L.scalar[i]), fabs(del_R.scalar[i]));
      Real const lim_slope_b = fmin(fabs(del_C.scalar[i]), fabs(del_G.scalar[i]));
      del_m_i.scalar[i]      = sgn_CUDA(del_C.scalar[i]) * fmin(2.0 * lim_slope_a, lim_slope_b);
    }
  }
#endif  // SCALAR

  // Project the monotonized difference in the characteristic variables back
  // onto the primitive variables Stone Eqn 39
  del_m_i.density    = del_a_m.a0 + del_a_m.a1 + del_a_m.a4;
  del_m_i.velocity_x = -sound_speed * del_a_m.a0 / cell_i.density + sound_speed * del_a_m.a4 / cell_i.density;
  del_m_i.velocity_y = del_a_m.a2;
  del_m_i.velocity_z = del_a_m.a3;
  del_m_i.pressure   = sound_speed_squared * del_a_m.a0 + sound_speed_squared * del_a_m.a4;

  // Compute the left and right interface values using the monotonized
  // difference in the primitive variables

  plmc_utils::PlmcPrimitive interface_L_iph, interface_R_imh;

  interface_R_imh.density    = cell_i.density - 0.5 * del_m_i.density;
  interface_R_imh.velocity_x = cell_i.velocity_x - 0.5 * del_m_i.velocity_x;
  interface_R_imh.velocity_y = cell_i.velocity_y - 0.5 * del_m_i.velocity_y;
  interface_R_imh.velocity_z = cell_i.velocity_z - 0.5 * del_m_i.velocity_z;
  interface_R_imh.pressure   = cell_i.pressure - 0.5 * del_m_i.pressure;

  interface_L_iph.density    = cell_i.density + 0.5 * del_m_i.density;
  interface_L_iph.velocity_x = cell_i.velocity_x + 0.5 * del_m_i.velocity_x;
  interface_L_iph.velocity_y = cell_i.velocity_y + 0.5 * del_m_i.velocity_y;
  interface_L_iph.velocity_z = cell_i.velocity_z + 0.5 * del_m_i.velocity_z;
  interface_L_iph.pressure   = cell_i.pressure + 0.5 * del_m_i.pressure;

#ifdef DE
  interface_R_imh.gas_energy = cell_i.gas_energy - 0.5 * del_m_i.gas_energy;
  interface_L_iph.gas_energy = cell_i.gas_energy + 0.5 * del_m_i.gas_energy;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar[i] = cell_i.scalar[i] - 0.5 * del_m_i.scalar[i];
    interface_L_iph.scalar[i] = cell_i.scalar[i] + 0.5 * del_m_i.scalar[i];
  }
#endif  // SCALAR

  // try removing this on shock tubes
  Real C                  = interface_R_imh.density + interface_L_iph.density;
  interface_R_imh.density = fmax(fmin(cell_i.density, cell_imo.density), interface_R_imh.density);
  interface_R_imh.density = fmin(fmax(cell_i.density, cell_imo.density), interface_R_imh.density);
  interface_L_iph.density = C - interface_R_imh.density;
  interface_L_iph.density = fmax(fmin(cell_i.density, cell_ipo.density), interface_L_iph.density);
  interface_L_iph.density = fmin(fmax(cell_i.density, cell_ipo.density), interface_L_iph.density);
  interface_R_imh.density = C - interface_L_iph.density;

  C                          = interface_R_imh.velocity_x + interface_L_iph.velocity_x;
  interface_R_imh.velocity_x = fmax(fmin(cell_i.velocity_x, cell_imo.velocity_x), interface_R_imh.velocity_x);
  interface_R_imh.velocity_x = fmin(fmax(cell_i.velocity_x, cell_imo.velocity_x), interface_R_imh.velocity_x);
  interface_L_iph.velocity_x = C - interface_R_imh.velocity_x;
  interface_L_iph.velocity_x = fmax(fmin(cell_i.velocity_x, cell_ipo.velocity_x), interface_L_iph.velocity_x);
  interface_L_iph.velocity_x = fmin(fmax(cell_i.velocity_x, cell_ipo.velocity_x), interface_L_iph.velocity_x);
  interface_R_imh.velocity_x = C - interface_L_iph.velocity_x;

  C                          = interface_R_imh.velocity_y + interface_L_iph.velocity_y;
  interface_R_imh.velocity_y = fmax(fmin(cell_i.velocity_y, cell_imo.velocity_y), interface_R_imh.velocity_y);
  interface_R_imh.velocity_y = fmin(fmax(cell_i.velocity_y, cell_imo.velocity_y), interface_R_imh.velocity_y);
  interface_L_iph.velocity_y = C - interface_R_imh.velocity_y;
  interface_L_iph.velocity_y = fmax(fmin(cell_i.velocity_y, cell_ipo.velocity_y), interface_L_iph.velocity_y);
  interface_L_iph.velocity_y = fmin(fmax(cell_i.velocity_y, cell_ipo.velocity_y), interface_L_iph.velocity_y);
  interface_R_imh.velocity_y = C - interface_L_iph.velocity_y;

  C                          = interface_R_imh.velocity_z + interface_L_iph.velocity_z;
  interface_R_imh.velocity_z = fmax(fmin(cell_i.velocity_z, cell_imo.velocity_z), interface_R_imh.velocity_z);
  interface_R_imh.velocity_z = fmin(fmax(cell_i.velocity_z, cell_imo.velocity_z), interface_R_imh.velocity_z);
  interface_L_iph.velocity_z = C - interface_R_imh.velocity_z;
  interface_L_iph.velocity_z = fmax(fmin(cell_i.velocity_z, cell_ipo.velocity_z), interface_L_iph.velocity_z);
  interface_L_iph.velocity_z = fmin(fmax(cell_i.velocity_z, cell_ipo.velocity_z), interface_L_iph.velocity_z);
  interface_R_imh.velocity_z = C - interface_L_iph.velocity_z;

  C                        = interface_R_imh.pressure + interface_L_iph.pressure;
  interface_R_imh.pressure = fmax(fmin(cell_i.pressure, cell_imo.pressure), interface_R_imh.pressure);
  interface_R_imh.pressure = fmin(fmax(cell_i.pressure, cell_imo.pressure), interface_R_imh.pressure);
  interface_L_iph.pressure = C - interface_R_imh.pressure;
  interface_L_iph.pressure = fmax(fmin(cell_i.pressure, cell_ipo.pressure), interface_L_iph.pressure);
  interface_L_iph.pressure = fmin(fmax(cell_i.pressure, cell_ipo.pressure), interface_L_iph.pressure);
  interface_R_imh.pressure = C - interface_L_iph.pressure;

  del_m_i.density    = interface_L_iph.density - interface_R_imh.density;
  del_m_i.velocity_x = interface_L_iph.velocity_x - interface_R_imh.velocity_x;
  del_m_i.velocity_y = interface_L_iph.velocity_y - interface_R_imh.velocity_y;
  del_m_i.velocity_z = interface_L_iph.velocity_z - interface_R_imh.velocity_z;
  del_m_i.pressure   = interface_L_iph.pressure - interface_R_imh.pressure;

#ifdef DE
  C                          = interface_R_imh.gas_energy + interface_L_iph.gas_energy;
  interface_R_imh.gas_energy = fmax(fmin(cell_i.gas_energy, cell_imo.gas_energy), interface_R_imh.gas_energy);
  interface_R_imh.gas_energy = fmin(fmax(cell_i.gas_energy, cell_imo.gas_energy), interface_R_imh.gas_energy);
  interface_L_iph.gas_energy = C - interface_R_imh.gas_energy;
  interface_L_iph.gas_energy = fmax(fmin(cell_i.gas_energy, cell_ipo.gas_energy), interface_L_iph.gas_energy);
  interface_L_iph.gas_energy = fmin(fmax(cell_i.gas_energy, cell_ipo.gas_energy), interface_L_iph.gas_energy);
  interface_R_imh.gas_energy = C - interface_L_iph.gas_energy;
  del_m_i.gas_energy         = interface_L_iph.gas_energy - interface_R_imh.gas_energy;
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    C                         = interface_R_imh.scalar[i] + interface_L_iph.scalar[i];
    interface_R_imh.scalar[i] = fmax(fmin(cell_i.scalar[i], cell_imo.scalar[i]), interface_R_imh.scalar[i]);
    interface_R_imh.scalar[i] = fmin(fmax(cell_i.scalar[i], cell_imo.scalar[i]), interface_R_imh.scalar[i]);
    interface_L_iph.scalar[i] = C - interface_R_imh.scalar[i];
    interface_L_iph.scalar[i] = fmax(fmin(cell_i.scalar[i], cell_ipo.scalar[i]), interface_L_iph.scalar[i]);
    interface_L_iph.scalar[i] = fmin(fmax(cell_i.scalar[i], cell_ipo.scalar[i]), interface_L_iph.scalar[i]);
    interface_R_imh.scalar[i] = C - interface_L_iph.scalar[i];
    del_m_i.scalar[i]         = interface_L_iph.scalar[i] - interface_R_imh.scalar[i];
  }
#endif  // SCALAR

#ifndef VL

  Real const dtodx = dt / dx;

  // Compute the eigenvalues of the linearized equations in the
  // primitive variables using the cell-centered primitive variables
  Real const lambda_m = cell_i.velocity_x - sound_speed;
  Real const lambda_0 = cell_i.velocity_x;
  Real const lambda_p = cell_i.velocity_x + sound_speed;

  // Integrate linear interpolation function over domain of dependence
  // defined by max(min) eigenvalue
  Real qx                    = -0.5 * fmin(lambda_m, 0.0) * dtodx;
  interface_R_imh.density    = interface_R_imh.density + qx * del_m_i.density;
  interface_R_imh.velocity_x = interface_R_imh.velocity_x + qx * del_m_i.velocity_x;
  interface_R_imh.velocity_y = interface_R_imh.velocity_y + qx * del_m_i.velocity_y;
  interface_R_imh.velocity_z = interface_R_imh.velocity_z + qx * del_m_i.velocity_z;
  interface_R_imh.pressure   = interface_R_imh.pressure + qx * del_m_i.pressure;

  qx                         = 0.5 * fmax(lambda_p, 0.0) * dtodx;
  interface_L_iph.density    = interface_L_iph.density - qx * del_m_i.density;
  interface_L_iph.velocity_x = interface_L_iph.velocity_x - qx * del_m_i.velocity_x;
  interface_L_iph.velocity_y = interface_L_iph.velocity_y - qx * del_m_i.velocity_y;
  interface_L_iph.velocity_z = interface_L_iph.velocity_z - qx * del_m_i.velocity_z;
  interface_L_iph.pressure   = interface_L_iph.pressure - qx * del_m_i.pressure;

  #ifdef DE
  interface_R_imh.gas_energy = interface_R_imh.gas_energy + qx * del_m_i.gas_energy;
  interface_L_iph.gas_energy = interface_L_iph.gas_energy - qx * del_m_i.gas_energy;
  #endif  // DE

  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar[i] = interface_R_imh.scalar[i] + qx * del_m_i.scalar[i];
    interface_L_iph.scalar[i] = interface_L_iph.scalar[i] - qx * del_m_i.scalar[i];
  }
  #endif  // SCALAR

  // Perform the characteristic tracing
  // Stone Eqns 42 & 43

  // left-hand interface value, i+1/2
  Real sum_0 = 0.0, sum_1 = 0.0, sum_2 = 0.0, sum_3 = 0.0, sum_4 = 0.0;
  #ifdef DE
  Real sum_ge = 0;
  #endif  // DE
  #ifdef SCALAR
  Real sum_scalar[NSCALARS];
  for (int i = 0; i < NSCALARS; i++) {
    sum_scalar[i] = 0.0;
  }
  #endif  // SCALAR
  if (lambda_m >= 0) {
    Real lamdiff = lambda_p - lambda_m;

    sum_0 += lamdiff *
             (-cell_i.density * del_m_i.velocity_x / (2 * sound_speed) + del_m_i.pressure / (2 * sound_speed_squared));
    sum_1 += lamdiff * (del_m_i.velocity_x / 2.0 - del_m_i.pressure / (2 * sound_speed * cell_i.density));
    sum_4 += lamdiff * (-cell_i.density * del_m_i.velocity_x * sound_speed / 2.0 + del_m_i.pressure / 2.0);
  }
  if (lambda_0 >= 0) {
    Real lamdiff = lambda_p - lambda_0;

    sum_0 += lamdiff * (del_m_i.density - del_m_i.pressure / (sound_speed_squared));
    sum_2 += lamdiff * del_m_i.velocity_y;
    sum_3 += lamdiff * del_m_i.velocity_z;
  #ifdef DE
    sum_ge += lamdiff * del_m_i.gas_energy;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += lamdiff * del_m_i.scalar[i];
    }
  #endif  // SCALAR
  }
  if (lambda_p >= 0) {
    Real lamdiff = lambda_p - lambda_p;

    sum_0 += lamdiff *
             (cell_i.density * del_m_i.velocity_x / (2 * sound_speed) + del_m_i.pressure / (2 * sound_speed_squared));
    sum_1 += lamdiff * (del_m_i.velocity_x / 2.0 + del_m_i.pressure / (2 * sound_speed * cell_i.density));
    sum_4 += lamdiff * (cell_i.density * del_m_i.velocity_x * sound_speed / 2.0 + del_m_i.pressure / 2.0);
  }

  // add the corrections to the initial guesses for the interface values
  interface_L_iph.density += 0.5 * dtodx * sum_0;
  interface_L_iph.velocity_x += 0.5 * dtodx * sum_1;
  interface_L_iph.velocity_y += 0.5 * dtodx * sum_2;
  interface_L_iph.velocity_z += 0.5 * dtodx * sum_3;
  interface_L_iph.pressure += 0.5 * dtodx * sum_4;
  #ifdef DE
  interface_L_iph.gas_energy += 0.5 * dtodx * sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar[i] += 0.5 * dtodx * sum_scalar[i];
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
    Real lamdiff = lambda_m - lambda_m;

    sum_0 += lamdiff *
             (-cell_i.density * del_m_i.velocity_x / (2 * sound_speed) + del_m_i.pressure / (2 * sound_speed_squared));
    sum_1 += lamdiff * (del_m_i.velocity_x / 2.0 - del_m_i.pressure / (2 * sound_speed * cell_i.density));
    sum_4 += lamdiff * (-cell_i.density * del_m_i.velocity_x * sound_speed / 2.0 + del_m_i.pressure / 2.0);
  }
  if (lambda_0 <= 0) {
    Real lamdiff = lambda_m - lambda_0;

    sum_0 += lamdiff * (del_m_i.density - del_m_i.pressure / (sound_speed_squared));
    sum_2 += lamdiff * del_m_i.velocity_y;
    sum_3 += lamdiff * del_m_i.velocity_z;
  #ifdef DE
    sum_ge += lamdiff * del_m_i.gas_energy;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += lamdiff * del_m_i.scalar[i];
    }
  #endif  // SCALAR
  }
  if (lambda_p <= 0) {
    Real lamdiff = lambda_m - lambda_p;

    sum_0 += lamdiff *
             (cell_i.density * del_m_i.velocity_x / (2 * sound_speed) + del_m_i.pressure / (2 * sound_speed_squared));
    sum_1 += lamdiff * (del_m_i.velocity_x / 2.0 + del_m_i.pressure / (2 * sound_speed * cell_i.density));
    sum_4 += lamdiff * (cell_i.density * del_m_i.velocity_x * sound_speed / 2.0 + del_m_i.pressure / 2.0);
  }

  // add the corrections
  interface_R_imh.density += 0.5 * dtodx * sum_0;
  interface_R_imh.velocity_x += 0.5 * dtodx * sum_1;
  interface_R_imh.velocity_y += 0.5 * dtodx * sum_2;
  interface_R_imh.velocity_z += 0.5 * dtodx * sum_3;
  interface_R_imh.pressure += 0.5 * dtodx * sum_4;
  #ifdef DE
  interface_R_imh.gas_energy += 0.5 * dtodx * sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar[i] += 0.5 * dtodx * sum_scalar[i];
  }
  #endif  // SCALAR
#endif    // CTU

  // apply minimum constraints
  interface_R_imh.density  = fmax(interface_R_imh.density, (Real)TINY_NUMBER);
  interface_L_iph.density  = fmax(interface_L_iph.density, (Real)TINY_NUMBER);
  interface_R_imh.pressure = fmax(interface_R_imh.pressure, (Real)TINY_NUMBER);
  interface_L_iph.pressure = fmax(interface_L_iph.pressure, (Real)TINY_NUMBER);

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

  dev_bounds_R[id]                = interface_R_imh.density;
  dev_bounds_R[o1 * n_cells + id] = interface_R_imh.density * interface_R_imh.velocity_x;
  dev_bounds_R[o2 * n_cells + id] = interface_R_imh.density * interface_R_imh.velocity_y;
  dev_bounds_R[o3 * n_cells + id] = interface_R_imh.density * interface_R_imh.velocity_z;
  dev_bounds_R[4 * n_cells + id] =
      (interface_R_imh.pressure / (gamma - 1.0)) + 0.5 * interface_R_imh.density *
                                                       (interface_R_imh.velocity_x * interface_R_imh.velocity_x +
                                                        interface_R_imh.velocity_y * interface_R_imh.velocity_y +
                                                        interface_R_imh.velocity_z * interface_R_imh.velocity_z);
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_bounds_R[(5 + i) * n_cells + id] = interface_R_imh.density * interface_R_imh.scalar[i];
  }
#endif  // SCALAR
#ifdef DE
  dev_bounds_R[(n_fields - 1) * n_cells + id] = interface_R_imh.density * interface_R_imh.gas_energy;
#endif  // DE
  // bounds_L refers to the left side of the i+1/2 interface
  id                              = xid + yid * nx + zid * nx * ny;
  dev_bounds_L[id]                = interface_L_iph.density;
  dev_bounds_L[o1 * n_cells + id] = interface_L_iph.density * interface_L_iph.velocity_x;
  dev_bounds_L[o2 * n_cells + id] = interface_L_iph.density * interface_L_iph.velocity_y;
  dev_bounds_L[o3 * n_cells + id] = interface_L_iph.density * interface_L_iph.velocity_z;
  dev_bounds_L[4 * n_cells + id] =
      (interface_L_iph.pressure / (gamma - 1.0)) + 0.5 * interface_L_iph.density *
                                                       (interface_L_iph.velocity_x * interface_L_iph.velocity_x +
                                                        interface_L_iph.velocity_y * interface_L_iph.velocity_y +
                                                        interface_L_iph.velocity_z * interface_L_iph.velocity_z);
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_bounds_L[(5 + i) * n_cells + id] = interface_L_iph.density * interface_L_iph.scalar[i];
  }
#endif  // SCALAR
#ifdef DE
  dev_bounds_L[(n_fields - 1) * n_cells + id] = interface_L_iph.density * interface_L_iph.gas_energy;
#endif  // DE
}

namespace plmc_utils
{
// =====================================================================================================================
PlmcPrimitive __device__ __host__ Load_Data(Real const *dev_conserved, size_t const &xid, size_t const &yid,
                                            size_t const &zid, size_t const &nx, size_t const &ny,
                                            size_t const &n_cells, size_t const &o1, size_t const &o2, size_t const &o3,
                                            Real const &gamma)
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
// =====================================================================================================================

// =====================================================================================================================
PlmcPrimitive __device__ __host__ Compute_Slope(PlmcPrimitive const &left, PlmcPrimitive const &right, Real const &coef)
{
  PlmcPrimitive slopes;

  slopes.density    = coef * (left.density - right.density);
  slopes.velocity_x = coef * (left.velocity_x - right.velocity_x);
  slopes.velocity_y = coef * (left.velocity_y - right.velocity_y);
  slopes.velocity_z = coef * (left.velocity_z - right.velocity_z);
  slopes.pressure   = coef * (left.pressure - right.pressure);

#ifdef MHD
  slopes.magnetic_y = coef * (left.magnetic_y - right.magnetic_y);
  slopes.magnetic_z = coef * (left.magnetic_z - right.magnetic_z);
#endif  // MHD

#ifdef DE
  slopes.gas_energy = coef * (left.gas_energy - right.gas_energy);
#endif  // DE

#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    slopes.scalar[i] = coef * (left.scalar[i] - right.scalar[i]);
  }
#endif  // SCALAR

  return slopes;
}
// =====================================================================================================================

// =====================================================================================================================
PlmcPrimitive __device__ __host__ Van_Leer_Slope(PlmcPrimitive const &left_slope, PlmcPrimitive const &right_slope)
{
  PlmcPrimitive vl_slopes;

  auto Calc_Vl_Slope = [](Real const &left, Real const &right) -> Real {
    if (left * right > 0.0) {
      return 2.0 * left * right / (left + right);
    } else {
      return 0.0;
    }
  };

  vl_slopes.density    = Calc_Vl_Slope(left_slope.density, right_slope.density);
  vl_slopes.velocity_x = Calc_Vl_Slope(left_slope.velocity_x, right_slope.velocity_x);
  vl_slopes.velocity_y = Calc_Vl_Slope(left_slope.velocity_y, right_slope.velocity_y);
  vl_slopes.velocity_z = Calc_Vl_Slope(left_slope.velocity_z, right_slope.velocity_z);
  vl_slopes.pressure   = Calc_Vl_Slope(left_slope.pressure, right_slope.pressure);

#ifdef MHD
  vl_slopes.magnetic_y = Calc_Vl_Slope(left_slope.magnetic_y, right_slope.magnetic_y);
  vl_slopes.magnetic_z = Calc_Vl_Slope(left_slope.magnetic_z, right_slope.magnetic_z);
#endif  // MHD

#ifdef DE
  vl_slopes.gas_energy = Calc_Vl_Slope(left_slope.gas_energy, right_slope.gas_energy);
#endif  // DE

#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    vl_slopes.scalar[i] = Calc_Vl_Slope(left_slope.scalar[i], right_slope.scalar[i]);
  }
#endif  // SCALAR

  return vl_slopes;
}
// =====================================================================================================================
}  // namespace plmc_utils

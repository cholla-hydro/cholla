/*! \file ppmc_cuda.cu
 *  \brief Functions definitions for the ppm kernels, using characteristic
 tracing. Written following Stone et al. 2008. */

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../reconstruction/reconstruction.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif

/*!
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using ppm. */
__global__ void PPMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                          Real dt, Real gamma, int dir)
{
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Thread guard to prevent overrun
  if (xid < 2 or xid >= nx - 3 or yid < 2 or yid >= ny - 3 or zid < 2 or zid >= nz - 3) {
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

  // declare primitive variables for each stencil
  // these will be placed into registers for each thread
  reconstruction::Primitive cell_i, cell_im1, cell_im2, cell_ip1, cell_ip2;

  // declare other variables to be used
  Real a;
  Real del_d_L, del_vx_L, del_vy_L, del_vz_L, del_p_L;
  Real del_d_R, del_vx_R, del_vy_R, del_vz_R, del_p_R;
  Real del_d_C, del_vx_C, del_vy_C, del_vz_C, del_p_C;
  Real del_d_G, del_vx_G, del_vy_G, del_vz_G, del_p_G;
  Real del_a_0_L, del_a_1_L, del_a_2_L, del_a_3_L, del_a_4_L;
  Real del_a_0_R, del_a_1_R, del_a_2_R, del_a_3_R, del_a_4_R;
  Real del_a_0_C, del_a_1_C, del_a_2_C, del_a_3_C, del_a_4_C;
  Real del_a_0_G, del_a_1_G, del_a_2_G, del_a_3_G, del_a_4_G;
  Real del_a_0_m, del_a_1_m, del_a_2_m, del_a_3_m, del_a_4_m;
  Real lim_slope_a, lim_slope_b;
  Real del_d_m_imo, del_vx_m_imo, del_vy_m_imo, del_vz_m_imo, del_p_m_imo;
  Real del_d_m_i, del_vx_m_i, del_vy_m_i, del_vz_m_i, del_p_m_i;
  Real del_d_m_ipo, del_vx_m_ipo, del_vy_m_ipo, del_vz_m_ipo, del_p_m_ipo;
  Real d_L, vx_L, vy_L, vz_L, p_L;
  Real d_R, vx_R, vy_R, vz_R, p_R;

// #ifdef CTU
#ifndef VL
  Real dtodx = dt / dx;
  Real d_6, vx_6, vy_6, vz_6, p_6;
  Real lambda_m, lambda_0, lambda_p;
  Real lambda_max, lambda_min;
  Real A, B, C, D;
  Real chi_1, chi_2, chi_3, chi_4, chi_5;
  Real sum_1, sum_2, sum_3, sum_4, sum_5;
#endif  // VL

#ifdef DE
  Real del_ge_L, del_ge_R, del_ge_C, del_ge_G;
  Real del_ge_m_imo, del_ge_m_i, del_ge_m_ipo;
  Real ge_L, ge_R;
  Real E_kin, E, dge;
  // #ifdef CTU
  #ifndef VL
  Real chi_ge, sum_ge, ge_6;
  #endif  // VL
#endif    // DE
#ifdef SCALAR
  Real del_scalar_L[NSCALARS], del_scalar_R[NSCALARS], del_scalar_C[NSCALARS], del_scalar_G[NSCALARS];
  Real del_scalar_m_imo[NSCALARS], del_scalar_m_i[NSCALARS], del_scalar_m_ipo[NSCALARS];
  Real scalar_L[NSCALARS], scalar_R[NSCALARS];
  // #ifdef CTU
  #ifndef VL
  Real chi_scalar[NSCALARS], sum_scalar[NSCALARS], scalar_6[NSCALARS];
  #endif  // VL
#endif    // SCALAR

  // load the 5-cell stencil into registers
  // cell i
  int id            = xid + yid * nx + zid * nx * ny;
  cell_i.density    = dev_conserved[id];
  cell_i.velocity_x = dev_conserved[o1 * n_cells + id] / cell_i.density;
  cell_i.velocity_y = dev_conserved[o2 * n_cells + id] / cell_i.density;
  cell_i.velocity_z = dev_conserved[o3 * n_cells + id] / cell_i.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_i.density *
          (cell_i.velocity_x * cell_i.velocity_x + cell_i.velocity_y * cell_i.velocity_y +
           cell_i.velocity_z * cell_i.velocity_z);
  dge             = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_i.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, dge, gamma);
#else   // not DE
  cell_i.pressure = (dev_conserved[4 * n_cells + id] -
                     0.5 * cell_i.density *
                         (cell_i.velocity_x * cell_i.velocity_x + cell_i.velocity_y * cell_i.velocity_y +
                          cell_i.velocity_z * cell_i.velocity_z)) *
                    (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_i.pressure = fmax(cell_i.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_i.gas_energy = dge / cell_i.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_i.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_i.density;
  }
#endif  // SCALAR
  // cell i-1
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

  cell_im1.density    = dev_conserved[id];
  cell_im1.velocity_x = dev_conserved[o1 * n_cells + id] / cell_im1.density;
  cell_im1.velocity_y = dev_conserved[o2 * n_cells + id] / cell_im1.density;
  cell_im1.velocity_z = dev_conserved[o3 * n_cells + id] / cell_im1.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_im1.density *
          (cell_im1.velocity_x * cell_im1.velocity_x + cell_im1.velocity_y * cell_im1.velocity_y +
           cell_im1.velocity_z * cell_im1.velocity_z);
  dge               = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_im1.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, dge, gamma);
#else   // not DE
  cell_im1.pressure = (dev_conserved[4 * n_cells + id] -
                       0.5 * cell_im1.density *
                           (cell_im1.velocity_x * cell_im1.velocity_x + cell_im1.velocity_y * cell_im1.velocity_y +
                            cell_im1.velocity_z * cell_im1.velocity_z)) *
                      (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_im1.pressure = fmax(cell_im1.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_im1.gas_energy = dge / cell_im1.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_im1.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_im1.density;
  }
#endif  // SCALAR
  // cell i+1
  switch (dir) {
    case 0:
      id = xid + 1 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid + 1) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid + 1) * nx * ny;
      break;
  }
  cell_ip1.density    = dev_conserved[id];
  cell_ip1.velocity_x = dev_conserved[o1 * n_cells + id] / cell_ip1.density;
  cell_ip1.velocity_y = dev_conserved[o2 * n_cells + id] / cell_ip1.density;
  cell_ip1.velocity_z = dev_conserved[o3 * n_cells + id] / cell_ip1.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_ip1.density *
          (cell_ip1.velocity_x * cell_ip1.velocity_x + cell_ip1.velocity_y * cell_ip1.velocity_y +
           cell_ip1.velocity_z * cell_ip1.velocity_z);
  dge               = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_ip1.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, dge, gamma);
#else   // not DE
  cell_ip1.pressure = (dev_conserved[4 * n_cells + id] -
                       0.5 * cell_ip1.density *
                           (cell_ip1.velocity_x * cell_ip1.velocity_x + cell_ip1.velocity_y * cell_ip1.velocity_y +
                            cell_ip1.velocity_z * cell_ip1.velocity_z)) *
                      (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_ip1.pressure = fmax(cell_ip1.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_ip1.gas_energy = dge / cell_ip1.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_ip1.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_ip1.density;
  }
#endif  // SCALAR
  // cell i-2
  switch (dir) {
    case 0:
      id = xid - 2 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid - 2) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid - 2) * nx * ny;
      break;
  }
  cell_im2.density    = dev_conserved[id];
  cell_im2.velocity_x = dev_conserved[o1 * n_cells + id] / cell_im2.density;
  cell_im2.velocity_y = dev_conserved[o2 * n_cells + id] / cell_im2.density;
  cell_im2.velocity_z = dev_conserved[o3 * n_cells + id] / cell_im2.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_im2.density *
          (cell_im2.velocity_x * cell_im2.velocity_x + cell_im2.velocity_y * cell_im2.velocity_y +
           cell_im2.velocity_z * cell_im2.velocity_z);
  dge               = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_im2.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, dge, gamma);
#else   // not DE
  cell_im2.pressure = (dev_conserved[4 * n_cells + id] -
                       0.5 * cell_im2.density *
                           (cell_im2.velocity_x * cell_im2.velocity_x + cell_im2.velocity_y * cell_im2.velocity_y +
                            cell_im2.velocity_z * cell_im2.velocity_z)) *
                      (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_im2.pressure = fmax(cell_im2.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_im2.gas_energy = dge / cell_im2.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_im2.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_im2.density;
  }
#endif  // SCALAR
  // cell i+2
  switch (dir) {
    case 0:
      id = xid + 2 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid + 2) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid + 2) * nx * ny;
      break;
  }
  cell_ip2.density    = dev_conserved[id];
  cell_ip2.velocity_x = dev_conserved[o1 * n_cells + id] / cell_ip2.density;
  cell_ip2.velocity_y = dev_conserved[o2 * n_cells + id] / cell_ip2.density;
  cell_ip2.velocity_z = dev_conserved[o3 * n_cells + id] / cell_ip2.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_ip2.density *
          (cell_ip2.velocity_x * cell_ip2.velocity_x + cell_ip2.velocity_y * cell_ip2.velocity_y +
           cell_ip2.velocity_z * cell_ip2.velocity_z);
  dge               = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_ip2.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, dge, gamma);
#else   // not DE
  cell_ip2.pressure = (dev_conserved[4 * n_cells + id] -
                       0.5 * cell_ip2.density *
                           (cell_ip2.velocity_x * cell_ip2.velocity_x + cell_ip2.velocity_y * cell_ip2.velocity_y +
                            cell_ip2.velocity_z * cell_ip2.velocity_z)) *
                      (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_ip2.pressure = fmax(cell_ip2.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_ip2.gas_energy = dge / cell_ip2.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_ip2.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_ip2.density;
  }
#endif  // SCALAR

  // printf("%d %d %d %f %f %f %f %f\n", xid, yid, zid, cell_i.density, cell_i.velocity_x, cell_i.velocity_y,
  // cell_i.velocity_z, cell_i.pressure);

  // Steps 2 - 5 are repeated for cell i-1, i, and i+1
  // Step 2 - Compute the left, right, centered, and van Leer differences of
  // the primitive variables
  //          Note that here L and R refer to locations relative to the cell
  //          center Stone Eqn 36

  // calculate the adiabatic sound speed in cell imo
  a = sqrt(gamma * cell_im1.pressure / cell_im1.density);

  // left
  del_d_L  = cell_im1.density - cell_im2.density;
  del_vx_L = cell_im1.velocity_x - cell_im2.velocity_x;
  del_vy_L = cell_im1.velocity_y - cell_im2.velocity_y;
  del_vz_L = cell_im1.velocity_z - cell_im2.velocity_z;
  del_p_L  = cell_im1.pressure - cell_im2.pressure;

  // right
  del_d_R  = cell_i.density - cell_im1.density;
  del_vx_R = cell_i.velocity_x - cell_im1.velocity_x;
  del_vy_R = cell_i.velocity_y - cell_im1.velocity_y;
  del_vz_R = cell_i.velocity_z - cell_im1.velocity_z;
  del_p_R  = cell_i.pressure - cell_im1.pressure;

  // centered
  del_d_C  = 0.5 * (cell_i.density - cell_im2.density);
  del_vx_C = 0.5 * (cell_i.velocity_x - cell_im2.velocity_x);
  del_vy_C = 0.5 * (cell_i.velocity_y - cell_im2.velocity_y);
  del_vz_C = 0.5 * (cell_i.velocity_z - cell_im2.velocity_z);
  del_p_C  = 0.5 * (cell_i.pressure - cell_im2.pressure);

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
  del_ge_L = cell_im1.gas_energy - cell_im2.gas_energy;
  del_ge_R = cell_i.gas_energy - cell_im1.gas_energy;
  del_ge_C = 0.5 * (cell_i.gas_energy - cell_im2.gas_energy);
  if (del_ge_L * del_ge_R > 0.0) {
    del_ge_G = 2.0 * del_ge_L * del_ge_R / (del_ge_L + del_ge_R);
  } else {
    del_ge_G = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_scalar_L[i] = cell_im1.scalar[i] - cell_im2.scalar[i];
    del_scalar_R[i] = cell_i.scalar[i] - cell_im1.scalar[i];
    del_scalar_C[i] = 0.5 * (cell_i.scalar[i] - cell_im2.scalar[i]);
    if (del_scalar_L[i] * del_scalar_R[i] > 0.0) {
      del_scalar_G[i] = 2.0 * del_scalar_L[i] * del_scalar_R[i] / (del_scalar_L[i] + del_scalar_R[i]);
    } else {
      del_scalar_G[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 3 - Project the left, right, centered and van Leer differences onto
  // the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables,
  //          see Stone for notation) Use the eigenvectors given in Stone
  //          2008, Appendix A

  del_a_0_L = -0.5 * cell_im1.density * del_vx_L / a + 0.5 * del_p_L / (a * a);
  del_a_1_L = del_d_L - del_p_L / (a * a);
  del_a_2_L = del_vy_L;
  del_a_3_L = del_vz_L;
  del_a_4_L = 0.5 * cell_im1.density * del_vx_L / a + 0.5 * del_p_L / (a * a);

  del_a_0_R = -0.5 * cell_im1.density * del_vx_R / a + 0.5 * del_p_R / (a * a);
  del_a_1_R = del_d_R - del_p_R / (a * a);
  del_a_2_R = del_vy_R;
  del_a_3_R = del_vz_R;
  del_a_4_R = 0.5 * cell_im1.density * del_vx_R / a + 0.5 * del_p_R / (a * a);

  del_a_0_C = -0.5 * cell_im1.density * del_vx_C / a + 0.5 * del_p_C / (a * a);
  del_a_1_C = del_d_C - del_p_C / (a * a);
  del_a_2_C = del_vy_C;
  del_a_3_C = del_vz_C;
  del_a_4_C = 0.5 * cell_im1.density * del_vx_C / a + 0.5 * del_p_C / (a * a);

  del_a_0_G = -0.5 * cell_im1.density * del_vx_G / a + 0.5 * del_p_G / (a * a);
  del_a_1_G = del_d_G - del_p_G / (a * a);
  del_a_2_G = del_vy_G;
  del_a_3_G = del_vz_G;
  del_a_4_G = 0.5 * cell_im1.density * del_vx_G / a + 0.5 * del_p_G / (a * a);

  // Step 4 - Apply monotonicity constraints to the differences in the
  // characteristic variables
  //          Stone Eqn 38

  del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;

  if (del_a_0_L * del_a_0_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
    lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
    del_a_0_m   = sgn_CUDA(del_a_0_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_1_L * del_a_1_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
    lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
    del_a_1_m   = sgn_CUDA(del_a_1_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_2_L * del_a_2_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
    lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
    del_a_2_m   = sgn_CUDA(del_a_2_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_3_L * del_a_3_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
    lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
    del_a_3_m   = sgn_CUDA(del_a_3_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_4_L * del_a_4_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
    lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
    del_a_4_m   = sgn_CUDA(del_a_4_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
#ifdef DE
  if (del_ge_L * del_ge_R > 0.0) {
    lim_slope_a  = fmin(fabs(del_ge_L), fabs(del_ge_R));
    lim_slope_b  = fmin(fabs(del_ge_C), fabs(del_ge_G));
    del_ge_m_imo = sgn_CUDA(del_ge_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  } else {
    del_ge_m_imo = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    if (del_scalar_L[i] * del_scalar_R[i] > 0.0) {
      lim_slope_a         = fmin(fabs(del_scalar_L[i]), fabs(del_scalar_R[i]));
      lim_slope_b         = fmin(fabs(del_scalar_C[i]), fabs(del_scalar_G[i]));
      del_scalar_m_imo[i] = sgn_CUDA(del_scalar_C[i]) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
    } else {
      del_scalar_m_imo[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 5 - Project the monotonized difference in the characteristic
  // variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_imo  = del_a_0_m + del_a_1_m + del_a_4_m;
  del_vx_m_imo = -a * del_a_0_m / cell_im1.density + a * del_a_4_m / cell_im1.density;
  del_vy_m_imo = del_a_2_m;
  del_vz_m_imo = del_a_3_m;
  del_p_m_imo  = a * a * del_a_0_m + a * a * del_a_4_m;

  // Step 2 - Compute the left, right, centered, and van Leer differences of
  // the primitive variables
  //          Note that here L and R refer to locations relative to the cell
  //          center Stone Eqn 36

  // calculate the adiabatic sound speed in cell i
  a = sqrt(gamma * cell_i.pressure / cell_i.density);

  // left
  del_d_L  = cell_i.density - cell_im1.density;
  del_vx_L = cell_i.velocity_x - cell_im1.velocity_x;
  del_vy_L = cell_i.velocity_y - cell_im1.velocity_y;
  del_vz_L = cell_i.velocity_z - cell_im1.velocity_z;
  del_p_L  = cell_i.pressure - cell_im1.pressure;

  // right
  del_d_R  = cell_ip1.density - cell_i.density;
  del_vx_R = cell_ip1.velocity_x - cell_i.velocity_x;
  del_vy_R = cell_ip1.velocity_y - cell_i.velocity_y;
  del_vz_R = cell_ip1.velocity_z - cell_i.velocity_z;
  del_p_R  = cell_ip1.pressure - cell_i.pressure;

  // centered
  del_d_C  = 0.5 * (cell_ip1.density - cell_im1.density);
  del_vx_C = 0.5 * (cell_ip1.velocity_x - cell_im1.velocity_x);
  del_vy_C = 0.5 * (cell_ip1.velocity_y - cell_im1.velocity_y);
  del_vz_C = 0.5 * (cell_ip1.velocity_z - cell_im1.velocity_z);
  del_p_C  = 0.5 * (cell_ip1.pressure - cell_im1.pressure);

  // van Leer
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
  del_ge_L = cell_i.gas_energy - cell_im1.gas_energy;
  del_ge_R = cell_ip1.gas_energy - cell_i.gas_energy;
  del_ge_C = 0.5 * (cell_ip1.gas_energy - cell_im1.gas_energy);
  if (del_ge_L * del_ge_R > 0.0) {
    del_ge_G = 2.0 * del_ge_L * del_ge_R / (del_ge_L + del_ge_R);
  } else {
    del_ge_G = 0.0;
  }
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_scalar_L[i] = cell_i.scalar[i] - cell_im1.scalar[i];
    del_scalar_R[i] = cell_ip1.scalar[i] - cell_i.scalar[i];
    del_scalar_C[i] = 0.5 * (cell_ip1.scalar[i] - cell_im1.scalar[i]);
    if (del_scalar_L[i] * del_scalar_R[i] > 0.0) {
      del_scalar_G[i] = 2.0 * del_scalar_L[i] * del_scalar_R[i] / (del_scalar_L[i] + del_scalar_R[i]);
    } else {
      del_scalar_G[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 3 - Project the left, right, centered, and van Leer differences onto
  // the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables,
  //          see Stone for notation) Use the eigenvectors given in Stone
  //          2008, Appendix A

  del_a_0_L = -0.5 * cell_i.density * del_vx_L / a + 0.5 * del_p_L / (a * a);
  del_a_1_L = del_d_L - del_p_L / (a * a);
  del_a_2_L = del_vy_L;
  del_a_3_L = del_vz_L;
  del_a_4_L = 0.5 * cell_i.density * del_vx_L / a + 0.5 * del_p_L / (a * a);

  del_a_0_R = -0.5 * cell_i.density * del_vx_R / a + 0.5 * del_p_R / (a * a);
  del_a_1_R = del_d_R - del_p_R / (a * a);
  del_a_2_R = del_vy_R;
  del_a_3_R = del_vz_R;
  del_a_4_R = 0.5 * cell_i.density * del_vx_R / a + 0.5 * del_p_R / (a * a);

  del_a_0_C = -0.5 * cell_i.density * del_vx_C / a + 0.5 * del_p_C / (a * a);
  del_a_1_C = del_d_C - del_p_C / (a * a);
  del_a_2_C = del_vy_C;
  del_a_3_C = del_vz_C;
  del_a_4_C = 0.5 * cell_i.density * del_vx_C / a + 0.5 * del_p_C / (a * a);

  del_a_0_G = -0.5 * cell_i.density * del_vx_G / a + 0.5 * del_p_G / (a * a);
  del_a_1_G = del_d_G - del_p_G / (a * a);
  del_a_2_G = del_vy_G;
  del_a_3_G = del_vz_G;
  del_a_4_G = 0.5 * cell_i.density * del_vx_G / a + 0.5 * del_p_G / (a * a);

  // Step 4 - Apply monotonicity constraints to the differences in the
  // characteristic variables
  //          Stone Eqn 38

  del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;

  if (del_a_0_L * del_a_0_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
    lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
    del_a_0_m   = sgn_CUDA(del_a_0_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_1_L * del_a_1_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
    lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
    del_a_1_m   = sgn_CUDA(del_a_1_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_2_L * del_a_2_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
    lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
    del_a_2_m   = sgn_CUDA(del_a_2_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_3_L * del_a_3_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
    lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
    del_a_3_m   = sgn_CUDA(del_a_3_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_4_L * del_a_4_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
    lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
    del_a_4_m   = sgn_CUDA(del_a_4_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
#ifdef DE
  if (del_ge_L * del_ge_R > 0.0) {
    lim_slope_a = fmin(fabs(del_ge_L), fabs(del_ge_R));
    lim_slope_b = fmin(fabs(del_ge_C), fabs(del_ge_G));
    del_ge_m_i  = sgn_CUDA(del_ge_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  } else {
    del_ge_m_i = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    if (del_scalar_L[i] * del_scalar_R[i] > 0.0) {
      lim_slope_a       = fmin(fabs(del_scalar_L[i]), fabs(del_scalar_R[i]));
      lim_slope_b       = fmin(fabs(del_scalar_C[i]), fabs(del_scalar_G[i]));
      del_scalar_m_i[i] = sgn_CUDA(del_scalar_C[i]) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
    } else {
      del_scalar_m_i[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 5 - Project the monotonized difference in the characteristic
  // variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_i  = del_a_0_m + del_a_1_m + del_a_4_m;
  del_vx_m_i = -a * del_a_0_m / cell_i.density + a * del_a_4_m / cell_i.density;
  del_vy_m_i = del_a_2_m;
  del_vz_m_i = del_a_3_m;
  del_p_m_i  = a * a * del_a_0_m + a * a * del_a_4_m;

  // Step 2 - Compute the left, right, centered, and van Leer differences of
  // the primitive variables
  //          Note that here L and R refer to locations relative to the cell
  //          center Stone Eqn 36

  // calculate the adiabatic sound speed in cell ipo
  a = sqrt(gamma * cell_ip1.pressure / cell_ip1.density);

  // left
  del_d_L  = cell_ip1.density - cell_i.density;
  del_vx_L = cell_ip1.velocity_x - cell_i.velocity_x;
  del_vy_L = cell_ip1.velocity_y - cell_i.velocity_y;
  del_vz_L = cell_ip1.velocity_z - cell_i.velocity_z;
  del_p_L  = cell_ip1.pressure - cell_i.pressure;

  // right
  del_d_R  = cell_ip2.density - cell_ip1.density;
  del_vx_R = cell_ip2.velocity_x - cell_ip1.velocity_x;
  del_vy_R = cell_ip2.velocity_y - cell_ip1.velocity_y;
  del_vz_R = cell_ip2.velocity_z - cell_ip1.velocity_z;
  del_p_R  = cell_ip2.pressure - cell_ip1.pressure;

  // centered
  del_d_C  = 0.5 * (cell_ip2.density - cell_i.density);
  del_vx_C = 0.5 * (cell_ip2.velocity_x - cell_i.velocity_x);
  del_vy_C = 0.5 * (cell_ip2.velocity_y - cell_i.velocity_y);
  del_vz_C = 0.5 * (cell_ip2.velocity_z - cell_i.velocity_z);
  del_p_C  = 0.5 * (cell_ip2.pressure - cell_i.pressure);

  // van Leer
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
  del_ge_L = cell_ip1.gas_energy - cell_i.gas_energy;
  del_ge_R = cell_ip2.gas_energy - cell_ip1.gas_energy;
  del_ge_C = 0.5 * (cell_ip2.gas_energy - cell_i.gas_energy);
  if (del_ge_L * del_ge_R > 0.0) {
    del_ge_G = 2.0 * del_ge_L * del_ge_R / (del_ge_L + del_ge_R);
  } else {
    del_ge_G = 0.0;
  }
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_scalar_L[i] = cell_ip1.scalar[i] - cell_i.scalar[i];
    del_scalar_R[i] = cell_ip2.scalar[i] - cell_ip1.scalar[i];
    del_scalar_C[i] = 0.5 * (cell_ip2.scalar[i] - cell_i.scalar[i]);
    if (del_scalar_L[i] * del_scalar_R[i] > 0.0) {
      del_scalar_G[i] = 2.0 * del_scalar_L[i] * del_scalar_R[i] / (del_scalar_L[i] + del_scalar_R[i]);
    } else {
      del_scalar_G[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 3 - Project the left, right, centered, and van Leer differences onto
  // the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables,
  //          see Stone for notation) Use the eigenvectors given in Stone
  //          2008, Appendix A

  del_a_0_L = -0.5 * cell_ip1.density * del_vx_L / a + 0.5 * del_p_L / (a * a);
  del_a_1_L = del_d_L - del_p_L / (a * a);
  del_a_2_L = del_vy_L;
  del_a_3_L = del_vz_L;
  del_a_4_L = 0.5 * cell_ip1.density * del_vx_L / a + 0.5 * del_p_L / (a * a);

  del_a_0_R = -0.5 * cell_ip1.density * del_vx_R / a + 0.5 * del_p_R / (a * a);
  del_a_1_R = del_d_R - del_p_R / (a * a);
  del_a_2_R = del_vy_R;
  del_a_3_R = del_vz_R;
  del_a_4_R = 0.5 * cell_ip1.density * del_vx_R / a + 0.5 * del_p_R / (a * a);

  del_a_0_C = -0.5 * cell_ip1.density * del_vx_C / a + 0.5 * del_p_C / (a * a);
  del_a_1_C = del_d_C - del_p_C / (a * a);
  del_a_2_C = del_vy_C;
  del_a_3_C = del_vz_C;
  del_a_4_C = 0.5 * cell_ip1.density * del_vx_C / a + 0.5 * del_p_C / (a * a);

  del_a_0_G = -0.5 * cell_ip1.density * del_vx_G / a + 0.5 * del_p_G / (a * a);
  del_a_1_G = del_d_G - del_p_G / (a * a);
  del_a_2_G = del_vy_G;
  del_a_3_G = del_vz_G;
  del_a_4_G = 0.5 * cell_ip1.density * del_vx_G / a + 0.5 * del_p_G / (a * a);

  // Step 4 - Apply monotonicity constraints to the differences in the
  // characteristic variables
  //          Stone Eqn 38

  del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;

  if (del_a_0_L * del_a_0_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
    lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
    del_a_0_m   = sgn_CUDA(del_a_0_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_1_L * del_a_1_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
    lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
    del_a_1_m   = sgn_CUDA(del_a_1_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_2_L * del_a_2_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
    lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
    del_a_2_m   = sgn_CUDA(del_a_2_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_3_L * del_a_3_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
    lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
    del_a_3_m   = sgn_CUDA(del_a_3_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_4_L * del_a_4_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
    lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
    del_a_4_m   = sgn_CUDA(del_a_4_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
#ifdef DE
  if (del_ge_L * del_ge_R > 0.0) {
    lim_slope_a  = fmin(fabs(del_ge_L), fabs(del_ge_R));
    lim_slope_b  = fmin(fabs(del_ge_C), fabs(del_ge_G));
    del_ge_m_ipo = sgn_CUDA(del_ge_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  } else {
    del_ge_m_ipo = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    if (del_scalar_L[i] * del_scalar_R[i] > 0.0) {
      lim_slope_a         = fmin(fabs(del_scalar_L[i]), fabs(del_scalar_R[i]));
      lim_slope_b         = fmin(fabs(del_scalar_C[i]), fabs(del_scalar_G[i]));
      del_scalar_m_ipo[i] = sgn_CUDA(del_scalar_C[i]) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
    } else {
      del_scalar_m_ipo[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 5 - Project the monotonized difference in the characteristic
  // variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_ipo  = del_a_0_m + del_a_1_m + del_a_4_m;
  del_vx_m_ipo = -a * del_a_0_m / cell_ip1.density + a * del_a_4_m / cell_ip1.density;
  del_vy_m_ipo = del_a_2_m;
  del_vz_m_ipo = del_a_3_m;
  del_p_m_ipo  = a * a * del_a_0_m + a * a * del_a_4_m;

  // Step 6 - Use parabolic interpolation to compute values at the left and
  // right of each cell center
  //          Here, the subscripts L and R refer to the left and right side of
  //          the ith cell center Stone Eqn 46

  d_L  = 0.5 * (cell_i.density + cell_im1.density) - (del_d_m_i - del_d_m_imo) / 6.0;
  vx_L = 0.5 * (cell_i.velocity_x + cell_im1.velocity_x) - (del_vx_m_i - del_vx_m_imo) / 6.0;
  vy_L = 0.5 * (cell_i.velocity_y + cell_im1.velocity_y) - (del_vy_m_i - del_vy_m_imo) / 6.0;
  vz_L = 0.5 * (cell_i.velocity_z + cell_im1.velocity_z) - (del_vz_m_i - del_vz_m_imo) / 6.0;
  p_L  = 0.5 * (cell_i.pressure + cell_im1.pressure) - (del_p_m_i - del_p_m_imo) / 6.0;

  d_R  = 0.5 * (cell_ip1.density + cell_i.density) - (del_d_m_ipo - del_d_m_i) / 6.0;
  vx_R = 0.5 * (cell_ip1.velocity_x + cell_i.velocity_x) - (del_vx_m_ipo - del_vx_m_i) / 6.0;
  vy_R = 0.5 * (cell_ip1.velocity_y + cell_i.velocity_y) - (del_vy_m_ipo - del_vy_m_i) / 6.0;
  vz_R = 0.5 * (cell_ip1.velocity_z + cell_i.velocity_z) - (del_vz_m_ipo - del_vz_m_i) / 6.0;
  p_R  = 0.5 * (cell_ip1.pressure + cell_i.pressure) - (del_p_m_ipo - del_p_m_i) / 6.0;

#ifdef DE
  ge_L = 0.5 * (cell_i.gas_energy + cell_im1.gas_energy) - (del_ge_m_i - del_ge_m_imo) / 6.0;
  ge_R = 0.5 * (cell_ip1.gas_energy + cell_i.gas_energy) - (del_ge_m_ipo - del_ge_m_i) / 6.0;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    scalar_L[i] = 0.5 * (cell_i.scalar[i] + cell_im1.scalar[i]) - (del_scalar_m_i[i] - del_scalar_m_imo[i]) / 6.0;
    scalar_R[i] = 0.5 * (cell_ip1.scalar[i] + cell_i.scalar[i]) - (del_scalar_m_ipo[i] - del_scalar_m_i[i]) / 6.0;
  }
#endif  // SCALAR

  // Step 7 - Apply further monotonicity constraints to ensure the values on
  // the left and right side
  //          of cell center lie between neighboring cell-centered values
  //          Stone Eqns 47 - 53

  if ((d_R - cell_i.density) * (cell_i.density - d_L) <= 0) {
    d_L = d_R = cell_i.density;
  }
  if ((vx_R - cell_i.velocity_x) * (cell_i.velocity_x - vx_L) <= 0) {
    vx_L = vx_R = cell_i.velocity_x;
  }
  if ((vy_R - cell_i.velocity_y) * (cell_i.velocity_y - vy_L) <= 0) {
    vy_L = vy_R = cell_i.velocity_y;
  }
  if ((vz_R - cell_i.velocity_z) * (cell_i.velocity_z - vz_L) <= 0) {
    vz_L = vz_R = cell_i.velocity_z;
  }
  if ((p_R - cell_i.pressure) * (cell_i.pressure - p_L) <= 0) {
    p_L = p_R = cell_i.pressure;
  }

  if (6.0 * (d_R - d_L) * (cell_i.density - 0.5 * (d_L + d_R)) > (d_R - d_L) * (d_R - d_L)) {
    d_L = 3.0 * cell_i.density - 2.0 * d_R;
  }
  if (6.0 * (vx_R - vx_L) * (cell_i.velocity_x - 0.5 * (vx_L + vx_R)) > (vx_R - vx_L) * (vx_R - vx_L)) {
    vx_L = 3.0 * cell_i.velocity_x - 2.0 * vx_R;
  }
  if (6.0 * (vy_R - vy_L) * (cell_i.velocity_y - 0.5 * (vy_L + vy_R)) > (vy_R - vy_L) * (vy_R - vy_L)) {
    vy_L = 3.0 * cell_i.velocity_y - 2.0 * vy_R;
  }
  if (6.0 * (vz_R - vz_L) * (cell_i.velocity_z - 0.5 * (vz_L + vz_R)) > (vz_R - vz_L) * (vz_R - vz_L)) {
    vz_L = 3.0 * cell_i.velocity_z - 2.0 * vz_R;
  }
  if (6.0 * (p_R - p_L) * (cell_i.pressure - 0.5 * (p_L + p_R)) > (p_R - p_L) * (p_R - p_L)) {
    p_L = 3.0 * cell_i.pressure - 2.0 * p_R;
  }

  if (6.0 * (d_R - d_L) * (cell_i.density - 0.5 * (d_L + d_R)) < -(d_R - d_L) * (d_R - d_L)) {
    d_R = 3.0 * cell_i.density - 2.0 * d_L;
  }
  if (6.0 * (vx_R - vx_L) * (cell_i.velocity_x - 0.5 * (vx_L + vx_R)) < -(vx_R - vx_L) * (vx_R - vx_L)) {
    vx_R = 3.0 * cell_i.velocity_x - 2.0 * vx_L;
  }
  if (6.0 * (vy_R - vy_L) * (cell_i.velocity_y - 0.5 * (vy_L + vy_R)) < -(vy_R - vy_L) * (vy_R - vy_L)) {
    vy_R = 3.0 * cell_i.velocity_y - 2.0 * vy_L;
  }
  if (6.0 * (vz_R - vz_L) * (cell_i.velocity_z - 0.5 * (vz_L + vz_R)) < -(vz_R - vz_L) * (vz_R - vz_L)) {
    vz_R = 3.0 * cell_i.velocity_z - 2.0 * vz_L;
  }
  if (6.0 * (p_R - p_L) * (cell_i.pressure - 0.5 * (p_L + p_R)) < -(p_R - p_L) * (p_R - p_L)) {
    p_R = 3.0 * cell_i.pressure - 2.0 * p_L;
  }

  d_L  = fmax(fmin(cell_i.density, cell_im1.density), d_L);
  d_L  = fmin(fmax(cell_i.density, cell_im1.density), d_L);
  d_R  = fmax(fmin(cell_i.density, cell_ip1.density), d_R);
  d_R  = fmin(fmax(cell_i.density, cell_ip1.density), d_R);
  vx_L = fmax(fmin(cell_i.velocity_x, cell_im1.velocity_x), vx_L);
  vx_L = fmin(fmax(cell_i.velocity_x, cell_im1.velocity_x), vx_L);
  vx_R = fmax(fmin(cell_i.velocity_x, cell_ip1.velocity_x), vx_R);
  vx_R = fmin(fmax(cell_i.velocity_x, cell_ip1.velocity_x), vx_R);
  vy_L = fmax(fmin(cell_i.velocity_y, cell_im1.velocity_y), vy_L);
  vy_L = fmin(fmax(cell_i.velocity_y, cell_im1.velocity_y), vy_L);
  vy_R = fmax(fmin(cell_i.velocity_y, cell_ip1.velocity_y), vy_R);
  vy_R = fmin(fmax(cell_i.velocity_y, cell_ip1.velocity_y), vy_R);
  vz_L = fmax(fmin(cell_i.velocity_z, cell_im1.velocity_z), vz_L);
  vz_L = fmin(fmax(cell_i.velocity_z, cell_im1.velocity_z), vz_L);
  vz_R = fmax(fmin(cell_i.velocity_z, cell_ip1.velocity_z), vz_R);
  vz_R = fmin(fmax(cell_i.velocity_z, cell_ip1.velocity_z), vz_R);
  p_L  = fmax(fmin(cell_i.pressure, cell_im1.pressure), p_L);
  p_L  = fmin(fmax(cell_i.pressure, cell_im1.pressure), p_L);
  p_R  = fmax(fmin(cell_i.pressure, cell_ip1.pressure), p_R);
  p_R  = fmin(fmax(cell_i.pressure, cell_ip1.pressure), p_R);

#ifdef DE
  if ((ge_R - cell_i.gas_energy) * (cell_i.gas_energy - ge_L) <= 0) {
    ge_L = ge_R = cell_i.gas_energy;
  }
  if (6.0 * (ge_R - ge_L) * (cell_i.gas_energy - 0.5 * (ge_L + ge_R)) > (ge_R - ge_L) * (ge_R - ge_L)) {
    ge_L = 3.0 * cell_i.gas_energy - 2.0 * ge_R;
  }
  if (6.0 * (ge_R - ge_L) * (cell_i.gas_energy - 0.5 * (ge_L + ge_R)) < -(ge_R - ge_L) * (ge_R - ge_L)) {
    ge_R = 3.0 * cell_i.gas_energy - 2.0 * ge_L;
  }
  ge_L = fmax(fmin(cell_i.gas_energy, cell_im1.gas_energy), ge_L);
  ge_L = fmin(fmax(cell_i.gas_energy, cell_im1.gas_energy), ge_L);
  ge_R = fmax(fmin(cell_i.gas_energy, cell_ip1.gas_energy), ge_R);
  ge_R = fmin(fmax(cell_i.gas_energy, cell_ip1.gas_energy), ge_R);
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    if ((scalar_R[i] - cell_i.scalar[i]) * (cell_i.scalar[i] - scalar_L[i]) <= 0) {
      scalar_L[i] = scalar_R[i] = cell_i.scalar[i];
    }
    if (6.0 * (scalar_R[i] - scalar_L[i]) * (cell_i.scalar[i] - 0.5 * (scalar_L[i] + scalar_R[i])) >
        (scalar_R[i] - scalar_L[i]) * (scalar_R[i] - scalar_L[i])) {
      scalar_L[i] = 3.0 * cell_i.scalar[i] - 2.0 * scalar_R[i];
    }
    if (6.0 * (scalar_R[i] - scalar_L[i]) * (cell_i.scalar[i] - 0.5 * (scalar_L[i] + scalar_R[i])) <
        -(scalar_R[i] - scalar_L[i]) * (scalar_R[i] - scalar_L[i])) {
      scalar_R[i] = 3.0 * cell_i.scalar[i] - 2.0 * scalar_L[i];
    }
    scalar_L[i] = fmax(fmin(cell_i.scalar[i], cell_im1.scalar[i]), scalar_L[i]);
    scalar_L[i] = fmin(fmax(cell_i.scalar[i], cell_im1.scalar[i]), scalar_L[i]);
    scalar_R[i] = fmax(fmin(cell_i.scalar[i], cell_ip1.scalar[i]), scalar_R[i]);
    scalar_R[i] = fmin(fmax(cell_i.scalar[i], cell_ip1.scalar[i]), scalar_R[i]);
  }
#endif  // SCALAR

// #ifdef CTU
#ifndef VL

  // Step 8 - Compute the coefficients for the monotonized parabolic
  // interpolation function
  //          Stone Eqn 54

  del_d_m_i  = d_R - d_L;
  del_vx_m_i = vx_R - vx_L;
  del_vy_m_i = vy_R - vy_L;
  del_vz_m_i = vz_R - vz_L;
  del_p_m_i  = p_R - p_L;

  d_6  = 6.0 * (cell_i.density - 0.5 * (d_L + d_R));
  vx_6 = 6.0 * (cell_i.velocity_x - 0.5 * (vx_L + vx_R));
  vy_6 = 6.0 * (cell_i.velocity_y - 0.5 * (vy_L + vy_R));
  vz_6 = 6.0 * (cell_i.velocity_z - 0.5 * (vz_L + vz_R));
  p_6  = 6.0 * (cell_i.pressure - 0.5 * (p_L + p_R));

  #ifdef DE
  del_ge_m_i = ge_R - ge_L;
  ge_6       = 6.0 * (cell_i.gas_energy - 0.5 * (ge_L + ge_R));
  #endif  // DE

  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_scalar_m_i[i] = scalar_R[i] - scalar_L[i];
    scalar_6[i]       = 6.0 * (cell_i.scalar[i] - 0.5 * (scalar_L[i] + scalar_R[i]));
  }
  #endif  // SCALAR

  // Compute the eigenvalues of the linearized equations in the
  // primitive variables using the cell-centered primitive variables

  // recalculate the adiabatic sound speed in cell i
  a = sqrt(gamma * cell_i.pressure / cell_i.density);

  lambda_m = cell_i.velocity_x - a;
  lambda_0 = cell_i.velocity_x;
  lambda_p = cell_i.velocity_x + a;

  // Step 9 - Compute the left and right interface values using monotonized
  // parabolic interpolation
  //          Stone Eqns 55 & 56

  // largest eigenvalue
  lambda_max = fmax(lambda_p, (Real)0);
  // smallest eigenvalue
  lambda_min = fmin(lambda_m, (Real)0);

  // left interface value, i+1/2
  d_R  = d_R - lambda_max * (0.5 * dtodx) * (del_d_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * d_6);
  vx_R = vx_R - lambda_max * (0.5 * dtodx) * (del_vx_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vx_6);
  vy_R = vy_R - lambda_max * (0.5 * dtodx) * (del_vy_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vy_6);
  vz_R = vz_R - lambda_max * (0.5 * dtodx) * (del_vz_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vz_6);
  p_R  = p_R - lambda_max * (0.5 * dtodx) * (del_p_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * p_6);

  // right interface value, i-1/2
  d_L  = d_L - lambda_min * (0.5 * dtodx) * (del_d_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * d_6);
  vx_L = vx_L - lambda_min * (0.5 * dtodx) * (del_vx_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vx_6);
  vy_L = vy_L - lambda_min * (0.5 * dtodx) * (del_vy_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vy_6);
  vz_L = vz_L - lambda_min * (0.5 * dtodx) * (del_vz_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vz_6);
  p_L  = p_L - lambda_min * (0.5 * dtodx) * (del_p_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * p_6);

  #ifdef DE
  ge_R = ge_R - lambda_max * (0.5 * dtodx) * (del_ge_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * ge_6);
  ge_L = ge_L - lambda_min * (0.5 * dtodx) * (del_ge_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * ge_6);
  #endif  // DE

  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    scalar_R[i] = scalar_R[i] - lambda_max * (0.5 * dtodx) *
                                    (del_scalar_m_i[i] - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * scalar_6[i]);
    scalar_L[i] = scalar_L[i] - lambda_min * (0.5 * dtodx) *
                                    (del_scalar_m_i[i] + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * scalar_6[i]);
  }
  #endif  // SCALAR

  // Step 10 - Perform the characteristic tracing
  //           Stone Eqns 57 - 60

  // left-hand interface value, i+1/2
  sum_1 = 0;
  sum_2 = 0;
  sum_3 = 0;
  sum_4 = 0;
  sum_5 = 0;
  #ifdef DE
  sum_ge = 0;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    sum_scalar[i] = 0;
  }
  #endif  // SCALAR

  if (lambda_m >= 0) {
    A = (0.5 * dtodx) * (lambda_p - lambda_m);
    B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_m * lambda_m);

    chi_1 = A * (del_d_m_i - d_6) + B * d_6;
    chi_2 = A * (del_vx_m_i - vx_6) + B * vx_6;
    chi_3 = A * (del_vy_m_i - vy_6) + B * vy_6;
    chi_4 = A * (del_vz_m_i - vz_6) + B * vz_6;
    chi_5 = A * (del_p_m_i - p_6) + B * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / a - chi_5 / (a * a));
    sum_2 += 0.5 * (chi_2 - chi_5 / (a * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * a - chi_5);
  }
  if (lambda_0 >= 0) {
    A = (0.5 * dtodx) * (lambda_p - lambda_0);
    B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_0 * lambda_0);

    chi_1 = A * (del_d_m_i - d_6) + B * d_6;
    chi_2 = A * (del_vx_m_i - vx_6) + B * vx_6;
    chi_3 = A * (del_vy_m_i - vy_6) + B * vy_6;
    chi_4 = A * (del_vz_m_i - vz_6) + B * vz_6;
    chi_5 = A * (del_p_m_i - p_6) + B * p_6;
  #ifdef DE
    chi_ge = A * (del_ge_m_i - ge_6) + B * ge_6;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = A * (del_scalar_m_i[i] - scalar_6[i]) + B * scalar_6[i];
    }
  #endif  // SCALAR

    sum_1 += chi_1 - chi_5 / (a * a);
    sum_3 += chi_3;
    sum_4 += chi_4;
  #ifdef DE
    sum_ge += chi_ge;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += chi_scalar[i];
    }
  #endif  // SCALAR
  }
  if (lambda_p >= 0) {
    A = (0.5 * dtodx) * (lambda_p - lambda_p);
    B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_p * lambda_p);

    chi_1 = A * (del_d_m_i - d_6) + B * d_6;
    chi_2 = A * (del_vx_m_i - vx_6) + B * vx_6;
    chi_3 = A * (del_vy_m_i - vy_6) + B * vy_6;
    chi_4 = A * (del_vz_m_i - vz_6) + B * vz_6;
    chi_5 = A * (del_p_m_i - p_6) + B * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / a + chi_5 / (a * a));
    sum_2 += 0.5 * (chi_2 + chi_5 / (a * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * a + chi_5);
  }

  // add the corrections to the initial guesses for the interface values
  d_R += sum_1;
  vx_R += sum_2;
  vy_R += sum_3;
  vz_R += sum_4;
  p_R += sum_5;
  #ifdef DE
  ge_R += sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    scalar_R[i] += sum_scalar[i];
  }
  #endif  // SCALAR

  // right-hand interface value, i-1/2
  sum_1 = 0;
  sum_2 = 0;
  sum_3 = 0;
  sum_4 = 0;
  sum_5 = 0;
  #ifdef DE
  sum_ge = 0;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    sum_scalar[i] = 0;
  }
  #endif  // SCALAR
  if (lambda_m <= 0) {
    C = (0.5 * dtodx) * (lambda_m - lambda_m);
    D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_m * lambda_m);

    chi_1 = C * (del_d_m_i + d_6) + D * d_6;
    chi_2 = C * (del_vx_m_i + vx_6) + D * vx_6;
    chi_3 = C * (del_vy_m_i + vy_6) + D * vy_6;
    chi_4 = C * (del_vz_m_i + vz_6) + D * vz_6;
    chi_5 = C * (del_p_m_i + p_6) + D * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / a - chi_5 / (a * a));
    sum_2 += 0.5 * (chi_2 - chi_5 / (a * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * a - chi_5);
  }
  if (lambda_0 <= 0) {
    C = (0.5 * dtodx) * (lambda_m - lambda_0);
    D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_0 * lambda_0);

    chi_1 = C * (del_d_m_i + d_6) + D * d_6;
    chi_2 = C * (del_vx_m_i + vx_6) + D * vx_6;
    chi_3 = C * (del_vy_m_i + vy_6) + D * vy_6;
    chi_4 = C * (del_vz_m_i + vz_6) + D * vz_6;
    chi_5 = C * (del_p_m_i + p_6) + D * p_6;
  #ifdef DE
    chi_ge = C * (del_ge_m_i + ge_6) + D * ge_6;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = C * (del_scalar_m_i[i] + scalar_6[i]) + D * scalar_6[i];
    }
  #endif  // SCALAR

    sum_1 += chi_1 - chi_5 / (a * a);
    sum_3 += chi_3;
    sum_4 += chi_4;
  #ifdef DE
    sum_ge += chi_ge;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += chi_scalar[i];
    }
  #endif  // SCALAR
  }
  if (lambda_p <= 0) {
    C = (0.5 * dtodx) * (lambda_m - lambda_p);
    D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_p * lambda_p);

    chi_1 = C * (del_d_m_i + d_6) + D * d_6;
    chi_2 = C * (del_vx_m_i + vx_6) + D * vx_6;
    chi_3 = C * (del_vy_m_i + vy_6) + D * vy_6;
    chi_4 = C * (del_vz_m_i + vz_6) + D * vz_6;
    chi_5 = C * (del_p_m_i + p_6) + D * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / a + chi_5 / (a * a));
    sum_2 += 0.5 * (chi_2 + chi_5 / (a * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * a + chi_5);
  }

  // add the corrections
  d_L += sum_1;
  vx_L += sum_2;
  vy_L += sum_3;
  vz_L += sum_4;
  p_L += sum_5;
  #ifdef DE
  ge_L += sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    scalar_L[i] += sum_scalar[i];
  }
  #endif  // SCALAR

#endif  // VL, i.e. CTU was used for this section

  // enforce minimum values
  d_L = fmax(d_L, (Real)TINY_NUMBER);
  d_R = fmax(d_R, (Real)TINY_NUMBER);
  p_L = fmax(p_L, (Real)TINY_NUMBER);
  p_R = fmax(p_R, (Real)TINY_NUMBER);

  // Step 11 - Send final values back from kernel

  // bounds_R refers to the right side of the i-1/2 interface
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
  dev_bounds_R[id]                = d_L;
  dev_bounds_R[o1 * n_cells + id] = d_L * vx_L;
  dev_bounds_R[o2 * n_cells + id] = d_L * vy_L;
  dev_bounds_R[o3 * n_cells + id] = d_L * vz_L;
  dev_bounds_R[4 * n_cells + id]  = p_L / (gamma - 1.0) + 0.5 * d_L * (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L);
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_bounds_R[(5 + i) * n_cells + id] = d_L * scalar_L[i];
  }
#endif  // SCALAR
#ifdef DE
  dev_bounds_R[grid_enum::GasEnergy * n_cells + id] = d_L * ge_L;
#endif  // DE
  // bounds_L refers to the left side of the i+1/2 interface
  id                              = xid + yid * nx + zid * nx * ny;
  dev_bounds_L[id]                = d_R;
  dev_bounds_L[o1 * n_cells + id] = d_R * vx_R;
  dev_bounds_L[o2 * n_cells + id] = d_R * vy_R;
  dev_bounds_L[o3 * n_cells + id] = d_R * vz_R;
  dev_bounds_L[4 * n_cells + id]  = p_R / (gamma - 1.0) + 0.5 * d_R * (vx_R * vx_R + vy_R * vy_R + vz_R * vz_R);
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_bounds_L[(5 + i) * n_cells + id] = d_R * scalar_R[i];
  }
#endif  // SCALAR
#ifdef DE
  dev_bounds_L[grid_enum::GasEnergy * n_cells + id] = d_R * ge_R;
#endif  // DE
}

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

  // load the 5-cell stencil into registers
  // cell i
  reconstruction::Primitive const cell_i =
      reconstruction::Load_Data(dev_conserved, xid, yid, zid, nx, ny, n_cells, o1, o2, o3, gamma);

  // cell i-1. The equality checks check the direction and subtracts one from the direction
  // im1 stands for "i minus 1"
  reconstruction::Primitive const cell_im1 = reconstruction::Load_Data(
      dev_conserved, xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny, n_cells, o1, o2, o3, gamma);

  // cell i+1. The equality checks check the direction and adds one to the direction
  // ip1 stands for "i plus 1"
  reconstruction::Primitive const cell_ip1 = reconstruction::Load_Data(
      dev_conserved, xid + int(dir == 0), yid + int(dir == 1), zid + int(dir == 2), nx, ny, n_cells, o1, o2, o3, gamma);

  // cell i-2. The equality checks check the direction and subtracts one from the direction
  // im2 stands for "i minus 2"
  reconstruction::Primitive const cell_im2 =
      reconstruction::Load_Data(dev_conserved, xid - 2 * int(dir == 0), yid - 2 * int(dir == 1),
                                zid - 2 * int(dir == 2), nx, ny, n_cells, o1, o2, o3, gamma);

  // cell i+2. The equality checks check the direction and adds one to the direction
  // ip2 stands for "i plus 2"
  reconstruction::Primitive const cell_ip2 =
      reconstruction::Load_Data(dev_conserved, xid + 2 * int(dir == 0), yid + 2 * int(dir == 1),
                                zid + 2 * int(dir == 2), nx, ny, n_cells, o1, o2, o3, gamma);

  // Steps 2 - 5 are repeated for cell i-1, i, and i+1

  // ===============
  // Cell i-1 slopes
  // ===============

  // calculate the adiabatic sound speed in cell im1
  Real sound_speed = hydro_utilities::Calc_Sound_Speed(cell_im1.pressure, cell_im1.density, gamma);

  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables. Note that here L
  // and R refer to locations relative to the cell center Stone Eqn 36

  // left
  reconstruction::Primitive del_L = reconstruction::Compute_Slope(cell_im1, cell_im2);

  // right
  reconstruction::Primitive del_R = reconstruction::Compute_Slope(cell_i, cell_im1);

  // centered
  reconstruction::Primitive del_C = reconstruction::Compute_Slope(cell_i, cell_im2, 0.5);

  // Van Leer
  reconstruction::Primitive del_G = reconstruction::Van_Leer_Slope(del_L, del_R);

  // Step 3 - Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  reconstruction::Characteristic del_a_L =
      reconstruction::Primitive_To_Characteristic(cell_im1, del_L, sound_speed, sound_speed * sound_speed, gamma);

  reconstruction::Characteristic del_a_R =
      reconstruction::Primitive_To_Characteristic(cell_im1, del_R, sound_speed, sound_speed * sound_speed, gamma);

  reconstruction::Characteristic del_a_C =
      reconstruction::Primitive_To_Characteristic(cell_im1, del_C, sound_speed, sound_speed * sound_speed, gamma);

  reconstruction::Characteristic del_a_G =
      reconstruction::Primitive_To_Characteristic(cell_im1, del_G, sound_speed, sound_speed * sound_speed, gamma);

  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  // Step 5 - and project the monotonized difference in the characteristic variables back onto the primitive variables
  // Stone Eqn 39
  reconstruction::Primitive const del_m_im1 = reconstruction::Monotonize_Characteristic_Return_Primitive(
      cell_im1, del_L, del_R, del_C, del_G, del_a_L, del_a_R, del_a_C, del_a_G, sound_speed, sound_speed * sound_speed,
      gamma);

  // =============
  // Cell i slopes
  // =============

  // calculate the adiabatic sound speed in cell i
  sound_speed = hydro_utilities::Calc_Sound_Speed(cell_i.pressure, cell_i.density, gamma);

  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables. Note that here L
  // and R refer to locations relative to the cell center Stone Eqn 36

  // left
  del_L = reconstruction::Compute_Slope(cell_i, cell_im1);

  // right
  del_R = reconstruction::Compute_Slope(cell_ip1, cell_i);

  // centered
  del_C = reconstruction::Compute_Slope(cell_ip1, cell_im1, 0.5);

  // Van Leer
  del_G = reconstruction::Van_Leer_Slope(del_L, del_R);

  // Step 3 - Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  del_a_L = reconstruction::Primitive_To_Characteristic(cell_i, del_L, sound_speed, sound_speed * sound_speed, gamma);

  del_a_R = reconstruction::Primitive_To_Characteristic(cell_i, del_R, sound_speed, sound_speed * sound_speed, gamma);

  del_a_C = reconstruction::Primitive_To_Characteristic(cell_i, del_C, sound_speed, sound_speed * sound_speed, gamma);

  del_a_G = reconstruction::Primitive_To_Characteristic(cell_i, del_G, sound_speed, sound_speed * sound_speed, gamma);

  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  // Step 5 - and project the monotonized difference in the characteristic variables back onto the primitive variables
  // Stone Eqn 39
  reconstruction::Primitive del_m_i = reconstruction::Monotonize_Characteristic_Return_Primitive(
      cell_i, del_L, del_R, del_C, del_G, del_a_L, del_a_R, del_a_C, del_a_G, sound_speed, sound_speed * sound_speed,
      gamma);

  // ===============
  // Cell i+1 slopes
  // ===============

  // calculate the adiabatic sound speed in cell ipo
  sound_speed = hydro_utilities::Calc_Sound_Speed(cell_ip1.pressure, cell_ip1.density, gamma);

  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables. Note that here L
  // and R refer to locations relative to the cell center Stone Eqn 36

  // left
  del_L = reconstruction::Compute_Slope(cell_ip1, cell_i);

  // right
  del_R = reconstruction::Compute_Slope(cell_ip2, cell_ip1);

  // centered
  del_C = reconstruction::Compute_Slope(cell_ip2, cell_i, 0.5);

  // Van Leer
  del_G = reconstruction::Van_Leer_Slope(del_L, del_R);

  // Step 3 - Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  del_a_L = reconstruction::Primitive_To_Characteristic(cell_ip1, del_L, sound_speed, sound_speed * sound_speed, gamma);

  del_a_R = reconstruction::Primitive_To_Characteristic(cell_ip1, del_R, sound_speed, sound_speed * sound_speed, gamma);

  del_a_C = reconstruction::Primitive_To_Characteristic(cell_ip1, del_C, sound_speed, sound_speed * sound_speed, gamma);

  del_a_G = reconstruction::Primitive_To_Characteristic(cell_ip1, del_G, sound_speed, sound_speed * sound_speed, gamma);

  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  // Step 5 - and project the monotonized difference in the characteristic variables back onto the primitive variables
  // Stone Eqn 39
  reconstruction::Primitive const del_m_ip1 = reconstruction::Monotonize_Characteristic_Return_Primitive(
      cell_ip1, del_L, del_R, del_C, del_G, del_a_L, del_a_R, del_a_C, del_a_G, sound_speed, sound_speed * sound_speed,
      gamma);

  // Step 6 - Use parabolic interpolation to compute values at the left and right of each cell center Here, the
  // subscripts L and R refer to the left and right side of the ith cell center Stone Eqn 46
  reconstruction::Primitive interface_L_iph =
      reconstruction::Calc_Interface_Parabolic(cell_ip1, cell_i, del_m_ip1, del_m_i);

  reconstruction::Primitive interface_R_imh =
      reconstruction::Calc_Interface_Parabolic(cell_i, cell_im1, del_m_i, del_m_im1);

  // Step 7 - Apply further monotonicity constraints to ensure the values on the left and right side of cell center lie
  // between neighboring cell-centered values Stone Eqns 47 - 53
  reconstruction::Monotonize_Parabolic_Interface(cell_i, cell_im1, cell_ip1, interface_L_iph, interface_R_imh);

#ifndef VL
  // Step 8 - Compute the coefficients for the monotonized parabolic
  // interpolation function
  //          Stone Eqn 54

  del_m_i.density    = interface_L_iph.density - interface_R_imh.density;
  del_m_i.velocity_x = interface_L_iph.velocity_x - interface_R_imh.velocity_x;
  del_m_i.velocity_y = interface_L_iph.velocity_y - interface_R_imh.velocity_y;
  del_m_i.velocity_z = interface_L_iph.velocity_z - interface_R_imh.velocity_z;
  del_m_i.pressure   = interface_L_iph.pressure - interface_R_imh.pressure;

  Real const d_6  = 6.0 * (cell_i.density - 0.5 * (interface_R_imh.density + interface_L_iph.density));
  Real const vx_6 = 6.0 * (cell_i.velocity_x - 0.5 * (interface_R_imh.velocity_x + interface_L_iph.velocity_x));
  Real const vy_6 = 6.0 * (cell_i.velocity_y - 0.5 * (interface_R_imh.velocity_y + interface_L_iph.velocity_y));
  Real const vz_6 = 6.0 * (cell_i.velocity_z - 0.5 * (interface_R_imh.velocity_z + interface_L_iph.velocity_z));
  Real const p_6  = 6.0 * (cell_i.pressure - 0.5 * (interface_R_imh.pressure + interface_L_iph.pressure));

  #ifdef DE
  del_m_i.gas_energy = interface_L_iph.gas_energy - interface_R_imh.gas_energy;
  Real const ge_6    = 6.0 * (cell_i.gas_energy - 0.5 * (interface_R_imh.gas_energy + interface_L_iph.gas_energy));
  #endif  // DE

  #ifdef SCALAR
  Real scalar_6[NSCALARS] : for (int i = 0; i < NSCALARS; i++)
  {
    del_m_i.scalar[i] = interface_L_iph.scalar[i] - interface_R_imh.scalar[i];
    scalar_6[i]       = 6.0 * (cell_i.scalar[i] - 0.5 * (interface_R_imh.scalar[i] + interface_L_iph.scalar[i]));
  }
  #endif  // SCALAR

  // Compute the eigenvalues of the linearized equations in the
  // primitive variables using the cell-centered primitive variables

  // recalculate the adiabatic sound speed in cell i
  sound_speed = hydro_utilities::Calc_Sound_Speed(cell_i.pressure, cell_i.density, gamma);

  Real const lambda_m = cell_i.velocity_x - sound_speed;
  Real const lambda_0 = cell_i.velocity_x;
  Real const lambda_p = cell_i.velocity_x + sound_speed;

  // Step 9 - Compute the left and right interface values using monotonized
  // parabolic interpolation
  //          Stone Eqns 55 & 56

  // largest eigenvalue
  Real const lambda_max = fmax(lambda_p, (Real)0);
  // smallest eigenvalue
  Real const lambda_min = fmin(lambda_m, (Real)0);

  // left interface value, i+1/2
  Real const dtodx = dt / dx;
  interface_L_iph.density =
      interface_L_iph.density -
      lambda_max * (0.5 * dtodx) * (del_m_i.density - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * d_6);
  interface_L_iph.velocity_x =
      interface_L_iph.velocity_x -
      lambda_max * (0.5 * dtodx) * (del_m_i.velocity_x - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vx_6);
  interface_L_iph.velocity_y =
      interface_L_iph.velocity_y -
      lambda_max * (0.5 * dtodx) * (del_m_i.velocity_y - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vy_6);
  interface_L_iph.velocity_z =
      interface_L_iph.velocity_z -
      lambda_max * (0.5 * dtodx) * (del_m_i.velocity_z - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vz_6);
  interface_L_iph.pressure =
      interface_L_iph.pressure -
      lambda_max * (0.5 * dtodx) * (del_m_i.pressure - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * p_6);

  // right interface value, i-1/2
  interface_R_imh.density =
      interface_R_imh.density -
      lambda_min * (0.5 * dtodx) * (del_m_i.density + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * d_6);
  interface_R_imh.velocity_x =
      interface_R_imh.velocity_x -
      lambda_min * (0.5 * dtodx) * (del_m_i.velocity_x + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vx_6);
  interface_R_imh.velocity_y =
      interface_R_imh.velocity_y -
      lambda_min * (0.5 * dtodx) * (del_m_i.velocity_y + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vy_6);
  interface_R_imh.velocity_z =
      interface_R_imh.velocity_z -
      lambda_min * (0.5 * dtodx) * (del_m_i.velocity_z + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vz_6);
  interface_R_imh.pressure =
      interface_R_imh.pressure -
      lambda_min * (0.5 * dtodx) * (del_m_i.pressure + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * p_6);

  #ifdef DE
  interface_L_iph.gas_energy =
      interface_L_iph.gas_energy -
      lambda_max * (0.5 * dtodx) * (del_m_i.gas_energy - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * ge_6);
  interface_R_imh.gas_energy =
      interface_R_imh.gas_energy -
      lambda_min * (0.5 * dtodx) * (del_m_i.gas_energy + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * ge_6);
  #endif  // DE

  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar[i] =
        interface_L_iph.scalar[i] -
        lambda_max * (0.5 * dtodx) * (del_m_i.scalar[i] - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * scalar_6[i]);
    interface_R_imh.scalar[i] =
        interface_R_imh.scalar[i] -
        lambda_min * (0.5 * dtodx) * (del_m_i.scalar[i] + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * scalar_6[i]);
  }
  #endif  // SCALAR

  // Step 10 - Perform the characteristic tracing
  //           Stone Eqns 57 - 60

  // left-hand interface value, i+1/2
  Real sum_1 = 0, sum_2 = 0, sum_3 = 0, sum_4 = 0, sum_5 = 0;
  #ifdef DE
  Real sum_ge = 0;
  #endif  // DE
  #ifdef SCALAR
  Real sum_scalar[NSCALARS];
  for (int i = 0; i < NSCALARS; i++) {
    sum_scalar[i] = 0;
  }
  #endif  // SCALAR

  if (lambda_m >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_m);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_m * lambda_m);

    Real const chi_1 = A * (del_m_i.density - d_6) + B * d_6;
    Real const chi_2 = A * (del_m_i.velocity_x - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_m_i.velocity_y - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_m_i.velocity_z - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_m_i.pressure - p_6) + B * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / sound_speed - chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 - chi_5 / (sound_speed * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * sound_speed - chi_5);
  }
  if (lambda_0 >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_0);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_0 * lambda_0);

    Real const chi_1 = A * (del_m_i.density - d_6) + B * d_6;
    Real const chi_2 = A * (del_m_i.velocity_x - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_m_i.velocity_y - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_m_i.velocity_z - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_m_i.pressure - p_6) + B * p_6;
  #ifdef DE
    Real chi_ge = A * (del_m_i.gas_energy - ge_6) + B * ge_6;
  #endif  // DE
  #ifdef SCALAR
    Real chi_scalar[NSCALARS];
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = A * (del_m_i.scalar[i] - scalar_6[i]) + B * scalar_6[i];
    }
  #endif  // SCALAR

    sum_1 += chi_1 - chi_5 / (sound_speed * sound_speed);
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
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_p);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_p * lambda_p);

    Real const chi_1 = A * (del_m_i.density - d_6) + B * d_6;
    Real const chi_2 = A * (del_m_i.velocity_x - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_m_i.velocity_y - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_m_i.velocity_z - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_m_i.pressure - p_6) + B * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / sound_speed + chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 + chi_5 / (sound_speed * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * sound_speed + chi_5);
  }

  // add the corrections to the initial guesses for the interface values
  interface_L_iph.density += sum_1;
  interface_L_iph.velocity_x += sum_2;
  interface_L_iph.velocity_y += sum_3;
  interface_L_iph.velocity_z += sum_4;
  interface_L_iph.pressure += sum_5;
  #ifdef DE
  interface_L_iph.gas_energy += sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar[i] += sum_scalar[i];
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
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_m);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_m * lambda_m);

    Real const chi_1 = C * (del_m_i.density + d_6) + D * d_6;
    Real const chi_2 = C * (del_m_i.velocity_x + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_m_i.velocity_y + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_m_i.velocity_z + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_m_i.pressure + p_6) + D * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / sound_speed - chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 - chi_5 / (sound_speed * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * sound_speed - chi_5);
  }
  if (lambda_0 <= 0) {
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_0);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_0 * lambda_0);

    Real const chi_1 = C * (del_m_i.density + d_6) + D * d_6;
    Real const chi_2 = C * (del_m_i.velocity_x + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_m_i.velocity_y + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_m_i.velocity_z + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_m_i.pressure + p_6) + D * p_6;
  #ifdef DE
    chi_ge = C * (del_m_i.gas_energy + ge_6) + D * ge_6;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = C * (del_m_i.scalar[i] + scalar_6[i]) + D * scalar_6[i];
    }
  #endif  // SCALAR

    sum_1 += chi_1 - chi_5 / (sound_speed * sound_speed);
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
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_p);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_p * lambda_p);

    Real const chi_1 = C * (del_m_i.density + d_6) + D * d_6;
    Real const chi_2 = C * (del_m_i.velocity_x + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_m_i.velocity_y + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_m_i.velocity_z + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_m_i.pressure + p_6) + D * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / sound_speed + chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 + chi_5 / (sound_speed * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * sound_speed + chi_5);
  }

  // add the corrections
  interface_R_imh.density += sum_1;
  interface_R_imh.velocity_x += sum_2;
  interface_R_imh.velocity_y += sum_3;
  interface_R_imh.velocity_z += sum_4;
  interface_R_imh.pressure += sum_5;
  #ifdef DE
  interface_R_imh.gas_energy += sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar[i] += sum_scalar[i];
  }
  #endif  // SCALAR

#endif  // not VL, i.e. CTU or SIMPLE was used for this section

  // enforce minimum values
  interface_R_imh.density  = fmax(interface_R_imh.density, (Real)TINY_NUMBER);
  interface_L_iph.density  = fmax(interface_L_iph.density, (Real)TINY_NUMBER);
  interface_R_imh.pressure = fmax(interface_R_imh.pressure, (Real)TINY_NUMBER);
  interface_L_iph.pressure = fmax(interface_L_iph.pressure, (Real)TINY_NUMBER);

  // Step 11 - Send final values back from kernel

  // Convert the left and right states in the primitive to the conserved variables send final values back from kernel
  // bounds_R refers to the right side of the i-1/2 interface
  size_t id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
  reconstruction::Write_Data(interface_L_iph, dev_bounds_L, dev_conserved, id, n_cells, o1, o2, o3, gamma);

  id = cuda_utilities::compute1DIndex(xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny);
  reconstruction::Write_Data(interface_R_imh, dev_bounds_R, dev_conserved, id, n_cells, o1, o2, o3, gamma);
}
/*! \file ppmc_cuda.cu
 *  \brief Functions definitions for the ppm kernels, using characteristic
 tracing. Written following Stone et al. 2008. */

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../reconstruction/reconstruction_internals.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif

// =====================================================================================================================
/*!
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using ppm. */
template <int dir>
__global__ void PPMC_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                         Real dt, Real gamma)
{
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  if (reconstruction::Thread_Guard<3>(nx, ny, nz, xid, yid, zid)) {
    return;
  }

  // Compute the total number of cells
  int const n_cells = nx * ny * nz;

  // Set the field indices for the various directions
  int o1, o2, o3;
  if constexpr (dir == 0) {
    o1 = grid_enum::momentum_x;
    o2 = grid_enum::momentum_y;
    o3 = grid_enum::momentum_z;
  } else if constexpr (dir == 1) {
    o1 = grid_enum::momentum_y;
    o2 = grid_enum::momentum_z;
    o3 = grid_enum::momentum_x;
  } else if constexpr (dir == 2) {
    o1 = grid_enum::momentum_z;
    o2 = grid_enum::momentum_x;
    o3 = grid_enum::momentum_y;
  }

  // load the 5-cell stencil into registers
  // cell i
  hydro_utilities::Primitive const cell_i =
      hydro_utilities::Load_Cell_Primitive<dir>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);

  // cell i-1. The equality checks check the direction and subtracts one from the direction
  // im1 stands for "i minus 1"
  hydro_utilities::Primitive const cell_im1 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny, n_cells, gamma);

  // cell i+1. The equality checks check the direction and adds one to the direction
  // ip1 stands for "i plus 1"
  hydro_utilities::Primitive const cell_ip1 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid + int(dir == 0), yid + int(dir == 1), zid + int(dir == 2), nx, ny, n_cells, gamma);

  // cell i-2. The equality checks check the direction and subtracts one from the direction
  // im2 stands for "i minus 2"
  hydro_utilities::Primitive const cell_im2 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid - 2 * int(dir == 0), yid - 2 * int(dir == 1), zid - 2 * int(dir == 2), nx, ny, n_cells, gamma);

  // cell i+2. The equality checks check the direction and adds one to the direction
  // ip2 stands for "i plus 2"
  hydro_utilities::Primitive const cell_ip2 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid + 2 * int(dir == 0), yid + 2 * int(dir == 1), zid + 2 * int(dir == 2), nx, ny, n_cells, gamma);

  // Steps 2 - 5 are repeated for cell i-1, i, and i+1

  // ===============
  // Cell i-1 slopes
  // ===============

  // Compute the eigenvectors for this cell
  reconstruction::EigenVecs eigenvectors = reconstruction::Compute_Eigenvectors(cell_im1, gamma);

  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables. Note that here L
  // and R refer to locations relative to the cell center Stone Eqn 36

  // left
  hydro_utilities::Primitive del_L = reconstruction::Compute_Slope(cell_im2, cell_im1);

  // right
  hydro_utilities::Primitive del_R = reconstruction::Compute_Slope(cell_im1, cell_i);

  // centered
  hydro_utilities::Primitive del_C = reconstruction::Compute_Slope(cell_im2, cell_i, 0.5);

  // Van Leer
  hydro_utilities::Primitive del_G = reconstruction::Van_Leer_Slope(del_L, del_R);

  // Step 3 - Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  reconstruction::Characteristic del_a_L =
      reconstruction::Primitive_To_Characteristic(cell_im1, del_L, eigenvectors, gamma);

  reconstruction::Characteristic del_a_R =
      reconstruction::Primitive_To_Characteristic(cell_im1, del_R, eigenvectors, gamma);

  reconstruction::Characteristic del_a_C =
      reconstruction::Primitive_To_Characteristic(cell_im1, del_C, eigenvectors, gamma);

  reconstruction::Characteristic del_a_G =
      reconstruction::Primitive_To_Characteristic(cell_im1, del_G, eigenvectors, gamma);

  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  reconstruction::Characteristic const del_a_m_im1 =
      reconstruction::Van_Leer_Limiter(del_a_L, del_a_R, del_a_C, del_a_G);

  // Step 5 - and project the monotonized difference in the characteristic variables back onto the primitive variables
  // Stone Eqn 39
  hydro_utilities::Primitive del_m_im1 = Characteristic_To_Primitive(cell_im1, del_a_m_im1, eigenvectors, gamma);

  // Limit the variables that aren't transformed by the characteristic projection
#ifdef DE
  del_m_im1.gas_energy_specific = reconstruction::Van_Leer_Limiter(
      del_L.gas_energy_specific, del_R.gas_energy_specific, del_C.gas_energy_specific, del_G.gas_energy_specific);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_m_im1.scalar_specific[i] = reconstruction::Van_Leer_Limiter(del_L.scalar_specific[i], del_R.scalar_specific[i],
                                                                    del_C.scalar_specific[i], del_G.scalar_specific[i]);
  }
#endif  // SCALAR

  // =============
  // Cell i slopes
  // =============

  // Compute the eigenvectors for this cell
  eigenvectors = reconstruction::Compute_Eigenvectors(cell_i, gamma);

  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables. Note that here L
  // and R refer to locations relative to the cell center Stone Eqn 36

  // left
  del_L = reconstruction::Compute_Slope(cell_im1, cell_i);

  // right
  del_R = reconstruction::Compute_Slope(cell_i, cell_ip1);

  // centered
  del_C = reconstruction::Compute_Slope(cell_im1, cell_ip1, 0.5);

  // Van Leer
  del_G = reconstruction::Van_Leer_Slope(del_L, del_R);

  // Step 3 - Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  del_a_L = reconstruction::Primitive_To_Characteristic(cell_i, del_L, eigenvectors, gamma);

  del_a_R = reconstruction::Primitive_To_Characteristic(cell_i, del_R, eigenvectors, gamma);

  del_a_C = reconstruction::Primitive_To_Characteristic(cell_i, del_C, eigenvectors, gamma);

  del_a_G = reconstruction::Primitive_To_Characteristic(cell_i, del_G, eigenvectors, gamma);

  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  reconstruction::Characteristic const del_a_m_i = reconstruction::Van_Leer_Limiter(del_a_L, del_a_R, del_a_C, del_a_G);

  // Step 5 - and project the monotonized difference in the characteristic variables back onto the primitive variables
  // Stone Eqn 39
  hydro_utilities::Primitive del_m_i = Characteristic_To_Primitive(cell_ip1, del_a_m_i, eigenvectors, gamma);

  // Limit the variables that aren't transformed by the characteristic projection
#ifdef DE
  del_m_i.gas_energy_specific = reconstruction::Van_Leer_Limiter(del_L.gas_energy_specific, del_R.gas_energy_specific,
                                                                 del_C.gas_energy_specific, del_G.gas_energy_specific);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_m_i.scalar_specific[i] = reconstruction::Van_Leer_Limiter(del_L.scalar_specific[i], del_R.scalar_specific[i],
                                                                  del_C.scalar_specific[i], del_G.scalar_specific[i]);
  }
#endif  // SCALAR

  // ===============
  // Cell i+1 slopes
  // ===============

  // Compute the eigenvectors for this cell
  eigenvectors = reconstruction::Compute_Eigenvectors(cell_ip1, gamma);

  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables. Note that here L
  // and R refer to locations relative to the cell center Stone Eqn 36

  // left
  del_L = reconstruction::Compute_Slope(cell_i, cell_ip1);

  // right
  del_R = reconstruction::Compute_Slope(cell_ip1, cell_ip2);

  // centered
  del_C = reconstruction::Compute_Slope(cell_i, cell_ip2, 0.5);

  // Van Leer
  del_G = reconstruction::Van_Leer_Slope(del_L, del_R);

  // Step 3 - Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  del_a_L = reconstruction::Primitive_To_Characteristic(cell_ip1, del_L, eigenvectors, gamma);

  del_a_R = reconstruction::Primitive_To_Characteristic(cell_ip1, del_R, eigenvectors, gamma);

  del_a_C = reconstruction::Primitive_To_Characteristic(cell_ip1, del_C, eigenvectors, gamma);

  del_a_G = reconstruction::Primitive_To_Characteristic(cell_ip1, del_G, eigenvectors, gamma);

  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  reconstruction::Characteristic const del_a_m_ip1 =
      reconstruction::Van_Leer_Limiter(del_a_L, del_a_R, del_a_C, del_a_G);

  // Step 5 - and project the monotonized difference in the characteristic variables back onto the primitive variables
  // Stone Eqn 39
  hydro_utilities::Primitive del_m_ip1 = Characteristic_To_Primitive(cell_ip1, del_a_m_ip1, eigenvectors, gamma);

  // Limit the variables that aren't transformed by the characteristic projection
#ifdef DE
  del_m_ip1.gas_energy_specific = reconstruction::Van_Leer_Limiter(
      del_L.gas_energy_specific, del_R.gas_energy_specific, del_C.gas_energy_specific, del_G.gas_energy_specific);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_m_ip1.scalar_specific[i] = reconstruction::Van_Leer_Limiter(del_L.scalar_specific[i], del_R.scalar_specific[i],
                                                                    del_C.scalar_specific[i], del_G.scalar_specific[i]);
  }
#endif  // SCALAR

  // Step 6 - Use parabolic interpolation to compute values at the left and right of each cell center Here, the
  // subscripts L and R refer to the left and right side of the ith cell center Stone Eqn 46
  hydro_utilities::Primitive interface_L_iph =
      reconstruction::Calc_Interface_Parabolic(cell_ip1, cell_i, del_m_ip1, del_m_i);

  hydro_utilities::Primitive interface_R_imh =
      reconstruction::Calc_Interface_Parabolic(cell_i, cell_im1, del_m_i, del_m_im1);

  // Step 7 - Apply further monotonicity constraints to ensure the values on the left and right side of cell center lie
  // between neighboring cell-centered values Stone Eqns 47 - 53
  reconstruction::Monotonize_Parabolic_Interface(cell_i, cell_im1, cell_ip1, interface_L_iph, interface_R_imh);

  // This is the beginning of the characteristic tracing
  // Step 8 - Compute the coefficients for the monotonized parabolic
  // interpolation function
  //          Stone Eqn 54

  del_m_i.density    = interface_L_iph.density - interface_R_imh.density;
  del_m_i.velocity.x = interface_L_iph.velocity.x - interface_R_imh.velocity.x;
  del_m_i.velocity.y = interface_L_iph.velocity.y - interface_R_imh.velocity.y;
  del_m_i.velocity.z = interface_L_iph.velocity.z - interface_R_imh.velocity.z;
  del_m_i.pressure   = interface_L_iph.pressure - interface_R_imh.pressure;

  Real const d_6  = 6.0 * (cell_i.density - 0.5 * (interface_R_imh.density + interface_L_iph.density));
  Real const vx_6 = 6.0 * (cell_i.velocity.x - 0.5 * (interface_R_imh.velocity.x + interface_L_iph.velocity.x));
  Real const vy_6 = 6.0 * (cell_i.velocity.y - 0.5 * (interface_R_imh.velocity.y + interface_L_iph.velocity.y));
  Real const vz_6 = 6.0 * (cell_i.velocity.z - 0.5 * (interface_R_imh.velocity.z + interface_L_iph.velocity.z));
  Real const p_6  = 6.0 * (cell_i.pressure - 0.5 * (interface_R_imh.pressure + interface_L_iph.pressure));

#ifdef DE
  del_m_i.gas_energy_specific = interface_L_iph.gas_energy_specific - interface_R_imh.gas_energy_specific;
  Real const ge_6             = 6.0 * (cell_i.gas_energy_specific -
                           0.5 * (interface_R_imh.gas_energy_specific + interface_L_iph.gas_energy_specific));
#endif  // DE

#ifdef SCALAR
  Real scalar_6[NSCALARS];
  for (int i = 0; i < NSCALARS; i++) {
    del_m_i.scalar_specific[i] = interface_L_iph.scalar_specific[i] - interface_R_imh.scalar_specific[i];
    scalar_6[i]                = 6.0 * (cell_i.scalar_specific[i] -
                         0.5 * (interface_R_imh.scalar_specific[i] + interface_L_iph.scalar_specific[i]));
  }
#endif  // SCALAR

  // Compute the eigenvalues of the linearized equations in the
  // primitive variables using the cell-centered primitive variables

  // recalculate the adiabatic sound speed in cell i
  Real const sound_speed = hydro_utilities::Calc_Sound_Speed(cell_i.pressure, cell_i.density, gamma);

  Real const lambda_m = cell_i.velocity.x - sound_speed;
  Real const lambda_0 = cell_i.velocity.x;
  Real const lambda_p = cell_i.velocity.x + sound_speed;

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
  interface_L_iph.velocity.x =
      interface_L_iph.velocity.x -
      lambda_max * (0.5 * dtodx) * (del_m_i.velocity.x - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vx_6);
  interface_L_iph.velocity.y =
      interface_L_iph.velocity.y -
      lambda_max * (0.5 * dtodx) * (del_m_i.velocity.y - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vy_6);
  interface_L_iph.velocity.z =
      interface_L_iph.velocity.z -
      lambda_max * (0.5 * dtodx) * (del_m_i.velocity.z - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vz_6);
  interface_L_iph.pressure =
      interface_L_iph.pressure -
      lambda_max * (0.5 * dtodx) * (del_m_i.pressure - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * p_6);

  // right interface value, i-1/2
  interface_R_imh.density =
      interface_R_imh.density -
      lambda_min * (0.5 * dtodx) * (del_m_i.density + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * d_6);
  interface_R_imh.velocity.x =
      interface_R_imh.velocity.x -
      lambda_min * (0.5 * dtodx) * (del_m_i.velocity.x + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vx_6);
  interface_R_imh.velocity.y =
      interface_R_imh.velocity.y -
      lambda_min * (0.5 * dtodx) * (del_m_i.velocity.y + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vy_6);
  interface_R_imh.velocity.z =
      interface_R_imh.velocity.z -
      lambda_min * (0.5 * dtodx) * (del_m_i.velocity.z + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vz_6);
  interface_R_imh.pressure =
      interface_R_imh.pressure -
      lambda_min * (0.5 * dtodx) * (del_m_i.pressure + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * p_6);

#ifdef DE
  interface_L_iph.gas_energy_specific =
      interface_L_iph.gas_energy_specific -
      lambda_max * (0.5 * dtodx) * (del_m_i.gas_energy_specific - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * ge_6);
  interface_R_imh.gas_energy_specific =
      interface_R_imh.gas_energy_specific -
      lambda_min * (0.5 * dtodx) * (del_m_i.gas_energy_specific + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * ge_6);
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar_specific[i] =
        interface_L_iph.scalar_specific[i] -
        lambda_max * (0.5 * dtodx) *
            (del_m_i.scalar_specific[i] - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * scalar_6[i]);
    interface_R_imh.scalar_specific[i] =
        interface_R_imh.scalar_specific[i] -
        lambda_min * (0.5 * dtodx) *
            (del_m_i.scalar_specific[i] + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * scalar_6[i]);
  }
#endif  // SCALAR

  // Step 10 - Perform the characteristic tracing
  //           Stone Eqns 57 - 60

  // left-hand interface value, i+1/2
  Real sum_1 = 0, sum_2 = 0, sum_3 = 0, sum_4 = 0, sum_5 = 0;
#ifdef DE
  Real sum_ge = 0;
  Real chi_ge = 0;
#endif  // DE
#ifdef SCALAR
  Real chi_scalar[NSCALARS];
  Real sum_scalar[NSCALARS];
  for (Real &val : sum_scalar) {
    val = 0;
  }
#endif  // SCALAR

  if (lambda_m >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_m);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_m * lambda_m);

    Real const chi_1 = A * (del_m_i.density - d_6) + B * d_6;
    Real const chi_2 = A * (del_m_i.velocity.x - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_m_i.velocity.y - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_m_i.velocity.z - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_m_i.pressure - p_6) + B * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / sound_speed - chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 - chi_5 / (sound_speed * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * sound_speed - chi_5);
  }
  if (lambda_0 >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_0);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_0 * lambda_0);

    Real const chi_1 = A * (del_m_i.density - d_6) + B * d_6;
    Real const chi_2 = A * (del_m_i.velocity.x - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_m_i.velocity.y - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_m_i.velocity.z - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_m_i.pressure - p_6) + B * p_6;
#ifdef DE
    chi_ge = A * (del_m_i.gas_energy_specific - ge_6) + B * ge_6;
#endif  // DE
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = A * (del_m_i.scalar_specific[i] - scalar_6[i]) + B * scalar_6[i];
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
    Real const chi_2 = A * (del_m_i.velocity.x - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_m_i.velocity.y - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_m_i.velocity.z - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_m_i.pressure - p_6) + B * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / sound_speed + chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 + chi_5 / (sound_speed * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * sound_speed + chi_5);
  }

  // add the corrections to the initial guesses for the interface values
  interface_L_iph.density += sum_1;
  interface_L_iph.velocity.x += sum_2;
  interface_L_iph.velocity.y += sum_3;
  interface_L_iph.velocity.z += sum_4;
  interface_L_iph.pressure += sum_5;
#ifdef DE
  interface_L_iph.gas_energy_specific += sum_ge;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar_specific[i] += sum_scalar[i];
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
  for (Real &val : sum_scalar) {
    val = 0;
  }
#endif  // SCALAR
  if (lambda_m <= 0) {
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_m);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_m * lambda_m);

    Real const chi_1 = C * (del_m_i.density + d_6) + D * d_6;
    Real const chi_2 = C * (del_m_i.velocity.x + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_m_i.velocity.y + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_m_i.velocity.z + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_m_i.pressure + p_6) + D * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / sound_speed - chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 - chi_5 / (sound_speed * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * sound_speed - chi_5);
  }
  if (lambda_0 <= 0) {
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_0);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_0 * lambda_0);

    Real const chi_1 = C * (del_m_i.density + d_6) + D * d_6;
    Real const chi_2 = C * (del_m_i.velocity.x + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_m_i.velocity.y + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_m_i.velocity.z + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_m_i.pressure + p_6) + D * p_6;
#ifdef DE
    chi_ge = C * (del_m_i.gas_energy_specific + ge_6) + D * ge_6;
#endif  // DE
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = C * (del_m_i.scalar_specific[i] + scalar_6[i]) + D * scalar_6[i];
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
    Real const chi_2 = C * (del_m_i.velocity.x + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_m_i.velocity.y + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_m_i.velocity.z + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_m_i.pressure + p_6) + D * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / sound_speed + chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 + chi_5 / (sound_speed * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * sound_speed + chi_5);
  }

  // add the corrections
  interface_R_imh.density += sum_1;
  interface_R_imh.velocity.x += sum_2;
  interface_R_imh.velocity.y += sum_3;
  interface_R_imh.velocity.z += sum_4;
  interface_R_imh.pressure += sum_5;
#ifdef DE
  interface_R_imh.gas_energy_specific += sum_ge;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar_specific[i] += sum_scalar[i];
  }
#endif  // SCALAR

  // This is the end of the characteristic tracing

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
// =====================================================================================================================

// =====================================================================================================================
template <int dir>
__global__ __launch_bounds__(TPB) void PPMC_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                               int ny, int nz, Real gamma)
{
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Ensure that we are only operating on cells that will be used
  if (reconstruction::Thread_Guard<3>(nx, ny, nz, xid, yid, zid)) {
    return;
  }

  // Compute the total number of cells
  int const n_cells = nx * ny * nz;

  // Set the field indices for the various directions
  int o1, o2, o3;
  if constexpr (dir == 0) {
    o1 = grid_enum::momentum_x;
    o2 = grid_enum::momentum_y;
    o3 = grid_enum::momentum_z;
  } else if constexpr (dir == 1) {
    o1 = grid_enum::momentum_y;
    o2 = grid_enum::momentum_z;
    o3 = grid_enum::momentum_x;
  } else if constexpr (dir == 2) {
    o1 = grid_enum::momentum_z;
    o2 = grid_enum::momentum_x;
    o3 = grid_enum::momentum_y;
  }

  // load the 5-cell stencil into registers
  // cell i
  hydro_utilities::Primitive const cell_i =
      hydro_utilities::Load_Cell_Primitive<dir>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);

  // cell i-1. The equality checks the direction and will subtract one from the correct direction
  // im1 stands for "i minus 1"
  hydro_utilities::Primitive const cell_im1 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny, n_cells, gamma);

  // cell i+1.  The equality checks the direction and add one to the correct direction
  // ip1 stands for "i plus 1"
  hydro_utilities::Primitive const cell_ip1 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid + int(dir == 0), yid + int(dir == 1), zid + int(dir == 2), nx, ny, n_cells, gamma);

  // cell i-2. The equality checks the direction and will subtract two from the correct direction
  // im2 stands for "i minus 2"
  hydro_utilities::Primitive const cell_im2 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid - 2 * int(dir == 0), yid - 2 * int(dir == 1), zid - 2 * int(dir == 2), nx, ny, n_cells, gamma);

  // cell i+2.  The equality checks the direction and add two to the correct direction
  // ip2 stands for "i plus 2"
  hydro_utilities::Primitive const cell_ip2 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid + 2 * int(dir == 0), yid + 2 * int(dir == 1), zid + 2 * int(dir == 2), nx, ny, n_cells, gamma);

  // Compute the eigenvectors
  reconstruction::EigenVecs const eigenvectors = reconstruction::Compute_Eigenvectors(cell_i, gamma);

  // Cell i
  reconstruction::Characteristic const cell_i_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_i, eigenvectors, gamma);

  // Cell i-1
  reconstruction::Characteristic const cell_im1_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_im1, eigenvectors, gamma);

  // Cell i-2
  reconstruction::Characteristic const cell_im2_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_im2, eigenvectors, gamma);

  // Cell i+1
  reconstruction::Characteristic const cell_ip1_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_ip1, eigenvectors, gamma);

  // Cell i+2
  reconstruction::Characteristic const cell_ip2_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_ip2, eigenvectors, gamma);

  // Compute the interface states for each field
  reconstruction::Characteristic interface_R_imh_characteristic, interface_L_iph_characteristic;

  reconstruction::PPM_Single_Variable(cell_im2_characteristic.a0, cell_im1_characteristic.a0, cell_i_characteristic.a0,
                                      cell_ip1_characteristic.a0, cell_ip2_characteristic.a0,
                                      interface_L_iph_characteristic.a0, interface_R_imh_characteristic.a0);
  reconstruction::PPM_Single_Variable(cell_im2_characteristic.a1, cell_im1_characteristic.a1, cell_i_characteristic.a1,
                                      cell_ip1_characteristic.a1, cell_ip2_characteristic.a1,
                                      interface_L_iph_characteristic.a1, interface_R_imh_characteristic.a1);
  reconstruction::PPM_Single_Variable(cell_im2_characteristic.a2, cell_im1_characteristic.a2, cell_i_characteristic.a2,
                                      cell_ip1_characteristic.a2, cell_ip2_characteristic.a2,
                                      interface_L_iph_characteristic.a2, interface_R_imh_characteristic.a2);
  reconstruction::PPM_Single_Variable(cell_im2_characteristic.a3, cell_im1_characteristic.a3, cell_i_characteristic.a3,
                                      cell_ip1_characteristic.a3, cell_ip2_characteristic.a3,
                                      interface_L_iph_characteristic.a3, interface_R_imh_characteristic.a3);
  reconstruction::PPM_Single_Variable(cell_im2_characteristic.a4, cell_im1_characteristic.a4, cell_i_characteristic.a4,
                                      cell_ip1_characteristic.a4, cell_ip2_characteristic.a4,
                                      interface_L_iph_characteristic.a4, interface_R_imh_characteristic.a4);

#ifdef MHD
  reconstruction::PPM_Single_Variable(cell_im2_characteristic.a5, cell_im1_characteristic.a5, cell_i_characteristic.a5,
                                      cell_ip1_characteristic.a5, cell_ip2_characteristic.a5,
                                      interface_L_iph_characteristic.a5, interface_R_imh_characteristic.a5);
  reconstruction::PPM_Single_Variable(cell_im2_characteristic.a6, cell_im1_characteristic.a6, cell_i_characteristic.a6,
                                      cell_ip1_characteristic.a6, cell_ip2_characteristic.a6,
                                      interface_L_iph_characteristic.a6, interface_R_imh_characteristic.a6);
#endif  // MHD

  // Convert back to primitive variables
  hydro_utilities::Primitive interface_L_iph =
      reconstruction::Characteristic_To_Primitive(cell_i, interface_L_iph_characteristic, eigenvectors, gamma);
  hydro_utilities::Primitive interface_R_imh =
      reconstruction::Characteristic_To_Primitive(cell_i, interface_R_imh_characteristic, eigenvectors, gamma);

  // Compute the interfaces for the variables that don't have characteristics
#ifdef DE
  reconstruction::PPM_Single_Variable(cell_im2.gas_energy_specific, cell_im1.gas_energy_specific,
                                      cell_i.gas_energy_specific, cell_ip1.gas_energy_specific,
                                      cell_ip2.gas_energy_specific, interface_L_iph.gas_energy_specific,
                                      interface_R_imh.gas_energy_specific);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    reconstruction::PPM_Single_Variable(cell_im2.scalar_specific[i], cell_im1.scalar_specific[i],
                                        cell_i.scalar_specific[i], cell_ip1.scalar_specific[i],
                                        cell_ip2.scalar_specific[i], interface_L_iph.scalar_specific[i],
                                        interface_R_imh.scalar_specific[i]);
  }
#endif  // SCALAR

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
// Instantiate the relevant template specifications
template __global__ void PPMC_CTU<0>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny,
                                     int nz, Real dx, Real dt, Real gamma);
template __global__ void PPMC_CTU<1>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny,
                                     int nz, Real dx, Real dt, Real gamma);
template __global__ void PPMC_CTU<2>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny,
                                     int nz, Real dx, Real dt, Real gamma);

template __global__ __launch_bounds__(TPB) void PPMC_VL<0>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                           int nx, int ny, int nz, Real gamma);
template __global__ __launch_bounds__(TPB) void PPMC_VL<1>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                           int nx, int ny, int nz, Real gamma);
template __global__ __launch_bounds__(TPB) void PPMC_VL<2>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                           int nx, int ny, int nz, Real gamma);
// =====================================================================================================================

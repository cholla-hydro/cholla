/*! \file plmc_cuda.cu
 *  \brief Definitions of the piecewise linear reconstruction functions with
           limiting applied in the characteristic variables, as described
           in Stone et al., 2008. */

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../reconstruction/reconstruction_internals.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif  // DE

/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
 *dev_bounds_R, int nx, int ny, int nz, Real dx, Real dt, Real
 gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using plm. */
__global__ __launch_bounds__(TPB) void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                                 int ny, int nz, Real dx, Real dt, Real gamma, int dir, int n_fields)
{
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Ensure that we are only operating on cells that will be used
  if (reconstruction::Thread_Guard<2>(nx, ny, nz, xid, yid, zid)) {
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
  hydro_utilities::Primitive const cell_i =
      reconstruction::Load_Data(dev_conserved, xid, yid, zid, nx, ny, n_cells, o1, o2, o3, gamma);

  // cell i-1. The equality checks the direction and will subtract one from the correct direction
  hydro_utilities::Primitive const cell_imo = reconstruction::Load_Data(
      dev_conserved, xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny, n_cells, o1, o2, o3, gamma);

  // cell i+1. The equality checks the direction and add one to the correct direction
  hydro_utilities::Primitive const cell_ipo = reconstruction::Load_Data(
      dev_conserved, xid + int(dir == 0), yid + int(dir == 1), zid + int(dir == 2), nx, ny, n_cells, o1, o2, o3, gamma);

  // calculate the adiabatic sound speed in cell i
  Real const sound_speed         = hydro_utilities::Calc_Sound_Speed(cell_i.pressure, cell_i.density, gamma);
  Real const sound_speed_squared = sound_speed * sound_speed;

// Compute the eigenvectors
#ifdef MHD
  reconstruction::EigenVecs const eigenvectors =
      reconstruction::Compute_Eigenvectors(cell_i, sound_speed, sound_speed_squared, gamma);
#else
  reconstruction::EigenVecs eigenvectors;
#endif  // MHD

  // Compute the left, right, centered, and van Leer differences of the
  // primitive variables Note that here L and R refer to locations relative to
  // the cell center

  // left
  hydro_utilities::Primitive const del_L = reconstruction::Compute_Slope(cell_imo, cell_i);

  // right
  hydro_utilities::Primitive const del_R = reconstruction::Compute_Slope(cell_i, cell_ipo);

  // centered
  hydro_utilities::Primitive const del_C = reconstruction::Compute_Slope(cell_imo, cell_ipo, 0.5);

  // Van Leer
  hydro_utilities::Primitive const del_G = reconstruction::Van_Leer_Slope(del_L, del_R);

  // Project the left, right, centered and van Leer differences onto the
  // characteristic variables Stone Eqn 37 (del_a are differences in
  // characteristic variables, see Stone for notation) Use the eigenvectors
  // given in Stone 2008, Appendix A
  reconstruction::Characteristic const del_a_L =
      reconstruction::Primitive_To_Characteristic(cell_i, del_L, eigenvectors, sound_speed, sound_speed_squared, gamma);

  reconstruction::Characteristic const del_a_R =
      reconstruction::Primitive_To_Characteristic(cell_i, del_R, eigenvectors, sound_speed, sound_speed_squared, gamma);

  reconstruction::Characteristic const del_a_C =
      reconstruction::Primitive_To_Characteristic(cell_i, del_C, eigenvectors, sound_speed, sound_speed_squared, gamma);

  reconstruction::Characteristic const del_a_G =
      reconstruction::Primitive_To_Characteristic(cell_i, del_G, eigenvectors, sound_speed, sound_speed_squared, gamma);

  // Apply monotonicity constraints to the differences in the characteristic variables and project the monotonized
  // difference in the characteristic variables back onto the primitive variables Stone Eqn 39
  hydro_utilities::Primitive del_m_i = reconstruction::Monotonize_Characteristic_Return_Primitive(
      cell_i, del_L, del_R, del_C, del_G, del_a_L, del_a_R, del_a_C, del_a_G, eigenvectors, sound_speed,
      sound_speed_squared, gamma);

  // Compute the left and right interface values using the monotonized difference in the primitive variables
  hydro_utilities::Primitive interface_L_iph = reconstruction::Calc_Interface_Linear(cell_i, del_m_i, 1.0);
  hydro_utilities::Primitive interface_R_imh = reconstruction::Calc_Interface_Linear(cell_i, del_m_i, -1.0);

#ifndef VL

  Real const dtodx = dt / dx;

  // Compute the eigenvalues of the linearized equations in the
  // primitive variables using the cell-centered primitive variables
  Real const lambda_m = cell_i.velocity.x - sound_speed;
  Real const lambda_0 = cell_i.velocity.x;
  Real const lambda_p = cell_i.velocity.x + sound_speed;

  // Integrate linear interpolation function over domain of dependence
  // defined by max(min) eigenvalue
  Real qx                    = -0.5 * fmin(lambda_m, 0.0) * dtodx;
  interface_R_imh.density    = interface_R_imh.density + qx * del_m_i.density;
  interface_R_imh.velocity.x = interface_R_imh.velocity.x + qx * del_m_i.velocity.x;
  interface_R_imh.velocity.y = interface_R_imh.velocity.y + qx * del_m_i.velocity.y;
  interface_R_imh.velocity.z = interface_R_imh.velocity.z + qx * del_m_i.velocity.z;
  interface_R_imh.pressure   = interface_R_imh.pressure + qx * del_m_i.pressure;

  qx                         = 0.5 * fmax(lambda_p, 0.0) * dtodx;
  interface_L_iph.density    = interface_L_iph.density - qx * del_m_i.density;
  interface_L_iph.velocity.x = interface_L_iph.velocity.x - qx * del_m_i.velocity.x;
  interface_L_iph.velocity.y = interface_L_iph.velocity.y - qx * del_m_i.velocity.y;
  interface_L_iph.velocity.z = interface_L_iph.velocity.z - qx * del_m_i.velocity.z;
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
             (-cell_i.density * del_m_i.velocity.x / (2 * sound_speed) + del_m_i.pressure / (2 * sound_speed_squared));
    sum_1 += lamdiff * (del_m_i.velocity.x / 2.0 - del_m_i.pressure / (2 * sound_speed * cell_i.density));
    sum_4 += lamdiff * (-cell_i.density * del_m_i.velocity.x * sound_speed / 2.0 + del_m_i.pressure / 2.0);
  }
  if (lambda_0 >= 0) {
    Real lamdiff = lambda_p - lambda_0;

    sum_0 += lamdiff * (del_m_i.density - del_m_i.pressure / (sound_speed_squared));
    sum_2 += lamdiff * del_m_i.velocity.y;
    sum_3 += lamdiff * del_m_i.velocity.z;
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
             (cell_i.density * del_m_i.velocity.x / (2 * sound_speed) + del_m_i.pressure / (2 * sound_speed_squared));
    sum_1 += lamdiff * (del_m_i.velocity.x / 2.0 + del_m_i.pressure / (2 * sound_speed * cell_i.density));
    sum_4 += lamdiff * (cell_i.density * del_m_i.velocity.x * sound_speed / 2.0 + del_m_i.pressure / 2.0);
  }

  // add the corrections to the initial guesses for the interface values
  interface_L_iph.density += 0.5 * dtodx * sum_0;
  interface_L_iph.velocity.x += 0.5 * dtodx * sum_1;
  interface_L_iph.velocity.y += 0.5 * dtodx * sum_2;
  interface_L_iph.velocity.z += 0.5 * dtodx * sum_3;
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
             (-cell_i.density * del_m_i.velocity.x / (2 * sound_speed) + del_m_i.pressure / (2 * sound_speed_squared));
    sum_1 += lamdiff * (del_m_i.velocity.x / 2.0 - del_m_i.pressure / (2 * sound_speed * cell_i.density));
    sum_4 += lamdiff * (-cell_i.density * del_m_i.velocity.x * sound_speed / 2.0 + del_m_i.pressure / 2.0);
  }
  if (lambda_0 <= 0) {
    Real lamdiff = lambda_m - lambda_0;

    sum_0 += lamdiff * (del_m_i.density - del_m_i.pressure / (sound_speed_squared));
    sum_2 += lamdiff * del_m_i.velocity.y;
    sum_3 += lamdiff * del_m_i.velocity.z;
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
             (cell_i.density * del_m_i.velocity.x / (2 * sound_speed) + del_m_i.pressure / (2 * sound_speed_squared));
    sum_1 += lamdiff * (del_m_i.velocity.x / 2.0 + del_m_i.pressure / (2 * sound_speed * cell_i.density));
    sum_4 += lamdiff * (cell_i.density * del_m_i.velocity.x * sound_speed / 2.0 + del_m_i.pressure / 2.0);
  }

  // add the corrections
  interface_R_imh.density += 0.5 * dtodx * sum_0;
  interface_R_imh.velocity.x += 0.5 * dtodx * sum_1;
  interface_R_imh.velocity.y += 0.5 * dtodx * sum_2;
  interface_R_imh.velocity.z += 0.5 * dtodx * sum_3;
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

  // Convert the left and right states in the primitive to the conserved variables send final values back from kernel
  // bounds_R refers to the right side of the i-1/2 interface
  size_t id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
  reconstruction::Write_Data(interface_L_iph, dev_bounds_L, dev_conserved, id, n_cells, o1, o2, o3, gamma);

  id = cuda_utilities::compute1DIndex(xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny);
  reconstruction::Write_Data(interface_R_imh, dev_bounds_R, dev_conserved, id, n_cells, o1, o2, o3, gamma);
}

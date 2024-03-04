/*! \file plmc_cuda.h
 *  \brief Declarations of the cuda plm kernels, characteristic reconstruction
 * version. */

#ifndef PLMC_CUDA_H
#define PLMC_CUDA_H

#include "../global/global.h"
#include "../grid/grid_enum.h"
#include "../reconstruction/reconstruction_internals.h"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"

/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
 *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real
 gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using plm. */
template <int dir>
__global__ __launch_bounds__(TPB) void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                                 int ny, int nz, Real dx, Real dt, Real gamma);

namespace reconstruction
{
template <size_t direction>
auto __device__ __inline__ PLMC_Reconstruction(Real *dev_conserved, int const xid, int const yid, int const zid,
                                               int const nx, int const ny, int const nz, Real const dx, Real const dt,
                                               Real const gamma)
{
  // Compute the total number of cells
  int const n_cells = nx * ny * nz;

  // load the 3-cell stencil into registers
  // cell i
  hydro_utilities::Primitive const cell_i =
      hydro_utilities::Load_Cell_Primitive<direction>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);

  // cell i-1. The equality checks the direction and will subtract one from the correct direction
  hydro_utilities::Primitive const cell_imo = hydro_utilities::Load_Cell_Primitive<direction>(
      dev_conserved, xid - int(direction == 0), yid - int(direction == 1), zid - int(direction == 2), nx, ny, n_cells,
      gamma);

  // cell i+1. The equality checks the direction and add one to the correct direction
  hydro_utilities::Primitive const cell_ipo = hydro_utilities::Load_Cell_Primitive<direction>(
      dev_conserved, xid + int(direction == 0), yid + int(direction == 1), zid + int(direction == 2), nx, ny, n_cells,
      gamma);

  // Compute the eigenvectors
  reconstruction::EigenVecs const eigenvectors = reconstruction::Compute_Eigenvectors(cell_i, gamma);

  // Compute the left, right, centered, and van Leer differences of the primitive variables Note that here L and R refer
  // to locations relative to the cell center

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
      reconstruction::Primitive_To_Characteristic(cell_i, del_L, eigenvectors, gamma);

  reconstruction::Characteristic const del_a_R =
      reconstruction::Primitive_To_Characteristic(cell_i, del_R, eigenvectors, gamma);

  reconstruction::Characteristic const del_a_C =
      reconstruction::Primitive_To_Characteristic(cell_i, del_C, eigenvectors, gamma);

  reconstruction::Characteristic const del_a_G =
      reconstruction::Primitive_To_Characteristic(cell_i, del_G, eigenvectors, gamma);

  // Apply monotonicity constraints to the differences in the characteristic variables and project the monotonized
  // difference in the characteristic variables back onto the primitive variables Stone Eqn 39
  reconstruction::Characteristic const del_a_m = reconstruction::Van_Leer_Limiter(del_a_L, del_a_R, del_a_C, del_a_G);

  // Project back into the primitive variables.
  hydro_utilities::Primitive del_m = Characteristic_To_Primitive(cell_i, del_a_m, eigenvectors, gamma);

  // Limit the variables that aren't transformed by the characteristic projection
#ifdef DE
  del_m.gas_energy_specific = Van_Leer_Limiter(del_L.gas_energy_specific, del_R.gas_energy_specific,
                                               del_C.gas_energy_specific, del_G.gas_energy_specific);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_m.scalar_specific[i] = Van_Leer_Limiter(del_L.scalar_specific[i], del_R.scalar_specific[i],
                                                del_C.scalar_specific[i], del_G.scalar_specific[i]);
  }
#endif  // SCALAR

  // Compute the left and right interface values using the monotonized difference in the primitive variables
  hydro_utilities::Primitive interface_L_iph = reconstruction::Calc_Interface_Linear(cell_i, del_m, 1.0);
  hydro_utilities::Primitive interface_R_imh = reconstruction::Calc_Interface_Linear(cell_i, del_m, -1.0);

  // Limit the interfaces
  reconstruction::Plm_Limit_Interfaces(interface_L_iph, interface_R_imh, cell_imo, cell_i, cell_ipo);

#ifndef VL

  Real const dtodx = dt / dx;

  // Compute the eigenvalues of the linearized equations in the
  // primitive variables using the cell-centered primitive variables
  Real const lambda_m = cell_i.velocity.x - eigenvectors.sound_speed;
  Real const lambda_0 = cell_i.velocity.x;
  Real const lambda_p = cell_i.velocity.x + eigenvectors.sound_speed;

  // Integrate linear interpolation function over domain of dependence
  // defined by max(min) eigenvalue
  Real qx                    = -0.5 * fmin(lambda_m, 0.0) * dtodx;
  interface_R_imh.density    = interface_R_imh.density + qx * del_m.density;
  interface_R_imh.velocity.x = interface_R_imh.velocity.x + qx * del_m.velocity.x;
  interface_R_imh.velocity.y = interface_R_imh.velocity.y + qx * del_m.velocity.y;
  interface_R_imh.velocity.z = interface_R_imh.velocity.z + qx * del_m.velocity.z;
  interface_R_imh.pressure   = interface_R_imh.pressure + qx * del_m.pressure;

  qx                         = 0.5 * fmax(lambda_p, 0.0) * dtodx;
  interface_L_iph.density    = interface_L_iph.density - qx * del_m.density;
  interface_L_iph.velocity.x = interface_L_iph.velocity.x - qx * del_m.velocity.x;
  interface_L_iph.velocity.y = interface_L_iph.velocity.y - qx * del_m.velocity.y;
  interface_L_iph.velocity.z = interface_L_iph.velocity.z - qx * del_m.velocity.z;
  interface_L_iph.pressure   = interface_L_iph.pressure - qx * del_m.pressure;

  #ifdef DE
  interface_R_imh.gas_energy_specific = interface_R_imh.gas_energy_specific + qx * del_m.gas_energy_specific;
  interface_L_iph.gas_energy_specific = interface_L_iph.gas_energy_specific - qx * del_m.gas_energy_specific;
  #endif  // DE

  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar[i] = interface_R_imh.scalar[i] + qx * del_m.scalar[i];
    interface_L_iph.scalar[i] = interface_L_iph.scalar[i] - qx * del_m.scalar[i];
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

    sum_0 += lamdiff * (-cell_i.density * del_m.velocity.x / (2 * eigenvectors.sound_speed) +
                        del_m.pressure / (2 * eigenvectors.sound_speed * eigenvectors.sound_speed));
    sum_1 += lamdiff * (del_m.velocity.x / 2.0 - del_m.pressure / (2 * eigenvectors.sound_speed * cell_i.density));
    sum_4 += lamdiff * (-cell_i.density * del_m.velocity.x * eigenvectors.sound_speed / 2.0 + del_m.pressure / 2.0);
  }
  if (lambda_0 >= 0) {
    Real lamdiff = lambda_p - lambda_0;

    sum_0 += lamdiff * (del_m.density - del_m.pressure / (eigenvectors.sound_speed * eigenvectors.sound_speed));
    sum_2 += lamdiff * del_m.velocity.y;
    sum_3 += lamdiff * del_m.velocity.z;
  #ifdef DE
    sum_ge += lamdiff * del_m.gas_energy_specific;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += lamdiff * del_m.scalar[i];
    }
  #endif  // SCALAR
  }
  if (lambda_p >= 0) {
    Real lamdiff = lambda_p - lambda_p;

    sum_0 += lamdiff * (cell_i.density * del_m.velocity.x / (2 * eigenvectors.sound_speed) +
                        del_m.pressure / (2 * eigenvectors.sound_speed * eigenvectors.sound_speed));
    sum_1 += lamdiff * (del_m.velocity.x / 2.0 + del_m.pressure / (2 * eigenvectors.sound_speed * cell_i.density));
    sum_4 += lamdiff * (cell_i.density * del_m.velocity.x * eigenvectors.sound_speed / 2.0 + del_m.pressure / 2.0);
  }

  // add the corrections to the initial guesses for the interface values
  interface_L_iph.density += 0.5 * dtodx * sum_0;
  interface_L_iph.velocity.x += 0.5 * dtodx * sum_1;
  interface_L_iph.velocity.y += 0.5 * dtodx * sum_2;
  interface_L_iph.velocity.z += 0.5 * dtodx * sum_3;
  interface_L_iph.pressure += 0.5 * dtodx * sum_4;
  #ifdef DE
  interface_L_iph.gas_energy_specific += 0.5 * dtodx * sum_ge;
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

    sum_0 += lamdiff * (-cell_i.density * del_m.velocity.x / (2 * eigenvectors.sound_speed) +
                        del_m.pressure / (2 * eigenvectors.sound_speed * eigenvectors.sound_speed));
    sum_1 += lamdiff * (del_m.velocity.x / 2.0 - del_m.pressure / (2 * eigenvectors.sound_speed * cell_i.density));
    sum_4 += lamdiff * (-cell_i.density * del_m.velocity.x * eigenvectors.sound_speed / 2.0 + del_m.pressure / 2.0);
  }
  if (lambda_0 <= 0) {
    Real lamdiff = lambda_m - lambda_0;

    sum_0 += lamdiff * (del_m.density - del_m.pressure / (eigenvectors.sound_speed * eigenvectors.sound_speed));
    sum_2 += lamdiff * del_m.velocity.y;
    sum_3 += lamdiff * del_m.velocity.z;
  #ifdef DE
    sum_ge += lamdiff * del_m.gas_energy_specific;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += lamdiff * del_m.scalar[i];
    }
  #endif  // SCALAR
  }
  if (lambda_p <= 0) {
    Real lamdiff = lambda_m - lambda_p;

    sum_0 += lamdiff * (cell_i.density * del_m.velocity.x / (2 * eigenvectors.sound_speed) +
                        del_m.pressure / (2 * eigenvectors.sound_speed * eigenvectors.sound_speed));
    sum_1 += lamdiff * (del_m.velocity.x / 2.0 + del_m.pressure / (2 * eigenvectors.sound_speed * cell_i.density));
    sum_4 += lamdiff * (cell_i.density * del_m.velocity.x * eigenvectors.sound_speed / 2.0 + del_m.pressure / 2.0);
  }

  // add the corrections
  interface_R_imh.density += 0.5 * dtodx * sum_0;
  interface_R_imh.velocity.x += 0.5 * dtodx * sum_1;
  interface_R_imh.velocity.y += 0.5 * dtodx * sum_2;
  interface_R_imh.velocity.z += 0.5 * dtodx * sum_3;
  interface_R_imh.pressure += 0.5 * dtodx * sum_4;
  #ifdef DE
  interface_R_imh.gas_energy_specific += 0.5 * dtodx * sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar[i] += 0.5 * dtodx * sum_scalar[i];
  }
  #endif  // SCALAR
#endif    // CTU

  struct LocalReturnStruct {
    hydro_utilities::Primitive left, right;
  };
  return LocalReturnStruct{interface_L_iph, interface_R_imh};
}
}  // namespace reconstruction

#endif  // PLMC_CUDA_H

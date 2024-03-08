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

namespace reconstruction
{

// sign = -1 for r_imh, +1 for l_iph
void __device__ __host__ __inline__ PLMC_Characteristic_Evolution(hydro_utilities::Primitive const &cell_i,
                                                                  hydro_utilities::Primitive const &del_m,
                                                                  Real const dt, Real const dx, Real const sound_speed,
                                                                  Real const sign,
                                                                  hydro_utilities::Primitive &interface)
{
  Real const dtodx = dt / dx;

  // Compute the eigenvalues of the linearized equations in the primitive variables using the cell-centered primitive
  // variables
  Real const lambda_m = cell_i.velocity.x - sound_speed;
  Real const lambda_0 = cell_i.velocity.x;
  Real const lambda_p = cell_i.velocity.x + sound_speed;

  // Integrate linear interpolation function over domain of dependence defined by max(min) eigenvalue
  Real qx = (sign > 0) ? fmax(lambda_p, 0.0) : fmin(lambda_m, 0.0);
  qx *= sign * 0.5 * dtodx;
  interface.density    = interface.density - sign * qx * del_m.density;
  interface.velocity.x = interface.velocity.x - sign * qx * del_m.velocity.x;
  interface.velocity.y = interface.velocity.y - sign * qx * del_m.velocity.y;
  interface.velocity.z = interface.velocity.z - sign * qx * del_m.velocity.z;
  interface.pressure   = interface.pressure - sign * qx * del_m.pressure;

#ifdef DE
  interface.gas_energy_specific = interface.gas_energy_specific - sign * qx * del_m.gas_energy_specific;
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface.scalar_specific[i] = interface.scalar_specific[i] - sign * qx * del_m.scalar_specific[i];
  }
#endif  // SCALAR

  // Perform the characteristic tracing
  // Stone Eqns 42 & 43

  // ================================================================================================================
  // left-hand interface value, i+1/2
  // ================================================================================================================
  if (sign > 0) {
    Real sum_0 = 0.0, sum_1 = 0.0, sum_2 = 0.0, sum_3 = 0.0, sum_4 = 0.0;
#ifdef DE
    Real sum_ge = 0.0;
#endif  // DE
#ifdef SCALAR
    Real sum_scalar[NSCALARS];
#endif  // SCALAR
    if (lambda_m >= 0) {
      Real lamdiff = lambda_p - lambda_m;

      sum_0 += lamdiff * (-cell_i.density * del_m.velocity.x / (2 * sound_speed) +
                          del_m.pressure / (2 * sound_speed * sound_speed));
      sum_1 += lamdiff * (del_m.velocity.x / 2.0 - del_m.pressure / (2 * sound_speed * cell_i.density));
      sum_4 += lamdiff * (-cell_i.density * del_m.velocity.x * sound_speed / 2.0 + del_m.pressure / 2.0);
    }
    if (lambda_0 >= 0) {
      Real lamdiff = lambda_p - lambda_0;

      sum_0 += lamdiff * (del_m.density - del_m.pressure / (sound_speed * sound_speed));
      sum_2 += lamdiff * del_m.velocity.y;
      sum_3 += lamdiff * del_m.velocity.z;
#ifdef DE
      Real const sum_ge = lamdiff * del_m.gas_energy_specific;
#endif  // DE
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        sum_scalar[i] = lamdiff * del_m.scalar_specific[i];
      }
#endif  // SCALAR
    }
    if (lambda_p >= 0) {
      Real lamdiff = lambda_p - lambda_p;

      sum_0 += lamdiff * (cell_i.density * del_m.velocity.x / (2 * sound_speed) +
                          del_m.pressure / (2 * sound_speed * sound_speed));
      sum_1 += lamdiff * (del_m.velocity.x / 2.0 + del_m.pressure / (2 * sound_speed * cell_i.density));
      sum_4 += lamdiff * (cell_i.density * del_m.velocity.x * sound_speed / 2.0 + del_m.pressure / 2.0);
    }

    // add the corrections to the initial guesses for the interface values
    interface.density += 0.5 * dtodx * sum_0;
    interface.velocity.x += 0.5 * dtodx * sum_1;
    interface.velocity.y += 0.5 * dtodx * sum_2;
    interface.velocity.z += 0.5 * dtodx * sum_3;
    interface.pressure += 0.5 * dtodx * sum_4;
#ifdef DE
    interface.gas_energy_specific += 0.5 * dtodx * sum_ge;
#endif  // DE
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      interface.scalar_specific[i] += 0.5 * dtodx * sum_scalar[i];
    }
#endif  // SCALAR
  }
  // ================================================================================================================
  // right-hand interface value, i-1/2
  // ================================================================================================================
  else {
    Real sum_0 = 0.0, sum_1 = 0.0, sum_2 = 0.0, sum_3 = 0.0, sum_4 = 0.0;

#ifdef DE
    Real sum_ge = 0.0;
#endif  // DE
#ifdef SCALAR
    Real sum_scalar[NSCALARS];
#endif  // SCALAR
    if (lambda_m <= 0) {
      Real lamdiff = lambda_m - lambda_m;

      sum_0 += lamdiff * (-cell_i.density * del_m.velocity.x / (2 * sound_speed) +
                          del_m.pressure / (2 * sound_speed * sound_speed));
      sum_1 += lamdiff * (del_m.velocity.x / 2.0 - del_m.pressure / (2 * sound_speed * cell_i.density));
      sum_4 += lamdiff * (-cell_i.density * del_m.velocity.x * sound_speed / 2.0 + del_m.pressure / 2.0);
    }
    if (lambda_0 <= 0) {
      Real lamdiff = lambda_m - lambda_0;

      sum_0 += lamdiff * (del_m.density - del_m.pressure / (sound_speed * sound_speed));
      sum_2 += lamdiff * del_m.velocity.y;
      sum_3 += lamdiff * del_m.velocity.z;
#ifdef DE
      sum_ge += lamdiff * del_m.gas_energy_specific;
#endif  // DE
#ifdef SCALAR
      for (int i = 0; i < NSCALARS; i++) {
        sum_scalar[i] += lamdiff * del_m.scalar_specific[i];
      }
#endif  // SCALAR
    }
    if (lambda_p <= 0) {
      Real lamdiff = lambda_m - lambda_p;

      sum_0 += lamdiff * (cell_i.density * del_m.velocity.x / (2 * sound_speed) +
                          del_m.pressure / (2 * sound_speed * sound_speed));
      sum_1 += lamdiff * (del_m.velocity.x / 2.0 + del_m.pressure / (2 * sound_speed * cell_i.density));
      sum_4 += lamdiff * (cell_i.density * del_m.velocity.x * sound_speed / 2.0 + del_m.pressure / 2.0);
    }

    // add the corrections
    interface.density += 0.5 * dtodx * sum_0;
    interface.velocity.x += 0.5 * dtodx * sum_1;
    interface.velocity.y += 0.5 * dtodx * sum_2;
    interface.velocity.z += 0.5 * dtodx * sum_3;
    interface.pressure += 0.5 * dtodx * sum_4;
#ifdef DE
    interface.gas_energy_specific += 0.5 * dtodx * sum_ge;
#endif  // DE
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      interface.scalar_specific[i] += 0.5 * dtodx * sum_scalar[i];
    }
#endif  // SCALAR
  }
}

template <uint direction>
auto __device__ __inline__ PLMC_Reconstruction(hydro_utilities::Primitive const &cell_im1,
                                               hydro_utilities::Primitive const &cell_i,
                                               hydro_utilities::Primitive const &cell_ip1, Real const dx, Real const dt,
                                               Real const gamma, Real const sign)
{
  // Compute the eigenvectors
  reconstruction::EigenVecs const eigenvectors = reconstruction::Compute_Eigenvectors(cell_i, gamma);

  // Compute the left, right, centered, and van Leer differences of the primitive variables Note that here L and R refer
  // to locations relative to the cell center

  // left
  hydro_utilities::Primitive const del_L = reconstruction::Compute_Slope(cell_im1, cell_i);

  // right
  hydro_utilities::Primitive const del_R = reconstruction::Compute_Slope(cell_i, cell_ip1);

  // centered
  hydro_utilities::Primitive const del_C = reconstruction::Compute_Slope(cell_im1, cell_ip1, 0.5);

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

  // Compute the interface values using the monotonized difference in the primitive variables
  hydro_utilities::Primitive interface = reconstruction::Calc_Interface_Linear(cell_i, del_m, sign);

// Do the characteristic tracing
#ifndef VL
  PLMC_Characteristic_Evolution(cell_i, del_m, dt, dx, eigenvectors.sound_speed, sign, interface);
#endif  // VL

  // apply minimum constraints
  interface.density  = fmax(interface.density, (Real)TINY_NUMBER);
  interface.pressure = fmax(interface.pressure, (Real)TINY_NUMBER);

  return interface;
}
}  // namespace reconstruction

#endif  // PLMC_CUDA_H

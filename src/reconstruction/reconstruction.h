/*!
 * \file reconstruction.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contain the various structs and device functions needed for interface reconstruction
 *
 */

#pragma once

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"

/*!
 * \brief Namespace to contain various utilities for the interface reconstruction kernels
 *
 */
namespace reconstruction
{
// =====================================================================================================================
/*!
 * \brief A struct for the primitive variables
 *
 */
struct Primitive {
  // Hydro variables
  Real density, velocity_x, velocity_y, velocity_z, pressure;

#ifdef MHD
  // These are all cell centered values
  Real magnetic_x, magnetic_y, magnetic_z;
#endif  // MHD

#ifdef DE
  Real gas_energy;
#endif  // DE

#ifdef SCALAR
  Real scalar[grid_enum::nscalars];
#endif  // SCALAR
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief A struct for the characteristic variables
 *
 */
struct Characteristic {
  // Hydro variables
  Real a0, a1, a2, a3, a4;

#ifdef MHD
  Real a5, a6;
#endif  // MHD
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Load the data for reconstruction
 *
 * \param[in] dev_conserved The conserved array
 * \param[in] xid The xid of the cell to load data from
 * \param[in] yid The yid of the cell to load data from
 * \param[in] zid The zid of the cell to load data from
 * \param[in] nx Size in the X direction
 * \param[in] ny Size in the Y direction
 * \param[in] n_cells The total number of cells
 * \param[in] o1 Directional parameter
 * \param[in] o2 Directional parameter
 * \param[in] o3 Directional parameter
 * \param[in] gamma The adiabatic index
 * \return Primitive The loaded cell data
 */
Primitive __device__ __host__ __inline__ Load_Data(Real const *dev_conserved, size_t const &xid, size_t const &yid,
                                                   size_t const &zid, size_t const &nx, size_t const &ny,
                                                   size_t const &n_cells, size_t const &o1, size_t const &o2,
                                                   size_t const &o3, Real const &gamma)
{  // Compute index
  size_t const id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);

  // Declare the variable we will return
  Primitive loaded_data;

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
      loaded_data.magnetic_x = magnetic_centered.x;
      loaded_data.magnetic_y = magnetic_centered.y;
      loaded_data.magnetic_z = magnetic_centered.z;
      break;
    case grid_enum::momentum_y:
      loaded_data.magnetic_x = magnetic_centered.y;
      loaded_data.magnetic_y = magnetic_centered.z;
      loaded_data.magnetic_z = magnetic_centered.x;
      break;
    case grid_enum::momentum_z:
      loaded_data.magnetic_x = magnetic_centered.z;
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
  loaded_data.pressure = hydro_utilities::Calc_Pressure_Primitive(
      dev_conserved[grid_enum::Energy * n_cells + id], loaded_data.density, loaded_data.velocity_x,
      loaded_data.velocity_y, loaded_data.velocity_z, gamma, loaded_data.magnetic_x, loaded_data.magnetic_y,
      loaded_data.magnetic_z);
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
/*!
 * \brief Compute a simple slope. Equation is `coef * (left - right)`.
 *
 * \param[in] left The data on the positive side of the slope
 * \param[in] right The data on the negative side of the slope
 * \param[in] coef The coefficient to multiply the slope by. Defaults to 1.0
 * \return Primitive The slopes
 */
Primitive __device__ __host__ __inline__ Compute_Slope(Primitive const &left, Primitive const &right,
                                                       Real const &coef = 1.0)
{
  Primitive slopes;

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
/*!
 * \brief Compute the Van Lear slope from the left and right slopes
 *
 * \param[in] left_slope The left slope
 * \param[in] right_slope The right slope
 * \return Primitive The Van Leer slope
 */
Primitive __device__ __host__ __inline__ Van_Leer_Slope(Primitive const &left_slope, Primitive const &right_slope)
{
  Primitive vl_slopes;

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

// =====================================================================================================================
/*!
 * \brief Project from the primitive variables slopes to the characteristic variables slopes. Stone Eqn 37. Use the
 * eigenvectors given in Stone 2008, Appendix A
 *
 * \param[in] primitive The primitive variables
 * \param[in] primitive_slope The primitive variables slopes
 * \param[in] sound_speed The speed of sound
 * \param[in] sound_speed_squared The speed of sound squared
 * \param[in] gamma The adiabatic index
 * \return Characteristic
 */
Characteristic __device__ __inline__ Primitive_To_Characteristic(Primitive const &primitive,
                                                                 Primitive const &primitive_slope,
                                                                 Real const &sound_speed,
                                                                 Real const &sound_speed_squared, Real const &gamma)
{
  Characteristic output;

#ifdef MHD
  // This is taken from Stone et al. 2008, appendix A. Equation numbers will be quoted as relevant

  // First, compute some basic quantities we will need later
  Real const inverse_sqrt_density = rsqrt(primitive.density);

  // Compute wave speeds and their squares
  Real const magnetosonic_speed_fast = mhd::utils::fastMagnetosonicSpeed(
      primitive.density, primitive.pressure, primitive.magnetic_x, primitive.magnetic_y, primitive.magnetic_z, gamma);
  Real const magnetosonic_speed_slow = mhd::utils::slowMagnetosonicSpeed(
      primitive.density, primitive.pressure, primitive.magnetic_x, primitive.magnetic_y, primitive.magnetic_z, gamma);

  Real const magnetosonic_speed_fast_squared = magnetosonic_speed_fast * magnetosonic_speed_fast;
  Real const magnetosonic_speed_slow_squared = magnetosonic_speed_slow * magnetosonic_speed_slow;

  // Compute Alphas (equation A16)
  Real alpha_fast, alpha_slow;
  if (Real const denom = (magnetosonic_speed_fast_squared - magnetosonic_speed_slow_squared),
      numerator_2      = (magnetosonic_speed_fast_squared - sound_speed_squared);
      denom <= 0.0 or numerator_2 <= 0.0) {
    alpha_fast = 1.0;
    alpha_slow = 0.0;
  } else if (Real const numerator_1 = (sound_speed_squared - magnetosonic_speed_slow_squared); numerator_1 <= 0.0) {
    alpha_fast = 0.0;
    alpha_slow = 1.0;
  } else {
    alpha_fast = sqrt(numerator_1 / denom);
    alpha_slow = sqrt(numerator_2 / denom);
  }

  // Compute Betas (equation A17). Note that rhypot can return an inf if By and Bz are both zero, the isfinite check
  // handles that case
  Real const beta_denom = rhypot(primitive.magnetic_y, primitive.magnetic_z);
  Real const beta_y     = (isfinite(beta_denom)) ? primitive.magnetic_y * beta_denom : 1.0;
  Real const beta_z     = (isfinite(beta_denom)) ? primitive.magnetic_z * beta_denom : 0.0;

  // Compute Q(s) (equation A14)
  Real const n_fs   = 0.5 / sound_speed_squared;  // equation A19
  Real const sign   = copysign(1.0, primitive.magnetic_x);
  Real const q_fast = sign * n_fs * alpha_fast * magnetosonic_speed_fast;
  Real const q_slow = sign * n_fs * alpha_slow * magnetosonic_speed_slow;

  // Compute A(s) (equation A15)
  Real const a_prime_fast = 0.5 * alpha_fast / (sound_speed * sqrt(primitive.density));
  Real const a_prime_slow = 0.5 * alpha_slow / (sound_speed * sqrt(primitive.density));

  // Multiply the slopes by the left eigenvector matrix given in equation 18
  output.a0 =
      n_fs * alpha_fast *
          (primitive_slope.pressure / primitive.density - magnetosonic_speed_fast * primitive_slope.velocity_x) +
      q_slow * (beta_y * primitive_slope.velocity_y + beta_z * primitive_slope.velocity_z) +
      a_prime_slow * (beta_y * primitive_slope.magnetic_y + beta_z * primitive_slope.magnetic_z);

  output.a1 = 0.5 * (beta_y * (primitive_slope.magnetic_z * sign * inverse_sqrt_density + primitive_slope.velocity_z) -
                     beta_z * (primitive_slope.magnetic_y * sign * inverse_sqrt_density + primitive_slope.velocity_y));

  output.a2 =
      n_fs * alpha_slow *
          (primitive_slope.pressure / primitive.density - magnetosonic_speed_slow * primitive_slope.velocity_x) -
      q_fast * (beta_y * primitive_slope.velocity_y + beta_z * primitive_slope.velocity_z) -
      a_prime_fast * (beta_y * primitive_slope.magnetic_y + beta_z * primitive_slope.magnetic_z);

  output.a3 = primitive_slope.density - primitive_slope.pressure / sound_speed_squared;

  output.a4 =
      n_fs * alpha_slow *
          (primitive_slope.pressure / primitive.density + magnetosonic_speed_slow * primitive_slope.velocity_x) +
      q_fast * (beta_y * primitive_slope.velocity_y + beta_z * primitive_slope.velocity_z) -
      a_prime_fast * (beta_y * primitive_slope.magnetic_y + beta_z * primitive_slope.magnetic_z);
  output.a5 = 0.5 * (beta_y * (primitive_slope.magnetic_z * sign * inverse_sqrt_density - primitive_slope.velocity_z) -
                     beta_z * (primitive_slope.magnetic_y * sign * inverse_sqrt_density - primitive_slope.velocity_y));

  output.a6 =
      n_fs * alpha_fast *
          (primitive_slope.pressure / primitive.density + magnetosonic_speed_fast * primitive_slope.velocity_x) -
      q_slow * (beta_y * primitive_slope.velocity_y + beta_z * primitive_slope.velocity_z) +
      a_prime_slow * (beta_y * primitive_slope.magnetic_y + beta_z * primitive_slope.magnetic_z);

#else   // not MHD
  output.a0 = -primitive.density * primitive_slope.velocity_x / (2.0 * sound_speed) +
              primitive_slope.pressure / (2.0 * sound_speed_squared);
  output.a1 = primitive_slope.density - primitive_slope.pressure / (sound_speed_squared);
  output.a2 = primitive_slope.velocity_y;
  output.a3 = primitive_slope.velocity_z;
  output.a4 = primitive.density * primitive_slope.velocity_x / (2.0 * sound_speed) +
              primitive_slope.pressure / (2.0 * sound_speed_squared);
#endif  // MHD

  return output;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Project from the characteristic variables slopes to the primitive variables slopes. Stone Eqn 39. Use the
 * eigenvectors given in Stone 2008, Appendix A
 *
 * \param[in] primitive The primitive variables
 * \param[in] characteristic_slope The characteristic slopes
 * \param[in] sound_speed The sound speed
 * \param[in] sound_speed_squared The sound speed squared
 * \param[in] gamma The adiabatic index
 * \param[out] output The primitive slopes
 */
void __device__ __inline__ Characteristic_To_Primitive(Primitive const &primitive,
                                                       Characteristic const &characteristic_slope,
                                                       Real const &sound_speed, Real const &sound_speed_squared,
                                                       Real const &gamma, Primitive &output)
{
#ifdef MHD
  // This is taken from Stone et al. 2008, appendix A. Equation numbers will be quoted as relevant

  // Compute wave speeds and their squares
  Real const magnetosonic_speed_fast = mhd::utils::fastMagnetosonicSpeed(
      primitive.density, primitive.pressure, primitive.magnetic_x, primitive.magnetic_y, primitive.magnetic_z, gamma);
  Real const magnetosonic_speed_slow = mhd::utils::slowMagnetosonicSpeed(
      primitive.density, primitive.pressure, primitive.magnetic_x, primitive.magnetic_y, primitive.magnetic_z, gamma);

  Real const magnetosonic_speed_fast_squared = magnetosonic_speed_fast * magnetosonic_speed_fast;
  Real const magnetosonic_speed_slow_squared = magnetosonic_speed_slow * magnetosonic_speed_slow;

  // Compute Alphas (equation A16)
  Real alpha_fast, alpha_slow;
  if (Real const denom = (magnetosonic_speed_fast_squared - magnetosonic_speed_slow_squared),
      numerator_2      = (magnetosonic_speed_fast_squared - sound_speed_squared);
      denom <= 0.0 or numerator_2 <= 0.0) {
    alpha_fast = 1.0;
    alpha_slow = 0.0;
  } else if (Real const numerator_1 = (sound_speed_squared - magnetosonic_speed_slow_squared); numerator_1 <= 0.0) {
    alpha_fast = 0.0;
    alpha_slow = 1.0;
  } else {
    alpha_fast = sqrt(numerator_1 / denom);
    alpha_slow = sqrt(numerator_2 / denom);
  }

  // Compute Betas (equation A17). Note that rhypot can return an inf if By and Bz are both zero, the isfinite check
  // handles that case
  Real const beta_denom = rhypot(primitive.magnetic_y, primitive.magnetic_z);
  Real const beta_y     = (isfinite(beta_denom)) ? primitive.magnetic_y * beta_denom : 1.0;
  Real const beta_z     = (isfinite(beta_denom)) ? primitive.magnetic_z * beta_denom : 0.0;

  // Compute Q(s) (equation A14)
  Real const sign   = copysign(1.0, primitive.magnetic_x);
  Real const q_fast = sign * alpha_fast * magnetosonic_speed_fast;
  Real const q_slow = sign * alpha_slow * magnetosonic_speed_slow;

  // Compute A(s) (equation A15)
  Real const a_prime_fast = alpha_fast * sound_speed * sqrt(primitive.density);
  Real const a_prime_slow = alpha_slow * sound_speed * sqrt(primitive.density);

  // Multiply the slopes by the right eigenvector matrix given in equation 12
  output.density = primitive.density * (alpha_fast * (characteristic_slope.a0 + characteristic_slope.a6) +
                                        alpha_slow * (characteristic_slope.a2 + characteristic_slope.a4)) +
                   characteristic_slope.a3;
  output.velocity_x = magnetosonic_speed_fast * alpha_fast * (characteristic_slope.a6 - characteristic_slope.a0) +
                      magnetosonic_speed_slow * alpha_slow * (characteristic_slope.a4 - characteristic_slope.a2);
  output.velocity_y = beta_y * (q_slow * (characteristic_slope.a0 - characteristic_slope.a6) +
                                q_fast * (characteristic_slope.a4 - characteristic_slope.a2)) +
                      beta_z * (characteristic_slope.a5 - characteristic_slope.a1);
  output.velocity_z = beta_z * (q_slow * (characteristic_slope.a0 - characteristic_slope.a6) +
                                q_fast * (characteristic_slope.a4 - characteristic_slope.a2)) +
                      beta_y * (characteristic_slope.a1 - characteristic_slope.a5);
  output.pressure = primitive.density * sound_speed_squared *
                    (alpha_fast * (characteristic_slope.a0 + characteristic_slope.a6) +
                     alpha_slow * (characteristic_slope.a2 + characteristic_slope.a4));
  output.magnetic_y = beta_y * (a_prime_slow * (characteristic_slope.a0 + characteristic_slope.a6) -
                                a_prime_fast * (characteristic_slope.a2 + characteristic_slope.a4)) -
                      beta_z * sign * sqrt(primitive.density) * (characteristic_slope.a5 + characteristic_slope.a1);
  output.magnetic_z = beta_z * (a_prime_slow * (characteristic_slope.a0 + characteristic_slope.a6) -
                                a_prime_fast * (characteristic_slope.a2 + characteristic_slope.a4)) +
                      beta_y * sign * sqrt(primitive.density) * (characteristic_slope.a5 + characteristic_slope.a1);

#else   // not MHD
  output.density    = characteristic_slope.a0 + characteristic_slope.a1 + characteristic_slope.a4;
  output.velocity_x = sound_speed / primitive.density * (characteristic_slope.a4 - characteristic_slope.a0);
  output.velocity_y = characteristic_slope.a2;
  output.velocity_z = characteristic_slope.a3;
  output.pressure   = sound_speed_squared * (characteristic_slope.a0 + characteristic_slope.a4);
#endif  // MHD
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Monotonize the characteristic slopes and project back into the primitive slopes
 *
 * \param[in] primitive The primitive variables
 * \param[in] del_L The left primitive slopes
 * \param[in] del_R The right primitive slopes
 * \param[in] del_C The centered primitive slopes
 * \param[in] del_G The Van Leer primitive slopes
 * \param[in] del_a_L The left characteristic slopes
 * \param[in] del_a_R The right characteristic slopes
 * \param[in] del_a_C The centered characteristic slopes
 * \param[in] del_a_G The Van Leer characteristic slopes
 * \param[in] sound_speed The sound speed
 * \param[in] sound_speed_squared The sound speed squared
 * \param[in] gamma The adiabatic index
 * \return Primitive The Monotonized primitive slopes
 */
Primitive __device__ __inline__ Monotonize_Characteristic_Return_Primitive(
    Primitive const &primitive, Primitive const &del_L, Primitive const &del_R, Primitive const &del_C,
    Primitive const &del_G, Characteristic const &del_a_L, Characteristic const &del_a_R, Characteristic const &del_a_C,
    Characteristic const &del_a_G, Real const &sound_speed, Real const &sound_speed_squared, Real const &gamma)
{
  // The function that will actually do the monotozation
  auto Monotonize = [](Real const &left, Real const &right, Real const &centered, Real const &van_leer) -> Real {
    if (left * right > 0.0) {
      Real const lim_slope_a = 2.0 * fmin(fabs(left), fabs(right));
      Real const lim_slope_b = fmin(fabs(centered), fabs(van_leer));
      return copysign(fmin(lim_slope_a, lim_slope_b), centered);
    } else {
      return 0.0;
    }
  };

  // the monotonized difference in the characteristic variables
  Characteristic del_a_m;
  // The monotonized difference in the characteristic variables projected into the primitive variables
  Primitive output;

  // Monotonize the slopes
  del_a_m.a0 = Monotonize(del_a_L.a0, del_a_R.a0, del_a_C.a0, del_a_G.a0);
  del_a_m.a1 = Monotonize(del_a_L.a1, del_a_R.a1, del_a_C.a1, del_a_G.a1);
  del_a_m.a2 = Monotonize(del_a_L.a2, del_a_R.a2, del_a_C.a2, del_a_G.a2);
  del_a_m.a3 = Monotonize(del_a_L.a3, del_a_R.a3, del_a_C.a3, del_a_G.a3);
  del_a_m.a4 = Monotonize(del_a_L.a4, del_a_R.a4, del_a_C.a4, del_a_G.a4);

#ifdef MHD
  del_a_m.a5 = Monotonize(del_a_L.a5, del_a_R.a5, del_a_C.a5, del_a_G.a5);
  del_a_m.a6 = Monotonize(del_a_L.a6, del_a_R.a6, del_a_C.a6, del_a_G.a6);
#endif  // MHD

#ifdef DE
  output.gas_energy = Monotonize(del_L.gas_energy, del_R.gas_energy, del_C.gas_energy, del_G.gas_energy);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    output.scalar[i] = Monotonize(del_L.scalar[i], del_R.scalar[i], del_C.scalar[i], del_G.scalar[i]);
  }
#endif  // SCALAR

  // Project into the primitive variables. Note the return by reference to preserve the values in the gas_energy and
  // scalars
  Characteristic_To_Primitive(primitive, del_a_m, sound_speed, sound_speed_squared, gamma, output);

  return output;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Monotonize the parabolic interface states
 *
 * \param[in] cell_i The state in cell i
 * \param[in] cell_im1 The state in cell i-1
 * \param[in] cell_ip1 The state in cell i+1
 * \param[in,out] interface_L_iph The left interface state at i+1/2
 * \param[in,out] interface_R_imh The right interface state at i-1/2
 * \return Primitive
 */
void __device__ __host__ __inline__ Monotonize_Parabolic_Interface(Primitive const &cell_i, Primitive const &cell_im1,
                                                                   Primitive const &cell_ip1,
                                                                   Primitive &interface_L_iph,
                                                                   Primitive &interface_R_imh)
{
  // The function that will actually do the monotozation. Note the return by refernce of the interface state
  auto Monotonize = [](Real const &state_i, Real const &state_im1, Real const &state_ip1, Real &interface_L,
                       Real &interface_R) {
    // Some terms we need for the comparisons
    Real const term_1 = 6.0 * (interface_L - interface_R) * (state_i - 0.5 * (interface_R + interface_L));
    Real const term_2 = pow(interface_L - interface_R, 2.0);

    // First monotonicity constraint. Equations 47-49 in Stone et al. 2008
    if ((interface_L - state_i) * (state_i - interface_R) <= 0.0) {
      interface_L = state_i;
      interface_R = state_i;
    }
    // Second monotonicity constraint. Equations 50 & 51 in Stone et al. 2008
    else if (term_1 > term_2) {
      interface_R = 3.0 * state_i - 2.0 * interface_L;
    }
    // Third monotonicity constraint. Equations 52 & 53 in Stone et al. 2008
    else if (term_1 < -term_2) {
      interface_L = 3.0 * state_i - 2.0 * interface_R;
    }

    // Bound the interface to lie between adjacent cell centered values
    interface_R = fmax(fmin(state_i, state_im1), interface_R);
    interface_R = fmin(fmax(state_i, state_im1), interface_R);
    interface_L = fmax(fmin(state_i, state_ip1), interface_L);
    interface_L = fmin(fmax(state_i, state_ip1), interface_L);
  };

  // Monotonize each interface state
  Monotonize(cell_i.density, cell_im1.density, cell_ip1.density, interface_L_iph.density, interface_R_imh.density);
  Monotonize(cell_i.velocity_x, cell_im1.velocity_x, cell_ip1.velocity_x, interface_L_iph.velocity_x,
             interface_R_imh.velocity_x);
  Monotonize(cell_i.velocity_y, cell_im1.velocity_y, cell_ip1.velocity_y, interface_L_iph.velocity_y,
             interface_R_imh.velocity_y);
  Monotonize(cell_i.velocity_z, cell_im1.velocity_z, cell_ip1.velocity_z, interface_L_iph.velocity_z,
             interface_R_imh.velocity_z);
  Monotonize(cell_i.pressure, cell_im1.pressure, cell_ip1.pressure, interface_L_iph.pressure, interface_R_imh.pressure);

#ifdef MHD
  Monotonize(cell_i.magnetic_y, cell_im1.magnetic_y, cell_ip1.magnetic_y, interface_L_iph.magnetic_y,
             interface_R_imh.magnetic_y);
  Monotonize(cell_i.magnetic_z, cell_im1.magnetic_z, cell_ip1.magnetic_z, interface_L_iph.magnetic_z,
             interface_R_imh.magnetic_z);
#endif  // MHD

#ifdef DE
  Monotonize(cell_i.gas_energy, cell_im1.gas_energy, cell_ip1.gas_energy, interface_L_iph.gas_energy,
             interface_R_imh.gas_energy);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    Monotonize(cell_i.scalar[i], cell_im1.scalar[i], cell_ip1.scalar[i], interface_L_iph.scalar[i],
               interface_R_imh.scalar[i]);
  }
#endif  // SCALAR
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Compute the interface state from the slope and cell centered state using linear interpolation
 *
 * \param[in] primitive The cell centered state
 * \param[in] slopes The slopes
 * \param[in] sign Whether to add or subtract the slope. +1 to add it and -1 to subtract it
 * \return Primitive The interface state
 */
Primitive __device__ __host__ __inline__ Calc_Interface_Linear(Primitive const &primitive, Primitive const &slopes,
                                                               Real const &sign)
{
  Primitive output;

  auto interface = [&sign](Real const &state, Real const &slope) -> Real { return state + sign * 0.5 * slope; };

  output.density    = interface(primitive.density, slopes.density);
  output.velocity_x = interface(primitive.velocity_x, slopes.velocity_x);
  output.velocity_y = interface(primitive.velocity_y, slopes.velocity_y);
  output.velocity_z = interface(primitive.velocity_z, slopes.velocity_z);
  output.pressure   = interface(primitive.pressure, slopes.pressure);

#ifdef MHD
  output.magnetic_y = interface(primitive.magnetic_y, slopes.magnetic_y);
  output.magnetic_z = interface(primitive.magnetic_z, slopes.magnetic_z);
#endif  // MHD

#ifdef DE
  output.gas_energy = interface(primitive.gas_energy, slopes.gas_energy);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    output.scalar[i] = interface(primitive.scalar[i], slopes.scalar[i]);
  }
#endif  // SCALAR

  return output;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Compute the interface state for the CTU version fo the reconstructor from the slope and cell centered state
 * using parabolic interpolation
 *
 * \param[in] cell_i The state in cell i
 * \param[in] cell_im1 The state in cell i-1
 * \param[in] slopes_i The slopes in cell i
 * \param[in] slopes_im1 The slopes in cell i-1
 * \return Primitive The interface state
 */
Primitive __device__ __host__ __inline__ Calc_Interface_Parabolic(Primitive const &cell_i, Primitive const &cell_im1,
                                                                  Primitive const &slopes_i,
                                                                  Primitive const &slopes_im1)
{
  Primitive output;

  auto interface = [](Real const &state_i, Real const &state_im1, Real const &slope_i, Real const &slope_im1) -> Real {
    return 0.5 * (state_i + state_im1) - (slope_i - slope_im1) / 6.0;
  };

  output.density    = interface(cell_i.density, cell_im1.density, slopes_i.density, slopes_im1.density);
  output.velocity_x = interface(cell_i.velocity_x, cell_im1.velocity_x, slopes_i.velocity_x, slopes_im1.velocity_x);
  output.velocity_y = interface(cell_i.velocity_y, cell_im1.velocity_y, slopes_i.velocity_y, slopes_im1.velocity_y);
  output.velocity_z = interface(cell_i.velocity_z, cell_im1.velocity_z, slopes_i.velocity_z, slopes_im1.velocity_z);
  output.pressure   = interface(cell_i.pressure, cell_im1.pressure, slopes_i.pressure, slopes_im1.pressure);

#ifdef MHD
  output.magnetic_y = interface(cell_i.magnetic_y, cell_im1.magnetic_y, slopes_i.magnetic_y, slopes_im1.magnetic_y);
  output.magnetic_z = interface(cell_i.magnetic_z, cell_im1.magnetic_z, slopes_i.magnetic_z, slopes_im1.magnetic_z);
#endif  // MHD

#ifdef DE
  output.gas_energy = interface(cell_i.gas_energy, cell_im1.gas_energy, slopes_i.gas_energy, slopes_im1.gas_energy);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    output.scalar[i] = interface(cell_i.scalar[i], cell_im1.scalar[i], slopes_i.scalar[i], slopes_im1.scalar[i]);
  }
#endif  // SCALAR

  return output;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Compute the PPM interface state for a given field/stencil.
 *
 * \details This method is heavily based on the implementation in Athena++. See the following papers for details
 *  - K. Felker & J. Stone, "A fourth-order accurate finite volume method for ideal MHD via upwind constrained
 * transport", JCP, 375, (2018)
 *  - P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations", JCP, 54, 174
 * (1984)
 *  - P. Colella & M. Sekora, "A limiter for PPM that preserves accuracy at smooth extrema", JCP, 227, 7069 (2008)
 *  - P. McCorquodale & P. Colella,  "A high-order finite-volume method for conservation laws on locally refined grids",
 * CAMCoS, 6, 1 (2011)
 *  - P. Colella, M.R. Dorr, J. Hittinger, D. Martin, "High-order, finite-volume methods in mapped coordinates", JCP,
 * 230, 2952 (2011)
 *
 * \param[in] cell_im2 The value of the field/stencil at i-2
 * \param[in] cell_im1 The value of the field/stencil at i-1
 * \param[in] cell_i The value of the field/stencil at i
 * \param[in] cell_ip1 The value of the field/stencil at i+1
 * \param[in] cell_ip2 The value of the field/stencil at i+2
 * \param[out] interface_L_iph The left interface at the i+1/2 face
 * \param[out] interface_R_imh The right interface at the i-1/2 face
 */
void __device__ __host__ __inline__ PPM_Single_Variable(Real const &cell_im2, Real const &cell_im1, Real const &cell_i,
                                                        Real const &cell_ip1, Real const &cell_ip2,
                                                        Real &interface_L_iph, Real &interface_R_imh)
{
  // This method is heavily based on the implementation in Athena++. See the following papers for details
  // - K. Felker & J. Stone, "A fourth-order accurate finite volume method for ideal MHD via upwind constrained
  // transport", JCP, 375, (2018)
  // - P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations", JCP, 54, 174
  // (1984)
  // - P. Colella & M. Sekora, "A limiter for PPM that preserves accuracy at smooth extrema", JCP, 227, 7069 (2008)
  // - P. McCorquodale & P. Colella,  "A high-order finite-volume method for conservation laws on locally refined
  // grids", CAMCoS, 6, 1 (2011)
  // - P. Colella, M.R. Dorr, J. Hittinger, D. Martin, "High-order, finite-volume methods in mapped coordinates", JCP,
  // 230, 2952 (2011)

  // Let's start by setting up some things that we'll need later

  // Colella & Sekora 2008 constant used in second derivative limiter
  Real const C2 = 1.25;

  // This lambda function is used for limiting the interfaces
  auto limit_interface = [&C2](Real const &cell_i, Real const &cell_im1, Real const &interface, Real const &slope_2nd_i,
                               Real const &slope_2nd_im1) -> Real {
    // Colella et al. 2011 eq. 85b.
    // 85a is slope_2nd_im1 and 85c is slope_2nd_i
    Real slope_2nd_centered = 3.0 * (cell_im1 + cell_i - 2.0 * interface);

    Real limited_slope = 0.0;
    if (SIGN(slope_2nd_centered) == SIGN(slope_2nd_im1) and SIGN(slope_2nd_centered) == SIGN(slope_2nd_i)) {
      limited_slope =
          SIGN(slope_2nd_centered) * min(C2 * abs(slope_2nd_im1), min(C2 * abs(slope_2nd_i), abs(slope_2nd_centered)));
    }

    // Collela et al. 2011 eq. 84a & 84b
    Real const diff_left  = interface - cell_im1;
    Real const diff_right = cell_i - interface;
    if (diff_left * diff_right < 0.0) {
      // Local extrema detected at the interface
      return 0.5 * (cell_im1 + cell_i) - limited_slope / 6.0;
    } else {
      return interface;
    }
  };

  // Now that the setup is done we can start computing the interface states

  // Compute average slopes
  Real const slope_left    = (cell_i - cell_im1);
  Real const slope_right   = (cell_ip1 - cell_i);
  Real const slope_avg_im1 = 0.5 * slope_left + 0.5 * (cell_im1 - cell_im2);
  Real const slope_avg_i   = 0.5 * slope_right + 0.5 * slope_left;
  Real const slope_avg_ip1 = 0.5 * (cell_ip2 - cell_ip1) + 0.5 * slope_right;

  // Approximate interface average at i-1/2 and i+1/2 using PPM
  // P. Colella & P. Woodward 1984 eq. 1.6
  interface_R_imh = 0.5 * (cell_im1 + cell_i) + (slope_avg_im1 - slope_avg_i) / 6.0;
  interface_L_iph = 0.5 * (cell_i + cell_ip1) + (slope_avg_i - slope_avg_ip1) / 6.0;

  // Limit interpolated interface states (Colella et al. 2011 section 4.3.1)

  // Approximate second derivative at interfaces for smooth extrema preservation
  // Colella et al. 2011 eq 85a
  Real const slope_2nd_im1 = cell_im2 + cell_i - 2.0 * cell_im1;
  Real const slope_2nd_i   = cell_im1 + cell_ip1 - 2.0 * cell_i;
  Real const slope_2nd_ip1 = cell_i + cell_ip2 - 2.0 * cell_ip1;

  interface_R_imh = limit_interface(cell_i, cell_im1, interface_R_imh, slope_2nd_i, slope_2nd_im1);
  interface_L_iph = limit_interface(cell_ip1, cell_i, interface_L_iph, slope_2nd_ip1, slope_2nd_i);

  // Compute cell-centered difference stencils (McCorquodale & Colella 2011 section 2.4.1)

  // Apply Colella & Sekora limiters to parabolic interpolant
  Real slope_2nd_face = 6.0 * (interface_R_imh + interface_L_iph - 2.0 * cell_i);

  Real slope_2nd_limited = 0.0;
  if (SIGN(slope_2nd_im1) == SIGN(slope_2nd_i) and SIGN(slope_2nd_im1) == SIGN(slope_2nd_ip1) and
      SIGN(slope_2nd_im1) == SIGN(slope_2nd_face)) {
    // Extrema is smooth
    // Colella & Sekora eq. 22
    slope_2nd_limited = SIGN(slope_2nd_face) * min(min(C2 * abs(slope_2nd_im1), C2 * abs(slope_2nd_i)),
                                                   min(C2 * abs(slope_2nd_ip1), abs(slope_2nd_face)));
  }

  // Check if 2nd derivative is close to roundoff error
  Real cell_max = max(abs(cell_im2), abs(cell_im1));
  cell_max      = max(cell_max, abs(cell_i));
  cell_max      = max(cell_max, abs(cell_ip1));
  cell_max      = max(cell_max, abs(cell_ip2));

  // If this condition is true then the limiter is not sensitive to roundoff and we use the limited ratio
  // McCorquodale & Colella 2011 eq. 27
  Real const rho = (abs(slope_2nd_face) > (1.0e-12) * cell_max) ? slope_2nd_limited / slope_2nd_face : 0.0;

  // Colella & Sekora eq. 25
  Real slope_face_left  = cell_i - interface_R_imh;
  Real slope_face_right = interface_L_iph - cell_i;

  // Check for local extrema
  if ((slope_face_left * slope_face_right) <= 0.0 or ((cell_ip1 - cell_i) * (cell_i - cell_im1)) <= 0.0) {
    // Extrema detected
    // Check if relative change in limited 2nd deriv is > roundoff
    if (rho <= (1.0 - (1.0e-12))) {
      // Limit smooth extrema
      // Colella & Sekora eq. 23
      interface_R_imh = cell_i - rho * slope_face_left;
      interface_L_iph = cell_i + rho * slope_face_right;
    }
  } else {
    // No extrema detected
    // Overshoot i-1/2,R / i,(-) state
    if (abs(slope_face_left) >= 2.0 * abs(slope_face_right)) {
      interface_R_imh = cell_i - 2.0 * slope_face_right;
    }
    // Overshoot i+1/2,L / i,(+) state
    if (abs(slope_face_right) >= 2.0 * abs(slope_face_left)) {
      interface_L_iph = cell_i + 2.0 * slope_face_left;
    }
  }
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Write the interface data to the appropriate arrays
 *
 * \param[in] interface_state The interface state to write
 * \param[out] dev_interface The interface array
 * \param[in] dev_conserved The conserved variables
 * \param[in] id The cell id to write to
 * \param[in] n_cells The total number of cells
 * \param[in] o1 Directional parameter
 * \param[in] o2 Directional parameter
 * \param[in] o3 Directional parameter
 * \param[in] gamma The adiabatic index
 */
void __device__ __host__ __inline__ Write_Data(Primitive const &interface_state, Real *dev_interface,
                                               Real const *dev_conserved, size_t const &id, size_t const &n_cells,
                                               size_t const &o1, size_t const &o2, size_t const &o3, Real const &gamma)
{
  // Write out density and momentum
  dev_interface[grid_enum::density * n_cells + id] = interface_state.density;
  dev_interface[o1 * n_cells + id]                 = interface_state.density * interface_state.velocity_x;
  dev_interface[o2 * n_cells + id]                 = interface_state.density * interface_state.velocity_y;
  dev_interface[o3 * n_cells + id]                 = interface_state.density * interface_state.velocity_z;

#ifdef MHD
  // Write the Y and Z interface states and load the X magnetic face needed to compute the energy
  Real magnetic_x;
  switch (o1) {
    case grid_enum::momentum_x:
      dev_interface[grid_enum::Q_x_magnetic_y * n_cells + id] = interface_state.magnetic_y;
      dev_interface[grid_enum::Q_x_magnetic_z * n_cells + id] = interface_state.magnetic_z;
      magnetic_x                                              = dev_conserved[grid_enum::magnetic_x * n_cells + id];
      break;
    case grid_enum::momentum_y:
      dev_interface[grid_enum::Q_y_magnetic_z * n_cells + id] = interface_state.magnetic_y;
      dev_interface[grid_enum::Q_y_magnetic_x * n_cells + id] = interface_state.magnetic_z;
      magnetic_x                                              = dev_conserved[grid_enum::magnetic_y * n_cells + id];
      break;
    case grid_enum::momentum_z:
      dev_interface[grid_enum::Q_z_magnetic_x * n_cells + id] = interface_state.magnetic_y;
      dev_interface[grid_enum::Q_z_magnetic_y * n_cells + id] = interface_state.magnetic_z;
      magnetic_x                                              = dev_conserved[grid_enum::magnetic_z * n_cells + id];
      break;
  }

  // Compute the MHD energy
  dev_interface[grid_enum::Energy * n_cells + id] = hydro_utilities::Calc_Energy_Primitive(
      interface_state.pressure, interface_state.density, interface_state.velocity_x, interface_state.velocity_y,
      interface_state.velocity_z, gamma, magnetic_x, interface_state.magnetic_y, interface_state.magnetic_z);
#else   // not MHD
  // Compute the hydro energy
  dev_interface[grid_enum::Energy * n_cells + id] = hydro_utilities::Calc_Energy_Primitive(
      interface_state.pressure, interface_state.density, interface_state.velocity_x, interface_state.velocity_y,
      interface_state.velocity_z, gamma);
#endif  // MHD

#ifdef DE
  dev_interface[grid_enum::GasEnergy * n_cells + id] = interface_state.density * interface_state.gas_energy;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_interface[(grid_enum::scalar + i) * n_cells + id] = interface_state.density * interface_state.scalar[i];
  }
#endif  // SCALAR
}
// =====================================================================================================================
}  // namespace reconstruction

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
  Real const beta_y     = (isfinite(beta_denom)) ? primitive.magnetic_y * beta_denom : 0.0;
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
  Real const beta_y     = (isfinite(beta_denom)) ? primitive.magnetic_y * beta_denom : 0.0;
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
 * \brief Compute the interface state from the slope and cell centered state.
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

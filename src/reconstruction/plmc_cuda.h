/*! \file plmc_cuda.h
 *  \brief Declarations of the cuda plm kernels, characteristic reconstruction
 * version. */

#ifndef PLMC_CUDA_H
#define PLMC_CUDA_H

#include "../global/global.h"
#include "../grid/grid_enum.h"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"

/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
 *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real
 gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using plm. */
__global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                          Real dt, Real gamma, int dir, int n_fields);

namespace plmc_utils
{
/*!
 * \brief A struct for the primitive variables
 *
 */
struct PlmcPrimitive {
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

/*!
 * \brief A struct for the characteristic variables
 *
 */
struct PlmcCharacteristic {
  // Hydro variables
  Real a0, a1, a2, a3, a4;

#ifdef MHD
  Real a5, a6;
#endif  // MHD
};

/*!
 * \brief Load the data for PLMC reconstruction
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
 * \return PlmcPrimitive The loaded cell data
 */
PlmcPrimitive __device__ __host__ Load_Data(Real const *dev_conserved, size_t const &xid, size_t const &yid,
                                            size_t const &zid, size_t const &nx, size_t const &ny,
                                            size_t const &n_cells, size_t const &o1, size_t const &o2, size_t const &o3,
                                            Real const &gamma);

/*!
 * \brief Compute a simple slope. Equation is `coef * (left - right)`.
 *
 * \param[in] left The data on the positive side of the slope
 * \param[in] right The data on the negative side of the slope
 * \param[in] coef The coefficient to multiply the slope by. Defaults to zero
 * \return PlmcPrimitive The slopes
 */
PlmcPrimitive __device__ __host__ Compute_Slope(PlmcPrimitive const &left, PlmcPrimitive const &right,
                                                Real const &coef = 1.0);

/*!
 * \brief Compute the Van Lear slope from the left and right slopes
 *
 * \param[in] left_slope The left slope
 * \param[in] right_slope The right slope
 * \return PlmcPrimitive The Van Leer slope
 */
PlmcPrimitive __device__ __host__ Van_Leer_Slope(PlmcPrimitive const &left_slope, PlmcPrimitive const &right_slope);

/*!
 * \brief Project from the primitive variables slopes to the characteristic variables slopes. Stone Eqn 37. Use the
 * eigenvectors given in Stone 2008, Appendix A
 *
 * \param[in] primitive The primitive variables
 * \param[in] primitive_slope The primitive variables slopes
 * \param[in] sound_speed The speed of sound
 * \param[in] sound_speed_squared The speed of sound squared
 * \param[in] gamma The adiabatic index
 * \return PlmcCharacteristic
 */
PlmcCharacteristic __device__ __inline__ Primitive_To_Characteristic(PlmcPrimitive const &primitive,
                                                                     PlmcPrimitive const &primitive_slope,
                                                                     Real const &sound_speed,
                                                                     Real const &sound_speed_squared, Real const &gamma)
{
  PlmcCharacteristic output;

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

  // Compute Betas (equation A17)
  Real const beta_denom = rhypot(primitive.magnetic_y, primitive.magnetic_z);
  Real const beta_y     = (beta_denom == 0) ? 0.0 : primitive.magnetic_y * beta_denom;
  Real const beta_z     = (beta_denom == 0) ? 0.0 : primitive.magnetic_z * beta_denom;

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
void __device__ __inline__ Characteristic_To_Primitive(PlmcPrimitive const &primitive,
                                                       PlmcCharacteristic const &characteristic_slope,
                                                       Real const &sound_speed, Real const &sound_speed_squared,
                                                       Real const &gamma, PlmcPrimitive &output)
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

  // Compute Betas (equation A17)
  Real const beta_denom = rhypot(primitive.magnetic_y, primitive.magnetic_z);
  Real const beta_y     = (beta_denom == 0) ? 0.0 : primitive.magnetic_y * beta_denom;
  Real const beta_z     = (beta_denom == 0) ? 0.0 : primitive.magnetic_z * beta_denom;

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
 * \return PlmcPrimitive The Monotonized primitive slopes
 */
PlmcPrimitive __device__ Monotonize_Characteristic_Return_Primitive(
    PlmcPrimitive const &primitive, PlmcPrimitive const &del_L, PlmcPrimitive const &del_R, PlmcPrimitive const &del_C,
    PlmcPrimitive const &del_G, PlmcCharacteristic const &del_a_L, PlmcCharacteristic const &del_a_R,
    PlmcCharacteristic const &del_a_C, PlmcCharacteristic const &del_a_G, Real const &sound_speed,
    Real const &sound_speed_squared, Real const &gamma);

/*!
 * \brief Compute the interface state from the slope and cell centered state.
 *
 * \param[in] primitive The cell centered state
 * \param[in] slopes The slopes
 * \param[in] sign Whether to add or subtract the slope. +1 to add it and -1 to subtract it
 * \return plmc_utils::PlmcPrimitive The interface state
 */
PlmcPrimitive __device__ __host__ Calc_Interface(PlmcPrimitive const &primitive, PlmcPrimitive const &slopes,
                                                 Real const &sign);

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
void __device__ __host__ Write_Data(PlmcPrimitive const &interface_state, Real *dev_interface,
                                    Real const *dev_conserved, size_t const &id, size_t const &n_cells,
                                    size_t const &o1, size_t const &o2, size_t const &o3, Real const &gamma);
}  // namespace plmc_utils
#endif  // PLMC_CUDA_H

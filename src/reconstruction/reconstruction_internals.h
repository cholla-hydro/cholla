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
// #include "../reconstruction/pcm_cuda.h"

/*!
 * \brief Namespace to contain various utilities for the interface reconstruction kernels
 *
 */
namespace reconstruction
{
// =====================================================================================================================
/*!
 * \brief This enum is used to select which reconstructor to use. The idea is that either one of its implicitly defined
 * members (i.e. not `chosen`) can be used to tell a kernel which reconstruction to perform and the member `chosen` can
 * be used to indicate which reconstruction method was chosen at compile time. I.e. in a Van Leer integrator the `pcm`
 * member would be passed to the first riemann solve to tell it to use PCM reconstruction and `chosen` would be passed
 * to the second riemann solve to indicate which higher order reconstruction it should use.
 *
 */
enum Kind {
  pcm,
  plmp,
  plmc,
  ppmp,
  ppmc,

#if defined(PCM)
  chosen = pcm
#elif defined(PLMP)
  chosen = plmp
#elif defined(PLMC)
  chosen = plmc
#elif defined(PPMP)
  chosen = ppmp
#elif defined(PPMC)
  chosen = ppmc
#else
  #error "no reconstruction selected"
#endif
};
// =====================================================================================================================

// =====================================================================================================================
struct EigenVecs {
  Real sound_speed;
#ifdef MHD
  Real magnetosonic_speed_fast, magnetosonic_speed_slow, magnetosonic_speed_fast_squared,
      magnetosonic_speed_slow_squared;
  Real alpha_fast, alpha_slow;
  Real beta_y, beta_z;
  Real n_fs, sign;
  /// The non-primed values are used in the conversion from characteristic to primitive variables
  Real q_fast, q_slow;
  Real a_fast, a_slow;
  /// The primed values are used in the conversion from primitive to characteristic variables
  Real q_prime_fast, q_prime_slow;
  Real a_prime_fast, a_prime_slow;
#endif  // MHD
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief A struct for the characteristic variables. We use the same notation as Stone et al. 2008 where the variable
 # `ai` is the ith characteristic variable.
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
 * \brief Determine if a thread is within the allowed range
 *
 * \tparam order The order of the reconstruction. 2 for PLM, 3 for PPM
 * \param nx The number of cells in the X-direction
 * \param ny The number of cells in the Y-direction
 * \param nz The number of cells in the Z-direction
 * \param xid The X thread index
 * \param yid The Y thread index
 * \param zid The Z thread index
 * \return true The thread is NOT in the allowed range
 * \return false The thread is in the allowed range
 */
template <int order>
bool __device__ __host__ __inline__ Thread_Guard(int const &nx, int const &ny, int const &nz, int const &xid,
                                                 int const &yid, int const &zid)
{
  // These checks all make sure that the xid is such that the thread won't try to load any memory that is out of bounds

  // X check
  bool out_of_bounds_thread = xid < order - 1 or xid >= nx - order;

  // Y check, only used for 2D and 3D
  if (ny > 1) {
    out_of_bounds_thread = yid < order - 1 or yid >= ny - order or out_of_bounds_thread;
  }

  // z check, only used for 3D
  if (nz > 1) {
    out_of_bounds_thread = zid < order - 1 or zid >= nz - order or out_of_bounds_thread;
  }
  // This is needed in the case that nz == 1 to avoid overrun
  else {
    out_of_bounds_thread = zid >= nz or out_of_bounds_thread;
  }

  return out_of_bounds_thread;
}
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
 * \return hydro_utilities::Primitive The loaded cell data
 */
hydro_utilities::Primitive __device__ __host__ __inline__ Load_Data(
    Real const *dev_conserved, size_t const &xid, size_t const &yid, size_t const &zid, size_t const &nx,
    size_t const &ny, size_t const &n_cells, size_t const &o1, size_t const &o2, size_t const &o3, Real const &gamma)
{  // Compute index
  size_t const id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);

  // Declare the variable we will return
  hydro_utilities::Primitive loaded_data;

  // Load hydro variables except pressure
  loaded_data.density      = dev_conserved[grid_enum::density * n_cells + id];
  loaded_data.velocity.x() = dev_conserved[o1 * n_cells + id] / loaded_data.density;
  loaded_data.velocity.y() = dev_conserved[o2 * n_cells + id] / loaded_data.density;
  loaded_data.velocity.z() = dev_conserved[o3 * n_cells + id] / loaded_data.density;

  // Load MHD variables. Note that I only need the centered values for the transverse fields except for the initial
  // computation of the primitive variables
#ifdef MHD
  auto magnetic_centered = mhd::utils::cellCenteredMagneticFields(dev_conserved, id, xid, yid, zid, n_cells, nx, ny);
  switch (o1) {
    case grid_enum::momentum_x:
      loaded_data.magnetic.x() = magnetic_centered.x();
      loaded_data.magnetic.y() = magnetic_centered.y();
      loaded_data.magnetic.z() = magnetic_centered.z();
      break;
    case grid_enum::momentum_y:
      loaded_data.magnetic.x() = magnetic_centered.y();
      loaded_data.magnetic.y() = magnetic_centered.z();
      loaded_data.magnetic.z() = magnetic_centered.x();
      break;
    case grid_enum::momentum_z:
      loaded_data.magnetic.x() = magnetic_centered.z();
      loaded_data.magnetic.y() = magnetic_centered.x();
      loaded_data.magnetic.z() = magnetic_centered.y();
      break;
  }
#endif  // MHD

// Load pressure accounting for dual energy if enabled
#ifdef DE  // DE
  Real const energy     = dev_conserved[grid_enum::Energy * n_cells + id];
  Real const gas_energy = dev_conserved[grid_enum::GasEnergy * n_cells + id];

  Real energy_non_thermal = hydro_utilities::Calc_Kinetic_Energy_From_Velocity(
      loaded_data.density, loaded_data.velocity.x(), loaded_data.velocity.y(), loaded_data.velocity.z());

  #ifdef MHD
  energy_non_thermal +=
      mhd::utils::computeMagneticEnergy(magnetic_centered.x(), magnetic_centered.y(), magnetic_centered.z());
  #endif  // MHD

  loaded_data.pressure = hydro_utilities::Get_Pressure_From_DE(energy, energy - energy_non_thermal, gas_energy, gamma);
  loaded_data.gas_energy_specific = gas_energy / loaded_data.density;
#else  // not DE
  #ifdef MHD
  loaded_data.pressure = hydro_utilities::Calc_Pressure_Primitive(
      dev_conserved[grid_enum::Energy * n_cells + id], loaded_data.density, loaded_data.velocity.x(),
      loaded_data.velocity.y(), loaded_data.velocity.z(), gamma, loaded_data.magnetic.x(), loaded_data.magnetic.y(),
      loaded_data.magnetic.z());
  #else   // not MHD
  loaded_data.pressure = hydro_utilities::Calc_Pressure_Primitive(
      dev_conserved[grid_enum::Energy * n_cells + id], loaded_data.density, loaded_data.velocity.x(),
      loaded_data.velocity.y(), loaded_data.velocity.z(), gamma);
  #endif  // MHD
#endif    // DE

#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    loaded_data.scalar_specific[i] = dev_conserved[(grid_enum::scalar + i) * n_cells + id] / loaded_data.density;
  }
#endif  // SCALAR

  return loaded_data;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Determine if a thread is within the allowed range
 *
 * \tparam reconstruction A member of reconstruction::Kind used to determine the order of reconstruction
 * \param nx The number of cells in the X-direction
 * \param ny The number of cells in the Y-direction
 * \param nz The number of cells in the Z-direction
 * \param xid The X thread index
 * \param yid The Y thread index
 * \param zid The Z thread index
 * \return true The thread is NOT in the allowed range
 * \return false The thread is in the allowed range
 */
template <int reconstruction>
bool __device__ __host__ __inline__ Riemann_Thread_Guard(size_t const nx, size_t const ny, size_t const nz,
                                                         size_t const xid, size_t const yid, size_t const zid)
{
  int order;
  if constexpr (reconstruction == reconstruction::Kind::pcm) {
    order = 1;
  } else if constexpr (reconstruction == reconstruction::Kind::plmc or reconstruction == reconstruction::Kind::plmp) {
    order = 3;
  } else if constexpr (reconstruction == reconstruction::Kind::ppmc or reconstruction == reconstruction::Kind::ppmp) {
    order = 4;
  }

  bool out_of_bounds_thread = false;
  // X check
  if (nx > 1) {
    out_of_bounds_thread = xid < order - 1 or xid >= nx - order or out_of_bounds_thread;
  }
  // Y check, only used for 2D and 3D
  if (ny > 1) {
    out_of_bounds_thread = yid < order - 1 or yid >= ny - order or out_of_bounds_thread;
  }

  // z check, only used for 3D
  if (nz > 1) {
    out_of_bounds_thread = zid < order - 1 or zid >= nz - order or out_of_bounds_thread;
  }
  // This is needed in the case that nz == 1 to avoid overrun
  else {
    out_of_bounds_thread = zid >= nz or out_of_bounds_thread;
  }

  return out_of_bounds_thread;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Compute a simple slope. Equation is `coef * (right - left)`.
 *
 * \param[in] left The data with the lower index (on the "left" side)
 * \param[in] right The data with the higher index (on the "right" side)
 * \param[in] coef The coefficient to multiply the slope by. Defaults to 1.0
 * \return hydro_utilities::Primitive The slopes
 */
hydro_utilities::Primitive __device__ __host__ __inline__ Compute_Slope(hydro_utilities::Primitive const &left,
                                                                        hydro_utilities::Primitive const &right,
                                                                        Real const &coef = 1.0)
{
  hydro_utilities::Primitive slopes;

  slopes.density      = coef * (right.density - left.density);
  slopes.velocity.x() = coef * (right.velocity.x() - left.velocity.x());
  slopes.velocity.y() = coef * (right.velocity.y() - left.velocity.y());
  slopes.velocity.z() = coef * (right.velocity.z() - left.velocity.z());
  slopes.pressure     = coef * (right.pressure - left.pressure);

#ifdef MHD
  slopes.magnetic.y() = coef * (right.magnetic.y() - left.magnetic.y());
  slopes.magnetic.z() = coef * (right.magnetic.z() - left.magnetic.z());
#endif  // MHD

#ifdef DE
  slopes.gas_energy_specific = coef * (right.gas_energy_specific - left.gas_energy_specific);
#endif  // DE

#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    slopes.scalar_specific[i] = coef * (right.scalar_specific[i] - left.scalar_specific[i]);
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
 * \return hydro_utilities::Primitive The Van Leer slope
 */
hydro_utilities::Primitive __device__ __host__ __inline__ Compute_Van_Leer_Slope(
    hydro_utilities::Primitive const &left_slope, hydro_utilities::Primitive const &right_slope)
{
  hydro_utilities::Primitive vl_slopes;

  auto Calc_Vl_Slope = [](Real const &left, Real const &right) -> Real {
    if (left * right > 0.0) {
      return 2.0 * left * right / (left + right);
    } else {
      return 0.0;
    }
  };

  vl_slopes.density      = Calc_Vl_Slope(left_slope.density, right_slope.density);
  vl_slopes.velocity.x() = Calc_Vl_Slope(left_slope.velocity.x(), right_slope.velocity.x());
  vl_slopes.velocity.y() = Calc_Vl_Slope(left_slope.velocity.y(), right_slope.velocity.y());
  vl_slopes.velocity.z() = Calc_Vl_Slope(left_slope.velocity.z(), right_slope.velocity.z());
  vl_slopes.pressure     = Calc_Vl_Slope(left_slope.pressure, right_slope.pressure);

#ifdef MHD
  vl_slopes.magnetic.y() = Calc_Vl_Slope(left_slope.magnetic.y(), right_slope.magnetic.y());
  vl_slopes.magnetic.z() = Calc_Vl_Slope(left_slope.magnetic.z(), right_slope.magnetic.z());
#endif  // MHD

#ifdef DE
  vl_slopes.gas_energy_specific = Calc_Vl_Slope(left_slope.gas_energy_specific, right_slope.gas_energy_specific);
#endif  // DE

#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    vl_slopes.scalar_specific[i] = Calc_Vl_Slope(left_slope.scalar_specific[i], right_slope.scalar_specific[i]);
  }
#endif  // SCALAR

  return vl_slopes;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Compute the eigenvectors in the given cell
 *
 * \param[in] primitive The primitive variables in a particular cell
 * \param[in] gamma The adiabatic index
 * \return EigenVecs
 */
EigenVecs __device__ __inline__ Compute_Eigenvectors(hydro_utilities::Primitive const &primitive, Real const &gamma)
{
  EigenVecs output;

  output.sound_speed = hydro_utilities::Calc_Sound_Speed(primitive.pressure, primitive.density, gamma);

#ifdef MHD
  // This is taken from Stone et al. 2008, appendix A. Equation numbers will be quoted as relevant

  // Compute wave speeds and their squares
  output.magnetosonic_speed_fast =
      mhd::utils::fastMagnetosonicSpeed(primitive.density, primitive.pressure, primitive.magnetic.x(),
                                        primitive.magnetic.y(), primitive.magnetic.z(), gamma);
  output.magnetosonic_speed_slow =
      mhd::utils::slowMagnetosonicSpeed(primitive.density, primitive.pressure, primitive.magnetic.x(),
                                        primitive.magnetic.y(), primitive.magnetic.z(), gamma);

  output.magnetosonic_speed_fast_squared = output.magnetosonic_speed_fast * output.magnetosonic_speed_fast;
  output.magnetosonic_speed_slow_squared = output.magnetosonic_speed_slow * output.magnetosonic_speed_slow;

  Real const sound_speed_squared = output.sound_speed * output.sound_speed;

  // Compute Alphas (equation A16)
  if (Real const denom = (output.magnetosonic_speed_fast_squared - output.magnetosonic_speed_slow_squared),
      numerator_2      = (output.magnetosonic_speed_fast_squared - sound_speed_squared);
      denom <= 0.0 or numerator_2 <= 0.0) {
    output.alpha_fast = 1.0;
    output.alpha_slow = 0.0;
  } else if (Real const numerator_1 = (sound_speed_squared - output.magnetosonic_speed_slow_squared);
             numerator_1 <= 0.0) {
    output.alpha_fast = 0.0;
    output.alpha_slow = 1.0;
  } else {
    output.alpha_fast = sqrt(numerator_1 / denom);
    output.alpha_slow = sqrt(numerator_2 / denom);
  }

  // Compute Betas (equation A17). Note that rhypot can return an inf if By and Bz are both zero, the isfinite check
  // handles that case
  Real const beta_denom = rhypot(primitive.magnetic.y(), primitive.magnetic.z());
  output.beta_y         = (isfinite(beta_denom)) ? primitive.magnetic.y() * beta_denom : 1.0;
  output.beta_z         = (isfinite(beta_denom)) ? primitive.magnetic.z() * beta_denom : 0.0;

  // Compute Q(s) (equation A14)
  output.sign         = copysign(1.0, primitive.magnetic.x());
  output.n_fs         = 0.5 / sound_speed_squared;  // equation A19
  output.q_prime_fast = output.sign * output.n_fs * output.alpha_fast * output.magnetosonic_speed_fast;
  output.q_prime_slow = output.sign * output.n_fs * output.alpha_slow * output.magnetosonic_speed_slow;
  output.q_fast       = output.sign * output.alpha_fast * output.magnetosonic_speed_fast;
  output.q_slow       = output.sign * output.alpha_slow * output.magnetosonic_speed_slow;

  // Compute A(s) (equation A15)
  output.a_fast       = output.alpha_fast * output.sound_speed * sqrt(primitive.density);
  output.a_slow       = output.alpha_slow * output.sound_speed * sqrt(primitive.density);
  output.a_prime_fast = 0.5 * output.alpha_fast / (output.sound_speed * sqrt(primitive.density));
  output.a_prime_slow = 0.5 * output.alpha_slow / (output.sound_speed * sqrt(primitive.density));
#endif  // MHD

  return output;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Project from the primitive variables slopes to the characteristic variables slopes. Stone Eqn 37. Use the
 * eigenvectors given in Stone 2008, Appendix A
 *
 * \param[in] primitive The primitive variables
 * \param[in] primitive_slope The primitive variables slopes
 * \param[in] EigenVecs The eigenvectors
 * \param[in] gamma The adiabatic index
 * \return Characteristic
 */
Characteristic __device__ __inline__ Primitive_To_Characteristic(hydro_utilities::Primitive const &primitive,
                                                                 hydro_utilities::Primitive const &primitive_slope,
                                                                 EigenVecs const &eigen, Real const &gamma)
{
  Characteristic output;

#ifdef MHD
  // Multiply the slopes by the left eigenvector matrix given in equation 18
  Real const inverse_sqrt_density = rsqrt(primitive.density);
  output.a0 =
      eigen.n_fs * eigen.alpha_fast *
          (primitive_slope.pressure / primitive.density -
           eigen.magnetosonic_speed_fast * primitive_slope.velocity.x()) +
      eigen.q_prime_slow * (eigen.beta_y * primitive_slope.velocity.y() + eigen.beta_z * primitive_slope.velocity.z()) +
      eigen.a_prime_slow * (eigen.beta_y * primitive_slope.magnetic.y() + eigen.beta_z * primitive_slope.magnetic.z());

  output.a1 =
      0.5 * (eigen.beta_y *
                 (primitive_slope.magnetic.z() * eigen.sign * inverse_sqrt_density + primitive_slope.velocity.z()) -
             eigen.beta_z *
                 (primitive_slope.magnetic.y() * eigen.sign * inverse_sqrt_density + primitive_slope.velocity.y()));

  output.a2 =
      eigen.n_fs * eigen.alpha_slow *
          (primitive_slope.pressure / primitive.density -
           eigen.magnetosonic_speed_slow * primitive_slope.velocity.x()) -
      eigen.q_prime_fast * (eigen.beta_y * primitive_slope.velocity.y() + eigen.beta_z * primitive_slope.velocity.z()) -
      eigen.a_prime_fast * (eigen.beta_y * primitive_slope.magnetic.y() + eigen.beta_z * primitive_slope.magnetic.z());

  output.a3 = primitive_slope.density - primitive_slope.pressure / (eigen.sound_speed * eigen.sound_speed);

  output.a4 =
      eigen.n_fs * eigen.alpha_slow *
          (primitive_slope.pressure / primitive.density +
           eigen.magnetosonic_speed_slow * primitive_slope.velocity.x()) +
      eigen.q_prime_fast * (eigen.beta_y * primitive_slope.velocity.y() + eigen.beta_z * primitive_slope.velocity.z()) -
      eigen.a_prime_fast * (eigen.beta_y * primitive_slope.magnetic.y() + eigen.beta_z * primitive_slope.magnetic.z());
  output.a5 =
      0.5 * (eigen.beta_y *
                 (primitive_slope.magnetic.z() * eigen.sign * inverse_sqrt_density - primitive_slope.velocity.z()) -
             eigen.beta_z *
                 (primitive_slope.magnetic.y() * eigen.sign * inverse_sqrt_density - primitive_slope.velocity.y()));

  output.a6 =
      eigen.n_fs * eigen.alpha_fast *
          (primitive_slope.pressure / primitive.density +
           eigen.magnetosonic_speed_fast * primitive_slope.velocity.x()) -
      eigen.q_prime_slow * (eigen.beta_y * primitive_slope.velocity.y() + eigen.beta_z * primitive_slope.velocity.z()) +
      eigen.a_prime_slow * (eigen.beta_y * primitive_slope.magnetic.y() + eigen.beta_z * primitive_slope.magnetic.z());

#else   // not MHD
  output.a0 = -primitive.density * primitive_slope.velocity.x() / (2.0 * eigen.sound_speed) +
              primitive_slope.pressure / (2.0 * (eigen.sound_speed * eigen.sound_speed));
  output.a1 = primitive_slope.density - primitive_slope.pressure / ((eigen.sound_speed * eigen.sound_speed));
  output.a2 = primitive_slope.velocity.y();
  output.a3 = primitive_slope.velocity.z();
  output.a4 = primitive.density * primitive_slope.velocity.x() / (2.0 * eigen.sound_speed) +
              primitive_slope.pressure / (2.0 * (eigen.sound_speed * eigen.sound_speed));
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
 * \param[in] eigen The eigenvectors
 * \param[in] gamma The adiabatic index
 * \return hydro_utilities::Primitive The state in primitive variables
 */
hydro_utilities::Primitive __device__ __host__ __inline__ Characteristic_To_Primitive(
    hydro_utilities::Primitive const &primitive, Characteristic const &characteristic_slope, EigenVecs const &eigen,
    Real const &gamma)
{
  hydro_utilities::Primitive output;
#ifdef MHD
  // Multiply the slopes by the right eigenvector matrix given in equation 12
  output.density = primitive.density * (eigen.alpha_fast * (characteristic_slope.a0 + characteristic_slope.a6) +
                                        eigen.alpha_slow * (characteristic_slope.a2 + characteristic_slope.a4)) +
                   characteristic_slope.a3;
  output.velocity.x() =
      eigen.magnetosonic_speed_fast * eigen.alpha_fast * (characteristic_slope.a6 - characteristic_slope.a0) +
      eigen.magnetosonic_speed_slow * eigen.alpha_slow * (characteristic_slope.a4 - characteristic_slope.a2);
  output.velocity.y() = eigen.beta_y * (eigen.q_slow * (characteristic_slope.a0 - characteristic_slope.a6) +
                                        eigen.q_fast * (characteristic_slope.a4 - characteristic_slope.a2)) +
                        eigen.beta_z * (characteristic_slope.a5 - characteristic_slope.a1);
  output.velocity.z() = eigen.beta_z * (eigen.q_slow * (characteristic_slope.a0 - characteristic_slope.a6) +
                                        eigen.q_fast * (characteristic_slope.a4 - characteristic_slope.a2)) +
                        eigen.beta_y * (characteristic_slope.a1 - characteristic_slope.a5);
  output.pressure = primitive.density * (eigen.sound_speed * eigen.sound_speed) *
                    (eigen.alpha_fast * (characteristic_slope.a0 + characteristic_slope.a6) +
                     eigen.alpha_slow * (characteristic_slope.a2 + characteristic_slope.a4));
  output.magnetic.y() =
      eigen.beta_y * (eigen.a_slow * (characteristic_slope.a0 + characteristic_slope.a6) -
                      eigen.a_fast * (characteristic_slope.a2 + characteristic_slope.a4)) -
      eigen.beta_z * eigen.sign * sqrt(primitive.density) * (characteristic_slope.a5 + characteristic_slope.a1);
  output.magnetic.z() =
      eigen.beta_z * (eigen.a_slow * (characteristic_slope.a0 + characteristic_slope.a6) -
                      eigen.a_fast * (characteristic_slope.a2 + characteristic_slope.a4)) +
      eigen.beta_y * eigen.sign * sqrt(primitive.density) * (characteristic_slope.a5 + characteristic_slope.a1);

#else   // not MHD
  output.density      = characteristic_slope.a0 + characteristic_slope.a1 + characteristic_slope.a4;
  output.velocity.x() = eigen.sound_speed / primitive.density * (characteristic_slope.a4 - characteristic_slope.a0);
  output.velocity.y() = characteristic_slope.a2;
  output.velocity.z() = characteristic_slope.a3;
  output.pressure     = (eigen.sound_speed * eigen.sound_speed) * (characteristic_slope.a0 + characteristic_slope.a4);
#endif  // MHD

  return output;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Compute the limited slope using the Van Leer limiter
 *
 * \param[in] left The left slope
 * \param[in] right The right slope
 * \param[in] centered The centered slope
 * \param[in] van_leer The Van Leer slope
 * \return Real The limited slope
 */
Real __device__ __host__ __inline__ Van_Leer_Limiter(Real const &left, Real const &right, Real const &centered,
                                                     Real const &van_leer)
{
  if (left * right > 0.0) {
    Real const lim_slope_a = 2.0 * fmin(fabs(left), fabs(right));
    Real const lim_slope_b = fmin(fabs(centered), fabs(van_leer));
    return copysign(fmin(lim_slope_a, lim_slope_b), centered);
  } else {
    return 0.0;
  }
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Limit the charactistic slopes. This is an overload that take reconstruction::Characteristic variables instead
 * of Reals as arguments. Note that it does not limit the gas energy or scalars
 *
 * \param[in] del_a_L The left characteristic slopes
 * \param[in] del_a_R The right characteristic slopes
 * \param[in] del_a_C The centered characteristic slopes
 * \param[in] del_a_G The Van Leer characteristic slopes
 * \return Characteristic The limited characteristic slopes
 */
Characteristic __device__ __host__ __inline__ Van_Leer_Limiter(Characteristic const &del_a_L,
                                                               Characteristic const &del_a_R,
                                                               Characteristic const &del_a_C,
                                                               Characteristic const &del_a_G)
{
  // the monotonized difference in the characteristic variables
  Characteristic del_a_m;

  // Monotonize the slopes
  del_a_m.a0 = Van_Leer_Limiter(del_a_L.a0, del_a_R.a0, del_a_C.a0, del_a_G.a0);
  del_a_m.a1 = Van_Leer_Limiter(del_a_L.a1, del_a_R.a1, del_a_C.a1, del_a_G.a1);
  del_a_m.a2 = Van_Leer_Limiter(del_a_L.a2, del_a_R.a2, del_a_C.a2, del_a_G.a2);
  del_a_m.a3 = Van_Leer_Limiter(del_a_L.a3, del_a_R.a3, del_a_C.a3, del_a_G.a3);
  del_a_m.a4 = Van_Leer_Limiter(del_a_L.a4, del_a_R.a4, del_a_C.a4, del_a_G.a4);

#ifdef MHD
  del_a_m.a5 = Van_Leer_Limiter(del_a_L.a5, del_a_R.a5, del_a_C.a5, del_a_G.a5);
  del_a_m.a6 = Van_Leer_Limiter(del_a_L.a6, del_a_R.a6, del_a_C.a6, del_a_G.a6);
#endif  // MHD

  return del_a_m;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Limit the primitive slopes. This is an overload that take reconstruction::Primitive variables instead
 * of Reals as arguments.
 *
 * \param[in] del_L The left primitive slopes
 * \param[in] del_R The right primitive slopes
 * \param[in] del_C The centered primitive slopes
 * \param[in] del_G The Van Leer primitive slopes
 * \return hydro_utilities::Primitive The limited primitive slopes
 */
hydro_utilities::Primitive __device__ __host__ __inline__ Van_Leer_Limiter(hydro_utilities::Primitive const &del_L,
                                                                           hydro_utilities::Primitive const &del_R,
                                                                           hydro_utilities::Primitive const &del_C,
                                                                           hydro_utilities::Primitive const &del_G)
{
  // the monotonized difference in the primitive variables
  hydro_utilities::Primitive del_m;

  // Monotonize the slopes
  del_m.density      = Van_Leer_Limiter(del_L.density, del_R.density, del_C.density, del_G.density);
  del_m.velocity.x() = Van_Leer_Limiter(del_L.velocity.x(), del_R.velocity.x(), del_C.velocity.x(), del_G.velocity.x());
  del_m.velocity.y() = Van_Leer_Limiter(del_L.velocity.y(), del_R.velocity.y(), del_C.velocity.y(), del_G.velocity.y());
  del_m.velocity.z() = Van_Leer_Limiter(del_L.velocity.z(), del_R.velocity.z(), del_C.velocity.z(), del_G.velocity.z());
  del_m.pressure     = Van_Leer_Limiter(del_L.pressure, del_R.pressure, del_C.pressure, del_G.pressure);

#ifdef MHD
  del_m.magnetic.y() = Van_Leer_Limiter(del_L.magnetic.y(), del_R.magnetic.y(), del_C.magnetic.y(), del_G.magnetic.y());
  del_m.magnetic.z() = Van_Leer_Limiter(del_L.magnetic.z(), del_R.magnetic.z(), del_C.magnetic.z(), del_G.magnetic.z());
#endif  // MHD

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

  return del_m;
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Compute the interface state from the slope and cell centered state using linear interpolation
 *
 * \param[in] primitive The cell centered state
 * \param[in] slopes The slopes
 * \param[in] sign Whether to add or subtract the slope. +1 to add it and -1 to subtract it
 * \return hydro_utilities::Primitive The interface state
 */
hydro_utilities::Primitive __device__ __host__ __inline__ Calc_Interface_Linear(
    hydro_utilities::Primitive const &primitive, hydro_utilities::Primitive const &slopes, Real const &sign)
{
  hydro_utilities::Primitive output;

  auto interface = [&sign](Real const &state, Real const &slope) -> Real { return state + sign * 0.5 * slope; };

  output.density      = interface(primitive.density, slopes.density);
  output.velocity.x() = interface(primitive.velocity.x(), slopes.velocity.x());
  output.velocity.y() = interface(primitive.velocity.y(), slopes.velocity.y());
  output.velocity.z() = interface(primitive.velocity.z(), slopes.velocity.z());
  output.pressure     = interface(primitive.pressure, slopes.pressure);

#ifdef MHD
  output.magnetic.y() = interface(primitive.magnetic.y(), slopes.magnetic.y());
  output.magnetic.z() = interface(primitive.magnetic.z(), slopes.magnetic.z());
#endif  // MHD

#ifdef DE
  output.gas_energy_specific = interface(primitive.gas_energy_specific, slopes.gas_energy_specific);
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    output.scalar_specific[i] = interface(primitive.scalar_specific[i], slopes.scalar_specific[i]);
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
 * - K. Felker & J. Stone, "A fourth-order accurate finite volume method for ideal MHD via upwind constrained
 * transport", JCP, 375, (2018)
 * - P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations", JCP, 54, 174
 * (1984)
 * - P. Colella & M. Sekora, "A limiter for PPM that preserves accuracy at smooth extrema", JCP, 227, 7069 (2008)
 * - P. McCorquodale & P. Colella,  "A high-order finite-volume method for conservation laws on locally refined
 * grids", CAMCoS, 6, 1 (2011)
 * - P. Colella, M.R. Dorr, J. Hittinger, D. Martin, "High-order, finite-volume methods in mapped coordinates", JCP,
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
      limited_slope = SIGN(slope_2nd_centered) *
                      fmin(C2 * abs(slope_2nd_im1), fmin(C2 * abs(slope_2nd_i), abs(slope_2nd_centered)));
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
    slope_2nd_limited = SIGN(slope_2nd_face) * fmin(fmin(C2 * abs(slope_2nd_im1), C2 * abs(slope_2nd_i)),
                                                    fmin(C2 * abs(slope_2nd_ip1), abs(slope_2nd_face)));
  }

  // Check if 2nd derivative is close to roundoff error
  Real cell_max = fmax(abs(cell_im2), abs(cell_im1));
  cell_max      = fmax(cell_max, abs(cell_i));
  cell_max      = fmax(cell_max, abs(cell_ip1));
  cell_max      = fmax(cell_max, abs(cell_ip2));

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
 * \brief Compute the primitive PPM interfaces. Calls PPM_Single_Variable on each field
 *
 * \param[in] cell_im2 The state of the cell at i-2
 * \param[in] cell_im1 The state of the cell at i-1
 * \param[in] cell_i The state of the cell at i
 * \param[in] cell_ip1 The state of the cell at i+1
 * \param[in] cell_ip2 The state of the cell at i+2
 * \return auto The left interface at i+1/2 and the right interface at i-1/2 in that order
 */
auto __device__ __host__ __inline__ PPM_Interfaces(hydro_utilities::Primitive const &cell_im2,
                                                   hydro_utilities::Primitive const &cell_im1,
                                                   hydro_utilities::Primitive const &cell_i,
                                                   hydro_utilities::Primitive const &cell_ip1,
                                                   hydro_utilities::Primitive const &cell_ip2)
{
  hydro_utilities::Primitive interface_R_imh, interface_L_iph;

  reconstruction::PPM_Single_Variable(cell_im2.density, cell_im1.density, cell_i.density, cell_ip1.density,
                                      cell_ip2.density, interface_L_iph.density, interface_R_imh.density);
  reconstruction::PPM_Single_Variable(cell_im2.velocity.x(), cell_im1.velocity.x(), cell_i.velocity.x(),
                                      cell_ip1.velocity.x(), cell_ip2.velocity.x(), interface_L_iph.velocity.x(),
                                      interface_R_imh.velocity.x());
  reconstruction::PPM_Single_Variable(cell_im2.velocity.y(), cell_im1.velocity.y(), cell_i.velocity.y(),
                                      cell_ip1.velocity.y(), cell_ip2.velocity.y(), interface_L_iph.velocity.y(),
                                      interface_R_imh.velocity.y());
  reconstruction::PPM_Single_Variable(cell_im2.velocity.z(), cell_im1.velocity.z(), cell_i.velocity.z(),
                                      cell_ip1.velocity.z(), cell_ip2.velocity.z(), interface_L_iph.velocity.z(),
                                      interface_R_imh.velocity.z());
  reconstruction::PPM_Single_Variable(cell_im2.pressure, cell_im1.pressure, cell_i.pressure, cell_ip1.pressure,
                                      cell_ip2.pressure, interface_L_iph.pressure, interface_R_imh.pressure);

#ifdef MHD
  reconstruction::PPM_Single_Variable(cell_im2.magnetic.y(), cell_im1.magnetic.y(), cell_i.magnetic.y(),
                                      cell_ip1.magnetic.y(), cell_ip2.magnetic.y(), interface_L_iph.magnetic.y(),
                                      interface_R_imh.magnetic.y());
  reconstruction::PPM_Single_Variable(cell_im2.magnetic.z(), cell_im1.magnetic.z(), cell_i.magnetic.z(),
                                      cell_ip1.magnetic.z(), cell_ip2.magnetic.z(), interface_L_iph.magnetic.z(),
                                      interface_R_imh.magnetic.z());
#endif  // MHD

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
#endif  // DE

  struct LocalReturnStruct {
    hydro_utilities::Primitive left, right;
  };
  return LocalReturnStruct{interface_L_iph, interface_R_imh};
}
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief Compute the characteristic PPM interfaces. Calls PPM_Single_Variable on each field
 *
 * \param[in] cell_im2 The state of the cell at i-2
 * \param[in] cell_im1 The state of the cell at i-1
 * \param[in] cell_i The state of the cell at i
 * \param[in] cell_ip1 The state of the cell at i+1
 * \param[in] cell_ip2 The state of the cell at i+2
 * \return auto The left interface at i+1/2 and the right interface at i-1/2 in that order
 */
auto __device__ __host__ __inline__ PPM_Interfaces(Characteristic const &cell_im2, Characteristic const &cell_im1,
                                                   Characteristic const &cell_i, Characteristic const &cell_ip1,
                                                   Characteristic const &cell_ip2)
{
  Characteristic interface_R_imh, interface_L_iph;

  reconstruction::PPM_Single_Variable(cell_im2.a0, cell_im1.a0, cell_i.a0, cell_ip1.a0, cell_ip2.a0, interface_L_iph.a0,
                                      interface_R_imh.a0);
  reconstruction::PPM_Single_Variable(cell_im2.a1, cell_im1.a1, cell_i.a1, cell_ip1.a1, cell_ip2.a1, interface_L_iph.a1,
                                      interface_R_imh.a1);
  reconstruction::PPM_Single_Variable(cell_im2.a2, cell_im1.a2, cell_i.a2, cell_ip1.a2, cell_ip2.a2, interface_L_iph.a2,
                                      interface_R_imh.a2);
  reconstruction::PPM_Single_Variable(cell_im2.a3, cell_im1.a3, cell_i.a3, cell_ip1.a3, cell_ip2.a3, interface_L_iph.a3,
                                      interface_R_imh.a3);
  reconstruction::PPM_Single_Variable(cell_im2.a4, cell_im1.a4, cell_i.a4, cell_ip1.a4, cell_ip2.a4, interface_L_iph.a4,
                                      interface_R_imh.a4);

#ifdef MHD
  reconstruction::PPM_Single_Variable(cell_im2.a5, cell_im1.a5, cell_i.a5, cell_ip1.a5, cell_ip2.a5, interface_L_iph.a5,
                                      interface_R_imh.a5);
  reconstruction::PPM_Single_Variable(cell_im2.a6, cell_im1.a6, cell_i.a6, cell_ip1.a6, cell_ip2.a6, interface_L_iph.a6,
                                      interface_R_imh.a6);
#endif  // MHD

  struct LocalReturnStruct {
    Characteristic left, right;
  };
  return LocalReturnStruct{interface_L_iph, interface_R_imh};
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
void __device__ __host__ __inline__ Write_Data(hydro_utilities::Primitive const &interface_state, Real *dev_interface,
                                               Real const *dev_conserved, size_t const &id, size_t const &n_cells,
                                               size_t const &o1, size_t const &o2, size_t const &o3, Real const &gamma)
{
  // Write out density and momentum
  dev_interface[grid_enum::density * n_cells + id] = interface_state.density;
  dev_interface[o1 * n_cells + id]                 = interface_state.density * interface_state.velocity.x();
  dev_interface[o2 * n_cells + id]                 = interface_state.density * interface_state.velocity.y();
  dev_interface[o3 * n_cells + id]                 = interface_state.density * interface_state.velocity.z();

#ifdef MHD
  // Write the Y and Z interface states and load the X magnetic face needed to compute the energy
  Real magnetic_x;
  switch (o1) {
    case grid_enum::momentum_x:
      dev_interface[grid_enum::Q_x_magnetic_y * n_cells + id] = interface_state.magnetic.y();
      dev_interface[grid_enum::Q_x_magnetic_z * n_cells + id] = interface_state.magnetic.z();
      magnetic_x                                              = dev_conserved[grid_enum::magnetic_x * n_cells + id];
      break;
    case grid_enum::momentum_y:
      dev_interface[grid_enum::Q_y_magnetic_z * n_cells + id] = interface_state.magnetic.y();
      dev_interface[grid_enum::Q_y_magnetic_x * n_cells + id] = interface_state.magnetic.z();
      magnetic_x                                              = dev_conserved[grid_enum::magnetic_y * n_cells + id];
      break;
    case grid_enum::momentum_z:
      dev_interface[grid_enum::Q_z_magnetic_x * n_cells + id] = interface_state.magnetic.y();
      dev_interface[grid_enum::Q_z_magnetic_y * n_cells + id] = interface_state.magnetic.z();
      magnetic_x                                              = dev_conserved[grid_enum::magnetic_z * n_cells + id];
      break;
  }

  // Compute the MHD energy
  dev_interface[grid_enum::Energy * n_cells + id] = hydro_utilities::Calc_Energy_Primitive(
      interface_state.pressure, interface_state.density, interface_state.velocity.x(), interface_state.velocity.y(),
      interface_state.velocity.z(), gamma, magnetic_x, interface_state.magnetic.y(), interface_state.magnetic.z());
#else   // not MHD
  // Compute the hydro energy
  dev_interface[grid_enum::Energy * n_cells + id] = hydro_utilities::Calc_Energy_Primitive(
      interface_state.pressure, interface_state.density, interface_state.velocity.x(), interface_state.velocity.y(),
      interface_state.velocity.z(), gamma);
#endif  // MHD

#ifdef DE
  dev_interface[grid_enum::GasEnergy * n_cells + id] = interface_state.density * interface_state.gas_energy_specific;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_interface[(grid_enum::scalar + i) * n_cells + id] =
        interface_state.density * interface_state.scalar_specific[i];
  }
#endif  // SCALAR
}
// =====================================================================================================================
}  // namespace reconstruction

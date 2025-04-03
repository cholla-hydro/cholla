/*!
 * \file basic_structs.h
 * \brief Constains some basic structs to be used around the code. Mostly this is here instead of hydro_utilities.h to
 * avoid circulary dependencies with mhd_utils.h
 *
 */

#pragma once

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"

namespace hydro_utilities
{
// =====================================================================================================================
// Here are some basic structs that can be used in various places when needed
// =====================================================================================================================
/*!
 * \brief A data only struct that acts as a simple 3 element vector.
 *
 */
struct VectorXYZ {
  /// Tracks the values held by the class. To ensure the class is an aggregate, it needs to be public. With that said,
  /// it should be treated as an implementation detail and not accessed directly
  Real arr_[3];

  /// To ensure this class is an aggregate, constructors are implicitly defined. The destructor & move/copy assignment
  /// operations are also implicitly defined

  /*!
   * \brief Returns the pointer to the data array
   *
   * \return Real* The pointer to the data array
   */
  __device__ __host__ Real* data() { return arr_; }

  /*!
   * \brief Overload for the [] operator to allow array-like access. Const version is needed if the object instance is
   * declared as const
   *
   * \param i Which element to access. Allowable values are 0, 1, and 2, all other values have undefined behaviour that
   * will result in a segfault or illegally memory access \return Real& Reference to the vector element at the ith
   * location
   */
  ///@{
  __device__ __host__ Real& operator[](std::size_t i) { return arr_[i]; }
  __device__ __host__ const Real& operator[](std::size_t i) const { return arr_[i]; }
  ///@}

  /*!
   * \brief Directly access the x, y, and z elements. Const version is needed if the object instance is declared as
   * const
   *
   * \return Real& Reference to the vector element at the ith location
   */
  ///@{
  __device__ __host__ Real& x() noexcept { return arr_[0]; }
  __device__ __host__ const Real& x() const noexcept { return arr_[0]; }
  __device__ __host__ Real& y() noexcept { return arr_[1]; }
  __device__ __host__ const Real& y() const noexcept { return arr_[1]; }
  __device__ __host__ Real& z() noexcept { return arr_[2]; }
  __device__ __host__ const Real& z() const noexcept { return arr_[2]; }
  ///@}
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief A data only struct for the conserved variables
 *
 */
struct Conserved {
  // Hydro variables
  Real density, energy;
  VectorXYZ momentum;

#ifdef MHD
  // These are all cell centered values
  VectorXYZ magnetic;
#endif  // MHD

#ifdef DE
  Real gas_energy;
#endif  // DE

#ifdef SCALAR
  Real scalar[grid_enum::nscalars];
#endif  // SCALAR

  /// Default constructor, should init everything to zero
  Conserved() = default;
  /// Manual constructor, mostly used for testing and doesn't init all members
  Conserved(Real const in_density, VectorXYZ const& in_momentum, Real const in_energy,
            VectorXYZ const& in_magnetic = {0, 0, 0}, Real const in_gas_energy = 0.0)
      : density(in_density), momentum(in_momentum), energy(in_energy)
  {
#ifdef MHD
    magnetic = in_magnetic;
#endif  // mhd

#ifdef DE
    gas_energy = in_gas_energy;
#endif  // DE
  };
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief A data only struct for the primitive variables
 *
 */
struct Primitive {
  // Hydro variable
  Real density, pressure;
  VectorXYZ velocity;

#ifdef MHD
  // These are all cell centered values
  VectorXYZ magnetic;
#endif  // MHD

#ifdef DE
  /// The specific thermal energy in the gas
  Real gas_energy;
#endif  // DE

#ifdef SCALAR
  Real scalar[grid_enum::nscalars];
#endif  // SCALAR

  /// Default constructor, should init everything to zero
  Primitive() = default;
  /// Manual constructor, mostly used for testing and doesn't init all members. The `in_` prefix stands for input,
  /// mostly to avoid name collision with the member variables
  Primitive(Real const in_density, VectorXYZ const& in_velocity, Real const in_pressure,
            VectorXYZ const& in_magnetic = {0, 0, 0}, Real const in_gas_energy = 0.0)
      : density(in_density), velocity(in_velocity), pressure(in_pressure)
  {
#ifdef MHD
    magnetic = in_magnetic;
#endif  // mhd

#ifdef DE
    gas_energy = in_gas_energy;
#endif  // DE
  };
};
// =====================================================================================================================
}  // namespace hydro_utilities

namespace reconstruction
{
struct InterfaceState {
  // Hydro variables
  Real density, energy;
  /// Note that `pressure` here is the gas pressure not the total pressure which would include the magnetic component
  Real pressure;
  hydro_utilities::VectorXYZ velocity, momentum;

#ifdef MHD
  // These are all cell centered values
  Real total_pressure;
  hydro_utilities::VectorXYZ magnetic;
#endif  // MHD

#ifdef DE
  Real gas_energy;
#endif  // DE

#ifdef SCALAR
  Real scalar[grid_enum::nscalars];
#endif  // SCALAR

  // Define the constructors
  /// Default constructor, should set everything to 0
  InterfaceState() = default;
  /// Initializing constructor: used to initialize to specific values, mostly used in tests. It only initializes a
  /// subset of the member variables since that is what is used in tests at the time of writing.
  InterfaceState(Real const in_density, hydro_utilities::VectorXYZ const in_velocity, Real const in_energy,
                 Real const in_pressure, hydro_utilities::VectorXYZ const in_magnetic = {0, 0, 0},
                 Real const in_total_pressure = 0.0)
      : density(in_density), velocity(in_velocity), energy(in_energy), pressure(in_pressure)
  {
    momentum.x() = velocity.x() * density;
    momentum.y() = velocity.y() * density;
    momentum.z() = velocity.z() * density;
#ifdef MHD
    magnetic       = in_magnetic;
    total_pressure = in_total_pressure;
#endif  // MHD
#ifdef DE
    gas_energy = 0.0;
#endif  // DE
  };
};
}  // namespace reconstruction

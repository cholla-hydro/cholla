/*! \file plmc_cuda.h
 *  \brief Declarations of the cuda plm kernels, characteristic reconstruction
 * version. */

#ifndef PLMC_CUDA_H
#define PLMC_CUDA_H

#include "../global/global.h"
#include "../grid/grid_enum.h"

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
  Real magnetic_y, magnetic_z;
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
 * \param dev_conserved The conserved array
 * \param xid The xid of the cell to load data from
 * \param yid The yid of the cell to load data from
 * \param zid The zid of the cell to load data from
 * \param nx Size in the X direction
 * \param ny Size in the Y direction
 * \param n_cells The total number of cells
 * \param o1 Directional parameter
 * \param o2 Directional parameter
 * \param o3 Directional parameter
 * \param gamma The adiabatic index
 * \return PlmcPrimitive The loaded cell data
 */
PlmcPrimitive __device__ __host__ Load_Data(Real const *dev_conserved, size_t const &xid, size_t const &yid,
                                            size_t const &zid, size_t const &nx, size_t const &ny,
                                            size_t const &n_cells, size_t const &o1, size_t const &o2, size_t const &o3,
                                            Real const &gamma);

/*!
 * \brief Compute a simple slope. Equation is `coef * (left - right)`.
 *
 * \param left The data on the positive side of the slope
 * \param right The data on the negative side of the slope
 * \param coef The coefficient to multiply the slope by. Defaults to zero
 * \return PlmcPrimitive The slopes
 */
PlmcPrimitive __device__ __host__ Compute_Slope(PlmcPrimitive const &left, PlmcPrimitive const &right,
                                                Real const &coef = 1.0);

/*!
 * \brief Compute the Van Lear slope from the left and right slopes
 *
 * \param left_slope The left slope
 * \param right_slope The right slope
 * \return PlmcPrimitive The Van Leer slope
 */
PlmcPrimitive __device__ __host__ Van_Leer_Slope(PlmcPrimitive const &left_slope, PlmcPrimitive const &right_slope);

/*!
 * \brief Project from the primitive variables slopes to the characteristic variables slopes. Stone Eqn 37. Use the
 * eigenvectors given in Stone 2008, Appendix A
 *
 * \param primitive The primitive variables
 * \param primitive_slope The primitive variables slopes
 * \param sound_speed The speed of sound
 * \param sound_speed_squared The speed of sound squared
 * \return PlmcCharacteristic
 */
PlmcCharacteristic __device__ __host__ Primitive_To_Characteristic(PlmcPrimitive const &primitive,
                                                                   PlmcPrimitive const &primitive_slope,
                                                                   Real const &sound_speed,
                                                                   Real const &sound_speed_squared);
}  // namespace plmc_utils
#endif  // PLMC_CUDA_H

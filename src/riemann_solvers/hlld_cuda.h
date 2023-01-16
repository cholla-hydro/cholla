/*!
 * \file hlld_cuda.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declaration of the HLLD solver from Miyoshi & Kusano 2005
 * "A multi-state HLL approximate Riemann solver for ideal
 * magnetohydrodynamics", hereafter referred to as M&K 2005
 *
 */

#pragma once

// External Includes

// Local Includes
#include "../global/global.h"

#ifdef CUDA
/*!
 * \brief Namespace for MHD code
 *
 */
namespace mhd
{
/*!
 * \brief Compute the HLLD fluxes from Miyoshi & Kusano 2005
 *
 * \param[in]  dev_bounds_L The interface states on the left side of the
 * interface \param[in]  dev_bounds_R The interface states on the right side of
 * the interface \param[in]  dev_magnetic_face A pointer to the begining of the
 * conserved magnetic field array that is stored at the interface. I.e. for the
 * X-direction solve this would be the begining of the X-direction fields
 * \param[out] dev_flux The output flux
 * \param[in]  nx Number of cells in the X-direction
 * \param[in]  ny Number of cells in the Y-direction
 * \param[in]  nz Number of cells in the Z-direction
 * \param[in]  n_ghost Number of ghost cells on each side
 * \param[in]  gamma The adiabatic index
 * \param[in]  dir The direction that the solve is taking place in. 0=X, 1=Y,
 * 2=Z \param[in]  n_fields The total number of fields
 */
__global__ void Calculate_HLLD_Fluxes_CUDA(Real *dev_bounds_L,
                                           Real *dev_bounds_R,
                                           Real *dev_magnetic_face,
                                           Real *dev_flux, int nx, int ny,
                                           int nz, int n_ghost, Real gamma,
                                           int direction, int n_fields);

/*!
 * \brief Namespace to hold private functions used within the HLLD
 * solver
 *
 */
namespace _internal
{
/*!
 * \brief Used for some comparisons. Value was chosen to match what is
 * used in Athena
 */
Real static const _hlldSmallNumber = 1.0e-8;

/*!
 * \brief Compute the left, right, star, and middle wave speeds. Also
 * returns the densities in the star states. M&K 2005 equations 38, 43,
 * 51, and 67
 *
 * \param[in] densityL Density, left side
 * \param[in] momentumXL Momentum in the X-direction, left side
 * \param[in] momentumYL Momentum in the Y-direction, left side
 * \param[in] momentumZL Momentum in the Z-direction, left side
 * \param[in] velocityXL Velocity in the X-direction, left side
 * \param[in] velocityYL Velocity in the Y-direction, left side
 * \param[in] velocityZL Velocity in the Z-direction, left side
 * \param[in] gasPressureL Gas pressure, left side
 * \param[in] totalPressureL Total MHD pressure, left side
 * \param[in] magneticX Magnetic field in the X-direction, left side
 * \param[in] magneticYL Magnetic field in the Y-direction, left side
 * \param[in] magneticZL Magnetic field in the Z-direction, left side
 * \param[in] densityR Density, right side
 * \param[in] momentumXR Momentum in the X-direction, right side
 * \param[in] momentumYR Momentum in the Y-direction, right side
 * \param[in] momentumZR Momentum in the Z-direction, right side
 * \param[in] velocityXR Velocity in the X-direction, right side
 * \param[in] velocityYR Velocity in the Y-direction, right side
 * \param[in] velocityZR Velocity in the Z-direction, right side
 * \param[in] gasPressureR Gas pressure, right side
 * \param[in] totalPressureR Total MHD pressure, right side
 * \param[in] magneticYR Magnetic field in the Y-direction, right side
 * \param[in] magneticZR Magnetic field in the Z-direction, right side
 * \param[in] gamma Adiabatic index
 * \param[out] speedL Approximate speed of the left most wave
 * \param[out] speedR Approximate speed of the right most wave
 * \param[out] speedM Speed of the middle wave
 * \param[out] speedStarL Speed of the left star state wave
 * \param[out] speedStarR Speed of the right star state wave
 * \param[out] densityStarL Density in left star region
 * \param[out] densityStarR Density in right star region
 */
__device__ __host__ void _approximateWaveSpeeds(
    Real const &densityL, Real const &momentumXL, Real const &momentumYL,
    Real const &momentumZL, Real const &velocityXL, Real const &velocityYL,
    Real const &velocityZL, Real const &gasPressureL,
    Real const &totalPressureL, Real const &magneticX, Real const &magneticYL,
    Real const &magneticZL, Real const &densityR, Real const &momentumXR,
    Real const &momentumYR, Real const &momentumZR, Real const &velocityXR,
    Real const &velocityYR, Real const &velocityZR, Real const &gasPressureR,
    Real const &totalPressureR, Real const &magneticYR, Real const &magneticZR,
    Real const &gamma, Real &speedL, Real &speedR, Real &speedM,
    Real &speedStarL, Real &speedStarR, Real &densityStarL, Real &densityStarR);

/*!
 * \brief Compute the fluxes in the left or right non-star state
 *
 * \param[in] momentumX Momentum in the X-direction
 * \param[in] velocityX Velocity in the X-direction
 * \param[in] velocityY Velocity in the Y-direction
 * \param[in] velocityZ Velocity in the Z-direction
 * \param[in] totalPressure Total MHD pressure
 * \param[in] energy Energy
 * \param[in] magneticX Magnetic field in -direction
 * \param[in] magneticY Magnetic field in -direction
 * \param[in] magneticZ Magnetic field in -direction
 * \param[out] densityFlux The density flux
 * \param[out] momentumFluxX The momentum flux in the X-direction
 * \param[out] momentumFluxY The momentum flux in the Y-direction
 * \param[out] momentumFluxZ The momentum flux in the Z-direction
 * \param[out] magneticFluxY The magnetic field flux in the Y-direction
 * \param[out] magneticFluxZ The magnetic field flux in the Z-direction
 * \param[out] energyFlux The energy flux
 */
__device__ __host__ void _nonStarFluxes(
    Real const &momentumX, Real const &velocityX, Real const &velocityY,
    Real const &velocityZ, Real const &totalPressure, Real const &energy,
    Real const &magneticX, Real const &magneticY, Real const &magneticZ,
    Real &densityFlux, Real &momentumFluxX, Real &momentumFluxY,
    Real &momentumFluxZ, Real &magneticFluxY, Real &magneticFluxZ,
    Real &energyFlux);

/*!
 * \brief Assign the given flux values to the dev_flux array
 *
 * \param[in] threadId The thread ID
 * \param[in] o1 Offset to get indexing right
 * \param[in] o2 Offset to get indexing right
 * \param[in] o3 Offset to get indexing right
 * \param[in] n_cells Number of cells
 * \param[out] dev_flux The flux array
 * \param[in] densityFlux The density flux
 * \param[in] momentumFluxX The momentum flux in the X-direction
 * \param[in] momentumFluxY The momentum flux in the Y-direction
 * \param[in] momentumFluxZ The momentum flux in the Z-direction
 * \param[in] magneticFluxY The magnetic field flux in the X-direction
 * \param[in] magneticFluxZ The magnetic field flux in the Y-direction
 * \param[in] energyFlux The energy flux
 */
__device__ __host__ void _returnFluxes(
    int const &threadId, int const &o1, int const &o2, int const &o3,
    int const &n_cells, Real *dev_flux, Real const &densityFlux,
    Real const &momentumFluxX, Real const &momentumFluxY,
    Real const &momentumFluxZ, Real const &magneticFluxY,
    Real const &magneticFluxZ, Real const &energyFlux);

/*!
 * \brief Compute the fluxes in the left or right star state. M&K 2005
 * equations 44-48, 64
 *
 * \param[in] speedM Speed of the central wave
 * \param[in] speedSide Speed of the non-star wave on the side being computed
 * \param[in] density Density
 * \param[in] velocityX Velocity in the X-direction
 * \param[in] velocityY Velocity in the Y-direction
 * \param[in] velocityZ Velocity in the Z-direction
 * \param[in] momentumX Momentum in the X-direction
 * \param[in] momentumY Momentum in the Y-direction
 * \param[in] momentumZ Momentum in the Z-direction
 * \param[in] energy Energy
 * \param[in] totalPressure Total MHD pressure
 * \param[in] magneticX Magnetic field in the X-direction
 * \param[in] magneticY Magnetic field in the Y-direction
 * \param[in] magneticZ Magnetic field in the Z-direction
 * \param[in] densityStar Density in the star state
 * \param[in] totalPressureStar Total MHD pressure in the star state
 * \param[in] densityFlux Density Flux from the non-star state
 * \param[in] momentumFluxX Momentum flux from the non-star state in the
 * X-direction \param[in] momentumFluxY Momentum flux from the non-star state in
 * the Y-direction \param[in] momentumFluxZ Momentum flux from the non-star
 * state in the Z-direction \param[in] energyFlux Energy flux from the non-star
 * state \param[in] magneticFluxY Magnetic flux from the non-star state in the
 * X-direction \param[in] magneticFluxZ Magnetic flux from the non-star state in
 * the Y-direction \param[out] velocityStarY Velocity in the star state in the
 * Y-direction \param[out] velocityStarZ Velocity in the star state in the
 * Z-direction \param[out] energyStar Energy in the star state \param[out]
 * magneticStarY Magnetic field in the star state in the X-direction \param[out]
 * magneticStarZ Magnetic field in the star state in the Y-direction \param[out]
 * densityStarFlux Density flux in the star state \param[out] momentumStarFluxX
 * Momentum flux in the star state in the X-direction \param[out]
 * momentumStarFluxY Momentum flux in the star state in the Y-direction
 * \param[out] momentumStarFluxZ Momentum flux in the star state in the
 * Z-direction \param[out] energyStarFlux Energy flux in the star state
 * \param[out] magneticStarFluxY Magnetic field flux in the star state in the
 * X-direction \param[out] magneticStarFluxZ Magnetic field flux in the star
 * state in the Y-direction
 *
 */
__device__ __host__ void _starFluxes(
    Real const &speedM, Real const &speedSide, Real const &density,
    Real const &velocityX, Real const &velocityY, Real const &velocityZ,
    Real const &momentumX, Real const &momentumY, Real const &momentumZ,
    Real const &energy, Real const &totalPressure, Real const &magneticX,
    Real const &magneticY, Real const &magneticZ, Real const &densityStar,
    Real const &totalPressureStar, Real const &densityFlux,
    Real const &momentumFluxX, Real const &momentumFluxY,
    Real const &momentumFluxZ, Real const &energyFlux,
    Real const &magneticFluxY, Real const &magneticFluxZ, Real &velocityStarY,
    Real &velocityStarZ, Real &energyStar, Real &magneticStarY,
    Real &magneticStarZ, Real &densityStarFlux, Real &momentumStarFluxX,
    Real &momentumStarFluxY, Real &momentumStarFluxZ, Real &energyStarFlux,
    Real &magneticStarFluxY, Real &magneticStarFluxZ);

/*!
 * \brief Compute the double star state. M&K 2005 equations 59-63
 *
 * \param[in] speedM
 * \param[in] magneticX
 * \param[in] totalPressureStar
 * \param[in] densityStarL
 * \param[in] velocityStarYL
 * \param[in] velocityStarZL
 * \param[in] energyStarL
 * \param[in] magneticStarYL
 * \param[in] magneticStarZL
 * \param[in] densityStarR
 * \param[in] velocityStarYR
 * \param[in] velocityStarZR
 * \param[in] energyStarR
 * \param[in] magneticStarYR
 * \param[in] magneticStarZR
 * \param[out] velocityDoubleStarY
 * \param[out] velocityDoubleStarZ
 * \param[out] magneticDoubleStarY
 * \param[out] magneticDoubleStarZ
 * \param[out] energyDoubleStarL
 * \param[out] energyDoubleStarR
 */
__device__ __host__ void _doubleStarState(
    Real const &speedM, Real const &magneticX, Real const &totalPressureStar,
    Real const &densityStarL, Real const &velocityStarYL,
    Real const &velocityStarZL, Real const &energyStarL,
    Real const &magneticStarYL, Real const &magneticStarZL,
    Real const &densityStarR, Real const &velocityStarYR,
    Real const &velocityStarZR, Real const &energyStarR,
    Real const &magneticStarYR, Real const &magneticStarZR,
    Real &velocityDoubleStarY, Real &velocityDoubleStarZ,
    Real &magneticDoubleStarY, Real &magneticDoubleStarZ,
    Real &energyDoubleStarL, Real &energyDoubleStarR);

/*!
 * \brief Compute the double star state fluxes. M&K 2005 equation 65
 *
 * \param[in] speedStarSide The star speed on the side being computed
 * \param[in] momentumStarFluxX
 * \param[in] momentumStarFluxY
 * \param[in] momentumStarFluxZ
 * \param[in] energyStarFlux
 * \param[in] magneticStarFluxY
 * \param[in] magneticStarFluxZ
 * \param[in] densityStar
 * \param[in] velocityStarX
 * \param[in] velocityStarY
 * \param[in] velocityStarZ
 * \param[in] energyStar
 * \param[in] magneticStarY
 * \param[in] magneticStarZ
 * \param[in] velocityDoubleStarX
 * \param[in] velocityDoubleStarY
 * \param[in] velocityDoubleStarZ
 * \param[in] energyDoubleStar
 * \param[in] magneticDoubleStarY
 * \param[in] magneticDoubleStarZ
 * \param[out] momentumDoubleStarFluxX
 * \param[out] momentumDoubleStarFluxY
 * \param[out] momentumDoubleStarFluxZ
 * \param[out] energyDoubleStarFlux
 * \param[out] magneticDoubleStarFluxY
 * \param[out] magneticDoubleStarFluxZ
 */
__device__ __host__ void _doubleStarFluxes(
    Real const &speedStarSide, Real const &momentumStarFluxX,
    Real const &momentumStarFluxY, Real const &momentumStarFluxZ,
    Real const &energyStarFlux, Real const &magneticStarFluxY,
    Real const &magneticStarFluxZ, Real const &densityStar,
    Real const &velocityStarX, Real const &velocityStarY,
    Real const &velocityStarZ, Real const &energyStar,
    Real const &magneticStarY, Real const &magneticStarZ,
    Real const &velocityDoubleStarX, Real const &velocityDoubleStarY,
    Real const &velocityDoubleStarZ, Real const &energyDoubleStar,
    Real const &magneticDoubleStarY, Real const &magneticDoubleStarZ,
    Real &momentumDoubleStarFluxX, Real &momentumDoubleStarFluxY,
    Real &momentumDoubleStarFluxZ, Real &energyDoubleStarFlux,
    Real &magneticDoubleStarFluxY, Real &magneticDoubleStarFluxZ);

}  // namespace _internal
}  // end namespace mhd
#endif  // CUDA

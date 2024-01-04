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
#include "../utils/hydro_utilities.h"

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
 * interface
 * \param[in]  dev_bounds_R The interface states on the right side of
 * the interface
 * \param[in]  dev_magnetic_face A pointer to the begining of the
 * conserved magnetic field array that is stored at the interface. I.e. for the
 * X-direction solve this would be the begining of the X-direction fields
 * \param[out] dev_flux The output flux
 * \param[in]  n_cells Total number of cells
 * \param[in]  n_ghost Number of ghost cells on each side
 * \param[in]  dir The direction that the solve is taking place in. 0=X, 1=Y,
 * 2=Z
 * \param[in]  n_fields The total number of fields
 */
__global__ void Calculate_HLLD_Fluxes_CUDA(Real const *dev_bounds_L, Real const *dev_bounds_R,
                                           Real const *dev_magnetic_face, Real *dev_flux, int const n_cells,
                                           Real const gamma, int const direction, int const n_fields);

/*!
 * \brief Namespace to hold private functions used within the HLLD
 * solver
 *
 */
namespace internal
{
/*!
 * \brief Used for some comparisons. Value was chosen to match what is
 * used in Athena
 */
Real static const _hlldSmallNumber = 1.0e-8;

/*!
 * \brief Holds all the data needed for the non-star states of the HLLD solver
 *
 */
struct State {
  Real density, velocityX, velocityY, velocityZ, energy, magneticY, magneticZ, gasPressure, totalPressure;
  #ifdef SCALAR
  Real scalarSpecific[grid_enum::nscalars];
  #endif  // SCALAR
  #ifdef DE
  Real thermalEnergySpecific;
  #endif  // DE
};

/*!
 * \brief Holds all the data needed for the star states of the HLLD solver
 * except total pressure and x velocity as those are shared between the left and
 * right states
 *
 */
struct StarState {
  // velocityStarX = Speeds.M
  // Total pressure is computed on its own since it's shared
  Real density, velocityY, velocityZ, energy, magneticY, magneticZ;
};

/*!
 * \brief Holds all the data needed for the double star states of the HLLD
 * solver except the x velocity, density, and total pressure since those are all
 * inherited from the star state.
 *
 */
struct DoubleStarState {
  // velocityDoubleStarX = Speeds.M
  // densityDoubleStar = densityStar
  // pressureDoubleStar = pressureStar
  // Shared values
  Real velocityY, velocityZ, magneticY, magneticZ;
  // Different values. Initializing these since one or the other can be uninitializing leading to bad tests
  Real energyL = 0.0, energyR = 0.0;
};

/*!
 * \brief Holds all the data needed for the fluxes in the HLLD solver
 *
 */
struct Flux {
  Real density, momentumX, momentumY, momentumZ, energy, magneticY, magneticZ;
};

/*!
 * \brief Holds all the data needed for the speeds in the HLLD solver
 *
 */
struct Speeds {
  Real L, LStar, M, RStar, R;
};

/*!
 * \brief Load and compute the left or right state
 *
 * \param interfaceArr The interface array to load from
 * \param magneticX The X magnetic field
 * \param gamma The adiabatic index
 * \param threadId The thread ID
 * \param n_cells Total number of cells
 * \param o1 Direction parameter
 * \param o2 Direction parameter
 * \param o3 Direction parameter
 * \return mhd::internal::State The loaded state
 */
__device__ __host__ mhd::internal::State loadState(Real const *interfaceArr, Real const &magneticX, Real const &gamma,
                                                   int const &threadId, int const &n_cells, int const &o1,
                                                   int const &o2, int const &o3);

/*!
 * \brief Compute the approximate left and right wave speeds. M&K 2005 equation
 * 67
 */
__device__ __host__ mhd::internal::Speeds approximateLRWaveSpeeds(mhd::internal::State const &stateL,
                                                                  mhd::internal::State const &stateR,
                                                                  Real const &magneticX, Real const &gamma);

/*!
 * \brief Compute the approximate middle wave speed. M&K 2005 equation 38
 */
__device__ __host__ Real approximateMiddleWaveSpeed(mhd::internal::State const &stateL,
                                                    mhd::internal::State const &stateR,
                                                    mhd::internal::Speeds const &speed);

/*!
 * \brief Compute the approximate left and right wave speeds. M&K 2005 equation
 * 51
 */
__device__ __host__ Real approximateStarWaveSpeed(mhd::internal::StarState const &starState,
                                                  mhd::internal::Speeds const &speed, Real const &magneticX,
                                                  Real const &side);

/*!
 * \brief Compute the fluxes in the left or right non-star state. M&K 2005
 * equation 2
 *
 * \param state The state to compute the flux of
 * \param magneticX The X magnetic field
 * \return mhd::internal::Flux The flux in the state
 */
__device__ __host__ mhd::internal::Flux nonStarFluxes(mhd::internal::State const &state, Real const &magneticX);

/*!
 * \brief Write the given flux values to the dev_flux array
 *
 * \param[in] threadId The thread ID
 * \param[in] o1 Offset to get indexing right
 * \param[in] o2 Offset to get indexing right
 * \param[in] o3 Offset to get indexing right
 * \param[in] n_cells Number of cells
 * \param[out] dev_flux The flux array
 * \param[in] flux The fluxes to write out
 * \param[in] state The left or right state depending on if this is a return for
 * one of the left states or one of the right states
 */
__device__ __host__ void returnFluxes(int const &threadId, int const &o1, int const &o2, int const &o3,
                                      int const &n_cells, Real *dev_flux, mhd::internal::Flux const &flux,
                                      mhd::internal::State const &state);

/*!
 * \brief Compute the total pressure in the star states. M&K 2005 equation 41
 *
 * \param stateL The left state
 * \param stateR The right state
 * \param speed The wave speeds
 * \return Real The total pressure in the star state
 */
__device__ __host__ Real starTotalPressure(mhd::internal::State const &stateL, mhd::internal::State const &stateR,
                                           mhd::internal::Speeds const &speed);

/*!
 * \brief Compute the L* or R* state. M&K 2005 equations 43-48
 *
 * \param state The non-star state on the same side as the desired star
 * state \param speed The wavespeeds \param speedSide The wave speed on the
 * same side as the desired star state \param magneticX The magnetic field
 * in the x direction \param totalPressureStar The total pressure in the
 * star state \return mhd::internal::StarState The computed star state
 */
__device__ __host__ mhd::internal::StarState computeStarState(mhd::internal::State const &state,
                                                              mhd::internal::Speeds const &speed, Real const &speedSide,
                                                              Real const &magneticX, Real const &totalPressureStar);

/*!
 * \brief Compute the flux in the star state. M&K 2005 equation 64
 *
 * \param starState The star state to compute the flux of
 * \param state The non-star state on the same side as the star state
 * \param flux The non-star flux on the same side as the star state
 * \param speed The wave speeds
 * \param speedSide The non-star wave speed on the same side as the star state
 * \return mhd::internal::Flux The flux in the star state
 */
__device__ __host__ mhd::internal::Flux starFluxes(mhd::internal::StarState const &starState,
                                                   mhd::internal::State const &state, mhd::internal::Flux const &flux,
                                                   mhd::internal::Speeds const &speed, Real const &speedSide);

/*!
 * \brief Compute the double star state. M&K 2005 equations 59-63
 *
 * \param starStateL The Left star state
 * \param starStateR The Right star state
 * \param magneticX The x magnetic field
 * \param totalPressureStar The total pressure in the star state
 * \param speed The approximate wave speeds
 * \return mhd::internal::DoubleStarState The double star state
 */
__device__ __host__ mhd::internal::DoubleStarState computeDoubleStarState(mhd::internal::StarState const &starStateL,
                                                                          mhd::internal::StarState const &starStateR,
                                                                          Real const &magneticX,
                                                                          Real const &totalPressureStar,
                                                                          mhd::internal::Speeds const &speed);

/*!
 * \brief Compute the double star state fluxes. M&K 2005 equation 65
 *
 * \param doubleStarState The double star states
 * \param starState The star state on the same side
 * \param state The non-star state on the same side
 * \param flux The non-star flux on the same side
 * \param speed The approximate wave speeds
 * \param speedSide The wave speed on the same side
 * \param speedSideStar The star wave speed on the same side
 * \return __device__
 */
__device__ __host__ mhd::internal::Flux computeDoubleStarFluxes(
    mhd::internal::DoubleStarState const &doubleStarState, Real const &doubleStarStateEnergy,
    mhd::internal::StarState const &starState, mhd::internal::State const &state, mhd::internal::Flux const &flux,
    mhd::internal::Speeds const &speed, Real const &speedSide, Real const &speedSideStar);

/*!
 * \brief Specialization of mhd::utils::computeGasPressure for use in the HLLD solver
 *
 * \param state The State to compute the gas pressure of
 * \param magneticX The X magnetic field
 * \param gamma The adiabatic index
 * \return Real The gas pressure
 */
inline __host__ __device__ Real Calc_Pressure_Primitive(mhd::internal::State const &state, Real const &magneticX,
                                                        Real const &gamma)
{
  return hydro_utilities::Calc_Pressure_Primitive(state.energy, state.density, state.velocityX, state.velocityY,
                                                  state.velocityZ, gamma, magneticX, state.magneticY, state.magneticZ);
}
}  // namespace internal
}  // end namespace mhd
#endif  // CUDA

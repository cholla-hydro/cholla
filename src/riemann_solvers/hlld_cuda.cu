/*!
 * \file hlld_cuda.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the implementation of the HLLD solver from Miyoshi & Kusano
 * 2005 "A multi-state HLL approximate Riemann solver for ideal
 * magnetohydrodynamics", hereafter referred to as M&K 2005
 *
 */

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../grid/grid_enum.h"
#include "../riemann_solvers/hlld_cuda.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"
#include "../utils/math_utilities.h"
#include "../utils/mhd_utilities.h"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif  // DE

#ifdef CUDA

  #ifdef MHD
namespace mhd
{
// =========================================================================
__global__ void Calculate_HLLD_Fluxes_CUDA(Real const *dev_bounds_L, Real const *dev_bounds_R,
                                           Real const *dev_magnetic_face, Real *dev_flux, int const n_cells,
                                           Real const gamma, int const direction, int const n_fields)
{
  // get a thread index
  int const threadId = threadIdx.x + blockIdx.x * blockDim.x;

  // Thread guard to avoid overrun
  if (threadId >= n_cells) {
    return;
  }

  // Offsets & indices
  int o1, o2, o3;
  switch (direction) {
    case 0:
      o1 = grid_enum::momentum_x;
      o2 = grid_enum::momentum_y;
      o3 = grid_enum::momentum_z;
      break;
    case 1:
      o1 = grid_enum::momentum_y;
      o2 = grid_enum::momentum_z;
      o3 = grid_enum::momentum_x;
      break;
    case 2:
      o1 = grid_enum::momentum_z;
      o2 = grid_enum::momentum_x;
      o3 = grid_enum::momentum_y;
      break;
  }

  // ============================
  // Retrieve state variables
  // ============================
  // The magnetic field in the X-direction
  Real const magneticX = dev_magnetic_face[threadId];

  mhd::_internal::State const stateL =
      mhd::_internal::loadState(dev_bounds_L, magneticX, gamma, threadId, n_cells, o1, o2, o3);
  mhd::_internal::State const stateR =
      mhd::_internal::loadState(dev_bounds_R, magneticX, gamma, threadId, n_cells, o1, o2, o3);

  // Compute the approximate Left and Right wave speeds
  mhd::_internal::Speeds speed = mhd::_internal::approximateLRWaveSpeeds(stateL, stateR, magneticX, gamma);

  // =================================================================
  // Compute the fluxes in the non-star states
  // =================================================================
  // Left state
  mhd::_internal::Flux fluxL = mhd::_internal::nonStarFluxes(stateL, magneticX);

  // If we're in the L state then assign fluxes and return.
  // In this state the flow is supersonic
  // M&K 2005 equation 66
  if (speed.L > 0.0) {
    mhd::_internal::returnFluxes(threadId, o1, o2, o3, n_cells, dev_flux, fluxL, stateL);
    return;
  }
  // Right state
  mhd::_internal::Flux fluxR = mhd::_internal::nonStarFluxes(stateR, magneticX);

  // If we're in the R state then assign fluxes and return.
  // In this state the flow is supersonic
  // M&K 2005 equation 66
  if (speed.R < 0.0) {
    mhd::_internal::returnFluxes(threadId, o1, o2, o3, n_cells, dev_flux, fluxR, stateR);
    return;
  }

  // =================================================================
  // Compute the fluxes in the star states
  // =================================================================
  // Shared quantities:
  // - velocityStarX = speedM
  // - totalPrssureStar is the same on both sides
  speed.M                      = approximateMiddleWaveSpeed(stateL, stateR, speed);
  Real const totalPressureStar = mhd::_internal::starTotalPressure(stateL, stateR, speed);

  // Left star state
  mhd::_internal::StarState const starStateL =
      mhd::_internal::computeStarState(stateL, speed, speed.L, magneticX, totalPressureStar);

  // Left star speed
  speed.LStar = mhd::_internal::approximateStarWaveSpeed(starStateL, speed, magneticX, -1);

  // If we're in the L* state then assign fluxes and return.
  // In this state the flow is subsonic
  // M&K 2005 equation 66
  if (speed.LStar > 0.0 and speed.L <= 0.0) {
    fluxL = mhd::_internal::starFluxes(starStateL, stateL, fluxL, speed, speed.L);
    mhd::_internal::returnFluxes(threadId, o1, o2, o3, n_cells, dev_flux, fluxL, stateL);
    return;
  }

  // Right star state
  mhd::_internal::StarState const starStateR =
      mhd::_internal::computeStarState(stateR, speed, speed.R, magneticX, totalPressureStar);

  // Right star speed
  speed.RStar = mhd::_internal::approximateStarWaveSpeed(starStateR, speed, magneticX, 1);

  // If we're in the R* state then assign fluxes and return.
  // In this state the flow is subsonic
  // M&K 2005 equation 66
  if (speed.RStar <= 0.0 and speed.R >= 0.0) {
    fluxR = mhd::_internal::starFluxes(starStateR, stateR, fluxR, speed, speed.R);
    mhd::_internal::returnFluxes(threadId, o1, o2, o3, n_cells, dev_flux, fluxR, stateR);
    return;
  }

  // =================================================================
  // Compute the fluxes in the double star states
  // =================================================================
  mhd::_internal::DoubleStarState const doubleStarState =
      mhd::_internal::computeDoubleStarState(starStateL, starStateR, magneticX, totalPressureStar, speed);

  // Compute and return L** fluxes
  // M&K 2005 equation 66
  if (speed.M > 0.0 and speed.LStar <= 0.0) {
    fluxL = mhd::_internal::computeDoubleStarFluxes(doubleStarState, doubleStarState.energyL, starStateL, stateL, fluxL,
                                                    speed, speed.L, speed.LStar);
    mhd::_internal::returnFluxes(threadId, o1, o2, o3, n_cells, dev_flux, fluxL, stateL);
    return;
  }
  // Compute and return R** fluxes
  // M&K 2005 equation 66
  if (speed.RStar > 0.0 and speed.M <= 0.0) {
    fluxR = mhd::_internal::computeDoubleStarFluxes(doubleStarState, doubleStarState.energyR, starStateR, stateR, fluxR,
                                                    speed, speed.R, speed.RStar);
    mhd::_internal::returnFluxes(threadId, o1, o2, o3, n_cells, dev_flux, fluxR, stateR);
    return;
  }
}
// =========================================================================

namespace _internal
{
// =====================================================================
__device__ __host__ mhd::_internal::State loadState(Real const *interfaceArr, Real const &magneticX, Real const &gamma,
                                                    int const &threadId, int const &n_cells, int const &o1,
                                                    int const &o2, int const &o3)
{
  mhd::_internal::State state;
  state.density   = interfaceArr[threadId + n_cells * grid_enum::density];
  state.density   = fmax(state.density, (Real)TINY_NUMBER);
  state.velocityX = interfaceArr[threadId + n_cells * o1] / state.density;
  state.velocityY = interfaceArr[threadId + n_cells * o2] / state.density;
  state.velocityZ = interfaceArr[threadId + n_cells * o3] / state.density;
  state.energy    = interfaceArr[threadId + n_cells * grid_enum::Energy];
  state.energy    = fmax(state.energy, (Real)TINY_NUMBER);
  state.magneticY = interfaceArr[threadId + n_cells * grid_enum::Q_x_magnetic_y];
  state.magneticZ = interfaceArr[threadId + n_cells * grid_enum::Q_x_magnetic_z];

    #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    state.scalarSpecific[i] = interfaceArr[threadId + n_cells * (grid_enum::scalar + i)] / state.density;
  }
    #endif  // SCALAR
    #ifdef DE
  state.thermalEnergySpecific = interfaceArr[threadId + n_cells * grid_enum::GasEnergy] / state.density;

  Real energyNonThermal = hydro_utilities::Calc_Kinetic_Energy_From_Velocity(state.density, state.velocityX,
                                                                             state.velocityY, state.velocityZ) +
                          mhd::utils::computeMagneticEnergy(magneticX, state.magneticY, state.magneticZ);

  state.gasPressure = fmax(hydro_utilities::Get_Pressure_From_DE(state.energy, state.energy - energyNonThermal,
                                                                 state.thermalEnergySpecific * state.density, gamma),
                           (Real)TINY_NUMBER);
    #else
  // Note that this function does the positive pressure check
  // internally
  state.gasPressure = mhd::_internal::Calc_Pressure_Primitive(state, magneticX, gamma);
    #endif  // DE

  state.totalPressure =
      mhd::utils::computeTotalPressure(state.gasPressure, magneticX, state.magneticY, state.magneticZ);

  return state;
}
// =====================================================================

// =====================================================================
__device__ __host__ mhd::_internal::Speeds approximateLRWaveSpeeds(mhd::_internal::State const &stateL,
                                                                   mhd::_internal::State const &stateR,
                                                                   Real const &magneticX, Real const &gamma)
{
  // Get the fast magnetosonic wave speeds
  Real magSonicL = mhd::utils::fastMagnetosonicSpeed(stateL.density, stateL.gasPressure, magneticX, stateL.magneticY,
                                                     stateL.magneticZ, gamma);
  Real magSonicR = mhd::utils::fastMagnetosonicSpeed(stateR.density, stateR.gasPressure, magneticX, stateR.magneticY,
                                                     stateR.magneticZ, gamma);

  // Compute the S_L and S_R wave speeds.
  // Version suggested by Miyoshi & Kusano 2005 and used in Athena
  // M&K 2005 equation 67
  Real magSonicMax = fmax(magSonicL, magSonicR);
  mhd::_internal::Speeds speed;
  speed.L = fmin(stateL.velocityX, stateR.velocityX) - magSonicMax;
  speed.R = fmax(stateL.velocityX, stateR.velocityX) + magSonicMax;

  return speed;
}
// =====================================================================

// =====================================================================
__device__ __host__ Real approximateMiddleWaveSpeed(mhd::_internal::State const &stateL,
                                                    mhd::_internal::State const &stateR,
                                                    mhd::_internal::Speeds const &speed)
{
  // Compute the S_M wave speed
  // M&K 2005 equation 38
  Real const speed_r_diff = speed.R - stateR.velocityX;
  Real const speed_l_diff = speed.L - stateL.velocityX;

  return  // Numerator
      (speed_r_diff * stateR.density * stateR.velocityX - speed_l_diff * stateL.density * stateL.velocityX -
       stateR.totalPressure + stateL.totalPressure) /
      // Denominator
      (speed_r_diff * stateR.density - speed_l_diff * stateL.density);
}
// =====================================================================

// =====================================================================
__device__ __host__ Real approximateStarWaveSpeed(mhd::_internal::StarState const &starState,
                                                  mhd::_internal::Speeds const &speed, Real const &magneticX,
                                                  Real const &side)
{
  // Compute the S_L^* and S_R^* wave speeds
  // M&K 2005 equation 51
  return speed.M + side * mhd::utils::alfvenSpeed(magneticX, starState.density);
}
// =====================================================================

// =====================================================================
__device__ __host__ mhd::_internal::Flux nonStarFluxes(mhd::_internal::State const &state, Real const &magneticX)
{
  mhd::_internal::Flux flux;
  // M&K 2005 equation 2
  flux.density = state.density * state.velocityX;

  flux.momentumX = flux.density * state.velocityX + state.totalPressure - magneticX * magneticX;
  flux.momentumY = flux.density * state.velocityY - magneticX * state.magneticY;
  flux.momentumZ = flux.density * state.velocityZ - magneticX * state.magneticZ;

  flux.magneticY = state.magneticY * state.velocityX - magneticX * state.velocityY;
  flux.magneticZ = state.magneticZ * state.velocityX - magneticX * state.velocityZ;

  // Group transverse terms for FP associative symmetry
  flux.energy = state.velocityX * (state.energy + state.totalPressure) -
                magneticX * (state.velocityX * magneticX +
                             ((state.velocityY * state.magneticY) + (state.velocityZ * state.magneticZ)));

  return flux;
}
// =====================================================================

// =====================================================================
__device__ __host__ void returnFluxes(int const &threadId, int const &o1, int const &o2, int const &o3,
                                      int const &n_cells, Real *dev_flux, mhd::_internal::Flux const &flux,
                                      mhd::_internal::State const &state)
{
  // Note that the direction of the grid_enum::fluxX_magnetic_DIR is the
  // direction of the electric field that the magnetic flux is, not the magnetic
  // flux
  dev_flux[threadId + n_cells * grid_enum::density]          = flux.density;
  dev_flux[threadId + n_cells * o1]                          = flux.momentumX;
  dev_flux[threadId + n_cells * o2]                          = flux.momentumY;
  dev_flux[threadId + n_cells * o3]                          = flux.momentumZ;
  dev_flux[threadId + n_cells * grid_enum::Energy]           = flux.energy;
  dev_flux[threadId + n_cells * grid_enum::fluxX_magnetic_z] = flux.magneticY;
  dev_flux[threadId + n_cells * grid_enum::fluxX_magnetic_y] = flux.magneticZ;

    #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_flux[threadId + n_cells * (grid_enum::scalar + i)] = state.scalarSpecific[i] * flux.density;
  }
    #endif  // SCALAR
    #ifdef DE
  dev_flux[threadId + n_cells * grid_enum::GasEnergy] = state.thermalEnergySpecific * flux.density;
    #endif  // DE
}
// =====================================================================

// =====================================================================
__device__ __host__ Real starTotalPressure(mhd::_internal::State const &stateL, mhd::_internal::State const &stateR,
                                           mhd::_internal::Speeds const &speed)
{
  // M&K 2005 equation 41
  return  // Numerator
      (stateR.density * stateL.totalPressure * (speed.R - stateR.velocityX) -
       stateL.density * stateR.totalPressure * (speed.L - stateL.velocityX) +
       stateL.density * stateR.density * (speed.R - stateR.velocityX) * (speed.L - stateL.velocityX) *
           (stateR.velocityX - stateL.velocityX)) /
      // Denominator
      (stateR.density * (speed.R - stateR.velocityX) - stateL.density * (speed.L - stateL.velocityX));
}
// =====================================================================

// =====================================================================
__device__ __host__ mhd::_internal::StarState computeStarState(mhd::_internal::State const &state,
                                                               mhd::_internal::Speeds const &speed,
                                                               Real const &speedSide, Real const &magneticX,
                                                               Real const &totalPressureStar)
{
  mhd::_internal::StarState starState;

  // Compute the densities in the star state
  // M&K 2005 equation 43
  starState.density = state.density * (speedSide - state.velocityX) / (speedSide - speed.M);

  // Check for and handle the degenerate case
  // Explained at the top of page 326 in M&K 2005
  if (fabs(state.density * (speedSide - state.velocityX) * (speedSide - speed.M) - (magneticX * magneticX)) <
      totalPressureStar * mhd::_internal::_hlldSmallNumber) {
    starState.velocityY = state.velocityY;
    starState.velocityZ = state.velocityZ;
    starState.magneticY = state.magneticY;
    starState.magneticZ = state.magneticZ;
  } else {
    // Denominator for M&K 2005 equations 44-47
    Real const denom = state.density * (speedSide - state.velocityX) * (speedSide - speed.M) - (magneticX * magneticX);

    // Compute the velocity and magnetic field in the star state
    // M&K 2005 equations 44 & 46
    Real coef           = magneticX * (speed.M - state.velocityX) / denom;
    starState.velocityY = state.velocityY - state.magneticY * coef;
    starState.velocityZ = state.velocityZ - state.magneticZ * coef;

    // M&K 2005 equations 45 & 47
    Real tmpPower       = (speedSide - state.velocityX);
    tmpPower            = tmpPower * tmpPower;
    coef                = (state.density * tmpPower - (magneticX * magneticX)) / denom;
    starState.magneticY = state.magneticY * coef;
    starState.magneticZ = state.magneticZ * coef;
  }

  // M&K 2005 equation 48
  starState.energy = (state.energy * (speedSide - state.velocityX) - state.totalPressure * state.velocityX +
                      totalPressureStar * speed.M +
                      magneticX * (math_utils::dotProduct(state.velocityX, state.velocityY, state.velocityZ, magneticX,
                                                          state.magneticY, state.magneticZ) -
                                   math_utils::dotProduct(speed.M, starState.velocityY, starState.velocityZ, magneticX,
                                                          starState.magneticY, starState.magneticZ))) /
                     (speedSide - speed.M);

  return starState;
}
// =====================================================================

// =====================================================================
__device__ __host__ mhd::_internal::Flux starFluxes(mhd::_internal::StarState const &starState,
                                                    mhd::_internal::State const &state,
                                                    mhd::_internal::Flux const &flux,
                                                    mhd::_internal::Speeds const &speed, Real const &speedSide)
{
  mhd::_internal::Flux starFlux;

  // Now compute the star state fluxes
  // M&K 2005 equations 64
  starFlux.density   = flux.density + speedSide * (starState.density - state.density);
  starFlux.momentumX = flux.momentumX + speedSide * (starState.density * speed.M - state.density * state.velocityX);
  starFlux.momentumY =
      flux.momentumY + speedSide * (starState.density * starState.velocityY - state.density * state.velocityY);
  starFlux.momentumZ =
      flux.momentumZ + speedSide * (starState.density * starState.velocityZ - state.density * state.velocityZ);
  starFlux.energy    = flux.energy + speedSide * (starState.energy - state.energy);
  starFlux.magneticY = flux.magneticY + speedSide * (starState.magneticY - state.magneticY);
  starFlux.magneticZ = flux.magneticZ + speedSide * (starState.magneticZ - state.magneticZ);

  return starFlux;
}
// =====================================================================

// =====================================================================
__device__ __host__ mhd::_internal::DoubleStarState computeDoubleStarState(mhd::_internal::StarState const &starStateL,
                                                                           mhd::_internal::StarState const &starStateR,
                                                                           Real const &magneticX,
                                                                           Real const &totalPressureStar,
                                                                           mhd::_internal::Speeds const &speed)
{
  mhd::_internal::DoubleStarState doubleStarState;

  // if Bx is zero then just return the star state
  // Explained at the top of page 328 in M&K 2005. Essentially when
  // magneticX is 0 this reduces to the HLLC solver
  if (0.5 * (magneticX * magneticX) < mhd::_internal::_hlldSmallNumber * totalPressureStar) {
    if (speed.M >= 0.0) {
      // We're in the L** state but Bx=0 so return L* state
      doubleStarState.velocityY = starStateL.velocityY;
      doubleStarState.velocityZ = starStateL.velocityZ;
      doubleStarState.magneticY = starStateL.magneticY;
      doubleStarState.magneticZ = starStateL.magneticZ;
      doubleStarState.energyL   = starStateL.energy;
    } else {
      // We're in the L** state but Bx=0 so return L* state
      doubleStarState.velocityY = starStateR.velocityY;
      doubleStarState.velocityZ = starStateR.velocityZ;
      doubleStarState.magneticY = starStateR.magneticY;
      doubleStarState.magneticZ = starStateR.magneticZ;
      doubleStarState.energyR   = starStateR.energy;
    }
  } else {
    // Setup some variables we'll need later
    Real sqrtDL           = sqrt(starStateL.density);
    Real sqrtDR           = sqrt(starStateR.density);
    Real inverseDensities = 1.0 / (sqrtDL + sqrtDR);
    Real magXSign         = copysign(1.0, magneticX);

    // All we need to do now is compute the transverse velocities
    // and magnetic fields along with the energy

    // Double Star velocities
    // M&K 2005 equations 59 & 60
    doubleStarState.velocityY = inverseDensities * (sqrtDL * starStateL.velocityY + sqrtDR * starStateR.velocityY +
                                                    magXSign * (starStateR.magneticY - starStateL.magneticY));
    doubleStarState.velocityZ = inverseDensities * (sqrtDL * starStateL.velocityZ + sqrtDR * starStateR.velocityZ +
                                                    magXSign * (starStateR.magneticZ - starStateL.magneticZ));

    // Double star magnetic fields
    // M&K 2005 equations 61 & 62
    doubleStarState.magneticY =
        inverseDensities * (sqrtDL * starStateR.magneticY + sqrtDR * starStateL.magneticY +
                            magXSign * (sqrtDL * sqrtDR) * (starStateR.velocityY - starStateL.velocityY));
    doubleStarState.magneticZ =
        inverseDensities * (sqrtDL * starStateR.magneticZ + sqrtDR * starStateL.magneticZ +
                            magXSign * (sqrtDL * sqrtDR) * (starStateR.velocityZ - starStateL.velocityZ));

    // Double star energy
    Real velDblStarDotMagDblStar =
        math_utils::dotProduct(speed.M, doubleStarState.velocityY, doubleStarState.velocityZ, magneticX,
                               doubleStarState.magneticY, doubleStarState.magneticZ);
    // M&K 2005 equation 63
    doubleStarState.energyL =
        starStateL.energy - sqrtDL * magXSign *
                                (math_utils::dotProduct(speed.M, starStateL.velocityY, starStateL.velocityZ, magneticX,
                                                        starStateL.magneticY, starStateL.magneticZ) -
                                 velDblStarDotMagDblStar);
    doubleStarState.energyR =
        starStateR.energy + sqrtDR * magXSign *
                                (math_utils::dotProduct(speed.M, starStateR.velocityY, starStateR.velocityZ, magneticX,
                                                        starStateR.magneticY, starStateR.magneticZ) -
                                 velDblStarDotMagDblStar);
  }

  return doubleStarState;
}
// =====================================================================

// =====================================================================
__device__ __host__ mhd::_internal::Flux computeDoubleStarFluxes(
    mhd::_internal::DoubleStarState const &doubleStarState, Real const &doubleStarStateEnergy,
    mhd::_internal::StarState const &starState, mhd::_internal::State const &state, mhd::_internal::Flux const &flux,
    mhd::_internal::Speeds const &speed, Real const &speedSide, Real const &speedSideStar)
{
  mhd::_internal::Flux doubleStarFlux;

  Real const speed_diff = speedSideStar - speedSide;

  // M&K 2005 equation 65
  doubleStarFlux.density =
      flux.density - speedSide * state.density - speed_diff * starState.density + speedSideStar * starState.density;

  doubleStarFlux.momentumX = flux.momentumX - speedSide * (state.density * state.velocityX) -
                             speed_diff * (starState.density * speed.M) + speedSideStar * (starState.density * speed.M);
  doubleStarFlux.momentumY = flux.momentumY - speedSide * (state.density * state.velocityY) -
                             speed_diff * (starState.density * starState.velocityY) +
                             speedSideStar * (starState.density * doubleStarState.velocityY);
  doubleStarFlux.momentumZ = flux.momentumZ - speedSide * (state.density * state.velocityZ) -
                             speed_diff * (starState.density * starState.velocityZ) +
                             speedSideStar * (starState.density * doubleStarState.velocityZ);
  doubleStarFlux.energy =
      flux.energy - speedSide * state.energy - speed_diff * starState.energy + speedSideStar * doubleStarStateEnergy;
  doubleStarFlux.magneticY = flux.magneticY - speedSide * state.magneticY - speed_diff * starState.magneticY +
                             speedSideStar * doubleStarState.magneticY;
  doubleStarFlux.magneticZ = flux.magneticZ - speedSide * state.magneticZ - speed_diff * starState.magneticZ +
                             speedSideStar * doubleStarState.magneticZ;

  return doubleStarFlux;
}
// =====================================================================

}  // namespace _internal
}  // end namespace mhd
  #endif  // MHD
#endif    // CUDA

/*!
 * \file hlld_cuda.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the implementation of the HLLD solver
 *
*/

// External Includes

// Local Includes
#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/mhd_utilities.h"
#include "../riemann_solvers/hlld_cuda.h"

#ifdef DE //PRESSURE_DE
    #include "../hydro/hydro_cuda.h"
#endif // DE

#ifdef CUDA
    // =========================================================================
    __global__ void Calculate_HLLD_Fluxes_CUDA(Real *dev_bounds_L,
                                               Real *dev_bounds_R,
                                               Real *dev_flux,
                                               int nx,
                                               int ny,
                                               int nz,
                                               int n_ghost,
                                               Real gamma,
                                               int direction,
                                               int n_fields)
    {
        // get a thread index
        int blockId  = blockIdx.x + blockIdx.y*gridDim.x;
        int threadId = threadIdx.x + blockId * blockDim.x;
        int zid = threadId / (nx*ny);
        int yid = (threadId - zid*nx*ny) / nx;
        int xid = threadId - zid*nx*ny - yid*nx;

        // Number of cells
        int n_cells = nx*ny*nz;

        // Offsets & indices
        int o1, o2, o3;
        if (direction==0) {o1 = 1; o2 = 2; o3 = 3;}
        if (direction==1) {o1 = 2; o2 = 3; o3 = 1;}
        if (direction==2) {o1 = 3; o2 = 1; o3 = 2;}

        // Thread guard to avoid overrun
        if (xid < nx and yid < ny and zid < nz)
        {
            // ============================
            // Retrieve conserved variables
            // ============================
            // Left interface
            Real densityL   = dev_bounds_L[threadId];
            Real momentumXL = dev_bounds_L[threadId + n_cells * o1];
            Real momentumYL = dev_bounds_L[threadId + n_cells * o2];
            Real momentumZL = dev_bounds_L[threadId + n_cells * o3];
            Real energyL    = dev_bounds_L[threadId + n_cells * 4];
            Real magneticXL = dev_bounds_L[threadId + n_cells * (o1 + 4 + NSCALARS)];
            Real magneticYL = dev_bounds_L[threadId + n_cells * (o2 + 4 + NSCALARS)];
            Real magneticZL = dev_bounds_L[threadId + n_cells * (o3 + 4 + NSCALARS)];

            #ifdef SCALAR
                Real scalarConservedL[NSCALARS];
                for (int i=0; i<NSCALARS; i++)
                {
                    scalarConservedL[i] = dev_bounds_L[threadId + n_cells * (5+i)];
                }
            #endif // SCALAR
            #ifdef DE
                Real thermalEnergyConservedL = dev_bounds_L[threadId + n_cells * (n_fields-1)];
            #endif // DE

            // Right interface
            Real densityR   = dev_bounds_R[threadId];
            Real momentumXR = dev_bounds_R[threadId + n_cells * o1];
            Real momentumYR = dev_bounds_R[threadId + n_cells * o2];
            Real momentumZR = dev_bounds_R[threadId + n_cells * o3];
            Real energyR    = dev_bounds_R[threadId + n_cells * 4];
            Real magneticXR = dev_bounds_R[threadId + n_cells * (o1 + 4 + NSCALARS)];
            Real magneticYR = dev_bounds_R[threadId + n_cells * (o2 + 4 + NSCALARS)];
            Real magneticZR = dev_bounds_R[threadId + n_cells * (o3 + 4 + NSCALARS)];

            #ifdef SCALAR
                Real scalarConservedR[NSCALARS];
                for (int i=0; i<NSCALARS; i++)
                {
                    scalarConservedR[i] = dev_bounds_R[threadId + n_cells * (5+i)];
                }
            #endif // SCALAR
            #ifdef DE
                Real thermalEnergyConservedR = dev_bounds_R[threadId + n_cells * (n_fields-1)];
            #endif // DE

            // Check for unphysical values
            densityL = fmax(densityL, (Real) TINY_NUMBER);
            densityR = fmax(densityR, (Real) TINY_NUMBER);
            energyL  = fmax(energyL,  (Real) TINY_NUMBER);
            energyR  = fmax(energyR,  (Real) TINY_NUMBER);

            // ============================
            // Compute primitive variables
            // ============================
            // Left interface
            Real const velocityXL = momentumXL / densityL;
            Real const velocityYL = momentumYL / densityL;
            Real const velocityZL = momentumZL / densityL;

            #ifdef DE //PRESSURE_DE
                Real const energyKineticL = 0.5 * densityL
                    * _hlldInternal::_dotProduct(velocityXL, velocityYL, velocityZL,
                                                 velocityXL, velocityYL, velocityZL);

                Real const energyMagneticL = 0.5
                    * _hlldInternal::_dotProduct(magneticXL, magneticYL, magneticZL,
                                                 magneticXL, magneticYL, magneticZL);

                Real const gasPressureL   = fmax(Get_Pressure_From_DE(energyL,
                                                                      energyL - energyKineticL - energyMagneticL,
                                                                      thermalEnergyConservedL,
                                                                      gamma),
                                                 (Real) TINY_NUMBER);
            #else
                // Note that this function does the positive pressure check
                // internally
                Real const gasPressureL  = mhdUtils::computeGasPressure(energyL,
                                                                        densityL,
                                                                        momentumXL,
                                                                        momentumYL,
                                                                        momentumZL,
                                                                        magneticXL,
                                                                        magneticYL,
                                                                        magneticZL,
                                                                        gamma);
            #endif //PRESSURE_DE

            Real const totalPressureL = mhdUtils::computeTotalPressure(gasPressureL,
                                                                       magneticXL,
                                                                       magneticYL,
                                                                       magneticZL);

            // Right interface
            Real const velocityXR = momentumXR / densityR;
            Real const velocityYR = momentumYR / densityR;
            Real const velocityZR = momentumZR / densityR;

            #ifdef DE //PRESSURE_DE
                Real const energyKineticR = 0.5 * densityR
                    * _hlldInternal::_dotProduct(velocityXR, velocityYR, velocityZR,
                                                 velocityXR, velocityYR, velocityZR);

                Real const energyMagneticR = 0.5
                    * _hlldInternal::_dotProduct(magneticXR, magneticYR, magneticZR,
                                                 magneticXR, magneticYR, magneticZR);

                Real const gasPressureR   = fmax(Get_Pressure_From_DE(energyR,
                                                                      energyR - energyKineticR - energyMagneticR,
                                                                      thermalEnergyConservedR,
                                                                      gamma),
                                                 (Real) TINY_NUMBER);
            #else
                // Note that this function does the positive pressure check
                // internally
                Real const gasPressureR  = mhdUtils::computeGasPressure(energyR,
                                                                  densityR,
                                                                  momentumXR,
                                                                  momentumYR,
                                                                  momentumZR,
                                                                  magneticXR,
                                                                  magneticYR,
                                                                  magneticZR,
                                                                  gamma);
            #endif //PRESSURE_DE

            Real const totalPressureR = mhdUtils::computeTotalPressure(gasPressureR,
                                                                 magneticXR,
                                                                 magneticYR,
                                                                 magneticZR);

            // Compute the approximate wave speeds and density in the star
            // regions
            Real speedL, speedR, speedM, speedStarL, speedStarR, densityStarL, densityStarR;
            _hlldInternal::_approximateWaveSpeeds(densityL,
                                                  momentumXL,
                                                  momentumYL,
                                                  momentumZL,
                                                  velocityXL,
                                                  velocityYL,
                                                  velocityZL,
                                                  gasPressureL,
                                                  totalPressureL,
                                                  magneticXL,
                                                  magneticYL,
                                                  magneticZL,
                                                  densityR,
                                                  momentumXR,
                                                  momentumYR,
                                                  momentumZR,
                                                  velocityXR,
                                                  velocityYR,
                                                  velocityZR,
                                                  gasPressureR,
                                                  totalPressureR,
                                                  magneticXR,
                                                  magneticYR,
                                                  magneticZR,
                                                  gamma,
                                                  speedL,
                                                  speedR,
                                                  speedM,
                                                  speedStarL,
                                                  speedStarR,
                                                  densityStarL,
                                                  densityStarR);

            // =================================================================
            // Compute the fluxes in the non-star states
            // =================================================================
            // Left state
            Real densityFluxL, momentumFluxXL, momentumFluxYL, momentumFluxZL,
                 magneticFluxYL, magneticFluxZL, energyFluxL;
            _hlldInternal::_nonStarFluxes(momentumXL,
                                          velocityXL,
                                          velocityYL,
                                          velocityZL,
                                          totalPressureL,
                                          energyL,
                                          magneticXL,
                                          magneticYL,
                                          magneticZL,
                                          densityFluxL,
                                          momentumFluxXL,
                                          momentumFluxYL,
                                          momentumFluxZL,
                                          magneticFluxYL,
                                          magneticFluxZL,
                                          energyFluxL);

            // If we're in the L state then assign fluxes and return.
            // In this state the flow is supersonic
            if (speedL >= 0.0)
            {
                _hlldInternal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityFluxL,
                                             momentumFluxXL, momentumFluxYL, momentumFluxZL,
                                             energyFluxL,
                                             magneticFluxYL, magneticFluxZL);
                #ifdef SCALAR
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId]  = (scalarConservedL[i] / densityL) * densityFluxL;
                    }
                #endif  // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedL / densityL) * densityFluxL;
                #endif  // DE
                return;
            }
            // Right state
            Real densityFluxR, momentumFluxXR, momentumFluxYR, momentumFluxZR,
                 magneticFluxYR, magneticFluxZR, energyFluxR;
            _hlldInternal::_nonStarFluxes(momentumXR,
                                          velocityXR,
                                          velocityYR,
                                          velocityZR,
                                          totalPressureR,
                                          energyR,
                                          magneticXR,
                                          magneticYR,
                                          magneticZR,
                                          densityFluxR,
                                          momentumFluxXR,
                                          momentumFluxYR,
                                          momentumFluxZR,
                                          magneticFluxYR,
                                          magneticFluxZR,
                                          energyFluxR);

            // If we're in the R state then assign fluxes and return.
            // In this state the flow is supersonic
            if (speedR <= 0.0)
            {
                _hlldInternal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityFluxR,
                                             momentumFluxXR, momentumFluxYR, momentumFluxZR,
                                             energyFluxR,
                                             magneticFluxYR, magneticFluxZR);
                #ifdef SCALAR
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId]  = (scalarConservedR[i] / densityR) * densityFluxR;
                    }
                #endif  // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedR / densityR) * densityFluxR;
                #endif  // DE
                return;
            }

            // =================================================================
            // Compute the fluxes in the star states
            // =================================================================
            // Shared quantity
            // note that velocityStarX = speedM
            Real totalPressureStar = totalPressureL + densityL
                                                      * (speedL - velocityXL)
                                                      * (speedM - velocityXL);

            // Left star state
            Real velocityStarYL, velocityStarZL,
                 energyStarL, magneticStarYL, magneticStarZL,
                 densityStarFluxL,
                 momentumStarFluxXL, momentumStarFluxYL, momentumStarFluxZL,
                 magneticStarFluxYL, magneticStarFluxZL, energyStarFluxL;
            _hlldInternal::_starFluxes(speedM,
                                       speedL,
                                       densityL,
                                       velocityXL,
                                       velocityYL,
                                       velocityZL,
                                       momentumXL,
                                       momentumYL,
                                       momentumZL,
                                       energyL,
                                       totalPressureL,
                                       magneticXL,
                                       magneticYL,
                                       magneticZL,
                                       densityStarL,
                                       totalPressureStar,
                                       densityFluxL,
                                       momentumFluxXL,
                                       momentumFluxYL,
                                       momentumFluxZL,
                                       energyFluxL,
                                       magneticFluxYL,
                                       magneticFluxZL,
                                       velocityStarYL,
                                       velocityStarZL,
                                       energyStarL,
                                       magneticStarYL,
                                       magneticStarZL,
                                       densityStarFluxL,
                                       momentumStarFluxXL,
                                       momentumStarFluxYL,
                                       momentumStarFluxZL,
                                       energyStarFluxL,
                                       magneticStarFluxYL,
                                       magneticStarFluxZL);

            // If we're in the L* state then assign fluxes and return.
            // In this state the flow is subsonic
            if (speedStarL >= 0.0)
            {
                _hlldInternal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityStarFluxL,
                                             momentumStarFluxXL, momentumStarFluxYL, momentumStarFluxZL,
                                             energyStarFluxL,
                                             magneticStarFluxYL, magneticStarFluxZL);
                #ifdef SCALAR
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId] = (scalarConservedL[i] / densityL) * densityStarFluxL;
                    }
                #endif  // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedL / densityL) * densityStarFluxL;
                #endif  // DE
                return;
            }

            // Right star state
            Real velocityStarYR, velocityStarZR,
                 energyStarR, magneticStarYR, magneticStarZR,
                 densityStarFluxR,
                 momentumStarFluxXR, momentumStarFluxYR, momentumStarFluxZR,
                 magneticStarFluxYR, magneticStarFluxZR, energyStarFluxR;
            _hlldInternal::_starFluxes(speedM,
                                       speedR,
                                       densityR,
                                       velocityXR,
                                       velocityYR,
                                       velocityZR,
                                       momentumXR,
                                       momentumYR,
                                       momentumZR,
                                       energyR,
                                       totalPressureR,
                                       magneticXR,
                                       magneticYR,
                                       magneticZR,
                                       densityStarR,
                                       totalPressureStar,
                                       densityFluxR,
                                       momentumFluxXR,
                                       momentumFluxYR,
                                       momentumFluxZR,
                                       energyFluxR,
                                       magneticFluxYR,
                                       magneticFluxZR,
                                       velocityStarYR,
                                       velocityStarZR,
                                       energyStarR,
                                       magneticStarYR,
                                       magneticStarZR,
                                       densityStarFluxR,
                                       momentumStarFluxXR,
                                       momentumStarFluxYR,
                                       momentumStarFluxZR,
                                       energyStarFluxR,
                                       magneticStarFluxYR,
                                       magneticStarFluxZR);

            // If we're in the R* state then assign fluxes and return.
            // In this state the flow is subsonic
            if (speedStarR <= 0.0)
            {
                _hlldInternal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityStarFluxR,
                                             momentumStarFluxXR, momentumStarFluxYR, momentumStarFluxZR,
                                             energyStarFluxR,
                                             magneticStarFluxYR, magneticStarFluxZR);
                #ifdef SCALAR
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId] = (scalarConservedR[i] / densityR) * densityStarFluxR;
                    }
                #endif  // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedR / densityR) * densityStarFluxR;
                #endif  // DE
                return;
            }

            // =================================================================
            // Compute the fluxes in the double star states
            // =================================================================
            Real velocityDoubleStarY, velocityDoubleStarZ,
                 magneticDoubleStarY, magneticDoubleStarZ,
                 energyDoubleStarL, energyDoubleStarR;
            _hlldInternal::_doubleStarState(speedM,
                                            magneticXL,
                                            totalPressureStar,
                                            densityStarL,
                                            velocityStarYL,
                                            velocityStarZL,
                                            energyStarL,
                                            magneticStarYL,
                                            magneticStarZL,
                                            densityStarR,
                                            velocityStarYR,
                                            velocityStarZR,
                                            energyStarR,
                                            magneticStarYR,
                                            magneticStarZR,
                                            velocityDoubleStarY,
                                            velocityDoubleStarZ,
                                            magneticDoubleStarY,
                                            magneticDoubleStarZ,
                                            energyDoubleStarL,
                                            energyDoubleStarR);

            // Compute and return L** fluxes
            if (speedM >= 0.0)
            {
                Real momentumDoubleStarFluxX, momentumDoubleStarFluxY, momentumDoubleStarFluxZ,
                     energyDoubleStarFlux,
                     magneticDoubleStarFluxY, magneticDoubleStarFluxZ;
                _hlldInternal::_doubleStarFluxes(speedStarL,
                                                 momentumStarFluxXL,
                                                 momentumStarFluxYL,
                                                 momentumStarFluxZL,
                                                 energyStarFluxL,
                                                 magneticStarFluxYL,
                                                 magneticStarFluxZL,
                                                 densityStarL,
                                                 speedM,
                                                 velocityStarYL,
                                                 velocityStarZL,
                                                 energyStarL,
                                                 magneticStarYL,
                                                 magneticStarZL,
                                                 speedM,
                                                 velocityDoubleStarY,
                                                 velocityDoubleStarZ,
                                                 energyDoubleStarL,
                                                 magneticDoubleStarY,
                                                 magneticDoubleStarZ,
                                                 momentumDoubleStarFluxX,
                                                 momentumDoubleStarFluxY,
                                                 momentumDoubleStarFluxZ,
                                                 energyDoubleStarFlux,
                                                 magneticDoubleStarFluxY,
                                                 magneticDoubleStarFluxZ);

                _hlldInternal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityStarFluxL,
                                             momentumDoubleStarFluxX, momentumDoubleStarFluxY, momentumDoubleStarFluxZ,
                                             energyDoubleStarFlux,
                                             magneticDoubleStarFluxY, magneticDoubleStarFluxZ);

                #ifdef SCALAR
                    // Return the passive scalar fluxes
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId] = (scalarConservedL[i] / densityL) * densityStarFluxL;
                    }
                #endif // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedL / densityL) * densityStarFluxL;
                #endif  // DE
                return;
            }
            // Compute and return R** fluxes
            else if (speedStarR >= 0.0)
            {
                Real momentumDoubleStarFluxX, momentumDoubleStarFluxY, momentumDoubleStarFluxZ,
                     energyDoubleStarFlux,
                     magneticDoubleStarFluxY, magneticDoubleStarFluxZ;
                _hlldInternal::_doubleStarFluxes(speedStarR,
                                                 momentumStarFluxXR,
                                                 momentumStarFluxYR,
                                                 momentumStarFluxZR,
                                                 energyStarFluxR,
                                                 magneticStarFluxYR,
                                                 magneticStarFluxZR,
                                                 densityStarR,
                                                 speedM,
                                                 velocityStarYR,
                                                 velocityStarZR,
                                                 energyStarR,
                                                 magneticStarYR,
                                                 magneticStarZR,
                                                 speedM,
                                                 velocityDoubleStarY,
                                                 velocityDoubleStarZ,
                                                 energyDoubleStarR,
                                                 magneticDoubleStarY,
                                                 magneticDoubleStarZ,
                                                 momentumDoubleStarFluxX,
                                                 momentumDoubleStarFluxY,
                                                 momentumDoubleStarFluxZ,
                                                 energyDoubleStarFlux,
                                                 magneticDoubleStarFluxY,
                                                 magneticDoubleStarFluxZ);

                _hlldInternal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityStarFluxR,
                                             momentumDoubleStarFluxX, momentumDoubleStarFluxY, momentumDoubleStarFluxZ,
                                             energyDoubleStarFlux,
                                             magneticDoubleStarFluxY, magneticDoubleStarFluxZ);

                #ifdef SCALAR
                    // Return the passive scalar fluxes
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId] = (scalarConservedR[i] / densityR) * densityStarFluxR;
                    }
                #endif // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedR / densityR) * densityStarFluxR;
                #endif  // DE
                return;
            }
        } // End thread guard
    };
    // =========================================================================

    namespace _hlldInternal
    {
        // =====================================================================
        __device__ __host__ void _approximateWaveSpeeds(Real const &densityL,
                                                        Real const &momentumXL,
                                                        Real const &momentumYL,
                                                        Real const &momentumZL,
                                                        Real const &velocityXL,
                                                        Real const &velocityYL,
                                                        Real const &velocityZL,
                                                        Real const &gasPressureL,
                                                        Real const &totalPressureL,
                                                        Real const &magneticXL,
                                                        Real const &magneticYL,
                                                        Real const &magneticZL,
                                                        Real const &densityR,
                                                        Real const &momentumXR,
                                                        Real const &momentumYR,
                                                        Real const &momentumZR,
                                                        Real const &velocityXR,
                                                        Real const &velocityYR,
                                                        Real const &velocityZR,
                                                        Real const &gasPressureR,
                                                        Real const &totalPressureR,
                                                        Real const &magneticXR,
                                                        Real const &magneticYR,
                                                        Real const &magneticZR,
                                                        Real const &gamma,
                                                        Real &speedL,
                                                        Real &speedR,
                                                        Real &speedM,
                                                        Real &speedStarL,
                                                        Real &speedStarR,
                                                        Real &densityStarL,
                                                        Real &densityStarR)
        {
            // Get the fast magnetosonic wave speeds
            Real magSonicL = mhdUtils::fastMagnetosonicSpeed(densityL,
                                                             gasPressureL,
                                                             magneticXL,
                                                             magneticYL,
                                                             magneticZL,
                                                             gamma);
            Real magSonicR = mhdUtils::fastMagnetosonicSpeed(densityR,
                                                             gasPressureR,
                                                             magneticXR,
                                                             magneticYR,
                                                             magneticZR,
                                                             gamma);

            // Compute the S_L and S_R wave speeds.
            // Version suggested by Miyoshi & Kusano 2005 and used in Athena
            Real magSonicMax = fmax(magSonicL, magSonicR);
            speedL = fmin(velocityXL, velocityXR) - magSonicMax;
            speedR = fmax(velocityXL, velocityXR) + magSonicMax;

            // Compute the S_M wave speed
            speedM = // Numerator
                          ( momentumXR * (speedR - velocityXR)
                          - momentumXL * (speedL - velocityXL)
                          + (totalPressureL - totalPressureR))
                          /
                          // Denominator
                          ( densityR * (speedR - velocityXR)
                          - densityL * (speedL - velocityXL));

            // Compute the densities in the star state
            densityStarL = densityL * (speedL - velocityXL) / (speedL - speedM);
            densityStarR = densityR * (speedR - velocityXR) / (speedR - speedM);

            // Compute the S_L^* and S_R^* wave speeds
            speedStarL = speedM - mhdUtils::alfvenSpeed(magneticXL, densityStarL);
            speedStarR = speedM + mhdUtils::alfvenSpeed(magneticXR, densityStarR);
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__ void _nonStarFluxes(Real const &momentumX,
                                                Real const &velocityX,
                                                Real const &velocityY,
                                                Real const &velocityZ,
                                                Real const &totalPressure,
                                                Real const &energy,
                                                Real const &magneticX,
                                                Real const &magneticY,
                                                Real const &magneticZ,
                                                Real &densityFlux,
                                                Real &momentumFluxX,
                                                Real &momentumFluxY,
                                                Real &momentumFluxZ,
                                                Real &magneticFluxY,
                                                Real &magneticFluxZ,
                                                Real &energyFlux)
        {
            densityFlux   = momentumX;

            momentumFluxX = momentumX * velocityX + totalPressure - magneticX * magneticX;
            momentumFluxY = momentumX * velocityY - magneticX * magneticY;
            momentumFluxZ = momentumX * velocityZ - magneticX * magneticZ;

            magneticFluxY = magneticY * velocityX - magneticX * velocityY;
            magneticFluxZ = magneticZ * velocityX - magneticX * velocityZ;

            // Group transverse terms for FP associative symmetry
            energyFlux    = velocityX * (energy + totalPressure) - magneticX
                            * (velocityX * magneticX
                               + ((velocityY * magneticY)
                               + (velocityZ * magneticZ)));
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__  void _returnFluxes(int const &threadId,
                                                int const &o1,
                                                int const &o2,
                                                int const &o3,
                                                int const &n_cells,
                                                Real *dev_flux,
                                                Real const &densityFlux,
                                                Real const &momentumFluxX,
                                                Real const &momentumFluxY,
                                                Real const &momentumFluxZ,
                                                Real const &energyFlux,
                                                Real const &magneticFluxY,
                                                Real const &magneticFluxZ)
        {
            dev_flux[threadId]                                 = densityFlux;
            dev_flux[threadId + n_cells * o1]                  = momentumFluxX;
            dev_flux[threadId + n_cells * o2]                  = momentumFluxY;
            dev_flux[threadId + n_cells * o3]                  = momentumFluxZ;
            dev_flux[threadId + n_cells * 4]                   = energyFlux;
            dev_flux[threadId + n_cells * (o2 + 4 + NSCALARS)] = magneticFluxY;
            dev_flux[threadId + n_cells * (o3 + 4 + NSCALARS)] = magneticFluxZ;
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__ void _starFluxes(Real const &speedM,
                                             Real const &speedSide,
                                             Real const &density,
                                             Real const &velocityX,
                                             Real const &velocityY,
                                             Real const &velocityZ,
                                             Real const &momentumX,
                                             Real const &momentumY,
                                             Real const &momentumZ,
                                             Real const &energy,
                                             Real const &totalPressure,
                                             Real const &magneticX,
                                             Real const &magneticY,
                                             Real const &magneticZ,
                                             Real const &densityStar,
                                             Real const &totalPressureStar,
                                             Real const &densityFlux,
                                             Real const &momentumFluxX,
                                             Real const &momentumFluxY,
                                             Real const &momentumFluxZ,
                                             Real const &energyFlux,
                                             Real const &magneticFluxY,
                                             Real const &magneticFluxZ,
                                             Real &velocityStarY,
                                             Real &velocityStarZ,
                                             Real &energyStar,
                                             Real &magneticStarY,
                                             Real &magneticStarZ,
                                             Real &densityStarFlux,
                                             Real &momentumStarFluxX,
                                             Real &momentumStarFluxY,
                                             Real &momentumStarFluxZ,
                                             Real &energyStarFlux,
                                             Real &magneticStarFluxY,
                                             Real &magneticStarFluxZ)
        {
            // Check for and handle the degenerate case
            if (fabs(density * (speedSide - velocityX)
                             * (speedSide - speedM)
                             - (magneticX * magneticX))
                < totalPressureStar * _hlldInternal::_hlldSmallNumber)
            {
                velocityStarY = velocityY;
                velocityStarZ = velocityZ;
                magneticStarY = magneticY;
                magneticStarZ = magneticZ;
            }
            else
            {
                Real const denom = density * (speedSide - velocityX)
                                           * (speedSide - speedM)
                                           - (magneticX * magneticX);

                // Compute the velocity and magnetic field in the star state
                Real coef     = magneticX  * (speedM - velocityX) / denom;
                velocityStarY = velocityY - magneticY * coef;
                velocityStarZ = velocityZ - magneticZ * coef;

                Real tmpPower = (speedSide - velocityX);
                tmpPower = tmpPower * tmpPower;
                coef = (density * tmpPower - (magneticX * magneticX)) / denom;
                magneticStarY = magneticY * coef;
                magneticStarZ = magneticZ * coef;
            }

            energyStar = ( energy * (speedSide - velocityX)
                        - totalPressure * velocityX
                        + totalPressureStar * speedM
                        + magneticX * (_hlldInternal::_dotProduct(velocityX, velocityY, velocityZ, magneticX, magneticY, magneticZ)
                                     - _hlldInternal::_dotProduct(speedM, velocityStarY, velocityStarZ, magneticX, magneticStarY, magneticStarZ)))
                        / (speedSide - speedM);

            // Now compute the star state fluxes
            densityStarFlux   = densityFlux   + speedSide * (densityStar - density);;
            momentumStarFluxX = momentumFluxX + speedSide * (densityStar * speedM - momentumX);;
            momentumStarFluxY = momentumFluxY + speedSide * (densityStar * velocityStarY - momentumY);;
            momentumStarFluxZ = momentumFluxZ + speedSide * (densityStar * velocityStarZ - momentumZ);;
            energyStarFlux    = energyFlux    + speedSide * (energyStar  - energy);
            magneticStarFluxY = magneticFluxY + speedSide * (magneticStarY - magneticY);
            magneticStarFluxZ = magneticFluxZ + speedSide * (magneticStarZ - magneticZ);
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__ void _doubleStarState(Real const &speedM,
                                                  Real const &magneticX,
                                                  Real const &totalPressureStar,
                                                  Real const &densityStarL,
                                                  Real const &velocityStarYL,
                                                  Real const &velocityStarZL,
                                                  Real const &energyStarL,
                                                  Real const &magneticStarYL,
                                                  Real const &magneticStarZL,
                                                  Real const &densityStarR,
                                                  Real const &velocityStarYR,
                                                  Real const &velocityStarZR,
                                                  Real const &energyStarR,
                                                  Real const &magneticStarYR,
                                                  Real const &magneticStarZR,
                                                  Real &velocityDoubleStarY,
                                                  Real &velocityDoubleStarZ,
                                                  Real &magneticDoubleStarY,
                                                  Real &magneticDoubleStarZ,
                                                  Real &energyDoubleStarL,
                                                  Real &energyDoubleStarR)
        {
            // if Bx is zero then just return the star state
            if (magneticX < _hlldInternal::_hlldSmallNumber * totalPressureStar)
            {
                velocityDoubleStarY = velocityStarYL;
                velocityDoubleStarZ = velocityStarZL;
                magneticDoubleStarY = magneticStarYL;
                magneticDoubleStarZ = magneticStarZL;
                energyDoubleStarL    = energyStarL;
                energyDoubleStarR    = energyStarR;
            }
            else
            {
                // Setup some variables we'll need later
                Real sqrtDL = sqrt(densityStarL);
                Real sqrtDR = sqrt(densityStarR);
                Real inverseDensities = 1.0 / (sqrtDL + sqrtDR);
                Real magXSign = copysign(1.0, magneticX);

                // All we need to do now is compute the transverse velocities
                // and magnetic fields along with the energy

                // Double Star velocities
                velocityDoubleStarY = inverseDensities * (sqrtDL * velocityStarYL
                                      + sqrtDR * velocityStarYR
                                      + magXSign * (magneticStarYR - magneticStarYL));
                velocityDoubleStarZ = inverseDensities * (sqrtDL * velocityStarZL
                                      + sqrtDR * velocityStarZR
                                      + magXSign * (magneticStarZR - magneticStarZL));

                // Double star magnetic fields
                magneticDoubleStarY = inverseDensities * (sqrtDL * magneticStarYR
                                      + sqrtDR * magneticStarYL
                                      + magXSign * (sqrtDL * sqrtDR) * (velocityStarYR - velocityStarYL));
                magneticDoubleStarZ = inverseDensities * (sqrtDL * magneticStarZR
                                      + sqrtDR * magneticStarZL
                                      + magXSign * (sqrtDL * sqrtDR) * (velocityStarZR - velocityStarZL));

                // Double star energy
                Real velDblStarDotMagDblStar = _hlldInternal::_dotProduct(speedM,
                                                                          velocityDoubleStarY,
                                                                          velocityDoubleStarZ,
                                                                          magneticX,
                                                                          magneticDoubleStarY,
                                                                          magneticDoubleStarZ);
                energyDoubleStarL = energyStarL - sqrtDL * magXSign
                    * (_hlldInternal::_dotProduct(speedM, velocityStarYL, velocityStarZL, magneticX, magneticStarYL, magneticStarZL)
                    - velDblStarDotMagDblStar);
                energyDoubleStarR = energyStarR + sqrtDR * magXSign
                    * (_hlldInternal::_dotProduct(speedM, velocityStarYR, velocityStarZR, magneticX, magneticStarYR, magneticStarZR)
                    - velDblStarDotMagDblStar);
            }
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__ void _doubleStarFluxes(Real const &speedStarSide,
                                                   Real const &momentumStarFluxX,
                                                   Real const &momentumStarFluxY,
                                                   Real const &momentumStarFluxZ,
                                                   Real const &energyStarFlux,
                                                   Real const &magneticStarFluxY,
                                                   Real const &magneticStarFluxZ,
                                                   Real const &densityStar,
                                                   Real const &velocityStarX,
                                                   Real const &velocityStarY,
                                                   Real const &velocityStarZ,
                                                   Real const &energyStar,
                                                   Real const &magneticStarY,
                                                   Real const &magneticStarZ,
                                                   Real const &velocityDoubleStarX,
                                                   Real const &velocityDoubleStarY,
                                                   Real const &velocityDoubleStarZ,
                                                   Real const &energyDoubleStar,
                                                   Real const &magneticDoubleStarY,
                                                   Real const &magneticDoubleStarZ,
                                                   Real &momentumDoubleStarFluxX,
                                                   Real &momentumDoubleStarFluxY,
                                                   Real &momentumDoubleStarFluxZ,
                                                   Real &energyDoubleStarFlux,
                                                   Real &magneticDoubleStarFluxY,
                                                   Real &magneticDoubleStarFluxZ)
        {
            momentumDoubleStarFluxX = momentumStarFluxX + speedStarSide * (velocityDoubleStarX - velocityStarX) * densityStar;
            momentumDoubleStarFluxY = momentumStarFluxY + speedStarSide * (velocityDoubleStarY - velocityStarY) * densityStar;
            momentumDoubleStarFluxZ = momentumStarFluxZ + speedStarSide * (velocityDoubleStarZ - velocityStarZ) * densityStar;
            energyDoubleStarFlux    = energyStarFlux    + speedStarSide * (energyDoubleStar    - energyStar);
            magneticDoubleStarFluxY = magneticStarFluxY + speedStarSide * (magneticDoubleStarY - magneticStarY);
            magneticDoubleStarFluxZ = magneticStarFluxZ + speedStarSide * (magneticDoubleStarZ - magneticStarZ);
        }
        // =====================================================================

    } // _hlldInternal namespace


#endif // CUDA
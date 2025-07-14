/*! \file hydro_cuda.h
 *  \brief Declarations of functions used in all cuda integration algorithms. */

#ifndef HYDRO_CUDA_H
#define HYDRO_CUDA_H

#include "../global/global.h"
#include "../hydro/average_cells.h"
#include "../utils/mhd_utilities.h"

__global__ void Update_Conserved_Variables_1D(Real *dev_conserved, Real *dev_F, int n_cells, int x_off, int n_ghost,
                                              Real dx, Real xbound, Real dt, Real gamma, int n_fields, int custom_grav);

__global__ void Update_Conserved_Variables_2D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny,
                                              int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound,
                                              Real ybound, Real dt, Real gamma, int n_fields, int custom_grav);

__global__ void Update_Conserved_Variables_3D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly, Real *Q_Ry,
                                              Real *Q_Lz, Real *Q_Rz, Real *dev_F_x, Real *dev_F_y, Real *dev_F_z,
                                              int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost,
                                              Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt,
                                              Real gamma, int n_fields, int custom_grav, Real density_floor,
                                              Real *dev_potential);

__global__ void PostUpdate_Conserved_Correct_Crashed_3D(Real *dev_conserved, int nx, int ny, int nz, int x_off,
                                                        int y_off, int z_off, int n_ghost, Real gamma, int n_fields,
                                                        SlowCellConditionChecker slow_check, int *any_error);

/*!
 * \brief Determine the maximum inverse crossing time in a specific cell
 *
 * \param[in] E The energy
 * \param[in] d The density
 * \param[in] d_inv The inverse density
 * \param[in] vx The velocity in the x-direction
 * \param[in] vy The velocity in the y-direction
 * \param[in] vz The velocity in the z-direction
 * \param[in] dx The size of each cell in the x-direction
 * \param[in] dy The size of each cell in the y-direction
 * \param[in] dz The size of each cell in the z-direction
 * \param[in] gamma The adiabatic index
 * \return Real The maximum inverse crossing time in the cell
 */
__device__ __host__ Real hydroInverseCrossingTime(Real const &E, Real const &d, Real const &d_inv, Real const &vx,
                                                  Real const &vy, Real const &vz, Real const &dx, Real const &dy,
                                                  Real const &dz, Real const &gamma);

/*!
 * \brief Determine the maximum inverse crossing time in a specific cell
 *
 * \param[in] E The energy
 * \param[in] d The density
 * \param[in] d_inv The inverse density
 * \param[in] vx The velocity in the x-direction
 * \param[in] vy The velocity in the y-direction
 * \param[in] vz The velocity in the z-direction
 * \param[in] avgBx The cell centered magnetic field in the x-direction
 * \param[in] avgBy The cell centered magnetic field in the y-direction
 * \param[in] avgBz The cell centered magnetic field in the z-direction
 * \param[in] dx The size of each cell in the x-direction
 * \param[in] dy The size of each cell in the y-direction
 * \param[in] dz The size of each cell in the z-direction
 * \param[in] gamma The adiabatic index
 * \return Real The maximum inverse crossing time in the cell
 */
__device__ __host__ Real mhdInverseCrossingTime(Real const &E, Real const &d, Real const &d_inv, Real const &vx,
                                                Real const &vy, Real const &vz, Real const &avgBx, Real const &avgBy,
                                                Real const &avgBz, Real const &dx, Real const &dy, Real const &dz,
                                                Real const &gamma);

__global__ void Calc_dt_3D(Real *dev_conserved, Real *dev_dti, Real gamma, int n_ghost, int n_fields, int nx, int ny,
                           int nz, Real dx, Real dy, Real dz);

Real Calc_dt_GPU(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy, Real dz,
                 Real gamma);

__global__ void Sync_Energies_1D(Real *dev_conserved, int nx, int n_ghost, Real gamma, int n_fields);

__global__ void Sync_Energies_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real gamma, int n_fields);

__global__ void Sync_Energies_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real gamma, int n_fields);

#ifdef TEMPERATURE_CEILING
void Temperature_Ceiling(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real gamma,
                         Real T_ceiling);
#endif  // TEMPERATURE CEILING

void Apply_Temperature_Floor(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real U_floor);

__global__ void Temperature_Floor_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                                         Real U_floor, int *counter);

void Apply_Scalar_Floor(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int field_num, Real scalar_floor);

__global__ void Scalar_Floor_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int field_num,
                                    Real scalar_floor);

__global__ void Partial_Update_Advected_Internal_Energy_1D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, int nx,
                                                           int n_ghost, Real dx, Real dt, Real gamma, int n_fields);

__global__ void Partial_Update_Advected_Internal_Energy_2D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly,
                                                           Real *Q_Ry, int nx, int ny, int n_ghost, Real dx, Real dy,
                                                           Real dt, Real gamma, int n_fields);

__global__ void Partial_Update_Advected_Internal_Energy_3D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly,
                                                           Real *Q_Ry, Real *Q_Lz, Real *Q_Rz, int nx, int ny, int nz,
                                                           int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma,
                                                           int n_fields);

__global__ void Select_Internal_Energy_1D(Real *dev_conserved, int nx, int n_ghost, int n_fields);

__global__ void Select_Internal_Energy_2D(Real *dev_conserved, int nx, int ny, int n_ghost, int n_fields);

__global__ void Select_Internal_Energy_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields);

/*! \brief Overwrites the values in the specified cell with the average of all the values from the (up to) 26
 *  neighboring cells.
 *
 *  Care is taken when applying this logic to a cell near the edge of a block (where the entire simulation domain
 *  is decomposed into 1 or more blocks).
 *    * Recall that the entire reason we have ghost zones is that the stencil for computing flux-divergence can't
 *      be applied uniformly to all cells -- the cells in the ghost zone can't be properly updated with the rest
 *      the local block when applying the flux-divergence. We might refer to these cells that aren't properly
 *      updated as being "stale". We refer to the width of the outer ring of stale values as the ``stale-depth``
 *    * For concreteness, consider a pure hydro/mhd simulation using the VL integrator:
 *       - Right after refreshing the ghost-zones, the stale_depth is 0
 *       - After the partial time-step, the stale_depth is 1.
 *       - After the full timestep, the stale depth depends on the choice of reconstruction. (e.g. it is 2 for
 *         for nearest neighbor and 3 for plmp).
 *       - The ghost-depth should always be equal to the max stale-depth at the end of a simulation cycle (if
 *         ghost-depth is bigger, extra work is done. If it's smaller, then your simulation is wrong)
 *    * To respect the simulations boundaries, values in "stale" cells are excluded from the averages. If
 *      stale-depth is 0, then values from beyond the edge of the simulation are excluded from averages
 *
 *  \pre When using the dual-energy formalism, this function should **ONLY** be called in contexts where all
 *    "non-crashed" cells are known to have *reconciled* "advected internal energy" and "total energy" field values.
 *    This is important for averaging.
 *        - In practice, this limitation is mostly meaningful inside of a hydro solver
 *        - Outside of a hydro-solver, you can definitely come up with pathological cases where applying source terms,
 *          but you would need to go out of your way to do something to violate this precondition
 *
 *  \returns A value of ``true`` indicates that the operation succeeded. A value of ``false`` indicates a failure.
 *    To succeed, this function requires that there are at least 2 neighboring cells with valid values that can be
 *    used to compute the average.
 *
 *  \note
 *  From a perfectionist's perspective, one could argue that we really should increment the stale-depth whenever
 *  we call this function (in other words, we should have an extra layer of ghost zones for each time we call
 *  this function).
 *    * rationale: if we don't, then the the number of neighbors considered results of the simulation can vary
 *      based on how close a cell is to a block-edge (the number of cells varies from 7 to 26).
 *    * more pragmatically: this probably doesn't matter a whole lot given that this piece of machinery is a
 *      band-aid solution to begin with.
 *    * Aside: a similar argument could be made for the energy-synchronization step of the dual-energy formalism.
 */
__device__ bool Average_Cell_All_Fields(int i, int j, int k, int nx, int ny, int nz, int ncells, int n_fields,
                                        Real gamma, Real *conserved, int stale_depth,
                                        SlowCellConditionChecker slow_check);

__device__ Real Average_Cell_Single_Field(int field_indx, int i, int j, int k, int nx, int ny, int nz, int ncells,
                                          Real *conserved);

#endif  // HYDRO_CUDA_H

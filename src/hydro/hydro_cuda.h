/*! \file hydro_cuda.h
 *  \brief Declarations of functions used in all cuda integration algorithms. */

#ifndef HYDRO_CUDA_H
#define HYDRO_CUDA_H

#include "../global/global.h"
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

#ifdef AVERAGE_SLOW_CELLS

void Average_Slow_Cells(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy,
                        Real dz, Real gamma, Real max_dti_slow, Real xbound, Real ybound, Real zbound, int nx_offset,
                        int ny_offset, int nz_offset);

__global__ void Average_Slow_Cells_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx,
                                      Real dy, Real dz, Real gamma, Real max_dti_slow, Real xbound, Real ybound,
                                      Real zbound, int nx_offset, int ny_offset, int nz_offset);
#endif

void Apply_Temperature_Floor(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real U_floor);

__global__ void Temperature_Floor_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                                         Real U_floor);

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

__device__ void Average_Cell_All_Fields(int i, int j, int k, int nx, int ny, int nz, int ncells, int n_fields,
                                        Real gamma, Real *conserved);

__device__ Real Average_Cell_Single_Field(int field_indx, int i, int j, int k, int nx, int ny, int nz, int ncells,
                                          Real *conserved);

#endif  // HYDRO_CUDA_H

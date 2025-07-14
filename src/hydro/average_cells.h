/*! \file average_cells.h
 *  \brief Definitions of functions and classes that implement logic related to averaging cells with
 *         neighbors. */

#ifndef AVERAGE_CELLS_H
#define AVERAGE_CELLS_H

#include <math.h>

#include "../global/global.h"

/*! \brief Object that checks whether a given cell meets the conditions for slow-cell averaging.
 *          The main motivation for creating this class is reducing ifdef statements (and allow to modify the
 *          actual slow-cell-condition. */
struct SlowCellConditionChecker {
// omit data-members if they aren't used for anything
#ifdef AVERAGE_SLOW_CELLS
  Real max_dti_slow, dx, dy, dz;
#endif

  /*! \brief Construct a new object. */
  __host__ __device__ SlowCellConditionChecker(Real max_dti_slow, Real dx, Real dy, Real dz)
#ifdef AVERAGE_SLOW_CELLS
      : max_dti_slow{max_dti_slow}, dx{dx}, dy{dy}, dz{dz}
#endif
  {
  }

  /*! \brief Returns whether the cell meets the condition for being considered a slow cell that must
   *  be averaged. */
  template <bool verbose = false>
  __device__ bool is_slow(Real total_energy, Real density, Real density_inv, Real velocity_x, Real velocity_y,
                          Real velocity_z, Real gamma) const
  {
    return this->max_dti_if_slow(total_energy, density, density_inv, velocity_x, velocity_y, velocity_z, gamma) > 0.0;
  }

  /*! \brief Returns the max inverse timestep of the specified cell, if it meets the criteria for being
   *  a slow cell. If it doesn't, return a negative value instead.
   */
  __device__ Real max_dti_if_slow(Real total_energy, Real density, Real density_inv, Real velocity_x, Real velocity_y,
                                  Real velocity_z, Real gamma) const;
};

#ifdef AVERAGE_SLOW_CELLS

void Average_Slow_Cells(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real gamma,
                        SlowCellConditionChecker slow_check, Real xbound, Real ybound, Real zbound, int nx_offset,
                        int ny_offset, int nz_offset);

__global__ void Average_Slow_Cells_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                                      Real gamma, SlowCellConditionChecker slow_check, Real xbound, Real ybound,
                                      Real zbound, int nx_offset, int ny_offset, int nz_offset);
#endif

#endif /* AVERAGE_CELLS_H */
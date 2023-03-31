/*!
 * \file dust_cuda.h
 * \author Helena Richie (helenarichie@pitt.edu)
 * \brief Contains the declaration for the kernel that updates the dust density scalar in dev_conserved.
 *
 */

#ifdef CUDA
  #ifdef DUST

    #ifndef DUST_CUDA_H
      #define DUST_CUDA_H

      #include <math.h>

      #include "../global/global.h"
      #include "../utils/gpu.hpp"

/*!
 * \brief Launch the dust kernel.
 *
 * \param[in] dev_conserved The device conserved variable array.
 * \param[in] nx Number of cells in the x-direction
 * \param[in] ny Number of cells in the y-direction
 * \param[in] nz Number of cells in the z-direction
 * \param[in] n_ghost Number of ghost cells
 * \param[in] n_fields
 * \param[in] dt Simulation timestep
 * \param[in] gamma Specific heat ratio
 *
 * \return None
 */
void Dust_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma);

/*!
 * \brief Compute the change in dust density for a cell and update its value in dev_conserved.
 *
 * \param[out] dev_conserved The device conserved variable array. The dust field is updated in this function. If dual
 * energy is turned on, then the dual energy field is updated, as well.
 * \param[in] nx Number of cells in the x-direction
 * \param[in] ny Number of cells in the y-direction
 * \param[in] nz Number of cells in the z-direction
 * \param[in] n_ghost Number of ghost cells
 * \param[in] n_fields
 * \param[in] dt Simulation timestep
 * \param[in] gamma Specific heat ratio
 *
 * \return None
 */
__global__ void Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                            Real gamma);

/*!
 * \brief Compute the sputtering timescale based on a cell's density and temperature.
 *
 * \param[in] n Gas number density in cm^-3
 * \param[in] T Gas temperature in K
 *
 * \return Real Sputtering timescale in seconds (McKinnon et al. 2017)
 */
__device__ __host__ Real calc_tau_sp(Real n, Real T);

/*!
 * \brief Compute the rate of change in dust density based on the current dust density and sputtering timescale.
 *
 * \param[in] d_dust Dust mass density in M_sun/kpc^3
 * \param[in] tau_sp Sputtering timescale in kyr
 *
 * \return Real Dust density rate of change (McKinnon et al. 2017)
 */
__device__ __host__ Real calc_dd_dt(Real d_dust, Real tau_sp);

    #endif  // DUST
  #endif    // CUDA
#endif      // DUST_CUDA_H
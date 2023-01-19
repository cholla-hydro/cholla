/*!
 * \file ct_electric_fields.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declaration for the kernel that computes the CT electric
 * fields. Method from Stone & Gardiner 2009 "A simple unsplit Godunov method
 * for multidimensional MHD" hereafter referred to as "S&G 2009"
 *
 */

#pragma once

// STL Includes

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"

#ifdef MHD
namespace mhd
{
/*!
 * \brief Namespace for functions required by functions within the mhd
 * namespace. Everything in this name space should be regarded as private
 * but is made accesible for testing
 *
 */
namespace _internal
{
// =====================================================================
/*!
 * \brief Compute and return the slope of the electric field used to
 * compute the CT electric fields. This function implements S&G 2009
 * equation 24
 *
 * \param[in] flux The flux array
 * \param[in] dev_conserved The conserved variable array
 * \param[in] fluxSign The sign of the flux to convert it to magnetic
 * field. Also serves to choose which magnetic flux is used, i.e. the Y
 * or Z field
 * \param[in] ctDirection The direction of the CT field that this slope
   will be used to calculate
 * \param[in] conservedQuadrent1 Which index should be reduced by one to get the
 correct conserved variables. Options are -1 for no reduction, 0 for reducing
 xid, 1 for reducing yid, and 2 for reducing zid
 * \param[in] conservedQuadrent2 Which index should be reduced by one to get the
 correct conserved variables. Options are -1 for no reduction, 0 for reducing
 xid, 1 for reducing yid, and 2 for reducing zid
 * \param[in] fluxQuadrent1 Which index should be reduced by one to get the
 correct flux variable. Options are -1 for no reduction, 0 for reducing xid, 1
 for reducing yid, and 2 for reducing zid
 * \param[in] fluxQuadrent2 Which index should be reduced by one to get the
 correct flux variable. Options are -1 for no reduction, 0 for reducing xid, 1
 for reducing yid, and 2 for reducing zid
 * \param[in] xid The x index
 * \param[in] yid The y index
 * \param[in] zid The z index
 * \param[in] nx The number of cells in the x-direction
 * \param[in] ny The number of cells in the y-direction
 * \param[in] n_cells The total number of cells
 * \return Real The slope of the electric field
 */
inline __host__ __device__ Real
_ctSlope(Real const *flux, Real const *dev_conserved, Real const &fluxSign,
         int const &ctDirection, int const &conservedQuadrent1,
         int const &conservedQuadrent2, int const &fluxQuadrent1,
         int const &fluxQuadrent2, int const &xid, int const &yid,
         int const &zid, int const &nx, int const &ny, int const &n_cells)
{
  // Compute the various required indices

  // Get the shifted modulos of the ctDirection.
  int const modPlus1 = (ctDirection + 1) % 3;
  int const modPlus2 = (ctDirection + 2) % 3;

  // Indices for the cell centered values
  int const xidCentered =
      xid - int(conservedQuadrent1 == 0) - int(conservedQuadrent2 == 0);
  int const yidCentered =
      yid - int(conservedQuadrent1 == 1) - int(conservedQuadrent2 == 1);
  int const zidCentered =
      zid - int(conservedQuadrent1 == 2) - int(conservedQuadrent2 == 2);
  int const idxCentered = cuda_utilities::compute1DIndex(
      xidCentered, yidCentered, zidCentered, nx, ny);

  // Index for the flux
  int const idxFlux = cuda_utilities::compute1DIndex(
      xid - int(fluxQuadrent1 == 0) - int(fluxQuadrent2 == 0),
      yid - int(fluxQuadrent1 == 1) - int(fluxQuadrent2 == 1),
      zid - int(fluxQuadrent1 == 2) - int(fluxQuadrent2 == 2), nx, ny);

  // Indices for the face centered magnetic fields that need to be averaged
  int const idxB2Shift = cuda_utilities::compute1DIndex(
      xidCentered - int(modPlus1 == 0), yidCentered - int(modPlus1 == 1),
      zidCentered - int(modPlus1 == 2), nx, ny);
  int const idxB3Shift = cuda_utilities::compute1DIndex(
      xidCentered - int(modPlus2 == 0), yidCentered - int(modPlus2 == 1),
      zidCentered - int(modPlus2 == 2), nx, ny);

  // Load values for cell centered electric field. B1 (not present) is
  // the magnetic field in the same direction as the `ctDirection`
  // variable, B2 and B3 are the next two fields cyclically. i.e. if
  // B1=Bx then B2=By and B3=Bz, if B1=By then B2=Bz and B3=Bx. The
  // same rules apply for the momentum
  Real const density   = dev_conserved[idxCentered];
  Real const Momentum2 = dev_conserved[idxCentered + (modPlus1 + 1) * n_cells];
  Real const Momentum3 = dev_conserved[idxCentered + (modPlus2 + 1) * n_cells];
  Real const B2Centered =
      0.5 * (dev_conserved[idxCentered +
                           (modPlus1 + grid_enum::magnetic_start) * n_cells] +
             dev_conserved[idxB2Shift +
                           (modPlus1 + grid_enum::magnetic_start) * n_cells]);
  Real const B3Centered =
      0.5 * (dev_conserved[idxCentered +
                           (modPlus2 + grid_enum::magnetic_start) * n_cells] +
             dev_conserved[idxB3Shift +
                           (modPlus2 + grid_enum::magnetic_start) * n_cells]);

  // Compute the electric field in the center with a cross product
  Real const electric_centered =
      (Momentum3 * B2Centered - Momentum2 * B3Centered) / density;

  // Load face centered electric field, note fluxSign to correctly do
  // the shift from magnetic flux to EMF/electric field and to choose
  // which field to use
  Real const electric_face =
      fluxSign *
      flux[idxFlux +
           (int(fluxSign == 1) + grid_enum::magnetic_start) * n_cells];

  // Compute the slope and return it
  // S&G 2009 equation 24
  return electric_face - electric_centered;
}
// =====================================================================
}  // namespace _internal

// =========================================================================
/*!
 * \brief Compute the Constrained Transport electric fields used to evolve
 * the magnetic field. Note that this function requires that the density be
 * non-zero or it will return Nans.
 *
 * \param[in] fluxX The flux on the x+1/2 face of each cell
 * \param[in] fluxY The flux on the y+1/2 face of each cell
 * \param[in] fluxZ The flux on the z+1/2 face of each cell
 * \param[in] dev_conserved The device resident grid
 * \param[out] ctElectricFields The CT electric fields
 * \param[in] nx The number of cells in the x-direction
 * \param[in] ny The number of cells in the y-direction
 * \param[in] nz The number of cells in the z-direction
 * \param[in] n_cells The total number of cells
 */
__global__ void Calculate_CT_Electric_Fields(
    Real const *fluxX, Real const *fluxY, Real const *fluxZ,
    Real const *dev_conserved, Real *ctElectricFields, int const nx,
    int const ny, int const nz, int const n_cells);
// =========================================================================
}  // end  namespace mhd
#endif  // MHD
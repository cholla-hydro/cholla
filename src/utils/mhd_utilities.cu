/*!
 * \file mhd_utilities.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the implementation of various utility functions for MHD and
 * for the various kernels, functions, and tools required for the 3D VL+CT MHD
 * integrator. Due to the CUDA/HIP compiler requiring that device functions be
 * directly accessible to the file they're used in most device functions will be
 * implemented in the header file
 *
 */

// STL Includes

// External Includes

// Local Includes
#include "../utils/mhd_utilities.h"

namespace mhd::utils
{
#ifdef MHD
void Init_Magnetic_Field_With_Vector_Potential(Header const &H, Grid3D::Conserved const &C,
                                               std::vector<Real> const &vectorPotential)
{
  // Compute the magnetic field
  for (size_t k = 1; k < H.nz; k++) {
    for (size_t j = 1; j < H.ny; j++) {
      for (size_t i = 1; i < H.nx; i++) {
        // Get cell index. The "xmo" means: X direction Minus One
        size_t const id    = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
        size_t const idxmo = cuda_utilities::compute1DIndex(i - 1, j, k, H.nx, H.ny);
        size_t const idymo = cuda_utilities::compute1DIndex(i, j - 1, k, H.nx, H.ny);
        size_t const idzmo = cuda_utilities::compute1DIndex(i, j, k - 1, H.nx, H.ny);

        C.magnetic_x[id] = (vectorPotential.at(id + 2 * H.n_cells) - vectorPotential.at(idymo + 2 * H.n_cells)) / H.dy -
                           (vectorPotential.at(id + 1 * H.n_cells) - vectorPotential.at(idzmo + 1 * H.n_cells)) / H.dz;
        C.magnetic_y[id] = (vectorPotential.at(id + 0 * H.n_cells) - vectorPotential.at(idzmo + 0 * H.n_cells)) / H.dz -
                           (vectorPotential.at(id + 2 * H.n_cells) - vectorPotential.at(idxmo + 2 * H.n_cells)) / H.dx;
        C.magnetic_z[id] = (vectorPotential.at(id + 1 * H.n_cells) - vectorPotential.at(idxmo + 1 * H.n_cells)) / H.dx -
                           (vectorPotential.at(id + 0 * H.n_cells) - vectorPotential.at(idymo + 0 * H.n_cells)) / H.dy;
      }
    }
  }
}
#endif  // MHD
}  // end namespace mhd::utils
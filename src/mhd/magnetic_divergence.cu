/*!
 * \file mhd_utilities.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the implementation of various utility functions for MHD and
 * for the various kernels, functions, and tools required for the 3D VL+CT MHD
 * integrator. Due to the CUDA/HIP compiler requiring that device functions be
 * directly accessible to the file they're used in most device functions will be
 * implemented in the header file. Uses the same method described in Stone et
 * al. 2008 "ATHENA: A new code for astrophysical MHD", hereafter referred to as
 * Stone et al. 2008
 *
 */

// STL Includes
#include <cfloat>
#include <limits>

// External Includes

// Local Includes
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../mhd/magnetic_divergence.h"
#include "../utils/DeviceVector.h"
#include "../utils/cuda_utilities.h"
#include "../utils/error_handling.h"
#include "../utils/reduction_utilities.h"
#ifdef MHD

namespace mhd
{
// =========================================================================
__global__ void calculateMagneticDivergence(Real const *dev_conserved, Real *dev_maxDivergence, Real const dx,
                                            Real const dy, Real const dz, int const nx, int const ny, int const nz,
                                            int const n_cells)
{
  // Variables to store the divergence
  Real cellDivergence;
  Real maxDivergence = 0.0;

  // Index variables
  int xid, yid, zid, id_xMin1, id_yMin1, id_zMin1;

  // Grid stride loop to perform as much of the reduction as possible
  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    // compute the real indices
    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

    // Thread guard to avoid overrun and to skip ghost cells that cannot
    // have their divergences computed due to a missing face;
    if (xid > 1 and yid > 1 and zid > 1 and xid < nx and yid < ny and zid < nz) {
      // Compute the various offset indices
      id_xMin1 = cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny);
      id_yMin1 = cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny);
      id_zMin1 = cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny);

      // Compute divergence
      // Stone et al. 2008 equation 25
      cellDivergence = ((dev_conserved[id + (grid_enum::magnetic_x)*n_cells] -
                         dev_conserved[id_xMin1 + (grid_enum::magnetic_x)*n_cells]) /
                        dx) +
                       ((dev_conserved[id + (grid_enum::magnetic_y)*n_cells] -
                         dev_conserved[id_yMin1 + (grid_enum::magnetic_y)*n_cells]) /
                        dy) +
                       ((dev_conserved[id + (grid_enum::magnetic_z)*n_cells] -
                         dev_conserved[id_zMin1 + (grid_enum::magnetic_z)*n_cells]) /
                        dz);

      maxDivergence = max(maxDivergence, fabs(cellDivergence));
    }
  }

  // Perform reduction across the entire grid
  reduction_utilities::gridReduceMax(maxDivergence, dev_maxDivergence);
}
// =========================================================================

// =============================================================================
Real checkMagneticDivergence(Grid3D const &G)
{
  // Compute the local value of the divergence
  // First let's create some variables we'll need.
  cuda_utilities::AutomaticLaunchParams static const launchParams(mhd::calculateMagneticDivergence);
  cuda_utilities::DeviceVector<Real> static dev_maxDivergence(1);

  // Set the device side divergence to the smallest possible double so that
  // the reduction isn't using the maximum value of the previous iteration
  dev_maxDivergence.assign(std::numeric_limits<Real>::lowest());

  // Now lets get the local maximum divergence
  hipLaunchKernelGGL(mhd::calculateMagneticDivergence, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(),
                     0, 0, G.C.device, dev_maxDivergence.data(), G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny, G.H.nz,
                     G.H.n_cells);
  GPU_Error_Check();
  Real max_magnetic_divergence = dev_maxDivergence[0];

  #ifdef MPI_CHOLLA
  // Now that we have the local maximum let's get the global maximum
  max_magnetic_divergence = ReduceRealMax(max_magnetic_divergence);
  #endif  // MPI_CHOLLA

  // If the magnetic divergence is greater than the limit then raise a warning and exit.
  // This maximum value of divergence was chosen after a discussion with Chris White of the Flatiron institute and an
  // Athena dev. He said that in his experience issues start showing up at around 1E-8 divergence so this is set with an
  // order of magnitude margin.
  Real static const magnetic_divergence_limit = 1.0E-9;
  if (max_magnetic_divergence > magnetic_divergence_limit) {
    // Report the error and exit
    chprintf(
        "The magnetic divergence has exceeded the maximum allowed value. "
        "Divergence = %7.4e, the maximum allowed divergence = %7.4e\n",
        max_magnetic_divergence, magnetic_divergence_limit);
    chexit(-1);
  } else if (max_magnetic_divergence < 0.0) {
    // Report the error and exit
    chprintf("The magnetic divergence is negative. Divergence = %7.4e\n", max_magnetic_divergence);
    chexit(-1);
  } else  // The magnetic divergence is within acceptable bounds
  {
    chprintf("Global maximum magnetic divergence = %7.4e\n", max_magnetic_divergence);
  }

  return max_magnetic_divergence;
}
// =============================================================================
}  // end namespace mhd
#endif  // MHD

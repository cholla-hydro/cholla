/*!
 * \file hydro_utilities.h
 * \author Helena Richie (helenarichie@pitt.edu)
 * \brief Contains the declaration of various utility functions for CUDA
 *
 */

#pragma once

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"

namespace cuda_utilities
{
/*!
 * \brief Compute the x, y, and z indices based off of the 1D index
 *
 * \param[in] id The 1D index
 * \param[in] nx The total number of cells in the x direction
 * \param[in] ny The total number of cells in the y direction
 * \param[out] xid The x index
 * \param[out] yid The y index
 * \param[out] zid The z index
 */
inline __host__ __device__ void compute3DIndices(int const &id, int const &nx,
                                                 int const &ny, int &xid,
                                                 int &yid, int &zid)
{
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;
}

/*!
 * \brief Compute the 1D index based off of the 3D indices
 *
 * \param xid The x index
 * \param yid The y index
 * \param zid The z index
 * \param nx The total number of cells in the x direction
 * \param ny The total number of cells in the y direction
 * \return int The 1D index
 */
inline __host__ __device__ int compute1DIndex(int const &xid, int const &yid,
                                              int const &zid, int const &nx,
                                              int const &ny)
{
  return xid + yid * nx + zid * nx * ny;
}

inline __host__ __device__ void Get_Real_Indices(int const &n_ghost,
                                                 int const &nx, int const &ny,
                                                 int const &nz, int &is,
                                                 int &ie, int &js, int &je,
                                                 int &ks, int &ke)
{
  is = n_ghost;
  ie = nx - n_ghost;
  if (ny == 1) {
    js = 0;
    je = 1;
  } else {
    js = n_ghost;
    je = ny - n_ghost;
  }
  if (nz == 1) {
    ks = 0;
    ke = 1;
  } else {
    ks = n_ghost;
    ke = nz - n_ghost;
  }
}

/*!
 * \brief Initialize GPU memory
 *
 * \param[in] ptr The pointer to GPU memory
 * \param[in] N The size of the array in bytes
 */
inline void initGpuMemory(Real *ptr, size_t N)
{
  CudaSafeCall(cudaMemset(ptr, 0, N));
}

// =====================================================================
/*!
 * \brief Struct to determine the optimal number of blocks and threads
 * per block to use when launching a kernel. The member
 * variables are `threadsPerBlock` and `numBlocks` which are chosen with
 the occupancy API. Can target any device on the system through the
 * optional constructor argument.
 * NOTE: On AMD there's currently an issue that stops kernels from being
 * passed. As a workaround for now this struct just returns the maximum
 * number of blocks and threads per block that a MI250X can run at once.
 *
 */
template <typename T>
struct AutomaticLaunchParams {
 public:
  /*!
   * \brief Construct a new Reduction Launch Params object. By default it
   * generates values of numBlocks and threadsPerBlock suitable for a
   * kernel with a grid-stride loop. For a kernel with one thread per
   * element set the optional `numElements` argument to the number of
   * elements
   *
   * \param[in] kernel The kernel to determine the launch parameters for
   * \param[in] numElements The number of elements in the array that
   the kernel operates on
   */
  AutomaticLaunchParams(T &kernel, size_t numElements = 0)
  {
    cudaOccupancyMaxPotentialBlockSize(&numBlocks, &threadsPerBlock, kernel, 0,
                                       0);

    if (numElements > 0) {
      numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    }
  }

  /// Defaulted Destructor
  ~AutomaticLaunchParams() = default;

  /// The maximum number of threads per block that the device supports
  int threadsPerBlock;
  /// The maximum number of scheduleable blocks on the device
  int numBlocks;
};
// =====================================================================
}  // end namespace cuda_utilities

/*!
 * \file hydro_utilities.h
 * \author Helena Richie (helenarichie@pitt.edu)
 * \brief Contains the declaration of various utility functions for CUDA
 *
 */

#pragma once

#include <string>

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
inline __host__ __device__ void compute3DIndices(int const &id, int const &nx, int const &ny, int &xid, int &yid,
                                                 int &zid)
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
inline __host__ __device__ int compute1DIndex(int const &xid, int const &yid, int const &zid, int const &nx,
                                              int const &ny)
{
  return xid + yid * nx + zid * nx * ny;
}

inline __host__ __device__ void Get_Real_Indices(int const &n_ghost, int const &nx, int const &ny, int const &nz,
                                                 int &is, int &ie, int &js, int &je, int &ks, int &ke)
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
inline void initGpuMemory(Real *ptr, size_t N) { GPU_Error_Check(cudaMemset(ptr, 0, N)); }

// =====================================================================
/*!
 * \brief Struct to determine the optimal number of blocks and threads
 * per block to use when launching a kernel. The member
 * variables are `threadsPerBlock` and `numBlocks` which are chosen with
 * the occupancy API.
 *
 */
template <typename T>
struct AutomaticLaunchParams {
 public:
  /*!
   * \brief Construct a new AutomaticLaunchParams object. By default it
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
    cudaOccupancyMaxPotentialBlockSize(&numBlocks, &threadsPerBlock, kernel, 0, 0);

    if (numElements > 0) {
      // This line is needed to check that threadsPerBlock isn't zero. Somewhere inside
      // cudaOccupancyMaxPotentialBlockSize threadsPerBlock can be zero according to clang-tidy so this line sets it to
      // a more reasonable value
      threadsPerBlock = (threadsPerBlock == 0) ? TPB : threadsPerBlock;

      // Compute the number of blocks
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

// =====================================================================
/*!
 * \brief Print the current GPU memory usage to standard out
 *
 * \param additional_text Any additional text to be appended to the end of the message
 */
void Print_GPU_Memory_Usage(std::string const &additional_text = "");
// =====================================================================
}  // end namespace cuda_utilities
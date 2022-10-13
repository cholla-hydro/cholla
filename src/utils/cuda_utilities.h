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
    inline __host__ __device__ void compute3DIndices(int const &id,
                                                     int const &nx,
                                                     int const &ny,
                                                     int &xid,
                                                     int &yid,
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
    inline __host__ __device__ int compute1DIndex(int const &xid,
                                                  int const &yid,
                                                  int const &zid,
                                                  int const &nx,
                                                  int const &ny)
    {
        return xid + yid*nx + zid*nx*ny;
    }

    inline __host__ __device__ void Get_Real_Indices(int const &n_ghost, int const &nx, int const &ny, int const &nz, int &is, int &ie, int &js, int &je, int &ks, int &ke) {
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

    // =========================================================================
    /*!
    * \brief Set the value that `pointer` points at in GPU memory to `value`.
    * This only sets the first value in memory so if `pointer` points to an
    * array then only `pointer[0]` will be set; i.e. this effectively does
    * `pointer = &value`
    *
    * \tparam T Any scalar type
    * \param[in] pointer The location in GPU memory
    * \param[in] value The value to set `*pointer` to
    */
    template <typename T>
    void setScalarDeviceMemory(T *pointer, T const value)
    {
        CudaSafeCall(
            cudaMemcpy(pointer,  // destination
                       &value,   // source
                       sizeof(T),
                       cudaMemcpyHostToDevice));
    }
    // =========================================================================
}
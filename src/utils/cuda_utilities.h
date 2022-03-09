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

namespace cudaUtils {
    namespace {
        
        inline __host__ __device__ void Get_GTID(int &id, int &xid, int &yid, int &zid, int &tid, int const nx, int const ny, int const nz) {
            int blockId = blockIdx + blockIdy * gridDimx;
            id = threadIdx + blockId * blockDimx;
            zid = id / (nx * ny);
            yid = (id - zid * nx * ny) / nx;
            xid = id - zid * nx * ny - yid * nx;
            // add a thread id within the block
            tid = threadIdx;
        }
    }
}
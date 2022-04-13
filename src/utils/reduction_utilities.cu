/*!
 * \file reduction_utilities.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the implementation of the GPU resident reduction utilities
 *
 */

// STL Includes
#include <float.h>

// External Includes

// Local Includes
#include "../utils/reduction_utilities.h"

#ifdef CUDA
    namespace reduction_utilities
    {
        // =====================================================================
        __global__ void kernelReduceMax(Real *in, Real* out, size_t N)
        {
            // Initialize variable to store the max value
            Real maxVal = -DBL_MAX;

            // Grid stride loop to perform as much of the reduction as possible
            for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
            {
                // A transformation could go here

                // Grid stride reduction
                maxVal = max(maxVal,in[i]);
            }

            // Find the maximum val in the grid and write it to `out`. Note that there
            // is no execution/memory barrier after this and so the reduced scalar is
            // not available for use in this kernel. The grid wide barrier can be
            // accomplished by ending this kernel here and then launching a new one or
            // by using cooperative groups. If this becomes a need it can be added later
            gridReduceMax(maxVal, out);
        }
        // =====================================================================

        // =====================================================================
        void reductionLaunchParams(uint &numBlocks, uint &threadsPerBlock, uint const &deviceNum)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, deviceNum);

            // Divide the total number of allowed threads by the number of
            // threads per block
            threadsPerBlock =  prop.maxThreadsPerBlock;
            numBlocks       = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount)
                              / threadsPerBlock;
        }
        // =====================================================================
    }//reduction_utilities
#endif  //CUDA
/*!
 * \file reduction_utilities.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declaration of the GPU resident reduction utilities
 *
 */

#pragma once

// STL Includes
#include <float.h>

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"

#ifdef CUDA
    /*!
    * \brief Namespace to contain device resident reduction functions. Includes
    * functions and kernels for array reduction, warp level, block level, and
    * grid level reductions.
    *
    */
    namespace reduction_utilities
    {
        // =====================================================================
        /*!
        * \brief Perform a reduction within the warp/wavefront to find the
        * maximum value of `val`
        *
        * \param[in] val The thread local variable to find the maximum of across
        * the warp
        * \return Real The maximum value of `val` within the warp
        */
        __inline__ __device__ Real warpReduceMax(Real val)
        {
            for (int offset = warpSize/2; offset > 0; offset /= 2)
            {
                val = fmax(val, __shfl_down(val, offset));
            }
            return val;
        }
        // =====================================================================

        // =====================================================================
        /*!
        * \brief Perform a reduction within the block to find the maximum value
        * of `val`
        *
        * \param[in] val The thread local variable to find the maximum of across
        * the block
        * \return Real The maximum value of `val` within the block
        */
        __inline__ __device__ Real blockReduceMax(Real val)
        {
            // Shared memory for storing the results of each warp-wise partial
            // reduction
            __shared__ Real shared[::maxWarpsPerBlock];

            int lane   = threadIdx.x % warpSize;  // thread ID within the warp,
            int warpId = threadIdx.x / warpSize;  // ID of the warp itself

            val = warpReduceMax(val);     // Each warp performs partial reduction

            if (lane==0) shared[warpId]=val; // Write reduced value to shared memory

            __syncthreads();              // Wait for all partial reductions

            //read from shared memory only if that warp existed
            val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

            if (warpId==0) val = warpReduceMax(val); //Final reduce within first warp

            return val;
        }
        // =====================================================================

        // =====================================================================
        /*!
        * \brief Perform an atomic reduction to find the maximum value of `val`
        *
        * \param[out] address The pointer to where to store the reduced scalar
        * value in device memory
        * \param[in] val The thread local variable to find the maximum of across
        * the grid. Typically this should be a partial reduction that has
        * already been reduced to the block level
        */
        __inline__ __device__ double atomicMax_double(double* address, double val)
        {
            unsigned long long int* address_as_ull = (unsigned long long int*) address;
            unsigned long long int old = *address_as_ull, assumed;
            // Explanation of loop here:
            // https://stackoverflow.com/questions/16077464/atomicadd-for-double-on-gpu
            // The loop is to make sure the value at address doesn't change
            // between the load at the atomic since the entire operation isn't
            // atomic

            // While it appears that this could result in many times more atomic
            // operations than required, in practice it's only a handful of
            // extra operation even in the worst case. Running with 16,000
            // blocks gives ~8-37 atomics after brief testing
            do {
                assumed = old;
                old = atomicCAS(address_as_ull,
                                assumed,
                                __double_as_longlong(fmax(__longlong_as_double(assumed),val)));
            } while (assumed != old);
            return __longlong_as_double(old);
        }
        // =====================================================================

        // =====================================================================
        /*!
         * \brief Perform a reduction within the grid to find the maximum value
         * of `val`. Note that this will overwrite the value in `out` with
         * the value of the optional argument `lowLimit` which defaults to
         * `-DBL_MAX`
         *
         * \details This function can perform a reduction to find the maximum of
         * the thread local variable `val` across the entire grid. It relies on a
         * warp-wise reduction using registers followed by a block-wise reduction
         * using shared memory, and finally a grid-wise reduction using atomics.
         * As a result the performance of this function is substantally improved
         * by using as many threads per block as possible and as few blocks as
         * possible since each block has to perform an atomic operation. To
         * accomplish this it is recommened that you use the
         * `reductionLaunchParams` functions to get the optimal number of blocks
         * and threads per block to launch rather than relying on Cholla defaults
         * and then within the kernel using a grid-stride loop to make sure the
         * kernel works with any combination of threads and blocks. Note that
         * after this function call you cannot use the reduced value in global
         * memory since there is no grid wide sync. You can get around this by
         * either launching a second kernel to do the next steps or by using
         * cooperative groups to perform a grid wide sync. During it's execution
         * it also calls multiple __synchThreads and so cannot be called from
         * within any kind of thread guard.
         *
         * \param[in] val The thread local variable to find the maximum of across
         * the grid
         * \param[out] out The pointer to where to store the reduced scalar value
         * in device memory
         * \param[in] lowLimit (optional)What value to initilize global memory
         * with. Defaults to -DBL_MAX. This value will be the lowest possible
         * value that the reduction can produce since all other values are
         * compared against it
         */
        __inline__ __device__ void gridReduceMax(Real val, Real* out, Real lowLimit = -DBL_MAX)
        {
            __syncthreads();              // Wait for all threads to calculate val;
	  
            // Set the value in global memory so meaningful comparisons can be
            // performed
            if (threadIdx.x + blockIdx.x * blockDim.x == 0) *out = lowLimit;

            // Reduce the entire block in parallel
            val = blockReduceMax(val);

            // Write block level reduced value to the output scalar atomically
            // if (threadIdx.x == 0) atomicMax_double(out, val);
	    if (threadIdx.x == 0) out[blockIdx.x] = val;
        }
        // =====================================================================

        // =====================================================================
        /*!
         * \brief Find the maximum value in the array. Note that this will
         * overwrite the value in `out` with the value of the optional argument
         * `lowLimit` which defaults to `-DBL_MAX`. If `in` and `out` are the
         * same array that's ok, all the loads are completed before the
         * overwrite occurs
         *
         * \param[in] in The pointer to the array to reduce in device memory
         * \param[out] out The pointer to where to store the reduced scalar
         * value in device memory
         * \param[in] N The size of the `in` array
         * \param[in] lowLimit (optional) What value to initilize global memory
         * with. Defaults to -DBL_MAX. This value will be the lowest possible
         * value that the reduction can produce since all other values are
         * compared against it
         */
        __global__ void kernelReduceMax(Real *in, Real* out, size_t N, Real lowLimit = -DBL_MAX);
        // =====================================================================

        // =====================================================================
        /*!
        * \brief Determine the optimal number of blocks and threads per block to
        * use when launching a reduction kernel
        *
        * \param[out] numBlocks The maximum number of blocks that are
        * scheduleable by the device in use when each block has the maximum
        * number of threads
        * \param[out] threadsPerBlock The maximum threads per block supported by
        * the device in use
        * \param[in] deviceNum optional: which device is being targeted.
        * Defaults to zero
        */
        void reductionLaunchParams(uint &numBlocks,
                                   uint &threadsPerBlock,
                                   uint const &deviceNum=0);
        // =====================================================================
    }  // namespace reduction_utilities
#endif  //CUDA

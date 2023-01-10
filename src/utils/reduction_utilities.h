/*!
 * \file reduction_utilities.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declaration of the GPU resident reduction utilities
 *
 */

#pragma once

// STL Includes
#include <cstdint>

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
                val = max(val, __shfl_down(val, offset));
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

        #ifndef O_HIP
        // =====================================================================
        // This section handles the atomics. It is complicated because CUDA
        // doesn't currently support atomics with non-integral types.
        // This code is taken from
        // https://github.com/rapidsai/cuml/blob/dc14361ba11c41f7a4e1e6a3625bbadd0f52daf7/cpp/src_prims/stats/minmax.cuh
        // with slight tweaks for our use case.
        // =====================================================================
        /*!
         * \brief Do a device side bit cast
         *
         * \tparam To The output type
         * \tparam From The input type
         * \param from The input value
         * \return To The bit cast version of From as type To
         */
        template <class To, class From>
        __device__ constexpr To bit_cast(const From& from) noexcept
        {
            // TODO: replace with `std::bitcast` once we adopt C++20 or libcu++ adds it
            To to{};
            static_assert(sizeof(To) == sizeof(From));
            memcpy(&to, &from, sizeof(To));
            return to;
        }

        /*!
         * \brief Encode a float as an int
         *
         * \param val The float to encode
         * \return int The encoded int
         */
        inline __device__ int encode(float val)
        {
            int i = bit_cast<int>(val);
            return i >= 0 ? i : (1 << 31) | ~i;
        }

        /*!
         * \brief Encode a double as a long long int
         *
         * \param val The double to encode
         * \return long long The encoded long long int
         */
        inline __device__ long long encode(double val)
        {
            std::int64_t i = bit_cast<std::int64_t>(val);
            return i >= 0 ? i : (1ULL << 63) | ~i;
        }

        /*!
         * \brief Decodes an int as a float
         *
         * \param val The int to decode
         * \return float The decoded float
         */
        inline __device__ float decode(int val)
        {
            if (val < 0) val = (1 << 31) | ~val;
            return bit_cast<float>(val);
        }

        /*!
         * \brief Decodes a long long int as a double
         *
         * \param val The long long to decode
         * \return double The decoded double
         */
        inline __device__ double decode(long long val)
        {
            if (val < 0) val = (1ULL << 63) | ~val;
            return bit_cast<double>(val);
        }
        #endif  //O_HIP
        /*!
        * \brief Perform an atomic reduction to find the maximum value of `val`
        *
        * \param[out] address The pointer to where to store the reduced scalar
        * value in device memory
        * \param[in] val The thread local variable to find the maximum of across
        * the grid. Typically this should be a partial reduction that has
        * already been reduced to the block level
        */
        inline __device__ float atomicMaxBits(float* address, float val)
        {
            #ifdef O_HIP
                return atomicMax(address, val);
            #else //O_HIP
                int old = atomicMax((int*)address, encode(val));
                return decode(old);
            #endif //O_HIP
        }

        /*!
        * \brief Perform an atomic reduction to find the maximum value of `val`
        *
        * \param[out] address The pointer to where to store the reduced scalar
        * value in device memory
        * \param[in] val The thread local variable to find the maximum of across
        * the grid. Typically this should be a partial reduction that has
        * already been reduced to the block level
        */
        inline __device__ double atomicMaxBits(double* address, double val)
        {
            #ifdef O_HIP
                return atomicMax(address, val);
            #else //O_HIP
                long long old = atomicMax((long long*)address, encode(val));
                return decode(old);
            #endif //O_HIP
        }

        /*!
        * \brief Perform an atomic reduction to find the minimum value of `val`
        *
        * \param[out] address The pointer to where to store the reduced scalar
        * value in device memory
        * \param[in] val The thread local variable to find the minimum of across
        * the grid. Typically this should be a partial reduction that has
        * already been reduced to the block level
        */
        inline __device__ float atomicMinBits(float* address, float val)
        {
            #ifdef O_HIP
                return atomicMin(address, val);
            #else //O_HIP
                int old = atomicMin((int*)address, encode(val));
                return decode(old);
            #endif //O_HIP
        }

        /*!
        * \brief Perform an atomic reduction to find the minimum value of `val`
        *
        * \param[out] address The pointer to where to store the reduced scalar
        * value in device memory
        * \param[in] val The thread local variable to find the minimum of across
        * the grid. Typically this should be a partial reduction that has
        * already been reduced to the block level
        */
        inline __device__ double atomicMinBits(double* address, double val)
        {
            #ifdef O_HIP
                return atomicMin(address, val);
            #else //O_HIP
                long long old = atomicMin((long long*)address, encode(val));
                return decode(old);
            #endif //O_HIP
        }
        // =====================================================================

        // =====================================================================
        /*!
         * \brief Perform a reduction within the grid to find the maximum value
         * of `val`. Note that the value of `out` should be set appropriately
         * before the kernel launch that uses this function to avoid any
         * potential race condition; the `cuda_utilities::setScalarDeviceMemory`
         * function exists for this purpose.
         * of `val`. Note that the value of `out` should be set appropriately
         * before the kernel launch that uses this function to avoid any
         * potential race condition; the `cuda_utilities::setScalarDeviceMemory`
         * function exists for this purpose.
         *
         * \details This function can perform a reduction to find the maximum of
         * the thread local variable `val` across the entire grid. It relies on a
         * warp-wise reduction using registers followed by a block-wise reduction
         * using shared memory, and finally a grid-wise reduction using atomics.
         * As a result the performance of this function is substantally improved
         * by using as many threads per block as possible and as few blocks as
         * possible since each block has to perform an atomic operation. To
         * accomplish this it is reccommened that you use the
         * `AutomaticLaunchParams` functions to get the optimal number of blocks
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
         */
        __inline__ __device__ void gridReduceMax(Real val, Real* out)
        {

            // Reduce the entire block in parallel
            val = blockReduceMax(val);

            // Write block level reduced value to the output scalar atomically
            if (threadIdx.x == 0) atomicMaxBits(out, val);
        }
        // =====================================================================

        // =====================================================================
        /*!
         * \brief Find the maximum value in the array. Make sure to initialize
         * `out` correctly before using this kernel; the
         * `cuda_utilities::setScalarDeviceMemory` function exists for this
         * purpose. If `in` and `out` are the same array that's ok, all the
         * loads are completed before the overwrite occurs.
         * \brief Find the maximum value in the array. Make sure to initialize
         * `out` correctly before using this kernel; the
         * `cuda_utilities::setScalarDeviceMemory` function exists for this
         * purpose. If `in` and `out` are the same array that's ok, all the
         * loads are completed before the overwrite occurs.
         *
         * \param[in] in The pointer to the array to reduce in device memory
         * \param[out] out The pointer to where to store the reduced scalar
         * value in device memory
         * \param[in] N The size of the `in` array
         */
        __global__ void kernelReduceMax(Real *in, Real* out, size_t N);
        // =====================================================================
    }  // namespace reduction_utilities
#endif  //CUDA

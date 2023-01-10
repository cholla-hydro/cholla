/*!
 * \file DeviceVector.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declartion and implementation of the DeviceVector
 * class. Note that since this is a templated class the implementation must be
 * in the header file
 *
 */

#pragma once

// STL Includes
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"

// =============================================================================
// Declaration of DeviceVector class
// =============================================================================
namespace cuda_utilities
{
    /*!
     * \brief A templatized class to encapsulate a device global memory pointer
     * in a std::vector like interface complete with most of the usual methods.
     * This class is intended to be used only in host code and does not work
     * device side; Passing the pointer to a kernel can be done with the
     * `data()` method. This class works for any device side pointer, scalar or
     * array valued.
     *
     * \tparam T Any serialized type where `sizeof(T)` returns correct results
     * should work but non-primitive types have not been tested.
     */
    template <typename T>
    class DeviceVector
    {
    public:
        /*!
         * \brief Construct a new Device Vector object by calling the
         * `_allocate` private method
         *
         * \param[in] size The number of elements desired in the array. Can be
         * any positive integer.
         * \param[in] initialize (optional) If true then initialize the GPU
         * memory to int(0)
         */
        DeviceVector(size_t const size, bool const initialize=false);

        /*!
         * \brief Destroy the Device Vector object by calling the `_deAllocate`
         * private method
         *
         */
        ~DeviceVector() {_deAllocate();}

        /*!
         * \brief Get the raw device pointer
         *
         * \return T* The pointer for the array in global memory
         */
        T* data() {return _ptr;}

        /*!
         * \brief Get the number of elements in the array.
         *
         * \return size_t The number of elements in the array
         */
        size_t size() {return _size;}

        /*!
         * \brief Overload the [] operator to return a value from device memory.
         * This method performs a cudaMemcpy to copy the desired element to the
         * host then returns it. Unlike the `at()` method this method does not
         * perform bounds checking
         *
         * \param[in] index The index of the desired value
         * \return T The value at dev_ptr[index]
         */
        T operator [] (size_t const &index);

        /*!
         * \brief Return a value from device memory. This method performs a
         * cudaMemcpy to copy the desired element to the host then returns it.
         * Unlike the `[]` overload this method perform bounds checking
         *
         * \param[in] index The index of the desired value
         * \return T The value at dev_ptr[index]
         */
        T const at(size_t const index);

        /*!
        * \brief Assign a single value in the array. Should generally only be
        * used when the pointer points to a scalar value. By default this
        * writes `hostValue` to the 0th element of the array.
        *
        * \param[in] hostValue The value to write to the device array
        * \param[in] index The location to write the value to, defaults to zero.
        */
        void assign(T const &hostValue, size_t const &index=0);

        /*!
         * \brief Resize the device container to contain `newSize` elements. If
         * `newSize` is greater than the current size then all the values are
         * kept and the rest of the array is default initialized. If `newSize`
         * is smaller than the current size then the array is truncated and
         * values at locations greater than `newSize` are lost. Keeping the
         * values in the array requires that the new array be allocated, the
         * values be copied, then the old array be freed; as such this method is
         * quite slow and can use a large amount of memory. If you don't care
         * about the values in the array then use the `reset` method
         *
         * \param[in] newSize The desired size of the array
         */
        void resize(size_t const newSize);

        /*!
         * \brief Reset the size of the array. This frees the old array and
         * allocates a new one; all values in the array may be lost. The values
         * in memory are not initialized and therefore the behaviour of the
         * default values is undefined
         *
         * \param newSize
         */
        void reset(size_t const newSize);

        /*!
         * \brief Copy the first `arrSize` elements of `arrIn` to the device.
         *
         * \param[in] arrIn The pointer to the array to be copied to the device
         * \param[in] arrSize The number of elements/size of the array to copy
         * to the device
         */
        void cpyHostToDevice(const T * arrIn, size_t const &arrSize);

        /*!
         * \brief Copy the contents of a std::vector to the device
         *
         * \param[in] vecIn The array whose contents are to be copied
         */
        void cpyHostToDevice(std::vector<T> const &vecIn)
            {cpyHostToDevice(vecIn.data(), vecIn.size());}

        /*!
         * \brief Copy the array from the device to a host array. Checks if the
         * host array is large enough based on the `arrSize` parameter.
         *
         * \param[out] arrOut The pointer to the host array
         * \param[in] arrSize The number of elements allocated in the host array
         */
        void cpyDeviceToHost(T * arrOut, size_t const &arrSize);

        /*!
         * \brief Copy the array from the device to a host std::vector. Checks
         * if the host array is large enough.
         *
         * \param[out] vecOut The std::vector to copy the device array into
         */
        void cpyDeviceToHost(std::vector<T> &vecOut)
            {cpyDeviceToHost(vecOut.data(), vecOut.size());}

    private:
        /// The size of the device array
        size_t _size;

        /// The pointer to the device array
        T *_ptr=nullptr;

        /*!
         * \brief Allocate the device side array
         *
         * \param[in] size The size of the array to allocate
         */
        void _allocate(size_t const size)
        {
            _size=size;
            CudaSafeCall(cudaMalloc(&_ptr, _size*sizeof(T)));
        }

        /*!
         * \brief Free the device side array
         *
         */
        void _deAllocate(){CudaSafeCall(cudaFree(_ptr));}
    };
}  // End of cuda_utilities namespace
// =============================================================================
// End declaration of DeviceVector class
// =============================================================================


// =============================================================================
// Definition of DeviceVector class
// =============================================================================
namespace cuda_utilities
{
    // =========================================================================
    // Public Methods
    // =========================================================================

    // =========================================================================
    template <typename T>
    DeviceVector<T>::DeviceVector(size_t const size, bool const initialize)
    {
        _allocate(size);

        if (initialize)
        {
            CudaSafeCall(cudaMemset(_ptr, 0, _size*sizeof(T)));
        }
    }
    // =========================================================================

    // =========================================================================
    template <typename T>
    void DeviceVector<T>::resize(size_t const newSize)
    {
        // Assign old array to a new pointer
        T * oldDevPtr = _ptr;

        // Determine how many elements to copy
        size_t const count = std::min(_size, newSize) * sizeof(T);

        // Allocate new array
        _allocate(newSize);

        // Copy the values from the old array to the new array
        CudaSafeCall(cudaMemcpyPeer(_ptr, 0, oldDevPtr, 0, count));

        // Free the old array
        CudaSafeCall(cudaFree(oldDevPtr));
    }
    // =========================================================================

    // =========================================================================
    template <typename T>
    void DeviceVector<T>::reset(size_t const newSize)
    {
        _deAllocate();
        _allocate(newSize);
    }
    // =========================================================================

    // =========================================================================
    template <typename T>
    T DeviceVector<T>::operator [] (size_t const &index)
    {
        T hostValue;
        CudaSafeCall(cudaMemcpy(&hostValue,
                                &(_ptr[index]),
                                sizeof(T),
                                cudaMemcpyDeviceToHost));
        return hostValue;
    }
    // =========================================================================

    // =========================================================================
    template <typename T>
    T const DeviceVector<T>::at(size_t const index)
    {
        if (index < _size)
        {
            // Use the overloaded [] operator to grab the value from GPU memory
            // into host memory
            return (*this)[index];
        }
        else
        {
            throw std::out_of_range("Warning: DeviceVector.at() detected an"
                                    " out of bounds memory access. Tried to"
                                    " access element "
                                    + std::to_string(index)
                                    + " of "
                                    + std::to_string(_size));
        }
    }
    // =========================================================================

    // =========================================================================
    template <typename T>
    void DeviceVector<T>::assign(T const &hostValue, size_t const &index)
    {
        CudaSafeCall(cudaMemcpy(&(_ptr[index]),  // destination
                                &hostValue,      // source
                                sizeof(T),
                                cudaMemcpyHostToDevice));
    }
    // =========================================================================

    // =========================================================================
    template <typename T>
    void DeviceVector<T>::cpyHostToDevice(const T * arrIn, size_t const &arrSize)
    {
        if (arrSize <= _size)
        {
            CudaSafeCall(cudaMemcpy(_ptr,
                                    arrIn,
                                    arrSize*sizeof(T),
                                    cudaMemcpyHostToDevice));
        }
        else
        {
            throw std::out_of_range("Warning: Couldn't copy array to device,"
                                    " device array is too small. Host array"
                                    " size="
                                    + std::to_string(arrSize)
                                    + ", device array size="
                                    + std::to_string(arrSize));
        }

    }
    // =========================================================================

    // =========================================================================
    template <typename T>
    void DeviceVector<T>::cpyDeviceToHost(T * arrOut, size_t const &arrSize)
    {
        if (_size <= arrSize)
        {
            CudaSafeCall(cudaMemcpy(arrOut,
                                    _ptr,
                                    _size*sizeof(T),
                                    cudaMemcpyDeviceToHost));
        }
        else
        {
            throw std::out_of_range("Warning: Couldn't copy array to host, "
                                    "host array is too small. Host array "
                                    "size="
                                    + std::to_string(arrSize)
                                    + ", device array size="
                                    + std::to_string(arrSize));
        }
    }
    // =========================================================================
} // end namespace cuda_utilities
// =============================================================================
// End definition of DeviceVector class
// =============================================================================
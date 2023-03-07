/*LICENSE*/

#ifndef DEVICES_GPU_POINTER_H
#define DEVICES_GPU_POINTER_H


#include <initializer_list>
#include <cuda_runtime.h>
#include "../../global/global_cuda.h"

namespace GPU
{
    template<typename T> class DeviceBuffer;


    //
    //  A simple autoptr that is host/device aware
    //
    template<typename T> class HostBuffer
    {

    public:

        HostBuffer() {}
        HostBuffer(size_t count) { this->Alloc(count); }
        ~HostBuffer() { this->Free(); };

        void Alloc(size_t count);
        void Free();
        void Zero();

        inline bool IsEmpty() const { return (mPtr == nullptr); }
        inline size_t Count() const { return mCount; }

        inline T* Ptr() { return mPtr; }
        inline const T* Ptr() const { return mPtr; }

        inline operator T*() { return mPtr; }
        inline operator const T*() const { return mPtr; }
        inline T* operator->() { return mPtr; }
        inline const T* operator->() const { return mPtr; }
        inline T& operator[](unsigned int i) { return mPtr[i]; }
        inline const T& operator[](unsigned int i) const { return mPtr[i]; }

        //
        //  All default stream memory transfers are synchronous.
        //  Offsets are {src,dest} for both directons.
        //
        void BlockingTransferToDevice(DeviceBuffer<T>& dest, size_t count = 0, const std::initializer_list<size_t>& offsets = {0,0}) const;
        void BlockingTransferFromDevice(const DeviceBuffer<T>& src, size_t count = 0, const std::initializer_list<size_t>& offsets = {0,0});

        HostBuffer(HostBuffer&&) = delete;
        HostBuffer(const HostBuffer&) = delete;
        void operator=(HostBuffer&&) = delete;
        void operator=(const HostBuffer&) = delete;

    protected:

        T* mPtr = nullptr;
        size_t mCount = 0;
    };


    template<typename T> class DeviceBuffer
    {

    public:

        DeviceBuffer() {}
        DeviceBuffer(size_t count) { this->Alloc(count); }
        ~DeviceBuffer() { this->Free(); };

        void Alloc(size_t count, bool zero = false);
        void Free();
        void BlockingZero();

        inline bool IsEmpty() const { return (mPtr == nullptr); }
        inline size_t Count() const { return mCount; }

        inline T* Ptr() { return mPtr; }
        inline const T* Ptr() const { return mPtr; }

        inline operator T*() { return mPtr; }
        inline operator const T*() const { return mPtr; }
        inline T* operator->() { return mPtr; }
        inline const T* operator->() const { return mPtr; }

        DeviceBuffer(DeviceBuffer&&) = delete;
        DeviceBuffer(const DeviceBuffer&) = delete;
        void operator=(DeviceBuffer&&) = delete;
        void operator=(const DeviceBuffer&) = delete;

    protected:

        T* mPtr = nullptr;
        size_t mCount = 0;
    };


    //
    //  Host & device buffer together
    //
    template<typename T> class TransferBuffers
    {

    public:

        TransferBuffers() = default;
        TransferBuffers(size_t count) { this->Alloc(count); }
        ~TransferBuffers() { this->Free(); }

        void Alloc(size_t count);
        void Free();

        inline bool IsEmpty() const { return this->hBuffer.IsEmpty(); }
        inline size_t Count() const { return this->hBuffer.Count(); }

        inline T* HostPtr() { return this->hBuffer.Ptr(); }
        inline const T* HostPtr() const { return this->hBuffer.Ptr(); }

        inline T* DevicePtr() { return this->dBuffer.Ptr(); }
        inline const T* DevicePtr() const { return this->dBuffer.Ptr(); }

        inline HostBuffer<T>& OnHost() { return this->hBuffer; }
        inline const HostBuffer<T>& OnHost() const { return this->hBuffer; }

        inline DeviceBuffer<T>& OnDevice() { return this->dBuffer; }
        inline const DeviceBuffer<T>& OnDevice() const { return this->dBuffer; }

        inline T& operator[](unsigned int i) { return this->hBuffer[i]; }
        inline const T& operator[](unsigned int i) const { return this->hBuffer[i]; }

        //
        //  All stream-less memory transfers are blocking.
        //
        inline void BlockingTransferToDevice(size_t count = 0, const std::initializer_list<size_t>& offsets = {0,0}) { this->hBuffer.BlockingTransferToDevice(this->dBuffer,count,offsets); }
        inline void BlockingTransferFromDevice(size_t count = 0, const std::initializer_list<size_t>& offsets = {0,0}) { this->hBuffer.BlockingTransferFromDevice(this->dBuffer,count,offsets); }

    private:

        HostBuffer<T> hBuffer;
        DeviceBuffer<T> dBuffer;
    };


    //
    //  Implementations
    //
    template<typename T> inline void HostBuffer<T>::Alloc(size_t count)
    {
        if(count != 0)
        {
            mCount = count;
            cudaHostAlloc(&mPtr,sizeof(T)*count,0);
        }
    }


    template<typename T> inline void HostBuffer<T>::Free()
    {
        if(mPtr != nullptr)
        {
            cudaFreeHost(mPtr);
            mPtr = nullptr;
        }
    }


    template<typename T> inline void HostBuffer<T>::Zero()
    {
        memset(mPtr,0,sizeof(T)*mCount);
    }


    template<typename T> inline void HostBuffer<T>::BlockingTransferToDevice(DeviceBuffer<T>& dest, size_t count, const std::initializer_list<size_t>& offsets) const
    {
        if(count == 0) count = mCount;
        cudaMemcpy(dest.Ptr()+offsets.begin()[1],mPtr+offsets.begin()[0],count*sizeof(T),cudaMemcpyHostToDevice);
    }


    template<typename T> inline void HostBuffer<T>::BlockingTransferFromDevice(const DeviceBuffer<T>& src, size_t count, const std::initializer_list<size_t>& offsets)
    {
        if(count == 0) count = mCount;
        cudaMemcpy(mPtr+offsets.begin()[1],src.Ptr()+offsets.begin()[0],count*sizeof(T),cudaMemcpyDeviceToHost);
    }


    template<typename T> inline void DeviceBuffer<T>::Alloc(size_t count, bool zero)
    {
        if(count != 0)
        {
            mCount = count;
            CudaSafeCall(cudaMalloc(&mPtr,count*sizeof(T)));
            if(zero) CudaSafeCall(cudaMemset(mPtr,0,sizeof(T)*mCount));
        }
    }


    template<typename T> inline void DeviceBuffer<T>::Free()
    {
        if(mPtr != nullptr)
        {
            cudaFree(mPtr);
            mPtr = nullptr;
        }
    }


    template<typename T> inline void DeviceBuffer<T>::BlockingZero()
    {
        cudaMemset(mPtr,0,mCount*sizeof(T));
    }


    template<typename T> inline void TransferBuffers<T>::Alloc(size_t count)
    {
        this->hBuffer.Alloc(count);
        this->dBuffer.Alloc(count);
    }


    template<typename T> inline void TransferBuffers<T>::Free()
    {
        this->hBuffer.Free();
        this->dBuffer.Free();
    }
};

#endif // DEVICES_GPU_POINTER_H

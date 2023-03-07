/*LICENSE*/


#include <cuda_runtime.h>
#include "static_table_gpu.h"
#include "static_table_gpu.cuh"
#include "../../global/global_cuda.h"


template<typename value_t, unsigned int N> StaticTableGPU<value_t,N,'x'>* StaticTableMakerGPU<value_t,N,'x'>::Create(value_t* ys, unsigned int numVars, const unsigned int* size, const value_t* xmin, const value_t* xmax)
{
    StaticTableGPU<value_t,N,'x'> *d;
    CudaSafeCall(cudaMalloc(&d,sizeof(StaticTableGPU<value_t,N,'x'>)));
    if(d != nullptr)
    {
        StaticTableGPU<value_t,N,'x'> h(ys,numVars,size,xmin,xmax);
        CudaSafeCall(cudaMemcpy(d,&h,sizeof(StaticTableGPU<value_t,N,'x'>),cudaMemcpyHostToDevice));
    }
    return d;
}
    

template<typename value_t, unsigned int N> void StaticTableMakerGPU<value_t,N,'x'>::Delete(StaticTableGPU<value_t,N,'x'>* d)
{
    if(d != nullptr)
    {
        CudaSafeCall(cudaFree(d));
    }
}


template struct StaticTableMakerGPU<float,1,'x'>;
template struct StaticTableMakerGPU<float,2,'x'>;
template struct StaticTableMakerGPU<float,3,'x'>;


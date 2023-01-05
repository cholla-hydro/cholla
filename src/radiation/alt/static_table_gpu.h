/*LICENSE*/

#ifndef STATIC_TABLE_GPU_CPU_GPU_H
#define STATIC_TABLE_GPU_CPU_GPU_H


//
//  Generic any-D table on GPU
//
template<typename value_t, unsigned int N, char Mode> class StaticTableGPU;
template<typename value_t, unsigned int N, char Mode> struct StaticTableMakerGPU;


template<typename value_t, unsigned int N> struct StaticTableMakerGPU<value_t,N,'x'>
{
    StaticTableGPU<value_t,N,'x'>* Create(value_t* ys, unsigned int numVars, const unsigned int* size, const value_t* xmin, const value_t* xmax);
    void Delete(StaticTableGPU<value_t,N,'x'>* d);
};

#endif // STATIC_TABLE_GPU_CPU_GPU_H

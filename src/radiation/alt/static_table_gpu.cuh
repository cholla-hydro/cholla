/*LICENSE*/

#ifndef STATIC_TABLE_GPU_GPU_H
#define STATIC_TABLE_GPU_GPU_H

//
//  Generic any-D table on GPU
//
#include "align.h"
#include "table_base.h"

template <typename value_t, unsigned int N, char Mode>
class StaticTableGPU;

template <typename value_t, unsigned int N>
class GPU_ALIGN_DECL StaticTableGPU<value_t, N, 'x'> : public TableBase<value_t, N, true>
{
 public:
  __device__ inline bool GetValues(const value_t x[N], value_t* ys, unsigned int begin, unsigned int end) const
  {
    return this->GetValuesImpl(x, ys, begin, end);
  }

  StaticTableGPU(value_t* ys, unsigned int numVars, const unsigned int size[N], const value_t xmin[N],
                 const value_t xmax[N])
  {
    this->Set(ys, numVars, size, xmin, xmax);
  }

  StaticTableGPU(value_t* ys, unsigned int numVars, unsigned int size, value_t xmin, value_t xmax)
  {
    unsigned int size1[N];
    value_t xmin1[N], xmax1[N];
    for (unsigned int n = 0; n < N; n++) {
      size1[n] = size;
      xmin1[n] = xmin;
      xmax1[n] = xmax;
    }
    this->Set(ys, numVars, size1, xmin1, xmax1);
  }
};

#endif  // STATIC_TABLE_GPU_GPU_H

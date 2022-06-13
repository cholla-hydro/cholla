#ifdef PARIS


#include "fft_3D.h"
#include "../utils/gpu.hpp"
#include "../io/io.h"
#include <cassert>
#include <cfloat>
#include <climits>


__host__ __device__ static inline double sqr(const double x) { return x*x; }

void FFT_3D::Filter_inv_k2( double *const input, double *const output, bool in_device ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;
  
  if ( in_device ){
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 
    
  // Provide FFT filter with a lambda that does 1/k^2 solve in frequency space
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
        // const double i2 = sqr(double(min(i,ni-i))*ddi);
        // const double j2 = sqr(double(min(j,nj-j))*ddj);
        // const double k2 = sqr(double(k)*ddk);
        // const double d = -1.0/(i2+j2+k2);
        // return cufftDoubleComplex{d*b.x,d*b.y};
        return cufftDoubleComplex{1.0*b.x, 1.0*b.y};
      } else {
        return cufftDoubleComplex{0.0,0.0};
      }
    });
    
  if ( in_device ){
    CHECK( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    CHECK( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  } 
}



#endif


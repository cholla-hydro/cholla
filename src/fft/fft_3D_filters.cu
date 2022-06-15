#ifdef PARIS


#include "fft_3D.h"
#include "../utils/gpu.hpp"
#include "../io/io.h"
#include <cassert>
#include <cfloat>
#include <climits>


__host__ __device__ static inline double sqr(const double x) { return x*x; }

__device__ Real linear_interpolation( Real x, Real *x_vals, Real *y_vals, int N ){
  if ( x <= x_vals[0] ){
    printf(" x: %f  outside of interplation range.\n", x );
    return y_vals[0];
  }
  if ( x >= x_vals[N-1] ){
    printf(" x: %f  outside of interplation range.\n", x );
    return y_vals[N-1];
  }
  int indx = 0;
  while( x_vals[indx] < x ) indx +=1;
  // printf( "%d \n", indx );
  Real xl, xr, yl, yr;
  xl = x_vals[indx-1];
  xr = x_vals[indx];
  yl = y_vals[indx-1];
  yr = y_vals[indx];  
  if ( x < xl || x > xr ) printf(" ##################### Interpolation error:   x: %e  xl: %e  xr: %e   indx: %d\n", x, xl, xr, indx );
  return  yl + ( x - xl ) / ( xr - xl ) * ( yr - yl );
}




void FFT_3D::Filter_rescale_by_k_k2( double *input, double *output, bool in_device, int direction, double D ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_, nk = nk_;
  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  if ( in_device ){
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 

  // Provide FFT filter with a lambda that multiplies by k / k^2 / D
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
      // const double kx = double(min(i,ni-i))*ddi;
      // const double ky = double(min(j,nj-j))*ddj;
      // const double kz = double(k)*ddk;
        int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        double kz = id_i * ddi;
        double ky = id_j * ddj;
        double kx = id_k * ddk;  
        double k2 = kx*kx + ky*ky + kz*kz ;
        if ( k2 == 0 ) k2 = 1.0;
        double factor;
        if      (direction == 0) factor = kz / k2 / D;
        else if (direction == 1) factor = ky / k2 / D;
        else if (direction == 2) factor = kx / k2 / D;
        else printf("Wrong direction %d\n", direction ); 
         // multiply b by 1j*factor ( Imaginary Number)
        return cufftDoubleComplex{-factor*b.y,factor*b.x};
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

void FFT_3D::Filter_rescale_by_power_spectrum( double *input, double *output, bool in_device, int size, double *dev_k, double *dev_pk ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_, nk = nk_;
  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;
  
  if ( in_device ){
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 
  
  // Provide FFT filter with a lambda that multiplies by P(k)
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
        // const double kx = double(min(i,ni-i))*ddi;
        // const double ky = double(min(j,nj-j))*ddj;
        // const double kz = double(k)*ddk;
        int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        double kz = id_i * ddi;
        double ky = id_j * ddj;
        double kx = id_k * ddk;  
        const double k_mag = sqrt( kx*kx + ky*ky + kz*kz );
        double pk = linear_interpolation( k_mag, dev_k, dev_pk, size );
        if ( i==1 && j==1 && k==1 ) printf("###### kx: %e  ky: %e  kz: %e  k_mag: %e  pk: %e \n", kx, ky, kz, k_mag, pk );  
        pk = sqrt(pk);
        return cufftDoubleComplex{pk*b.x,pk*b.y};
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
        const double i2 = sqr(double(min(i,ni-i))*ddi);
        const double j2 = sqr(double(min(j,nj-j))*ddj);
        const double k2 = sqr(double(k)*ddk);
        const double d = -1.0/(i2+j2+k2);
        return cufftDoubleComplex{d*b.x,d*b.y};
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


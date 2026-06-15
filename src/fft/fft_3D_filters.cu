#if defined(PARIS) && defined(FFT) 


#include "fft_3D.h"
#include "../utils/gpu.hpp"
#include "../io/io.h"
#include "../global/global.h"
#include <cassert>
#include <cfloat>
#include <climits>


__host__ __device__ static inline Real sqr(const Real x) { return x*x; }
__host__ __device__ static inline Real Sqr(const double x) { return x * x; }

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
  Real xl, xr, yl, yr;
  xl = x_vals[indx-1];
  xr = x_vals[indx];
  yl = y_vals[indx-1];
  yr = y_vals[indx];  
  if ( x < xl || x > xr ) printf(" ##################### Interpolation error:   x: %e  xl: %e  xr: %e   indx: %d\n", x, xl, xr, indx );
  return  yl + ( x - xl ) / ( xr - xl ) * ( yr - yl );
}

__device__ Real log_log_interpolation( Real x, Real *x_vals, Real *y_vals, int N ){
  if ( x <= x_vals[0] ){
    printf(" x: %f  outside of interplation range (xv0 = %f).\n", x, x_vals[0] );
    return y_vals[0];
  }
  if ( x >= x_vals[N-1] ){
    printf(" x: %f  outside of interplation range (xvn-1 = %f). \n", x, x_vals[N-1] );
    return y_vals[N-1];
  }
  int indx = 0;
  while( x_vals[indx] < x ) indx +=1;
  Real xl, xr, yl, yr;
  xl = log(x_vals[indx-1]);
  xr = log(x_vals[indx]);
  yl = log(y_vals[indx-1]);
  yr = log(y_vals[indx]);  
  if ( x < exp(xl) || x > exp(xr) ) printf(" ##################### Interpolation error:   x: %e  xl: %e  xr: %e   indx: %d\n", x, exp(xl), exp(xr), indx );
  return  exp(yl + ( log(x) - xl ) / ( xr - xl ) * ( yr - yl ));
}



void FFT_3D::Filter_rescale_by_k_k2( Real *input, Real *output, bool in_device, int direction, Real D ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_, nk = nk_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  if ( in_device ){
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 

  // Provide FFT filter with a lambda that multiplies by k / k^2 / D
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
        // Get the global indices 
        int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        // Compute kx, ky, and kz from the indices
        Real kz = id_i * ddi;
        Real ky = id_j * ddj;
        Real kx = id_k * ddk;  
        // Compute the magnitude of k squared
        Real k2 = kx*kx + ky*ky + kz*kz ;
        if ( k2 == 0 ) k2 = 1.0;
        Real factor;
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
    GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  } 
}
/*! void FFT_3D::Filter_rescale_by_power_spectrum( Real *input, Real *output, bool in_device, int size, Real *dev_k, Real *dev_pk ) const
 *  \brief Filter that rescales by a scale-dependent power spectrum */
void FFT_3D::Filter_rescale_by_power_spectrum( Real *input, Real *output, bool in_device, int size, Real *dev_k, Real *dev_pk ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_, nk = nk_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  chprintf("inputBytes %d outputBytes %d\n",inputBytes_,outputBytes_);
  
  if ( in_device ){
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 
  
  // Provide FFT filter with a lambda that multiplies by P(k)
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
        // Get the global indices 
        int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        // Compute kx, ky, and kz from the indices
        Real kz = id_i * ddi;
        Real ky = id_j * ddj;
        Real kx = id_k * ddk;  
        // Compute the magnitude of k 
        const Real k_mag = sqrt( kx*kx + ky*ky + kz*kz );
        // these give similar answers
        //Real pk = linear_interpolation( k_mag, dev_k, dev_pk, size ); // linear interp of P(k)
        Real pk = log_log_interpolation( k_mag, dev_k, dev_pk, size );  // log log interp of P(k)
        pk = sqrt(pk);
        return cufftDoubleComplex{pk*b.x,pk*b.y};
      } else {
        return cufftDoubleComplex{0.0,0.0};
      }
    });
    
  if ( in_device ){
    GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  } 
    
}

void FFT_3D::Filter_inv_k2( Real *const input, Real *const output, bool in_device ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;
  
  if ( in_device ){
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 

  // Poisson-solve constants that depend on divergence-operator approximation
  const int nk    = nk_;

  const Real si = M_PI / Real(ni);
  const Real sj = M_PI / Real(nj);
  const Real sk = M_PI / Real(nk);
/*
  #ifdef PARIS_3PT
  const Real si = M_PI / Real(ni);
  const Real sj = M_PI / Real(nj);
  const Real sk = M_PI / Real(nk);
  #elif defined PARIS_5PT
  //const int nk    = nk_;
  const Real si = 2.0 * M_PI / Real(ni);
  const Real sj = 2.0 * M_PI / Real(nj);
  const Real sk = 2.0 * M_PI / Real(nk);
  #endif
*/

  const int n = ni * nj * nk;

  // Provide FFT filter with a lambda that does 1/k^2 solve in frequency space
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {

        /*const Real i2 = Sqr(sin(Real(min(i, ni - i)) * si) * ddi);
        const Real j2 = Sqr(sin(Real(min(j, nj - j)) * sj) * ddj);
        const Real k2 = Sqr(sin(Real(k) * sk) * ddk);*/
        /*const Real i2 = Sqr(Real(min(i, ni - i)) * ddi);
        const Real j2 = Sqr(Real(min(j, nj - j)) * ddj);
        const Real k2 = Sqr(Real(k) * ddk);
        const Real d = -1./(i2+j2+k2);*/
/*
  #ifdef PARIS_3PT
        const Real i2 = Sqr(sin(Real(min(i, ni - i)) * si) * ddi);
        const Real j2 = Sqr(sin(Real(min(j, nj - j)) * sj) * ddj);
        const Real k2 = Sqr(sin(Real(k) * sk) * ddk);
  #elif defined PARIS_5PT
        const Real ci = cos(Real(min(i, ni - i)) * si);
        const Real cj = cos(Real(min(j, nj - j)) * sj);
        const Real ck = cos(Real(k) * sk);
        const Real i2 = ddi * (2.0 * ci * ci - 16.0 * ci + 14.0);
        const Real j2 = ddj * (2.0 * cj * cj - 16.0 * cj + 14.0);
        const Real k2 = ddk * (2.0 * ck * ck - 16.0 * ck + 14.0);
  #else
        const Real i2 = Sqr(Real(min(i, ni - i)) * ddi);
        const Real j2 = Sqr(Real(min(j, nj - j)) * ddj);
        const Real k2 = Sqr(Real(k) * ddk);
  #endif
*/



        // Get the global indices 
        //int id_i = i < ni/2 ? i : i - ni;
        //int id_j = j < nj/2 ? j : j - nj;
        //int id_k = k < nk/2 ? k : k - nk;
        // Compute kx, ky, and kz from the indices
        //Real kz = id_i * ddi;
        //Real ky = id_j * ddj;
        //Real kx = id_k * ddk; 
        //const Real dx = 1./(ddi/(2.*M_PI)); // grid size
        //const Real k2 = 2./(dx*dx) * (3 - cos(kx*dx) - cos(ky*dx) - cos(kz*dx) );
        //const Real f = 2.0*M_PI;
        //Real k2 = 2./(dx*dx) * (3 - cos(f*Real(id_i)) - cos(f*Real(id_j)) - cos(f*Real(id_k)) );
        //if(k2==0)
        //  return cufftDoubleComplex{0.0,0.0};
          //k2 = 1;
        // ddi = 2.0 * MPI/50000
        // dx  = (2.0*M_PI/(ddi*ni);
/*
        Real dx = (2.0*M_PI/(ddi*ni));
        Real dy = (2.0*M_PI/(ddj*nj));
        Real dz = (2.0*M_PI/(ddk*nk));
        Real kx = id_i * ddi;
        Real ky = id_j * ddj;
        Real kz = id_k * ddk; 
        Real i2 = (2/(dx*dx)) * (cos(kx*dx)-1);
        Real j2 = (2/(dy*dy)) * (cos(ky*dy)-1);
        Real k2 = (2/(dz*dz)) * (cos(kz*dz)-1);
        //Real k_sq = (i2+j2+k2);
        Real k_sq = (kx*kx + ky*ky + kz*kz);
        if(k_sq==0)
          return cufftDoubleComplex{0.0,0.0};
        //Real d = -1./k_sq;
        Real d = 1./k_sq;
        //error REVERT
*/
        /*const Real i2 = Sqr(Real(min(i, ni - i)) * ddi);
        const Real j2 = Sqr(Real(min(j, nj - j)) * ddj);
        const Real k2 = Sqr(Real(k) * ddk);*/

        //HERE
        /*const Real i2 = Sqr(Real(min(i, ni - i)) * ddi);
        const Real j2 = Sqr(Real(min(j, nj - j)) * ddj);
        const Real k2 = Sqr(Real(k) * ddk);
        const Real d = -1.0/(i2+j2+k2);*/

        //const Real i2 = Sqr(Real(min(i, ni - i)) * ddi);
        //const Real j2 = Sqr(Real(min(j, nj - j)) * ddj);
        //const Real k2 = Sqr(Real(k) * ddk);
        // Get the global indices 
        /*int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        // Compute kx, ky, and kz from the indices
        Real kz = id_i * ddi;
        Real ky = id_j * ddj;
        Real kx = id_k * ddk;  */
        // Compute the magnitude of k 
        //const Real k_mag = sqrt( kx*kx + ky*ky + kz*kz );

        /*// Get the global indices 
        int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        // Compute kx, ky, and kz from the indices
        Real kz = id_i * ddi;
        Real ky = id_j * ddj;
        Real kx = id_k * ddk;  
        // Compute the magnitude of k squared
        Real k2 = kx*kx + ky*ky + kz*kz ;
        if ( k2 == 0 ) k2 = 1.0;*/
        //const Real d = -1.0/(i2+j2+k2);
        //const Real d = -1.0/(kx*kx + ky*ky + kz*kz);

        // 1/k^2

        // Get the global indices 
        int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        // Compute kx, ky, and kz from the indices
        Real kz = id_i * ddi;
        Real ky = id_j * ddj;
        Real kx = id_k * ddk;  
        // Compute the magnitude of k 
        const Real k_sq =  kx*kx + ky*ky + kz*kz ;
        if(k_sq==0)
          return cufftDoubleComplex{0.0,0.0};
        //const Real d = -1.0/k_sq;
        const Real d = 1.0/k_sq; // sign change doesn't change oddness

        return cufftDoubleComplex{d*b.x,d*b.y};
      } else {
        return cufftDoubleComplex{0.0,0.0};
      }
    });
    
  if ( in_device ){
    GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  } 
}


/*! void FFT_3D::Filter_identity( const size_t bytes, Real *const input, Real *const output) const
 *  \brief The identity function filter */
void FFT_3D::Filter_identity( Real *const input, Real *output, bool in_device ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  // copy input into byte array
  if ( in_device ){
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 
  
  // Provide FFT filter that does nothing
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      return b;
    });

  // copy results to output
  if ( in_device ){
    GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  } 
}

/*! void FFT_3D::Filter_rescale( const size_t bytes, Real *const input, Real A, Real *const output) const
 *  \brief A filter that rescales the grid in Fourier space*/
void FFT_3D::Filter_rescale( Real *const input, Real A, Real *output, bool in_device ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const int nk    = nk_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  // copy input into byte array
  if ( in_device ){
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 
  
  const int n = ni * nj * nk;

  //rescale and apply offset
  //gpuFor(
  //    n, GPU_LAMBDA(const int i) { db_[i] = A * db_[i]; });
  // Provide FFT filter that simply rescales
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
        return cufftDoubleComplex{A*b.x,A*b.y};
      } else {
        return cufftDoubleComplex{0.0,0.0};
      }
    });

  // copy results to output
  if ( in_device ){
    //GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
    GPU_Error_Check( cudaMemcpy( output, db_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    //GPU_Error_Check( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
    GPU_Error_Check( cudaMemcpy( output, db_, outputBytes_, cudaMemcpyDeviceToHost));
  } 
}


#endif

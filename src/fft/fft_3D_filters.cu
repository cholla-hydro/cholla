/*! /file
 *  /brief Implements FFT filtering machinery */

#include "../global/global.h"

#if defined(PARIS) && defined(FFT)

  #include <cassert>
  #include <cfloat>
  #include <climits>

  #include "../io/io.h"
  #include "../utils/gpu.hpp"
  #include "fft_3D.h"

__host__ __device__ static inline Real sqr(const Real x) { return x * x; }

__device__ Real linear_interpolation(Real x, Real *x_vals, Real *y_vals, int N)
{
  if (x <= x_vals[0]) {
    printf(" x: %f  outside of interplation range.\n", x);
    return y_vals[0];
  }
  if (x >= x_vals[N - 1]) {
    printf(" x: %f  outside of interplation range.\n", x);
    return y_vals[N - 1];
  }
  int indx = 0;
  while (x_vals[indx] < x) indx += 1;
  Real xl, xr, yl, yr;
  xl = x_vals[indx - 1];
  xr = x_vals[indx];
  yl = y_vals[indx - 1];
  yr = y_vals[indx];
  if (x < xl || x > xr)
    printf(" ##################### Interpolation error:   x: %e  xl: %e  xr: %e   indx: %d\n", x, xl, xr, indx);
  return yl + (x - xl) / (xr - xl) * (yr - yl);
}

__device__ Real log_log_interpolation(Real x, Real *x_vals, Real *y_vals, int N)
{
  if (x <= x_vals[0]) {
    printf(" x: %f  outside of interplation range (xv0 = %f).\n", x, x_vals[0]);
    return y_vals[0];
  }
  if (x >= x_vals[N - 1]) {
    printf(" x: %f  outside of interplation range (xvn-1 = %f). \n", x, x_vals[N - 1]);
    return y_vals[N - 1];
  }
  int indx = 0;
  while (x_vals[indx] < x) indx += 1;
  Real xl, xr, yl, yr;
  xl = log(x_vals[indx - 1]);
  xr = log(x_vals[indx]);
  yl = log(y_vals[indx - 1]);
  yr = log(y_vals[indx]);
  if (x < exp(xl) || x > exp(xr))
    printf(" ##################### Interpolation error:   x: %e  xl: %e  xr: %e   indx: %d\n", x, exp(xl), exp(xr),
           indx);
  return exp(yl + (log(x) - xl) / (xr - xl) * (yr - yl));
}

void FFT_3D::Filter_rescale_by_k_k2(Real *input, Real *output, bool in_device, int direction, Real D) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_, nk = nk_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  if (in_device) {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyHostToDevice));
  }

  // Provide FFT filter with a lambda that multiplies by k / k^2 / D
  henry_->filter(bytes, db_, da_, [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) {
    if (i != 0 || j != 0 || k != 0) {
      // Get the global indices
      int id_i = i < ni / 2 ? i : i - ni;
      int id_j = j < nj / 2 ? j : j - nj;
      int id_k = k < nk / 2 ? k : k - nk;
      // Compute kx, ky, and kz from the indices
      Real kx = id_i * ddi;
      Real ky = id_j * ddj;
      Real kz = id_k * ddk;
      // Compute the magnitude of k squared
      Real k2 = kx * kx + ky * ky + kz * kz;
      if (k2 == 0) k2 = 1.0;
      Real factor;
      if (direction == 0)
        factor = kx / k2 / D;
      else if (direction == 1)
        factor = ky / k2 / D;
      else if (direction == 2)
        factor = kz / k2 / D;
      else
        printf("Wrong direction %d\n", direction);
      // multiply b by 1j*factor ( Imaginary Number)
      return cufftDoubleComplex{-factor * b.y, factor * b.x};
    } else {
      return cufftDoubleComplex{0.0, 0.0};
    }
  });

  if (in_device) {
    GPU_Error_Check(cudaMemcpy(output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  }
}
void FFT_3D::Filter_rescale_by_power_spectrum(Real *input, Real *output, bool in_device, int size, Real *dev_k,
                                              Real *dev_pk) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_, nk = nk_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  if (in_device) {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyHostToDevice));
  }

  // Provide FFT filter with a lambda that multiplies by P(k)
  henry_->filter(bytes, db_, da_, [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) {
    if (i != 0 || j != 0 || k != 0) {
      // Get the global indices
      int id_i = i < ni / 2 ? i : i - ni;
      int id_j = j < nj / 2 ? j : j - nj;
      // no difference?
      // int id_k = k < nk/2 ? k : k - nk;
      int id_k         = k;
      const Real k_mag = sqrt(sqr(id_i * ddi) + sqr(id_j * ddj) + sqr(id_k * ddk));

      // these give similar answers
      // Real pk = linear_interpolation( k_mag, dev_k, dev_pk, size ); // linear interp of P(k)
      // Real pk = log_log_interpolation( k_mag, dev_k, dev_pk, size );  // log log interp of P(k)
      Real pk = log_log_interpolation(k_mag, dev_k, dev_pk, size);  // log log interp of P(k)
      pk      = sqrt(pk);
      return cufftDoubleComplex{pk * b.x, pk * b.y};
    } else {
      return cufftDoubleComplex{0.0, 0.0};
    }
  });

  if (in_device) {
    GPU_Error_Check(cudaMemcpy(output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  }
}

void FFT_3D::Filter_inv_k2(Real *const input, Real *const output, bool in_device) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const double dx = dx_, dy = dy_, dz = dz_;
  const size_t bytes = minBytes_;

  /* For 128^3 in 25 Mpc/h
    ngpu = 1, Cosmological ICs: RMS  of phi field 1.474697e+07
    ngpu = 2, Cosmological ICs: RMS  of phi field 1.275145e+07
    ngpu = 4, Cosmological ICs: RMS  of phi field 1.430082e+07*/
  // ngpu=1, Cosmological ICs: RMS  of overdensity field after P(k) 5.331810e+00
  // ngpu=2, Cosmological ICs: RMS  of overdensity field after P(k) 5.333359e+00
  // ngpu=4, Cosmological ICs: RMS  of overdensity field after P(k) 5.308875e+00

  /* For 256^3 in 50 Mpc/h
    ngpu = 1, Cosmological ICs: RMS  of phi field 4.423029e+07
    ngpu = 2, Cosmological ICs: RMS  of phi field 3.297570e+07
    ngpu = 4, Cosmological ICs: RMS  of phi field 2.809177e+07*/
  // ngpu = 4, Cosmological ICs: RMS  of overdensity field after P(k) 5.301260e+00

  /* For 384^3 in 75 Mpc/h
    ngpu = 2, Cosmological ICs: RMS  of phi field 4.042792e+07
    ngpu = 4, Cosmological ICs: RMS  of phi field 5.219293e+07*/
  // ngpu=4, Cosmological ICs: RMS  of overdensity field after P(k) 5.310492e+00

  /* For 512^3 in 100 Mpc/h
    ngpu = 4, Cosmological ICs: RMS  of phi field 8.125767e+07 */
  // ngpu=4, Cosmological ICs: RMS  of overdensity field after P(k) 5.327733e+00

  // Poisson-solve constants that depend on divergence-operator approximation
  const int nk = nk_;
  // note that we may need these differentials for derivatives
  // keeping them here
  /*
  #ifdef PARIS_3PT
      ddi_=(2.0 * double(n[0] - 1) / (hi[0] - lo_[0]));
      ddj_=(2.0 * double(n[1] - 1) / (hi[1] - lo_[1]));
      ddk_=(2.0 * double(n[2] - 1) / (hi[2] - lo_[2]));
  #elif defined PARIS_5PT
      ddi_=(Sqr(double(n[0] - 1) / (hi[0] - lo_[0])) / 6.0);
      ddj_=(Sqr(double(n[1] - 1) / (hi[1] - lo_[1])) / 6.0);
      ddk_=(Sqr(double(n[2] - 1) / (hi[2] - lo_[2])) / 6.0);
  #else
      ddi_=(2.0 * M_PI * double(n[0] - 1) / (double(n[0]) * (hi[0] - lo_[0])));
      ddj_=(2.0 * M_PI * double(n[1] - 1) / (double(n[1]) * (hi[1] - lo_[1])));
      ddk_=(2.0 * M_PI * double(n[2] - 1) / (double(n[2]) * (hi[2] - lo_[2])));
  #endif*/
  if (in_device) {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyHostToDevice));
  }


  // Provide FFT filter with a lambda that does 1/k^2 solve in frequency space
  henry_->filter(bytes, db_, da_, [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) {
    if (i != 0 || j != 0 || k != 0) {
      // Get the global indices
      int id_i = i < ni / 2 ? i : i - ni;
      int id_j = j < nj / 2 ? j : j - nj;
      // no difference?
      // int id_k = k < nk/2 ? k : k - nk;
      int id_k = k;
      Real kx  = id_i * ddi;
      Real ky  = id_j * ddj;
      Real kz  = id_k * ddk;
      // spectral
      // double ksq = kx*kx + ky*ky + kz*kz;
      // this gives the same answer as spectral
      const double ci = cos(2.0 * M_PI * id_i / Real(ni));
      const double cj = cos(2.0 * M_PI * id_j / Real(nj));
      const double ck = cos(2.0 * M_PI * id_k / Real(nk));
      const double i2 = (2.0 * ci * ci - 16.0 * ci + 14.0) / (6 * dx * dx);
      const double j2 = (2.0 * cj * cj - 16.0 * cj + 14.0) / (6 * dy * dy);
      const double k2 = (2.0 * ck * ck - 16.0 * ck + 14.0) / (6 * dz * dz);
      double ksq      = i2 + j2 + k2;
      // this gives the same answer as spectral
      // double ksq = sqr(2*sin(0.5*kx*dx)/dx) + sqr(2*sin(0.5*ky*dy)/dy) + sqr(2*sin(0.5*kz*dz)/dz);
      double d = -1. / ksq;

      // heavily dependent on the box size, the largest modes
      // are up -weighted by the 1/k^2 dependence

      return cufftDoubleComplex{d * b.x, d * b.y};
    } else {
      return cufftDoubleComplex{0.0, 0.0};
    }
  });

  if (in_device) {
    GPU_Error_Check(cudaMemcpy(output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  }
}

void FFT_3D::Filter_identity(Real *const input, Real *output, bool in_device) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  // copy input into byte array
  if (in_device) {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyHostToDevice));
  }

  // Provide FFT filter that does nothing
  henry_->filter(bytes, db_, da_,
                 [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) { return b; });

  // copy results to output
  if (in_device) {
    GPU_Error_Check(cudaMemcpy(output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  }
}

void FFT_3D::Filter_rescale(Real *const input, Real A, Real *output, bool in_device) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const int nk   = nk_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  // copy input into byte array
  if (in_device) {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(db_, input, inputBytes_, cudaMemcpyHostToDevice));
  }

  const int n = ni * nj * nk;

  // Provide FFT filter that simply rescales
  henry_->filter(bytes, db_, da_, [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) {
    if (i != 0 || j != 0 || k != 0) {
      return cufftDoubleComplex{A * b.x, A * b.y};
    } else {
      return cufftDoubleComplex{0.0, 0.0};
    }
  });

  // copy results to output
  if (in_device) {
    GPU_Error_Check(cudaMemcpy(output, db_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    GPU_Error_Check(cudaMemcpy(output, db_, outputBytes_, cudaMemcpyDeviceToHost));
  }
}

#endif

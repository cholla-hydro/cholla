#ifdef PARIS

  #include <cmath>
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
#endif

  #include "ParisPeriodic.hpp"

__host__ __device__ static inline double sqr(const double x) { return x * x; }

ParisPeriodic::ParisPeriodic(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3], double dx)
    : ni_(n[0]),
      nj_(n[1]),
  #ifdef PARIS_3PT
      nk_(n[2]),
      ddi_(2.0 * double(n[0] - 1) / (hi[0] - lo[0])),
      ddj_(2.0 * double(n[1] - 1) / (hi[1] - lo[1])),
      ddk_(2.0 * double(n[2] - 1) / (hi[2] - lo[2])),
  #elif defined PARIS_5PT
      nk_(n[2]),
      ddi_(sqr(double(n[0] - 1) / (hi[0] - lo[0])) / 6.0),
      ddj_(sqr(double(n[1] - 1) / (hi[1] - lo[1])) / 6.0),
      ddk_(sqr(double(n[2] - 1) / (hi[2] - lo[2])) / 6.0),
  #else
      ddi_{2.0 * M_PI * double(n[0] - 1) / (double(n[0]) * (hi[0] - lo[0]))},
      ddj_{2.0 * M_PI * double(n[1] - 1) / (double(n[1]) * (hi[1] - lo[1]))},
      ddk_{2.0 * M_PI * double(n[2] - 1) / (double(n[2]) * (hi[2] - lo[2]))},
  #endif
      henry(n, lo, hi, m, id),
      dx_(dx)
{
}

void ParisPeriodic::solvePotential(const size_t bytes, double *const density, double *const potential) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;

  // Poisson-solve constants that depend on divergence-operator approximation
  #ifdef PARIS_3PT
  const int nk    = nk_;
  const double si = M_PI / double(ni);
  const double sj = M_PI / double(nj);
  const double sk = M_PI / double(nk);
  #elif defined PARIS_5PT
  const int nk    = nk_;
  const double si = 2.0 * M_PI / double(ni);
  const double sj = 2.0 * M_PI / double(nj);
  const double sk = 2.0 * M_PI / double(nk);
  #endif

  // Provide FFT filter with a lambda that does Poisson solve in frequency space
  henry.filter(bytes, density, potential,
               [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) {
                 if (i || j || k) {
  #ifdef PARIS_3PT
                   const double i2 = sqr(sin(double(min(i, ni - i)) * si) * ddi);
                   const double j2 = sqr(sin(double(min(j, nj - j)) * sj) * ddj);
                   const double k2 = sqr(sin(double(k) * sk) * ddk);
  #elif defined PARIS_5PT
          const double ci = cos(double(min(i, ni - i)) * si);
          const double cj = cos(double(min(j, nj - j)) * sj);
          const double ck = cos(double(k) * sk);
          const double i2 = ddi * (2.0 * ci * ci - 16.0 * ci + 14.0);
          const double j2 = ddj * (2.0 * cj * cj - 16.0 * cj + 14.0);
          const double k2 = ddk * (2.0 * ck * ck - 16.0 * ck + 14.0);
  #else
          const double i2 = sqr(double(min(i, ni - i)) * ddi);
          const double j2 = sqr(double(min(j, nj - j)) * ddj);
          const double k2 = sqr(double(k) * ddk);
  #endif
                   const double d = -1.0 / (i2 + j2 + k2);
                   return cufftDoubleComplex{d * b.x, d * b.y};
                 } else {
                   return cufftDoubleComplex{0.0, 0.0};
                 }
               });
}


template<int IJ> __device__ cufftDoubleComplex EddingtonTensorGF(double dx, int ni, int nj, double ddi, double ddj, double ddk, const int i, const int j, const int k, const cufftDoubleComplex b)
{
    const double z = (i<=ni/2 ? i : i - ni) * ddi;
    const double y = (j<=nj/2 ? j : j - nj) * ddj;
    const double x = k * ddk;

    double w = sqrt(x*x+y*y+z*z);
    if(i==0 && j==0 && k==0)
    {
        w = 0.5*ddi; // reasonable approximation for a DC != 0 GF
    }

    //
    //  1/(4pi(r^2+eps^2)) -> pi/(2k)*exp(-k*eps)
    //  r^ir^j/(4pi(r^2+eps^2)^2) -> pi/(4k)*exp(-k*eps)*[delta^{ij} - k^ik^j/k^2(1+eps*k)]
    //  eps^2/3/(4pi(r^2+eps^2)^2) -> Pi eps/12*exp(-k*eps)
    //  See comment in initial_conditions.cpp for Iliev15 tests
    //
    const double epsOT = dx;
    const double GwOT = 0.5*M_PI/w*exp(-epsOT*w);

    const double epsET = 2*epsOT;
    const double GwET1 = 0.25*M_PI*exp(-epsET*w)/w*(1+epsET*w/3);
    const double GwET2 = 0.25*M_PI*exp(-epsET*w)/pow(w,3)*(1+epsET*w);

    switch(IJ)
    {
        case 0: return cufftDoubleComplex{(GwET1-GwET2*x*x)*b.x,(GwET1-GwET2*x*x)*b.y};
        case 1: return cufftDoubleComplex{(    0-GwET2*y*x)*b.x,(    0-GwET2*y*x)*b.y};
        case 2: return cufftDoubleComplex{(GwET1-GwET2*y*y)*b.x,(GwET1-GwET2*y*y)*b.y};
        case 3: return cufftDoubleComplex{(    0-GwET2*z*x)*b.x,(    0-GwET2*z*x)*b.y};
        case 4: return cufftDoubleComplex{(    0-GwET2*z*y)*b.x,(    0-GwET2*z*y)*b.y};
        case 5: return cufftDoubleComplex{(GwET1-GwET2*z*z)*b.x,(GwET1-GwET2*z*z)*b.y};
        case 6: return cufftDoubleComplex{GwOT*b.x,GwOT*b.y};
        default: return cufftDoubleComplex{0,0};
    };
}

void ParisPeriodic::solveEddingtonTensor(size_t bytes, double *source, double *tensor, int component) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;
  auto dx = dx_;

  // Provide FFT filter with a lambda that does Poisson solve in frequency space
  henry.filter(bytes, source, tensor,
               [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) {
                    static decltype(&EddingtonTensorGF<0>) gfs[] = { EddingtonTensorGF<0>, EddingtonTensorGF<1>, EddingtonTensorGF<2>, EddingtonTensorGF<3>, EddingtonTensorGF<4>, EddingtonTensorGF<5>, EddingtonTensorGF<6> };
                    return gfs[component](dx,ni,nj,ddi,ddj,ddk,i,j,k,b);
               });
}

#endif

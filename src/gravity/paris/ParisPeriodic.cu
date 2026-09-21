#ifdef PARIS

  #include <cmath>

  #include "ParisPeriodic.hpp"

__host__ __device__ static inline double Sqr(const double x) { return x * x; }

ParisPeriodic::ParisPeriodic(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3],
                             double dx)
    : ni_(n[0]),
      nj_(n[1]),
  #ifndef GRAVITY_5_POINTS_GRADIENT
      nk_(n[2]),
      ddi_(2.0 * double(n[0] - 1) / (hi[0] - lo[0])),
      ddj_(2.0 * double(n[1] - 1) / (hi[1] - lo[1])),
      ddk_(2.0 * double(n[2] - 1) / (hi[2] - lo[2])),
  #else
      nk_(n[2]),
      ddi_(Sqr(double(n[0] - 1) / (hi[0] - lo[0])) / 6.0),
      ddj_(Sqr(double(n[1] - 1) / (hi[1] - lo[1])) / 6.0),
      ddk_(Sqr(double(n[2] - 1) / (hi[2] - lo[2])) / 6.0),
  #endif
      henry(n, lo, hi, m, id),
      dx_(dx),                                                                  // RT
      dd2i_(2.0 * M_PI * double(n[0] - 1) / (double(n[0]) * (hi[0] - lo[0]))),  // RT
      dd2j_(2.0 * M_PI * double(n[1] - 1) / (double(n[1]) * (hi[1] - lo[1]))),  // RT
      dd2k_(2.0 * M_PI * double(n[2] - 1) / (double(n[2]) * (hi[2] - lo[2])))   // RT
{
}

void ParisPeriodic::solve(const size_t bytes, double *const density, double *const potential) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;

  // Poisson-solve constants that depend on divergence-operator approximation
  #ifndef GRAVITY_5_POINTS_GRADIENT
  const int nk    = nk_;
  const double si = M_PI / double(ni);
  const double sj = M_PI / double(nj);
  const double sk = M_PI / double(nk);
  #else
  const int nk    = nk_;
  const double si = 2.0 * M_PI / double(ni);
  const double sj = 2.0 * M_PI / double(nj);
  const double sk = 2.0 * M_PI / double(nk);
  #endif

  // Provide FFT filter with a lambda that does Poisson solve in frequency space
  henry.filter(bytes, density, potential,
               [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) {
                 if (i || j || k) {
  #ifndef GRAVITY_5_POINTS_GRADIENT
                   const double i2 = Sqr(sin(double(min(i, ni - i)) * si) * ddi);
                   const double j2 = Sqr(sin(double(min(j, nj - j)) * sj) * ddj);
                   const double k2 = Sqr(sin(double(k) * sk) * ddk);
  #else
          const double ci = cos(double(min(i, ni - i)) * si);
          const double cj = cos(double(min(j, nj - j)) * sj);
          const double ck = cos(double(k) * sk);
          const double i2 = ddi * (2.0 * ci * ci - 16.0 * ci + 14.0);
          const double j2 = ddj * (2.0 * cj * cj - 16.0 * cj + 14.0);
          const double k2 = ddk * (2.0 * ck * ck - 16.0 * ck + 14.0);
  #endif
                   const double d = -1.0 / (i2 + j2 + k2);
                   return cufftDoubleComplex{d * b.x, d * b.y};
                 } else {
                   return cufftDoubleComplex{0.0, 0.0};
                 }
               });
}

  #ifdef RT
template <int IJ>
__device__ cufftDoubleComplex EddingtonTensorGF(double dx, int ni, int nj, double dd2i, double dd2j, double dd2k,
                                                const int i, const int j, const int k, const cufftDoubleComplex b)
{
  const double z = (i <= ni / 2 ? i : i - ni) * dd2i;
  const double y = (j <= nj / 2 ? j : j - nj) * dd2j;
  const double x = k * dd2k;

  double w = sqrt(x * x + y * y + z * z);
  if (i == 0 && j == 0 && k == 0) {
    w = 0.5 * dd2i;  // reasonable approximation for a DC != 0 GF
  }

  //
  //  1/(4pi(r^2+eps^2)) -> pi/(2k)*exp(-k*eps)
  //  r^ir^j/(4pi(r^2+eps^2)^2) -> pi/(4k)*exp(-k*eps)*[delta^{ij} - k^ik^j/k^2(1+eps*k)]
  //  eps^2/3/(4pi(r^2+eps^2)^2) -> Pi eps/12*exp(-k*eps)
  //  See comment in initial_conditions.cpp for Iliev15 tests
  //
  const double epsOT = dx;
  const double GwOT  = 0.5 * M_PI / w * exp(-epsOT * w);

  const double epsET = 2 * epsOT;
  const double GwET1 = 0.25 * M_PI * exp(-epsET * w) / w * (1 + epsET * w / 3);
  const double GwET2 = 0.25 * M_PI * exp(-epsET * w) / pow(w, 3) * (1 + epsET * w);

  switch (IJ) {
    case 0:
      return cufftDoubleComplex{(GwET1 - GwET2 * x * x) * b.x, (GwET1 - GwET2 * x * x) * b.y};
    case 1:
      return cufftDoubleComplex{(0 - GwET2 * y * x) * b.x, (0 - GwET2 * y * x) * b.y};
    case 2:
      return cufftDoubleComplex{(GwET1 - GwET2 * y * y) * b.x, (GwET1 - GwET2 * y * y) * b.y};
    case 3:
      return cufftDoubleComplex{(0 - GwET2 * z * x) * b.x, (0 - GwET2 * z * x) * b.y};
    case 4:
      return cufftDoubleComplex{(0 - GwET2 * z * y) * b.x, (0 - GwET2 * z * y) * b.y};
    case 5:
      return cufftDoubleComplex{(GwET1 - GwET2 * z * z) * b.x, (GwET1 - GwET2 * z * z) * b.y};
    case 6:
      return cufftDoubleComplex{GwOT * b.x, GwOT * b.y};
    default:
      return cufftDoubleComplex{0, 0};
  };
}

void ParisPeriodic::solveEddingtonTensor(size_t bytes, double *source, double *tensor, int component) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const double dd2i = dd2i_, dd2j = dd2j_, dd2k = dd2k_;
  auto dx = dx_;

  // Provide FFT filter with a lambda that does Poisson solve in frequency space
  henry.filter(bytes, source, tensor,
               [=] __device__(const int i, const int j, const int k, const cufftDoubleComplex b) {
                 static decltype(&EddingtonTensorGF<0>) gfs[] = {
                     EddingtonTensorGF<0>, EddingtonTensorGF<1>, EddingtonTensorGF<2>, EddingtonTensorGF<3>,
                     EddingtonTensorGF<4>, EddingtonTensorGF<5>, EddingtonTensorGF<6>};
                 return gfs[component](dx, ni, nj, dd2i, dd2j, dd2k, i, j, k, b);
               });
}
  #endif  // RT

#endif

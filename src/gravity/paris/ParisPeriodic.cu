#ifdef PARIS

#include "ParisPeriodic.hpp"

#include <cmath>

__host__ __device__ static inline double sqr(const double x) { return x*x; }

ParisPeriodic::ParisPeriodic(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]):
  ni_(n[0]),
  nj_(n[1]),
#ifdef PARIS_3PT
  nk_(n[2]),
  ddi_(2.0*double(n[0]-1)/(hi[0]-lo[0])),
  ddj_(2.0*double(n[1]-1)/(hi[1]-lo[1])),
  ddk_(2.0*double(n[2]-1)/(hi[2]-lo[2])),
#elif defined PARIS_5PT
  nk_(n[2]),
  ddi_(sqr(double(n[0]-1)/(hi[0]-lo[0]))/6.0),
  ddj_(sqr(double(n[1]-1)/(hi[1]-lo[1]))/6.0),
  ddk_(sqr(double(n[2]-1)/(hi[2]-lo[2]))/6.0),
#else
  ddi_{2.0*M_PI*double(n[0]-1)/(double(n[0])*(hi[0]-lo[0]))},
  ddj_{2.0*M_PI*double(n[1]-1)/(double(n[1])*(hi[1]-lo[1]))},
  ddk_{2.0*M_PI*double(n[2]-1)/(double(n[2])*(hi[2]-lo[2]))},
#endif
  henry(n,lo,hi,m,id)
{ }

void ParisPeriodic::solve(const size_t bytes, double *const density, double *const potential) const
{
  const int ni = ni_, nj = nj_;
  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;

#ifdef PARIS_3PT
  const int nk = nk_;
  const double si = M_PI/double(ni);
  const double sj = M_PI/double(nj);
  const double sk = M_PI/double(nk);
#elif defined PARIS_5PT
  const int nk = nk_;
  const double si = 2.0*M_PI/double(ni);
  const double sj = 2.0*M_PI/double(nj);
  const double sk = 2.0*M_PI/double(nk);
#endif

  henry.filter(bytes,density,potential,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
#ifdef PARIS_3PT
        const double i2 = sqr(sin(double(min(i,ni-i))*si)*ddi);
        const double j2 = sqr(sin(double(min(j,nj-j))*sj)*ddj);
        const double k2 = sqr(sin(double(k)*sk)*ddk);
#elif defined PARIS_5PT
        const double ci = cos(double(min(i,ni-i))*si);
        const double cj = cos(double(min(j,nj-j))*sj);
        const double ck = cos(double(k)*sk);
        const double i2 = ddi*(2.0*ci*ci-16.0*ci+14.0);
        const double j2 = ddj*(2.0*cj*cj-16.0*cj+14.0);
        const double k2 = ddk*(2.0*ck*ck-16.0*ck+14.0);
#else
        const double i2 = sqr(double(min(i,ni-i))*ddi);
        const double j2 = sqr(double(min(j,nj-j))*ddj);
        const double k2 = sqr(double(k)*ddk);
#endif
        const double d = -1.0/(i2+j2+k2);
        return cufftDoubleComplex{d*b.x,d*b.y};
      } else {
        return cufftDoubleComplex{0.0,0.0};
      }
    });
}

#endif

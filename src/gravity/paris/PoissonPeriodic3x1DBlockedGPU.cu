#ifdef PARIS

#include "PoissonPeriodic3x1DBlockedGPU.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

__host__ __device__
static inline double sqr(const double x) { return x*x; }

PoissonPeriodic3x1DBlockedGPU::PoissonPeriodic3x1DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]):
#ifdef PARIS_3PT
  ddi_(2.0*double(n[0]-1)/(hi[0]-lo[0])),
  ddj_(2.0*double(n[1]-1)/(hi[1]-lo[1])),
  ddk_(2.0*double(n[2]-1)/(hi[2]-lo[2])),
#elif defined PARIS_5PT
  ddi_(sqr(double(n[0]-1)/(hi[0]-lo[0]))/6.0),
  ddj_(sqr(double(n[1]-1)/(hi[1]-lo[1]))/6.0),
  ddk_(sqr(double(n[2]-1)/(hi[2]-lo[2]))/6.0),
#else
  ddi_{2.0*M_PI*double(n[0]-1)/(double(n[0])*(hi[0]-lo[0]))},
  ddj_{2.0*M_PI*double(n[1]-1)/(double(n[1])*(hi[1]-lo[1]))},
  ddk_{2.0*M_PI*double(n[2]-1)/(double(n[2])*(hi[2]-lo[2]))},
#endif
  idi_(id[0]),
  idj_(id[1]),
  idk_(id[2]),
  mi_(m[0]),
  mj_(m[1]),
  mk_(m[2]),
  nh_(n[2]/2+1),
  ni_(n[0]),
  nj_(n[1]),
  nk_(n[2]),
  bytes_(0)
{
  mq_ = int(round(sqrt(mk_)));
  while (mk_%mq_) mq_--;
  mp_ = mk_/mq_;
  assert(mp_*mq_ == mk_);

  idp_ = idk_/mq_;
  idq_ = idk_%mq_;

  {
    const int color = idi_*mj_+idj_;
    const int key = idk_;
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&commK_);
  }
  {
    const int color = idi_*mp_+idp_;
    const int key = idj_*mq_+idq_;
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&commJ_);
  }
  {
    const int color = idj_*mq_+idq_;
    const int key = idi_*mp_+idp_;
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&commI_);
  }
  const int dh = (nh_+mk_-1)/mk_;
  di_ = (ni_+mi_-1)/mi_;
  dj_ = (nj_+mj_-1)/mj_;
  dk_ = (nk_+mk_-1)/mk_;

  dip_ = (di_+mp_-1)/mp_;
  djq_ = (dj_+mq_-1)/mq_;
  const int mjq = mj_*mq_;
  dhq_ = (nh_+mjq-1)/mjq;
  const int mip = mi_*mp_;
  djp_ = (nj_+mip-1)/mip;

  const long nMax = std::max(
    { long(di_)*long(dj_)*long(dk_),
      long(mp_)*long(mq_)*long(dip_)*long(djq_)*long(dk_),
      long(2)*long(dip_)*long(djq_)*long(mk_)*long(dh),
      long(2)*long(dip_)*long(mp_)*long(djq_)*long(mq_)*long(dh),
      long(2)*long(dip_)*long(djq_)*long(mjq)*long(dhq_),
      long(2)*long(dip_)*long(dhq_)*long(mip)*long(djp_),
      long(2)*djp_*long(dhq_)*long(mip)*long(dip_)
    });
  assert(nMax <= INT_MAX);
  bytes_ = nMax*sizeof(double);

  CHECK(cufftPlanMany(&c2ci_,1,&ni_,&ni_,1,ni_,&ni_,1,ni_,CUFFT_Z2Z,djp_*dhq_));
  CHECK(cufftPlanMany(&c2cj_,1,&nj_,&nj_,1,nj_,&nj_,1,nj_,CUFFT_Z2Z,dip_*dhq_));
  CHECK(cufftPlanMany(&c2rk_,1,&nk_,&nh_,1,nh_,&nk_,1,nk_,CUFFT_Z2D,dip_*djq_));
  CHECK(cufftPlanMany(&r2ck_,1,&nk_,&nk_,1,nk_,&nh_,1,nh_,CUFFT_D2Z,dip_*djq_));

#ifdef PARIS_NO_GPU_MPI
  CHECK(cudaHostAlloc(&ha_,bytes_+bytes_,cudaHostAllocDefault));
  assert(ha_);
  hb_ = ha_+nMax;
#endif
}

PoissonPeriodic3x1DBlockedGPU::~PoissonPeriodic3x1DBlockedGPU()
{
#ifdef PARIS_NO_GPU_MPI
  CHECK(cudaFreeHost(ha_));
  ha_ = hb_ = nullptr;
#endif
  CHECK(cufftDestroy(r2ck_));
  CHECK(cufftDestroy(c2rk_));
  CHECK(cufftDestroy(c2cj_));
  CHECK(cufftDestroy(c2ci_));
  MPI_Comm_free(&commI_);
  MPI_Comm_free(&commJ_);
  MPI_Comm_free(&commK_);
}

void PoissonPeriodic3x1DBlockedGPU::solve(const size_t bytes, double *const density, double *const potential) const
{
  assert(bytes >= bytes_);

  double *const a = potential;
  double *const b = density;
  cufftDoubleComplex *const ac = reinterpret_cast<cufftDoubleComplex*>(a);
  cufftDoubleComplex *const bc = reinterpret_cast<cufftDoubleComplex*>(b);

  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const int di = di_, dj = dj_, dk = dk_;
  const int dhq = dhq_, dip = dip_, djp = djp_, djq = djq_;
  const int idi = idi_, idj = idj_, idk = idk_;
  const int idp = idp_, idq = idq_;
  const int mi = mi_, mj = mj_, mk = mk_;
  const int mp = mp_, mq = mq_;
  const int nh = nh_, ni = ni_, nj = nj_, nk = nk_;

  const int idip = idi*mp+idp;
  const int idjq = idj*mq+idq;
  const int mip = mi*mp;
  const int mjq = mj*mq;

  gpuFor(
    mp,mq,dip,djq,dk,
    GPU_LAMBDA(const int p, const int q, const int i, const int j, const int k) {
      const int ii = p*dip+i; 
      const int jj = q*djq+j;
      const int ia = k+dk*(j+djq*(i+dip*(q+mq*p)));
      const int ib = k+dk*(jj+dj*ii);
      a[ia] = b[ib];
    });

  const int countK = dip*djq*dk;
#ifdef PARIS_NO_GPU_MPI
  CHECK(cudaMemcpy(ha_,a,bytes,cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_,countK,MPI_DOUBLE,hb_,countK,MPI_DOUBLE,commK_);
  CHECK(cudaMemcpy(b,hb_,bytes,cudaMemcpyHostToDevice));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(a,countK,MPI_DOUBLE,b,countK,MPI_DOUBLE,commK_);
#endif

  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int jLo = idj*dj+idq*djq;
    const int jHi = std::min({jLo+djq,(idj+1)*dj,nj});
    gpuFor(
      iHi-iLo,jHi-jLo,mk,dk,
      GPU_LAMBDA(const int i, const int j, const int pq, const int k) {
        const int kk = pq*dk+k;
        if (kk < nk) {
          const int ia = kk+nk*(j+djq*i);
          const int ib = k+dk*(j+djq*(i+dip*pq));
          a[ia] = b[ib];
        }
      });
  }
  CHECK(cufftExecD2Z(r2ck_,a,bc));

  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int jLo = idj_*dj_+idq*djq;
    const int jHi = std::min({jLo+djq,(idj+1)*dj,nj});
    gpuFor(
      mjq,iHi-iLo,jHi-jLo,dhq,
      GPU_LAMBDA(const int q, const int i, const int j, const int k) {
        const int kk = q*dhq+k;
        if (kk < nh) {
          const int ia = k+dhq*(j+djq*(i+dip*q));
          const int ib = kk+nh*(j+djq*i);
          ac[ia] = bc[ib];
        }
      });
  }

  const int countJ = 2*dip*djq*dhq;
#ifdef PARIS_NO_GPU_MPI
  CHECK(cudaMemcpy(ha_,a,bytes,cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_,countJ,MPI_DOUBLE,hb_,countJ,MPI_DOUBLE,commJ_);
  CHECK(cudaMemcpy(b,hb_,bytes,cudaMemcpyHostToDevice));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(a,countJ,MPI_DOUBLE,b,countJ,MPI_DOUBLE,commJ_);
#endif

  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      kHi-kLo,iHi-iLo,mj,mq,djq,
      GPU_LAMBDA(const int k, const int i, const int r, const int q, const int j) {
        const int rdj = r*dj;
        const int jj = rdj+q*djq+j;
        if ((jj < nj) && (jj < rdj+dj)) {
          const int ia = jj+nj*(i+dip*k);
          const int ib = k+dhq*(j+djq*(i+dip*(q+mq*r)));
          ac[ia] = bc[ib];
        }
      });
  }

  CHECK(cufftExecZ2Z(c2cj_,ac,bc,CUFFT_FORWARD));

  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      mip,kHi-kLo,iHi-iLo,djp,
      GPU_LAMBDA(const int p, const int k, const int i, const int j) {
        const int jj = p*djp+j;
        if (jj < nj) {
            const int ia = j+djp*(i+dip*(k+dhq*p));
            const int ib = jj+nj*(i+dip*k);
            ac[ia] = bc[ib];
          }
        });
  }
 
  const int countI = 2*dip*djp*dhq;
#ifdef PARIS_NO_GPU_MPI
  CHECK(cudaMemcpy(ha_,a,bytes,cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_,countI,MPI_DOUBLE,hb_,countI,MPI_DOUBLE,commI_);
  CHECK(cudaMemcpy(b,hb_,bytes,cudaMemcpyHostToDevice));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(a,countI,MPI_DOUBLE,b,countI,MPI_DOUBLE,commI_);
#endif

  {
    const int jLo = idip*djp;
    const int jHi = std::min(jLo+djp,nj);
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      jHi-jLo,kHi-kLo,mi,mp,dip,
      GPU_LAMBDA(const int j, const int k, const int r, const int p, const int i) {
        const int rdi = r*di;
        const int ii = rdi+p*dip+i;
        if ((ii < ni) && (ii < rdi+di)) {
          const int ia = ii+ni*(k+dhq*j);
          const int ib = j+djp*(i+dip*(k+dhq*(p+mp*r)));
          ac[ia] = bc[ib];
        }
      });
  }

  CHECK(cufftExecZ2Z(c2ci_,ac,bc,CUFFT_FORWARD));

#ifdef PARIS_3PT
  const double si = M_PI/double(ni);
  const double sj = M_PI/double(nj);
  const double sk = M_PI/double(nk);
#elif defined PARIS_5PT
  const double si = 2.0*M_PI/double(ni);
  const double sj = 2.0*M_PI/double(nj);
  const double sk = 2.0*M_PI/double(nk);
#endif

  const int jLo = idip*djp;
  const int jHi = std::min(jLo+djp,nj);
  const int kLo = idjq*dhq;
  const int kHi = std::min(kLo+dhq,nh);

  gpuFor(
    jHi-jLo,kHi-kLo,ni,
    GPU_LAMBDA(const int j0, const int k0, const int i) {
      const int j = jLo+j0;
      const int k = kLo+k0;
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
        const int iab = i+ni*(k0+dhq*j0);
        ac[iab] = d*bc[iab];
      } else {
        ac[0].x = ac[0].y = 0;
      }
    });

  CHECK(cufftExecZ2Z(c2ci_,ac,bc,CUFFT_INVERSE));
 
  {
    const int jLo = idip*djp;
    const int jHi = std::min(jLo+djp,nj);
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      mi,mp,jHi-jLo,kHi-kLo,dip,
      GPU_LAMBDA(const int r, const int p, const int j, const int k, const int i) {
        const int rdi = r*di;
        const int ii = rdi+p*dip+i;
        if ((ii < ni) && (ii < rdi+di)) {
          const int ia = i+dip*(k+dhq*(j+djp*(p+mp*r)));
          const int ib = ii+ni*(k+dhq*j);
          ac[ia] = bc[ib];
        }
      });
  }

#ifdef PARIS_NO_GPU_MPI
  CHECK(cudaMemcpy(ha_,a,bytes,cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_,countI,MPI_DOUBLE,hb_,countI,MPI_DOUBLE,commI_);
  CHECK(cudaMemcpy(b,hb_,bytes,cudaMemcpyHostToDevice));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(a,countI,MPI_DOUBLE,b,countI,MPI_DOUBLE,commI_);
#endif

  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      kHi-kLo,iHi-iLo,mip,djp,
      GPU_LAMBDA(const int k, const int i, const int p, const int j) {
        const int jj = p*djp+j;
        if (jj < nj) {
          const int ia = jj+nj*(i+dip*k);
          const int ib = i+dip*(k+dhq*(j+djp*p));
          ac[ia] = bc[ib];
        }
      });
  }

  CHECK(cufftExecZ2Z(c2cj_,ac,bc,CUFFT_INVERSE));

  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      mj,mq,kHi-kLo,iHi-iLo,djq,
      GPU_LAMBDA(const int r, const int q, const int k, const int i, const int j) {
        const int rdj = r*dj;
        const int jj = rdj+q*djq+j;
        if ((jj < nj) && (jj < rdj+dj)) {
          const int ia = j+djq*(i+dip*(k+dhq*(q+mq*r)));
          const int ib = jj+nj*(i+dip*k);
          ac[ia] = bc[ib];
        }
      });
  }

#ifdef PARIS_NO_GPU_MPI
  CHECK(cudaMemcpy(ha_,a,bytes,cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_,countJ,MPI_DOUBLE,hb_,countJ,MPI_DOUBLE,commJ_);
  CHECK(cudaMemcpy(b,hb_,bytes,cudaMemcpyHostToDevice));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(a,countJ,MPI_DOUBLE,b,countJ,MPI_DOUBLE,commJ_);
#endif

  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int jLo = idj*dj+idq*djq;
    const int jHi = std::min({jLo+djq,(idj+1)*dj,nj});
    gpuFor(
      iHi-iLo,jHi-jLo,mjq,dhq,
      GPU_LAMBDA(const int i, const int j, const int q, const int k) {
        const int kk = q*dhq+k;
        if (kk < nh) {
          const int ia = kk+nh*(j+djq*i);
          const int ib = j+djq*(i+dip*(k+dhq*q));
          ac[ia] = bc[ib];
        }
      });
  }

  CHECK(cufftExecZ2D(c2rk_,ac,b));

  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int jLo = idj*dj+idq*djq;
    const int jHi = std::min({jLo+djq,(idj+1)*dj,nj});
    gpuFor(
      mk,iHi-iLo,jHi-jLo,dk,
      GPU_LAMBDA(const int pq, const int i, const int j, const int k) {
        const int kk = pq*dk+k;
        if (kk < nk) {
          const int ia = k+dk*(j+djq*(i+dip*pq));
          const int ib = kk+nk*(j+djq*i);
          a[ia] = b[ib];
        }
      });
  }

#ifdef PARIS_NO_GPU_MPI
  CHECK(cudaMemcpy(ha_,a,bytes,cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_,countK,MPI_DOUBLE,hb_,countK,MPI_DOUBLE,commK_);
  CHECK(cudaMemcpy(b,hb_,bytes,cudaMemcpyHostToDevice));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(a,countK,MPI_DOUBLE,b,countK,MPI_DOUBLE,commK_);
#endif

  {
    const double divN = 1.0/(double(ni)*double(nj)*double(nk));
    const int kLo = idk*dk;
    const int kHi = std::min(kLo+dk,nk);
    gpuFor(
      mp,dip,mq,djq,kHi-kLo,
      GPU_LAMBDA(const int p, const int i, const int q, const int j, const int k) {
        const int ii = p*dip+i;
        const int jj = q*djq+j;
        if ((ii < di) && (jj < dj)) {
          const int ia = k+dk*(jj+dj*ii);
          const int ib = k+dk*(j+djq*(i+dip*(q+mq*p)));
          a[ia] = divN*b[ib];
        }
      });
  }
}

#endif

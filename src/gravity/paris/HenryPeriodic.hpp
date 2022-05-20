#pragma once

#include <algorithm>
#include <mpi.h>

#include "../../utils/gpu.hpp"

class HenryPeriodic {
  public:
    HenryPeriodic(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);
    ~HenryPeriodic();
    size_t bytes() const { return bytes_; }

    template <typename F>
    void filter(const size_t bytes, double *const density, double *const potential, const F f) const;

  private:
    int idi_,idj_,idk_;
    int mi_,mj_,mk_;
    int nh_,ni_,nj_,nk_;
    int mp_,mq_;
    int idp_,idq_;
    MPI_Comm commI_,commJ_,commK_;
    int dh_,di_,dj_,dk_;
    int dhq_,dip_,djp_,djq_;
    size_t bytes_;
    cufftHandle c2ci_,c2cj_,c2rk_,r2ck_;
#ifndef MPI_GPU
    double *ha_, *hb_;
#endif
};

#if defined(__HIP__) || defined(__CUDACC__)

template <typename F>
void HenryPeriodic::filter(const size_t bytes, double *const density, double *const potential, const F f) const
{
  assert(bytes >= bytes_);

  double *const a = potential;
  double *const b = density;
  cufftDoubleComplex *const ac = reinterpret_cast<cufftDoubleComplex*>(a);
  cufftDoubleComplex *const bc = reinterpret_cast<cufftDoubleComplex*>(b);

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
#ifndef MPI_GPU
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
#ifndef MPI_GPU
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
#ifndef MPI_GPU
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

  const int jLo = idip*djp;
  const int jHi = std::min(jLo+djp,nj);
  const int kLo = idjq*dhq;
  const int kHi = std::min(kLo+dhq,nh);

  gpuFor(
    jHi-jLo,kHi-kLo,ni,
    GPU_LAMBDA(const int j0, const int k0, const int i) {
      const int j = jLo+j0;
      const int k = kLo+k0;
      const int iab = i+ni*(k0+dhq*j0);
      ac[iab] = f(i,j,k,bc[iab]);
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

#ifndef MPI_GPU
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

#ifndef MPI_GPU
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

#ifndef MPI_GPU
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


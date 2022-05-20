#ifdef PARIS

#include "HenryPeriodic.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>

HenryPeriodic::HenryPeriodic(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]):
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
  dh_ = (nh_+mk_-1)/mk_;
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
      long(2)*long(dip_)*long(djq_)*long(mk_)*long(dh_),
      long(2)*long(dip_)*long(mp_)*long(djq_)*long(mq_)*long(dh_),
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

#ifndef MPI_GPU
  CHECK(cudaHostAlloc(&ha_,bytes_+bytes_,cudaHostAllocDefault));
  assert(ha_);
  hb_ = ha_+nMax;
#endif
}

HenryPeriodic::~HenryPeriodic()
{
#ifndef MPI_GPU
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

#endif

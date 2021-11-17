#ifdef PARIS

#include "PoissonPeriodic3DBlockedGPU.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define ERROR(X,...) if (X) { fprintf(stderr,"%s(%d): ",__FILE__,__LINE__); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); MPI_Abort(MPI_COMM_WORLD,(X)); }

static inline __host__ __device__ double sqr(const double x) { return x*x; }

PoissonPeriodic3DBlockedGPU::PoissonPeriodic3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]):
#ifdef PARIS_3PT
  di_(2.0*double(n[0]-1)/(hi[0]-lo[0])),
  dj_(2.0*double(n[1]-1)/(hi[1]-lo[1])),
  dk_(2.0*double(n[2]-1)/(hi[2]-lo[2])),
#elif defined PARIS_5PT
  di_(sqr(double(n[0]-1)/(hi[0]-lo[0]))/6.0),
  dj_(sqr(double(n[1]-1)/(hi[1]-lo[1]))/6.0),
  dk_(sqr(double(n[2]-1)/(hi[2]-lo[2]))/6.0),
#else
  di_{2.0*M_PI*double(n[0]-1)/(double(n[0])*(hi[0]-lo[0]))},
  dj_{2.0*M_PI*double(n[1]-1)/(double(n[1])*(hi[1]-lo[1]))},
  dk_{2.0*M_PI*double(n[2]-1)/(double(n[2])*(hi[2]-lo[2]))},
#endif
  mi_(m[0]),
  mj_(m[1]),
  mk_(m[2]),
  ni_(n[0]),
  nj_(n[1]),
  nk_(n[2])
{
  {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    ERROR(mi_*mj_*mk_ != size,"Task grid %dx%dx%d does not equal total tasks %d",mi_,mj_,mk_,size);
  }
  {
    const int color = id[0];
    const int key = id[2]+mk_*id[1];
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&commSlab_);
  }
  {
    const int color = 1;
    const int key = id[2]+mk_*(id[1]+mj_*id[0]);
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&commWorld_);
  }

  ERROR(ni_%mi_,"%d X elements are not divisible into %d X tasks",ni_,mi_);
  const int niBlock = ni_/mi_;
  ERROR(nj_%mj_,"%d Y elements are not divisible into %d Y tasks",nj_,mj_);
  const int njBlock = nj_/mj_;
  ERROR(nk_%mk_,"%d Z elements are not divisible into %d Z tasks",nk_,mk_);
  const int nkBlock = nk_/mk_;
  const int nBlock = niBlock*njBlock*nkBlock;

  const int nh = nk_/2+1;
  const int mjk = mj_*mk_;
  ERROR(niBlock%mjk,"%d X layers per XYZ block not divisible into %dx%d=%d YZ slabs",niBlock,mj_,mk_,mjk);
  const int niSlab = niBlock/mjk;
  const int nSlab = niSlab*nj_*2*nh;

  const int mijk = mi_*mj_*mk_;
  const int njk = nj_*nk_;
  const int njh = nj_*nh;
  ERROR(njh%mijk,"%dx(%d/2+1)=%d X pencils not divisible into %d tasks",nj_,nk_,njh,mijk);
  const int njhPencil = njh/mijk;
  const int nPencil = ni_*2*njhPencil;
  const int nMax = std::max({nBlock,nSlab,nPencil});

  bytes_ = sizeof(double)*nMax;
  {
    int njnk[2] = {nj_,nk_};
    int njnh[2] = {nj_,nh};
    CHECK(cufftPlanMany(&dz2d_,2,njnk,njnk,1,njk,njnh,1,njh,CUFFT_D2Z,niSlab));
    CHECK(cufftPlanMany(&zd2d_,2,njnk,njnh,1,njh,njnk,1,njk,CUFFT_Z2D,niSlab));
    CHECK(cufftPlanMany(&zz1d_,1,&ni_,&ni_,1,ni_,&ni_,1,ni_,CUFFT_Z2Z,njhPencil));
  }
#ifndef MPI_GPU
  CHECK(cudaHostAlloc(&ha_,bytes_+bytes_,cudaHostAllocDefault));
  assert(ha_);
  hb_ = ha_+nMax;
#endif
}

PoissonPeriodic3DBlockedGPU::~PoissonPeriodic3DBlockedGPU()
{
#ifndef MPI_GPU
  CHECK(cudaFreeHost(ha_));
  ha_ = hb_ = nullptr;
#endif
  CHECK(cufftDestroy(zz1d_));
  CHECK(cufftDestroy(zd2d_));
  CHECK(cufftDestroy(dz2d_));
  MPI_Comm_free(&commWorld_);
  MPI_Comm_free(&commSlab_);
}

void PoissonPeriodic3DBlockedGPU::solve(const long bytes, double *const da, double *const db)
{
  // Make local copies for lambda kernels
  const double di = di_;
  const double dj = dj_;
  const double dk = dk_;
  const int mj = mj_;
  const int mk = mk_;
  const int ni = ni_;
  const int nj = nj_;

  cufftDoubleComplex *const ca = reinterpret_cast<cufftDoubleComplex *>(da);
  cufftDoubleComplex *const cb = reinterpret_cast<cufftDoubleComplex *>(db);

  const int nh = nk_/2+1;
  const int njh = nj*nh;
  const int niBlock = (ni+mi_-1)/mi_;
  const int njBlock = (nj+mj-1)/mj;
  const int nkBlock = (nk_+mk-1)/mk;

  const int mjk = mj*mk;
  const int niSlab = (niBlock+mjk-1)/mjk;
  const int nBlockSlab = niSlab*njBlock*nkBlock;

  ERROR(bytes < bytes_,"Vector bytes %ld less than %ld local elements = %ld bytes",bytes,bytes_/sizeof(double),bytes_);

  // Copy blocks to slabs

#ifndef MPI_GPU
  CHECK(cudaMemcpy(ha_,da,bytes_,cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_,nBlockSlab,MPI_DOUBLE,hb_,nBlockSlab,MPI_DOUBLE,commSlab_);
  CHECK(cudaMemcpyAsync(db,hb_,bytes_,cudaMemcpyHostToDevice,0));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(da,nBlockSlab,MPI_DOUBLE,db,nBlockSlab,MPI_DOUBLE,commSlab_);
#endif

  gpuFor(
    niSlab,mj,njBlock,mk,nkBlock,
    GPU_LAMBDA(const int i, const int p, const int j, const int q, const int k) {
      const int ia = k+nkBlock*(q+mk*(j+njBlock*(p+mj*i)));
      const int ib = k+nkBlock*(j+njBlock*(i+niSlab*(q+mk*p)));
      da[ia] = db[ib];
    });

  // da -> cb
  CHECK(cufftExecD2Z(dz2d_,da,cb));

  // Copy slabs to pencils

  gpuFor(
    njh,niSlab,
    GPU_LAMBDA(const int jk, const int i) {
      const int ia = i+niSlab*jk;
      const int ib = jk+njh*i;
      ca[ia].x = cb[ib].x;
      ca[ia].y = cb[ib].y;
    });

  const int m = mi_*mj*mk;
  const int njhPencil = (njh+m-1)/m;
  const int nSlabPencil = 2*njhPencil*niSlab;

#ifndef MPI_GPU
  CHECK(cudaMemcpy(ha_,da,bytes_,cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_,nSlabPencil,MPI_DOUBLE,hb_,nSlabPencil,MPI_DOUBLE,MPI_COMM_WORLD);
  CHECK(cudaMemcpyAsync(db,hb_,bytes_,cudaMemcpyHostToDevice,0));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(da,nSlabPencil,MPI_DOUBLE,db,nSlabPencil,MPI_DOUBLE,MPI_COMM_WORLD);
#endif

  gpuFor(
    njhPencil,m,niSlab,
    GPU_LAMBDA(const int jk, const int pq, const int i) {
      const int ia = i+niSlab*(pq+m*jk);
      const int ib = i+niSlab*(jk+njhPencil*pq);
      ca[ia].x = cb[ib].x;
      ca[ia].y = cb[ib].y;
    });

  // ca -> cb
  CHECK(cufftExecZ2Z(zz1d_,ca,cb,CUFFT_FORWARD));

  // Solve Poisson equation

  {
#ifdef PARIS_3PT
    const double si = M_PI/double(ni_);
    const double sj = M_PI/double(nj_);
    const double sk = M_PI/double(nk_);
#elif defined PARIS_5PT
    const double si = 2.0*M_PI/double(ni_);
    const double sj = 2.0*M_PI/double(nj_);
    const double sk = 2.0*M_PI/double(nk_);
#endif

    int rank = MPI_PROC_NULL;
    MPI_Comm_rank(commWorld_,&rank);
    const int jkLo = rank*njhPencil;
    const int jkHi = std::min(jkLo+njhPencil,njh);
    const int djk = jkHi-jkLo;
    gpuFor(
      djk,ni,
      GPU_LAMBDA(int jk, const int i) {
        const int ijk = i+ni*jk;
        if ((ijk == 0) && (jkLo == 0)) {
          cb[0].x = cb[0].y = 0;
        } else {
#ifdef PARIS_3PT
          const double ii = sqr(sin(double(min(i,ni-i))*si)*di);
#elif defined PARIS_5PT
          const double ci = cos(double(min(i,ni-i))*si);
          const double ii = di*(2.0*ci*ci-16.0*ci+14.0);
#else
          const double ii = sqr(double(min(i,ni-i))*di);
#endif
          jk += jkLo;
          const int j = jk/nh;
#ifdef PARIS_3PT
          const double jj = sqr(sin(double(min(j,nj-j))*sj)*dj);
#elif defined PARIS_5PT
          const double cj = cos(double(min(j,nj-j))*sj);
          const double jj = dj*(2.0*cj*cj-16.0*cj+14.0);
#else
          const double jj = sqr(double(min(j,nj-j))*dj);
#endif
          const int k = jk-j*nh;
#ifdef PARIS_3PT
          const double kk = sqr(sin(double(k)*sk)*dk);
#elif defined PARIS_5PT
          const double ck = cos(double(k)*sk);
          const double kk = dk*(2.0*ck*ck-16.0*ck+14.0);
#else
          const double kk = sqr(double(k)*dk);
#endif
          const double d = -1.0/(ii+jj+kk);
          cb[ijk].x *= d;
          cb[ijk].y *= d;
        }
      });
  }

  // cb -> ca
  CHECK(cufftExecZ2Z(zz1d_,cb,ca,CUFFT_INVERSE));

  // Copy pencils to slabs

  gpuFor(
    m,njhPencil,niSlab,
    GPU_LAMBDA(const int pq, const int jk, const int i) {
      const int ia = i+niSlab*(pq+m*jk);
      const int ib = i+niSlab*(jk+njhPencil*pq);
      cb[ib].x = ca[ia].x;
      cb[ib].y = ca[ia].y;
    });

#ifndef MPI_GPU
  CHECK(cudaMemcpy(hb_,db,bytes_,cudaMemcpyDeviceToHost));
  MPI_Alltoall(hb_,nSlabPencil,MPI_DOUBLE,ha_,nSlabPencil,MPI_DOUBLE,commWorld_);
  CHECK(cudaMemcpyAsync(da,ha_,bytes_,cudaMemcpyHostToDevice,0));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(db,nSlabPencil,MPI_DOUBLE,da,nSlabPencil,MPI_DOUBLE,commWorld_);
#endif

  gpuFor(
    niSlab,njh,
    GPU_LAMBDA(const int i, const int jk) {
      const int ia = i+jk*niSlab;
      const int ib = jk+njh*i;
      cb[ib].x = ca[ia].x;
      cb[ib].y = ca[ia].y;
    });

  // cb -> da
  CHECK(cufftExecZ2D(zd2d_,cb,da));

  // Copy slabs to blocks

  const double divN = 1.0/(long(ni)*long(nj)*long(nk_));

  gpuFor(
    mj,mk,niSlab,njBlock,nkBlock,
    GPU_LAMBDA(const int p, const int q, const int i, const int j, const int k) {
      const int ia = k+nkBlock*(q+mk*(j+njBlock*(p+mj*i)));
      const int ib = k+nkBlock*(j+njBlock*(i+niSlab*(q+mk*p)));
      db[ib] = divN*da[ia];
    });

#ifndef MPI_GPU
  CHECK(cudaMemcpy(hb_,db,bytes_,cudaMemcpyDeviceToHost));
  MPI_Alltoall(hb_,nBlockSlab,MPI_DOUBLE,ha_,nBlockSlab,MPI_DOUBLE,commSlab_);
  CHECK(cudaMemcpyAsync(da,ha_,bytes_,cudaMemcpyHostToDevice,0));
#else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(db,nBlockSlab,MPI_DOUBLE,da,nBlockSlab,MPI_DOUBLE,commSlab_);
#endif
}

#endif

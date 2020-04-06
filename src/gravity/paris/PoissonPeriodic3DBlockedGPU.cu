#include "PoissonPeriodic3DBlockedGPU.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "gpu.hpp"

#define ERROR(X,...) if (X) { fprintf(stderr,"%s(%d): ",__FILE__,__LINE__); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); MPI_Abort(MPI_COMM_WORLD,(X)); }

static constexpr double pi = 3.141592653589793238462643383279502884197169399375105820974;

static inline __host__ __device__ double sqr(const double x) { return x*x; }

PoissonPeriodic3DBlockedGPU::PoissonPeriodic3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]):
  di_{2.0*pi*double(n[0]-1)/(double(n[0])*(hi[0]-lo[0]))},
  dj_{2.0*pi*double(n[1]-1)/(double(n[1])*(hi[1]-lo[1]))},
  dk_{2.0*pi*double(n[2]-1)/(double(n[2])*(hi[2]-lo[2]))},
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
  const long nBlock = long(niBlock)*long(njBlock)*long(nkBlock);

  const int nh = nk_/2+1;
  const int mjk = mj_*mk_;
  ERROR(niBlock%mjk,"%d X layers per XYZ block not divisible into %dx%d=%d YZ slabs",niBlock,mj_,mk_,mjk);
  const int niSlab = niBlock/mjk;
  const long nSlab = long(niSlab)*(nj_)*2L*long(nh);

  const int mijk = mi_*mj_*mk_;
  const long njk = nj_*nk_;
  const long njh = nj_*nh;
  ERROR(njh%mijk,"%dx(%d/2+1)=%ld X pencils not divisible into %d tasks",nj_,nk_,njh,mijk);
  const int njhPencil = njh/mijk;
  const long nPencil = ni_*2*njhPencil;

  bytes_ = sizeof(double)*std::max({nBlock,nSlab,nPencil});
  {
    int njnk[2] = {nj_,nk_};
    int njnh[2] = {nj_,nh};
    CHECK(cufftPlanMany(&dz2d_,2,njnk,njnk,1,njk,njnh,1,njh,CUFFT_D2Z,niSlab));
    CHECK(cufftPlanMany(&zd2d_,2,njnk,njnh,1,njh,njnk,1,njk,CUFFT_Z2D,niSlab));
    CHECK(cufftPlanMany(&zz1d_,1,&ni_,&ni_,1,ni_,&ni_,1,ni_,CUFFT_Z2Z,njhPencil));
  }
}

PoissonPeriodic3DBlockedGPU::~PoissonPeriodic3DBlockedGPU()
{
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
  const long njh = nj*nh;
  const int niBlock = (ni+mi_-1)/mi_;
  const int njBlock = (nj+mj-1)/mj;
  const int nkBlock = (nk_+mk-1)/mk;

  const int mjk = mj*mk;
  const int niSlab = (niBlock+mjk-1)/mjk;
  const int nBlockSlab = niSlab*njBlock*nkBlock;

  ERROR(bytes < bytes_,"Vector bytes %ld less than %ld local elements = %ld bytes",bytes,bytes_/sizeof(double),bytes_);

  // Copy blocks to slabs

  MPI_Alltoall(da,nBlockSlab,MPI_DOUBLE,db,nBlockSlab,MPI_DOUBLE,commSlab_);

  gpuFor(
    niSlab,mj,njBlock,mk,nkBlock,
    GPU_LAMBDA(const long ia, const long i, const int p, const int j, const int q, const int k) {
      const long ib = k+nkBlock*(j+njBlock*(i+niSlab*(q+mk*p)));
      da[ia] = db[ib];
    });

  // da -> cb
  CHECK(cufftExecD2Z(dz2d_,da,cb));

  // Copy slabs to pencils

  gpuFor(
    njh,niSlab,
    GPU_LAMBDA(const long ia, const long jk, const long i) {
      const long ib = jk+njh*i;
      ca[ia].x = cb[ib].x;
      ca[ia].y = cb[ib].y;
    });

  const int m = mi_*mj*mk;
  const int njhPencil = (njh+m-1)/m;
  const int nSlabPencil = 2*njhPencil*niSlab;

  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(da,nSlabPencil,MPI_DOUBLE,db,nSlabPencil,MPI_DOUBLE,MPI_COMM_WORLD);

  gpuFor(
    njhPencil,m,niSlab,
    GPU_LAMBDA(const long ia, const int jk, const int pq, const int i) {
      const long ib = i+niSlab*(jk+njhPencil*pq);
      ca[ia].x = cb[ib].x;
      ca[ia].y = cb[ib].y;
    });

  // ca -> cb
  CHECK(cufftExecZ2Z(zz1d_,ca,cb,CUFFT_FORWARD));

  // Solve Poisson equation

  {
    int rank = MPI_PROC_NULL;
    MPI_Comm_rank(commWorld_,&rank);
    const long jkLo = rank*njhPencil;
    const long jkHi = std::min(jkLo+njhPencil,njh);
    gpuFor(
      jkHi-jkLo,ni,
      GPU_LAMBDA(const long ijk, long jk, const long i) {
        if ((ijk == 0) && (jkLo == 0)) {
          ca[0].x = ca[0].y = 0;
        } else {
          const double ii = sqr(double(min(i,ni-i))*di);
          jk += jkLo;
          const int j = jk/nh;
          const double jj = sqr(double(min(j,nj-j))*dj);
          const int k = jk-j*nh;
          const double kk = sqr(double(k)*dk);
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
    GPU_LAMBDA(const long ib, const int pq, const int jk, const int i) {
      const long ia = i+niSlab*(pq+m*jk);
      cb[ib].x = ca[ia].x;
      cb[ib].y = ca[ia].y;
    });

  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(db,nSlabPencil,MPI_DOUBLE,da,nSlabPencil,MPI_DOUBLE,commWorld_);

  gpuFor(
    niSlab,njh,
    GPU_LAMBDA(const long ib, const long i, const long jk) {
      const long ia = i+jk*niSlab;
      cb[ib].x = ca[ia].x;
      cb[ib].y = ca[ia].y;
    });

  // cb -> da 
  CHECK(cufftExecZ2D(zd2d_,cb,da));

  // Copy slabs to blocks

  const double divN = 1.0/(long(ni)*long(nj)*long(nk_));

  gpuFor(
    mj,mk,niSlab,njBlock,nkBlock,
    GPU_LAMBDA(const long ib, const int p, const int q, const int i, const int j, const int k) {
      const long ia = k+nkBlock*(q+mk*(j+njBlock*(p+mj*i)));
      db[ib] = divN*da[ia];
    });

  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(db,nBlockSlab,MPI_DOUBLE,da,nBlockSlab,MPI_DOUBLE,commSlab_);
}

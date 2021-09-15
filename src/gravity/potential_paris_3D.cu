#if defined(GRAVITY) && defined(PARIS)

#include "../gravity/potential_paris_3D.h"
#include "../utils/gpu.hpp"
#include "../io/io.h"
#include <cassert>
#include <cfloat>
#include <climits>

static void __attribute__((unused)) printDiff(const Real *p, const Real *q, const int ng, const int nx, const int ny, const int nz, const bool plot = false)
{
  Real dMax = 0, dSum = 0, dSum2 = 0;
  Real qMax = 0, qSum = 0, qSum2 = 0;
#pragma omp parallel for reduction(max:dMax,qMax) reduction(+:dSum,dSum2,qSum,qSum2)
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        const int ijk = i+ng+(nx+ng+ng)*(j+ng+(ny+ng+ng)*(k+ng));
        const Real qAbs = fabs(q[ijk]);
        qMax = std::max(qMax,qAbs);
        qSum += qAbs;
        qSum2 += qAbs*qAbs;
        const Real d = fabs(q[ijk]-p[ijk]);
        dMax = std::max(dMax,d);
        dSum += d;
        dSum2 += d*d;
      }
    }
  }
  Real maxs[2] = {qMax,dMax};
  Real sums[4] = {qSum,qSum2,dSum,dSum2};
  MPI_Allreduce(MPI_IN_PLACE,&maxs,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&sums,4,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  chprintf(" Poisson-Solver Diff: L1 %g L2 %g Linf %g\n",sums[2]/sums[0],sqrt(sums[3]/sums[1]),maxs[1]/maxs[0]);
  fflush(stdout);
  if (!plot) return;

  printf("###\n");
  const int k = nz/2;
  //for (int j = 0; j < ny; j++) {
  const int j = ny/2;
    for (int i = 0; i < nx; i++) {
      const int ijk = i+ng+(nx+ng+ng)*(j+ng+(ny+ng+ng)*(k+ng));
      //printf("%d %d %g %g %g\n",j,i,q[ijk],p[ijk],q[ijk]-p[ijk]);
      printf("%d %g %g %g\n",i,q[ijk],p[ijk],q[ijk]-p[ijk]);
    }
    printf("\n");
  //}

  MPI_Finalize();
  exit(0);
}

Potential_Paris_3D::Potential_Paris_3D():
  dn_{0,0,0},
  dr_{0,0,0},
  lo_{0,0,0},
  lr_{0,0,0},
  myLo_{0,0,0},
  pp_(nullptr),
  minBytes_(0),
  densityBytes_(0),
  potentialBytes_(0),
  da_(nullptr),
  db_(nullptr)
{}

Potential_Paris_3D::~Potential_Paris_3D() { Reset(); }

void Potential_Paris_3D::Get_Potential(const Real *const density, Real *const potential, const Real g, const Real offset, const Real a)
{
#ifdef COSMOLOGY
  const Real scale = Real(4)*M_PI*g/a;
#else
  const Real scale = Real(4)*M_PI*g;
#endif
  assert(da_);
  Real *const da = da_;
  Real *const db = db_;
  assert(density);

  const int ni = dn_[2];
  const int nj = dn_[1];
  const int nk = dn_[0];

  const int n = ni*nj*nk;
  #ifdef GRAVITY_GPU
  CHECK(cudaMemcpy(db,density,densityBytes_,cudaMemcpyDeviceToDevice));
  #else
  CHECK(cudaMemcpy(db,density,densityBytes_,cudaMemcpyHostToDevice));
  #endif
  const int ngi = ni+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const int ngj = nj+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;

  gpuFor(n,GPU_LAMBDA(const int i) { db[i] = scale*(db[i]-offset); });
  pp_->solve(minBytes_,db,da);
  gpuFor(
    nk,nj,ni,
    GPU_LAMBDA(const int k, const int j, const int i) {
      const int ia = i+ni*(j+nj*k);
      const int ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
      db[ib] = da[ia];
    });

  assert(potential);
  #ifdef GRAVITY_GPU
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToDevice));
  #else
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToHost));
  #endif
}

void Potential_Paris_3D::Initialize(const Real lx, const Real ly, const Real lz, const Real xMin, const Real yMin, const Real zMin, const int nx, const int ny, const int nz, const int nxReal, const int nyReal, const int nzReal, const Real dx, const Real dy, const Real dz)
{
  chprintf(" Using Poisson Solver: Paris Periodic");
#ifdef PARIS_5PT
  chprintf(" 5-Point\n");
#elif defined PARIS_3PT
  chprintf(" 3-Point\n");
#else
  chprintf(" Spectral\n");
#endif

  const long nl012 = long(nxReal)*long(nyReal)*long(nzReal);
  assert(nl012 <= INT_MAX);

  dn_[0] = nzReal;
  dn_[1] = nyReal;
  dn_[2] = nxReal;

  dr_[0] = dz;
  dr_[1] = dy;
  dr_[2] = dx;

  lr_[0] = lz;
  lr_[1] = ly;
  lr_[2] = lx;

  myLo_[0] = zMin;
  myLo_[1] = yMin;
  myLo_[2] = xMin;
  MPI_Allreduce(myLo_,lo_,3,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);

  const Real hi[3] = {lo_[0]+lz-dr_[0],lo_[1]+ly-dr_[1],lo_[2]+lx-dr_[2]};
  const int n[3] = {nz,ny,nx};
  const int m[3] = {n[0]/nzReal,n[1]/nyReal,n[2]/nxReal};
  const int id[3] = {int(round((zMin-lo_[0])/(dn_[0]*dr_[0]))),int(round((yMin-lo_[1])/(dn_[1]*dr_[1]))),int(round((xMin-lo_[2])/(dn_[2]*dr_[2])))};
  chprintf("  Paris: [ %g %g %g ]-[ %g %g %g ] N_local[ %d %d %d ] Tasks[ %d %d %d ]\n",lo_[2],lo_[1],lo_[0],lo_[2]+lx,lo_[1]+ly,lo_[0]+lz,dn_[2],dn_[1],dn_[0],m[2],m[1],m[0]);

  assert(dn_[0] == n[0]/m[0]);
  assert(dn_[1] == n[1]/m[1]);
  assert(dn_[2] == n[2]/m[2]);

  pp_ = new PoissonPeriodic3x1DBlockedGPU(n,lo_,hi,m,id);
  assert(pp_);
  minBytes_ = pp_->bytes();
  densityBytes_ = long(sizeof(Real))*dn_[0]*dn_[1]*dn_[2];
  const long gg = N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  potentialBytes_ = long(sizeof(Real))*(dn_[0]+gg)*(dn_[1]+gg)*(dn_[2]+gg);

  CHECK(cudaMalloc(reinterpret_cast<void **>(&da_),std::max(minBytes_,densityBytes_)));
  assert(da_);

  CHECK(cudaMalloc(reinterpret_cast<void **>(&db_),std::max(minBytes_,potentialBytes_)));
  assert(db_);
}

void Potential_Paris_3D::Reset()
{
  if (db_) CHECK(cudaFree(db_));
  db_ = nullptr;

  if (da_) CHECK(cudaFree(da_));
  da_ = nullptr;

  potentialBytes_ = densityBytes_ = minBytes_ = 0;

  if (pz_) delete pz_;
  pz_ = nullptr;

  if (pp_) delete pp_;
  pp_ = nullptr;

  myLo_[2] = myLo_[1] = myLo_[0] = 0;
  lr_[2] = lr_[1] = lr_[0] = 0;
  lo_[2] = lo_[1] = lo_[0] = 0;
  dr_[2] = dr_[1] = dr_[0] = 0;
  dn_[2] = dn_[1] = dn_[0] = 0;
}

#endif

#if defined(GRAVITY) && defined(PARIS)

#include "potential_paris_3D.h"
#include "gpu.hpp"
#include "../io.h"
#include <cassert>

Potential_Paris_3D::Potential_Paris_3D():
  n_{0,0,0},
  p_(nullptr),
  minBytes_(0),
  densityBytes_(0),
  potentialBytes_(0),
  da_(nullptr),
  db_(nullptr)
{}

Potential_Paris_3D::~Potential_Paris_3D() { Reset(); }

void Potential_Paris_3D::Get_Potential(const Real *const density, Real *const potential, const Real g, const Real avgDensity, const Real a)
{
  constexpr Real pi = 3.141592653589793238462643383279502884197169399375105820974;
#ifdef COSMOLOGY
  const Real scale = Real(4)*pi*g/a;
  const Real offset = avgDensity;
#else
  const Real scale = Real(4)*pi*g;
  constexpr Real offset = Real(0);
#endif
  assert(da_);
  double *const da = da_;
  double *const db = db_;
  assert(density);
  CHECK(cudaMemcpy(db,density,densityBytes_,cudaMemcpyHostToDevice));
  const long ngi = n_[0]+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long ngj = n_[1]+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long n = long(n_[0])*n_[1]*n_[2];
  gpuFor(n,GPU_LAMBDA(const long i) { da[i] = scale*(db[i]-offset); });
  p_->solve(minBytes_,da,db);
  gpuFor(
    n_[2],n_[1],n_[0],
    GPU_LAMBDA(const long ia, const int k, const int j, const int i) {
      const long ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
      db[ib] = da[ia];
    });
  assert(potential);
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToHost));
}

void Potential_Paris_3D::Initialize(const Real lx, const Real ly, const Real lz, const Real xMin, const Real yMin, const Real zMin, const int nx, const int ny, const int nz, const int nxReal, const int nyReal, const int nzReal, const Real dx, const Real dy, const Real dz)
{
  chprintf(" Using Poisson Solver: Paris\n");
  n_[0] = nxReal;
  n_[1] = nyReal;
  n_[2] = nzReal;

  const int n[3] = {nz,ny,nx};
  const double lo[3] = {0,0,0};
  const double hi[3] = {lz-dz,ly-dy,lx-dx};
  const int m[3] = {n[0]/nzReal,n[1]/nyReal,n[2]/nxReal};
  const int id[3] = {int(round(zMin/dz)),int(round(yMin/dy)),int(round(xMin/dx))};
  chprintf("  Paris: L[ %g %g %g ] N_local[ %d %d %d ] Tasks[ %d %d %d ]\n",lx,ly,lz,n_[0],n_[1],n_[2],m[2],m[1],m[0]);
  assert(n_[0] == n[2]/m[2]);
  assert(n_[1] == n[1]/m[1]);
  assert(n_[2] == n[0]/m[0]);
  p_ = new PoissonPeriodic3DBlockedGPU(n,lo,hi,m,id);
  assert(p_);

  minBytes_ = p_->bytes();
  densityBytes_ = long(sizeof(Real))*n_[0]*n_[1]*n_[2];
  const long gg = N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  potentialBytes_ = long(sizeof(Real))*(n_[0]+gg)*(n_[1]+gg)*(n_[2]+gg);

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

  potentialBytes_ = 0;
  densityBytes_ = 0;
  minBytes_ = 0;

  delete p_;
  p_ = nullptr;

  n_[0] = n_[1] = n_[2] = 0;
}

#endif

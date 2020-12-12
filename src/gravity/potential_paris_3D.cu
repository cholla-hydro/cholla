#if defined(GRAVITY) && defined(PARIS)

#include "potential_paris_3D.h"
#include "gpu.hpp"
#include "../io.h"
#include <cassert>

Potential_Paris_3D::Potential_Paris_3D():
  dn_{0,0,0},
  n_{0,0,0},
  dr_{0,0,0},
  hi_{0,0,0},
  lo_{0,0,0},
  pp_(nullptr),
  pz_(nullptr),
  minBytes_(0),
  densityBytes_(0),
  potentialBytes_(0),
  da_(nullptr),
  db_(nullptr)
{}

Potential_Paris_3D::~Potential_Paris_3D() { Reset(); }

void Potential_Paris_3D::Get_Potential(const Real *const density, Real *const potential, const Real g, const Real avgDensity, const Real a)
{
  if (!pp_) return;
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
  const long ngi = dn_[2]+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long ngj = dn_[1]+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long n = long(dn_[2])*dn_[1]*dn_[0];
  gpuFor(n,GPU_LAMBDA(const long i) { da[i] = scale*(db[i]-offset); });
  pp_->solve(minBytes_,da,db);
  gpuFor(
    dn_[0],dn_[1],dn_[2],
    GPU_LAMBDA(const long ia, const int k, const int j, const int i) {
      const long ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
      db[ib] = da[ia];
    });
  assert(potential);
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToHost));
}

void Potential_Paris_3D::Initialize(const Real lx, const Real ly, const Real lz, const Real xMin, const Real yMin, const Real zMin, const int nx, const int ny, const int nz, const int nxReal, const int nyReal, const int nzReal, const Real dx, const Real dy, const Real dz, const bool periodic)
{
  chprintf(" Using Poisson Solver: Paris ");
  if (periodic) chprintf("Periodic\n");
  else chprintf("Antisymmetric\n");

  dn_[0] = nzReal;
  dn_[1] = nyReal;
  dn_[2] = nxReal;

  dr_[0] = dz;
  dr_[1] = dy;
  dr_[2] = dx;

  n_[0] = nz;
  n_[1] = ny;
  n_[2] = nx;

  const double myLo[3] = {zMin,yMin,xMin};
  MPI_Allreduce(myLo,lo_,3,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
  hi_[0] = lo_[0]+lz-dr_[0];
  hi_[1] = lo_[1]+ly-dr_[1];
  hi_[2] = lo_[2]+lx-dr_[2];
  const int m[3] = {n_[0]/nzReal,n_[1]/nyReal,n_[2]/nxReal};
  const int id[3] = {int(round((zMin-lo_[0])/dr_[0])),int(round((yMin-lo_[1])/dr_[1])),int(round((xMin-lo_[2])/dr_[2]))};
  chprintf("  Paris: [ %g %g %g ]-[ %g %g %g ] N_local[ %d %d %d ] Tasks[ %d %d %d ]\n",lo_[2],lo_[1],lo_[0],lo_[2]+lx,lo_[1]+ly,lo_[0]+lz,dn_[2],dn_[1],dn_[0],m[2],m[1],m[0]);
  assert(dn_[0] == n_[0]/m[0]);
  assert(dn_[1] == n_[1]/m[1]);
  assert(dn_[2] == n_[2]/m[2]);

  if (periodic) {
    pp_ = new PoissonPeriodic3DBlockedGPU(n_,lo_,hi_,m,id);
    assert(pp_);
    minBytes_ = pp_->bytes();
  } else {
    pz_ = new PoissonZero3DBlockedGPU(n_,lo_,hi_,m,id);
    assert(pz_);
    minBytes_ = pz_->bytes();
  }

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

  potentialBytes_ = 0;
  densityBytes_ = 0;
  minBytes_ = 0;

  if (pz_) delete pz_;
  pz_ = nullptr;

  if (pp_) delete pp_;
  pp_ = nullptr;

  dn_[2] = dn_[1] = dn_[0] = 0;
}

#endif

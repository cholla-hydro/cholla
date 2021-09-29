#if defined(GRAVITY) && defined(PARIS_BC)

#include "../gravity/potential_paris_BC_3D.h"
#include "../io/io.h"
#include "../utils/gpu.hpp"
#include <cassert>

Potential_Paris_BC_3D::Potential_Paris_BC_3D():
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

Potential_Paris_BC_3D::~Potential_Paris_BC_3D() { Reset(); }

void Potential_Paris_BC_3D::Get_Potential(const Real *const density, Real *const potential, const Real g, const Real offset, const Real a)
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

  const int ngi = ni+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const int ngj = nj+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;

  const Real ddx = 1.0/(scale*dr_[2]*dr_[2]);
  const Real ddy = 1.0/(scale*dr_[1]*dr_[1]);
  const Real ddz = 1.0/(scale*dr_[0]*dr_[0]);
  const int nij = ni*nj;

  {
#ifdef GRAVITY_GPU
    const Real *const phi = potential;
    Real *const rho = density;
#else
    CHECK(cudaMemcpyAsync(db,potential,potentialBytes_,cudaMemcpyHostToDevice,0));
    CHECK(cudaMemcpyAsync(da,density,densityBytes_,cudaMemcpyHostToDevice,0));
    const Real *const phi = db;
    Real *const rho = da;
#endif
    gpuFor(
      nk,nj,ni,
      GPU_LAMBDA(const int k, const int j, const int i) {
        const int ia = i+ni*(j+nj*k);
        const int ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
        const Real div = ddx*(phi[ib-1]+phi[ib+1]-2.0*phi[ib])+ddy*(phi[ib-ni]+phi[ib+ni]-2.0*phi[ib])+ddz*(phi[ib-nij]+phi[ib+nij]-2.0*phi[ib]);
        da[ia] = scale*(rho[ia]-offset-div);
      });
  }

  pp_->solve(minBytes_,da,db);

  {
#ifdef GRAVITY_GPU
    Real *const phi = potential;
    const Real *const dPhi = db;
#else
    CHECK(cudaMemcpyAsync(da,db,densityBytes_,cudaMemcpyDeviceToDevice,0));
    CHECK(cudaMemcpyAsync(db,potential,potentialBytes_,cudaMemcpyHostToDevice,0));
    Real *const phi = db;
    const Real *const dPhi = da;
#endif
    gpuFor(
      nk,nj,ni,
      GPU_LAMBDA(const int k, const int j, const int i) {
        const int ia = i+ni*(j+nj*k);
        const int ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
        phi[ib] += dPhi[ia];
      });
#ifndef GRAVITY_GPU
    CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToHost));
#endif
  }
}

void Potential_Paris_BC_3D::Initialize(const Real lx, const Real ly, const Real lz, const Real xMin, const Real yMin, const Real zMin, const int nx, const int ny, const int nz, const int nxReal, const int nyReal, const int nzReal, const Real dx, const Real dy, const Real dz)
{
  chprintf(" using poisson solver: ");
  chprintf("3-point");
  chprintf(" Paris with boundary conditions\n");

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
  chprintf("  Paris BC: [ %g %g %g ]-[ %g %g %g ] n_local[ %d %d %d ] tasks[ %d %d %d ]\n",lo_[2],lo_[1],lo_[0],lo_[2]+lx,lo_[1]+ly,lo_[0]+lz,dn_[2],dn_[1],dn_[0],m[2],m[1],m[0]);

  assert(dn_[0] == n[0]/m[0]);
  assert(dn_[1] == n[1]/m[1]);
  assert(dn_[2] == n[2]/m[2]);

  pp_ = new PoissonZero3DBlockedGPU(n,lo_,hi,m,id);
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

void Potential_Paris_BC_3D::Reset()
{
  if (db_) CHECK(cudaFree(db_));
  db_ = nullptr;

  if (da_) CHECK(cudaFree(da_));
  da_ = nullptr;

  potentialBytes_ = densityBytes_ = minBytes_ = 0;

  if (pp_) delete pp_;
  pp_ = nullptr;

  myLo_[2] = myLo_[1] = myLo_[0] = 0;
  lr_[2] = lr_[1] = lr_[0] = 0;
  lo_[2] = lo_[1] = lo_[0] = 0;
  dr_[2] = dr_[1] = dr_[0] = 0;
  dn_[2] = dn_[1] = dn_[0] = 0;
}

#endif

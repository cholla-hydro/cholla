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
  da_(nullptr),
  db_(nullptr)
#ifndef GRAVITY_GPU
  ,potentialBytes_(0),
  dc_(nullptr)
#endif
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
  const int ngij = ngi*ngj;

#ifdef GRAVITY_GPU
  Real *const phi = potential;
  const Real *const rho = density;
#else
  CHECK(cudaMemcpyAsync(dc_,potential,potentialBytes_,cudaMemcpyHostToDevice,0));
  Real *const phi = dc_;
  CHECK(cudaMemcpyAsync(da,density,densityBytes_,cudaMemcpyHostToDevice,0));
  const Real *const rho = da;
#endif

  gpuFor(
    nk,nj,ni,
    GPU_LAMBDA(const int k, const int j, const int i) {
      const int ia = i+ni*(j+nj*k);
      const int ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
      const Real div = ddx*(phi[ib-1]+phi[ib+1]-2.0*phi[ib])+ddy*(phi[ib-ngi]+phi[ib+ngi]-2.0*phi[ib])+ddz*(phi[ib-ngij]+phi[ib+ngij]-2.0*phi[ib]);
      da[ia] = scale*(rho[ia]-offset-div);
    });

  pp_->solve(minBytes_,da,db);

  gpuFor(
    nk,nj,ni,
    GPU_LAMBDA(const int k, const int j, const int i) {
      const int ia = i+ni*(j+nj*k);
      const int ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
      phi[ib] += db[ia];
    });

#ifndef GRAVITY_GPU
  CHECK(cudaMemcpy(potential,phi,potentialBytes_,cudaMemcpyDeviceToHost));
#endif
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

  myLo_[0] = zMin+0.5*dr_[0];
  myLo_[1] = yMin+0.5*dr_[1];
  myLo_[2] = xMin+0.5*dr_[2];
  MPI_Allreduce(myLo_,lo_,3,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);

  const Real hi[3] = {lo_[0]+lr_[0]-dr_[0],lo_[1]+lr_[1]-dr_[1],lo_[2]+lr_[1]-dr_[2]};
  const int n[3] = {nz,ny,nx};
  const int m[3] = {n[0]/nzReal,n[1]/nyReal,n[2]/nxReal};
  const int id[3] = {int(round((myLo_[0]-lo_[0])/(dn_[0]*dr_[0]))),int(round((myLo_[1]-lo_[1])/(dn_[1]*dr_[1]))),int(round((myLo_[2]-lo_[2])/(dn_[2]*dr_[2])))};
  chprintf("  Paris BC: [ %g %g %g ]-[ %g %g %g ] n_local[ %d %d %d ] tasks[ %d %d %d ]\n",lo_[2],lo_[1],lo_[0],hi[2],hi[1],hi[0],dn_[2],dn_[1],dn_[0],m[2],m[1],m[0]);

  assert(dn_[0] == n[0]/m[0]);
  assert(dn_[1] == n[1]/m[1]);
  assert(dn_[2] == n[2]/m[2]);

  pp_ = new PoissonZero3DBlockedGPU(n,lo_,hi,m,id);
  assert(pp_);
  minBytes_ = pp_->bytes();
  densityBytes_ = long(sizeof(Real))*dn_[0]*dn_[1]*dn_[2];

  CHECK(cudaMalloc(reinterpret_cast<void **>(&da_),std::max(minBytes_,densityBytes_)));
  assert(da_);
  CHECK(cudaMalloc(reinterpret_cast<void **>(&db_),std::max(minBytes_,densityBytes_)));
  assert(db_);

#ifndef GRAVITY_GPU
  const long gg = N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  potentialBytes_ = long(sizeof(Real))*(dn_[0]+gg)*(dn_[1]+gg)*(dn_[2]+gg);
  CHECK(cudaMalloc(reinterpret_cast<void **>(&dc_),potentialBytes_));
  assert(dc_);
#endif
}

void Potential_Paris_BC_3D::Reset()
{
#ifndef GRAVITY_GPU
  if (dc_) CHECK(cudaFree(dc_));
  dc_ = nullptr;
  potentialBytes_ = 0;
#endif

  if (db_) CHECK(cudaFree(db_));
  db_ = nullptr;

  if (da_) CHECK(cudaFree(da_));
  da_ = nullptr;

  densityBytes_ = minBytes_ = 0;

  if (pp_) delete pp_;
  pp_ = nullptr;

  myLo_[2] = myLo_[1] = myLo_[0] = 0;
  lr_[2] = lr_[1] = lr_[0] = 0;
  lo_[2] = lo_[1] = lo_[0] = 0;
  dr_[2] = dr_[1] = dr_[0] = 0;
  dn_[2] = dn_[1] = dn_[0] = 0;
}

#endif

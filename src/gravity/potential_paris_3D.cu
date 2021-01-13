#if defined(GRAVITY) && defined(PARIS)

#include "potential_paris_3D.h"
#include "gpu.hpp"
#include "../io.h"
#include <cassert>
#include <cfloat>


static void __attribute__((unused)) printDiff(const Real *p, const Real *q, const int ng, const int nx, const int ny, const int nz, const bool plot = false)
{
  Real dMax = 0, dSum = 0, dSum2 = 0;
  Real qMax = 0, qSum = 0, qSum2 = 0;
#pragma omp parallel for reduction(max:dMax,qMax) reduction(+:dSum,dSum2,qSum,qSum2)
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        const long ijk = i+ng+(nx+ng+ng)*(j+ng+(ny+ng+ng)*(k+ng));
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
      const long ijk = i+ng+(nx+ng+ng)*(j+ng+(ny+ng+ng)*(k+ng));
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
  pz_(nullptr),
  minBytes_(0),
  densityBytes_(0),
  potentialBytes_(0),
  da_(nullptr),
  db_(nullptr)
{}

Potential_Paris_3D::~Potential_Paris_3D() { Reset(); }

__device__ static Real analyticD(const Real x, const Real y, const Real z, const Real ddlx, const Real ddly, const Real ddlz)
{
  return exp(-x*x-y*y-z*z)*((4.0*x*x-2.0)*ddlx+(4.0*y*y-2.0)*ddly+(4.0*z*z-2.0)*ddlz);
}

__device__ static Real analyticF(const Real x, const Real y, const Real z)
{
  return exp(-x*x-y*y-z*z);
}

void Potential_Paris_3D::Get_Analytic_Potential(const Real *const density, Real *const potential)
{

  const Real dx = dr_[2];
  const Real dy = dr_[1];
  const Real dz = dr_[0];
  const Real xLo = lo_[2];
  const Real yLo = lo_[1];
  const Real zLo = lo_[0];
  const Real lx = lr_[2];
  const Real ly = lr_[1];
  const Real lz = lr_[0];
  const Real xBegin = myLo_[2];
  const Real yBegin = myLo_[1];
  const Real zBegin = myLo_[0];

  assert(da_);
  Real *const da = da_;
  Real *const db = db_;
  assert(density);
  CHECK(cudaMemcpy(db,density,densityBytes_,cudaMemcpyHostToDevice));

  const Real dlx = 2.0/lx;
  const Real dly = 2.0/ly;
  const Real dlz = 2.0/lz;
  const Real bx = -dlx*(xLo+0.5*lx);
  const Real by = -dly*(yLo+0.5*ly);
  const Real bz = -dlz*(zLo+0.5*lz);
  const Real ddlx = dlx*dlx;
  const Real ddly = dly*dly;
  const Real ddlz = dlz*dlz;

  gpuFor(
    dn_[0],dn_[1],dn_[2],
    GPU_LAMBDA(const long kji, const int k, const int j, const int i) {
      const Real x = dlx*(xBegin+dx*(Real(i)+0.5))+bx;
      const Real y = dly*(yBegin+dy*(Real(j)+0.5))+by;
      const Real z = dlz*(zBegin+dz*(Real(k)+0.5))+bz;
      da[kji] = db[kji]-analyticD(x,y,z,ddlx,ddly,ddlz);
      });
      
  assert(pz_);
  pz_->solve(minBytes_,da,db);

  const long ngi = dn_[2]+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long ngj = dn_[1]+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;

  gpuFor(
    dn_[0],dn_[1],dn_[2],
    GPU_LAMBDA(const long ia, const int k, const int j, const int i) {
      const Real x = dlx*(xBegin+dx*(Real(i)+0.5))+bx;
      const Real y = dly*(yBegin+dy*(Real(j)+0.5))+by;
      const Real z = dlz*(zBegin+dz*(Real(k)+0.5))+bz;
      const long ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
      db[ib] = da[ia]+analyticF(x,y,z);
    });

  assert(potential);
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToHost));
}

void Potential_Paris_3D::Get_Potential(const Real *const density, Real *const potential, const Real g, const Real offset, const Real a)
{
#ifdef COSMOLOGY
  const Real scale = Real(4)*M_PI*g/a;
#else
  const Real scale = Real(4)*M_PI*g;
  const Real mass = a;
#endif
  assert(da_);
  Real *const da = da_;
  Real *const db = db_;
  assert(density);
  const long n = long(dn_[2])*dn_[1]*dn_[0];
  CHECK(cudaMemcpy(db,density,densityBytes_,cudaMemcpyHostToDevice));
  const long ngi = dn_[2]+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long ngj = dn_[1]+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;

  if (pp_) {

    gpuFor(n,GPU_LAMBDA(const long i) { da[i] = scale*(db[i]-offset); });
    pp_->solve(minBytes_,da,db);
    gpuFor(
        dn_[0],dn_[1],dn_[2],
        GPU_LAMBDA(const long ia, const int k, const int j, const int i) {
        const long ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
        db[ib] = da[ia];
        });

  } else {

    assert(pz_);
    constexpr Real fraction = 2*6;
    const Real r0 = std::min({lr_[0],lr_[1],lr_[2]})/fraction;
    const Real spi = sqrt(M_PI);
    const Real denom = r0*spi;
    const Real rho0 = mass/(denom*denom*denom);

    const Real dlx = lr_[2]/r0;
    const Real dly = lr_[1]/r0;
    const Real dlz = lr_[0]/r0;
    const Real dx = dr_[2]*dlx;
    const Real dy = dr_[1]*dly;
    const Real dz = dr_[0]*dlz;
    const Real x0 = lo_[2]+0.5*lr_[2];
    const Real y0 = lo_[1]+0.5*lr_[1];
    const Real z0 = lo_[0]+0.5*lr_[0];
    const Real xBegin = (myLo_[2]-x0)*dlx;
    const Real yBegin = (myLo_[1]-y0)*dly;
    const Real zBegin = (myLo_[0]-z0)*dlz;

    gpuFor(
        dn_[0],dn_[1],dn_[2],
        GPU_LAMBDA(const long kji, const int k, const int j, const int i) {
        const Real x = xBegin+dx*(Real(i)+0.5);
        const Real y = yBegin+dy*(Real(j)+0.5);
        const Real z = zBegin+dz*(Real(k)+0.5);
        da[kji] = scale*(db[kji]-offset-rho0*exp(-x*x-y*y-z*z));
        });

    pz_->solve(minBytes_,da,db);

    const Real ngmdr0 = -g*mass/r0;
    const Real lim0 = ngmdr0*Real(2)/spi;
    const Real lim2 = -lim0/Real(3);

    gpuFor(
        dn_[0],dn_[1],dn_[2],
        GPU_LAMBDA(const long ia, const int k, const int j, const int i) {
        const Real x = xBegin+dx*(Real(i)+0.5);
        const Real y = yBegin+dy*(Real(j)+0.5);
        const Real z = zBegin+dz*(Real(k)+0.5);
        const Real r = sqrt(x*x+y*y+z*z);
        const Real v0 = (r < DBL_EPSILON) ? lim0+lim2*r*r : ngmdr0*erf(r)/r;
        const long ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
        db[ib] = da[ia]+v0;
        });
  }
  assert(potential);
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToHost));
}

void Potential_Paris_3D::Initialize(const Real lx, const Real ly, const Real lz, const Real xMin, const Real yMin, const Real zMin, const int nx, const int ny, const int nz, const int nxReal, const int nyReal, const int nzReal, const Real dx, const Real dy, const Real dz, const bool periodic)
{
  chprintf(" Using Poisson Solver: Paris ");
  if (periodic) chprintf("Periodic");
  else chprintf("Antisymmetric");
#ifdef PARIS_5PT
  chprintf(" 5-Point\n");
#elif defined PARIS_3PT
  chprintf(" 3-Point\n");
#else
  chprintf(" Spectral\n");
#endif

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

  if (periodic) {
    pp_ = new PoissonPeriodic3DBlockedGPU(n,lo_,hi,m,id);
    assert(pp_);
    minBytes_ = pp_->bytes();
  } else {
    pz_ = new PoissonZero3DBlockedGPU(n,lo_,hi,m,id);
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

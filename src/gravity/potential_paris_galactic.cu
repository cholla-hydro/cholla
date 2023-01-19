#ifdef PARIS_GALACTIC

  #include <cassert>

  #include "../gravity/potential_paris_galactic.h"
  #include "../io/io.h"
  #include "../utils/gpu.hpp"

Potential_Paris_Galactic::Potential_Paris_Galactic()
    : dn_{0, 0, 0},
      dr_{0, 0, 0},
      lo_{0, 0, 0},
      lr_{0, 0, 0},
      myLo_{0, 0, 0},
      pp_(nullptr),
      densityBytes_(0),
      minBytes_(0),
      da_(nullptr),
      db_(nullptr)
  #ifndef GRAVITY_GPU
      ,
      potentialBytes_(0),
      dc_(nullptr)
  #endif
{
}

Potential_Paris_Galactic::~Potential_Paris_Galactic() { Reset(); }

void Potential_Paris_Galactic::Get_Potential(const Real *const density,
                                             Real *const potential,
                                             const Real g,
                                             const DiskGalaxy &galaxy)
{
  const Real scale = Real(4) * M_PI * g;

  assert(da_);
  Real *const da = da_;
  Real *const db = db_;
  assert(density);

  const int ni = dn_[2];
  const int nj = dn_[1];
  const int nk = dn_[0];

  const int ngi = ni + N_GHOST_POTENTIAL + N_GHOST_POTENTIAL;
  const int ngj = nj + N_GHOST_POTENTIAL + N_GHOST_POTENTIAL;

  #ifdef GRAVITY_GPU
  const Real *const rho = density;
  Real *const phi       = potential;
  #else
  CHECK(cudaMemcpyAsync(da, density, densityBytes_, cudaMemcpyHostToDevice, 0));
  CHECK(cudaMemcpyAsync(dc_, potential, potentialBytes_, cudaMemcpyHostToDevice,
                        0));
  const Real *const rho = da;
  Real *const phi       = dc_;
  #endif

  const Real xMin = myLo_[2];
  const Real yMin = myLo_[1];
  const Real zMin = myLo_[0];

  const Real dx = dr_[2];
  const Real dy = dr_[1];
  const Real dz = dr_[0];

  const Real md = SIMULATED_FRACTION * galaxy.getM_d();
  const Real rd = galaxy.getR_d();
  const Real zd = galaxy.getZ_d();

  const Real rho0 = md * zd * zd / (4.0 * M_PI);
  gpuFor(
      nk, nj, ni, GPU_LAMBDA(const int k, const int j, const int i) {
        const int ia = i + ni * (j + nj * k);

        const Real x = xMin + i * dx;
        const Real y = yMin + j * dy;
        const Real z = zMin + k * dz;

        const Real r = sqrt(x * x + y * y);
        const Real a = sqrt(z * z + zd * zd);
        const Real b = rd + a;
        const Real c = r * r + b * b;
        const Real dRho =
            rho0 * (rd * c + 3.0 * a * b * b) / (a * a * a * pow(c, 2.5));

        da[ia] = scale * (rho[ia] - dRho);
      });

  pp_->solve(minBytes_, da, db);

  const Real phi0 = -g * md;
  gpuFor(
      nk, nj, ni, GPU_LAMBDA(const int k, const int j, const int i) {
        const int ia = i + ni * (j + nj * k);
        const int ib =
            i + N_GHOST_POTENTIAL +
            ngi * (j + N_GHOST_POTENTIAL + ngj * (k + N_GHOST_POTENTIAL));

        const Real x = xMin + i * dx;
        const Real y = yMin + j * dy;
        const Real z = zMin + k * dz;

        const Real r    = sqrt(x * x + y * y);
        const Real a    = sqrt(z * z + zd * zd);
        const Real b    = a + rd;
        const Real c    = sqrt(r * r + b * b);
        const Real dPhi = phi0 / c;

        phi[ib] = db[ia] + dPhi;
      });

  #ifndef GRAVITY_GPU
  CHECK(cudaMemcpy(potential, dc_, potentialBytes_, cudaMemcpyDeviceToHost));
  #endif
}

void Potential_Paris_Galactic::Initialize(
    const Real lx, const Real ly, const Real lz, const Real xMin,
    const Real yMin, const Real zMin, const int nx, const int ny, const int nz,
    const int nxReal, const int nyReal, const int nzReal, const Real dx,
    const Real dy, const Real dz)
{
  const long nl012 = long(nxReal) * long(nyReal) * long(nzReal);
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

  myLo_[0] = zMin + 0.5 * dr_[0];
  myLo_[1] = yMin + 0.5 * dr_[1];
  myLo_[2] = xMin + 0.5 * dr_[2];
  MPI_Allreduce(myLo_, lo_, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  const Real hi[3] = {lo_[0] + lr_[0] - dr_[0], lo_[1] + lr_[1] - dr_[1],
                      lo_[2] + lr_[1] - dr_[2]};
  const int n[3]   = {nz, ny, nx};
  const int m[3]   = {n[0] / nzReal, n[1] / nyReal, n[2] / nxReal};
  const int id[3]  = {int(round((myLo_[0] - lo_[0]) / (dn_[0] * dr_[0]))),
                      int(round((myLo_[1] - lo_[1]) / (dn_[1] * dr_[1]))),
                      int(round((myLo_[2] - lo_[2]) / (dn_[2] * dr_[2])))};
  chprintf(
      " Paris Galactic: [ %g %g %g ]-[ %g %g %g ] n_local[ %d %d %d ] tasks[ "
      "%d %d %d ]\n",
      lo_[2], lo_[1], lo_[0], hi[2], hi[1], hi[0], dn_[2], dn_[1], dn_[0], m[2],
      m[1], m[0]);

  assert(dn_[0] == n[0] / m[0]);
  assert(dn_[1] == n[1] / m[1]);
  assert(dn_[2] == n[2] / m[2]);

  pp_ = new PoissonZero3DBlockedGPU(n, lo_, hi, m, id);
  assert(pp_);
  minBytes_     = pp_->bytes();
  densityBytes_ = long(sizeof(Real)) * dn_[0] * dn_[1] * dn_[2];

  CHECK(cudaMalloc(reinterpret_cast<void **>(&da_),
                   std::max(minBytes_, densityBytes_)));
  CHECK(cudaMalloc(reinterpret_cast<void **>(&db_),
                   std::max(minBytes_, densityBytes_)));

  #ifndef GRAVITY_GPU
  const long gg = N_GHOST_POTENTIAL + N_GHOST_POTENTIAL;
  potentialBytes_ =
      long(sizeof(Real)) * (dn_[0] + gg) * (dn_[1] + gg) * (dn_[2] + gg);
  CHECK(cudaMalloc(reinterpret_cast<void **>(&dc_), potentialBytes_));
  #endif
}

void Potential_Paris_Galactic::Reset()
{
  #ifndef GRAVITY_GPU
  if (dc_) CHECK(cudaFree(dc_));
  dc_             = nullptr;
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

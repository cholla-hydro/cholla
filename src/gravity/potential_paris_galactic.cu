#ifdef PARIS_GALACTIC

  #include <cassert>

  #include "../global/global.h"
  #include "../gravity/potential_paris_galactic.h"
  #include "../io/io.h"
  #include "../model/disk_galaxy.h"
  #include "../model/potentials.h"
  #include "../utils/error_handling.h"
  #include "../utils/gpu.hpp"

PotentialParisGalactic::PotentialParisGalactic()
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

PotentialParisGalactic::~PotentialParisGalactic() { Reset(); }

void PotentialParisGalactic::Get_Potential(const Real *const density, Real *const potential, const Real grav_const,
                                           const DiskGalaxy &galaxy)
{
  const Real scale = Real(4) * M_PI * grav_const;
  if (grav_const == GN) CHOLLA_ERROR("For consistency, grav_const must be equal to the GN macro");

  assert(da_);
  // we are (presumably) defining aliases for this->da_ and this->db_ since the aliases
  // can be more easily captured by the lambdas.
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
  GPU_Error_Check(cudaMemcpyAsync(da, density, densityBytes_, cudaMemcpyHostToDevice, 0));
  GPU_Error_Check(cudaMemcpyAsync(dc_, potential, potentialBytes_, cudaMemcpyHostToDevice, 0));
  const Real *const rho = da;
  Real *const phi       = dc_;
  #endif

  const Real xMin = myLo_[2];
  const Real yMin = myLo_[1];
  const Real zMin = myLo_[0];

  const Real dx = dr_[2];
  const Real dy = dr_[1];
  const Real dz = dr_[0];

  // We begin the actual calculation

  // The underlying solver of Poisson's equation (called by this function) will ONLY work correctly
  // when solves for a Gravitational potential, Phi, satisfying the following 2 invariants:
  //   1. The laplacian of Phi is 0 at all the boundaries (i.e. density is zero at the boundaries)
  //   2. The gradient of Phi is 0 at all the boundaries (i.e. gravity field -- aka gravitational
  //      acceleration -- is 0 at the boundaries)
  //
  // Consequently, we must pass the underlying solver a density field that implicitly corresponds to
  // a potential satisfying these 2 conditions. Let's call this density field `rho_solver`
  //
  // In general, the `rho_real` field (includes all "dynamic" density -- particles and gas) corresponds
  // to a potential (lets call it `Phi_real`) that does NOT satisfies these requirements. Consequently,
  // we need to do something "clever."
  //
  // This function assumes that we have an analytic approximation for what the potential of the
  // `rho_real` field looks like, called `Phi_approx`. Let's refer to the density field that EXACTLY
  // matches `Phi_approx` as `rho_approx`.
  //
  // The basic idea is:
  //   1. compute `rho_solver = rho_real - rho_approx`
  //   2. have the underlying solver compute `Phi_solver` from `rho_solver`
  //   3. return the value of `Phi_real = Phi_solver + Phi_approx`.
  //
  // It's important that `Phi_approx` accurately approximates the potential of `Phi_real` at the boundaries
  // (it's perfectly fine to be inaccurate inside the domain).
  //   - When this is true, then this means that the values, gradient, and laplacian of `Phi_solver`
  //     are all 0 at the boundaries.
  //   - Consequently, the invariants of the underlying solver are all satisfied.

  // Load in the object to approximate the gravitation potential
  // -> we are implicitly assuming that the gas disk is the only source of dynamical density
  //    (i.e. the `rho_real` array is dominated by gas density)
  // -> we are currently ignoring contributions from particles
  //
  // NOTE: At the time of writing, we aren't accounting for the fact that the gas-disk is truncated.
  //       This is almost certainly significant!
  const AprroxExponentialDisk3MN approx_potential = galaxies::MW.getGasDisk().selfgrav_approx_potential;

  // STEP 1: compute the RHS of Poisson's equation that will be passed to the solver
  //  -> in more detail we are computing `4 * pi * G * rho_solver`
  //  -> as noted above, `rho_solver = rho_real - rho_approx`
  gpuFor(
      nk, nj, ni, GPU_LAMBDA(const int k, const int j, const int i) {
        const int ia = i + ni * (j + nj * k);

        const Real x = xMin + i * dx;
        const Real y = yMin + j * dy;
        const Real z = zMin + k * dz;
        const Real R = sqrt((x * x) + (y * y));

        da[ia] = scale * (rho[ia] - approx_potential.rho_disk_D3D(R, z));
      });

  // STEP 2: actually solve poisson's equation for the density field `rho_solver = rho_real - rho_approx`.
  //         The resulting gravitational potential is stored in `db`
  pp_->solve(minBytes_, da, db);

  // STEP 3: Compute the gravitational potential corresponding to `rho_rea;` and store
  // it inside the phi pointer at each spatial location
  // - `db` currently holds gravitational potential, `Phi_solver = (Phi_real - Phi_approx)`, for the density
  //   field of `rho_solver = (rho_real - rho_approx)`
  // - To get `Phi_real`, we simply compute `Phi_approx + Phi_solver`.
  gpuFor(
      nk, nj, ni, GPU_LAMBDA(const int k, const int j, const int i) {
        const int ia = i + ni * (j + nj * k);
        const int ib = i + N_GHOST_POTENTIAL + ngi * (j + N_GHOST_POTENTIAL + ngj * (k + N_GHOST_POTENTIAL));

        const Real x = xMin + i * dx;
        const Real y = yMin + j * dy;
        const Real z = zMin + k * dz;
        const Real R = sqrt((x * x) + (y * y));

        phi[ib] = db[ia] + approx_potential.phi_disk_D3D(R, z);
      });

  #ifdef GRAVITY_GPU
  // in this case, potential is a device pointer and it directly aliases the phi pointer
  // (so we don't have to do anything)
  #else
  GPU_Error_Check(cudaMemcpy(potential, dc_, potentialBytes_, cudaMemcpyDeviceToHost));
  #endif
}

void PotentialParisGalactic::Initialize(const Real lx, const Real ly, const Real lz, const Real xMin, const Real yMin,
                                        const Real zMin, const int nx, const int ny, const int nz, const int nxReal,
                                        const int nyReal, const int nzReal, const Real dx, const Real dy, const Real dz)
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

  const Real hi[3] = {lo_[0] + lr_[0] - dr_[0], lo_[1] + lr_[1] - dr_[1], lo_[2] + lr_[1] - dr_[2]};
  const int n[3]   = {nz, ny, nx};
  const int m[3]   = {n[0] / nzReal, n[1] / nyReal, n[2] / nxReal};
  const int id[3]  = {int(round((myLo_[0] - lo_[0]) / (dn_[0] * dr_[0]))),
                      int(round((myLo_[1] - lo_[1]) / (dn_[1] * dr_[1]))),
                      int(round((myLo_[2] - lo_[2]) / (dn_[2] * dr_[2])))};
  chprintf(
      " Paris Galactic: [ %g %g %g ]-[ %g %g %g ] n_local[ %d %d %d ] tasks[ "
      "%d %d %d ]\n",
      lo_[2], lo_[1], lo_[0], hi[2], hi[1], hi[0], dn_[2], dn_[1], dn_[0], m[2], m[1], m[0]);

  assert(dn_[0] == n[0] / m[0]);
  assert(dn_[1] == n[1] / m[1]);
  assert(dn_[2] == n[2] / m[2]);

  pp_ = new PoissonZero3DBlockedGPU(n, lo_, hi, m, id);
  assert(pp_);
  minBytes_     = pp_->bytes();
  densityBytes_ = long(sizeof(Real)) * dn_[0] * dn_[1] * dn_[2];

  GPU_Error_Check(cudaMalloc(reinterpret_cast<void **>(&da_), std::max(minBytes_, densityBytes_)));
  GPU_Error_Check(cudaMalloc(reinterpret_cast<void **>(&db_), std::max(minBytes_, densityBytes_)));

  #ifndef GRAVITY_GPU
  const long gg   = N_GHOST_POTENTIAL + N_GHOST_POTENTIAL;
  potentialBytes_ = long(sizeof(Real)) * (dn_[0] + gg) * (dn_[1] + gg) * (dn_[2] + gg);
  GPU_Error_Check(cudaMalloc(reinterpret_cast<void **>(&dc_), potentialBytes_));
  #endif
}

void PotentialParisGalactic::Reset()
{
  #ifndef GRAVITY_GPU
  if (dc_) {
    GPU_Error_Check(cudaFree(dc_));
  }
  dc_             = nullptr;
  potentialBytes_ = 0;
  #endif

  if (db_) {
    GPU_Error_Check(cudaFree(db_));
  }
  db_ = nullptr;

  if (da_) {
    GPU_Error_Check(cudaFree(da_));
  }
  da_ = nullptr;

  densityBytes_ = minBytes_ = 0;

  if (pp_) {
    delete pp_;
  }
  pp_ = nullptr;

  myLo_[2] = myLo_[1] = myLo_[0] = 0;
  lr_[2] = lr_[1] = lr_[0] = 0;
  lo_[2] = lo_[1] = lo_[0] = 0;
  dr_[2] = dr_[1] = dr_[0] = 0;
  dn_[2] = dn_[1] = dn_[0] = 0;
}

#endif

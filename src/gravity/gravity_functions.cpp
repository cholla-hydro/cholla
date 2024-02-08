#ifdef GRAVITY

  #include <cstring>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../mpi/cuda_mpi_routines.h"
  #include "../utils/error_handling.h"

  #ifdef PARALLEL_OMP
    #include "../utils/parallel_omp.h"
  #endif

  #if defined(PARIS_TEST) || defined(PARIS_GALACTIC_TEST)
    #include <vector>
  #endif

  // #ifdef PARTICLES
  #include "../model/disk_galaxy.h"
// #endif

// Set delta_t when usi#ng gravity
void Grid3D::set_dt_Gravity()
{
  // Delta_t for the hydro
  Real dt_hydro = H.dt;

  #ifdef AVERAGE_SLOW_CELLS
  Real min_dt_slow;
  #endif

  #ifdef PARTICLES
  // Compute delta_t for particles and choose min(dt_particles, dt_hydro)
  Real dt_particles, dt_min;

    #ifdef COSMOLOGY
  chprintf("Current_z: %f \n", Cosmo.current_z);
  Real da_particles, da_min, dt_physical;

  // Compute the particles delta_t
  Particles.dt = Calc_Particles_dt_Cosmo();
  dt_particles = Particles.dt;
  // Convert delta_t to delta_a ( a = scale factor )
  da_particles = Cosmo.Get_da_from_dt(dt_particles);
  da_particles = fmin(da_particles, 1.0);  // Limit delta_a

      #ifdef ONLY_PARTICLES
  // If only particles da_min is only da_particles
  da_min = da_particles;
  chprintf(" Delta_a_particles: %f \n", da_particles);

      #else  // NOT ONLY_PARTICLES
  // Here da_min is the minumum between da_particles and da_hydro
  Real da_hydro;
  da_hydro =
      Cosmo.Get_da_from_dt(dt_hydro) * Cosmo.current_a * Cosmo.current_a / Cosmo.H0;  // Convet delta_t to delta_a
  da_min = fmin(da_hydro, da_particles);                                              // Find the minumum delta_a
  chprintf(" Delta_a_particles: %f      Delta_a_gas: %f   \n", da_particles, da_hydro);

      #endif  // ONLY_PARTICLES

  // Limit delta_a by the expansion rate
  Cosmo.max_delta_a = fmin(MAX_EXPANSION_RATE * Cosmo.current_a, MAX_DELTA_A);
  if (da_min > Cosmo.max_delta_a) {
    da_min = Cosmo.max_delta_a;
    chprintf(" Seting max delta_a: %f\n", da_min);
  }

      // Small delta_a when reionization starts
      #ifdef COOLING_GRACKLE
  if (fabs(Cosmo.current_a + da_min - Cool.scale_factor_UVB_on) < 0.005) {
    da_min /= 2;
    chprintf(" Starting UVB. Limiting delta_a:  %f \n", da_min);
  }
      #endif
      #ifdef CHEMISTRY_GPU
  if (fabs(Cosmo.current_a + da_min - Chem.scale_factor_UVB_on) < 0.005) {
    da_min /= 2;
    chprintf(" Starting UVB. Limiting delta_a:  %f \n", da_min);
  }
      #endif

  // Limit delta_a if it's time to output
  if ((Cosmo.current_a + da_min) > Cosmo.next_output) {
    da_min       = Cosmo.next_output - Cosmo.current_a;
    H.Output_Now = true;
  }

      #ifdef ANALYSIS
  // Limit delta_a if it's time to run analysis
  if (Analysis.next_output_indx < Analysis.n_outputs) {
    if (H.Output_Now && fabs(Cosmo.current_a + da_min - Analysis.next_output) < 1e-6)
      Analysis.Output_Now = true;
    else if (Cosmo.current_a + da_min > Analysis.next_output) {
      da_min              = Analysis.next_output - Cosmo.current_a;
      Analysis.Output_Now = true;
    }
  }
      #endif

  if (da_min < 0) {
    chprintf("ERROR: Negative delta_a");
    exit(-1);
  }

  // Set delta_a after it has been computed
  Cosmo.delta_a = da_min;
  // Convert delta_a back to delta_t
  dt_min = Cosmo.Get_dt_from_da(Cosmo.delta_a) * Cosmo.H0 / (Cosmo.current_a * Cosmo.current_a);
  // Set the new delta_t for the hydro step
  H.dt = dt_min;
  chprintf(" Current_a: %f    delta_a: %f     dt:  %f\n", Cosmo.current_a, Cosmo.delta_a, H.dt);

      #ifdef AVERAGE_SLOW_CELLS
  // Set the min_delta_t for averaging a slow cell
  da_particles = fmin(da_particles, Cosmo.max_delta_a);
  min_dt_slow  = Cosmo.Get_dt_from_da(da_particles) / Particles.C_cfl * Cosmo.H0 / (Cosmo.current_a * Cosmo.current_a) /
                SLOW_FACTOR;
  H.min_dt_slow = min_dt_slow;
      #endif

  // Compute the physical time
  dt_physical   = Cosmo.Get_dt_from_da(Cosmo.delta_a);
  Cosmo.dt_secs = dt_physical * Cosmo.time_conversion;
  Cosmo.t_secs += Cosmo.dt_secs;
  chprintf(" t_physical: %f Myr   dt_physical: %f Myr\n", Cosmo.t_secs / MYR, Cosmo.dt_secs / MYR);
  Particles.dt = dt_physical;

    #else  // Not Cosmology
  // If NOT using COSMOLOGY

  // Compute the particles delta_t
  dt_particles = Calc_Particles_dt();
  dt_particles = fmin(dt_particles, Particles.max_dt);
      #ifdef ONLY_PARTICLES
  dt_min = dt_particles;
  chprintf(" dt_particles: %f \n", dt_particles);
      #else
  chprintf(" dt_hydro: %f   dt_particles: %f \n", dt_hydro, dt_particles);
  // Get the minimum delta_t between hydro and particles
  dt_min = fmin(dt_hydro, dt_particles);
      #endif  // ONLY_PARTICLES

      #ifdef AVERAGE_SLOW_CELLS
  // Set the min_delta_t for averaging a slow cell
  // min_dt_slow = dt_particles / Particles.C_cfl / SLOW_FACTOR;
  min_dt_slow   = 3 * H.dx;
  H.min_dt_slow = min_dt_slow;
      #endif

  // Set the new delta_t
  H.dt         = dt_min;
  Particles.dt = H.dt;
    #endif  // COSMOLOGY
  #endif    // PARTICLES

  #if defined(AVERAGE_SLOW_CELLS) && !defined(PARTICLES)
  // Set the min_delta_t for averaging a slow cell ( for now the min_dt_slow is
  // set to a large value, change this with your condition ) min_dt_slow = H.dt
  // / C_cfl * 100 ;
  min_dt_slow   = 3 * H.dx;
  H.min_dt_slow = min_dt_slow;
  #endif

  // Set current and previous delta_t for the potential extrapolation
  if (Grav.INITIAL) {
    Grav.dt_prev = H.dt;
    Grav.dt_now  = H.dt;
  } else {
    Grav.dt_prev = Grav.dt_now;
    Grav.dt_now  = H.dt;
  }

  #if defined(PARTICLES_GPU) && defined(PRINT_MAX_MEMORY_USAGE)
  Particles.Print_Max_Memory_Usage();
  #endif
}

// NOT USED: Get Average density on the Global dommain
Real Grav3D::Get_Average_Density()
{
  Real dens_sum, dens_mean;

  #ifndef PARALLEL_OMP
  dens_sum = Get_Average_Density_function(0, nz_local);
  #else
  dens_sum = 0;
  Real dens_sum_all[N_OMP_THREADS];
    #pragma omp parallel num_threads(N_OMP_THREADS)
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id      = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs(nz_local, n_omp_procs, omp_id, &g_start, &g_end);
    dens_sum_all[omp_id] = Get_Average_Density_function(g_start, g_end);
  }
  for (Real dens_sum_all_element : dens_sum_all) {
    dens_sum += dens_sum_all_element;
  }
  #endif

  dens_mean = dens_sum / (nx_local * ny_local * nz_local);

  Real dens_avrg_all;
  #ifdef MPI_CHOLLA
  dens_avrg_all = ReduceRealAvg(dens_mean);
  #else
  dens_avrg_all = dens_mean;
  #endif

  dens_avrg = dens_avrg_all;

  return dens_avrg_all;
}

// NOT USED: Function to get Average density on the Global dommain
Real Grav3D::Get_Average_Density_function(int g_start, int g_end)
{
  int nx = nx_local;
  int ny = ny_local;
  int nz = nz_local;
  int k, j, i, id;
  Real dens_sum = 0;
  for (k = g_start; k < g_end; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        id = (i) + (j)*nx + (k)*nx * ny;
        dens_sum += F.density_h[id];
      }
    }
  }
  return dens_sum;
}

  #ifdef PARIS_TEST

static inline Real sqr(const Real x) { return x * x; }

static inline Real f1(const Real x) { return exp(-10.0 * sqr(2.0 * x - 1.0)) * sin(8.0 * M_PI * x); }

static inline Real d1(const Real x)
{
  return 16.0 * exp(-10.0 * sqr(2.0 * x - 1.0)) *
         ((400.0 * x * x - 400.0 * x - 4.0 * M_PI * M_PI + 95.0) * sin(8.0 * M_PI * x) +
          (40.0 * M_PI - 80.0 * M_PI * x) * cos(8.0 * M_PI * x));
}

static inline Real periodicF(const Real x, const Real y, const Real z) { return f1(x) * f1(y) * f1(z); }

static inline Real periodicD(const Real x, const Real y, const Real z, const Real ddlx, const Real ddly,
                             const Real ddlz)
{
  return ddlx * d1(x) * f1(y) * f1(z) + ddly * f1(x) * d1(y) * f1(z) + ddlz * f1(x) * f1(y) * d1(z);
}

static constexpr Real twoPi  = 2.0 * M_PI;
static constexpr Real fourPi = 4.0 * M_PI;
static constexpr Real sixPi2 = 6.0 * M_PI * M_PI;

static inline Real nonzeroF(const Real x, const Real y, const Real z)
{
  const Real sx = sin(twoPi * x);
  const Real sy = sin(twoPi * y);
  const Real sz = sin(twoPi * z);
  const Real f  = exp(-x * x - y * y - z * z);
  return sx * sx * sx * sy * sy * sy * sz * sz * sz + f;
}

static inline Real nonzeroD(const Real x, const Real y, const Real z, const Real ddlx, const Real ddly, const Real ddlz)
{
  const Real sx  = sin(twoPi * x);
  const Real sy  = sin(twoPi * y);
  const Real sz  = sin(twoPi * z);
  const Real sx3 = sx * sx * sx;
  const Real sy3 = sy * sy * sy;
  const Real sz3 = sz * sz * sz;
  const Real f   = exp(-x * x - y * y - z * z);
  const Real df  = ddlx * (4.0 * x * x - 2.0) + ddly * (4.0 * y * y - 2.0) + ddlz * (4.0 * z * z - 2.0);
  return (ddlx * sx * (3.0 * cos(fourPi * x) + 1.0) * sy3 * sz3 +
          ddly * sx3 * sy * (3.0 * cos(fourPi * y) + 1.0) * sz3 +
          ddlz * sx3 * sy3 * sz * (3.0 * cos(fourPi * z) + 1.0)) *
             sixPi2 +
         f * df;
}
  #endif

  #if defined(PARIS_TEST) || defined(PARIS_GALACTIC_TEST)
static void printDiff(const Real *p, const Real *q, const int nx, const int ny, const int nz,
                      const int ng = N_GHOST_POTENTIAL, const bool plot = false)
{
  Real dMax = 0, dSum = 0, dSum2 = 0;
  Real qMax = 0, qSum = 0, qSum2 = 0;
    #pragma omp parallel for reduction(max : dMax, qMax) reduction(+ : dSum, dSum2, qSum, qSum2)
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        const long ijk  = i + ng + (nx + ng + ng) * (j + ng + (ny + ng + ng) * (k + ng));
        const Real qAbs = fabs(q[ijk]);
        qMax            = std::max(qMax, qAbs);
        qSum += qAbs;
        qSum2 += qAbs * qAbs;
        const Real d = fabs(q[ijk] - p[ijk]);
        dMax         = std::max(dMax, d);
        dSum += d;
        dSum2 += d * d;
      }
    }
  }
  Real maxs[2] = {qMax, dMax};
  Real sums[4] = {qSum, qSum2, dSum, dSum2};
  MPI_Allreduce(MPI_IN_PLACE, &maxs, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &sums, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  chprintf(" Poisson-Solver Diff: L1 %g L2 %g Linf %g\n", sums[2] / sums[0], sqrt(sums[3] / sums[1]),
           maxs[1] / maxs[0]);
  fflush(stdout);
  if (!plot) return;

  printf("###\n");
    #if 0
  int kMax = -1;
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        const long ijk = i+ng+(nx+ng+ng)*(j+ng+(ny+ng+ng)*(k+ng));
        const Real qAbs = fabs(q[ijk]);
        if (qAbs == qMax) kMax = k;
      }
    }
    if (kMax > -1) {
    #endif
  const int k = nz / 2;
  for (int j = 0; j < ny + ng + ng; j++) {
    for (int i = 0; i < nx + ng + ng; i++) {
      const long ijk = i + (nx + ng + ng) * (j + (ny + ng + ng) * (k + ng));
      printf("%d %d %g %g %g\n", j, i, q[ijk], p[ijk], q[ijk] - p[ijk]);
    }
    printf("\n");
  }
    #if 0
      break;
    }
  }
    #endif
  fflush(stdout);
  MPI_Finalize();
  exit(0);
}
  #endif

// Initialize the Grav Object at the beginning of the simulation
void Grid3D::Initialize_Gravity(struct Parameters *P)
{
  chprintf("\nInitializing Gravity... \n");
  Grav.Initialize(H.xblocal, H.yblocal, H.zblocal, H.xblocal_max, H.yblocal_max, H.zblocal_max, H.xdglobal, H.ydglobal,
                  H.zdglobal, P->nx, P->ny, P->nz, H.nx_real, H.ny_real, H.nz_real, H.dx, H.dy, H.dz,
                  H.n_ghost_potential_offset, P);
  chprintf("Gravity Successfully Initialized. \n\n");

  if (P->bc_potential_type == 1) {
    const int ng    = N_GHOST_POTENTIAL;
    const int twoNG = ng + ng;
    const int nk    = Grav.nz_local + twoNG;
    const int nj    = Grav.ny_local + twoNG;
    const int ni    = Grav.nx_local + twoNG;
    const Real dr   = 0.5 - ng;

  #ifdef PARIS_GALACTIC_TEST
    chprintf("Analytic Test of Poisson Solvers:\n");
    std::vector<Real> exact(Grav.n_cells_potential);
    std::vector<Real> potential(Grav.n_cells_potential);
    const Real scale      = 4.0 * M_PI * Grav.Gconst;
    const Real ddx        = 1.0 / (scale * Grav.dx * Grav.dx);
    const Real ddy        = 1.0 / (scale * Grav.dy * Grav.dy);
    const Real ddz        = 1.0 / (scale * Grav.dz * Grav.dz);
    const Real *const phi = Grav.F.potential_h;
    const int nij         = ni * nj;
    const Real a0         = galaxies::MW.phi_disk_D3D(0, 0);
    const Real da0        = 2.0 / (25.0 * scale);
    #pragma omp parallel for
    for (int k = 0; k < nk; k++) {
      const Real z  = Grav.zMin + Grav.dz * (k + dr);
      const int njk = nj * k;
      for (int j = 0; j < nj; j++) {
        const Real y   = Grav.yMin + Grav.dy * (j + dr);
        const Real yy  = y * y;
        const int nijk = ni * (j + njk);
        for (int i = 0; i < ni; i++) {
          const Real x  = Grav.xMin + Grav.dx * (i + dr);
          const Real r  = sqrt(x * x + yy);
          const int ijk = i + nijk;
          exact[ijk] = potential[ijk] = Grav.F.potential_h[ijk] = galaxies::MW.phi_disk_D3D(r, z);
        }
      }
    }
    #pragma omp parallel for
    for (int k = 0; k < Grav.nz_local; k++) {
      const Real z  = Grav.zMin + Grav.dz * (k + 0.5);
      const Real zz = z * z;
      const int njk = Grav.ny_local * k;
      for (int j = 0; j < Grav.ny_local; j++) {
        const Real y   = Grav.yMin + Grav.dy * (j + 0.5);
        const Real yy  = y * y;
        const int nijk = Grav.nx_local * (j + njk);
        for (int i = 0; i < Grav.nx_local; i++) {
          const Real x          = Grav.xMin + Grav.dx * (i + 0.5);
          const Real r          = sqrt(x * x + yy);
          const int ijk         = i + nijk;
          const Real rr         = x * x + yy + zz;
          const Real f          = a0 * exp(-0.2 * rr);
          const Real df         = da0 * (15.0 - 2.0 * rr) * f;
          Grav.F.density_h[ijk] = galaxies::MW.rho_disk_D3D(r, z) + df;
          const int ib          = i + ng + ni * (j + ng + nj * (k + ng));
          exact[ib] -= f;
        }
      }
    }
    Grav.Poisson_solver_test.Get_Potential(Grav.F.density_h, Grav.F.potential_h, Grav.Gconst, galaxies::MW);
    chprintf(" Paris Galactic");
    printDiff(Grav.F.potential_h, exact.data(), Grav.nx_local, Grav.ny_local, Grav.nz_local);
    Get_Potential_SOR(Grav.Gconst, 0, 0, P);
    chprintf(" SOR");
    printDiff(Grav.F.potential_h, exact.data(), Grav.nx_local, Grav.ny_local, Grav.nz_local);
  #endif

  #ifdef SOR
    chprintf(" Initializing disk analytic potential\n");
    #pragma omp parallel for
    for (int k = 0; k < nk; k++) {
      const Real z  = Grav.zMin + Grav.dz * (k + dr);
      const int njk = nj * k;
      for (int j = 0; j < nj; j++) {
        const Real y   = Grav.yMin + Grav.dy * (j + dr);
        const Real yy  = y * y;
        const int nijk = ni * (j + njk);
        for (int i = 0; i < ni; i++) {
          const Real x            = Grav.xMin + Grav.dx * (i + dr);
          const Real r            = sqrt(x * x + yy);
          const int ijk           = i + nijk;
          Grav.F.potential_h[ijk] = galaxies::MW.phi_disk_D3D(r, z);
        }
      }
    }
  #endif
  }
}

// Compute the Gravitational Potential by solving Poisson Equation
void Grid3D::Compute_Gravitational_Potential(struct Parameters *P)
{
  #ifdef CPU_TIME
  Timer.Grav_Potential.Start();
  #endif

  #ifdef PARTICLES
  // Copy the particles density to the grav_density array
  Copy_Particles_Density_to_Gravity(*P);
  #endif

  #ifndef ONLY_PARTICLES
  // Copy the hydro density to the grav_density array
  Copy_Hydro_Density_to_Gravity();
  #endif

  #ifdef COSMOLOGY
  // If using cosmology, set the gravitational constant to the one in the
  // correct units
  const Real Grav_Constant = Cosmo.cosmo_G;
  const Real current_a     = Cosmo.current_a;
  const Real dens_avrg     = Cosmo.rho_0_gas;
  #else
  const Real Grav_Constant = Grav.Gconst;
  // If slowing the Sphere Collapse problem ( bc_potential_type=0 )
  const Real dens_avrg = (P->bc_potential_type == 0) ? H.sphere_background_density : 0;
  const Real r0        = H.sphere_radius;
  // Re-use current_a as the total mass of the sphere
  const Real current_a = (H.sphere_density - dens_avrg) * 4.0 * M_PI * r0 * r0 * r0 / 3.0;
  #endif

  if (!Grav.BC_FLAGS_SET) {
    Grav.TRANSFER_POTENTIAL_BOUNDARIES = true;
    Set_Boundary_Conditions(*P);
    Grav.TRANSFER_POTENTIAL_BOUNDARIES = false;
    // #ifdef MPI_CHOLLA
    // printf(" Pid: %d Gravity Boundary Flags: %d %d %d %d %d %d \n", procID,
    // Grav.boundary_flags[0], Grav.boundary_flags[1], Grav.boundary_flags[2],
    // Grav.boundary_flags[3], Grav.boundary_flags[4], Grav.boundary_flags[5] );
    // #endif
    Grav.BC_FLAGS_SET = true;
  }

  #ifdef GRAV_ISOLATED_BOUNDARY_X
  if (Grav.boundary_flags[0] == 3) {
    Compute_Potential_Boundaries_Isolated(0, P);
  }
  if (Grav.boundary_flags[1] == 3) {
    Compute_Potential_Boundaries_Isolated(1, P);
  }
  // chprintf("Isolated X\n");
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
  if (Grav.boundary_flags[2] == 3) {
    Compute_Potential_Boundaries_Isolated(2, P);
  }
  if (Grav.boundary_flags[3] == 3) {
    Compute_Potential_Boundaries_Isolated(3, P);
  }
  // chprintf("Isolated Y\n");
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  if (Grav.boundary_flags[4] == 3) {
    Compute_Potential_Boundaries_Isolated(4, P);
  }
  if (Grav.boundary_flags[5] == 3) {
    Compute_Potential_Boundaries_Isolated(5, P);
  }
  // chprintf("Isolated Z\n");
  #endif

  // Solve Poisson Equation to compute the potential
  // Poisson Equation: laplacian( phi ) = 4 * pi * G / scale_factor * ( dens -
  // dens_average )
  Real *input_density, *output_potential;
  #ifdef GRAVITY_GPU
  input_density    = Grav.F.density_d;
  output_potential = Grav.F.potential_d;
  #else
  input_density    = Grav.F.density_h;
  output_potential = Grav.F.potential_h;
  #endif

  #ifdef SOR

    #ifdef PARIS_GALACTIC_TEST
      #ifdef GRAVITY_GPU
        #error "GRAVITY_GPU not yet supported with PARIS_GALACTIC_TEST"
      #endif
  Grav.Poisson_solver_test.Get_Potential(input_density, output_potential, Grav_Constant, galaxies::MW);
  std::vector<Real> p(output_potential, output_potential + Grav.n_cells_potential);
  Get_Potential_SOR(Grav_Constant, dens_avrg, current_a, P);
  chprintf(" Paris vs SOR");
  printDiff(p.data(), output_potential, Grav.nx_local, Grav.ny_local, Grav.nz_local, N_GHOST_POTENTIAL, false);
    #else
  Get_Potential_SOR(Grav_Constant, dens_avrg, current_a, P);
    #endif

  #elif defined PARIS_GALACTIC
  Grav.Poisson_solver.Get_Potential(input_density, output_potential, Grav_Constant, galaxies::MW);
  #else
  Grav.Poisson_solver.Get_Potential(input_density, output_potential, Grav_Constant, dens_avrg, current_a);
  #endif  // SOR

  #ifdef CPU_TIME
  Timer.Grav_Potential.End();
  #endif
}

  #ifdef GRAVITY_ANALYTIC_COMP
void Grid3D::Setup_Analytic_Potential(struct Parameters *P)
{
    #ifndef PARALLEL_OMP
  Setup_Analytic_Galaxy_Potential(0, Grav.nz_local + 2 * N_GHOST_POTENTIAL, galaxies::MW);
    #else
      #pragma omp parallel num_threads(N_OMP_THREADS)
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id      = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs(Grav.nz_local + 2 * N_GHOST_POTENTIAL, n_omp_procs, omp_id, &g_start, &g_end);

    Setup_Analytic_Galaxy_Potential(g_start, g_end, galaxies::MW);
  }
    #endif

    #ifdef GRAVITY_GPU
  GPU_Error_Check(cudaMemcpy(Grav.F.analytic_potential_d, Grav.F.analytic_potential_h,
                             Grav.n_cells_potential * sizeof(Real), cudaMemcpyHostToDevice));
    #endif
}

void Grid3D::Add_Analytic_Potential()
{
    #ifdef GRAVITY_GPU
  Add_Analytic_Potential_GPU();
    #else
      #ifndef PARALLEL_OMP
  Add_Analytic_Potential(0, Grav.nz_local + 2 * N_GHOST_POTENTIAL);
      #else
        #pragma omp parallel num_threads(N_OMP_THREADS)
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id      = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs(Grav.nz_local + 2 * N_GHOST_POTENTIAL, n_omp_procs, omp_id, &g_start, &g_end);

    Add_Analytic_Potential(g_start, g_end);
  }
      #endif  // PARALLEL_OMP
    #endif    // GRAVITY_GPU else
}
  #endif  // GRAVITY_ANALYTIC_COMP

void Grid3D::Copy_Hydro_Density_to_Gravity_Function(int g_start, int g_end)
{
  // Copy the density array from hydro conserved to gravity density array

  Real dens;
  int i, j, k, id, id_grav;
  for (k = g_start; k < g_end; k++) {
    for (j = 0; j < Grav.ny_local; j++) {
      for (i = 0; i < Grav.nx_local; i++) {
        id      = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        id_grav = (i) + (j)*Grav.nx_local + (k)*Grav.nx_local * Grav.ny_local;

        dens = C.density[id];

  // If using cosmology the density must be rescaled to the physical coordinates
  #ifdef COSMOLOGY
        dens *= Cosmo.rho_0_gas;
  #endif

  #ifdef PARTICLES
        Grav.F.density_h[id_grav] += dens;  // Hydro density is added AFTER partices density
  #else
        Grav.F.density_h[id_grav] = dens;
  #endif
      }
    }
  }
}

void Grid3D::Copy_Hydro_Density_to_Gravity()
{
  #ifdef GRAVITY_GPU
  Copy_Hydro_Density_to_Gravity_GPU();
  #else

    #ifndef PARALLEL_OMP
  Copy_Hydro_Density_to_Gravity_Function(0, Grav.nz_local);
    #else

      #pragma omp parallel num_threads(N_OMP_THREADS)
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id      = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs(Grav.nz_local, n_omp_procs, omp_id, &g_start, &g_end);

    Copy_Hydro_Density_to_Gravity_Function(g_start, g_end);
  }
    #endif  // PARALLEL_OMP

  #endif  // GRAVITY_GPU
}

  #ifdef GRAVITY_ANALYTIC_COMP
void Grid3D::Setup_Analytic_Galaxy_Potential(int g_start, int g_end, DiskGalaxy &gal)
{
  int nx = Grav.nx_local + 2 * N_GHOST_POTENTIAL;
  int ny = Grav.ny_local + 2 * N_GHOST_POTENTIAL;
  int nz = Grav.nz_local + 2 * N_GHOST_POTENTIAL;

  // the fraction of the disk that's not modelled (and so its analytic
  // contribution must be added)
  Real non_mod_frac = 1 - SIMULATED_FRACTION;

  int k, j, i, id;
  Real x_pos, y_pos, z_pos, R;
  for (k = g_start; k < g_end; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        id                              = i + j * nx + k * nx * ny;
        x_pos                           = Grav.xMin + Grav.dx * (i - N_GHOST_POTENTIAL) + 0.5 * Grav.dx;
        y_pos                           = Grav.yMin + Grav.dy * (j - N_GHOST_POTENTIAL) + 0.5 * Grav.dy;
        z_pos                           = Grav.zMin + Grav.dz * (k - N_GHOST_POTENTIAL) + 0.5 * Grav.dz;
        R                               = sqrt(x_pos * x_pos + y_pos * y_pos);
        Grav.F.analytic_potential_h[id] = non_mod_frac * gal.phi_disk_D3D(R, z_pos) + gal.phi_halo_D3D(R, z_pos);
      }
    }
  }
}

/**
 * Adds a specified potential function to the potential calculated from solving
 * the Poisson equation. External grav potential not due to simulated matter.
 */
void Grid3D::Add_Analytic_Potential(int g_start, int g_end)
{
  int nx = Grav.nx_local + 2 * N_GHOST_POTENTIAL;
  int ny = Grav.ny_local + 2 * N_GHOST_POTENTIAL;
  int nz = Grav.nz_local + 2 * N_GHOST_POTENTIAL;

  int k, j, i, id;
  Real x_pos, y_pos, z_pos, R;
  for (k = g_start; k < g_end; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        id = i + j * nx + k * nx * ny;
        Grav.F.potential_h[id] += Grav.F.analytic_potential_h[id];
      }
    }
  }
}
  #endif  // GRAVITY_ANALYTIC_COMP

// Extrapolate the potential to obtain phi_n+1/2
void Grid3D::Extrapolate_Grav_Potential_Function(int g_start, int g_end)
{
  // Use phi_n-1 and phi_n to extrapolate the potential and obtain phi_n+1/2

  int nx_pot = Grav.nx_local + 2 * N_GHOST_POTENTIAL;
  int ny_pot = Grav.ny_local + 2 * N_GHOST_POTENTIAL;
  int nz_pot = Grav.nz_local + 2 * N_GHOST_POTENTIAL;

  int n_ghost_grid, nx_grid, ny_grid, nz_grid;
  Real *potential_in, *potential_out;

  // Input potential
  potential_in = Grav.F.potential_h;

  // Output potential
  potential_out = C.Grav_potential;
  // n_ghost for the output potential
  n_ghost_grid = H.n_ghost;

  // Grid size for the output potential
  nx_grid = Grav.nx_local + 2 * n_ghost_grid;
  ny_grid = Grav.ny_local + 2 * n_ghost_grid;
  nz_grid = Grav.nz_local + 2 * n_ghost_grid;

  int nGHST = n_ghost_grid - N_GHOST_POTENTIAL;
  Real pot_now, pot_prev, pot_extrp;
  int k, j, i, id_pot, id_grid;
  for (k = g_start; k < g_end; k++) {
    for (j = 0; j < ny_pot; j++) {
      for (i = 0; i < nx_pot; i++) {
        id_pot  = i + j * nx_pot + k * nx_pot * ny_pot;
        id_grid = (i + nGHST) + (j + nGHST) * nx_grid + (k + nGHST) * nx_grid * ny_grid;
        pot_now = potential_in[id_pot];  // Potential at the n-th timestep
        if (Grav.INITIAL) {
          pot_extrp = pot_now;  // The first timestep the extrapolated potential
                                // is phi_0
        } else {
          pot_prev = Grav.F.potential_1_h[id_pot];  // Potential at the (n-1)-th
                                                    // timestep ( previous step )
          // Compute the extrapolated potential from phi_n-1 and phi_n
          pot_extrp = pot_now + 0.5 * Grav.dt_now * (pot_now - pot_prev) / Grav.dt_prev;
        }

  #ifdef COSMOLOGY
        // For cosmological simulation the potential is tranformed to 'comoving
        // coordinates'
        pot_extrp *= Cosmo.current_a * Cosmo.current_a / Cosmo.phi_0_gas;
  #endif

        // Save the extrapolated potential
        potential_out[id_grid] = pot_extrp;
        // Set phi_n-1 = phi_n, to use it during the next step
        Grav.F.potential_1_h[id_pot] = pot_now;
      }
    }
  }
}

// Call the function to extrapolate the potential
void Grid3D::Extrapolate_Grav_Potential()
{
  #ifdef GRAVITY_GPU
  Extrapolate_Grav_Potential_GPU();
  #else

    #ifndef PARALLEL_OMP
  Extrapolate_Grav_Potential_Function(0, Grav.nz_local + 2 * N_GHOST_POTENTIAL);
    #else

      #pragma omp parallel num_threads(N_OMP_THREADS)
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id      = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs(Grav.nz_local + 2 * N_GHOST_POTENTIAL, n_omp_procs, omp_id, &g_start, &g_end);

    Extrapolate_Grav_Potential_Function(g_start, g_end);
  }
    #endif  // PARALLEL_OMP

  #endif  // GRAVITY_GPU

  // After the first timestep the INITIAL flag is set to false, that way the
  // potential is properly extrapolated afterwards
  Grav.INITIAL = false;
}

#endif  // GRAVITY

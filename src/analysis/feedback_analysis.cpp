#include "feedback_analysis.h"

#include "../io/io.h"
#include "../model/disk_galaxy.h"

#ifdef MPI_CHOLLA
  #include "../mpi/mpi_routines.h"
#endif

#define VRMS_CUTOFF_DENSITY (0.01 * 0.6 * MP / DENSITY_UNIT)

FeedbackAnalysis::FeedbackAnalysis(Grid3D& G, struct Parameters* P)
{
  // set distance limits beyond which contributions to the V_rms don't
  // make sense, such as too close to the simulation volume edge or
  // too high above the disk.
  r_max = P->xlen / 2 * 0.975;
  z_max = 0.15;  // 150 pc above/below the disk plane
  // allocate arrays
  h_circ_vel_x = (Real*)malloc(G.H.n_cells * sizeof(Real));
  h_circ_vel_y = (Real*)malloc(G.H.n_cells * sizeof(Real));

#ifdef PARTICLES_GPU
  GPU_Error_Check(cudaMalloc((void**)&d_circ_vel_x, G.H.n_cells * sizeof(Real)));
  GPU_Error_Check(cudaMalloc((void**)&d_circ_vel_y, G.H.n_cells * sizeof(Real)));
#endif

  // setup the (constant) circular speed arrays
  int id;
  Real vca, r, x, y, z;

  for (int k = G.H.n_ghost; k < G.H.nz - G.H.n_ghost; k++) {
    for (int j = G.H.n_ghost; j < G.H.ny - G.H.n_ghost; j++) {
      for (int i = G.H.n_ghost; i < G.H.nx - G.H.n_ghost; i++) {
        id = i + j * G.H.nx + k * G.H.nx * G.H.ny;

        G.Get_Position(i, j, k, &x, &y, &z);
        r = sqrt(x * x + y * y);

        vca              = sqrt(r * fabs(galaxies::MW.gr_total_D3D(r, z)));
        h_circ_vel_x[id] = -y / r * vca;
        h_circ_vel_y[id] = x / r * vca;
      }
    }
  }

#ifdef PARTICLES_GPU
  GPU_Error_Check(cudaMemcpy(d_circ_vel_x, h_circ_vel_x, G.H.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
  GPU_Error_Check(cudaMemcpy(d_circ_vel_y, h_circ_vel_y, G.H.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
#endif
}

FeedbackAnalysis::~FeedbackAnalysis()
{
  free(h_circ_vel_x);
  free(h_circ_vel_y);
#ifdef PARTICLES_GPU
  GPU_Error_Check(cudaFree(d_circ_vel_x));
  GPU_Error_Check(cudaFree(d_circ_vel_y));
#endif
}

void FeedbackAnalysis::Compute_Gas_Velocity_Dispersion(Grid3D& G)
{
#ifdef CPU_TIME
  G.Timer.FeedbackAnalysis.Start();
#endif

#ifdef PARTICLES_CPU
  int i, j, k, id, idm, idp;
  int id_grav;
  Real x, y, z, r, xpm, xpp, ypm, ypp, zpm, zpp;
  Real Pm, Pp;
  Real dPdx, dPdy, dPdr;
  Real vx, vy, vz, vrms_poisson, vrms_analytic, vcp, vca, vcxp, vcyp, vcxa, vcya;
  Real total_mass, partial_mass = 0, total_var_analytic = 0, total_var_poisson = 0, partial_var_poisson = 0,
                   partial_var_analytic = 0;

  int n_ghost_grav = G.Particles.G.n_ghost_particles_grid;
  int ghost_diff   = n_ghost_grav - G.H.n_ghost;
  int nx_grav      = G.Particles.G.nx_local + 2 * n_ghost_grav;
  int ny_grav      = G.Particles.G.ny_local + 2 * n_ghost_grav;

  for (k = 0; k < G.H.nz_real; k++) {
    for (j = 0; j < G.H.ny_real; j++) {
      for (i = 0; i < G.H.nx_real; i++) {
        id = (i + G.H.n_ghost) + (j + G.H.n_ghost) * G.H.nx + (k + G.H.n_ghost) * G.H.nx * G.H.ny;
        partial_mass += G.C.density[id];
      }
    }
  }
  #ifdef MPI_CHOLLA
  MPI_Allreduce(&partial_mass, &total_mass, 1, MPI_CHREAL, MPI_SUM, world);
  #else
  total_mass = partial_mass;
  #endif

  for (k = G.H.n_ghost; k < G.H.nz - G.H.n_ghost; k++) {
    for (j = G.H.n_ghost; j < G.H.ny - G.H.n_ghost; j++) {
      for (i = G.H.n_ghost; i < G.H.nx - G.H.n_ghost; i++) {
        id      = i + j * G.H.nx + k * G.H.nx * G.H.ny;
        id_grav = (i + ghost_diff) + (j + ghost_diff) * nx_grav + (k + ghost_diff) * nx_grav * ny_grav;

        if (G.C.density[id] < VRMS_CUTOFF_DENSITY) continue;  // in cgs, this is 0.01 cm^{-3}

        G.Get_Position(i, j, k, &x, &y, &z);
        r = sqrt(x * x + y * y);

        vcp  = sqrt(r * fabs(G.Particles.G.gravity_x[id_grav] * x / r + G.Particles.G.gravity_y[id_grav] * y / r));
        vcxp = -y / r * vcp;
        vcyp = x / r * vcp;
        vx   = G.C.momentum_x[id] / G.C.density[id];
        vy   = G.C.momentum_y[id] / G.C.density[id];
        vz   = G.C.momentum_z[id] / G.C.density[id];

        partial_var_poisson += ((vx - vcxp) * (vx - vcxp) + (vy - vcyp) * (vy - vcyp) + vz * vz) * G.C.density[id];
        partial_var_analytic += ((vx - h_circ_vel_x[id]) * (vx - h_circ_vel_x[id]) +
                                 (vy - h_circ_vel_y[id]) * (vy - h_circ_vel_y[id]) + (vz * vz)) *
                                G.C.density[id];
      }
    }
  }
  partial_var_poisson /= total_mass;
  partial_var_analytic /= total_mass;

  #ifdef MPI_CHOLLA
  MPI_Reduce(&partial_var_poisson, &total_var_poisson, 1, MPI_CHREAL, MPI_SUM, root, world);
  MPI_Reduce(&partial_var_analytic, &total_var_analytic, 1, MPI_CHREAL, MPI_SUM, root, world);

  #else
  total_var_poisson  = partial_var_poisson;
  total_var_analytic = partial_var_analytic;
  #endif

  vrms_poisson  = sqrt(total_var_poisson) * VELOCITY_UNIT / 1e5;  // output in km/s
  vrms_analytic = sqrt(total_var_analytic) * VELOCITY_UNIT / 1e5;

  chprintf("feedback: time %f, dt=%f, vrms_p = %f km/s, vrms_a = %f km/s\n", G.H.t, G.H.dt, vrms_poisson,
           vrms_analytic);

#elif defined(PARTICLES_GPU)
  Compute_Gas_Velocity_Dispersion_GPU(G);
#endif  // PARTICLES_CPU

#ifdef CPU_TIME
  G.Timer.FeedbackAnalysis.End();
#endif
}

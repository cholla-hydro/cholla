
  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>

  #include <cstring>
  #include <fstream>
  #include <sstream>
  #include <vector>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../utils/DeviceVector.h"
  #include "../utils/error_handling.h"
  #include "../utils/reduction_utilities.h"
  #include "../feedback/ratecalc.h"
  #include "feedback.h"

  #define TPB_FEEDBACK 128

/** the prescription for dividing a scalar quantity between 3x3x3 cells is done
   by imagining a 2x2x2 cell volume around the SN.  These fractions, then,
   represent the linear extent of this volume into the cell in question. For i=0
   this should be 1*1/2. For i=-1 this should be (1-dx)*1/2. For i=+1 this
   should be dx*1/2. In the above the 1/2 factor is normalize over 2
   cells/direction.
  */
inline __device__ Real Frac(int i, Real dx) { return (-0.5 * i * i - 0.5 * i + 1 + i * dx) * 0.5; }

inline __device__ Real D_Frac(int i, Real dx)
{
  return (dx > 0.5) * i * (1 - 2 * dx) + ((i + 1) * dx + 0.5 * (i - 1)) - 3 * (i - 1) * (i + 1) * (0.5 - dx);
}

/** This function used for debugging potential race conditions.  Feedback from neighboring
    particles could simultaneously alter one hydro cell's conserved quantities.
 */
inline __device__ bool Particle_Is_Alone(Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev, part_int_t n_local,
                                         int gtid, Real dx)
{
  Real x0 = pos_x_dev[gtid];
  Real y0 = pos_y_dev[gtid];
  Real z0 = pos_z_dev[gtid];
  // Brute force loop to see if particle is alone
  for (int i = 0; i < n_local; i++) {
    if (i == gtid) continue;
    if (abs(x0 - pos_x_dev[i]) > dx) continue;
    if (abs(y0 - pos_y_dev[i]) > dx) continue;
    if (abs(z0 - pos_z_dev[i]) > dx) continue;
    // If we made it here, something is too close.
    return false;
  }
  return true;
}

inline __device__ Real Get_Average_Density(Real* density, int xi, int yi, int zi, int nx_grid, int ny_grid, int n_ghost)
{
  Real d_average = 0.0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        d_average +=
            density[(xi + n_ghost + i) + (yi + n_ghost + j) * nx_grid + (zi + n_ghost + k) * nx_grid * ny_grid];
      }
    }
  }
  return d_average / 27;
}

inline __device__ Real Get_Average_Number_Density_CGS(Real* density, int xi, int yi, int zi, int nx_grid, int ny_grid,
                                                      int n_ghost)
{
  return Get_Average_Density(density, xi, yi, zi, nx_grid, ny_grid, n_ghost) * DENSITY_UNIT / (MU * MP);
}

__device__ void Set_Average_Density(int indx_x, int indx_y, int indx_z, int nx_g, int ny_g, int n_ghost, Real* density,
                                    Real ave_dens)
{
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        int indx = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;

        density[indx] = ave_dens;
      }
    }
  }
}

__device__ void Apply_Resolved_SN(Real pos_x, Real pos_y, Real pos_z, Real xMin, Real yMin, Real zMin, Real dx, Real dy,
                                  Real dz, int nx_g, int ny_g, int n_ghost, int n_cells,
                                  Real* conserved_device, Real feedback_density, Real feedback_energy)
{
  // For 2x2x2, a particle between 0-0.5 injects onto cell - 1
  int indx_x = (int)floor((pos_x - xMin - 0.5 * dx) / dx);
  int indx_y = (int)floor((pos_y - yMin - 0.5 * dy) / dy);
  int indx_z = (int)floor((pos_z - zMin - 0.5 * dz) / dz);

  Real cell_center_x = xMin + indx_x * dx + 0.5 * dx;
  Real cell_center_y = yMin + indx_y * dy + 0.5 * dy;
  Real cell_center_z = zMin + indx_z * dz + 0.5 * dz;

  Real delta_x = 1 - (pos_x - cell_center_x) / dx;
  Real delta_y = 1 - (pos_y - cell_center_y) / dy;
  Real delta_z = 1 - (pos_z - cell_center_z) / dz;

  Real* density    = conserved_device;
  Real* energy     = &conserved_device[n_cells * grid_enum::Energy];
#ifdef DE
  Real* gasEnergy  = &conserved_device[n_cells * grid_enum::GasEnergy];
#endif

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        int indx    = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;
        Real x_frac = i * (1 - delta_x) + (1 - i) * delta_x;
        Real y_frac = j * (1 - delta_y) + (1 - j) * delta_y;
        Real z_frac = k * (1 - delta_z) + (1 - k) * delta_z;

        atomicAdd(&density[indx], x_frac * y_frac * z_frac * feedback_density);
#ifdef DE
        atomicAdd(&gasEnergy[indx], x_frac * y_frac * z_frac * feedback_energy);
#endif
        atomicAdd(&energy[indx], x_frac * y_frac * z_frac * feedback_energy);

      }  // k loop
    }    // j loop
  }      // i loop
}

/* \brief Function used for depositing energy or momentum from an unresolved
 * supernova or from a stellar wind
 *
 * \note
 * Previously there were 2 separate functions defined to perform this operation.
 * They were functionally the same. They only differences were the names of
 * variables.
 *
 * \par
 * There are currently issues with the internals of this function:
 * - this requires the codebase to be compiled with the dual energy formalism
 * - momentum and total energy are not updated self-consistently
 */
__device__ void Apply_Energy_Momentum_Deposition(Real pos_x, Real pos_y, Real pos_z, Real xMin, Real yMin, Real zMin,
                                                 Real dx, Real dy, Real dz, int nx_g, int ny_g, int n_ghost,
                                                 int n_cells, Real* conserved_device,
                                                 Real feedback_density, Real feedback_momentum, Real feedback_energy,
                                                 int indx_x, int indx_y, int indx_z)
{
  Real delta_x = (pos_x - xMin - indx_x * dx) / dx;
  Real delta_y = (pos_y - yMin - indx_y * dy) / dy;
  Real delta_z = (pos_z - zMin - indx_z * dz) / dz;

  Real* density    = conserved_device;
  Real* momentum_x = &conserved_device[n_cells * grid_enum::momentum_x];
  Real* momentum_y = &conserved_device[n_cells * grid_enum::momentum_y];
  Real* momentum_z = &conserved_device[n_cells * grid_enum::momentum_z];
  Real* energy     = &conserved_device[n_cells * grid_enum::Energy];
#ifdef DE
  Real* gas_energy = &conserved_device[n_cells * grid_enum::GasEnergy];
#endif

  // loop over the 27 cells to add up all the allocated feedback
  // momentum magnitudes.  For each cell allocate density and
  // energy based on the ratio of allocated momentum to this overall sum.
  Real mag = 0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        Real x_frac = D_Frac(i, delta_x) * Frac(j, delta_y) * Frac(k, delta_z);
        Real y_frac = Frac(i, delta_x) * D_Frac(j, delta_y) * Frac(k, delta_z);
        Real z_frac = Frac(i, delta_x) * Frac(j, delta_y) * D_Frac(k, delta_z);

        mag += sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac);
      }
    }
  }

  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        // index in array of conserved quantities
        int indx = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;

        Real x_frac = D_Frac(i, delta_x) * Frac(j, delta_y) * Frac(k, delta_z);
        Real y_frac = Frac(i, delta_x) * D_Frac(j, delta_y) * Frac(k, delta_z);
        Real z_frac = Frac(i, delta_x) * Frac(j, delta_y) * D_Frac(k, delta_z);

        Real px       = x_frac * feedback_momentum;
        Real py       = y_frac * feedback_momentum;
        Real pz       = z_frac * feedback_momentum;
        Real f_dens   = sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac) / mag * feedback_density;
        Real f_energy = sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac) / mag * feedback_energy;

        atomicAdd(&density[indx], f_dens);
        atomicAdd(&momentum_x[indx], px);
        atomicAdd(&momentum_y[indx], py);
        atomicAdd(&momentum_z[indx], pz);
        atomicAdd(&energy[indx], f_energy);

#ifdef DE
        gas_energy[indx] = energy[indx] - (momentum_x[indx] * momentum_x[indx] + momentum_y[indx] * momentum_y[indx] +
                                           momentum_z[indx] * momentum_z[indx]) /
                                              (2 * density[indx]);
#endif
        /*
        energy[indx] = ( momentum_x[indx] * momentum_x[indx] +
                         momentum_y[indx] * momentum_y[indx] +
                         momentum_z[indx] * momentum_z[indx] ) /
                       2 / density[indx] + gasEnergy[indx];
        */
      }  // k loop
    }    // j loop
  }      // i loop
}

__device__ void SN_Feedback(Real pos_x, Real pos_y, Real pos_z, Real age, Real* mass_dev, part_int_t* id_dev, Real xMin,
                            Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g,
                            int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt,
                            int num_SN, Real* s_info, Real* conserved_dev, Real gamma, int indx_x, int indx_y, int indx_z)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  Real dV = dx * dy * dz;
  int n_cells    = nx_g * ny_g * nz_g;

  /*
  if (gtid == 0) {
    kernel_printf("SNUMBER n_step: %d, id: %lld, N: %d\n", n_step, id_dev[gtid], N);
  }
  */

  // no sense doing anything if there was no SN
  if (num_SN == 0) return;

  Real* density             = conserved_dev;
  Real n_0                  = Get_Average_Number_Density_CGS(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
  s_info[feedinfoLUT::LEN * tid + feedinfoLUT::countSN] += num_SN;

  Real feedback_energy  = num_SN * feedback::ENERGY_PER_SN / dV;
  Real feedback_density = num_SN * feedback::MASS_PER_SN / dV;

  Real shell_radius = feedback::R_SH * pow(n_0, -0.46) * pow(fabsf(num_SN), 0.29);
  #ifdef ONLY_RESOLVED
  bool is_resolved = true;
  #else
  bool is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius;
  #endif

  if (is_resolved) {
    // inject energy and density
    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::countResolved] += num_SN;
    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalEnergy]   += feedback_energy * dV;
    Apply_Resolved_SN(pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost, n_cells,
                      conserved_dev, feedback_density, feedback_energy);
  } else {
    // currently, only unresolved SN feedback involves averaging the densities.
    Real ave_dens = n_0 * MU * MP / DENSITY_UNIT;
    Set_Average_Density(indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost, density, ave_dens);

    // inject momentum and density
    Real feedback_momentum = feedback::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(fabsf(num_SN), 0.93) / dV / sqrt(3.0);
    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::countUnresolved]  += num_SN;
    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalMomentum]    += feedback_momentum * dV * sqrt(3.0);
    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalUnresEnergy] += feedback_energy * dV;
    Apply_Energy_Momentum_Deposition(
        pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost, n_cells, conserved_dev,
        feedback_density, feedback_momentum, feedback_energy, indx_x, indx_y, indx_z);
  }

  // update the cluster mass
  mass_dev[gtid] -= num_SN * feedback::MASS_PER_SN;
}

__device__ void Wind_Feedback(Real pos_x, Real pos_y, Real pos_z, Real age, Real* mass_dev, part_int_t* id_dev,
                              Real xMin, Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy,
                              Real dz, int nx_g, int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt,
                              const feedback::SWRateCalc sw_calc, Real* s_info,
                              Real* conserved_dev, Real gamma, int indx_x, int indx_y, int indx_z)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  Real dV = dx * dy * dz;
  int n_cells    = nx_g * ny_g * nz_g;

  if ((age < 0) or not sw_calc.is_active(age)) return;
  Real feedback_momentum = sw_calc.Get_Wind_Flux(age);
  // no sense in proceeding if there is no feedback.
  if (feedback_momentum == 0) return;
  Real feedback_energy  = sw_calc.Get_Wind_Power(age);
  Real feedback_density = sw_calc.Get_Wind_Mass(feedback_momentum, feedback_energy);

  // feedback_momentum now becomes momentum component along one direction.
  feedback_momentum *= mass_dev[gtid] * dt / dV / sqrt(3.0);
  feedback_density *= mass_dev[gtid] * dt / dV;
  feedback_energy *= mass_dev[gtid] * dt / dV;

  mass_dev[gtid]   -= feedback_density * dV;

  // we log net momentum, not momentum density, and magnitude (not the
  // component along a direction)
  s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalWindMomentum] += feedback_momentum * dV * sqrt(3.0);
  s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalWindEnergy]   += feedback_energy * dV;

  Apply_Energy_Momentum_Deposition(pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost,
                                   n_cells, conserved_dev, feedback_density,
                                   feedback_momentum, feedback_energy, indx_x, indx_y, indx_z);
}

__global__ void Cluster_Feedback_Kernel(part_int_t n_local, part_int_t* id_dev, Real* pos_x_dev, Real* pos_y_dev,
                                        Real* pos_z_dev, Real* mass_dev, Real* age_dev, Real xMin, Real yMin, Real zMin,
                                        Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g, int ny_g,
                                        int nz_g, int n_ghost, Real t, Real dt, Real* info, Real* conserved_dev,
                                        Real gamma, int* num_SN_dev, int n_step)
{
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * blockDim.x + tid;

  // prologoue: setup buffer for collecting SN feedback information
  __shared__ Real s_info[feedinfoLUT::LEN * TPB_FEEDBACK];
  for (unsigned int cur_ind = 0; cur_ind < feedinfoLUT::LEN; cur_ind++) {
    s_info[feedinfoLUT::LEN * tid + cur_ind] = 0;
  }

  // do the main work:
  if (gtid < n_local) { // Bounds check on particle arrays

    Real pos_x    = pos_x_dev[gtid];
    Real pos_y    = pos_y_dev[gtid];
    Real pos_z    = pos_z_dev[gtid];
    bool in_local = (pos_x >= xMin && pos_x < xMax) && (pos_y >= yMin && pos_y < yMax) && (pos_z >= zMin && pos_z < zMax);

    int indx_x  = (int)floor((pos_x - xMin) / dx);
    int indx_y  = (int)floor((pos_y - yMin) / dy);
    int indx_z  = (int)floor((pos_z - zMin) / dz);
    bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x >= nx_g - 2 * n_ghost ||
                  indx_y >= ny_g - 2 * n_ghost || indx_z >= nz_g - 2 * n_ghost;

    // ignore should always be not in_local, by definition

    if (in_local and (not ignore)) {
      // note age_dev is actually the time of birth
      Real age = t - age_dev[gtid];

      SN_Feedback(pos_x, pos_y, pos_z, age, mass_dev, id_dev, xMin, yMin, zMin, xMax, yMax, zMax, dx, dy, dz, nx_g,
                  ny_g, nz_g, n_ghost, n_step, t, dt, num_SN_dev[gtid],
                  s_info, conserved_dev, gamma, indx_x, indx_y, indx_z);
    }
  }


  // epilogue: sum the info from all threads (in all blocks) and add it into info
  __syncthreads();
  reduction_utilities::blockAccumulateIntoNReals<feedinfoLUT::LEN,TPB_FEEDBACK>(info, s_info);
}

/* determine the number of supernovae during the current step */
__global__ void Get_SN_Count_Kernel(part_int_t n_local, part_int_t* id_dev, Real* mass_dev,
                                    Real* age_dev, Real t, Real dt,
                                    const feedback::SNRateCalc snr_calc, int n_step, int* num_SN_dev)
{
  int tid = threadIdx.x;

  int gtid = blockIdx.x * blockDim.x + tid;
  // Bounds check on particle arrays
  if (gtid >= n_local) return;

  // note age_dev is actually the time of birth
  Real age = t - age_dev[gtid];

  Real average_num_sn = snr_calc.Get_SN_Rate(age) * mass_dev[gtid] * dt;
  num_SN_dev[gtid]    = snr_calc.Get_Number_Of_SNe_In_Cluster(average_num_sn, n_step, id_dev[gtid]);
}

namespace { // anonymous namespace

/* This functor is the callback used in the main part of cholla
 */
struct ClusterFeedbackMethod {

  ClusterFeedbackMethod(FeedbackAnalysis& analysis, bool use_snr_calc, feedback::SNRateCalc snr_calc)
    : analysis(analysis), use_snr_calc_(use_snr_calc), snr_calc_(snr_calc)
{ }

  /* Actually apply the stellar feedback (SNe and stellar winds) */
  void operator() (Grid3D& G);

private: // attributes

  FeedbackAnalysis& analysis;
  /* When false, ignore the snr_calc_ attribute. Instead, assume all clusters undergo a single
   * supernova during the very first cycle and then never have a supernova again. */
  const bool use_snr_calc_;
  feedback::SNRateCalc snr_calc_;
};

} // close anonymous namespace

/**
 * @brief Stellar feedback function (SNe and stellar winds)
 *
 * @param G
 */
void ClusterFeedbackMethod::operator()(Grid3D& G)
{
#if !(defined(PARTICLES_GPU) && defined(PARTICLE_AGE) && defined(PARTICLE_IDS))
  CHOLLA_ERROR("This function can't be called with the current compiler flags");
#else
  #ifdef CPU_TIME
  G.Timer.Feedback.Start();
  #endif

  if (G.H.dt == 0) return;

  // h_info is used to store feedback summary info on the host. The following
  // syntax sets all entries to 0 -- important if a process has no particles
  // (this is valid C++ syntax, but historically wasn't valid C syntax)
  Real h_info[feedinfoLUT::LEN] = {};

  // only apply feedback if we have clusters
  if (G.Particles.n_local > 0) {
    // compute the grid-size or the number of thread-blocks per grid. The number of threads in a block is
    // given by TPB_FEEDBACK
    int ngrid = (G.Particles.n_local - 1) / TPB_FEEDBACK + 1;

    // Declare/allocate device buffer for holding the number of supernovae per particle in the current cycle
    // (The following behavior can be accomplished without any memory allocations if we employ templates)
    cuda_utilities::DeviceVector<int> d_num_SN(G.Particles.n_local, true);  // initialized to 0

    if (use_snr_calc_) {
      hipLaunchKernelGGL(Get_SN_Count_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local,
                         G.Particles.partIDs_dev, G.Particles.mass_dev, G.Particles.age_dev, G.H.t, G.H.dt,
                         snr_calc_, G.H.n_step, d_num_SN.data());
      CHECK(cudaDeviceSynchronize());
    } else {
      // in this branch, ``this->use_snr_calc_ == false``. This means that we assume all particles undergo
      // a supernova during the very first cycle. Then there is never another supernova
      if (G.H.n_step == 0) {
        std::vector<int> tmp(G.Particles.n_local, 1);
        CHECK(cudaMemcpy(d_num_SN.data(), tmp.data(), sizeof(int)*G.Particles.n_local, cudaMemcpyHostToDevice));
      } else {
        // do nothing - the number of supernovae is already zero
      }
    }

    // Declare/allocate device buffer for accumulating summary information about feedback
    cuda_utilities::DeviceVector<Real> d_info(feedinfoLUT::LEN, true);  // initialized to 0

    hipLaunchKernelGGL(Cluster_Feedback_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local,
                       G.Particles.partIDs_dev, G.Particles.pos_x_dev, G.Particles.pos_y_dev, G.Particles.pos_z_dev,
                       G.Particles.mass_dev, G.Particles.age_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal,
                       G.H.xblocal_max, G.H.yblocal_max, G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny,
                       G.H.nz, G.H.n_ghost, G.H.t, G.H.dt, d_info.data(), G.C.d_density, gama, 
                       d_num_SN.data(), G.H.n_step);

    // copy summary data back to the host
    CHECK(cudaMemcpy(&h_info, d_info.data(), feedinfoLUT::LEN * sizeof(Real), cudaMemcpyDeviceToHost));
  }

  // now gather the feedback summary info into an array called info.
  #ifdef MPI_CHOLLA
  Real info[feedinfoLUT::LEN];
  MPI_Reduce(&h_info, &info, feedinfoLUT::LEN, MPI_CHREAL, MPI_SUM, root, world);
  #else
  Real* info = h_info;
  #endif

  #ifdef MPI_CHOLLA  // only do stats gathering on root rank
  if (procID == 0) {
  #endif

    analysis.countSN += (long)info[feedinfoLUT::countSN];
    analysis.countResolved += (long)info[feedinfoLUT::countResolved];
    analysis.countUnresolved += (long)info[feedinfoLUT::countUnresolved];
    analysis.totalEnergy += info[feedinfoLUT::totalEnergy];
    analysis.totalMomentum += info[feedinfoLUT::totalMomentum];
    analysis.totalUnresEnergy += info[feedinfoLUT::totalUnresEnergy];
    analysis.totalWindMomentum += info[feedinfoLUT::totalWindMomentum];
    analysis.totalWindEnergy += info[feedinfoLUT::totalWindEnergy];

    chprintf("iteration %d, t %.4e, dt %.4e", G.H.n_step, G.H.t, G.H.dt);

  #ifndef NO_SN_FEEDBACK
    Real global_resolved_ratio = 0.0;
    if (analysis.countResolved > 0 || analysis.countUnresolved > 0) {
      global_resolved_ratio = analysis.countResolved / (analysis.countResolved + analysis.countUnresolved);
    }
    chprintf(": number of SN: %d,(R: %d, UR: %d)\n", (int)info[feedinfoLUT::countSN], (long)info[feedinfoLUT::countResolved],
             (long)info[feedinfoLUT::countUnresolved]);
    chprintf("    cummulative: #SN: %d, ratio of resolved (R: %d, UR: %d) = %.3e\n", (long)analysis.countSN,
             (long)analysis.countResolved, (long)analysis.countUnresolved, global_resolved_ratio);
    chprintf("    sn  r energy  : %.5e erg, cumulative: %.5e erg\n", info[feedinfoLUT::totalEnergy] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalEnergy * FORCE_UNIT * LENGTH_UNIT);
    chprintf("    sn ur energy  : %.5e erg, cumulative: %.5e erg\n",
             info[feedinfoLUT::totalUnresEnergy] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalUnresEnergy * FORCE_UNIT * LENGTH_UNIT);
    chprintf("    sn momentum  : %.5e SM km/s, cumulative: %.5e SM km/s\n",
             info[feedinfoLUT::totalMomentum] * VELOCITY_UNIT / 1e5, analysis.totalMomentum * VELOCITY_UNIT / 1e5);
  #endif  // NO_SN_FEEDBACK

  #ifndef NO_WIND_FEEDBACK
    chprintf("    wind momentum: %.5e S.M. km/s,  cumulative: %.5e S.M. km/s\n",
             info[feedinfoLUT::totalWindMomentum] * VELOCITY_UNIT / 1e5, analysis.totalWindMomentum * VELOCITY_UNIT / 1e5);
    chprintf("    wind energy  : %.5e erg,  cumulative: %.5e erg\n", info[feedinfoLUT::totalWindEnergy] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalWindEnergy * FORCE_UNIT * LENGTH_UNIT);
  #endif  // NO_WIND_FEEDBACK

  #ifdef MPI_CHOLLA
  }  //   end if procID == 0
  #endif

  #ifdef CPU_TIME
  G.Timer.Feedback.End();
  #endif
#endif // the ifdef statement for Particle-stuff
}

std::function<void(Grid3D&)> feedback::configure_feedback_callback(struct parameters& P,
                                                                   FeedbackAnalysis& analysis)
{
  const std::string sn_model = P.feedback_sn_model;

  // check whether or not the user wants some kind of feedback
  if (sn_model == "none") {
    return {}; // intentionally returning an empty object

#if !(defined(FEEDBACK) && defined(PARTICLES_GPU) && defined(PARTICLE_AGE) && defined(PARTICLE_IDS))
  } else if (sn_model.empty()) {
    return {};
  } else {
    CHOLLA_ERROR("The way that cholla was compiled does not currently support feedback");
#endif
  }

  // at the moment, we just ignore feedback_sn_model (other than "none")
  // but possible values in the future: "none", "legacy-resolved", "legacy-combo", ...
  //
  // other parameters to accept in the future:
  // - feedback_sn_overlap_strat: "ignore", "choose_lower_id"
  // - feedback_sn_boundary_strat: "ignore", "snap"

  std::string sn_rate_model = P.feedback_sn_rate;
  if (sn_rate_model.empty()){
    sn_rate_model = "table";
  }

  std::function<void(Grid3D&)> out;
  if (sn_rate_model == "immediate_sn") {
    out = ClusterFeedbackMethod(analysis, false, feedback::SNRateCalc());
  } else if (sn_rate_model == "table") {
    out = ClusterFeedbackMethod(analysis, true, feedback::SNRateCalc(P));
  } else {
    CHOLLA_ERROR("Unrecognized option passed to sn_rate_model: %s", sn_rate_model.c_str());
  }

  return out;
}



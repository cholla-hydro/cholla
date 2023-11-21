
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
  #include "../feedback/feedback_model.h"
  #include "feedback.h"

  #define TPB_FEEDBACK 128



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


template<typename FeedbackModel>
__global__ void Cluster_Feedback_Kernel(part_int_t n_local, part_int_t* id_dev, Real* pos_x_dev, Real* pos_y_dev,
                                        Real* pos_z_dev, Real* mass_dev, Real* age_dev, Real xMin, Real yMin, Real zMin,
                                        Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g, int ny_g,
                                        int nz_g, int n_ghost, Real t, Real dt, Real* info, Real* conserved_dev,
                                        Real gamma, int* num_SN_dev, int n_step, FeedbackModel feedback_model)
{
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * blockDim.x + tid;

  // prologoue: setup buffer for collecting SN feedback information
  __shared__ Real s_info[feedinfoLUT::LEN * TPB_FEEDBACK];
  for (unsigned int cur_ind = 0; cur_ind < feedinfoLUT::LEN; cur_ind++) {
    s_info[feedinfoLUT::LEN * tid + cur_ind] = 0;
  }

  // do the main work:
  {
    // reduce branching
    part_int_t tmp_gtid_ = min(n_local - 1, part_int_t(gtid));

    Real pos_x    = pos_x_dev[tmp_gtid_];
    Real pos_y    = pos_y_dev[tmp_gtid_];
    Real pos_z    = pos_z_dev[tmp_gtid_];

    // compute the position in index-units (appropriate for a field with a ghost-zone)
    // - an integer value corresponds to the left edge of a cell
    const Real pos_x_indU = (pos_x - xMin) / dx + n_ghost;
    const Real pos_y_indU = (pos_y - yMin) / dy + n_ghost;
    const Real pos_z_indU = (pos_z - zMin) / dz + n_ghost;

    bool ignore = (((pos_x_indU < n_ghost) or (pos_x_indU >= (nx_g - n_ghost))) or
                   ((pos_y_indU < n_ghost) or (pos_y_indU >= (ny_g - n_ghost))) or
                   ((pos_z_indU < n_ghost) or (pos_z_indU >= (ny_g - n_ghost))));

    if ((not ignore) and (n_local > gtid)) {
      // note age_dev is actually the time of birth
      Real age = t - age_dev[gtid];

      feedback_model.apply_feedback(pos_x_indU, pos_y_indU, pos_z_indU, age, mass_dev, id_dev, dx, dy, dz,
                                    nx_g, ny_g, nz_g, n_ghost, num_SN_dev[gtid], s_info, conserved_dev);
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
template<typename FeedbackModel>
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
template<typename FeedbackModel>
void ClusterFeedbackMethod<FeedbackModel>::operator()(Grid3D& G)
{
#if !(defined(PARTICLES_GPU) && defined(PARTICLE_AGE) && defined(PARTICLE_IDS))
  CHOLLA_ERROR("This function can't be called with the current compiler flags");
#else
  #ifdef CPU_TIME
  G.Timer.Feedback.Start();
  #endif

  if (max(fabs(G.H.dy - G.H.dx), fabs(G.H.dz - G.H.dx))  > fabs(1e-15 * G.H.dx)) {
    CHOLLA_ERROR("dx, dy, dz must all approximately be the same with the current feedback prescriptions");
  }

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

    // initialize feedback_model
    FeedbackModel feedback_model{};

    hipLaunchKernelGGL(Cluster_Feedback_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local,
                       G.Particles.partIDs_dev, G.Particles.pos_x_dev, G.Particles.pos_y_dev, G.Particles.pos_z_dev,
                       G.Particles.mass_dev, G.Particles.age_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal,
                       G.H.xblocal_max, G.H.yblocal_max, G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny,
                       G.H.nz, G.H.n_ghost, G.H.t, G.H.dt, d_info.data(), G.C.d_density, gama, 
                       d_num_SN.data(), G.H.n_step, feedback_model);

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
#if !(defined(FEEDBACK) && defined(PARTICLES_GPU) && defined(PARTICLE_AGE) && defined(PARTICLE_IDS))
  const bool supports_feedback = false;
#else
  const bool supports_feedback = true;
#endif

  // retrieve the supernova-feedback model name
  std::string sn_model = P.feedback_sn_model;
  if (sn_model.empty() and (not supports_feedback)) {
    sn_model = "none";
  } else if (sn_model.empty()) {
#ifdef ONLY_RESOLVED
    sn_model = "resolvedCiC";
#else
    sn_model = "legacy";
#endif
    chprintf("the feedback_sn_model was not supplied. Right now, we are defaulting to \"%s\" (based "
             "on compiler flags) - in the future we will abort with an error instead",
             sn_model.c_str());
  }


  // handle the case when there is no feedback (or if the code can't support feedback)
  if (sn_model == "none") {  // return an empty objec
    return {};
  } else if (not supports_feedback) {
    CHOLLA_ERROR("The way that cholla was compiled does not currently support feedback");
  }


  // parse the supernova-rate-model to initialize some values
  SNRateCalc snr_calc{};
  bool use_snr_calc;

  const std::string sn_rate_model = P.feedback_sn_rate;
  if (sn_rate_model.empty() or (sn_rate_model == "table")) {
    use_snr_calc = true;
    snr_calc = feedback::SNRateCalc(P);
  } else if (sn_rate_model == "immediate_sn") {
    use_snr_calc = false;
  } else {
    CHOLLA_ERROR("Unrecognized option passed to sn_rate_model: %s", sn_rate_model.c_str());
  }

  // now lets initialize ClusterFeedbackMethod<> and return
  std::function<void(Grid3D&)> out;
  if (sn_model == "legacy") {
    out = ClusterFeedbackMethod<feedback_model::LegacySNe<feedback_model::CiCResolvedSNPrescription>>(analysis, use_snr_calc, snr_calc);
  } else if (sn_model == "resolvedCiC") {
    out = ClusterFeedbackMethod<feedback_model::CiCResolvedSNPrescription>(analysis, use_snr_calc, snr_calc);
  } else if (sn_model == "resolved27cell") {
    out = ClusterFeedbackMethod<feedback_model::Sphere27ResolvedSNPrescription>(analysis, use_snr_calc, snr_calc);
  } else if (sn_model == "resolvedExperimentalBinarySphere"){
    out = ClusterFeedbackMethod<feedback_model::SphereBinaryResolvedSNPrescription>(analysis, use_snr_calc, snr_calc);
  } else {
    CHOLLA_ERROR("Unrecognized sn_model: %s", sn_model.c_str());
  }
  return out;
}



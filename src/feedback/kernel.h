#pragma once
/* Define the main kernel that does most of the heavy lifting.
 *
 * This is primarily defined in a header so that we can directly test it.
 */

#include "../global/global.h"
#include "../feedback/feedback.h"
#include "../utils/reduction_utilities.h"
#include "../utils/DeviceVector.h"

#define TPB_FEEDBACK 128

// The following is only here to simplify testing. In the future it may make sense to move it to a different header
namespace feedback_details{

/* Applies cluster feedback.
 *
 * \tparam FeedbackModel type that encapsulates the actual feedback prescription
 *
 * \param[in,out] particle_props Encodes the actual particle data needed for feedback. If there is any feedback, the
 *     relevant particle properties (like particle mass) will be updated during this call.
 * \param[in]     spatial_props Encodes spatial information about the local domain and the fields
 * \param[in]     cycle_props Encodes details about the simulation's current (global) iteration cycle
 * \param[out]    info An array that will is intended to accumulate summary details about the feedback during the course of
 *     this kernel call. This function assumes it has feedinfoLUT::LEN entries that are all initialized to 0.
 * \param[out]    conserved_dev pointer to the fluid-fields that will be updated during this function call.
 * \param[in]     An array of ``particle_props.n_local`` non-negative integers that specify the number of supernovae that are
 *     are scheduled to occur during the current cycle (for each particle).
 * \param[in]     feedback_model represents the feedback prescription. At the time of writing, this is largely a dummy argument
 *     used to help with the inference of the FeedbackModel type.
 */
template<typename FeedbackModel>
__global__ void Cluster_Feedback_Kernel(const feedback_details::ParticleProps particle_props,
                                        const feedback_details::FieldSpatialProps spatial_props,
                                        const feedback_details::CycleProps cycle_props, Real* info, Real* conserved_dev,
                                        int* num_SN_dev, FeedbackModel feedback_model)
{
  const int tid = threadIdx.x;

  // prologoue: setup buffer for collecting SN feedback information
  __shared__ Real s_info[feedinfoLUT::LEN * TPB_FEEDBACK];
  for (unsigned int cur_ind = 0; cur_ind < feedinfoLUT::LEN; cur_ind++) {
    s_info[feedinfoLUT::LEN * tid + cur_ind] = 0;
  }

  // do the main work. All threads across the grid will iterate over the list of particles
  // - this is grid-strided loop. This is a common idiom that makes the kernel more flexible
  // - If there are more local particles than threads, some threads will visit more than 1 particle
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int loop_stride = blockDim.x * gridDim.x;
  for (int i = start; i < particle_props.n_local; i += loop_stride) {

    const int n_ghost = spatial_props.n_ghost;

    // compute the position in index-units (appropriate for a field with a ghost-zone)
    // - an integer value corresponds to the left edge of a cell
    const Real pos_x_indU = (particle_props.pos_x_dev[i] - spatial_props.xMin) / spatial_props.dx + n_ghost;
    const Real pos_y_indU = (particle_props.pos_y_dev[i] - spatial_props.yMin) / spatial_props.dy + n_ghost;
    const Real pos_z_indU = (particle_props.pos_z_dev[i] - spatial_props.zMin) / spatial_props.dz + n_ghost;

    bool ignore = (((pos_x_indU < n_ghost) or (pos_x_indU >= (spatial_props.nx_g - n_ghost))) or
                   ((pos_y_indU < n_ghost) or (pos_y_indU >= (spatial_props.ny_g - n_ghost))) or
                   ((pos_z_indU < n_ghost) or (pos_z_indU >= (spatial_props.nz_g - n_ghost))));

    if (not ignore) {
      // note age_dev is actually the time of birth
      const Real age = cycle_props.t - particle_props.age_dev[i];

      // holds a reference to the particle's mass (this will be updated after feedback is handled)
      Real& mass_ref = particle_props.mass_dev[i];

      feedback_model.apply_feedback(pos_x_indU, pos_y_indU, pos_z_indU, particle_props.vel_x_dev[i],
                                    particle_props.vel_y_dev[i], particle_props.vel_z_dev[i],
                                    age, mass_ref, particle_props.id_dev[i],
                                    spatial_props.dx, spatial_props.dy, spatial_props.dz, 
                                    spatial_props.nx_g, spatial_props.ny_g, spatial_props.nz_g, n_ghost,
                                    num_SN_dev[i], s_info, conserved_dev);
    }
  }

  // epilogue: sum the info from all threads (in all blocks) and add it into info
  __syncthreads(); // synchronize all threads in the current block. It's important to do this before
                   // the next function call because we accumulate values on the local block first
  reduction_utilities::blockAccumulateIntoNReals<feedinfoLUT::LEN,TPB_FEEDBACK>(info, s_info);
}


/* Launches the Kernel for ClusterFeedback
 *
 * \tparam FeedbackModel type that encapsulates the actual feedback prescription
 *
 * \param[in,out] particle_props Encodes the actual particle data needed for feedback. If there is any feedback, the
 *     relevant particle properties (like particle mass) will be updated during this call.
 * \param[in]     spatial_props Encodes spatial information about the local domain and the fields
 * \param[in]     cycle_props Encodes details about the simulation's current (global) iteration cycle
 * \param[out]    info An array on the host that will is intended to accumulate summary details about the feedback during the course of
 *     this kernel call. This function assumes it has feedinfoLUT::LEN entries that are all initialized to 0.
 * \param[out]    conserved_dev pointer to the fluid-fields that will be updated during this function call.
 * \param[in]     An array of ``particle_props.n_local`` non-negative integers that specify the number of supernovae that are
 *     are scheduled to occur during the current cycle (for each particle).
 * \param[in]     feedback_model represents the feedback prescription. At the time of writing, this is largely a dummy argument
 *     used to help with the inference of the FeedbackModel type.
 */
template<typename FeedbackModel>
void Exec_Cluster_Feedback_Kernel(const feedback_details::ParticleProps& particle_props,
                                  const feedback_details::FieldSpatialProps& spatial_props,
                                  const feedback_details::CycleProps& cycle_props, 
                                  Real* info, Real* conserved_dev, int* num_SN_dev)
{
  // compute the grid-size or the number of thread-blocks per grid. The number of threads in a block is
  // given by TPB_FEEDBACK
  const int blocks_per_grid = (particle_props.n_local - 1) / TPB_FEEDBACK + 1;

  // Declare/allocate device buffer for accumulating summary information about feedback
  cuda_utilities::DeviceVector<Real> d_info(feedinfoLUT::LEN, true);  // initialized to 0

  // initialize feedback_model
  FeedbackModel feedback_model{};

  hipLaunchKernelGGL(feedback_details::Cluster_Feedback_Kernel, blocks_per_grid, TPB_FEEDBACK, 0, 0,
                     particle_props, spatial_props, cycle_props, d_info.data(), conserved_dev, num_SN_dev, feedback_model);

  if (info != nullptr) {
    // copy summary data back to the host
    CHECK(cudaMemcpy(info, d_info.data(), feedinfoLUT::LEN * sizeof(Real), cudaMemcpyDeviceToHost));
  } else {
    CHECK(cudaDeviceSynchronize());
  }
}

} // namespace feedback_details
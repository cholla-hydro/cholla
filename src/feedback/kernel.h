#pragma once
/* Define the main kernel that does most of the heavy lifting.
 *
 * This is primarily defined in a header so that we can directly test it.
 */

#include "../global/global.h"
#include "../feedback/feedback.h"
#include "../feedback/feedback_stencil.h"  // Arr3
#include "../utils/reduction_utilities.h"
#include "../utils/DeviceVector.h"

#include <cooperative_groups.h>

#define TPB_FEEDBACK 128

// STENCIL OVERLAPS
// ----------------
// We refer to the pattern of cells around a particle that are affected by a given feedback perscription
// as a stencil
//
// Because we use separate GPU threads to simultaneously process the impact of feedback from separate
// particles on the local fluid fields, care must be taken when nearby particles have overlapping
// stencils.
//
// Our chosen approach
// -------------------
// When there is overlap, we essentially handle feedback one-at-a-time where the order is based on some
// deterministic particle property (nominally the particle id).
//
// Essentially, we first pre-register the stencils of all particles undergoing feedback in a "mask".
// Then we make a pass through all of the particles with pending feedback. 
// - For each particle with pending feedback, we use the previously-constructed "mask" to check if
//   its stencil overlaps with a stencil of a different particle (also with pending feedback) that 
//   has a larger particle id.
//   - if it doesn't, we perform feedback now!
//   - otherwise, we defer feedback until the next "pass" (and register its stencil in the "mask" used 
//     for the next pass)
// We make additional passes until we have applied all feedback.
//
// In the best case scenario (no overlap), we just do one pass. In the pathological worst case (all 
// particles are directly on top of each other), the number of passes is equal to the number of 
// particles with pending feedback.
//
// Alternative Options
// -------------------
// For a small subset of prescriptions, it's possible to handle overlapping regions by using atomic
// operations to update the fluid fields. This only works if you know how much each quantity will change
// ahead of time (i.e. the changes are independent of the current values). However, because we primarily
// parameterize the fluid's properties in terms of its mass density, momenta densities, and total energy 
// densities, the prescriptions that can be handled in this way are VERY limited. This approach is only
// applicable for prescriptions
// - that only inject a fixed amount of thermal energy density
// - OR are fairly pathological. Here are 2 simple examples:
//   1. if the prescription injects momentum (while not affecting mass), then assumptions need to be made 
//      that resulting changes to the kinetic energy are exactly balanced by changes to the thermal energy
//      (this is because you can't know how the kinetic energy density will change unless you you know the
//      current momentum and current density).
//   2. if the prescription injects mass (while not affecting momentum -- effectively reducing the bulk
//      velocity), makes a similar assumption that any changes to the kinetic energy density are exactly
//      balanced by changes to the thermal energy
// NOTE: If we adopted a strategy where we convert the total energy to the thermal energy, then apply all 
// feedback, and then at the end recompute the total energy afterwards this opens more options. It's worth
// considering in the future (although it is less flexible than our chosen approach).


// The following definitions are only in a header file to simplify testing.
namespace feedback_details{

inline __host__ void Ptr_Swap(part_int_t*& a, part_int_t*& b)
{
  part_int_t* tmp = a;
  a = b;
  b = tmp;
}

/* check if the current thread is the root-thread across all threadblocks
 *
 * \note
 * is this just equivalent to `(cooperative_groups::grid_group::thread_rank() == 0;)`?
 */
inline __host__ bool Is_GridWide_Root_Thread()
{ 
  return ( (blockIdx.x == 0) and (blockIdx.y == 0) and (blockIdx.z == 0) and
           (threadIdx.x == 0) and (threadIdx.y == 0) and (threadIdx.z == 0) );
}

/* Specifies the stategy for handling star-particles with overlapping stencils */
enum struct OverlapStrat {
  ignore, /*<! simply ignore that there are overlaps. Schedule everything at once (useful for profiling) */
  sequential /*<! Process feedback for all overlapping stencils (in order of the increasing particle ids) */
};

/* Class that implements most of the logic (and tracks associated data) for scheduling feedback from particles.
 * It's essentially a state-machine.
 *
 * As a shorthand, lets refer to the collection of particles who need to have their feedback during a given
 * simulation cycle, the "designated-feedback-particles"
 *
 * When configured with OverlapStrat::sequential (the primary use-case of this class), "designated-feedback-particles"
 * with overlapping stencils will be scheduled sequentially. The intention is for an instance of this class, lets 
 * call it `ov_scheduler`, to be used in the following control flow:
 *
 *  - at the start of the relevant kernel, call `ov_scheduler.Reset_State`
 *  - then iterate over all of the "designated-feedback-particles" (all particles that need to have their feedback 
 *    applied in the current kernel call). For each of these particles, call 
 *    `ov_scheduler.Register_Pending_Particle_Feedback`
 *  - now enter a while-loop where `ov_scheduler.Prepare_Next_Pass` is evaluated as the condition-expression
 *     - we refer to each evaluation of the loop body as a "pass".
 *     - The loop-body should consist of iterating over all of the "designated-feedback-particles" (in other words, each
 *       evaluation of the loop body corresponds to a "pass" through the "designated-feedback-particles"). For each of
 *       these particles, evaluate `ov_scheduler.Is_Scheduled_And_Update`
 *     - that method will return whether that is scheduled to have its feedback applied right now. It will also update
 *       the internal state of 
 *
 * \note
 * The code would probably be faster if we specified OverlapStrat as a template argument, but everything is already 
 * fairly template-heavy.
 *
 * \note
 * Problems could arise if you passed particle information to this class's methods (after you have reset the state) 
 * that aren't part of the "designated-feedback-particles".
 */
class OverlapScheduler {

private:
  /* the overlap strategy */
  OverlapStrat strat_ = OverlapStrat::ignore;
  /* counts the number of "passes" */
  part_int_t pass_count_ = 0;
  /* the total size of each shared register */
  std::size_t mask_size_ = 0;

  // the following attributes are all pointers to global device memory  

  ///@{
  /* the "masks" each are each pointers to have the same sizes as a fluid-field (including ghost zones) - each value corresponds
   * to a distinct cell in the simulation.
   *
   * In each mask, the value at each location will either store ``OverlapScheduler::DFLT_VAL`` or the minimum particle id
   * of a particle with pending feedback whose stencil "may" overlap with the location.
   *
   * In a given "pass", the values in curPass_mask_ are used to identify which particles can have their feedback applied in
   * the current pass and the values of nextPass_mask_ are updated to help which particles can have their feedback applied in
   * a future cycle.
   */
  part_int_t* curPass_mask_  = nullptr;
  part_int_t* nextPass_mask_ = nullptr;
  ///@}

  /* pointer to a global memory address that is used to track whether there are any particles with pending feedback that
   * will need to be applied in a future "pass".
   * 
   * At the start of each "pass" this is initialized to zero and it is gradually updated over the course of the "pass" (as particles
   * with pending feedback are encountered that must be handled in a future "pass"). If this has a non-zero value at the end of a
   * given "pass", then another "pass" is required.
   */
  part_int_t* any_pending_particles_ = nullptr;

public:
  /* the default value stored in a "mask".
   *
   * this is the max value representable by a 64-bit signed integer.
   */
  static inline constexpr part_int_t DFLT_VAL = 9223372036854775807;

public:

  /* default constructor */
  __host__ __device__ OverlapScheduler()
    : strat_(OverlapStrat::ignore),
      pass_count_(0),
      mask_size_(0),
      curPass_mask_(nullptr),
      nextPass_mask_(nullptr),
      any_pending_particles_(nullptr)
  { }


  /* Main constructor of OverlapScheduler.
   *
   * \note
   * the data_storage argument MUST persist longer than the lifetime of this
   * class. It manages the lifetime of the pointers used by this class.
   */
  __host__ OverlapScheduler(OverlapStrat strat,
                            cuda_utilities::DeviceVector<part_int_t>& data_storage,
                            int ng_x, int ng_y, int ng_z)
  : OverlapScheduler()
  {
    this->strat_ = strat;
    std::size_t mask_size = ng_x * ng_y * ng_z;

    switch (strat) {
      case OverlapStrat::ignore:
        // this is all redundant, but we opt for explicitness
        this->pass_count_            = 0;
        this->mask_size_             = 0;
        this->curPass_mask_          = nullptr;
        this->nextPass_mask_         = nullptr;
        this->any_pending_particles_ = nullptr;
        break;
      case OverlapStrat::sequential:
        // we may want to slightly over-allocate to ensure some particular alignments...
        data_storage.reset(1 + 2 * mask_size);
        this->pass_count_            = 0;
        this->mask_size_             = mask_size;
        this->curPass_mask_          = data_storage.data();
        this->nextPass_mask_         = this->curPass_mask_ + mask_size;
        this->any_pending_particles_ = this->nextPass_mask_ + mask_size;
        break;
    }
  }

  /* This must be called at the start of the kernel call where an OverlapScheduler will be used
   *
   * \note
   * This is a collective operation that must be executed by all threads (throughout the entire 
   * grid) at once. Deadlocks will occur if this is executed in a conditional branch that some
   * threads can't reach.
   */
  __device__ void Reset_State(const cooperative_groups::grid_group& g)
  {
    this->pass_count_ = 0;

    if (this->strat_ != OverlapStrat::ignore) {
      if (Is_GridWide_Root_Thread())  *(this->any_pending_particles_) = 0;
      // in the above operation, it should't logically matter whether one or more thread modifies
      // any_pending_particles_'s contents (but I suspect that it may affect performance)

      OverlapScheduler::clear_mask(this->nextPass_mask_, this->mask_size_);

      g.sync(); // this sync is required to ensure that all threads across the grid are done 
                // clearing the mask (before we start mutating the mask)
    }
  }

  /* try to prepare for the next pass through particles with pending feedback.
   *
   * \returns true if there are any particles with pending feedback remaining.
   *
   * \note
   * This is a collective operation that must be executed by all threads (throughout the entire 
   * grid) at once. Deadlocks will occur if this is executed in a conditional branch that some
   * threads can't reach.
   */
  __device__ bool Prepare_Next_Pass(const cooperative_groups::grid_group& g) {

    if (this->strat_ == OverlapStrat::ignore) { // in this scenario, only a single pass is required!
      if (this->pass_count_ != 0) return false;
      this->pass_count_ = 1;
      return true;
    }

    // the rest of this logic is for OverlapStrat::sequential

    g.sync();  // this is important! we need to be sure all threads across all thread-blocks are done
               // completing any previous "passes" through the data to make sure all threads agree that
               // another pass is required.

    // if there are no particles with pending feedback, we are done! We can exit now
    if (0 == *(this->any_pending_particles_))  return false;

    // otherwise we need to continue on. Let's synchronize since we will be resetting the value of
    // this->any_pending_particles_ (is this a place where we can use a memory fence?)
    g.sync();
    if (Is_GridWide_Root_Thread())  *(this->any_pending_particles_) = 0;
    // in the above operation, it should't logically matter whether one or more thread modifies
    // any_pending_particles_'s contents (but I suspect that it may affect performance)

    // increment the total pass-count
    this->pass_count_++;

    // Finally prepare the masks for the next loop
    Ptr_Swap(this->curPass_mask_, this->nextPass_mask_);
    OverlapScheduler::clear_mask(this->nextPass_mask_, this->mask_size_);

    g.sync(); // this last sync is required to make sure all threads across the grid are done clearing
              // the mask (before we start mutating the mask)
    return true;
  }


  /* External users of OverlapScheduler must calls this during initial setup for each particle that will
   * undergo feedback during the upcoming cycle.
   *
   * \note
   * This is also used internally.
   */
  template<typename Prescription>
  __device__ void Register_Pending_Particle_Feedback(Prescription p, long long int particle_id,
                                                     Arr3<Real> pos_indU, int ng_x, int ng_y)
  {
    if (strat_ == OverlapStrat::ignore) return;

    static_assert(sizeof(long long int) == sizeof(part_int_t));
    long long int* mask = (long long int*)(this->nextPass_mask_);
	  p.for_each_possible_overlap(
	    pos_indU[0], pos_indU[1], pos_indU[2], ng_x, ng_y,
      [mask, particle_id](Real dummy_arg, int ind3d) -> void {atomicMin(mask + ind3d, particle_id);}
    );
  }

  /* Checks whether a particle that applies feedback during the current cycle is scheduled to apply feedback right now
   * (in the current "pass"). This function MAY also update the internal state to prepare for the schdeduler for future
   * "passes".
   *
   * This will return `false` for particles that already applied feedback during the current simulation-cycle (but in a
   * prior "pass")  It will also return `false` if the specified particle's feedback is scheduled for a future pass. 
   */
  template<typename Prescription>
  __device__ bool Is_Scheduled_And_Update(Prescription p, part_int_t particle_id, Arr3<Real> pos_indU,
                                          int ng_x, int ng_y)
  {
    if (this->strat_ == OverlapStrat::ignore) return true;

    // retrieve the minimum particle_id in the current zone
    part_int_t min_id = OverlapScheduler::DFLT_VAL;
    part_int_t* mask = this->curPass_mask_;

    p.for_each_possible_overlap(
      pos_indU[0], pos_indU[1], pos_indU[2], ng_x, ng_y,
      [mask,&min_id](Real dummy_arg, int ind3d) -> void {min_id = min(min_id, mask[ind3d]);}
    );

    if (particle_id < min_id) {          // feedback was applied in prior "pass"
	    return false;
	  } else if (particle_id == min_id) {  // feedback is scheduled for current "pass"
      return true;
    } else {                             // particle feedback scheduled for future "pass"
      // record that another pass is necessary
      atomicAdd(this->any_pending_particles_, 1);
      // prepare the mask for the next pass
      Register_Pending_Particle_Feedback(p, particle_id, pos_indU, ng_x, ng_y);
      return false;
    }
  }

private:

  /* To be called across all threads and blocks at once
   *
   * \note
   * Assumes that blockDim.y, blockDim.z, gridDim.y, gridDim.z are all 1.
   * We can fix this!
   */
  static __device__ void clear_mask(part_int_t* ptr, std::size_t len) {
	  len *= std::size_t(ptr != nullptr);
    const std::size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t loop_stride = blockDim.x * gridDim.x;
    for (int i = start; i < len; i += loop_stride) { 
      ptr[i] = OverlapScheduler::DFLT_VAL;
    }
  }

};

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
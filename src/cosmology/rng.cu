
#include "../utils/cuda_utilities.h"
#include "rng.h"


/*! \fn void RNG_Init_GPU(int nx, int ny, int nz, int n_ghost, unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state)
 *  \brief Initialize a GPU-based RNG */
__global__ void RNG_Init_GPU(int nx, int ny, int nz, int n_ghost, unsigned long long seed, unsigned long long subsequence, unsigned long long offset, rng_parallel_state_t *states)
{

  // indices
  int xid, yid, zid;
  int const threadId = threadIdx.x + blockIdx.x * blockDim.x;

	// determine the cell location
  cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);

	// only real cells participate
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {

	  // copy state to local memory for efficiency
	  rng_parallel_state_t localState = states[threadId];

		// initialize the Philox RNG using the 
		// shared seed, the rank-specific subsequence
		// the rank-specific offset, and the philox state
		curand_init(seed, threadId, offset, &localState);
	
    states[threadId] = localState;
	}
}


/*! \fn void RNG_Normal_Field_GPU(Real *d_field, int nx, int ny, int nz, int n_ghost, curandStatePhilox4_32_10_t *state)
 *  \brief Generate a normal gaussian random field on a grid */
__global__ void RNG_Normal_Field_GPU(Real *d_field, int nx, int ny, int nz, int n_ghost, rng_parallel_state_t *states)
{
  // indices
  int xid, yid, zid;
  int const threadId = threadIdx.x + blockIdx.x * blockDim.x;

	// determine the cell location
  cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);


	// only real cells participate
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {

	  rng_parallel_state_t localState = states[threadId];
	
		// pull a gaussian random variate for each cell
		d_field[threadId] = gpurand_normal(&localState); // precision-aware wrapper

	  states[threadId] = localState;
	}
}



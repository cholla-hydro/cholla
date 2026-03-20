
#include "rng.h"


/*! \fn void NG_Init_GPU(unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state)
 *  \brief Initialize a GPU-based RNG */
__global__ void RNG_Init_GPU(int n_cells, int n_ghost, unsigned long long seed, unsigned long long subsequence, unsigned long long offset, rng_parallel_state_t *states)
{

	// determine the cell location
	int id;

	// get a global thread ID
	id = threadIdx.x + blockIdx.x * blockDim.x;

	// copy state to local memory for efficiency
	rng_parallel_state_t localState = states[id];

	// only real cells participate
	if (id > n_ghost - 1 && id < n_cells - n_ghost) {

		// initialize the Philox RNG using the 
		// shared seed, the rank-specific subsequence
		// the rank-specific offset, and the philox state
		curand_init(seed, id, offset, &localState);
	}
	states[id] = localState;
}


/*! \fn void RNG_Normal_Field_GPU(Real *d_field, int n_cells, int n_ghost, curandStatePhilox4_32_10_t *state)
 *  \brief Generate a normal gaussian random field on a grid */
__global__ void RNG_Normal_Field_GPU(Real *d_field, int n_cells, int n_ghost, rng_parallel_state_t *states)
{
	// determine the cell location
	int id;

	// get a global thread ID
	id = threadIdx.x + blockIdx.x * blockDim.x;

	rng_parallel_state_t localState = states[id];

	// only real cells participate
	if (id > n_ghost - 1 && id < n_cells - n_ghost) {

		// pull a gaussian random variate for each cell
//		d_field[id] = gpurand_normal(&states[id]);
//		d_field[id] = curand_normal(&states[id]);
//		d_field[id] = curand_normal(&localState);
		d_field[id] = curand_uniform(&localState);

	}
	states[id] = localState;
}



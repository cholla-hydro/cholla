
#include "rng.h"


/*! \fn void NG_Init_GPU(unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state)
 *  \brief Initialize a GPU-based RNG */
__device__ void RNG_Init_GPU(unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state)
{
	// initialize the Philox RNG using the 
	// shared seed, the rank-specific subsequence
	// the rank-specific offset, and the philox state
	curand_init(seed, subsequence, offset, state);
}


/*! \fn void RNG_Normal_Field_GPU(Real *d_field, int n_cells, int n_ghost, curandStatePhilox4_32_10_t *state)
 *  \brief Generate a normal gaussian random field on a grid */
inline __device__ void RNG_Normal_Field_GPU(Real *d_field, int n_cells, int n_ghost, curandStatePhilox4_32_10_t *state)
{
	// determine the cell location
	int id;

	// get a global thread ID
	id = threadIdx.x + blockIdx.x * blockDim.x;

	// only real cells participate
	if (id > n_ghost - 1 && id < n_cells - n_ghost) {

		// pull a gaussian random variate for each cell
		d_field[id] = gpurand_normal(state);
	}
}



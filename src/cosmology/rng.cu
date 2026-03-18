
#include "rng.h"

__device__ void RNG_Init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state)
{
	curand_init(seed, subsequence, offset, state);
}

inline __device__ void RNG_Normal_Field(Real *d_field, int n_cells, int n_ghost, curandStatePhilox4_32_10_t *state)
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
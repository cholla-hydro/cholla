
#include "../utils/cuda_utilities.h"
#include "rng.h"

/*
  // initialze
  cuda_utilities::AutomaticLaunchParams static const launchParams(RNG_Init_GPU, n_cells);
  hipLaunchKernelGGL(RNG_Init_GPU, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                     nx_local,ny_local,nz_local,0,CP.rng_seed, CP.rng_subsequence, CP.rng_offset, rng_states);
//                     H.nx,H.ny,H.nz,0,CP.rng_seed, CP.rng_subsequence, CP.rng_offset, rng_states);
*/
/*
  cuda_utilities::AutomaticLaunchParams static const launchParams(RNG_Normal_Field_GPU, n_cells);
  hipLaunchKernelGGL(RNG_Normal_Field_GPU, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                     d_field, nx_local, ny_local, nz_local, 0, state);
*/


/*! \fn void RNG_Init_GPU_TEST(int nx, int ny, int nz, int n_ghost, unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state)
 *  \brief Initialize a GPU-based RNG */
__global__ void RNG_Init_GPU_TEST(int nx, int ny, int nz, int nx_local_start, int ny_local_start, int nz_local_start, uint64_t seed, rng_parallel_state_t *states) {

  // indices
  int xid, yid, zid;
  int const threadId = threadIdx.x + blockIdx.x * blockDim.x;

	// determine the cell location
  cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);

	// only real cells participate
  if((xid>=0)&(xid<nx)&(yid>=0)&(yid<ny)&(zid>=0)&(zid<nz)) { // all cells are real

    uint64_t global_idx = (xid + nx_local_start);
    global_idx += (yid + ny_local_start)*nx;
    global_idx += (zid + nz_local_start)*nx*ny;

    uint64_t subsequence = global_idx >> 32;
    uint64_t offset = global_idx & 0xFFFFFFFFULL;

	  // copy state to local memory for efficiency
	  rng_parallel_state_t localState = states[threadId];

		// initialize the Philox RNG using the 
		// shared seed, the rank-specific subsequence
		// the rank-specific offset, and the philox state
		curand_init(seed, subsequence, offset, &localState);
	
    states[threadId] = localState;
	}
}



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
  if((xid>=0)&(xid<nx)&(yid>=0)&(yid<ny)&(zid>=0)&(zid<nz)) { // all cells are real

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
  if((xid>=0)&(xid<nx)&(yid>=0)&(yid<ny)&(zid>=0)&(zid<nz)) { // all cells are real

	  rng_parallel_state_t localState = states[threadId];
	
		// pull a gaussian random variate for each cell
		d_field[threadId] = gpurand_normal(&localState); // precision-aware wrapper

	  states[threadId] = localState;
	}
}




#include "../utils/cuda_utilities.h"
#include "rng.h"

#if PRECISION == 1
  #ifndef TYPEDEF_DEFINED_REAL
typedef float4 Real4;
  #endif
#endif
#if PRECISION == 2
  #ifndef TYPEDEF_DEFINED_REAL
typedef double4 Real4;
  #endif
#endif

/*! \fn void RNG_Init_GPU(int nx, int ny, int nz, int n_ghost, unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state)
 *  \brief Initialize a GPU-based RNG */
__global__ void RNG_Init_GPU(int nx_local, int ny_local, int nz_local, int nx_local_start, int ny_local_start, int nz_local_start, int nx, int ny, int nz, uint64_t seed, rng_parallel_state_t *states) {

  // indices
  int xid, yid, zid;
  int const threadId = threadIdx.x + blockIdx.x * blockDim.x;

	// determine the cell location
  cuda_utilities::compute3DIndices(threadId, nx_local, ny_local, xid, yid, zid);

	// only real cells participate
  if((xid>=0)&(xid<nx_local)&(yid>=0)&(yid<ny_local)&(zid>=0)&(zid<nz_local)) { // all cells are real

    // create a global real-cell index
    uint64_t global_idx = (xid + nx_local_start);
    global_idx += (yid + ny_local_start)*nx;
    global_idx += (zid + nz_local_start)*nx*ny;

    // create a reproducible subsequence and offset
    //uint64_t subsequence = global_idx >> 32;
    //uint64_t offset = global_idx & 0xFFFFFFFFULL;
    //uint64_t subsequence = global_idx >> 48;
    uint64_t subsequence = global_idx;
    //uint64_t offset = global_idx & 0xFFFFFFFFFFFFULL;
    uint64_t offset = 0;

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
__global__ void RNG_Init_TEST(int procID, int nx_local, int ny_local, int nz_local, int nx_local_start, int ny_local_start, int nz_local_start, int nx, int ny, int nz, uint64_t seed, rng_parallel_state_t *states) {

  // indices
  int xid, yid, zid;
  int const threadId = threadIdx.x + blockIdx.x * blockDim.x;

	// determine the cell location
  cuda_utilities::compute3DIndices(threadId, nx_local, ny_local, xid, yid, zid);

	// only real cells participate
  if((xid>=0)&(xid<nx_local)&(yid>=0)&(yid<ny_local)&(zid>=0)&(zid<nz_local)) { // all cells are real

    // create a global real-cell index
    uint64_t global_idx = (xid + nx_local_start);
    global_idx += (yid + ny_local_start)*nx;
    global_idx += (zid + nz_local_start)*nx*ny;

    // create a reproducible subsequence and offset
    //uint64_t subsequence = global_idx >> 32;
    //uint64_t offset = global_idx & 0xFFFFFFFFULL;
    //uint64_t subsequence = global_idx >> 48;
    //uint64_t subsequence = global_idx;
    //uint64_t offset = global_idx & 0xFFFFFFFFFFFFULL;
    uint64_t offset = 0;
    uint64_t subsequence = global_idx;
    //subsequence += (1ULL << 32)*procID; // explore sequence cycling every 2^32
    subsequence = 0;
    if(procID==1)
      subsequence = 1ULL << 32; //HERE
      subsequence += global_idx;

	  // copy state to local memory for efficiency
	  rng_parallel_state_t localState = states[threadId];

		// initialize the Philox RNG using the 
		// shared seed, the rank-specific subsequence
		// the rank-specific offset, and the philox state
		curand_init(seed, subsequence, offset, &localState);
	
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
    //Real4 variate = gpurand_normal4(&localState); 
    //d_field[threadId] = variate.x;
    d_field[threadId] = gpurand_normal(&localState); // precision-aware wrapper

    states[threadId] = localState;
  }
}




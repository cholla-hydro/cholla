
#include "../utils/cuda_utilities.h"
#include "rng.h"

#if PRECISION == 1
  #ifndef TYPEDEF_DEFINED_REAL
typedef float4 Real4;
typedef float2 Real2;
  #endif
#endif
#if PRECISION == 2
  #ifndef TYPEDEF_DEFINED_REAL
typedef double4 Real4;
typedef double2 Real2;
  #endif
#endif

__global__ void RNG_Init_GPU(int nx_local, int ny_local, int nz_local, int nx_local_start, int ny_local_start,
                             int nz_local_start, int nx, int ny, int nz, uint64_t seed, rng_parallel_state_t *states)
{
  // indices
  // int xid, yid, zid;
  uint64_t const threadId = threadIdx.x + blockIdx.x * blockDim.x;

  // determine the cell location
  // cuda_utilities::compute3DIndices(threadId, nx_local, ny_local, xid, yid, zid);
  uint64_t zid = threadId / (nx_local * ny_local);
  uint64_t yid = (threadId - zid * nx_local * ny_local) / nx_local;
  uint64_t xid = threadId - zid * nx_local * ny_local - yid * nx_local;

  // only real cells participate
  if ((xid < nx_local) and (yid < ny_local) and (zid < nz_local)) {  // all cells are real

    // create a global real-cell index
    // uint64_t global_idx = (xid + nx_local_start);
    // global_idx += (yid + ny_local_start)*nx;
    // global_idx += (zid + nz_local_start)*nx*ny;
    uint64_t global_idx = (xid + nx_local_start);
    global_idx += (yid + ny_local_start) * nx;
    global_idx += (zid + nz_local_start) * nx * ny;

    // create a reproducible subsequence and offset
    // uint64_t subsequence = global_idx >> 32;
    // uint64_t offset = global_idx & 0xFFFFFFFFULL;
    // uint64_t subsequence = global_idx >> 48;
    // uint64_t subsequence = global_idx;
    // uint64_t subsequence = 0;
    uint64_t offset = 0;
    // uint64_t subsequence = global_idx >> 32;
    uint64_t flag = global_idx >> 32;
    // uint64_t offset = global_idx & 0xFFFFFFFFFFFFULL;
    // uint64_t offset = global_idx;

    // copy state to local memory for efficiency
    rng_parallel_state_t localState = states[threadId];

    // initialize the Philox RNG using the
    // shared seed, the rank-specific subsequence
    // the rank-specific offset, and the philox state
    // BRANT ERROR this could only work for multiples of 2048^3
    // We need a larger iterator, maybe only CPU can work?
    // if(flag) {
    // 	curand_init(seed+1, subsequence, offset, &localState);
    //}else{
    // curand_init(seed, subsequence, offset, &localState);
    curand_init(seed + flag, 0, 0, &localState);  // same state
    //}

    states[threadId] = localState;
  }
}

__global__ void RNG_Init_TEST(int procID, int nx_local, int ny_local, int nz_local, int nx_local_start,
                              int ny_local_start, int nz_local_start, int nx, int ny, int nz, uint64_t seed,
                              rng_parallel_state_t *states)
{
  // indices
  int xid, yid, zid;
  int const threadId = threadIdx.x + blockIdx.x * blockDim.x;

  // determine the cell location
  cuda_utilities::compute3DIndices(threadId, nx_local, ny_local, xid, yid, zid);

  // only real cells participate
  if ((xid >= 0) and (xid < nx_local) and (yid >= 0) and (yid < ny_local) and
      (zid >= 0) & (zid < nz_local)) {  // all cells are real

    // create a global real-cell index
    uint64_t global_idx = (xid + nx_local_start);
    global_idx += (yid + ny_local_start) * nx;
    global_idx += (zid + nz_local_start) * nx * ny;

    // create a reproducible subsequence and offset
    uint64_t offset      = 0;
    uint64_t subsequence = global_idx;

    // copy state to local memory for efficiency
    rng_parallel_state_t localState = states[threadId];

    // initialize the Philox RNG using the
    // shared seed, the rank-specific subsequence
    // the rank-specific offset, and the philox state
    // curand_init(seed, subsequence, offset, &localState);
    // curand_init(seed, subsequence, offset, &localState);
    curand_init(seed, subsequence, offset, &localState);

    states[threadId] = localState;
  }
}

/*! \fn void RNG_Normal_Field_GPU(Real *d_field, int nx, int ny, int nz, int n_ghost, curandStatePhilox4_32_10_t *state)
 *  \brief Generate a normal gaussian random field on a grid */
__global__ void RNG_Normal_Field_GPU(Real *d_field, int nx_local, int ny_local, int nz_local, int nx_local_start,
                                     int ny_local_start, int nz_local_start, int nx, int ny, int nz, uint64_t seed,
                                     rng_parallel_state_t *states)
{
  // indices
  // int xid, yid, zid;
  uint64_t const threadId = threadIdx.x + blockIdx.x * blockDim.x;

  // determine the cell location
  // cuda_utilities::compute3DIndices(threadId, nx_local, ny_local, xid, yid, zid);
  // try again
  // let's generate the RNGs on the CPU
  uint64_t zid = threadId / (nx_local * ny_local);
  uint64_t yid = (threadId - zid * nx_local * ny_local) / nx_local;
  uint64_t xid = threadId - zid * nx_local * ny_local - yid * nx_local;

  // only real cells participate
  if ((xid < nx_local) and (yid < ny_local) and (zid < nz_local)) {  // all cells are real

    // create a global real-cell index
    uint64_t global_idx = (xid + nx_local_start);
    global_idx += (yid + ny_local_start) * nx;
    global_idx += (zid + nz_local_start) * nx * ny;

    rng_parallel_state_t localState = states[threadId];

    // skip ahead
    skipahead(global_idx, &localState);

    // uint64_t offset = global_idx;
    // uint64_t subsequence = global_idx;
    // uint64_t offset = 0;
    // uint64_t flag = global_idx >> 32;
    // uint64_t subsequence = flag;
    // if(flag)
    //{
    // curand_init(seed+flag, subsequence, offset, &localState);
    // d_field[threadId] = 0;
    // curand_init(seed, subsequence, offset, &localState);
    //}//else{
    // 	d_field[threadId] = gpurand_normal(&localState);
    //}
    // curand_init(seed, subsequence, offset, &localState);
    //	curand_init(seed, subsequence, flag, &localState);
    d_field[threadId] = gpurand_normal(&localState);

    /*

     // pull two random uniform numbers
     Real2 u = gpurand_uniform2(&localState);

     // do box-muller transform
     Real r = sqrt(-2.0 * log(u.x));
     Real theta = 2.0 * RNG_PI * u.y;
     */

    // force the 128-bit counter to advance perfectly across 64-bit boundaries.
    // state.v[0] and state.v[1] hold the lower 64 bits of the Philox counter.
    // localState.v[0] = (unsigned int)(global_index & 0xFFFFFFFFULL);
    // localState.v[1] = (unsigned int)(global_index  >> 32);

    // clear any internal Box-Muller tracking flags to prevent stale state masking
    // localState.boxmuller_index = 0;
    // localState.boxmuller_flag = 0;

    // pull a gaussian random variate for each cell
    // Real4 variate = gpurand_normal4(&localState);
    // d_field[threadId] = variate.x;
    // d_field[threadId] = gpurand_normal(&localState); // precision-aware wrapper
    // d_field[threadId] = r * cos(theta); // just need one
    // d_field[threadId] = gpurand_normal(&localState);

    states[threadId] = localState;
  }
}

/*! \fn void RNG_Normal_Field_GPU(Real *d_field, int nx, int ny, int nz, int n_ghost, curandStatePhilox4_32_10_t *state)
 *  \brief Generate a normal gaussian random field on a grid */
__global__ void RNG_Normal_Field_GPU_BAK(Real *d_field, int nx, int ny, int nz, int n_ghost,
                                         rng_parallel_state_t *states)
{
  // indices
  int xid, yid, zid;
  int const threadId = threadIdx.x + blockIdx.x * blockDim.x;

  // determine the cell location
  cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);

  // only real cells participate
  if ((xid >= 0) and (xid < nx) and (yid >= 0) and (yid < ny) and (zid >= 0) and (zid < nz)) {  // all cells are real
    rng_parallel_state_t localState = states[threadId];

    // pull a gaussian random variate for each cell
    // Real4 variate = gpurand_normal4(&localState);
    // d_field[threadId] = variate.x;
    d_field[threadId] = gpurand_normal(&localState);  // precision-aware wrapper

    states[threadId] = localState;
  }
}


#include "../utils/cuda_utilities.h"
#include "rng.h"

__global__ void RNG_Init_GPU(int nx_local, int ny_local, int nz_local, int nx_local_start, int ny_local_start,
                             int nz_local_start, int nx, int ny, int nz, uint64_t seed, rng_parallel_state_t *states)
{
  // indices
  uint64_t const threadId = threadIdx.x + blockIdx.x * blockDim.x;

  // determine the cell location
  uint64_t zid = threadId / (nx_local * ny_local);
  uint64_t yid = (threadId - zid * nx_local * ny_local) / nx_local;
  uint64_t xid = threadId - zid * nx_local * ny_local - yid * nx_local;

  // only real cells participate
  if ((xid < nx_local) and (yid < ny_local) and (zid < nz_local)) {  // all cells are real

    // create a global real-cell index
    uint64_t global_idx = (xid + nx_local_start);
    global_idx += (yid + ny_local_start) * nx;
    global_idx += (zid + nz_local_start) * nx * ny;

    // create a reproducible subsequence and offset
    uint64_t offset = 0;
    uint64_t flag   = global_idx >> 32;

    // copy state to local memory for efficiency
    rng_parallel_state_t localState = states[threadId];

    // initialize the HIP or CUDA RNG using the
    // shared seed, the rank-specific subsequence
    // the rank-specific offset, and the rng state
    curand_init(seed + flag, 0, 0, &localState);

    states[threadId] = localState;
  }
}

__global__ void RNG_Normal_Field_GPU(Real *d_field, int nx_local, int ny_local, int nz_local, int nx_local_start,
                                     int ny_local_start, int nz_local_start, int nx, int ny, int nz, uint64_t seed,
                                     rng_parallel_state_t *states)
{
  // indices
  uint64_t const threadId = threadIdx.x + blockIdx.x * blockDim.x;

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

    d_field[threadId] = gpurand_normal(&localState);

    states[threadId] = localState;
  }
}
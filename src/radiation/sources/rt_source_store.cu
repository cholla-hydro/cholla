/*! \file rt_source_store.cu
 *  \brief Device storage, drift, and MPI migration of RT sources. */

#ifdef RT

  #include <algorithm>
  #include <vector>

  #include "../../global/global.h"
  #include "../../global/global_cuda.h"
  #include "../../io/io.h"
  #include "../../utils/error_handling.h"
  #include "../../utils/gpu.hpp"
  #include "rt_source_store.h"
  #ifdef MPI_CHOLLA
    #include "../../mpi/mpi_routines.h"
  #endif

  #define TPB_RT_SOURCES 256

namespace rt_sources
{

namespace
{

int N_Blocks(int n) { return (n + TPB_RT_SOURCES - 1) / TPB_RT_SOURCES; }

/*! Copy n packed sources [i*NF + f] into field-major storage starting at slot offset. */
__global__ void Unpack_Kernel(int n, int offset, int cap, const Real *__restrict__ packed, Real *__restrict__ data)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int f = 0; f < SRC_NFIELDS; f++) data[(size_t)f * cap + offset + i] = packed[(size_t)i * SRC_NFIELDS + f];
}

/*! Copy n field-major sources into packed form [i*NF + f]. */
__global__ void Pack_Kernel(int n, int cap, const Real *__restrict__ data, Real *__restrict__ packed)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int f = 0; f < SRC_NFIELDS; f++) packed[(size_t)i * SRC_NFIELDS + f] = data[(size_t)f * cap + i];
}

__global__ void Drift_Kernel(int n, int cap, Real *data, Real drift_factor)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int d = 0; d < 3; d++)
    data[(size_t)(SRC_X + d) * cap + i] += drift_factor * data[(size_t)(SRC_VX + d) * cap + i];
}

/*! Split sources by position along dir. Sources inside [lo, hi) are copied to scratch;
 *  sources below lo or at/above hi are packed into send_lo / send_hi, with their position
 *  wrapped periodically into [gmin, gmin + L). The order within each output is not
 *  deterministic (atomic slot allocation); nothing downstream depends on it. */
__global__ void Partition_Kernel(int n, int cap, const Real *__restrict__ data, Real *__restrict__ scratch,
                                 Real *__restrict__ send_lo, Real *__restrict__ send_hi, int *counts, int dir, Real lo,
                                 Real hi, Real gmin, Real L)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  Real x = data[(size_t)(SRC_X + dir) * cap + i];

  if (x >= lo && x < hi) {
    const int k = atomicAdd(&counts[0], 1);
    for (int f = 0; f < SRC_NFIELDS; f++) scratch[(size_t)f * cap + k] = data[(size_t)f * cap + i];
    return;
  }

  const int side = (x < lo) ? 0 : 1;
  if (x < gmin) x += L;
  if (x >= gmin + L) x -= L;

  Real *buf   = (side == 0) ? send_lo : send_hi;
  const int k = atomicAdd(&counts[1 + side], 1);
  for (int f = 0; f < SRC_NFIELDS; f++) buf[(size_t)k * SRC_NFIELDS + f] = data[(size_t)f * cap + i];
  buf[(size_t)k * SRC_NFIELDS + SRC_X + dir] = x;
}

}  // namespace

SourceStore::~SourceStore()
{
  if (dev_data) cudaFree(dev_data);
  if (dev_scratch_) cudaFree(dev_scratch_);
  for (int s = 0; s < 2; s++)
    if (dev_send_[s]) cudaFree(dev_send_[s]);
  if (dev_unpack_) cudaFree(dev_unpack_);
  if (dev_counts_) cudaFree(dev_counts_);
}

void SourceStore::Reserve(int n)
{
  if (dev_counts_ == nullptr) GPU_Error_Check(cudaMalloc((void **)&dev_counts_, 3 * sizeof(int)));
  if (n <= capacity) return;

  const int new_cap        = std::max({n, 2 * capacity, 1024});
  const size_t field_bytes = (size_t)new_cap * SRC_NFIELDS * sizeof(Real);

  Real *new_data = nullptr;
  GPU_Error_Check(cudaMalloc((void **)&new_data, field_bytes));
  for (int f = 0; f < SRC_NFIELDS && n_local > 0; f++) {
    GPU_Error_Check(cudaMemcpy(new_data + (size_t)f * new_cap, dev_data + (size_t)f * capacity, n_local * sizeof(Real),
                               cudaMemcpyDeviceToDevice));
  }

  if (dev_data) cudaFree(dev_data);
  if (dev_scratch_) cudaFree(dev_scratch_);
  for (int s = 0; s < 2; s++)
    if (dev_send_[s]) cudaFree(dev_send_[s]);
  if (dev_unpack_) cudaFree(dev_unpack_);

  // The scratch, send and unpack buffers are each as large as the store, so a partition
  // or an append can never overflow them.
  dev_data = new_data;
  GPU_Error_Check(cudaMalloc((void **)&dev_scratch_, field_bytes));
  for (int s = 0; s < 2; s++) GPU_Error_Check(cudaMalloc((void **)&dev_send_[s], field_bytes));
  GPU_Error_Check(cudaMalloc((void **)&dev_unpack_, field_bytes));
  capacity = new_cap;
}

void SourceStore::Append_Device(const Real *dev_packed, int n)
{
  if (n <= 0) return;
  Reserve(n_local + n);
  hipLaunchKernelGGL(Unpack_Kernel, N_Blocks(n), TPB_RT_SOURCES, 0, 0, n, n_local, capacity, dev_packed, dev_data);
  GPU_Error_Check();
  n_local += n;
}

void SourceStore::Append_Host(const Real *packed, int n)
{
  // Appending in chunks keeps the unpack buffer bounded by the capacity at the time.
  int done = 0;
  while (done < n) {
    Reserve(n_local + (n - done));
    const int chunk = std::min(n - done, capacity);
    GPU_Error_Check(cudaMemcpy(dev_unpack_, packed + (size_t)done * SRC_NFIELDS,
                               (size_t)chunk * SRC_NFIELDS * sizeof(Real), cudaMemcpyHostToDevice));
    Append_Device(dev_unpack_, chunk);
    done += chunk;
  }
}

void SourceStore::Copy_To_Host(std::vector<Real> &packed) const
{
  packed.resize((size_t)n_local * SRC_NFIELDS);
  if (n_local == 0) return;
  hipLaunchKernelGGL(Pack_Kernel, N_Blocks(n_local), TPB_RT_SOURCES, 0, 0, n_local, capacity, dev_data, dev_unpack_);
  GPU_Error_Check(cudaMemcpy(packed.data(), dev_unpack_, packed.size() * sizeof(Real), cudaMemcpyDeviceToHost));
}

void SourceStore::Drift(Real drift_factor)
{
  if (n_local == 0 || drift_factor == 0) return;
  hipLaunchKernelGGL(Drift_Kernel, N_Blocks(n_local), TPB_RT_SOURCES, 0, 0, n_local, capacity, dev_data, drift_factor);
  GPU_Error_Check();
}

void SourceStore::Migrate(const RT_State &state)
{
  // One direction at a time, like the particle transfer: a source that crosses a corner
  // reaches the diagonal neighbor in two hops.
  for (int dir = 0; dir < 3; dir++) Migrate_Direction_(state, dir);
}

void SourceStore::Migrate_Direction_(const RT_State &state, int dir)
{
  Reserve(1);  // make sure the buffers exist even with no sources

  int counts[3] = {0, 0, 0};
  GPU_Error_Check(cudaMemset(dev_counts_, 0, 3 * sizeof(int)));
  if (n_local > 0) {
    hipLaunchKernelGGL(Partition_Kernel, N_Blocks(n_local), TPB_RT_SOURCES, 0, 0, n_local, capacity, dev_data,
                       dev_scratch_, dev_send_[0], dev_send_[1], dev_counts_, dir, state.Local_Min(dir),
                       state.Local_Max(dir), state.Global_Min(dir), state.Global_Len(dir));
    GPU_Error_Check();
    GPU_Error_Check(cudaMemcpy(counts, dev_counts_, 3 * sizeof(int), cudaMemcpyDeviceToHost));
    std::swap(dev_data, dev_scratch_);
    n_local = counts[0];
  }
  const int n_send[2] = {counts[1], counts[2]};

  // Collect what arrives on this rank, packed on the host.
  std::vector<Real> incoming;
  int n_dropped = 0;

  #ifdef MPI_CHOLLA
  // Tags: a message sent to the lower neighbor uses tag_base + 2*dir, to the upper neighbor
  // tag_base + 2*dir + 1. The receiver matches lower-sent messages from its upper neighbor.
  const int tag_base = 7100;
  const int tag[2]   = {tag_base + 2 * dir, tag_base + 2 * dir + 1};
  int n_recv[2]      = {0, 0};
  MPI_Request req[4];
  int nreq = 0;
  for (int s = 0; s < 2; s++) {
    const int face = 2 * dir + s;
    if (state.flags[face] != 5) continue;
    // receive through face s the count the neighbor sent toward us (opposite tag)
    MPI_Irecv(&n_recv[s], 1, MPI_INT, source[face], tag[1 - s], world, &req[nreq++]);
    MPI_Isend(&n_send[s], 1, MPI_INT, dest[face], tag[s], world, &req[nreq++]);
  }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

  std::vector<Real> h_send[2], h_recv[2];
  nreq = 0;
  for (int s = 0; s < 2; s++) {
    const int face = 2 * dir + s;
    if (state.flags[face] != 5) continue;
    h_recv[s].resize((size_t)n_recv[s] * SRC_NFIELDS);
    h_send[s].resize((size_t)n_send[s] * SRC_NFIELDS);
    if (n_send[s] > 0)
      GPU_Error_Check(
          cudaMemcpy(h_send[s].data(), dev_send_[s], h_send[s].size() * sizeof(Real), cudaMemcpyDeviceToHost));
    MPI_Irecv(h_recv[s].data(), (int)h_recv[s].size(), MPI_CHREAL, source[face], tag[1 - s], world, &req[nreq++]);
    MPI_Isend(h_send[s].data(), (int)h_send[s].size(), MPI_CHREAL, dest[face], tag[s], world, &req[nreq++]);
  }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
  for (int s = 0; s < 2; s++) incoming.insert(incoming.end(), h_recv[s].begin(), h_recv[s].end());
  #endif  // MPI_CHOLLA

  // Appending the periodic self-sends cannot reallocate dev_send_: kept + sent sources
  // equal the count before the partition, which already fit in the capacity.
  for (int s = 0; s < 2; s++) {
    const int flag = state.flags[2 * dir + s];
    if (flag == 1) {
      // periodic with ourselves: the wrapped sources come straight back
      Append_Device(dev_send_[s], n_send[s]);
    } else if (flag != 5) {
      n_dropped += n_send[s];  // outflow-type boundary: the source leaves the simulation
    }
  }
  if (!incoming.empty()) Append_Host(incoming.data(), (int)(incoming.size() / SRC_NFIELDS));

  if (n_dropped > 0) chprintf("RT sources: %d source(s) left the domain through a non-periodic boundary.\n", n_dropped);
}

}  // namespace rt_sources

#endif  // RT
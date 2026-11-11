/*! \file rt_source_apply.cu
 *  \brief Deposit sources into dev_rs and fold ghost-cell contributions back onto the
 *         ranks that own those cells. */

#ifdef RT

  #include <algorithm>
  #include <cmath>

  #include "../../global/global.h"
  #include "../../global/global_cuda.h"
  #include "../../utils/gpu.hpp"
  #include "rt_sources.h"
  #ifdef MPI_CHOLLA
    #include "../../mpi/mpi_routines.h"
  #endif

  #define TPB_RT_APPLY 256

namespace rt_sources
{

namespace
{

/*! Weights along one axis for the separable kernels. u is the source position in cell
 *  units, measured so that cell i spans [i, i+1). Returns the number of cells m and sets
 *  the first cell i0 and the weights w[0..m-1], which sum to 1. */
__device__ int Axis_Weights(int kernel, Real u, int &i0, Real w[3])
{
  switch (kernel) {
    case 0: {  // NGP: the cell containing the source
      i0   = (int)floor(u);
      w[0] = 1;
      return 1;
    }
    case 1: {  // CIC: the two cells whose centers bracket the source
      const Real s = u - 0.5;
      i0           = (int)floor(s);
      const Real f = s - i0;
      w[0]         = 1 - f;
      w[1]         = f;
      return 2;
    }
    default: {  // TSC: the containing cell and its two neighbors
      const int i  = (int)floor(u);
      const Real d = u - (i + 0.5);  // offset from the cell center, in [-0.5, 0.5)
      i0           = i - 1;
      w[0]         = 0.5 * (0.5 - d) * (0.5 - d);
      w[1]         = 0.75 - d * d;
      w[2]         = 0.5 * (0.5 + d) * (0.5 + d);
      return 3;
    }
  }
}

__device__ inline void Add_Cell(Real *rs, int i, int j, int k, int nx, int ny, int nz, Real v)
{
  if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return;  // cannot happen if half width <= n_ghost
  atomicAdd(&rs[i + nx * (j + ny * k)], v);
}

/*! One thread per source. rs receives lum * weight / cell volume, so the deposit of a
 *  lum = 1 source integrates to 1 over code volume, matching the initial conditions. */
__global__ void Deposit_Kernel(int n, int cap, const Real *__restrict__ data, Real *rs, int nx, int ny, int nz,
                               int n_ghost, Real xlo, Real ylo, Real zlo, Real dx, Real dy, Real dz, int kernel,
                               Real radius, Real t_mid)
{
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) return;

  const Real lum     = data[(size_t)SRC_LUM * cap + s];
  const Real t_birth = data[(size_t)SRC_TBIRTH * cap + s];
  const Real t_life  = data[(size_t)SRC_TLIFE * cap + s];
  if (lum == 0 || t_mid < t_birth) return;
  if (t_life > 0 && t_mid >= t_birth + t_life) return;

  // position in local cell units, including the ghost offset
  const Real u[3] = {(data[(size_t)SRC_X * cap + s] - xlo) / dx + n_ghost,
                     (data[(size_t)SRC_Y * cap + s] - ylo) / dy + n_ghost,
                     (data[(size_t)SRC_Z * cap + s] - zlo) / dz + n_ghost};
  const Real amp  = lum / (dx * dy * dz);

  if (kernel == 3) {
    // Top hat: every cell whose center lies within `radius` cells of the source, equally
    // weighted. Count first so the weights sum exactly to 1.
    const int lo[3] = {(int)floor(u[0] - radius), (int)floor(u[1] - radius), (int)floor(u[2] - radius)};
    const int hi[3] = {(int)floor(u[0] + radius), (int)floor(u[1] + radius), (int)floor(u[2] + radius)};
    const Real r2   = radius * radius;
    int count       = 0;
    for (int k = lo[2]; k <= hi[2]; k++)
      for (int j = lo[1]; j <= hi[1]; j++)
        for (int i = lo[0]; i <= hi[0]; i++) {
          const Real ex = i + 0.5 - u[0], ey = j + 0.5 - u[1], ez = k + 0.5 - u[2];
          if (ex * ex + ey * ey + ez * ez <= r2) count++;
        }
    if (count == 0) {  // radius smaller than the distance to any center: fall back to NGP
      Add_Cell(rs, (int)floor(u[0]), (int)floor(u[1]), (int)floor(u[2]), nx, ny, nz, amp);
      return;
    }
    const Real v = amp / count;
    for (int k = lo[2]; k <= hi[2]; k++)
      for (int j = lo[1]; j <= hi[1]; j++)
        for (int i = lo[0]; i <= hi[0]; i++) {
          const Real ex = i + 0.5 - u[0], ey = j + 0.5 - u[1], ez = k + 0.5 - u[2];
          if (ex * ex + ey * ey + ez * ez <= r2) Add_Cell(rs, i, j, k, nx, ny, nz, v);
        }
    return;
  }

  int i0[3], m[3];
  Real w[3][3];
  for (int d = 0; d < 3; d++) m[d] = Axis_Weights(kernel, u[d], i0[d], w[d]);

  for (int c = 0; c < m[2]; c++)
    for (int b = 0; b < m[1]; b++)
      for (int a = 0; a < m[0]; a++)
        Add_Cell(rs, i0[0] + a, i0[1] + b, i0[2] + c, nx, ny, nz, amp * w[0][a] * w[1][b] * w[2][c]);
}

/*! Copy a slab of n_ghost cells along `dir`, starting at index `start` along dir and
 *  spanning the full (ghost-inclusive) extent in the other two directions.
 *  mode 0: rs -> buf, then zero those cells.  mode 1: rs += buf. */
__global__ void Slab_Kernel(Real *rs, Real *buf, int nx, int ny, int nz, int n_ghost, int dir, int start, int mode)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j, k;
  if (dir == 0) {
    if (t >= n_ghost * ny * nz) return;
    const int g = t % n_ghost, jk = t / n_ghost;
    i = start + g;
    j = jk % ny;
    k = jk / ny;
  } else if (dir == 1) {
    if (t >= nx * n_ghost * nz) return;
    i            = t % nx;
    const int gk = t / nx;
    j            = start + gk % n_ghost;
    k            = gk / n_ghost;
  } else {
    if (t >= nx * ny * n_ghost) return;
    i            = t % nx;
    const int jg = t / nx;
    j            = jg % ny;
    k            = start + jg / ny;
  }
  const int id = i + nx * (j + ny * k);
  if (mode == 0) {
    buf[t] = rs[id];
    rs[id] = 0;
  } else {
    rs[id] += buf[t];
  }
}

int Slab_Size(const RT_State &s, int dir)
{
  if (dir == 0) return s.nghost * s.ny * s.nz;
  if (dir == 1) return s.nx * s.nghost * s.nz;
  return s.nx * s.ny * s.nghost;
}

int Dir_Len(const RT_State &s, int dir) { return dir == 0 ? s.nx : (dir == 1 ? s.ny : s.nz); }

void Launch_Slab(Real *rs, Real *buf, const RT_State &s, int dir, int start, int mode)
{
  const int n = Slab_Size(s, dir);
  hipLaunchKernelGGL(Slab_Kernel, (n + TPB_RT_APPLY - 1) / TPB_RT_APPLY, TPB_RT_APPLY, 0, 0, rs, buf, s.nx, s.ny, s.nz,
                     s.nghost, dir, start, mode);
  GPU_Error_Check();
}

}  // namespace

void FoldBuffers::Allocate(const RT_State &s)
{
  Free();
  size = std::max({Slab_Size(s, 0), Slab_Size(s, 1), Slab_Size(s, 2)});
  for (int b = 0; b < 4; b++) {
    GPU_Error_Check(cudaMalloc((void **)&dev[b], size * sizeof(Real)));
    host[b] = (Real *)malloc(size * sizeof(Real));
  }
}

void FoldBuffers::Free()
{
  for (int b = 0; b < 4; b++) {
    if (dev[b]) cudaFree(dev[b]);
    if (host[b]) free(host[b]);
    dev[b]  = nullptr;
    host[b] = nullptr;
  }
  size = 0;
}

void Deposit_Sources(const SourceStore &store, const RTPopulation &pop, const RT_State &s, Real *dev_rs)
{
  if (store.n_local == 0) return;
  hipLaunchKernelGGL(Deposit_Kernel, (store.n_local + TPB_RT_APPLY - 1) / TPB_RT_APPLY, TPB_RT_APPLY, 0, 0,
                     store.n_local, store.capacity, store.dev_data, dev_rs, s.nx, s.ny, s.nz, s.nghost, s.xblocal,
                     s.yblocal, s.zblocal, s.dx, s.dy, s.dz, static_cast<int>(pop.kernel), pop.kernel_radius,
                     s.t_mid_myr);
  GPU_Error_Check();
}

void Fold_Source_Ghosts(Real *dev_rs, const RT_State &s, FoldBuffers &buf)
{
  const int ng = s.nghost;
  for (int dir = 0; dir < 3; dir++) {
    const int n       = Dir_Len(s, dir);
    const int lo_face = 2 * dir, hi_face = 2 * dir + 1;

    // Ghost slabs: lower [0, ng) goes to the lower neighbor, upper [n-ng, n) to the upper.
    // They land in the receiver's active cells next to the shared face:
    //   from my upper neighbor (its lower ghosts) -> my [n-2ng, n-ng)
    //   from my lower neighbor (its upper ghosts) -> my [ng, 2ng)
    // Loading zeroes the slab, so cells already sent are not sent again in a later direction,
    // while corner cells still reach the diagonal neighbor through the next direction.
    Launch_Slab(dev_rs, buf.dev[0], s, dir, 0, 0);
    Launch_Slab(dev_rs, buf.dev[1], s, dir, n - ng, 0);

    if (s.flags[lo_face] == 1) Launch_Slab(dev_rs, buf.dev[0], s, dir, n - 2 * ng, 1);
    if (s.flags[hi_face] == 1) Launch_Slab(dev_rs, buf.dev[1], s, dir, ng, 1);

  #ifdef MPI_CHOLLA
    const int n_slab   = Slab_Size(s, dir);
    const int tag_base = 7200;
    const int tag[2]   = {tag_base + 2 * dir, tag_base + 2 * dir + 1};  // sent to lower / upper
    MPI_Request req[4];
    int nreq = 0;
    for (int side = 0; side < 2; side++) {
      const int face = 2 * dir + side;
      if (s.flags[face] != 5) continue;
    #ifdef MPI_GPU
      Real *send = buf.dev[side], *recv = buf.dev[2 + side];
    #else
      GPU_Error_Check(cudaMemcpy(buf.host[side], buf.dev[side], n_slab * sizeof(Real), cudaMemcpyDeviceToHost));
      Real *send = buf.host[side], *recv = buf.host[2 + side];
    #endif
      // what arrives through my lower face was sent "upward" by my lower neighbor, and vice versa
      MPI_Irecv(recv, n_slab, MPI_CHREAL, source[face], tag[1 - side], world, &req[nreq++]);
      MPI_Isend(send, n_slab, MPI_CHREAL, dest[face], tag[side], world, &req[nreq++]);
    }
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    for (int side = 0; side < 2; side++) {
      const int face = 2 * dir + side;
      if (s.flags[face] != 5) continue;
    #ifndef MPI_GPU
      GPU_Error_Check(
          cudaMemcpy(buf.dev[2 + side], buf.host[2 + side], n_slab * sizeof(Real), cudaMemcpyHostToDevice));
    #endif
      // from the lower neighbor -> my lowest active slab; from the upper -> my highest
      Launch_Slab(dev_rs, buf.dev[2 + side], s, dir, side == 0 ? ng : n - 2 * ng, 1);
    }
  #endif  // MPI_CHOLLA
  }
}

}  // namespace rt_sources

#endif  // RT
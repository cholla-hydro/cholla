#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "cuda_boundaries.h"

__device__ int FindIndex(int ig, int nx, int flag, int face, int n_ghost, Real *a);

__device__ int SetBoundaryMapping(int ig, int jg, int kg, Real *a, int flags[], int nx, int ny, int nz, int n_ghost);

__global__ void PackBuffers3DKernel(Real *buffer, Real *c_head, int isize, int jsize, int ksize, int nx, int ny,
                                    int idxoffset, int buffer_ncells, int n_fields, int n_cells)
{
  int id, i, j, k, idx, ii;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= buffer_ncells) {
    return;
  }
  k   = id / (isize * jsize);
  j   = (id - k * isize * jsize) / isize;
  i   = id - k * isize * jsize - j * isize;
  idx = i + (j + k * ny) * nx + idxoffset;
  // idxoffset contains offset terms from
  // idx = (i+ioffset) + (j+joffset)*H.nx + (k+koffset)*H.nx*H.ny;
  for (ii = 0; ii < n_fields; ii++) {
    *(buffer + id + ii * buffer_ncells) = c_head[idx + ii * n_cells];
  }
}

void PackBuffers3D(Real *buffer, Real *c_head, int nx, int ny, int n_fields, int n_cells, int idxoffset, int isize,
                   int jsize, int ksize)
{
  int buffer_ncells = isize * jsize * ksize;
  dim3 dim1dGrid((buffer_ncells + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(PackBuffers3DKernel, dim1dGrid, dim1dBlock, 0, 0, buffer, c_head, isize, jsize, ksize, nx, ny,
                     idxoffset, buffer_ncells, n_fields, n_cells);
  GPU_Error_Check(cudaDeviceSynchronize());
}

__global__ void UnpackBuffers3DKernel(Real *buffer, Real *c_head, int isize, int jsize, int ksize, int nx, int ny,
                                      int idxoffset, int buffer_ncells, int n_fields, int n_cells)
{
  int id, i, j, k, idx, ii;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= buffer_ncells) {
    return;
  }
  k   = id / (isize * jsize);
  j   = (id - k * isize * jsize) / isize;
  i   = id - k * isize * jsize - j * isize;
  idx = i + (j + k * ny) * nx + idxoffset;
  for (ii = 0; ii < n_fields; ii++) {
    c_head[idx + ii * n_cells] = *(buffer + id + ii * buffer_ncells);
  }
}

void UnpackBuffers3D(Real *buffer, Real *c_head, int nx, int ny, int n_fields, int n_cells, int idxoffset, int isize,
                     int jsize, int ksize)
{
  // void UnpackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize,
  // int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int
  // n_cells){
  int buffer_ncells = isize * jsize * ksize;
  dim3 dim1dGrid((buffer_ncells + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(UnpackBuffers3DKernel, dim1dGrid, dim1dBlock, 0, 0, buffer, c_head, isize, jsize, ksize, nx, ny,
                     idxoffset, buffer_ncells, n_fields, n_cells);
}

__global__ void SetGhostCellsKernel(Real *c_head, int nx, int ny, int nz, int n_fields, int n_cells, int n_ghost,
                                    int f0, int f1, int f2, int f3, int f4, int f5, int isize, int jsize, int ksize,
                                    int imin, int jmin, int kmin, int dir)
{
  int id, i, j, k, gidx, idx, ii;
  Real a[3]    = {1., 1., 1.};
  int flags[6] = {f0, f1, f2, f3, f4, f5};

  // using thread ID calculate which ghost cell this is
  // calculate which real cell using SetBoundaryMapping
  // and FindIndex
  // set variables, flipping momentum sign when needed
  // if flags[dir] == 3 correct the energy

  // calculate ghost cell ID and i,j,k
  id = threadIdx.x + blockIdx.x * blockDim.x;

  // not true i,j,k but relative i,j,k
  k = id / (isize * jsize);
  j = (id - k * isize * jsize) / isize;
  i = id - k * isize * jsize - j * isize;
  if (id >= isize * jsize * ksize) {
    return;
  }
  // true i,j,k conversion
  i += imin;
  j += jmin;
  k += kmin;
  gidx = i + j * nx + k * nx * ny;

  // calculate idx (index of real cell) and a[:] for reflection
  idx = SetBoundaryMapping(i, j, k, &a[0], flags, nx, ny, nz, n_ghost);

  if (idx >= 0) {
    for (ii = 0; ii < n_fields; ii++) {
      c_head[gidx + ii * n_cells] = c_head[idx + ii * n_cells];
    }
    // momentum correction for reflection
    // these are set to -1 whenever ghost cells in a direction are in a
    // reflective boundary condition
    if (flags[0] == 2 || flags[1] == 2) {
      c_head[gidx + n_cells] *= a[0];
    }
    if (flags[2] == 2 || flags[3] == 2) {
      c_head[gidx + 2 * n_cells] *= a[1];
    }
    if (flags[4] == 2 || flags[5] == 2) {
      c_head[gidx + 3 * n_cells] *= a[2];
    }

#ifndef MHD
    // energy and momentum correction for transmission
    // Diode: only allow outflow
    if (flags[dir] == 3) {
      //
      int momdex = gidx + (dir / 2 + 1) * n_cells;
      // (X) Dir 0,1 -> Mom 1 -> c_head[gidx+1*n_cells]
      // (Y) Dir 2,3 -> Mom 2 -> c_head[gidx+2*n_cells]
      // (Z) Dir 4,5 -> Mom 3 -> c_head[gidx+3*n_cells]
      // If a momentum is set to 0, subtract its kinetic energy [gidx+4*n_cells]
      if (dir % 2 == 0) {
        // Direction 0,2,4 are left-side, don't allow inflow with positive
        // momentum
        if (c_head[momdex] > 0.0) {
          c_head[gidx + 4 * n_cells] -= 0.5 * (c_head[momdex] * c_head[momdex]) / c_head[gidx];
          c_head[momdex] = 0.0;
        }
      } else {
        // Direction 1,3,5 are right-side, don't allow inflow with negative
        // momentum
        if (c_head[momdex] < 0.0) {
          c_head[gidx + 4 * n_cells] -= 0.5 * (c_head[momdex] * c_head[momdex]) / c_head[gidx];
          c_head[momdex] = 0.0;
        }
      }
    }   // end energy correction for transmissive boundaries
#endif  // not MHD
  }     // end idx>=0
}  // end function

void SetGhostCells(Real *c_head, int nx, int ny, int nz, int n_fields, int n_cells, int n_ghost, int flags[], int isize,
                   int jsize, int ksize, int imin, int jmin, int kmin, int dir)
{
  dim3 dim1dGrid((isize * jsize * ksize + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(SetGhostCellsKernel, dim1dGrid, dim1dBlock, 0, 0, c_head, nx, ny, nz, n_fields, n_cells, n_ghost,
                     flags[0], flags[1], flags[2], flags[3], flags[4], flags[5], isize, jsize, ksize, imin, jmin, kmin,
                     dir);
}

__device__ int SetBoundaryMapping(int ig, int jg, int kg, Real *a, int flags[], int nx, int ny, int nz, int n_ghost)
{
  // nx, ny, nz, n_ghost
  /* 1D */
  int ir, jr, kr, idx;
  ir = jr = kr = idx = 0;
  if (nx > 1) {
    // set index on -x face
    if (ig < n_ghost) {
      ir = FindIndex(ig, nx, flags[0], 0, n_ghost, &a[0]);
    }
    // set index on +x face
    else if (ig >= nx - n_ghost) {
      ir = FindIndex(ig, nx, flags[1], 1, n_ghost, &a[0]);
    }
    // set i index for multi-D problems
    else {
      ir = ig;
    }

    // if custom x boundaries are needed, set index to -1 and return
    if (ir < 0) {
      return idx = -1;
    }

    // otherwise add i index to ghost cell mapping
    idx += ir;
  }

  /* 2D */
  if (ny > 1) {
    // set index on -y face
    if (jg < n_ghost) {
      jr = FindIndex(jg, ny, flags[2], 0, n_ghost, &a[1]);
    }
    // set index on +y face
    else if (jg >= ny - n_ghost) {
      jr = FindIndex(jg, ny, flags[3], 1, n_ghost, &a[1]);
    }
    // set j index for multi-D problems
    else {
      jr = jg;
    }

    // if custom y boundaries are needed, set index to -1 and return
    if (jr < 0) {
      return idx = -1;
    }

    // otherwise add j index to ghost cell mapping
    idx += nx * jr;
  }

  /* 3D */
  if (nz > 1) {
    // set index on -z face
    if (kg < n_ghost) {
      kr = FindIndex(kg, nz, flags[4], 0, n_ghost, &a[2]);
    }
    // set index on +z face
    else if (kg >= nz - n_ghost) {
      kr = FindIndex(kg, nz, flags[5], 1, n_ghost, &a[2]);
    }
    // set k index for multi-D problems
    else {
      kr = kg;
    }

    // if custom z boundaries are needed, set index to -1 and return
    if (kr < 0) {
      return idx = -1;
    }

    // otherwise add k index to ghost cell mapping
    idx += nx * ny * kr;
  }
  return idx;
}

__device__ int FindIndex(int ig, int nx, int flag, int face, int n_ghost, Real *a)
{
  int id;

  // lower face
  if (face == 0) {
    switch (flag) {
      // periodic
      case 1:
        id = ig + nx - 2 * n_ghost;
        break;
      // reflective
      case 2:
        id   = 2 * n_ghost - ig - 1;
        *(a) = -1.0;
        break;
      // transmissive
      case 3:
        id = n_ghost;
        break;
      // custom
      case 4:
        id = -1;
        break;
      // MPI
      case 5:
        id = ig;
        break;
      // default is periodic
      default:
        id = ig + nx - 2 * n_ghost;
    }
  }
  // upper face
  else {
    switch (flag) {
      // periodic
      case 1:
        id = ig - nx + 2 * n_ghost;
        break;
      // reflective
      case 2:
        id   = 2 * (nx - n_ghost) - ig - 1;
        *(a) = -1.0;
        break;
      // transmissive
      case 3:
        id = nx - n_ghost - 1;
        break;
      // custom
      case 4:
        id = -1;
        break;
      // MPI
      case 5:
        id = ig;
        break;
      // default is periodic
      default:
        id = ig - nx + 2 * n_ghost;
    }
  }
  return id;
}

__global__ void Wind_Boundary_kernel(Real *c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off,
                                     int y_off, int z_off, Real dx, Real dy, Real dz, Real xbound, Real ybound,
                                     Real zbound, Real gamma, Real t)
{
  int id, xid, yid, zid, gid;
  Real n_0, T_0;
  Real mu = 0.6;
  Real vx, vy, vz, d_0, P_0;

  n_0 = 1e-2;  // same value as n_bg in cloud initial condition function (cm^-3)
  T_0 = 3e6;   // same value as T_bg in cloud initial condition function (K)

  // same values as rho_bg and p_bg in cloud initial condition function
  d_0 = n_0 * mu * MP / DENSITY_UNIT;
  P_0 = n_0 * KB * T_0 / PRESSURE_UNIT;

  vx = 100 * TIME_UNIT / KPC;  // km/s * (cholla unit conversion)
  vy = 0.0;
  vz = 0.0;

  // calculate ghost cell ID and i,j,k in GPU grid
  id = threadIdx.x + blockIdx.x * blockDim.x;

  // not true i,j,k but relative i,j,k in the GPU grid
  cuda_utilities::compute3DIndices(id, n_ghost, ny, xid, yid, zid);

  // map thread id to ghost cell id
  xid += 0;  // -x boundary
  gid = xid + yid * nx + zid * nx * ny;

  if (xid <= n_ghost && xid < nx && yid < ny && zid < nz) {
    // set conserved variables
    c_device[gid]               = d_0;
    c_device[gid + 1 * n_cells] = vx * d_0;
    c_device[gid + 2 * n_cells] = vy * d_0;
    c_device[gid + 3 * n_cells] = vz * d_0;
    c_device[gid + 4 * n_cells] = P_0 / (gamma - 1.0) + 0.5 * d_0 * (vx * vx + vy * vy + vz * vz);
  }
  __syncthreads();
}

__global__ void Noh_Boundary_kernel(Real *c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off,
                                    int y_off, int z_off, Real dx, Real dy, Real dz, Real xbound, Real ybound,
                                    Real zbound, Real gamma, Real t)
{
  int id, xid, yid, zid, gid;
  Real x_pos, y_pos, z_pos, r;
  Real vx, vy, vz, d_0, P_0;

  d_0 = 1.0;
  P_0 = 1.0e-6;

  // calculate ghost cell ID and i,j,k in GPU grid
  id = threadIdx.x + blockIdx.x * blockDim.x;

  int isize, jsize;
  // int ksize;

  // +x boundary first
  isize = n_ghost;
  jsize = ny;
  // ksize = nz;

  // not true i,j,k but relative i,j,k in the GPU grid
  zid = id / (isize * jsize);
  yid = (id - zid * isize * jsize) / isize;
  xid = id - zid * isize * jsize - yid * isize;

  // map thread id to ghost cell id
  xid += nx - n_ghost;  // +x boundary
  gid = xid + yid * nx + zid * nx * ny;

  if (xid >= nx - n_ghost && xid < nx && yid < ny && zid < nz) {
    // use the subgrid offset and global boundaries to calculate absolute
    // positions on the grid
    x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;
    y_pos = (y_off + yid - n_ghost + 0.5) * dy + ybound;
    z_pos = (z_off + zid - n_ghost + 0.5) * dz + zbound;

    // for 2D calculate polar r
    if (nz == 1) {
      r = sqrt(x_pos * x_pos + y_pos * y_pos);
      // for 3D calculate spherical r
    } else {
      r = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);
    }

    // calculate the velocities
    vx = -x_pos / r;
    vy = -y_pos / r;
    if (nz > 1) {
      vz = -z_pos / r;
    } else {
      vz = 0;
    }
    // set the conserved quantities
    if (nz > 1) {
      c_device[gid] = d_0 * (1.0 + t / r) * (1.0 + t / r);
    } else {
      c_device[gid] = d_0 * (1.0 + t / r);
    }
    c_device[gid + 1 * n_cells] = vx * c_device[gid];
    c_device[gid + 2 * n_cells] = vy * c_device[gid];
    c_device[gid + 3 * n_cells] = vz * c_device[gid];
    c_device[gid + 4 * n_cells] = P_0 / (gamma - 1.0) + 0.5 * c_device[gid];
  }

  // +y boundary next
  isize = nx;
  jsize = n_ghost;
  // ksize = nz;

  // not true i,j,k but relative i,j,k
  zid = id / (isize * jsize);
  yid = (id - zid * isize * jsize) / isize;
  xid = id - zid * isize * jsize - yid * isize;

  // map thread id to ghost cell id
  yid += ny - n_ghost;  // +y boundary
  gid = xid + yid * nx + zid * nx * ny;

  if (xid < nx && yid >= ny - n_ghost && yid < ny && zid < nz) {
    // use the subgrid offset and global boundaries to calculate absolute
    // positions on the grid
    x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;
    y_pos = (y_off + yid - n_ghost + 0.5) * dy + ybound;
    z_pos = (z_off + zid - n_ghost + 0.5) * dz + zbound;

    // for 2D calculate polar r
    if (nz == 1) {
      r = sqrt(x_pos * x_pos + y_pos * y_pos);
      // for 3D, calculate spherical r
    } else {
      r = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);
    }

    // calculate the velocities
    vx = -x_pos / r;
    vy = -y_pos / r;
    if (nz > 1) {
      vz = -z_pos / r;
    } else {
      vz = 0;
    }
    // set the conserved quantities
    if (nz > 1) {
      c_device[gid] = d_0 * (1.0 + t / r) * (1.0 + t / r);
    } else {
      c_device[gid] = d_0 * (1.0 + t / r);
    }
    c_device[gid + 1 * n_cells] = vx * c_device[gid];
    c_device[gid + 2 * n_cells] = vy * c_device[gid];
    c_device[gid + 3 * n_cells] = vz * c_device[gid];
    c_device[gid + 4 * n_cells] = P_0 / (gamma - 1.0) + 0.5 * c_device[gid];
  }
  __syncthreads();

  // +z boundary last (only if 3D)
  if (nz == 1) {
    return;
  }

  isize = nx;
  jsize = ny;
  // ksize = n_ghost;

  // not true i,j,k but relative i,j,k
  zid = id / (isize * jsize);
  yid = (id - zid * isize * jsize) / isize;
  xid = id - zid * isize * jsize - yid * isize;

  // map thread id to ghost cell id
  zid += nz - n_ghost;  // +z boundary
  gid = xid + yid * nx + zid * nx * ny;

  if (xid < nx && yid < ny && zid >= nz - n_ghost && zid < nz) {
    // use the subgrid offset and global boundaries to calculate absolute
    // positions on the grid
    x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;
    y_pos = (y_off + yid - n_ghost + 0.5) * dy + ybound;
    z_pos = (z_off + zid - n_ghost + 0.5) * dz + zbound;

    // for 2D calculate polar r
    if (nz == 1) {
      r = sqrt(x_pos * x_pos + y_pos * y_pos);
      // for 3D, calculate spherical r
    } else {
      r = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);
    }

    // calculate the velocities
    vx = -x_pos / r;
    vy = -y_pos / r;
    if (nz > 1) {
      vz = -z_pos / r;
    } else {
      vz = 0;
    }
    // set the conserved quantities
    if (nz > 1) {
      c_device[gid] = d_0 * (1.0 + t / r) * (1.0 + t / r);
    } else {
      c_device[gid] = d_0 * (1.0 + t / r);
    }
    c_device[gid + 1 * n_cells] = vx * c_device[gid];
    c_device[gid + 2 * n_cells] = vy * c_device[gid];
    c_device[gid + 3 * n_cells] = vz * c_device[gid];
    c_device[gid + 4 * n_cells] = P_0 / (gamma - 1.0) + 0.5 * c_device[gid];
  }
}

void Wind_Boundary_CUDA(Real *c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off, int y_off,
                        int z_off, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real gamma, Real t)
{
  // determine the size of the grid to launch
  // need at least as many threads as the largest boundary face
  // current implementation assumes the test is run on a cube...
  int isize, jsize, ksize;
  isize = n_ghost;
  jsize = ny;
  ksize = nz;

  dim3 dim1dGrid((isize * jsize * ksize + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  // launch the boundary kernel
  hipLaunchKernelGGL(Wind_Boundary_kernel, dim1dGrid, dim1dBlock, 0, 0, c_device, nx, ny, nz, n_cells, n_ghost, x_off,
                     y_off, z_off, dx, dy, dz, xbound, ybound, zbound, gamma, t);
}

void Noh_Boundary_CUDA(Real *c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off, int y_off,
                       int z_off, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real gamma, Real t)
{
  // determine the size of the grid to launch
  // need at least as many threads as the largest boundary face
  // current implementation assumes the test is run on a cube...
  int isize, jsize, ksize;
  isize = n_ghost;
  jsize = ny;
  ksize = nz;

  dim3 dim1dGrid((isize * jsize * ksize + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  // launch the boundary kernel
  hipLaunchKernelGGL(Noh_Boundary_kernel, dim1dGrid, dim1dBlock, 0, 0, c_device, nx, ny, nz, n_cells, n_ghost, x_off,
                     y_off, z_off, dx, dy, dz, xbound, ybound, zbound, gamma, t);
}
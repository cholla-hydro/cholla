#include "../utils/gpu.hpp"
#include "field_operations.h"

/*! \fn void Rescale_Field_GPU(Real *d_x, Real A, int nx, int ny, int nz, int n_ghost)
 *  \brief Multiply one field a multiplicative constant */
__global__ void Rescale_Field_GPU(Real *d_x, const Real A, int nx, int ny, int nz, int n_ghost)
{
  // Rescale a field by a multiplicative constant

  // determine the cell location
  int id, xid, yid, zid;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;

  // only real cells participate
  if ((xid >= 0) and (xid < nx) and (yid >= 0) and (yid < ny) and (zid >= 0) and (zid < nz)) {  // all cells are real
    // rescale the field
    d_x[id] *= A;
  }
}

/*! \fn void Field_Elementwise_Product_GPU(Real *d_x, Real *d_y, int nx, int ny, int nz, int n_ghost)
 *  \brief Multiply one field elementwise by another */
inline __device__ void Field_Elementwise_Product_GPU(Real *d_x, Real *d_y, int nx, int ny, int nz, int n_ghost)
{
  // determine the cell location
  int id, xid, yid, zid;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;

  // only real cells participate
  /*if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {*/
  if ((xid >= 0) & (xid < nx) & (yid >= 0) & (yid < ny) & (zid >= 0) & (zid < nz)) {  // all cells are real

    // rescale x by y
    d_x[id] *= d_y[id];
  }
}

/*! \fn void FFT_Populate_Wavevectors_GPU(Real *d_kx, Real *d_ky, Real *d_kz, Real *d_kk, int n_cells, int n_ghost)
 *  \brief Multiply one field elementwise by another */
/*
inline __device__ void FFT_Populate_Wavevectors_GPU(Real *d_kx, Real *d_ky, Real *d_kz, Real *d_kk, int n_cells, int
n_ghost)
{
        // determine the cell location
        int id;

        // get a global thread ID
        id = threadIdx.x + blockIdx.x * blockDim.x;

        // only real cells participate
        if (id > n_ghost - 1 && id < n_cells - n_ghost) {

                d_kx[id] = 0;
                d_ky[id] = 0;
                d_kz[id] = 0;
        }
}
*/

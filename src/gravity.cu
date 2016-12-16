/*! \file gravity.cu
 *  \brief Function definitions for the gravity source corrections.*/

#ifdef CUDA

#include<cuda.h>
#include<math.h>
#include<stdio.h>
#include"global.h"
#include"global_cuda.h"
#include"gravity.h"


/*! \fn Correct_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_Q_Ly, Real *dev_Q_Ry, int nx, int ny, Real dt)
 *  \brief Correct the input states to the Riemann solver based on gravitational source terms. */
__global__ void Correct_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_Q_Ly, Real *dev_Q_Ry, int nx, int ny, Real dt)
{
  int n_cells = nx*ny;

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId * blockDim.x;
  int yid = tid / nx;
  int xid = tid - yid*nx;
  int id = xid + yid*nx;

  Real d, vx, vy, gx, gy;
  gx = 0.0;
  gy = -0.1;

  if (xid < nx && yid < ny)
  {
    d = dev_Q_Lx[id];
    vx = dev_Q_Lx[  n_cells + id];
    vy = dev_Q_Lx[2*n_cells + id];
    dev_Q_Lx[  n_cells + id] += dt*d*gx;
    dev_Q_Lx[2*n_cells + id] += dt*d*gy;
    dev_Q_Lx[4*n_cells + id] += dt*d*(vx*gx + vy*gy);
    d = dev_Q_Rx[id];
    vx = dev_Q_Rx[  n_cells + id];
    vy = dev_Q_Rx[2*n_cells + id];
    dev_Q_Rx[  n_cells + id] += dt*d*gx;
    dev_Q_Rx[2*n_cells + id] += dt*d*gy;
    dev_Q_Rx[4*n_cells + id] += dt*d*(vx*gx + vy*gy);
    d = dev_Q_Ly[id];
    vx = dev_Q_Ly[  n_cells + id];
    vy = dev_Q_Ly[2*n_cells + id];
    dev_Q_Ly[  n_cells + id] += dt*d*gx;
    dev_Q_Ly[2*n_cells + id] += dt*d*gy;
    dev_Q_Ly[4*n_cells + id] += dt*d*(vx*gx + vy*gy);
    d = dev_Q_Ry[id];
    vx = dev_Q_Ry[  n_cells + id];
    vy = dev_Q_Ry[2*n_cells + id];
    dev_Q_Ry[  n_cells + id] += dt*d*gx;
    dev_Q_Ry[2*n_cells + id] += dt*d*gy;
    dev_Q_Ry[4*n_cells + id] += dt*d*(vx*gx + vy*gy);
  }

}

#endif // CUDA


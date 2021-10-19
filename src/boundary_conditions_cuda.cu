#include "grid3D.h"
#include "boundary_conditions_cuda.cuh"
#include "global_cuda.h"

#ifdef DEVICE_COMM

// This is a CUDA ported version of the `Set_Ghost_Cells` function

void Grid3D::Set_Ghost_Cells_Cuda(int imin[3], int imax[3], Real a[3], int flags[6], int dir) {

  int threads = (imax[0] - imin[0]) * (imax[1] - imin[1]) * (imax[2] - imin[2]);
  int block = 256;
  int grid = (threads + block - 1) / block;

  Set_Ghost_Cells_Cuda_Arg arg;
  for (int d = 0; d < 3; d++) {
    arg.imin[d] = imin[d];
    arg.imax[d] = imax[d];
  }
  for (int d = 0; d < 6; d++) {
    arg.flags[d] = flags[d];
  }
  arg.dir = dir;

  Set_Ghost_Cells_Cuda_Kernel<<<grid, block>>>(dev_conserved, H, arg);
  CudaSafeCall(cudaStreamSynchronize(0));

}

#endif


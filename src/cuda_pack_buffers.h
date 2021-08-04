#ifdef CUDA
#include"gpu.hpp"
#include"global.h"
#include"global_cuda.h"




__device__ void PackBuffers3DKernel(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

void PackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);
#endif

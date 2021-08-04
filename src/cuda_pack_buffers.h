#ifdef CUDA
#include"global.h"
__device__ void PackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

#endif

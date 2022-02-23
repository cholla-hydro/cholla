#ifdef CUDA
#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"

void PackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

void UnpackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

void PackGhostCells(Real * c_head,
		    int nx, int ny, int nz, int n_fields, int n_cells, int n_ghost, int flags[],
		    int isize, int jsize, int ksize,
		    int imin, int jmin, int kmin, int dir);

#endif

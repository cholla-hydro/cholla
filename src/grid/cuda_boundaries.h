#ifdef CUDA
  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../utils/gpu.hpp"

// void PackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int
// ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);
void PackBuffers3D(Real* buffer, Real* c_head, int nx, int ny, int n_fields, int n_cells, int idxoffset, int isize,
                   int jsize, int ksize);

void UnpackBuffers3D(Real* buffer, Real* c_head, int nx, int ny, int n_fields, int n_cells, int idxoffset, int isize,
                     int jsize, int ksize);
// void UnpackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int
// ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

void SetGhostCells(Real* c_head, int nx, int ny, int nz, int n_fields, int n_cells, int n_ghost, int flags[], int isize,
                   int jsize, int ksize, int imin, int jmin, int kmin, int dir);

void Wind_Boundary_CUDA(Real* c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off, int y_off,
                        int z_off, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real gamma,
                        Real t);

void Noh_Boundary_CUDA(Real* c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off, int y_off,
                       int z_off, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real gamma, Real t);

#endif

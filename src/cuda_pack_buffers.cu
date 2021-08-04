#include"gpu.hpp"
#include"global.h"
#include"global_cuda.h"


void PackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells){
  dim3 dim1dGrid((isize*jsize*ksize+TPB-1)/TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1); 
  hipLaunchKernelGGL(PackBuffers3DKernel,dim1dGrid,dim1dBlock,0,0,send_buffer_3d,c_head,isize,jsize,ksize,H.nx,H.ny,idxoffset,offset,H.n_fields,H.n_cells);
}

  
__device__ void PackBuffers3DKernel(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells)
{
  int id,i,j,k,idx,ii;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= offset){
    return;
  }
  k = id/(isize*jsize);
  j = (id - k*isize*jsize)/isize;
  i = id - k*isize*jsize - j*isize;
  idx  = i + (j+k*ny)*nx + idxoffset;
  //(i+ioffset) + (j+joffset)*H.nx + (k+koffset)*H.nx*H.ny;
  // gidx = id 
  //gidx = i+(j+k*jsize)*isize; 
  //gidx = i + j*isize + k*ijsize;//i+(j+k*jsize)*isize
  for (ii=0; ii<n_fields; ii++) {
    *(buffer + id + ii*offset) = c_head[idx + ii*n_cells]; 
  }
  
}

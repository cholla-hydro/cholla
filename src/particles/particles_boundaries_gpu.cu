#if defined(PARTICLES) && defined(PARTICLES_GPU)

#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include <iostream>
#include"../io.h"
#include"../global.h"
#include"../global_cuda.h"
#include "particles_3D.h"
#include "../grid3D.h"


__global__ void Set_Particles_Boundary_Kernel( int side, part_int_t n_local,  Real *pos_dev, Real d_min, Real d_max, Real d_length ){
  
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_local) return;
  
  Real pos;
  pos = pos_dev[tid];
  
  if ( side == 0 ){
    if ( pos < d_min ) pos += d_length;
  }
  
  if ( side == 1 ){
    if ( pos >= d_max ) pos -= d_length;
  }
  
  pos_dev[tid] = pos;
  
}


void Grid3D::Set_Particles_Boundary_GPU( int dir, int side ){
  
  Real d_min, d_max, L;
  Real *pos_dev;
  if ( dir == 0 ){
    d_min = Particles.G.zMin;
    d_max = Particles.G.zMax;
    pos_dev = Particles.pos_x_dev;
  }
  if ( dir == 1 ){
    d_min = Particles.G.yMin;
    d_max = Particles.G.yMax;
    pos_dev = Particles.pos_y_dev;
  }
  if ( dir == 2 ){
    d_min = Particles.G.zMin;
    d_max = Particles.G.zMax;
    pos_dev = Particles.pos_z_dev;
  }
  
  L = d_max - d_min;
  
  // set values for GPU kernels
  int ngrid =  (Particles.n_local + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  Set_Particles_Boundary_Kernel<<<dim1dGrid,dim1dBlock>>>( side, Particles.n_local, pos_dev, d_min, d_max, L  );
  CudaCheckError();

}


#endif //PARTICLES
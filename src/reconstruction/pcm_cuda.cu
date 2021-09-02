/*! \file pcm_cuda.cu
 *  \brief Definitions of the piecewise constant reconstruction functions */
#ifdef CUDA

#include "../utils/gpu.hpp"
#include <math.h>
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/pcm_cuda.h"


__global__ void PCM_Reconstruction_1D(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int n_cells, int n_ghost, Real gamma, int n_fields)
{
    
  // declare conserved variables for each stencil
  // these will be placed into registers for each thread
  Real d, mx, my, mz, E;

  #ifdef DE
  Real ge;
  #endif

  #ifdef SCALAR
  Real scalar[NSCALARS];
  #endif

  // get a global thread ID
  int xid = threadIdx.x + blockIdx.x*blockDim.x;
  int id;


  // threads corresponding to real cells plus one ghost cell do the calculation
  if (xid < n_cells-1)
  {
    // retrieve appropriate conserved variables
    id = xid;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id]; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id]; 
    #endif

    // send values back from the kernel
    dev_bounds_L[            id] = d;
    dev_bounds_L[  n_cells + id] = mx;
    dev_bounds_L[2*n_cells + id] = my;
    dev_bounds_L[3*n_cells + id] = mz;
    dev_bounds_L[4*n_cells + id] = E;    
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_L[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_L[(n_fields-1)*n_cells + id] = ge;    
    #endif

    // retrieve appropriate conserved variables
    id = xid+1;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id]; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id]; 
    #endif

    // send values back from the kernel
    id = xid;
    dev_bounds_R[            id] = d;
    dev_bounds_R[  n_cells + id] = mx;
    dev_bounds_R[2*n_cells + id] = my;
    dev_bounds_R[3*n_cells + id] = mz;
    dev_bounds_R[4*n_cells + id] = E;
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_R[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_R[(n_fields-1)*n_cells + id] = ge;
    #endif
    
  }

}


__global__ void PCM_Reconstruction_2D(Real *dev_conserved, Real *dev_bounds_Lx, Real *dev_bounds_Rx, Real *dev_bounds_Ly, Real *dev_bounds_Ry, int nx, int ny, int n_ghost, Real gamma, int n_fields)
{
    
  // declare conserved variables for each stencil
  // these will be placed into registers for each thread
  Real d, mx, my, mz, E;
  #ifdef DE
  Real ge;
  #endif
  #ifdef SCALAR
  Real scalar[NSCALARS];
  #endif
  
  int n_cells = nx*ny;

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId * blockDim.x;
  int yid = tid / nx;
  int xid = tid - yid*nx;
  int id;

  // threads corresponding to real cells plus one ghost cell do the calculation
  // x direction
  if (xid < nx-1 && yid < ny)
  {
    // retrieve appropriate conserved variables
    id = xid + yid*nx;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id]; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif

    // send values back from the kernel
    dev_bounds_Lx[            id] = d;
    dev_bounds_Lx[  n_cells + id] = mx;
    dev_bounds_Lx[2*n_cells + id] = my;
    dev_bounds_Lx[3*n_cells + id] = mz;
    dev_bounds_Lx[4*n_cells + id] = E; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Lx[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Lx[(n_fields-1)*n_cells + id] = ge; 
    #endif

    // retrieve appropriate conserved variables
    id = xid+1 + yid*nx;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id];    
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];    
    #endif

    // send values back from the kernel
    id = xid + yid*nx;
    dev_bounds_Rx[            id] = d;
    dev_bounds_Rx[  n_cells + id] = mx;
    dev_bounds_Rx[2*n_cells + id] = my;
    dev_bounds_Rx[3*n_cells + id] = mz;
    dev_bounds_Rx[4*n_cells + id] = E;    
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Rx[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Rx[(n_fields-1)*n_cells + id] = ge;    
    #endif
  }
  
  // y direction
  if (xid < nx && yid < ny-1)
  {
    // retrieve appropriate conserved variables
    id = xid + yid*nx;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id]; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id]; 
    #endif

    // send values back from the kernel
    dev_bounds_Ly[            id] = d;
    dev_bounds_Ly[  n_cells + id] = mx;
    dev_bounds_Ly[2*n_cells + id] = my;
    dev_bounds_Ly[3*n_cells + id] = mz;
    dev_bounds_Ly[4*n_cells + id] = E; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Ly[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Ly[(n_fields-1)*n_cells + id] = ge; 
    #endif

    // retrieve appropriate conserved variables
    id = xid + (yid+1)*nx;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id];
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif

    // send values back from the kernel
    id = xid + yid*nx;
    dev_bounds_Ry[            id] = d;
    dev_bounds_Ry[  n_cells + id] = mx;
    dev_bounds_Ry[2*n_cells + id] = my;
    dev_bounds_Ry[3*n_cells + id] = mz;
    dev_bounds_Ry[4*n_cells + id] = E;    
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Ry[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Ry[(n_fields-1)*n_cells + id] = ge;    
    #endif
  }

}


__global__ void PCM_Reconstruction_3D(Real *dev_conserved, 
                                      Real *dev_bounds_Lx, Real *dev_bounds_Rx,
                                      Real *dev_bounds_Ly, Real *dev_bounds_Ry,
                                      Real *dev_bounds_Lz, Real *dev_bounds_Rz,
                                      int nx, int ny, int nz, int n_ghost, Real gamma, int n_fields)
{
    
  // declare conserved variables for each stencil
  // these will be placed into registers for each thread
  Real d, mx, my, mz, E;
  #ifdef DE
  Real ge;
  #endif
  #ifdef SCALAR
  Real scalar[NSCALARS];
  #endif
  
  
  int n_cells = nx*ny*nz;

  // get a thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;
  int id = xid + yid*nx + zid*nx*ny;

  // x direction
  if (xid < nx-1 && yid < ny && zid < nz)
  {
    // retrieve appropriate conserved variables
    id = xid + yid*nx + zid*nx*ny;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id]; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id]; 
    #endif

    // send values back from the kernel
    dev_bounds_Lx[            id] = d;
    dev_bounds_Lx[  n_cells + id] = mx;
    dev_bounds_Lx[2*n_cells + id] = my;
    dev_bounds_Lx[3*n_cells + id] = mz;
    dev_bounds_Lx[4*n_cells + id] = E; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Lx[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Lx[(n_fields-1)*n_cells + id] = ge; 
    #endif

    // retrieve appropriate conserved variables
    id = xid+1 + yid*nx + zid*nx*ny;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id];    
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];    
    #endif

    // send values back from the kernel
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_Rx[            id] = d;
    dev_bounds_Rx[  n_cells + id] = mx;
    dev_bounds_Rx[2*n_cells + id] = my;
    dev_bounds_Rx[3*n_cells + id] = mz;
    dev_bounds_Rx[4*n_cells + id] = E;    
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Rx[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Rx[(n_fields-1)*n_cells + id] = ge;    
    #endif
  }
  
  // y direction
  if (xid < nx && yid < ny-1 && zid < nz)
  {
    // retrieve appropriate conserved variables
    id = xid + yid*nx + zid*nx*ny;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id]; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id]; 
    #endif

    // send values back from the kernel
    dev_bounds_Ly[            id] = d;
    dev_bounds_Ly[  n_cells + id] = mx;
    dev_bounds_Ly[2*n_cells + id] = my;
    dev_bounds_Ly[3*n_cells + id] = mz;
    dev_bounds_Ly[4*n_cells + id] = E; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Ly[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Ly[(n_fields-1)*n_cells + id] = ge; 
    #endif

    // retrieve appropriate conserved variables
    id = xid + (yid+1)*nx + zid*nx*ny;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id];
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif

    // send values back from the kernel
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_Ry[            id] = d;
    dev_bounds_Ry[  n_cells + id] = mx;
    dev_bounds_Ry[2*n_cells + id] = my;
    dev_bounds_Ry[3*n_cells + id] = mz;
    dev_bounds_Ry[4*n_cells + id] = E;    
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Ry[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Ry[(n_fields-1)*n_cells + id] = ge;    
    #endif
  }

  // z direction
  if (xid < nx && yid < ny && zid < nz-1)
  {
    // retrieve appropriate conserved variables
    id = xid + yid*nx + zid*nx*ny;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id]; 
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id]; 
    #endif

    // send values back from the kernel
    dev_bounds_Lz[            id] = d;
    dev_bounds_Lz[  n_cells + id] = mx;
    dev_bounds_Lz[2*n_cells + id] = my;
    dev_bounds_Lz[3*n_cells + id] = mz;
    dev_bounds_Lz[4*n_cells + id] = E;
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Lz[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Lz[(n_fields-1)*n_cells + id] = ge; 
    #endif
    
    // retrieve appropriate conserved variables
    id = xid + yid*nx + (zid+1)*nx*ny;
    d  = dev_conserved[            id]; 
    mx = dev_conserved[  n_cells + id]; 
    my = dev_conserved[2*n_cells + id]; 
    mz = dev_conserved[3*n_cells + id]; 
    E  = dev_conserved[4*n_cells + id];
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar[i] = dev_conserved[(5+i)*n_cells + id];
    }
    #endif
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif

    // send values back from the kernel
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_Rz[            id] = d;
    dev_bounds_Rz[  n_cells + id] = mx;
    dev_bounds_Rz[2*n_cells + id] = my;
    dev_bounds_Rz[3*n_cells + id] = mz;
    dev_bounds_Rz[4*n_cells + id] = E;    
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_Rz[(5+i)*n_cells + id] = scalar[i];
    }
    #endif
    #ifdef DE
    dev_bounds_Rz[(n_fields-1)*n_cells + id] = ge;    
    #endif

  }
}


#endif //CUDA

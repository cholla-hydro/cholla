/*! \file pcm_cuda.cu
 *  \brief Definitions of the piecewise constant reconstruction functions */
#ifdef CUDA

#include "../utils/gpu.hpp"
#include <math.h>
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/pcm_cuda.h"
#include "../utils/mhd_utilities.h"
#include "../utils/cuda_utilities.h"

__global__ void PCM_Reconstruction_1D(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int n_cells, int n_ghost, Real gamma, int n_fields)
{

  // declare conserved variables for each stencil
  // these will be placed into registers for each thread
  Real d, mx, my, mz, E;

  #ifdef DE
  Real ge;
  #endif  //DE

  #ifdef SCALAR
  Real scalar[NSCALARS];
  #endif  //SCALAR

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
    #endif  //SCALAR
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    dev_bounds_L[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    dev_bounds_R[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE

  }

}


__global__ void PCM_Reconstruction_2D(Real *dev_conserved, Real *dev_bounds_Lx, Real *dev_bounds_Rx, Real *dev_bounds_Ly, Real *dev_bounds_Ry, int nx, int ny, int n_ghost, Real gamma, int n_fields)
{

  // declare conserved variables for each stencil
  // these will be placed into registers for each thread
  Real d, mx, my, mz, E;
  #ifdef DE
  Real ge;
  #endif  //DE
  #ifdef SCALAR
  Real scalar[NSCALARS];
  #endif  //SCALAR

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
    #endif  //SCALAR
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    dev_bounds_Lx[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    dev_bounds_Rx[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE
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
    #endif  //SCALAR
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    dev_bounds_Ly[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif  //DE

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
    #endif  //SCALAR
    #ifdef DE
    dev_bounds_Ry[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE
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
  #ifdef SCALAR
  Real scalar[NSCALARS];
  #endif  //SCALAR

  int const n_cells = nx*ny*nz;

  // get a thread ID
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

  // Guard to avoid out of bounds threads
  if (xid < nx && yid < ny && zid < nz)
  {
    // ========================================
    // Retrieve appropriate conserved variables
    // ========================================
    Real const d  = dev_conserved[            id];
    Real const mx = dev_conserved[  n_cells + id];
    Real const my = dev_conserved[2*n_cells + id];
    Real const mz = dev_conserved[3*n_cells + id];
    Real const E  = dev_conserved[4*n_cells + id];
    #ifdef SCALAR
      for (int i=0; i<NSCALARS; i++)
      {
        scalar[i] = dev_conserved[(5+i)*n_cells + id];
      }
    #endif  //SCALAR
    #ifdef  MHD
      auto const [cellCenteredBx, cellCenteredBy, cellCenteredBz] = mhd::utils::cellCenteredMagneticFields(dev_conserved,
                                           id, xid, yid, zid, n_cells, nx, ny);
    #endif  //MHD
    #ifdef DE
      Real const ge = dev_conserved[(n_fields-1)*n_cells + id];
    #endif  //DE

    // ================================
    // Send values back from the kernel
    // ================================

    // Send the x+1/2 Left interface
    dev_bounds_Lx[            id] = d;
    dev_bounds_Lx[  n_cells + id] = mx;
    dev_bounds_Lx[2*n_cells + id] = my;
    dev_bounds_Lx[3*n_cells + id] = mz;
    dev_bounds_Lx[4*n_cells + id] = E;
    #ifdef SCALAR
      for (int i=0; i<NSCALARS; i++)
      {
        dev_bounds_Lx[(5+i)*n_cells + id] = scalar[i];
      }
    #endif  //SCALAR
    #ifdef  MHD
      dev_bounds_Lx[(grid_enum::Q_x_magnetic_y)*n_cells + id] = cellCenteredBy;
      dev_bounds_Lx[(grid_enum::Q_x_magnetic_z)*n_cells + id] = cellCenteredBz;
    #endif  //MHD
    #ifdef DE
      dev_bounds_Lx[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE

    // Send the y+1/2 Left interface
    dev_bounds_Ly[            id] = d;
    dev_bounds_Ly[  n_cells + id] = mx;
    dev_bounds_Ly[2*n_cells + id] = my;
    dev_bounds_Ly[3*n_cells + id] = mz;
    dev_bounds_Ly[4*n_cells + id] = E;
    #ifdef SCALAR
      for (int i=0; i<NSCALARS; i++)
      {
        dev_bounds_Ly[(5+i)*n_cells + id] = scalar[i];
      }
    #endif  //SCALAR
    #ifdef  MHD
      dev_bounds_Ly[(grid_enum::Q_y_magnetic_z)*n_cells + id] = cellCenteredBz;
      dev_bounds_Ly[(grid_enum::Q_y_magnetic_x)*n_cells + id] = cellCenteredBx;
    #endif  //MHD
    #ifdef DE
      dev_bounds_Ly[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE

    // Send the z+1/2 Left interface
    dev_bounds_Lz[            id] = d;
    dev_bounds_Lz[  n_cells + id] = mx;
    dev_bounds_Lz[2*n_cells + id] = my;
    dev_bounds_Lz[3*n_cells + id] = mz;
    dev_bounds_Lz[4*n_cells + id] = E;
    #ifdef SCALAR
      for (int i=0; i<NSCALARS; i++)
      {
        dev_bounds_Lz[(5+i)*n_cells + id] = scalar[i];
      }
    #endif  //SCALAR
    #ifdef  MHD
      dev_bounds_Lz[(grid_enum::Q_z_magnetic_x)*n_cells + id] = cellCenteredBx;
      dev_bounds_Lz[(grid_enum::Q_z_magnetic_y)*n_cells + id] = cellCenteredBy;
    #endif  //MHD
    #ifdef DE
      dev_bounds_Lz[(n_fields-1)*n_cells + id] = ge;
    #endif  //DE

    // Send the x-1/2 Right interface
    if (xid > 0)
    {
      id = cuda_utilities::compute1DIndex(xid-1, yid, zid, nx, ny);
      dev_bounds_Rx[            id] = d;
      dev_bounds_Rx[  n_cells + id] = mx;
      dev_bounds_Rx[2*n_cells + id] = my;
      dev_bounds_Rx[3*n_cells + id] = mz;
      dev_bounds_Rx[4*n_cells + id] = E;
      #ifdef SCALAR
        for (int i=0; i<NSCALARS; i++)
        {
          dev_bounds_Rx[(5+i)*n_cells + id] = scalar[i];
        }
      #endif  //SCALAR
      #ifdef  MHD
        dev_bounds_Rx[(grid_enum::Q_x_magnetic_y)*n_cells + id] = cellCenteredBy;
        dev_bounds_Rx[(grid_enum::Q_x_magnetic_z)*n_cells + id] = cellCenteredBz;
      #endif  //MHD
      #ifdef DE
        dev_bounds_Rx[(n_fields-1)*n_cells + id] = ge;
      #endif  //DE
    }

    if (yid > 0)
    {
      // Send the y-1/2 Right interface
      id = cuda_utilities::compute1DIndex(xid, yid-1, zid, nx, ny);
      dev_bounds_Ry[            id] = d;
      dev_bounds_Ry[  n_cells + id] = mx;
      dev_bounds_Ry[2*n_cells + id] = my;
      dev_bounds_Ry[3*n_cells + id] = mz;
      dev_bounds_Ry[4*n_cells + id] = E;
      #ifdef SCALAR
        for (int i=0; i<NSCALARS; i++)
        {
          dev_bounds_Ry[(5+i)*n_cells + id] = scalar[i];
        }
      #endif  //SCALAR
      #ifdef  MHD
        dev_bounds_Ry[(grid_enum::Q_y_magnetic_z)*n_cells + id] = cellCenteredBz;
        dev_bounds_Ry[(grid_enum::Q_y_magnetic_x)*n_cells + id] = cellCenteredBx;
      #endif  //MHD
      #ifdef DE
        dev_bounds_Ry[(n_fields-1)*n_cells + id] = ge;
      #endif  //DE
      }

    if (zid > 0)
    {
      // Send the z-1/2 Right interface
      id = cuda_utilities::compute1DIndex(xid, yid, zid-1, nx, ny);
      dev_bounds_Rz[            id] = d;
      dev_bounds_Rz[  n_cells + id] = mx;
      dev_bounds_Rz[2*n_cells + id] = my;
      dev_bounds_Rz[3*n_cells + id] = mz;
      dev_bounds_Rz[4*n_cells + id] = E;
      #ifdef SCALAR
        for (int i=0; i<NSCALARS; i++)
        {
          dev_bounds_Rz[(5+i)*n_cells + id] = scalar[i];
        }
      #endif  //SCALAR
      #ifdef  MHD
        dev_bounds_Rz[(grid_enum::Q_z_magnetic_x)*n_cells + id] = cellCenteredBx;
        dev_bounds_Rz[(grid_enum::Q_z_magnetic_y)*n_cells + id] = cellCenteredBy;
      #endif  //MHD
      #ifdef DE
        dev_bounds_Rz[(n_fields-1)*n_cells + id] = ge;
      #endif  //DE
    }
  }
}


#endif //CUDA

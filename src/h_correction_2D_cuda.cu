/*! \file h_correction_2D_cuda.cu
 *  \brief Functions definitions for the H correction kernels. 
           Written following Sanders et al. 1998. */
#ifdef CUDA
#ifdef H_CORRECTION 

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"h_correction_2D_cuda.h"



/*! \fn void calc_eta_x_2D(Real *dev_bounds_L, Real *dev_bounds_R, Real *eta_x, int nx, int ny, int n_ghost, Real gamma)
 *  \brief When passed the left and right boundary values at an interface, calculates
           the eta value for the interface according to the forumulation in Sanders et al, 1998. */
__global__ void calc_eta_x_2D(Real *dev_bounds_L, Real *dev_bounds_R, Real *eta_x, int nx, int ny, int n_ghost, Real gamma)
{
  int n_cells = nx*ny;

  // declare primative variables for each stencil
  // these will be placed into registers for each thread
  Real pl, pr, al, ar;

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId * blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;
  

  if (xid > n_ghost-2 && xid < nx-n_ghost && yid > n_ghost-2 && yid < ny-n_ghost+1)
  {
    // load the interface values into registers
    id = xid + yid*nx;
    pl  = (dev_bounds_L[4*n_cells + id] - 
      0.5*(dev_bounds_L[  n_cells+id]*dev_bounds_L[  n_cells+id] + 
           dev_bounds_L[2*n_cells+id]*dev_bounds_L[2*n_cells+id] + 
           dev_bounds_L[3*n_cells+id]*dev_bounds_L[3*n_cells+id])/dev_bounds_L[id]) * (gamma - 1.0);
    pl  = fmax(pl, (Real) 1.0e-20);
    pr  = (dev_bounds_R[4*n_cells + id] - 
      0.5*(dev_bounds_R[  n_cells+id]*dev_bounds_R[  n_cells+id] + 
           dev_bounds_R[2*n_cells+id]*dev_bounds_R[2*n_cells+id] + 
           dev_bounds_R[3*n_cells+id]*dev_bounds_R[3*n_cells+id])/dev_bounds_R[id]) * (gamma - 1.0);
    pr  = fmax(pr, (Real) 1.0e-20);

    al = sqrt(gamma*pl/dev_bounds_L[id]); 
    ar = sqrt(gamma*pl/dev_bounds_R[id]); 
  
    eta_x[id] = 0.5*fabs((dev_bounds_R[n_cells+id]/dev_bounds_R[id] + ar) - (dev_bounds_L[n_cells+id]/dev_bounds_L[id] - al));

  }

}



/*! \fn void calc_eta_y(Real *dev_bounds_L, Real *dev_bounds_R, Real *eta_y, int nx, int ny, int n_ghost, Real gamma)
 *  \brief When passed the left and right boundary values at an interface, calculates
           the eta value for the interface according to the forumulation in Sanders et al, 1998. */
__global__ void calc_eta_y_2D(Real *dev_bounds_L, Real *dev_bounds_R, Real *eta_y, int nx, int ny, int n_ghost, Real gamma)
{
  int n_cells = nx*ny;

  // declare primative variables for each stencil
  // these will be placed into registers for each thread
  Real pl, pr, al, ar;

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId*blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;


  if (yid > n_ghost-2 && yid < ny-n_ghost && xid > n_ghost-2 && xid < nx-n_ghost+1)
  {
    // load the interface values into registers
    id = xid + yid*nx;
    pl  = (dev_bounds_L[4*n_cells + id] - 
      0.5*(dev_bounds_L[2*n_cells+id]*dev_bounds_L[2*n_cells+id] + 
           dev_bounds_L[3*n_cells+id]*dev_bounds_L[3*n_cells+id] + 
           dev_bounds_L[  n_cells+id]*dev_bounds_L[  n_cells+id])/dev_bounds_L[id]) * (gamma - 1.0);
    pl  = fmax(pl, (Real) 1.0e-20);
    pr  = (dev_bounds_R[4*n_cells + id] - 
      0.5*(dev_bounds_R[2*n_cells+id]*dev_bounds_R[2*n_cells+id] + 
           dev_bounds_R[3*n_cells+id]*dev_bounds_R[3*n_cells+id] + 
           dev_bounds_R[  n_cells+id]*dev_bounds_R[  n_cells+id])/dev_bounds_R[id]) * (gamma - 1.0);
    pr  = fmax(pr, (Real) 1.0e-20);

    al = sqrt(gamma*pl/dev_bounds_L[id]); 
    ar = sqrt(gamma*pl/dev_bounds_R[id]); 
  
    eta_y[id] = 0.5*fabs((dev_bounds_R[2*n_cells+id]/dev_bounds_R[id] + ar) - (dev_bounds_L[2*n_cells+id]/dev_bounds_L[id] - al));

  }

}



/*! \fn void calc_etah_x_2D(Real *eta_x, Real *eta_y, Real *etah_x, int nx, int ny, int n_ghost)
 *  \brief When passed the eta values at every interface, calculates
           the eta_h value for the interface according to the forumulation in Sanders et al, 1998. */
__global__ void calc_etah_x_2D(Real *eta_x, Real *eta_y, Real *etah_x, int nx, int ny, int n_ghost)
{

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId*blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;
  
  Real etah;

  if (xid > n_ghost-2 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost)
  {
      id = xid + yid*nx;

      etah = fmax(eta_y[xid + (yid-1)*nx], eta_y[id]);
      etah = fmax(etah, eta_x[id]);
      etah = fmax(etah, eta_y[xid+1 + (yid-1)*nx]);
      etah = fmax(etah, eta_y[xid+1 + yid*nx]);

      etah_x[id] = etah;
  }

}


/*! \fn void calc_etah_y_2D(Real *eta_x, Real *eta_y, Real *etah_y, int nx, int ny, int n_ghost)
 *  \brief When passed the eta values at every interface, calculates
           the eta_h value for the interface according to the forumulation in Sanders et al, 1998. */
__global__ void calc_etah_y_2D(Real *eta_x, Real *eta_y, Real *etah_y, int nx, int ny, int n_ghost)
{

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId*blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;

  Real etah;

  if (yid > n_ghost-2 && yid < ny-n_ghost && xid > n_ghost-1 && xid < nx-n_ghost)
  {
      id = xid + yid*nx;

      etah = fmax(eta_x[xid-1 + yid*nx], eta_x[id]);
      etah = fmax(etah, eta_y[id]);
      etah = fmax(etah, eta_x[xid-1 + (yid+1)*nx]);
      etah = fmax(etah, eta_x[xid + (yid+1)*nx]);

      etah_y[id] = etah;
  }

}




#endif //H_CORRECTION
#endif //CUDA

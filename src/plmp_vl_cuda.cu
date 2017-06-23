/*! \file plmp_vl_cuda.cu
 *  \brief Definitions of the piecewise linear reconstruction functions for  
           use with the Van Leer integrator, as decribed
           in Stone et al., 2009. */
#ifdef CUDA

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"plmp_vl_cuda.h"


/*! \fn __global__ void PLMP_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using plm. */
__global__ void PLMP_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
{
  int n_cells = nx*ny*nz;
  int o1, o2, o3;
  if (dir == 0) {
    o1 = 1; o2 = 2; o3 = 3;
  }
  if (dir == 1) {
    o1 = 2; o2 = 3; o3 = 1;
  }
  if (dir == 2) {
    o1 = 3; o2 = 1; o3 = 2;
  }

  // declare primative variables in the stencil
  Real d_i, vx_i, vy_i, vz_i, p_i;
  Real d_imo, vx_imo, vy_imo, vz_imo, p_imo; 
  Real d_ipo, vx_ipo, vy_ipo, vz_ipo, p_ipo;

  // declare left and right interface values
  Real d_L, vx_L, vy_L, vz_L, p_L;
  Real d_R, vx_R, vy_R, vz_R, p_R;


  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId*blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;


  if (xid < nx && yid < ny && zid < nz)
  {
    // load the 3-cell stencil into registers
    // cell i
    id = xid + yid*nx + zid*nx*ny;
    d_i  =  dev_conserved[            id];
    vx_i =  dev_conserved[o1*n_cells + id] / d_i;
    vy_i =  dev_conserved[o2*n_cells + id] / d_i;
    vz_i =  dev_conserved[o3*n_cells + id] / d_i;
    p_i  = (dev_conserved[4*n_cells + id] - 0.5*d_i*(vx_i*vx_i + vy_i*vy_i + vz_i*vz_i)) * (gamma - 1.0);
    p_i  = fmax(p_i, (Real) TINY_NUMBER);
    // cell i-1
    if (dir == 0) id = xid-1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-1)*nx*ny;
    d_imo  =  dev_conserved[            id];
    vx_imo =  dev_conserved[o1*n_cells + id] / d_imo;
    vy_imo =  dev_conserved[o2*n_cells + id] / d_imo;
    vz_imo =  dev_conserved[o3*n_cells + id] / d_imo;
    p_imo  = (dev_conserved[4*n_cells + id] - 0.5*d_imo*(vx_imo*vx_imo + vy_imo*vy_imo + vz_imo*vz_imo)) * (gamma - 1.0);
    p_imo  = fmax(p_imo, (Real) TINY_NUMBER);
    // cell i+1
    if (dir == 0) id = xid+1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid+1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid+1)*nx*ny;
    d_ipo  =  dev_conserved[            id];
    vx_ipo =  dev_conserved[o1*n_cells + id] / d_ipo;
    vy_ipo =  dev_conserved[o2*n_cells + id] / d_ipo;
    vz_ipo =  dev_conserved[o3*n_cells + id] / d_ipo;
    p_ipo  = (dev_conserved[4*n_cells + id] - 0.5*d_ipo*(vx_ipo*vx_ipo + vy_ipo*vy_ipo + vz_ipo*vz_ipo)) * (gamma - 1.0);
    p_ipo  = fmax(p_ipo, (Real) TINY_NUMBER);


    // Calculate the interface values for each primitive variable
    Interface_Values_PLM(d_imo,  d_i,  d_ipo,  &d_L,  &d_R); 
    Interface_Values_PLM(vx_imo, vx_i, vx_ipo, &vx_L, &vx_R); 
    Interface_Values_PLM(vy_imo, vy_i, vy_ipo, &vy_L, &vy_R); 
    Interface_Values_PLM(vz_imo, vz_i, vz_ipo, &vz_L, &vz_R); 
    Interface_Values_PLM(p_imo,  p_i,  p_ipo,  &p_L,  &p_R); 

    // Apply mimimum constraints
    d_L = fmax(d_L, (Real) TINY_NUMBER);
    d_R = fmax(d_R, (Real) TINY_NUMBER);
    p_L = fmax(p_L, (Real) TINY_NUMBER);
    p_R = fmax(p_R, (Real) TINY_NUMBER);

    // Convert the left and right states in the primitive to the conserved variables
    // send final values back from kernel
    // bounds_R refers to the right side of the i-1/2 interface
    if (dir == 0) id = xid-1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-1)*nx*ny;
    dev_bounds_R[            id] = d_L;
    dev_bounds_R[o1*n_cells + id] = d_L*vx_L;
    dev_bounds_R[o2*n_cells + id] = d_L*vy_L;
    dev_bounds_R[o3*n_cells + id] = d_L*vz_L;
    dev_bounds_R[4*n_cells + id] = (p_L/(gamma-1.0)) + 0.5*d_L*(vx_L*vx_L + vy_L*vy_L + vz_L*vz_L);  
    // bounds_L refers to the left side of the i+1/2 interface
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_L[            id] = d_R;
    dev_bounds_L[o1*n_cells + id] = d_R*vx_R;
    dev_bounds_L[o2*n_cells + id] = d_R*vy_R;
    dev_bounds_L[o3*n_cells + id] = d_R*vz_R;
    dev_bounds_L[4*n_cells + id] = (p_R/(gamma-1.0)) + 0.5*d_R*(vx_R*vx_R + vy_R*vy_R + vz_R*vz_R);      

  }
}
    


__device__ void Interface_Values_PLM(Real q_imo, Real q_i, Real q_ipo, Real *q_L, Real *q_R)
{
  Real del_q_L, del_q_R, del_q_C, del_q_G; 
  Real lim_slope_a, lim_slope_b, del_q_m;  

  // Compute the left, right, and centered differences of the primative variables
  // Note that here L and R refer to locations relative to the cell center

  // left
  del_q_L  = q_i - q_imo;
  // right
  del_q_R  = q_ipo - q_i;
  // centered
  del_q_C  = 0.5*(q_ipo - q_imo);
  // Van Leer
  if (del_q_L*del_q_R > 0.0) { del_q_G = 2.0*del_q_L*del_q_R / (del_q_L+del_q_R); }
  else { del_q_G = 0.0; }

  // Monotonize the differences
  lim_slope_a = fmin(fabs(del_q_L), fabs(del_q_R));
  lim_slope_b = fmin(fabs(del_q_C), fabs(del_q_G));
  
  // Minmod limiter
  //del_q_m = sgn_CUDA(del_q_C)*fmin(2.0*lim_slope_a, fabs(del_q_C));

  // Van Leer limiter 
  del_q_m = sgn_CUDA(del_q_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);


  // Calculate the left and right interface values using the limited slopes
  *q_L = q_i - 0.5*del_q_m;
  *q_R = q_i + 0.5*del_q_m;

}


#endif //CUDA

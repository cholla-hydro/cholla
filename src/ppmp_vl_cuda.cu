/*! \file ppmp_vl_cuda.cu
 *  \brief Definitions of the piecewise parabolic reconstruction (Fryxell 200) functions for  
           use with the Van Leer integrator, as decribed in Stone et al., 2009. */
#ifdef CUDA
#ifdef PPMP

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"ppmp_vl_cuda.h"


/*! \fn __global__ void PPMP_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using ppm. */
__global__ void PPMP_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
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
  Real d_imt, vx_imt, vy_imt, vz_imt, p_imt; 
  Real d_ipt, vx_ipt, vy_ipt, vz_ipt, p_ipt;

  // declare left and right interface values
  Real d_L, vx_L, vy_L, vz_L, p_L;
  Real d_R, vx_R, vy_R, vz_R, p_R;

  // declare other variables
  Real del_q_imo, del_q_i, del_q_ipo;

  #ifdef DE
  Real ge_i, ge_imo, ge_ipo, ge_imt, ge_ipt, ge_L, ge_R;
  #endif

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId*blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;


  //if (xid > n_ghost-2 && xid < nx-n_ghost+1 && yid < ny && zid < nz)
  if (xid < nx && yid < ny && zid < nz)
  {
    // load the 5-cell stencil into registers
    // cell i
    id = xid + yid*nx + zid*nx*ny;
    d_i  =  dev_conserved[            id];
    vx_i =  dev_conserved[o1*n_cells + id] / d_i;
    vy_i =  dev_conserved[o2*n_cells + id] / d_i;
    vz_i =  dev_conserved[o3*n_cells + id] / d_i;
    p_i  = (dev_conserved[4*n_cells + id] - 0.5*d_i*(vx_i*vx_i + vy_i*vy_i + vz_i*vz_i)) * (gamma - 1.0);
    p_i  = fmax(p_i, (Real) TINY_NUMBER);
    #ifdef DE
    ge_i = dev_conserved[5*n_cells + id];
    #endif
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
    #ifdef DE
    ge_imo = dev_conserved[5*n_cells + id];
    #endif    
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
    #ifdef DE
    ge_ipo = dev_conserved[5*n_cells + id];
    #endif
    // cell i-2
    if (dir == 0) id = xid-2 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-2)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-2)*nx*ny;
    d_imt  =  dev_conserved[            id];
    vx_imt =  dev_conserved[o1*n_cells + id] / d_imt;
    vy_imt =  dev_conserved[o2*n_cells + id] / d_imt;
    vz_imt =  dev_conserved[o3*n_cells + id] / d_imt;
    p_imt  = (dev_conserved[4*n_cells + id] - 0.5*d_imt*(vx_imt*vx_imt + vy_imt*vy_imt + vz_imt*vz_imt)) * (gamma - 1.0);
    p_imt  = fmax(p_imt, (Real) TINY_NUMBER);
    #ifdef DE
    ge_imt = dev_conserved[5*n_cells + id];
    #endif
    // cell i+2
    if (dir == 0) id = xid+2 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid+2)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid+2)*nx*ny;
    d_ipt  =  dev_conserved[            id];
    vx_ipt =  dev_conserved[o1*n_cells + id] / d_ipt;
    vy_ipt =  dev_conserved[o2*n_cells + id] / d_ipt;
    vz_ipt =  dev_conserved[o3*n_cells + id] / d_ipt;
    p_ipt  = (dev_conserved[4*n_cells + id] - 0.5*d_ipt*(vx_ipt*vx_ipt + vy_ipt*vy_ipt + vz_ipt*vz_ipt)) * (gamma - 1.0);
    p_ipt  = fmax(p_ipt, (Real) TINY_NUMBER);
    #ifdef DE
    ge_ipt = dev_conserved[5*n_cells + id];
    #endif


    // Calculate the monotonized slopes for cells imo, i, ipo (density)
    del_q_imo = Calculate_Slope(d_imt, d_imo, d_i);
    del_q_i   = Calculate_Slope(d_imo, d_i,   d_ipo);
    del_q_ipo = Calculate_Slope(d_i,   d_ipo, d_ipt);

    // Calculate the interface values for density
    Interface_Values_PPM(d_imo,  d_i,  d_ipo,  del_q_imo, del_q_i, del_q_ipo, &d_L,  &d_R); 

    // Calculate the monotonized slopes for cells imo, i, ipo (x-velocity)
    del_q_imo = Calculate_Slope(vx_imt, vx_imo, vx_i);
    del_q_i   = Calculate_Slope(vx_imo, vx_i,   vx_ipo);
    del_q_ipo = Calculate_Slope(vx_i,   vx_ipo, vx_ipt);

    // Calculate the interface values for x-velocity
    Interface_Values_PPM(vx_imo, vx_i, vx_ipo, del_q_imo, del_q_i, del_q_ipo, &vx_L, &vx_R); 

    // Calculate the monotonized slopes for cells imo, i, ipo (y-velocity)
    del_q_imo = Calculate_Slope(vy_imt, vy_imo, vy_i);
    del_q_i   = Calculate_Slope(vy_imo, vy_i,   vy_ipo);
    del_q_ipo = Calculate_Slope(vy_i,   vy_ipo, vy_ipt);

    // Calculate the interface values for y-velocity
    Interface_Values_PPM(vy_imo, vy_i, vy_ipo, del_q_imo, del_q_i, del_q_ipo, &vy_L, &vy_R); 

    // Calculate the monotonized slopes for cells imo, i, ipo (z-velocity)
    del_q_imo = Calculate_Slope(vz_imt, vz_imo, vz_i);
    del_q_i   = Calculate_Slope(vz_imo, vz_i,   vz_ipo);
    del_q_ipo = Calculate_Slope(vz_i,   vz_ipo, vz_ipt);

    // Calculate the interface values for z-velocity
    Interface_Values_PPM(vz_imo, vz_i, vz_ipo, del_q_imo, del_q_i, del_q_ipo, &vz_L, &vz_R); 

    // Calculate the monotonized slopes for cells imo, i, ipo (pressure)
    del_q_imo = Calculate_Slope(p_imt, p_imo, p_i);
    del_q_i   = Calculate_Slope(p_imo, p_i,   p_ipo);
    del_q_ipo = Calculate_Slope(p_i,   p_ipo, p_ipt);

    // Calculate the interface values for pressure
    Interface_Values_PPM(p_imo,  p_i,  p_ipo,  del_q_imo, del_q_i, del_q_ipo, &p_L,  &p_R); 

    #ifdef DE
    // Calculate the monotonized slopes for cells imo, i, ipo (internal energy)
    del_q_imo = Calculate_Slope(ge_imt, ge_imo, ge_i);
    del_q_i   = Calculate_Slope(ge_imo, ge_i,   ge_ipo);
    del_q_ipo = Calculate_Slope(ge_i,   ge_ipo, ge_ipt);

    // Calculate the interface values for internal energy
    Interface_Values_PPM(ge_imo,  ge_i,  ge_ipo,  del_q_imo, del_q_i, del_q_ipo, &ge_L,  &ge_R); 
    #endif

    // ensure that the parabolic distribution of each of the primative variables is monotonic
    // local maximum or minimum criterion (Fryxell Eqn 52, Fig 11)
    if ( (d_R  - d_i)  * (d_i  - d_L)  <= 0)  { d_L  = d_R  = d_i; }
    if ( (vx_R - vx_i) * (vx_i - vx_L) <= 0)  { vx_L = vx_R = vx_i; }
    if ( (vy_R - vy_i) * (vy_i - vy_L) <= 0)  { vy_L = vy_R = vy_i; }
    if ( (vz_R - vz_i) * (vz_i - vz_L) <= 0)  { vz_L = vz_R = vz_i; }
    if ( (p_R  - p_i)  * (p_i  - p_L)  <= 0)  { p_L  = p_R  = p_i; }
    #ifdef DE
    if ( (ge_R - ge_i) * (ge_i - ge_L) <= 0)  { ge_L = ge_R = ge_i; }
    #endif
    // steep gradient criterion (Fryxell Eqn 53, Fig 12)
    if ( (d_R  - d_L)  * (d_L -  (3*d_i  - 2*d_R))  < 0)  { d_L  = 3*d_i  - 2*d_R;  }
    if ( (vx_R - vx_L) * (vx_L - (3*vx_i - 2*vx_R)) < 0)  { vx_L = 3*vx_i - 2*vx_R; }
    if ( (vy_R - vy_L) * (vy_L - (3*vy_i - 2*vy_R)) < 0)  { vy_L = 3*vy_i - 2*vy_R; }
    if ( (vz_R - vz_L) * (vz_L - (3*vz_i - 2*vz_R)) < 0)  { vz_L = 3*vz_i - 2*vz_R; }
    if ( (p_R  - p_L)  * (p_L -  (3*p_i  - 2*p_R))  < 0)  { p_L  = 3*p_i  - 2*p_R;  }
    #ifdef DE
    if ( (ge_R - ge_L) * (ge_L - (3*ge_i - 2*ge_R)) < 0)  { ge_L = 3*ge_i - 2*ge_R; }
    #endif
    if ( (d_R  - d_L)  * ((3*d_i  - 2*d_L)  - d_R)  < 0)  { d_R  = 3*d_i  - 2*d_L;  }
    if ( (vx_R - vx_L) * ((3*vx_i - 2*vx_L) - vx_R) < 0)  { vx_R = 3*vx_i - 2*vx_L; }
    if ( (vy_R - vy_L) * ((3*vy_i - 2*vy_L) - vy_R) < 0)  { vy_R = 3*vy_i - 2*vy_L; }
    if ( (vz_R - vz_L) * ((3*vz_i - 2*vz_L) - vz_R) < 0)  { vz_R = 3*vz_i - 2*vz_L; }
    if ( (p_R  - p_L)  * ((3*p_i  - 2*p_L)  - p_R)  < 0)  { p_R  = 3*p_i  - 2*p_L;  }
    #ifdef DE
    if ( (ge_R - ge_L) * ((3*ge_i - 2*ge_L) - ge_R) < 0)  { ge_R = 3*ge_i - 2*ge_L; }
    #endif

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
    dev_bounds_R[4*n_cells + id] = p_L/(gamma-1.0) + 0.5*d_L*(vx_L*vx_L + vy_L*vy_L + vz_L*vz_L);  
    #ifdef DE
    dev_bounds_R[5*n_cells + id] = ge_L;
    #endif
    // bounds_L refers to the left side of the i+1/2 interface
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_L[            id] = d_R;
    dev_bounds_L[o1*n_cells + id] = d_R*vx_R;
    dev_bounds_L[o2*n_cells + id] = d_R*vy_R;
    dev_bounds_L[o3*n_cells + id] = d_R*vz_R;
    dev_bounds_L[4*n_cells + id] = p_R/(gamma-1.0) + 0.5*d_R*(vx_R*vx_R + vy_R*vy_R + vz_R*vz_R);      
    #ifdef DE
    dev_bounds_L[5*n_cells + id] = ge_R;
    #endif
  
  }
}
    


/*! \fn __device__ Real Calculate_Slope(Real q_imo, Real q_i, Real q_ipo)
 *  \brief Calculates the limited slope across a cell.*/
__device__ Real Calculate_Slope(Real q_imo, Real q_i, Real q_ipo)
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
  //del_q_m = sgn(del_q_C)*fmin(2.0*lim_slope_a, fabs(del_q_C));

  // Van Leer limiter
  del_q_m = sgn(del_q_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);

  return del_q_m;

}


/*! \fn __device__ void Interface_Values_PPM(Real q_imo, Real q_i, Real q_ipo, Real del_q_imo, Real del_q_i, Real del_q_ipo, Real *q_L, Real *q_R)
 *  \brief Calculates the left and right interface values for a cell using parabolic reconstruction
           in the primitive variables with limited slopes provided. Applies further monotonicity constraints.*/
__device__ void Interface_Values_PPM(Real q_imo, Real q_i, Real q_ipo, Real del_q_imo, Real del_q_i, Real del_q_ipo, Real *q_L, Real *q_R)
{
  // Calculate the left and right interface values using the limited slopes
  *q_L = 0.5*(q_i + q_imo) - (1.0/6.0)*(del_q_i - del_q_imo);
  *q_R = 0.5*(q_ipo + q_i) - (1.0/6.0)*(del_q_ipo - del_q_i);

  // Apply further monotonicity constraints to ensure interface values lie between
  // neighboring cell-centered values

  if ((*q_R - q_i)*(q_i - *q_L) <= 0) *q_L = *q_R = q_i;

  if (6.0*(*q_R - *q_L)*(q_i - 0.5*(*q_L + *q_R)) > (*q_R - *q_L)*(*q_R - *q_L))  *q_L = 3.0*q_i - 2.0*(*q_R);

  if (6.0*(*q_R - *q_L)*(q_i - 0.5*(*q_L + *q_R)) < -(*q_R - *q_L)*(*q_R - *q_L)) *q_R = 3.0*q_i - 2.0*(*q_L);
  
  *q_L  = fmax( fmin(q_i, q_imo), *q_L );
  *q_L  = fmin( fmax(q_i, q_imo), *q_L );
  *q_R  = fmax( fmin(q_i, q_ipo), *q_R );
  *q_R  = fmin( fmax(q_i, q_ipo), *q_R );

}

#endif //PPMP
#endif //CUDA

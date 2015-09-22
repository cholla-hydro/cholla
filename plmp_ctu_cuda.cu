/*! \file plmp_ctu_cuda.cu
 *  \brief Definitions of the piecewise linear reconstruction functions, in the primitive variables, 
           as described in Toro... */
#ifdef CUDA
#ifdef PLMP

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"plmp_ctu_cuda.h"


/*! \fn __global__ void PLMP_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using plm. */
__global__ void PLMP_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
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

  // declare primative variables for each stencil
  // these will be placed into registers for each thread
  Real d_i, vx_i, vy_i, vz_i, p_i;
  Real d_imo, vx_imo, vy_imo, vz_imo, p_imo; 
  Real d_ipo, vx_ipo, vy_ipo, vz_ipo, p_ipo;

  Real dtodx = dt/dx;

  // declare other variables to be used
	Real dl, dr, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr;  
	Real mxl, mxr, myl, myr, mzl, mzr, El, Er;
  Real dfl, dfr, mxfl, mxfr, myfl, myfr, mzfl, mzfr, Efl, Efr;
  Real del_d_L, del_vx_L, del_vy_L, del_vz_L, del_p_L;
  Real del_d_R, del_vx_R, del_vy_R, del_vz_R, del_p_R;
	Real d_slope, vx_slope, vy_slope, vz_slope, p_slope;

  // variables needed for dual energy
  #ifdef DE
  Real ge_i, ge_imo, ge_ipo;
  Real gel, ger;
  Real del_ge_L, del_ge_R, ge_slope;
  #endif

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId*blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;


  //if (xid > n_ghost-3 && xid < nx-n_ghost+2 && yid < ny && zid < nz)
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
    // for dual energy pressure is a separately tracked quantity
    #ifdef DE
    ge_i =  dev_conserved[5*n_cells + id];
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
    ge_imo =  dev_conserved[5*n_cells + id];
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
    ge_ipo =  dev_conserved[5*n_cells + id];
    #endif


    // calculate the slope (in each primative variable) across cell i

    // Left
    del_d_L  = d_i  - d_imo;
    del_vx_L = vx_i - vx_imo;
    del_vy_L = vy_i - vy_imo;
    del_vz_L = vz_i - vz_imo;
    del_p_L  = p_i  - p_imo;
    #ifdef DE
    del_ge_L = ge_i - ge_imo;
    #endif

    // Right
    del_d_R  = d_ipo  - d_i;
    del_vx_R = vx_ipo - vx_i;
    del_vy_R = vy_ipo - vy_i;
    del_vz_R = vz_ipo - vz_i;
    del_p_R  = p_ipo  - p_i;
    #ifdef DE
    del_ge_R = ge_ipo - ge_i;
    #endif
    
    
    // limit the slopes (B=1 is minmod, B=2 is superbee) (see Eqn 13.229 of Toro 2009)
    Real B = 1;
    if (del_d_R >=0) d_slope  = fmax(0, fmax(fmin(B*del_d_L, del_d_R), fmin(del_d_L, B*del_d_R)));
    if (del_d_R<0)   d_slope  = fmin(0, fmin(fmax(B*del_d_L, del_d_R), fmax(del_d_L, B*del_d_R)));
    if (del_vx_R>=0) vx_slope = fmax(0, fmax(fmin(B*del_vx_L, del_vx_R), fmin(del_vx_L, B*del_vx_R)));
    if (del_vx_R<0)  vx_slope = fmin(0, fmin(fmax(B*del_vx_L, del_vx_R), fmax(del_vx_L, B*del_vx_R)));
    if (del_vy_R>=0) vy_slope = fmax(0, fmax(fmin(B*del_vy_L, del_vy_R), fmin(del_vy_L, B*del_vy_R)));
    if (del_vy_R<0)  vy_slope = fmin(0, fmin(fmax(B*del_vy_L, del_vy_R), fmax(del_vy_L, B*del_vy_R)));
    if (del_vz_R>=0) vz_slope = fmax(0, fmax(fmin(B*del_vz_L, del_vz_R), fmin(del_vz_L, B*del_vz_R)));
    if (del_vz_R<0)  vz_slope = fmin(0, fmin(fmax(B*del_vz_L, del_vz_R), fmax(del_vz_L, B*del_vz_R)));
    if (del_p_R>=0)  p_slope  = fmax(0, fmax(fmin(B*del_p_L, del_p_R), fmin(del_p_L, B*del_p_R)));
    if (del_p_R<0)   p_slope  = fmin(0, fmin(fmax(B*del_p_L, del_p_R), fmax(del_p_L, B*del_p_R)));
    #ifdef DE
    if (del_ge_R>=0) ge_slope  = fmax(0, fmax(fmin(B*del_ge_L, del_ge_R), fmin(del_ge_L, B*del_ge_R)));
    if (del_ge_R<0)  ge_slope  = fmin(0, fmin(fmax(B*del_ge_L, del_ge_R), fmax(del_ge_L, B*del_ge_R)));
    #endif
     

    // set the boundary values for cell i
    dl = d_i - 0.5 * d_slope;
    dr = d_i + 0.5 * d_slope;
    vxl = vx_i - 0.5 * vx_slope;
    vxr = vx_i + 0.5 * vx_slope;
    vyl = vy_i - 0.5 * vy_slope;
    vyr = vy_i + 0.5 * vy_slope;
    vzl = vz_i - 0.5 * vz_slope;
    vzr = vz_i + 0.5 * vz_slope;
    pl = p_i - 0.5 * p_slope;
    pr = p_i + 0.5 * p_slope;
    #ifdef DE
    gel = ge_i - 0.5 * ge_slope;
    ger = ge_i + 0.5 * ge_slope;
    #endif


    // calculate the conserved variables and fluxes at each interface
    mxl = dl*vxl;
    mxr = dr*vxr;
    myl = dl*vyl;
    myr = dr*vyr;
    mzl = dl*vzl; 
    mzr = dr*vzr;
    El = pl/(gamma-1.0) + 0.5*dl*(vxl*vxl + vyl*vyl + vzl*vzl);
    Er = pr/(gamma-1.0) + 0.5*dr*(vxr*vxr + vyr*vyr + vzr*vzr);


    dfl = mxl;
    dfr = mxr;
    mxfl = mxl*vxl + pl;
    mxfr = mxr*vxr + pr;
    myfl = mxl*vyl;
    myfr = mxr*vyr;
    mzfl = mxl*vzl;
    mzfr = mxr*vzr;
    Efl = (El + pl) * vxl;
    Efr = (Er + pr) * vxr;
    #ifdef DE
    gefl = gel*dl*vxl;
    gefr = ger*dr*vxr;
    #endif

    // Evolve the boundary extrapolated values half a timestep.
    dl += 0.5 * (dtodx) * (dfl - dfr);
    dr += 0.5 * (dtodx) * (dfl - dfr);
    mxl += 0.5 * (dtodx) * (mxfl - mxfr);
    mxr += 0.5 * (dtodx) * (mxfl - mxfr);
    myl += 0.5 * (dtodx) * (myfl - myfr);
    myr += 0.5 * (dtodx) * (myfl - myfr);
    mzl += 0.5 * (dtodx) * (mzfl - mzfr);
    mzr += 0.5 * (dtodx) * (mzfl - mzfr);
    El += 0.5 * (dtodx) * (Efl - Efr);
    Er += 0.5 * (dtodx) * (Efl - Efr);	
    #ifdef DE
    gel += 0.5 * (dtodx) * (gefl - gefr);
    ger += 0.5 * (dtodx) * (gefl - gefr);
    #endif

    // apply minimum constraints
    dl = fmax(dl, (Real) TINY_NUMBER);
    dr = fmax(dr, (Real) TINY_NUMBER);
    pl = fmax(pl, (Real) TINY_NUMBER);
    pr = fmax(pr, (Real) TINY_NUMBER);

    // send final values back from kernel
    // bounds_R refers to the right side of the i-1/2 interface
    if (dir == 0) id = xid-1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-1)*nx*ny;
    dev_bounds_R[            id] = dl;
    dev_bounds_R[o1*n_cells + id] = mxl;
    dev_bounds_R[o2*n_cells + id] = myl;
    dev_bounds_R[o3*n_cells + id] = mzl;
    dev_bounds_R[4*n_cells + id] = El;    
    // bounds_L refers to the left side of the i+1/2 interface
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_L[            id] = dr;
    dev_bounds_L[1*n_cells + id] = mxr;
    dev_bounds_L[2*n_cells + id] = myr;
    dev_bounds_L[3*n_cells + id] = mzr;
    dev_bounds_L[4*n_cells + id] = Er;

  }
}
    


#endif //PLMP
#endif //CUDA

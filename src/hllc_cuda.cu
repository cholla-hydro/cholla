/*! \file hllc_cuda.cu
 *  \brief Function definitions for the cuda HLLC Riemann solver.*/

#ifdef CUDA

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"roe_cuda.h"



/*! \fn Calculate_HLLC_Fluxes(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief HLLC Riemann solver based on the version described in Toro (2006), Sec. 10.4. */
__global__ void Calculate_HLLC_Fluxes(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
{
  // get a thread index
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId * blockDim.x;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;

  int n_cells = nx*ny*nz;

  Real dl, vxl, mxl, vyl, myl, vzl, mzl, pl, El;
  Real dr, vxr, mxr, vyr, myr, vzr, mzr, pr, Er;

  Real g1 = gamma - 1.0; 
  Real Hl, Hr;
  Real sqrtdl, sqrtdr, vx, vy, vz, H;
  Real vsq, asq, a;
  Real lambda_m, lambda_p;
  Real f_d_l, f_mx_l, f_my_l, f_mz_l, f_E_l;
  Real f_d_r, f_mx_r, f_my_r, f_mz_r, f_E_r;
  Real f_d, f_mx, f_my, f_mz, f_E;
  Real al, ar, bm, bp, tl, tr, bl, br, tmp, am, cp, sl, sr, sm, cfl, cfr;
  #ifdef DE
  Real gel, ger, f_ge;
  #endif

  int o1, o2, o3;
  if (dir==0) {
    o1 = 1; o2 = 2; o3 = 3;
  }
  if (dir==1) {
    o1 = 2; o2 = 3; o3 = 1;
  }
  if (dir==2) {
    o1 = 3; o2 = 1; o3 = 2;
  }

  // Each thread executes the solver independently
  //if (xid > n_ghost-3 && xid < nx-n_ghost+1 && yid < ny && zid < nz) 
  if (xid < nx && yid < ny && zid < nz) 
  {
    // retrieve conserved variables
    dl  = dev_bounds_L[             tid];
    mxl = dev_bounds_L[o1*n_cells + tid];
    myl = dev_bounds_L[o2*n_cells + tid];
    mzl = dev_bounds_L[o3*n_cells + tid];
    El  = dev_bounds_L[4*n_cells + tid];
    #ifdef DE
    gel = dev_bounds_L[5*n_cells + tid] / dl;
    #endif

    dr  = dev_bounds_R[            tid];
    mxr = dev_bounds_R[o1*n_cells + tid];
    myr = dev_bounds_R[o2*n_cells + tid];
    mzr = dev_bounds_R[o3*n_cells + tid];
    Er  = dev_bounds_R[4*n_cells + tid]; 
    #ifdef DE
    ger = dev_bounds_R[5*n_cells + tid] / dr;
    #endif

    // calculate primative variables
    vxl = mxl / dl;
    vyl = myl / dl;
    vzl = mzl / dl;
    pl  = (El - 0.5*dl*(vxl*vxl + vyl*vyl + vzl*vzl)) * (gamma - 1.0);
    pl  = fmax(pl, (Real) TINY_NUMBER);
    vxr = mxr / dr;
    vyr = myr / dr;
    vzr = mzr / dr;
    pr  = (Er - 0.5*dr*(vxr*vxr + vyr*vyr + vzr*vzr)) * (gamma - 1.0);
    pr  = fmax(pr, (Real) TINY_NUMBER);    

    // calculate the enthalpy in each cell
    Hl = (El + pl) / dl;
    Hr = (Er + pr) / dr;

    // calculate averages of the variables needed for the Roe Jacobian 
    // (see Stone et al., 2008, Eqn 65, or Toro 2009, 11.118)
    sqrtdl = sqrt(dl);
    sqrtdr = sqrt(dr);
    vx = (sqrtdl*vxl + sqrtdr*vxr) / (sqrtdl + sqrtdr);
    vy = (sqrtdl*vyl + sqrtdr*vyr) / (sqrtdl + sqrtdr);
    vz = (sqrtdl*vzl + sqrtdr*vzr) / (sqrtdl + sqrtdr);
    H  = (sqrtdl*Hl  + sqrtdr*Hr)  / (sqrtdl + sqrtdr); 

    // calculate the sound speed squared (Stone B2)
    vsq = (vx*vx + vy*vy + vz*vz);
    asq = g1*(H - 0.5*vsq);
    a = sqrt(asq);

    // calculate the averaged eigenvectors of the Roe matrix (Stone Eqn B2, Toro 11.107)
    lambda_m = vx - a; 
    lambda_p = vx + a;

    // compute max and min wave speeds
    cfl = sqrt(gamma*pl/dl);  // sound speed in left state
    cfr = sqrt(gamma*pr/dr);  // sound speed in right state

    // for signal speeds,
    // take max/min of Roe eigenvalues and left and right sound speeds
    al = fmin(lambda_m, vxl - cfl);
    ar = fmax(lambda_p, vxr + cfr);
   
    bm = fmin(al, (Real) 0.0);
    bp = fmax(ar, (Real) 0.0);


    // compute contact wave speed and pressure in star region (Toro eqn 10.37)
    tl = pl + (vxl - al)*mxl;
    tr = pr + (vxr - ar)*mxr;
    bl = mxl - dl*al;
    br = -(mxr - dr*ar);
    tmp = 1.0 / (bl + br);
    am = (tl - tr)*tmp; // contact wave speed
    cp = (dl*tr + dr*tl)*tmp; // star region pressure

    // compute flux weights
    if (am >= 0.0) {
      sl = am/(am - bm);
      sr = 0.0;
      sm = -bm/(am - bm);
    }
    else {
      sl = 0.0;
      sr = -am/(bp - am);
      sm = bp/(bp - am);
    }

    // caclulate the left and right fluxes 
    // along the bm/bp lines
    f_d_l = mxl - bm*dl;
    f_mx_l = mxl*(vxl - bm) + pl;
    f_my_l = myl*(vxl - bm);
    f_mz_l = mzl*(vxl - bm);
    f_E_l = El*(vxl - bm) + pl*vxl;

    f_d_r = mxr - bp*dr;
    f_mx_r = mxr*(vxr - bp) + pr;
    f_my_r = myr*(vxr - bp);
    f_mz_r = mzr*(vxr - bp);
    f_E_r = Er*(vxr - bp) + pr*vxr;

    // compute the hllc flux
    f_d = sl*f_d_l + sr*f_d_r;
    f_mx = sl*f_mx_l + sr*f_mx_r + sm*cp;
    f_my = sl*f_my_l + sr*f_my_r;
    f_mz = sl*f_mz_l + sr*f_mz_r;
    f_E = sl*f_E_l + sr*f_E_r + sm*cp*am;

    #ifdef DE
    if (f_d >= 0.0)
      f_ge = f_d * gel;
    else
      f_ge = f_d * ger;
    #endif

    // return the hllc fluxes
    dev_flux[          tid] = f_d;
    dev_flux[o1*n_cells+tid] = f_mx;
    dev_flux[o2*n_cells+tid] = f_my;
    dev_flux[o3*n_cells+tid] = f_mz ;
    dev_flux[4*n_cells+tid] = f_E;
    #ifdef DE
    dev_flux[5*n_cells+tid] = f_ge;
    #endif


  }

}


#endif //CUDA

/*! \file hllc_cuda.cu
 *  \brief Function definitions for the cuda HLLC Riemann solver.*/

#ifdef CUDA

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"hllc_cuda.h"



/*! \fn Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief HLLC Riemann solver based on the version described in Toro (2006), Sec. 10.4. */
__global__ void Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, Real *dev_etah, int dir)
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
  Real dls, drs, mxls, mxrs, myls, myrs, mzls, mzrs, Els, Ers;
  Real f_d, f_mx, f_my, f_mz, f_E;
  Real Sl, Sr, Sm, cfl, cfr, ps;
  #ifdef DE
  Real gel, ger, gels, gers, f_ge_l, f_ge_r, f_ge;
  #endif
  // Retrieve etah value
  Real etah = dev_etah[tid];


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

    // for signal speeds, take max/min of Roe eigenvalues and left and right sound speeds
    // Batten eqn. 48
    Sl = fmin(lambda_m, vxl - cfl);
    Sr = fmax(lambda_p, vxr + cfr);

    // if the H-correction is turned on, add cross-flux dissipation
    Sl = sgn_CUDA(Sl)*fmax(fabs(Sl), etah);
    Sr = sgn_CUDA(Sr)*fmax(fabs(Sr), etah);

 
    // left and right fluxes 
    f_d_l  = mxl;
    f_mx_l = mxl*vxl + pl;
    f_my_l = myl*vxl;
    f_mz_l = mzl*vxl;
    f_E_l  = (El + pl)*vxl;
    #ifdef DE
    f_ge_l = mxl*gel;
    #endif

    f_d_r  = mxr;
    f_mx_r = mxr*vxr + pr;
    f_my_r = myr*vxr;
    f_mz_r = mzr*vxr;
    f_E_r  = (Er + pr)*vxr;
    #ifdef DE
    f_ge_r = mxr*ger;
    #endif

    // return upwind flux if flow is supersonic 
    if (Sl > 0.0) {
      dev_flux[           tid] = f_d_l;
      dev_flux[o1*n_cells+tid] = f_mx_l;
      dev_flux[o2*n_cells+tid] = f_my_l;
      dev_flux[o3*n_cells+tid] = f_mz_l;
      dev_flux[4*n_cells+tid]  = f_E_l;
      #ifdef DE
      dev_flux[5*n_cells+tid]  = f_ge_l;
      #endif
      return;
    }
    else if (Sr < 0.0) {
      dev_flux[           tid] = f_d_r;
      dev_flux[o1*n_cells+tid] = f_mx_r;
      dev_flux[o2*n_cells+tid] = f_my_r;
      dev_flux[o3*n_cells+tid] = f_mz_r;
      dev_flux[4*n_cells+tid]  = f_E_r;
      #ifdef DE
      dev_flux[5*n_cells+tid]  = f_ge_r;
      #endif
      return;
    }
    // otherwise compute subsonic flux
    else { 

      // compute contact wave speed and pressure in star region (Batten eqns 34 & 36)
      Sm = (dr*vxr*(Sr - vxr) - dl*vxl*(Sl - vxl) + pl - pr) / (dr*(Sr - vxr) - dl*(Sl - vxl));
      ps = dl*(vxl - Sl)*(vxl - Sm) + pl;

      // conserved variables in the left star state (Batten eqns 35 - 40)
      dls = dl * (Sl - vxl) / (Sl - Sm);
      mxls = (mxl*(Sl - vxl) + ps - pl) / (Sl - Sm);
      myls = myl*(Sl - vxl) / (Sl - Sm);
      mzls = mzl*(Sl - vxl) / (Sl - Sm);
      Els = (El*(Sl - vxl) - pl*vxl + ps*Sm) / (Sl - Sm);
      #ifdef DE
      gels = dl*gel*(Sl - vxl) / (Sl - Sm);
      #endif

      // conserved variables in the right star state
      drs = dr * (Sr - vxr) / (Sr - Sm);
      mxrs = (mxr*(Sr - vxr) + ps - pr) / (Sr - Sm);
      myrs = myr*(Sr - vxr) / (Sr - Sm);
      mzrs = mzr*(Sr - vxr) / (Sr - Sm);
      Ers = (Er*(Sr - vxr) - pr*vxr + ps*Sm) / (Sr - Sm);
      #ifdef DE
      gers = dr*ger*(Sr - vxr) / (Sr - Sm);
      #endif


      // compute the hllc flux (Batten eqn 27)
      f_d  = 0.5*(f_d_l  + f_d_r  + (Sr - fabs(Sm))*drs  + (Sl + fabs(Sm))*dls  - Sl*dl  - Sr*dr);
      f_mx = 0.5*(f_mx_l + f_mx_r + (Sr - fabs(Sm))*mxrs + (Sl + fabs(Sm))*mxls - Sl*mxl - Sr*mxr);
      f_my = 0.5*(f_my_l + f_my_r + (Sr - fabs(Sm))*myrs + (Sl + fabs(Sm))*myls - Sl*myl - Sr*myr);
      f_mz = 0.5*(f_mz_l + f_mz_r + (Sr - fabs(Sm))*mzrs + (Sl + fabs(Sm))*mzls - Sl*mzl - Sr*mzr);
      f_E  = 0.5*(f_E_l  + f_E_r  + (Sr - fabs(Sm))*Ers  + (Sl + fabs(Sm))*Els  - Sl*El  - Sr*Er);
      #ifdef DE
      f_ge = 0.5*(f_ge_l + f_ge_r + (Sr - fabs(Sm))*gers + (Sl + fabs(Sm))*gels - Sl*dl*gel - Sr*dr*ger);
      #endif


      // return the hllc fluxes
      dev_flux[           tid] = f_d;
      dev_flux[o1*n_cells+tid] = f_mx;
      dev_flux[o2*n_cells+tid] = f_my;
      dev_flux[o3*n_cells+tid] = f_mz;
      dev_flux[4*n_cells+tid]  = f_E;
      #ifdef DE
      dev_flux[5*n_cells+tid]  = f_ge;
      #endif

    }
  }

}


#endif //CUDA

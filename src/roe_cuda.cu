/*! \file roe_cuda.cu
 *  \brief Function definitions for the cuda Roe Riemann solver.*/

#ifdef CUDA

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"roe_cuda.h"



/*! \fn Calculate_Roe_Fluxes(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, Real *dev_etah, int dir)
 *  \brief Roe Riemann solver based on the version described in Stone et al, 2008. */
__global__ void Calculate_Roe_Fluxes(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, Real *dev_etah, int dir)
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

  Real etah = 0.0;
  Real g1 = gamma - 1.0; 
  Real Hl, Hr;
  Real sqrtdl, sqrtdr, vx, vy, vz, H;
  Real vsq, asq, a;
  Real lambda_m, lambda_0, lambda_p;
  Real f_d_l, f_mx_l, f_my_l, f_mz_l, f_E_l;
  Real f_d_r, f_mx_r, f_my_r, f_mz_r, f_E_r;
  Real del_d, del_mx, del_my, del_mz, del_E;
  Real a0, a1, a2, a3, a4;
  a0 = a1 = a2 = a3 = a4 = 0.0;
  Real sum_0, sum_1, sum_2, sum_3, sum_4;
  sum_0 = sum_1 = sum_2 = sum_3 = sum_4 = 0.0;
  Real test0, test1, test2, test3, test4;
  int hlle_flag = 0;
  #ifdef DE
  Real dgel, gel, dger, ger, f_ge_l, f_ge_r;
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
    dl  = dev_bounds_L[            tid];
    mxl = dev_bounds_L[o1*n_cells + tid];
    myl = dev_bounds_L[o2*n_cells + tid];
    mzl = dev_bounds_L[o3*n_cells + tid];
    El  = dev_bounds_L[4*n_cells + tid];
    #ifdef DE
    dgel = dev_bounds_L[5*n_cells + tid];
    #endif

    dr  = dev_bounds_R[            tid];
    mxr = dev_bounds_R[o1*n_cells + tid];
    myr = dev_bounds_R[o2*n_cells + tid];
    mzr = dev_bounds_R[o3*n_cells + tid];
    Er  = dev_bounds_R[4*n_cells + tid]; 
    #ifdef DE
    dger = dev_bounds_R[5*n_cells + tid];
    #endif


    // retrieve etah value
    etah = dev_etah[tid];

    // calculate primative variables
    vxl = mxl / dl;
    vyl = myl / dl;
    vzl = mzl / dl;
    pl  = (El - 0.5*dl*(vxl*vxl + vyl*vyl + vzl*vzl)) * (gamma - 1.0);
    pl  = fmax(pl, (Real) TINY_NUMBER);
    #ifdef DE
    gel = dgel / dl;
    #endif
    vxr = mxr / dr;
    vyr = myr / dr;
    vzr = mzr / dr;
    pr  = (Er - 0.5*dr*(vxr*vxr + vyr*vyr + vzr*vzr)) * (gamma - 1.0);
    pr  = fmax(pr, (Real) TINY_NUMBER);    
    #ifdef DE
    ger = dger / dr;
    #endif

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
    asq = g1*fmax((H - 0.5*vsq), TINY_NUMBER);
    a = sqrt(asq);

    // calculate the averaged eigenvectors of the Roe matrix (Stone Eqn B2, Toro 11.107)
    lambda_m = vx - a; 
    lambda_0 = vx;
    lambda_p = vx + a;
  
    // caclulate the fluxes for the left and right input states,
    // based on the average values in either cell
    f_d_l = mxl;
    f_mx_l = mxl*vxl + pl;
    f_my_l = mxl*vyl;
    f_mz_l = mxl*vzl;
    f_E_l = (El + pl)*vxl;
    #ifdef DE
    f_ge_l = mxl*gel;
    #endif

    f_d_r = mxr;
    f_mx_r = mxr*vxr + pr;
    f_my_r = mxr*vyr;
    f_mz_r = mxr*vzr;
    f_E_r = (Er + pr)*vxr;
    #ifdef DE
    f_ge_r = mxr*ger;
    #endif

    // return upwind flux if flow is supersonic
    if (lambda_m >= 0.0) {
      dev_flux[          tid] = f_d_l;
      dev_flux[o1*n_cells+tid] = f_mx_l;
      dev_flux[o2*n_cells+tid] = f_my_l;
      dev_flux[o3*n_cells+tid] = f_mz_l;
      dev_flux[4*n_cells+tid] = f_E_l;
      #ifdef DE
      dev_flux[5*n_cells+tid] = f_ge_l;
      #endif
      return;
    }
    else if (lambda_p <= 0.0) {
      dev_flux[          tid] = f_d_r;
      dev_flux[o1*n_cells+tid] = f_mx_r;
      dev_flux[o2*n_cells+tid] = f_my_r;
      dev_flux[o3*n_cells+tid] = f_mz_r;
      dev_flux[4*n_cells+tid] = f_E_r;
      #ifdef DE
      dev_flux[5*n_cells+tid] = f_ge_r;
      #endif
      return;
    }
    // otherwise calculate the Roe fluxes
    else {
    
      // calculate the difference in conserved variables across the cell interface
      // Stone Eqn 68
      del_d  = dr  - dl;
      del_mx = mxr - mxl;
      del_my = myr - myl;
      del_mz = mzr - mzl;
      del_E  = Er  - El;


      // evaluate the flux function (Stone Eqn 66 & 67, Toro Eqn 11.29)

      Real Na = 0.5/asq;
      Real coeff = 0.0;

      // left eigenvector [0] * del_q
      a0 = del_d*Na*(0.5*g1*vsq + vx*a) - del_mx*Na*(g1*vx+a) - del_my*Na*g1*vy - del_mz*Na*g1*vz + del_E*Na*g1;
      coeff = a0 * fmax(fabs(lambda_m), etah);
      sum_0 += coeff;
      sum_1 += coeff * (vx-a);
      sum_2 += coeff * vy;
      sum_3 += coeff * vz;
      sum_4 += coeff * (H - vx*a);
      // left eigenvector [1] * del_q
      a1 = -del_d*vy + del_my;
      coeff = a1 * fmax(fabs(lambda_0), etah);
      sum_2 += coeff;
      sum_4 += coeff * vy;
      // left eigenvector [2] * del_q
      a2 = -del_d*vz + del_mz;
      coeff = a2 * fmax(fabs(lambda_0), etah);
      sum_3 += coeff;
      sum_4 += coeff * vz;
      // left eigenvector [3] * del_q
      a3 = del_d*(1.0 - Na*g1*vsq) + del_mx*g1*vx/asq + del_my*g1*vy/asq + del_mz*g1*vz/asq - del_E*g1/asq;
      coeff = a3 * fmax(fabs(lambda_0), etah);
      sum_0 += coeff;
      sum_1 += coeff * vx;
      sum_2 += coeff * vy;
      sum_3 += coeff * vz;
      sum_4 += coeff * 0.5*vsq;
      // left eigenvector [4] * del_q
      a4 = del_d*Na*(0.5*g1*vsq - vx*a) - del_mx*Na*(g1*vx-a) - del_my*Na*g1*vy - del_mz*Na*g1*vz + del_E*Na*g1;
      coeff = a4 * fmax(fabs(lambda_p), etah);
      sum_0 += coeff;
      sum_1 += coeff * (vx + a);
      sum_2 += coeff * vy;
      sum_3 += coeff * vz;
      sum_4 += coeff * (H + vx*a);


      // if density or pressure is negative, compute the HLLE fluxes
      // test intermediate states
      test0 = dl + a0;
      test1 = mxl + a0*(vx-a);
      test2 = myl + a0*vy;
      test3 = mzl + a0*vz;
      test4 = El + a0*(H-vx*a);

      if(lambda_0 > lambda_m) {
        if (test0 <= 0.0) { 
          hlle_flag=1; 
        }
        if (test4 - 0.5*(test1*test1 + test2*test2 + test3*test3)/test0 < 0.0) {
          hlle_flag=2;
        }
      }

      test0 += a3 + a4;
      test1 += a3*vx;
      test2 += a1 + a3*vy;
      test3 += a2 + a3*vz;
      test4 += a1*vy + a2*vz + a3*0.5*vsq;

      if(lambda_p > lambda_0) {
        if (test0 <= 0.0) { 
          hlle_flag=1; 
        }
        if (test4 - 0.5*(test1*test1 + test2*test2 + test3*test3)/test0 < 0.0) {
          hlle_flag=2;
        }
      }

      // if pressure or density is negative, and we have not already returned the supersonic fluxes,
      // return the HLLE fluxes
      if (hlle_flag != 0) {

        Real cfl, cfr, al, ar, bm, bp, tmp;

        // compute max and fmin wave speeds
        cfl = sqrt(gamma*pl/dl);  // sound speed in left state
        cfr = sqrt(gamma*pr/dr);  // sound speed in right state

        // take max/fmin of Roe eigenvalues and left and right sound speeds
        al = fmin(lambda_m, vxl - cfl);
        ar = fmax(lambda_p, vxr + cfr);
    
        bm = fmin(al, (Real) 0.0);
        bp = fmax(ar, (Real) 0.0);

        // compute left and right fluxes
        f_d_l = mxl - bm*dl;
        f_d_r = mxr - bp*dr;

        f_mx_l = mxl*(vxl - bm) + pl;
        f_mx_r = mxr*(vxr - bp) + pr;

        f_my_l = myl*(vxl - bm);
        f_my_r = myr*(vxr - bp);

        f_mz_l = mzl*(vxl - bm);
        f_mz_r = mzr*(vxr - bp);

        f_E_l = El*(vxl - bm) + pl*vxl;
        f_E_r = Er*(vxr - bp) + pr*vxr;

        #ifdef DE
        f_ge_l = dgel*(vxl - bm);
        f_ge_r = dger*(vxr - bp);
        #endif

        // compute the HLLE flux at the interface
        tmp = 0.5*(bp + bm)/(bp - bm);

        dev_flux[          tid] = 0.5*(f_d_l  + f_d_r)  + (f_d_l  - f_d_r)*tmp; 
        dev_flux[o1*n_cells+tid] = 0.5*(f_mx_l + f_mx_r) + (f_mx_l - f_mx_r)*tmp; 
        dev_flux[o2*n_cells+tid] = 0.5*(f_my_l + f_my_r) + (f_my_l - f_my_r)*tmp; 
        dev_flux[o3*n_cells+tid] = 0.5*(f_mz_l + f_mz_r) + (f_mz_l - f_mz_r)*tmp; 
        dev_flux[4*n_cells+tid] = 0.5*(f_E_l  + f_E_r)  + (f_E_l  - f_E_r)*tmp;
        #ifdef DE
        dev_flux[5*n_cells+tid] = 0.5*(f_ge_l + f_ge_r) + (f_ge_l - f_ge_r)*tmp;
        #endif
        return;
      }
      // otherwise return the roe fluxes
      else {
        dev_flux[          tid] = 0.5*(f_d_l  + f_d_r  - sum_0);
        dev_flux[o1*n_cells+tid] = 0.5*(f_mx_l + f_mx_r - sum_1);
        dev_flux[o2*n_cells+tid] = 0.5*(f_my_l + f_my_r - sum_2);
        dev_flux[o3*n_cells+tid] = 0.5*(f_mz_l + f_mz_r - sum_3);
        dev_flux[4*n_cells+tid] = 0.5*(f_E_l  + f_E_r  - sum_4);
        #ifdef DE
        if (dev_flux[tid] >= 0.0)
          dev_flux[5*n_cells+tid] = dev_flux[tid] * gel;
        else
          dev_flux[5*n_cells+tid] = dev_flux[tid] * ger;
        #endif
      }

    }

  }

}


#endif //CUDA

/*! \file roe.cpp
 *  \brief Function definitions for the Roe Riemann solver. */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"global.h"
#include"roe.h"


/* \fn Calculate_Roe_Fluxes(Real cW[], Real fluxes[], Real gamma, Real etah)
 * \brief Returns the density, momentum, and Energy fluxes at an interface.
   Inputs are an array containg left and right density, momentum, and Energy. */
void Calculate_Roe_Fluxes(Real cW[], Real fluxes[], Real gamma, Real etah)
{
  Real dl, vxl, mxl, vyl, myl, vzl, mzl, pl, El;
  Real dr, vxr, mxr, vyr, myr, vzr, mzr, pr, Er;

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


  // calculate primative variables from input array
  dl = cW[0];
  dr = cW[1];
  mxl = cW[2];
  vxl = mxl / dl;
  mxr = cW[3];
  vxr = mxr / dr;
  myl = cW[4];
  vyl = myl / dl;
  myr = cW[5];
  vyr = myr / dr;
  mzl = cW[6];
  vzl = mzl / dl;
  mzr = cW[7];
  vzr = mzr / dr;
  El  = cW[8];
  pl = (El - 0.5*dl*(vxl*vxl + vyl*vyl + vzl*vzl)) * g1;
  pl = fmax(pl, TINY_NUMBER);
  Er = cW[9];
  pr = (Er - 0.5*dr*(vxr*vxr + vyr*vyr + vzr*vzr)) * g1;
  pr = fmax(pr, TINY_NUMBER);


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
  

  // calculate the fluxes for the left and right input states,
  // based on the average values in either cell
  f_d_l = mxl;
  f_mx_l = mxl*vxl + pl;
  f_my_l = mxl*vyl;
  f_mz_l = mxl*vzl;
  f_E_l = (El + pl)*vxl;

  f_d_r = mxr;
  f_mx_r = mxr*vxr + pr;
  f_my_r = mxr*vyr;
  f_mz_r = mxr*vzr;
  f_E_r = (Er + pr)*vxr;


  // return upwind flux if flow is supersonic
  if (lambda_m >= 0.0) {
    fluxes[0] = f_d_l;
    fluxes[1] = f_mx_l;
    fluxes[2] = f_my_l;
    fluxes[3] = f_mz_l;
    fluxes[4] = f_E_l;
    return;
  }
  else if (lambda_p <= 0.0) {
    fluxes[0] = f_d_r;
    fluxes[1] = f_mx_r;
    fluxes[2] = f_my_r;
    fluxes[3] = f_mz_r;
    fluxes[4] = f_E_r;
    return;
  }
  //otherwise return the roe flux
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


    // test intermediate states
    test0 = dl + a0;
    test1 = mxl + a0*(vx-a);
    test2 = myl + a0*vy;
    test3 = mzl + a0*vz;
    test4 = El + a0*(H-vx*a);

    // first characteristic
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

    // second characteristic
    if(lambda_p > lambda_0) {
      if (test0 <= 0.0) { 
        hlle_flag=1; 
      }
      if (test4 - 0.5*(test1*test1 + test2*test2 + test3*test3)/test0 < 0.0) {
        hlle_flag=2;
      }
    }


    // if density or pressure is negative, compute the HLLE fluxes
    if (hlle_flag != 0) {

      Real cfl, cfr, al, ar, bm, bp, tmp;

      // compute max and min wave speeds
      cfl = sqrt(gamma*pl/dl);  // sound speed in left state
      cfr = sqrt(gamma*pr/dr);  // sound speed in right state

      // take max/min of Roe eigenvalues and left and right sound speeds
      al = fmin(lambda_m, vxl - cfl);
      ar = fmax(lambda_p, vxr + cfr);

      bm = fmin(al, 0.0);
      bp = fmax(ar, 0.0);

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

      // compute the HLLE flux at the interface
      tmp = 0.5*(bp + bm)/(bp - bm);

      fluxes[0] = 0.5*(f_d_l  + f_d_r)  + (f_d_l  - f_d_r)*tmp; 
      fluxes[1] = 0.5*(f_mx_l + f_mx_r) + (f_mx_l - f_mx_r)*tmp; 
      fluxes[2] = 0.5*(f_my_l + f_my_r) + (f_my_l - f_my_r)*tmp; 
      fluxes[3] = 0.5*(f_mz_l + f_mz_r) + (f_mz_l - f_mz_r)*tmp; 
      fluxes[4] = 0.5*(f_E_l  + f_E_r)  + (f_E_l  - f_E_r)*tmp; 
    }
    // otherwise return the Roe fluxes
    else { 
      fluxes[0] = 0.5*(f_d_l  + f_d_r  - sum_0);
      fluxes[1] = 0.5*(f_mx_l + f_mx_r - sum_1);
      fluxes[2] = 0.5*(f_my_l + f_my_r - sum_2);
      fluxes[3] = 0.5*(f_mz_l + f_mz_r - sum_3);
      fluxes[4] = 0.5*(f_E_l  + f_E_r  - sum_4);
    }      

  }



}


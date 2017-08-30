/*! \file hllc.cpp
 *  \brief Function definitions for the HLLC Riemann solver.*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"global.h"
#include"hllc.h"



/*! \fn Calculate_HLLC_Fluxes(Real cW[], Real fluxes[], Real gamma, Real etah, int dir)
 *  \brief HLLC Riemann solver based on the version described in Toro (2006), Sec. 10.4. */
void Calculate_HLLC_Fluxes(Real cW[], Real fluxes[], Real gamma, Real etah, int dir)
{
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

  // calculate primative variables from input array
  dl = cW[0];
  dr = cW[1];
  if (dir == 0) {
    mxl = cW[2];
    mxr = cW[3];
    myl = cW[4];
    myr = cW[5];
    mzl = cW[6];
    mzr = cW[7];
  }
  if (dir == 1) {
    mxl = cW[4];
    mxr = cW[5];
    myl = cW[6];
    myr = cW[7];
    mzl = cW[2];
    mzr = cW[3];
  }
  if (dir == 2) {
    mxl = cW[6];
    mxr = cW[7];
    myl = cW[2];
    myr = cW[3];
    mzl = cW[4];
    mzr = cW[5];
  }
  vxl = mxl / dl;
  vxr = mxr / dr;
  vyl = myl / dl;
  vyr = myr / dr;
  vzl = mzl / dl;
  vzr = mzr / dr;
  El  = cW[8];
  pl = (El - 0.5*dl*(vxl*vxl + vyl*vyl + vzl*vzl)) * g1;
  pl = fmax(pl, TINY_NUMBER);
  Er  = cW[9];
  pr = (Er - 0.5*dr*(vxr*vxr + vyr*vyr + vzr*vzr)) * g1;
  pr = fmax(pr, TINY_NUMBER);
  #ifdef DE
  gel = cW[10] / dl;
  ger = cW[11] / dr;
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
  Sl = sgn(Sl)*fmax(fabs(Sl), etah);
  Sr = sgn(Sr)*fmax(fabs(Sr), etah);


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
    fluxes[0] = f_d_l;
    if (dir == 0 ) {
      fluxes[1] = f_mx_l;
      fluxes[2] = f_my_l;
      fluxes[3] = f_mz_l;
    }
    if (dir == 1 ) {
      fluxes[1] = f_my_l;
      fluxes[2] = f_mz_l;
      fluxes[3] = f_mx_l;
    }
    if (dir == 2 ) {
      fluxes[1] = f_mz_l;
      fluxes[2] = f_mx_l;
      fluxes[3] = f_my_l;
    fluxes[4] = f_E_l;
    #ifdef DE
    fluxes[5] = f_ge_l;
    #endif
    return;
  }
  else if (Sr < 0.0) {
    fluxes[0] = f_d_r;
    if (dir == 0) {
      fluxes[1] = f_mx_r;
      fluxes[2] = f_my_r;
      fluxes[3] = f_mz_r;
    }
    if (dir == 1) {
      fluxes[1] = f_my_r;
      fluxes[2] = f_mz_r;
      fluxes[3] = f_mx_r;
    }
    if (dir == 2) {
      fluxes[1] = f_mz_r;
      fluxes[2] = f_mx_r;
      fluxes[3] = f_my_r;
    }
    fluxes[4] = f_E_r;
    #ifdef DE
    fluxes[5]  = f_ge_r;
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
    f_ge = 0.5*(f_ge_l + f_ge_r + (Sr - fabs(Sm))*gers + (Sl + fabs(Sm))*gels - Sl*gel - Sr*ger);
    #endif


    // return the hllc fluxes
    fluxes[0] = f_d;
    if (dir == 0) {
      fluxes[1] = f_mx;
      fluxes[2] = f_my;
      fluxes[3] = f_mz;
    }
    if (dir == 1) {
      fluxes[1] = f_my;
      fluxes[2] = f_mz;
      fluxes[3] = f_mx;
    }
    if (dir == 2) {
      fluxes[1] = f_mz;
      fluxes[2] = f_mx;
      fluxes[3] = f_my;
    }
    fluxes[4]  = f_E;
    #ifdef DE
    fluxes[5]  = f_ge;
    #endif

  }

}


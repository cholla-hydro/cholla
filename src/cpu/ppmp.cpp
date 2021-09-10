/*! \file ppmp.cpp
 *  \brief Definitions of the PPM reconstruction functions, written following Fryxell et al., 2000 */
#ifndef CUDA
#ifdef PPMP

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../cpu/ppmp.h"
#include "../global/global.h"

#define STEEPENING
#define FLATTENING


/*! \fn void ppmp(Real stencil[], Real bounds[], Real dx, Real dt)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using ppm. */
void ppmp(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
{
  // retrieve the values from the stencil
  Real d_i = stencil[0];
  Real vx_i = stencil[1] / d_i;
  Real vy_i = stencil[2] / d_i;
  Real vz_i = stencil[3] / d_i;
  Real p_i = (stencil[4] - 0.5*d_i*(vx_i*vx_i + vy_i*vy_i + vz_i*vz_i)) * (gamma-1.0);
  p_i = fmax(p_i, TINY_NUMBER);
  Real d_imo = stencil[5];
  Real vx_imo = stencil[6] / d_imo;
  Real vy_imo = stencil[7] / d_imo;
  Real vz_imo = stencil[8] / d_imo;
  Real p_imo = (stencil[9] - 0.5*d_imo*(vx_imo*vx_imo + vy_imo*vy_imo + vz_imo*vz_imo)) * (gamma-1.0);
  p_imo = fmax(p_imo, TINY_NUMBER);
  Real d_ipo = stencil[10];
  Real vx_ipo = stencil[11] / d_ipo;
  Real vy_ipo = stencil[12] / d_ipo;
  Real vz_ipo = stencil[13] / d_ipo;
  Real p_ipo = (stencil[14] - 0.5*d_ipo*(vx_ipo*vx_ipo + vy_ipo*vy_ipo + vz_ipo*vz_ipo)) * (gamma-1.0);
  p_ipo = fmax(p_ipo, TINY_NUMBER);
  Real d_imt = stencil[15];
  Real vx_imt = stencil[16] / d_imt;
  Real vy_imt = stencil[17] / d_imt;
  Real vz_imt = stencil[18] / d_imt;
  Real p_imt = (stencil[19] - 0.5*d_imt*(vx_imt*vx_imt + vy_imt*vy_imt + vz_imt*vz_imt)) * (gamma-1.0);
  p_imt = fmax(p_imt, TINY_NUMBER);
  Real d_ipt = stencil[20];
  Real vx_ipt = stencil[21] / d_ipt;
  Real vy_ipt = stencil[22] / d_ipt;
  Real vz_ipt = stencil[23] / d_ipt;
  Real p_ipt = (stencil[24] - 0.5*d_ipt*(vx_ipt*vx_ipt + vy_ipt*vy_ipt + vz_ipt*vz_ipt)) * (gamma-1.0);
  p_ipt = fmax(p_ipt, TINY_NUMBER);
  Real p_imth = (stencil[29] - 0.5*(stencil[26]*stencil[26] + stencil[27]*stencil[27] + stencil[28]*stencil[28])/stencil[25]) * (gamma-1.0);
  p_imth = fmax(p_imth, TINY_NUMBER);
  Real p_ipth = (stencil[34] - 0.5*(stencil[31]*stencil[31] + stencil[32]*stencil[32] + stencil[33]*stencil[33])/stencil[30]) * (gamma-1.0);
  p_ipth = fmax(p_ipth, TINY_NUMBER);
  Real dx_i, dx_imo, dx_ipo, dx_imt, dx_ipt;
  dx_i = dx_imo = dx_ipo = dx_imt = dx_ipt = dx;

  // declare the primative variables we are calculating
  // note: dl & dr refer to the left and right boundary values of cell i
  Real dl, dr, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr;

  //define constants to use in contact discontinuity detection
  Real d2_rho_imo, d2_rho_ipo; //second derivative of rho
  Real eta_i; //steepening coefficient derived from a dimensionless quantity relating
              //the first and third derivatives of  the density (Fryxell Eqns 36 & 40)
  Real delta_m_imo, delta_m_ipo; //monotonized slopes (Fryxell Eqn 26) 
  //define constants to use in shock flattening
  Real F_imo, F_i, F_ipo; //flattening coefficients (Fryxell Eqn 48)
  // define other constants
  Real cs, cl, cr; // sound speed in cell i, and at left and right boundaries
  Real del_d, del_vx, del_vy, del_vz, del_p; // "slope" accross cell i
  Real d_6, vx_6, vy_6, vz_6, p_6;
  Real beta_m, beta_0, beta_p;
  Real alpha_m, alpha_0, alpha_p;
  Real lambda_m, lambda_0, lambda_p; // speed of characteristics
  Real dL_m, dL_0;
  Real vxL_m, vyL_0, vzL_0, vxL_p;
  Real pL_m, pL_0, pL_p;
  Real dR_0, dR_p;
  Real vxR_m, vyR_0, vzR_0, vxR_p;
  Real pR_m, pR_0, pR_p;
  Real chi_L_m, chi_L_0, chi_L_p;
  Real chi_R_m, chi_R_0, chi_R_p;

  // use ppm routines to set cell boundary values (see Fryxell Sec. 3.1.1)
  // left
  dl  = interface_value(d_imt,  d_imo,  d_i,  d_ipo,  dx_imt, dx_imo, dx_i, dx_ipo);
  vxl = interface_value(vx_imt, vx_imo, vx_i, vx_ipo, dx_imt, dx_imo, dx_i, dx_ipo);
  vyl = interface_value(vy_imt, vy_imo, vy_i, vy_ipo, dx_imt, dx_imo, dx_i, dx_ipo);
  vzl = interface_value(vz_imt, vz_imo, vz_i, vz_ipo, dx_imt, dx_imo, dx_i, dx_ipo);
  pl  = interface_value(p_imt,  p_imo,  p_i,  p_ipo,  dx_imt, dx_imo, dx_i, dx_ipo);
  // right
  dr  = interface_value(d_imo,  d_i,  d_ipo,  d_ipt,  dx_imo, dx_i, dx_ipo, dx_ipt);
  vxr = interface_value(vx_imo, vx_i, vx_ipo, vx_ipt, dx_imo, dx_i, dx_ipo, dx_ipt);
  vyr = interface_value(vy_imo, vy_i, vy_ipo, vy_ipt, dx_imo, dx_i, dx_ipo, dx_ipt);
  vzr = interface_value(vz_imo, vz_i, vz_ipo, vz_ipt, dx_imo, dx_i, dx_ipo, dx_ipt);
  pr  = interface_value(p_imo,  p_i,  p_ipo,  p_ipt,  dx_imo, dx_i, dx_ipo, dx_ipt);


#ifdef STEEPENING
  //check for contact discontinuities & steepen if necessary (see Fryxell Sec 3.1.2)
  //if condition 4 (Fryxell Eqn 37) is true, check further conditions, otherwise do nothing
  if ((fabs(d_ipo - d_imo) / fmin(d_ipo, d_imo)) > 0.01)
  {
    //calculate the second derivative of the density in the imo and ipo cells
    d2_rho_imo = calc_d2_rho(d_imt, d_imo, d_i, dx, dx, dx);
    d2_rho_ipo = calc_d2_rho(d_i, d_ipo, d_ipt, dx, dx, dx);
    //if condition 1 (Fryxell Eqn 38) is true, check further conditions, otherwise do nothing
    if ((d2_rho_imo * d2_rho_ipo) < 0)
    {
      //calculate condition 5, pressure vs density jumps (Fryxell Eqn 39)
      //if c5 is true, set value of eta for discontinuity steepening
      if ((fabs(p_ipo - p_imo) / fmin(p_ipo, p_imo)) < 0.1 * gamma * (fabs(d_ipo - d_imo) / fmin(d_ipo, d_imo)))
      { 
        //calculate first eta value (Fryxell Eqn 36)
        eta_i = calc_eta(d2_rho_imo, d2_rho_ipo, dx, dx, dx, d_imo, d_ipo);
        //calculate steepening coefficient (Fryxell Eqn 40)
        eta_i = fmax(0, fmin(20*(eta_i-0.05), 1) );

        //calculate new left and right interface variables using monotonized slopes
        delta_m_imo = calc_delta_q(d_imt, d_imo, d_i, dx, dx, dx);
        delta_m_imo = limit_delta_q(delta_m_imo, d_imt, d_imo, d_i);
        delta_m_ipo = calc_delta_q(d_i, d_ipo, d_ipt, dx, dx, dx);
        delta_m_ipo = limit_delta_q(delta_m_ipo, d_i, d_ipo, d_ipt);

        //replace left and right interface values of density
        dl = dl*(1-eta_i) + (d_imo + 0.5 * delta_m_imo) * eta_i;
        dr = dr*(1-eta_i) + (d_ipo - 0.5 * delta_m_ipo) * eta_i;
      }
    }
  }
#endif

#ifdef FLATTENING
  //flatten shock fronts that are too narrow (see Fryxell Sec 3.1.3)
  //calculate the shock steepness parameters (Fryxell Eqn 43)
  //calculate the dimensionless flattening coefficients (Fryxell Eqn 45)
  F_imo = fmax( 0, fmin(1, 10*(((p_i - p_imt) / (p_ipo - p_imth))-0.75)) );
  F_i = fmax( 0, fmin(1, 10*(((p_ipo - p_imo) / (p_ipt - p_imt))-0.75)) );
  F_ipo = fmax( 0, fmin(1, 10*(((p_ipt - p_i) / (p_ipth - p_imo))-0.75)) );
  //ensure that we are encountering a shock (Fryxell Eqns 46 & 47)
  if (fabs(p_i - p_imt) / fmin(p_i, p_imt) < 1./3.)  {F_imo = 0;}
  if (fabs(p_ipo - p_imo) / fmin(p_ipo, p_imo) < 1./3.)  {F_i = 0;}
  if (fabs(p_ipt - p_i) / fmin(p_ipt, p_i) < 1./3.)  {F_ipo = 0;}
  if (vx_i - vx_imt > 0) {F_imo = 0;}
  if (vx_ipo - vx_imo > 0) {F_i = 0;}
  if (vx_ipt - vx_i > 0) {F_ipo = 0;}
  //set the flattening coefficient (Fryxell Eqn 48)
  if (p_ipo - p_imo < 0) {F_i = fmax(F_i, F_ipo);}
  else {F_i = fmax(F_i, F_imo);}
  //modify the interface values
  dl  = F_i * d_i  + (1 - F_i) * dl;
  vxl = F_i * vx_i + (1 - F_i) * vxl;
  vyl = F_i * vy_i + (1 - F_i) * vyl;
  vzl = F_i * vz_i + (1 - F_i) * vzl;
  pl  = F_i * p_i  + (1 - F_i) * pl;
  dr  = F_i * d_i  + (1 - F_i) * dr;
  vxr = F_i * vx_i + (1 - F_i) * vxr;
  vyr = F_i * vy_i + (1 - F_i) * vyr;
  vzr = F_i * vz_i + (1 - F_i) * vzr;
  pr  = F_i * p_i  + (1 - F_i) * pr;
#endif

  //ensure that the parabolic distribution of each of the primative variables is monotonic
  //local maximum or minimum criterion (Fryxell Eqn 52, Fig 11)
  if ( (dr  - d_i)  * (d_i  - dl)  <= 0)  { dl  = dr  = d_i; }
  if ( (vxr - vx_i) * (vx_i - vxl) <= 0)  { vxl = vxr = vx_i; }
  if ( (vyr - vy_i) * (vy_i - vyl) <= 0)  { vyl = vyr = vy_i; }
  if ( (vzr - vz_i) * (vz_i - vzl) <= 0)  { vzl = vzr = vz_i; }
  if ( (pr  - p_i)  * (p_i  - pl)  <= 0)  { pl  = pr  = p_i; }
  //steep gradient criterion (Fryxell Eqn 53, Fig 12)
  if ( (dr  - dl)  * (dl -  (3*d_i  - 2*dr))  < 0)  { dl  = 3*d_i  - 2*dr;  }
  if ( (vxr - vxl) * (vxl - (3*vx_i - 2*vxr)) < 0)  { vxl = 3*vx_i - 2*vxr; }
  if ( (vyr - vyl) * (vyl - (3*vy_i - 2*vyr)) < 0)  { vyl = 3*vy_i - 2*vyr; }
  if ( (vzr - vzl) * (vzl - (3*vz_i - 2*vzr)) < 0)  { vzl = 3*vz_i - 2*vzr; }
  if ( (pr  - pl)  * (pl -  (3*p_i  - 2*pr))  < 0)  { pl  = 3*p_i  - 2*pr;  }
  if ( (dr  - dl)  * ((3*d_i  - 2*dl)  - dr)  < 0)  { dr  = 3*d_i  - 2*dl;  }
  if ( (vxr - vxl) * ((3*vx_i - 2*vxl) - vxr) < 0)  { vxr = 3*vx_i - 2*vxl; }
  if ( (vyr - vyl) * ((3*vy_i - 2*vyl) - vyr) < 0)  { vyr = 3*vy_i - 2*vyl; }
  if ( (vzr - vzl) * ((3*vz_i - 2*vzl) - vzr) < 0)  { vzr = 3*vz_i - 2*vzl; }
  if ( (pr  - pl)  * ((3*p_i  - 2*pl)  - pr)  < 0)  { pr  = 3*p_i  - 2*pl;  }


  // compute sound speed in cell i
  cs = sqrt(gamma * p_i / d_i);

  // compute a first guess at the left and right states by taking the average
  // under the characteristic on each side that has the largest speed

  // compute deltas and 'sixes' (Fryxell Eqns 29 & 30)
  del_d  = dr  - dl;  // Fryxell Eqn 29
  del_vx = vxr - vxl; // Fryxell Eqn 29
  del_vy = vyr - vyl; // Fryxell Eqn 29
  del_vz = vzr - vzl; // Fryxell Eqn 29
  del_p  = pr  - pl;  // Fryxell Eqn 29

  d_6  = 6.0 * (d_i  - 0.5*(dl  + dr));  // Fryxell Eqn 30
  vx_6 = 6.0 * (vx_i - 0.5*(vxl + vxr)); // Fryxell Eqn 30
  vy_6 = 6.0 * (vy_i - 0.5*(vyl + vyr)); // Fryxell Eqn 30
  vz_6 = 6.0 * (vz_i - 0.5*(vzl + vzr)); // Fryxell Eqn 30
  p_6  = 6.0 * (p_i  - 0.5*(pl  + pr));  // Fryxell Eqn 30

  // set speed of characteristics (v-c, v, v+c) using average values of v and c
  lambda_m = vx_i - cs;
  lambda_0 = vx_i;
  lambda_p = vx_i + cs;

  // calculate betas (for right interface guesses)
  beta_m = fmax( (lambda_m * dt / dx_i) , 0 ); // Fryxell Eqn 59
  beta_0 = fmax( (lambda_0 * dt / dx_i) , 0 ); // Fryxell Eqn 59
  beta_p = fmax( (lambda_p * dt / dx_i) , 0 ); // Fryxell Eqn 59
 
  // calculate alphas (for left interface guesses)
  alpha_m = fmax( (-lambda_m * dt / dx_i), 0); // Fryxell Eqn 61
  alpha_0 = fmax( (-lambda_0 * dt / dx_i), 0); // Fryxell Eqn 61
  alpha_p = fmax( (-lambda_p * dt / dx_i), 0); // Fryxell Eqn 61

  // average values under characteristics for left interface (Fryxell Eqn 60)
  dL_m  = dl  + 0.5 * alpha_m * (del_d  + d_6  * (1 - (2./3.) * alpha_m));
  vxL_m = vxl + 0.5 * alpha_m * (del_vx + vx_6 * (1 - (2./3.) * alpha_m));
  pL_m  = pl  + 0.5 * alpha_m * (del_p  + p_6  * (1 - (2./3.) * alpha_m));
  dL_0  = dl  + 0.5 * alpha_0 * (del_d  + d_6  * (1 - (2./3.) * alpha_0));
  vyL_0 = vyl + 0.5 * alpha_0 * (del_vy + vy_6 * (1 - (2./3.) * alpha_0));
  vzL_0 = vzl + 0.5 * alpha_0 * (del_vz + vz_6 * (1 - (2./3.) * alpha_0));
  pL_0  = pl  + 0.5 * alpha_0 * (del_p  + p_6  * (1 - (2./3.) * alpha_0));
  vxL_p = vxl + 0.5 * alpha_p * (del_vx + vx_6 * (1 - (2./3.) * alpha_p));
  pL_p  = pl  + 0.5 * alpha_p * (del_p  + p_6  * (1 - (2./3.) * alpha_p));

  // average values under characteristics for right interface (Fryxell Eqn 58)
  vxR_m = vxr - 0.5 * beta_m * (del_vx - vx_6 * (1 - (2./3.) * beta_m));
  pR_m  = pr  - 0.5 * beta_m * (del_p  - p_6  * (1 - (2./3.) * beta_m));
  dR_0  = dr  - 0.5 * beta_0 * (del_d  - d_6  * (1 - (2./3.) * beta_0));
  vyR_0 = vyr - 0.5 * beta_0 * (del_vy - vy_6 * (1 - (2./3.) * beta_0));
  vzR_0 = vzr - 0.5 * beta_0 * (del_vz - vz_6 * (1 - (2./3.) * beta_0));
  pR_0  = pr  - 0.5 * beta_0 * (del_p  - p_6  * (1 - (2./3.) * beta_0));
  dR_p  = dr  - 0.5 * beta_p * (del_d  - d_6  * (1 - (2./3.) * beta_p));
  vxR_p = vxr - 0.5 * beta_p * (del_vx - vx_6 * (1 - (2./3.) * beta_p));
  pR_p  = pr  - 0.5 * beta_p * (del_p  - p_6  * (1 - (2./3.) * beta_p));

  // as a first guess, use characteristics with the largest speeds
  // for transverse velocities, use the 0 characteristic
  // left
  dl  = dL_m;
  vxl = vxL_m;
  vyl = vyL_0;
  vzl = vzL_0;
  pl  = pL_m;
  // right
  dr  = dR_p;
  vxr = vxR_p;
  vyr = vyR_0;
  vzr = vzR_0;
  pr  = pR_p;

  // correct these initial guesses by taking into account the number of 
  // characteristics on each side of the interface

  // calculate the 'guess' sound speeds 
  cl = sqrt(gamma * pl / dl);
  cr = sqrt(gamma * pr / dr);

  // calculate the chi values (Fryxell Eqns 62 & 63)
  chi_L_m =  1./(2*dl*cl) * (vxl - vxL_m - (pl - pL_m)/(dl*cl));
  chi_L_p = -1./(2*dl*cl) * (vxl - vxL_p + (pl - pL_p)/(dl*cl));
  chi_L_0 = (pl - pL_0)/(dl*dl*cl*cl) + 1./dl - 1./dL_0;
  chi_R_m =  1./(2*dr*cr) * (vxr - vxR_m - (pr - pR_m)/(dr*cr));
  chi_R_p = -1./(2*dr*cr) * (vxr - vxR_p + (pr - pR_p)/(dr*cr));
  chi_R_0 = (pr - pR_0)/(dr*dr*cr*cr) + 1./dr - 1./dR_0;

  // set chi to 0 if characteristic velocity has the wrong sign (Fryxell Eqn 64)
  if (lambda_m >= 0) { chi_L_m = 0; }
  if (lambda_0 >= 0) { chi_L_0 = 0; }
  if (lambda_p >= 0) { chi_L_p = 0; }
  if (lambda_m <= 0) { chi_R_m = 0; }
  if (lambda_0 <= 0) { chi_R_0 = 0; }
  if (lambda_p <= 0) { chi_R_p = 0; }

  // use the chi values to correct the initial guesses and calculate final input states
  pl = pl + (dl*dl * cl*cl) * (chi_L_p + chi_L_m);
  vxl = vxl + dl * cl * (chi_L_p - chi_L_m);
  dl = pow( ((1.0/dl) - (chi_L_m + chi_L_0 + chi_L_p)) , -1);
  pr = pr + (dr*dr * cr*cr) * (chi_R_p + chi_R_m);
  vxr = vxr + dr * cr * (chi_R_p - chi_R_m);
  dr = pow( ((1.0/dr) - (chi_R_m + chi_R_0 + chi_R_p)) , -1);

  // send final values back  (conserved variables)
  bounds[0] = dl;
  bounds[1] = dl*vxl;
  bounds[2] = dl*vyl;
  bounds[3] = dl*vzl;
  bounds[4] = (pl/(gamma-1.0)) + 0.5*dl*(vxl*vxl + vyl*vyl + vzl*vzl);
  bounds[5] = dr;
  bounds[6] = dr*vxr;
  bounds[7] = dr*vyr;
  bounds[8] = dr*vzr;
  bounds[9] = (pr/(gamma-1.0)) + 0.5*dr*(vxr*vxr + vyr*vyr + vzr*vzr);

}


/*! \fn interface_value 
 *  \brief Returns the interpolated value at the right hand interface of cell i.*/
Real interface_value(Real q_imo, Real q_i, Real q_ipo, Real q_ipt,
                 Real dx_imo, Real dx_i, Real dx_ipo, Real dx_ipt)
{
    Real Z_1;
    Real Z_2;
    Real X;
    Real dq_i;
    Real dq_ipo;
    Real q_R;

    Z_1 = (dx_imo + dx_i) / (2*dx_i + dx_ipo);
    Z_2 = (dx_ipt + dx_ipo) / (2*dx_ipo + dx_i);
    X = dx_imo + dx_i + dx_ipo + dx_ipt;

    dq_i = calc_delta_q(q_imo, q_i, q_ipo, dx_imo, dx_i, dx_ipo);
    dq_ipo = calc_delta_q(q_i, q_ipo, q_ipt, dx_i, dx_ipo, dx_ipt);

    dq_i = limit_delta_q(dq_i, q_imo, q_i, q_ipo);
    dq_ipo = limit_delta_q(dq_ipo, q_i, q_ipo, q_ipt);

    q_R = q_i + (dx_i / (dx_i + dx_ipo))*(q_ipo - q_i) + 
    (1 / X)*( ((2*dx_ipo*dx_i)/(dx_ipo+dx_i)) * (Z_1 - Z_2) * (q_ipo - q_i) - 
     dx_i*Z_1*dq_ipo + dx_ipo*Z_2*dq_i );

    return q_R;
}

/*! \fn calc_delta_q
 *  \brief Returns the average slope in zone i of the parabola with zone averages
     of imo, i, and ipo. See Fryxell Eqn 24. */
Real calc_delta_q(Real q_imo, Real q_i, Real q_ipo, 
                Real dx_imo, Real dx_i, Real dx_ipo)
{

    Real A;
    Real B;
    Real C;
    Real D;
    Real E;

    A = dx_i / (dx_imo + dx_i + dx_ipo);
    B = (2*dx_imo + dx_i) / (dx_ipo + dx_i);
    C = q_ipo - q_i;
    D = (dx_i + 2*dx_ipo) / (dx_imo + dx_i);
    E = q_i - q_imo;

    return A * (B*C + D*E);
}


/*! \fn limit_delta_q
 *  \brief Limits the value of delta_rho according to Fryxell Eqn 26
     to ensure monotonic interface values. */
Real limit_delta_q(Real del_in, Real q_imo, Real q_i, Real q_ipo)
{
  Real dq;

    if ( (q_ipo-q_i)*(q_i-q_imo) > 0)
    {
      dq = fmin(2.0*fabs(q_i-q_imo), 2.0*fabs(q_i - q_ipo));
      dq = fmin(dq, fabs(del_in));
      return sgn(del_in)*dq;
    }
    else return 0;
}


/*! \fn test_interface_value 
 *  \brief Returns the right hand interpolated value at imo | i cell interface,
     assuming equal cell widths. */
Real test_interface_value(Real q_imo, Real q_i, Real q_ipo, Real q_ipt)
{
    return (1./12.)*(-q_ipt + 7*q_ipo + 7*q_i - q_imo);
}


/*! \fn calc_d2_rho
 *  \brief Returns the second derivative of rho across zone i. (Fryxell Eqn 35) */
Real calc_d2_rho(Real rho_imo, Real rho_i, Real rho_ipo,
    Real dx_imo, Real dx_i, Real dx_ipo)
{
    Real A;
    Real B;
    Real C;

    A = 1 / (dx_imo + dx_i + dx_ipo);
    B = (rho_ipo - rho_i) / (dx_ipo + dx_i);
    C = (rho_i - rho_imo) / (dx_i + dx_imo);

    return A*(B - C);
} 


/*! \fn calc_eta
 *  \brief Returns a dimensionless quantity relating the 1st and 3rd derivatives
    See Fryxell Eqn 36. */
Real calc_eta(Real d2rho_imo, Real d2rho_ipo, Real dx_imo, Real dx_i, Real dx_ipo,
    Real rho_imo, Real rho_ipo)
{
    Real x_imo = 0.5*dx_imo;
    Real x_i = x_imo + 0.5*dx_imo + 0.5*dx_i;
    Real x_ipo = x_i + 0.5*dx_i + 0.5*dx_ipo;

    Real A;
    Real B;

    A = (d2rho_ipo - d2rho_imo) / (x_ipo - x_imo);
    B = (pow((x_i - x_imo),3) + pow((x_ipo - x_i),3)) / (rho_ipo - rho_imo);

   return -A * B;
}

#endif //PPMP
#endif //CUDA

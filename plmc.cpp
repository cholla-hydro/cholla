/*! \file plmc.cpp
 *  \brief Definitions of the piecewise linear reconstruction functions, as decribed
           in Stone et al., 2008. */
#ifndef CUDA
#ifdef PLMC

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"plmc.h"
#include"global.h"


/*! \fn plmc(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
*  \brief Use a piece-wise linear method to calculate boundary values for each cell. */
void plmc(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
{
  Real d_i, vx_i, vy_i, vz_i, p_i;
  Real d_imo, vx_imo, vy_imo, vz_imo, p_imo;
  Real d_ipo, vx_ipo, vy_ipo, vz_ipo, p_ipo;

  // retrieve the values from the stencil
  d_i = stencil[0];
  vx_i = stencil[1] / d_i;
  vy_i = stencil[2] / d_i;
  vz_i = stencil[3] / d_i;
  p_i = (stencil[4] - 0.5*d_i*(vx_i*vx_i + vy_i*vy_i + vz_i*vz_i)) * (gamma-1.0);
  p_i = fmax(p_i, TINY_NUMBER);
  d_imo = stencil[5];
  vx_imo = stencil[6] / d_imo;
  vy_imo = stencil[7] / d_imo;
  vz_imo = stencil[8] / d_imo;
  p_imo = (stencil[9] - 0.5*d_imo*(vx_imo*vx_imo + vy_imo*vy_imo + vz_imo*vz_imo)) * (gamma-1.0);
  p_imo = fmax(p_imo, TINY_NUMBER);
  d_ipo = stencil[10];
  vx_ipo = stencil[11] / d_ipo;
  vy_ipo = stencil[12] / d_ipo;
  vz_ipo = stencil[13] / d_ipo;
  p_ipo = (stencil[14] - 0.5*d_ipo*(vx_ipo*vx_ipo + vy_ipo*vy_ipo + vz_ipo*vz_ipo)) * (gamma-1.0);	
  p_ipo = fmax(p_ipo, TINY_NUMBER);

  Real dtodx = dt/dx;

  // adiabatic sound speed in cell imo, i, ipo
  Real a_i   = sqrt(gamma*p_i/d_i);


  // Step 1 - Compute the eigenvalues and eigenvectors of the linearized
  //          equations in the primative variables using the cell-centered
  //          primative variables
  //          (just compute eigenvalues; deal with eigenvectors later)
  Real lambda_m = vx_i-a_i;
  Real lambda_0 = vx_i;
  Real lambda_p = vx_i+a_i;


  // Step 2 - Compute the left, right, and centered differences of the primative variables
  //          Note that here L and R refer to locations relative to the cell center
  //          Stone Eqn 36

  Real del_d_L, del_vx_L, del_vy_L, del_vz_L, del_p_L;
  Real del_d_R, del_vx_R, del_vy_R, del_vz_R, del_p_R;
  Real del_d_C, del_vx_C, del_vy_C, del_vz_C, del_p_C;
  Real del_d_G, del_vx_G, del_vy_G, del_vz_G, del_p_G;

  // Left
  del_d_L  = d_i  - d_imo;
  del_vx_L = vx_i - vx_imo;
  del_vy_L = vy_i - vy_imo;
  del_vz_L = vz_i - vz_imo;
  del_p_L  = p_i  - p_imo;

  // Right
  del_d_R  = d_ipo  - d_i;
  del_vx_R = vx_ipo - vx_i;
  del_vy_R = vy_ipo - vy_i;
  del_vz_R = vz_ipo - vz_i;
  del_p_R  = p_ipo  - p_i;
  
  // Centered
  del_d_C  = d_ipo  - d_imo;
  del_vx_C = vx_ipo - vx_imo;
  del_vy_C = vy_ipo - vy_imo;
  del_vz_C = vz_ipo - vz_imo;
  del_p_C  = p_ipo  - p_imo;
  
  // Van Leer
  if (del_d_L*del_d_R > 0.0)   { del_d_G  = 2.0*del_d_L*del_d_R   / (del_d_L+del_d_R); }
  else { del_d_G = 0.0; }
  if (del_vx_L*del_vx_R > 0.0) { del_vx_G = 2.0*del_vx_L*del_vx_R / (del_vx_L+del_vx_R); }
  else { del_vx_G = 0.0; } 
  if (del_vy_L*del_vy_R > 0.0) { del_vy_G = 2.0*del_vy_L*del_vy_R / (del_vy_L+del_vy_R); }
  else { del_vy_G = 0.0; } 
  if (del_vz_L*del_vz_R > 0.0) { del_vz_G = 2.0*del_vz_L*del_vz_R / (del_vz_L+del_vz_R); }
  else { del_vz_G = 0.0; } 
  if (del_p_L*del_p_R > 0.0)   { del_p_G  = 2.0*del_p_L*del_p_R   / (del_p_L+del_p_R); }
  else { del_p_G = 0.0; } 

  
  // Step 3 - Project the left, right, and centered differences onto the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables, see Stone for notation)
  //          Use the eigenvectors given in Stone 2008, Appendix A
  Real del_a_0_L, del_a_1_L, del_a_2_L, del_a_3_L, del_a_4_L;
  Real del_a_0_R, del_a_1_R, del_a_2_R, del_a_3_R, del_a_4_R;
  Real del_a_0_C, del_a_1_C, del_a_2_C, del_a_3_C, del_a_4_C;
  Real del_a_0_G, del_a_1_G, del_a_2_G, del_a_3_G, del_a_4_G;

  del_a_0_L = -d_i * del_vx_L / (2*a_i) + del_p_L / (2*a_i*a_i);
  del_a_1_L = del_d_L - del_p_L / (a_i*a_i);
  del_a_2_L = del_vy_L;
  del_a_3_L = del_vz_L;
  del_a_4_L = d_i * del_vx_L / (2*a_i) + del_p_L / (2*a_i*a_i);

  del_a_0_R = -d_i * del_vx_R / (2*a_i) + del_p_R / (2*a_i*a_i);
  del_a_1_R = del_d_R - del_p_R / (a_i*a_i);
  del_a_2_R = del_vy_R;
  del_a_3_R = del_vz_R;
  del_a_4_R = d_i * del_vx_R / (2*a_i) + del_p_R / (2*a_i*a_i);

  del_a_0_C = -d_i * del_vx_C / (2*a_i) + del_p_C / (2*a_i*a_i);
  del_a_1_C = del_d_C - del_p_C / (a_i*a_i);
  del_a_2_C = del_vy_C;
  del_a_3_C = del_vz_C;
  del_a_4_C = d_i * del_vx_C / (2*a_i) + del_p_C / (2*a_i*a_i);

  del_a_0_G = -d_i * del_vx_G / (2*a_i) + del_p_G / (2*a_i*a_i);
  del_a_1_G = del_d_G - del_p_G / (a_i*a_i);
  del_a_2_G = del_vy_G;
  del_a_3_G = del_vz_G;
  del_a_4_G = d_i * del_vx_G / (2*a_i) + del_p_G / (2*a_i*a_i);

  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  //          Stone Eqn 38
  Real del_a_0_m, del_a_1_m, del_a_2_m, del_a_3_m, del_a_4_m;
  Real lim_slope_a, lim_slope_b;

/*
  del_a_0_m = SIGN(del_a_0_C) * minof3(2*fabs(del_a_0_L), 2*fabs(del_a_0_R), fabs(del_a_0_C));
  del_a_1_m = SIGN(del_a_1_C) * minof3(2*fabs(del_a_1_L), 2*fabs(del_a_1_R), fabs(del_a_1_C));
  del_a_2_m = SIGN(del_a_2_C) * minof3(2*fabs(del_a_2_L), 2*fabs(del_a_2_R), fabs(del_a_2_C));
  del_a_3_m = SIGN(del_a_3_C) * minof3(2*fabs(del_a_3_L), 2*fabs(del_a_3_R), fabs(del_a_3_C));
  del_a_4_m = SIGN(del_a_4_C) * minof3(2*fabs(del_a_4_L), 2*fabs(del_a_4_R), fabs(del_a_4_C));
*/

/************ New from Athena code ******************/
  del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;

  if (del_a_0_L * del_a_0_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
    lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
    del_a_0_m = SIGN(del_a_0_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }

  if (del_a_1_L * del_a_1_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
    lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
    del_a_1_m = SIGN(del_a_1_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }

  if (del_a_2_L * del_a_2_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
    lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
    del_a_2_m = SIGN(del_a_2_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }

  if (del_a_3_L * del_a_3_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
    lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
    del_a_3_m = SIGN(del_a_3_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }

  if (del_a_4_L * del_a_4_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
    lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
    del_a_4_m = SIGN(del_a_4_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
//********************************

  // Step 5 - Project the monotonized difference in the characteristic variables back onto the 
  //          primative variables
  //          Stone Eqn 39
  Real del_d_m_i, del_vx_m_i, del_vy_m_i, del_vz_m_i, del_p_m_i;

  del_d_m_i  = del_a_0_m + del_a_1_m + del_a_4_m;
  del_vx_m_i = -a_i*del_a_0_m / d_i + a_i*del_a_4_m / d_i;
  del_vy_m_i = del_a_2_m;
  del_vz_m_i = del_a_3_m;
  del_p_m_i  = del_a_0_m*a_i*a_i + del_a_4_m*a_i*a_i;


  // Step 6 - Compute the left and right interface values using the monotonized difference in the
  //          primative variables
  //          Stone Eqns 40 & 41
  Real d_L_iph, vx_L_iph, vy_L_iph, vz_L_iph, p_L_iph;
  Real d_R_imh, vx_R_imh, vy_R_imh, vz_R_imh, p_R_imh;
/*
  d_R_imh  = d_i  - (0.5 - fmin(lambda_m, 0) * 0.5*dtodx) * del_d_m_i;
  vx_R_imh = vx_i - (0.5 - fmin(lambda_m, 0) * 0.5*dtodx) * del_vx_m_i;
  vy_R_imh = vy_i - (0.5 - fmin(lambda_m, 0) * 0.5*dtodx) * del_vy_m_i;
  vz_R_imh = vz_i - (0.5 - fmin(lambda_m, 0) * 0.5*dtodx) * del_vz_m_i;
  p_R_imh  = p_i  - (0.5 - fmin(lambda_m, 0) * 0.5*dtodx) * del_p_m_i;

  d_L_iph  = d_i  + (0.5 - fmax(lambda_p, 0) * 0.5*dtodx) * del_d_m_i;
  vx_L_iph = vx_i + (0.5 - fmax(lambda_p, 0) * 0.5*dtodx) * del_vx_m_i;
  vy_L_iph = vy_i + (0.5 - fmax(lambda_p, 0) * 0.5*dtodx) * del_vy_m_i;
  vz_L_iph = vz_i + (0.5 - fmax(lambda_p, 0) * 0.5*dtodx) * del_vz_m_i;
  p_L_iph  = p_i  + (0.5 - fmax(lambda_p, 0) * 0.5*dtodx) * del_p_m_i;
*/

/************ New from Athena code ******************/
  // Step 7 Compute L/R values, ensure they lie between neighboring cell-centered values
  d_R_imh  = d_i  - 0.5*del_d_m_i; 
  vx_R_imh = vx_i - 0.5*del_vx_m_i;
  vy_R_imh = vy_i - 0.5*del_vy_m_i;
  vz_R_imh = vz_i - 0.5*del_vz_m_i;
  p_R_imh  = p_i  - 0.5*del_p_m_i;
 
  d_L_iph  = d_i  + 0.5*del_d_m_i; 
  vx_L_iph = vx_i + 0.5*del_vx_m_i;
  vy_L_iph = vy_i + 0.5*del_vy_m_i;
  vz_L_iph = vz_i + 0.5*del_vz_m_i;
  p_L_iph  = p_i  + 0.5*del_p_m_i; 


  Real C = d_R_imh + d_L_iph;
  d_R_imh = fmax( fmin(d_i, d_imo), d_R_imh );
  d_R_imh = fmin( fmax(d_i, d_imo), d_R_imh );
  d_L_iph = C - d_R_imh;
  d_L_iph = fmax( fmin(d_i, d_ipo), d_L_iph );
  d_L_iph = fmin( fmax(d_i, d_ipo), d_L_iph );
  d_R_imh = C - d_L_iph;

  C = vx_R_imh + vx_L_iph;
  vx_R_imh = fmax( fmin(vx_i, vx_imo), vx_R_imh );
  vx_R_imh = fmin( fmax(vx_i, vx_imo), vx_R_imh );
  vx_L_iph = C - vx_R_imh;
  vx_L_iph = fmax( fmin(vx_i, vx_ipo), vx_L_iph );
  vx_L_iph = fmin( fmax(vx_i, vx_ipo), vx_L_iph );
  vx_R_imh = C - vx_L_iph;  

  C = vy_R_imh + vy_L_iph;
  vy_R_imh = fmax( fmin(vy_i, vy_imo), vy_R_imh );
  vy_R_imh = fmin( fmax(vy_i, vy_imo), vy_R_imh );
  vy_L_iph = C - vy_R_imh;
  vy_L_iph = fmax( fmin(vy_i, vy_ipo), vy_L_iph );
  vy_L_iph = fmin( fmax(vy_i, vy_ipo), vy_L_iph );
  vy_R_imh = C - vy_L_iph;
 
  C = vz_R_imh + vz_L_iph;
  vz_R_imh = fmax( fmin(vz_i, vz_imo), vz_R_imh );
  vz_R_imh = fmin( fmax(vz_i, vz_imo), vz_R_imh );
  vz_L_iph = C - vz_R_imh; 
  vz_L_iph = fmax( fmin(vz_i, vz_ipo), vz_L_iph );
  vz_L_iph = fmin( fmax(vz_i, vz_ipo), vz_L_iph );
  vz_R_imh = C - vz_L_iph;

  C = p_R_imh + p_L_iph;
  p_R_imh = fmax( fmin(p_i, p_imo), p_R_imh );
  p_R_imh = fmin( fmax(p_i, p_imo), p_R_imh );
  p_L_iph = C - p_R_imh;
  p_L_iph = fmax( fmin(p_i, p_ipo), p_L_iph );
  p_L_iph = fmin( fmax(p_i, p_ipo), p_L_iph );
  p_R_imh = C - p_L_iph;

  del_d_m_i  = d_L_iph  - d_R_imh;
  del_vx_m_i = vx_L_iph - vx_R_imh;
  del_vy_m_i = vy_L_iph - vy_R_imh;
  del_vz_m_i = vz_L_iph - vz_R_imh;
  del_p_m_i  = p_L_iph  - p_R_imh;

  // Step 8 - Integrate linear interpolation function over domain of dependence
  //          defined by max(min) eigenvalue
  Real qx = -0.5*fmin(lambda_m, 0)*dtodx;
  d_R_imh  = d_R_imh  + qx * del_d_m_i;
  vx_R_imh = vx_R_imh + qx * del_vx_m_i;
  vy_R_imh = vy_R_imh + qx * del_vy_m_i;
  vz_R_imh = vz_R_imh + qx * del_vz_m_i;
  p_R_imh  = p_R_imh  + qx * del_p_m_i;

  qx = 0.5*fmax(lambda_p, 0)*dtodx;
  d_L_iph  = d_L_iph  - qx * del_d_m_i;
  vx_L_iph = vx_L_iph - qx * del_vx_m_i;
  vy_L_iph = vy_L_iph - qx * del_vy_m_i;
  vz_L_iph = vz_L_iph - qx * del_vz_m_i;
  p_L_iph  = p_L_iph  - qx * del_p_m_i;
//********************************


  // Step 7 - Perform the characteristic tracing
  //          Stone Eqns 42 & 43
  Real lamdiff;

  // left-hand interface value, i+1/2
  Real sum_0 = 0;
  Real sum_1 = 0;
  Real sum_2 = 0;
  Real sum_3 = 0;
  Real sum_4 = 0;
  if (lambda_m >= 0)
  {
    lamdiff = lambda_p - lambda_m;

    sum_0 += lamdiff * (-d_i*del_vx_m_i/(2*a_i) + del_p_m_i/(2*a_i*a_i));
    sum_1 += lamdiff * (del_vx_m_i/2.0 - del_p_m_i/(2*a_i*d_i));
    //sum_1 += lamdiff * (-d_i*del_vx_m_i/(2*a_i) + del_p_m_i/(2*a_i*a_i));
    sum_4 += lamdiff * (-d_i*del_vx_m_i*a_i/2.0 + del_p_m_i/2.0);
    //sum_4 += lamdiff * (-d_i*del_vx_m_i/(2*a_i) + del_p_m_i/(2*a_i*a_i));
  }
  if (lambda_0 >= 0)
  {
    lamdiff = lambda_p - lambda_0;
  
    sum_0 += lamdiff * (del_d_m_i - del_p_m_i/(a_i*a_i));
    //sum_0 += lamdiff * (-a_i*del_d_m_i/d_i + del_p_m_i / (a_i*d_i));
    sum_2 += lamdiff * del_vy_m_i;
    sum_3 += lamdiff * del_vz_m_i;
    //sum_4 += lamdiff * (a_i*del_d_m_i/d_i - del_p_m_i / (a_i*d_i));
  }
  if (lambda_p >= 0)
  {
    lamdiff = lambda_p - lambda_p;

    sum_0 += lamdiff * (d_i*del_vx_m_i/(2*a_i) + del_p_m_i/(2*a_i*a_i));
    //sum_0 += lamdiff * (d_i*del_vx_m_i*a_i/2.0 + del_p_m_i/2.0);
    sum_1 += lamdiff * (del_vx_m_i/2.0 + del_p_m_i/(2*a_i*d_i));
    sum_4 += lamdiff * (d_i*del_vx_m_i*a_i/2.0 + del_p_m_i/2.0);
  }

  // add the corrections to the initial guesses for the interface values
  d_L_iph  += 0.5*dtodx*sum_0;
  vx_L_iph += 0.5*dtodx*sum_1;
  vy_L_iph += 0.5*dtodx*sum_2;
  vz_L_iph += 0.5*dtodx*sum_3;
  p_L_iph  += 0.5*dtodx*sum_4;


  // right-hand interface value, i-1/2
  sum_0 = 0;
  sum_1 = 0;
  sum_2 = 0;
  sum_3 = 0;
  sum_4 = 0;
  if (lambda_m <= 0)
  {
    lamdiff = lambda_m - lambda_m; 

    sum_0 += lamdiff * (-d_i*del_vx_m_i/(2*a_i) + del_p_m_i/(2*a_i*a_i));
    sum_1 += lamdiff * (del_vx_m_i/2.0 - del_p_m_i/(2*a_i*d_i));
    //sum_1 += lamdiff * (-d_i*del_vx_m_i/(2*a_i) + del_p_m_i/(2*a_i*a_i));
    sum_4 += lamdiff * (-d_i*del_vx_m_i*a_i/2.0 + del_p_m_i/2.0);
    //sum_4 += lamdiff * (-d_i*del_vx_m_i/(2*a_i) + del_p_m_i/(2*a_i*a_i));
  }
  if (lambda_0 <= 0)
  {
    lamdiff = lambda_m - lambda_0;
  
    sum_0 += lamdiff * (del_d_m_i - del_p_m_i/(a_i*a_i));
    //sum_0 += lamdiff * (-a_i*del_d_m_i/d_i + del_p_m_i / (a_i*d_i));
    sum_2 += lamdiff * del_vy_m_i;
    sum_3 += lamdiff * del_vz_m_i;
    //sum_4 += lamdiff * (a_i*del_d_m_i/d_i - del_p_m_i / (a_i*d_i));
  }
  if (lambda_p <= 0)
  {
    lamdiff = lambda_m - lambda_p;

    sum_0 += lamdiff * (d_i*del_vx_m_i/(2*a_i) + del_p_m_i/(2*a_i*a_i));
    //sum_0 += lamdiff * (d_i*del_vx_m_i*a_i/2.0 + del_p_m_i/2.0);
    sum_1 += lamdiff * (del_vx_m_i/2.0 + del_p_m_i/(2*a_i*d_i));
    sum_4 += lamdiff * (d_i*del_vx_m_i*a_i/2.0 + del_p_m_i/2.0);
  }

  // add the corrections
  d_R_imh  += 0.5*dtodx*sum_0;
  vx_R_imh += 0.5*dtodx*sum_1;
  vy_R_imh += 0.5*dtodx*sum_2;
  vz_R_imh += 0.5*dtodx*sum_3;
  p_R_imh  += 0.5*dtodx*sum_4;

  // apply minimum constraints
  d_R_imh = fmax(d_R_imh, TINY_NUMBER);
  d_L_iph = fmax(d_L_iph, TINY_NUMBER);
  p_R_imh = fmax(p_R_imh, TINY_NUMBER);
  p_L_iph = fmax(p_L_iph, TINY_NUMBER);

  // Step 8 - Convert the left and right states in the primitive to the conserved variables
  bounds[0] = d_R_imh;
  bounds[1] = d_R_imh*vx_R_imh;
  bounds[2] = d_R_imh*vy_R_imh;
  bounds[3] = d_R_imh*vz_R_imh;
  bounds[4] = (p_R_imh/(gamma-1.0)) + 0.5*d_R_imh*(vx_R_imh*vx_R_imh + vy_R_imh*vy_R_imh + vz_R_imh*vz_R_imh);
  bounds[5] = d_L_iph;
  bounds[6] = d_L_iph*vx_L_iph;
  bounds[7] = d_L_iph*vy_L_iph;
  bounds[8] = d_L_iph*vz_L_iph;
  bounds[9] = (p_L_iph/(gamma-1.0)) + 0.5*d_L_iph*(vx_L_iph*vx_L_iph + vy_L_iph*vy_L_iph + vz_L_iph*vz_L_iph);



}

#endif //PLMP
#endif //CUDA

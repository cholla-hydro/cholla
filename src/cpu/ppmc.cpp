/*! \file ppmc.cpp
 *  \brief Definitions of the PPM reconstruction functions. Written following Stone et al. 2008. */
#ifndef CUDA
#ifdef PPMC

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../cpu/ppmc.h"
#include "../global/global.h"



/*! \fn void ppmc(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
 *  \brief When passed a stencil of conserved variables, returns the left and right
           boundary values for the interface calculated using ppm. */
void ppmc(Real stencil[], Real bounds[], const Real dx, const Real dt, Real gamma)
{
  // retrieve the values from the stencil
  Real d_i = stencil[0];
  Real vx_i = stencil[1] / d_i;
  Real vy_i = stencil[2] / d_i;
  Real vz_i = stencil[3] / d_i;
  Real p_i = (stencil[4] - 0.5*d_i*(vx_i*vx_i + vy_i*vy_i + vz_i*vz_i)) * (gamma - 1.0);
  p_i = fmax(p_i, TINY_NUMBER);
  Real d_imo = stencil[5];
  Real vx_imo = stencil[6] / d_imo;
  Real vy_imo = stencil[7] / d_imo;
  Real vz_imo = stencil[8] / d_imo;
  Real p_imo = (stencil[9] - 0.5*d_imo*(vx_imo*vx_imo + vy_imo*vy_imo + vz_imo*vz_imo)) * (gamma - 1.0);
  p_imo = fmax(p_imo, TINY_NUMBER);
  Real d_ipo = stencil[10];
  Real vx_ipo = stencil[11] / d_ipo;
  Real vy_ipo = stencil[12] / d_ipo;
  Real vz_ipo = stencil[13] / d_ipo;
  Real p_ipo = (stencil[14] - 0.5*d_ipo*(vx_ipo*vx_ipo + vy_ipo*vy_ipo + vz_ipo*vz_ipo)) * (gamma - 1.0);
  p_ipo = fmax(p_ipo, TINY_NUMBER);
  Real d_imt = stencil[15];
  Real vx_imt = stencil[16] / d_imt;
  Real vy_imt = stencil[17] / d_imt;
  Real vz_imt = stencil[18] / d_imt;
  Real p_imt = (stencil[19] - 0.5*d_imt*(vx_imt*vx_imt + vy_imt*vy_imt + vz_imt*vz_imt)) * (gamma - 1.0);
  p_imt = fmax(p_imt, TINY_NUMBER);
  Real d_ipt = stencil[20];
  Real vx_ipt = stencil[21] / d_ipt;
  Real vy_ipt = stencil[22] / d_ipt;
  Real vz_ipt = stencil[23] / d_ipt;
  Real p_ipt = (stencil[24] - 0.5*d_ipt*(vx_ipt*vx_ipt + vy_ipt*vy_ipt + vz_ipt*vz_ipt)) * (gamma - 1.0);
  p_ipt = fmax(p_ipt, TINY_NUMBER);

/*
printf("% 10.8f % 10.8f % 10.8f % 10.8f % 10.8f\n", d_imt, d_imo, d_i, d_ipo, d_ipt);
printf("% 10.8f % 10.8f % 10.8f % 10.8f % 10.8f\n", vx_imt, vx_imo, vx_i, vx_ipo, vx_ipt);
printf("% 10.8f % 10.8f % 10.8f % 10.8f % 10.8f\n", vy_imt, vy_imo, vy_i, vy_ipo, vy_ipt);
printf("% 10.8f % 10.8f % 10.8f % 10.8f % 10.8f\n", vz_imt, vz_imo, vz_i, vz_ipo, vz_ipt);
printf("% 10.8f % 10.8f % 10.8f % 10.8f % 10.8f\n", p_imt, p_imo, p_i, p_ipo, p_ipt);
*/


  const Real dtodx = dt/dx;

  Real a_imo, a_i, a_ipo;
  Real lambda_m, lambda_0, lambda_p;
  Real del_d_L, del_vx_L, del_vy_L, del_vz_L, del_p_L;
  Real del_d_R, del_vx_R, del_vy_R, del_vz_R, del_p_R;
  Real del_d_C, del_vx_C, del_vy_C, del_vz_C, del_p_C;
  Real del_d_G, del_vx_G, del_vy_G, del_vz_G, del_p_G;
  Real del_a_0_L, del_a_1_L, del_a_2_L, del_a_3_L, del_a_4_L;
  Real del_a_0_R, del_a_1_R, del_a_2_R, del_a_3_R, del_a_4_R;
  Real del_a_0_C, del_a_1_C, del_a_2_C, del_a_3_C, del_a_4_C;
  Real del_a_0_G, del_a_1_G, del_a_2_G, del_a_3_G, del_a_4_G;
  Real del_a_0_m, del_a_1_m, del_a_2_m, del_a_3_m, del_a_4_m;
  Real lim_slope_a, lim_slope_b;
  Real del_d_m_imo, del_vx_m_imo, del_vy_m_imo, del_vz_m_imo, del_p_m_imo;
  Real del_d_m_i, del_vx_m_i, del_vy_m_i, del_vz_m_i, del_p_m_i;
  Real del_d_m_ipo, del_vx_m_ipo, del_vy_m_ipo, del_vz_m_ipo, del_p_m_ipo;
  Real d_L, vx_L, vy_L, vz_L, p_L;
  Real d_R, vx_R, vy_R, vz_R, p_R;
  Real d_6, vx_6, vy_6, vz_6, p_6;
  Real lambda_max, lambda_min;
  Real A, B, C, D;
  Real chi_1, chi_2, chi_3, chi_4, chi_5;
  Real sum_1, sum_2, sum_3, sum_4, sum_5;


  // calculate the adiabatic sound speed in cell imo, i, ipo
  a_imo = sqrt(gamma*p_imo/d_imo);
  a_i   = sqrt(gamma*p_i/d_i);
  a_ipo = sqrt(gamma*p_ipo/d_ipo);


  // Step 1 - Compute the eigenvalues of the linearized equations in the
  //          primitive variables using the cell-centered primitive variables

  lambda_m = vx_i-a_i;
  lambda_0 = vx_i;
  lambda_p = vx_i+a_i;


  // Steps 2 - 5 are repeated for cell i-1, i, and i+1

  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables
  //          Note that here L and R refer to locations relative to the cell center
  //          Stone Eqn 36

  // left
  del_d_L  = d_imo - d_imt;
  del_vx_L = vx_imo - vx_imt;
  del_vy_L = vy_imo - vy_imt;
  del_vz_L = vz_imo - vz_imt;
  del_p_L  = p_imo  - p_imt;

  // right
  del_d_R  = d_i  - d_imo;
  del_vx_R = vx_i - vx_imo;
  del_vy_R = vy_i - vy_imo;
  del_vz_R = vz_i - vz_imo;
  del_p_R  = p_i  - p_imo;

  // centered
  del_d_C  = 0.5*(d_i - d_imt);
  del_vx_C = 0.5*(vx_i - vx_imt);
  del_vy_C = 0.5*(vy_i - vy_imt);
  del_vz_C = 0.5*(vz_i - vz_imt);
  del_p_C  = 0.5*(p_i - p_imt);

  // Van Leer
  if (del_d_L*del_d_R > 0.0) { del_d_G = 2.0*del_d_L*del_d_R / (del_d_L+del_d_R); }
  else { del_d_G = 0.0; }
  if (del_vx_L*del_vx_R > 0.0) { del_vx_G = 2.0*del_vx_L*del_vx_R / (del_vx_L+del_vx_R); }
  else { del_vx_G = 0.0; }
  if (del_vy_L*del_vy_R > 0.0) { del_vy_G = 2.0*del_vy_L*del_vy_R / (del_vy_L+del_vy_R); }
  else { del_vy_G = 0.0; }
  if (del_vz_L*del_vz_R > 0.0) { del_vz_G = 2.0*del_vz_L*del_vz_R / (del_vz_L+del_vz_R); }
  else { del_vz_G = 0.0; }
  if (del_p_L*del_p_R > 0.0) { del_p_G = 2.0*del_p_L*del_p_R / (del_p_L+del_p_R); }
  else { del_p_G = 0.0; }


  // Step 3 - Project the left, right, centered and van Leer differences onto the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables, see Stone for notation)
  //          Use the eigenvectors given in Stone 2008, Appendix A

  del_a_0_L = -d_imo * del_vx_L / (2*a_imo) + del_p_L / (2*a_imo*a_imo);
  del_a_1_L = del_d_L - del_p_L / (a_imo*a_imo);
  del_a_2_L = del_vy_L;
  del_a_3_L = del_vz_L;
  del_a_4_L = d_imo * del_vx_L / (2*a_imo) + del_p_L / (2*a_imo*a_imo);

  del_a_0_R = -d_imo * del_vx_R / (2*a_imo) + del_p_R / (2*a_imo*a_imo);
  del_a_1_R = del_d_R - del_p_R / (a_imo*a_imo);
  del_a_2_R = del_vy_R;
  del_a_3_R = del_vz_R;
  del_a_4_R = d_imo * del_vx_R / (2*a_imo) + del_p_R / (2*a_imo*a_imo);

  del_a_0_C = -d_imo * del_vx_C / (2*a_imo) + del_p_C / (2*a_imo*a_imo);
  del_a_1_C = del_d_C - del_p_C / (a_imo*a_imo);
  del_a_2_C = del_vy_C;
  del_a_3_C = del_vz_C;
  del_a_4_C = d_imo * del_vx_C / (2*a_imo) + del_p_C / (2*a_imo*a_imo);

  del_a_0_G = -d_imo * del_vx_G / (2*a_imo) + del_p_G / (2*a_imo*a_imo);
  del_a_1_G = del_d_G - del_p_G / (a_imo*a_imo);
  del_a_2_G = del_vy_G;
  del_a_3_G = del_vz_G;
  del_a_4_G = d_imo * del_vx_G / (2*a_imo) + del_p_G / (2*a_imo*a_imo);


  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  //          Stone Eqn 38

  del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;

  if (del_a_0_L*del_a_0_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
    lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
    del_a_0_m = sgn(del_a_0_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_1_L*del_a_1_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
    lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
    del_a_1_m = sgn(del_a_1_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_2_L*del_a_2_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
    lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
    del_a_2_m = sgn(del_a_2_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_3_L*del_a_3_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
    lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
    del_a_3_m = sgn(del_a_3_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_4_L*del_a_4_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
    lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
    del_a_4_m = sgn(del_a_4_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }


  // Step 5 - Project the monotonized difference in the characteristic variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_imo  = del_a_0_m + del_a_1_m + del_a_4_m;
  del_vx_m_imo = -a_imo*del_a_0_m / d_imo + a_imo* del_a_4_m / d_imo;
  del_vy_m_imo = del_a_2_m;
  del_vz_m_imo = del_a_3_m;
  del_p_m_imo  = a_imo*a_imo*del_a_0_m + a_imo*a_imo*del_a_4_m;


  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables
  //          Note that here L and R refer to locations relative to the cell center
  //          Stone Eqn 36

  // left
  del_d_L  = d_i  - d_imo;
  del_vx_L = vx_i - vx_imo;
  del_vy_L = vy_i - vy_imo;
  del_vz_L = vz_i - vz_imo;
  del_p_L  = p_i  - p_imo;

  // right
  del_d_R  = d_ipo  - d_i;
  del_vx_R = vx_ipo - vx_i;
  del_vy_R = vy_ipo - vy_i;
  del_vz_R = vz_ipo - vz_i;
  del_p_R  = p_ipo  - p_i;

  // centered
  del_d_C  = 0.5*(d_ipo - d_imo);
  del_vx_C = 0.5*(vx_ipo - vx_imo);
  del_vy_C = 0.5*(vy_ipo - vy_imo);
  del_vz_C = 0.5*(vz_ipo - vz_imo);
  del_p_C  = 0.5*(p_ipo - p_imo);

  // van Leer
  if (del_d_L*del_d_R > 0.0) { del_d_G = 2.0*del_d_L*del_d_R / (del_d_L+del_d_R); }
  else { del_d_G = 0.0; }
  if (del_vx_L*del_vx_R > 0.0) { del_vx_G = 2.0*del_vx_L*del_vx_R / (del_vx_L+del_vx_R); }
  else { del_vx_G = 0.0; }
  if (del_vy_L*del_vy_R > 0.0) { del_vy_G = 2.0*del_vy_L*del_vy_R / (del_vy_L+del_vy_R); }
  else { del_vy_G = 0.0; }
  if (del_vz_L*del_vz_R > 0.0) { del_vz_G = 2.0*del_vz_L*del_vz_R / (del_vz_L+del_vz_R); }
  else { del_vz_G = 0.0; }
  if (del_p_L*del_p_R > 0.0) { del_p_G = 2.0*del_p_L*del_p_R / (del_p_L+del_p_R); }
  else { del_p_G = 0.0; }


  // Step 3 - Project the left, right, centered, and van Leer differences onto the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables, see Stone for notation)
  //          Use the eigenvectors given in Stone 2008, Appendix A

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

  del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;

  if (del_a_0_L*del_a_0_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
    lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
    del_a_0_m = sgn(del_a_0_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_1_L*del_a_1_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
    lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
    del_a_1_m = sgn(del_a_1_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_2_L*del_a_2_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
    lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
    del_a_2_m = sgn(del_a_2_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_3_L*del_a_3_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
    lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
    del_a_3_m = sgn(del_a_3_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_4_L*del_a_4_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
    lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
    del_a_4_m = sgn(del_a_4_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }


  // Step 5 - Project the monotonized difference in the characteristic variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_i  = del_a_0_m + del_a_1_m + del_a_4_m;
  del_vx_m_i = -a_i*del_a_0_m / d_i + a_i* del_a_4_m / d_i;
  del_vy_m_i = del_a_2_m;
  del_vz_m_i = del_a_3_m;
  del_p_m_i  = a_i*a_i*del_a_0_m + a_i*a_i*del_a_4_m;


  // Step 2 - Compute the left, right, centered, and van Leer differences of the primitive variables
  //          Note that here L and R refer to locations relative to the cell center
  //          Stone Eqn 36

  // left
  del_d_L  = d_ipo - d_i;
  del_vx_L = vx_ipo - vx_i;
  del_vy_L = vy_ipo - vy_i;
  del_vz_L = vz_ipo - vz_i;
  del_p_L  = p_ipo  - p_i;

  // right
  del_d_R  = d_ipt  - d_ipo;
  del_vx_R = vx_ipt - vx_ipo;
  del_vy_R = vy_ipt - vy_ipo;
  del_vz_R = vz_ipt - vz_ipo;
  del_p_R  = p_ipt  - p_ipo;

  // centered
  del_d_C  = 0.5*(d_ipt - d_i);
  del_vx_C = 0.5*(vx_ipt- vx_i);
  del_vy_C = 0.5*(vy_ipt - vy_i);
  del_vz_C = 0.5*(vz_ipt - vz_i);
  del_p_C  = 0.5*(p_ipt - p_i);

  // van Leer
  if (del_d_L*del_d_R > 0.0) { del_d_G = 2.0*del_d_L*del_d_R / (del_d_L+del_d_R); }
  else { del_d_G = 0.0; }
  if (del_vx_L*del_vx_R > 0.0) { del_vx_G = 2.0*del_vx_L*del_vx_R / (del_vx_L+del_vx_R); }
  else { del_vx_G = 0.0; }
  if (del_vy_L*del_vy_R > 0.0) { del_vy_G = 2.0*del_vy_L*del_vy_R / (del_vy_L+del_vy_R); }
  else { del_vy_G = 0.0; }
  if (del_vz_L*del_vz_R > 0.0) { del_vz_G = 2.0*del_vz_L*del_vz_R / (del_vz_L+del_vz_R); }
  else { del_vz_G = 0.0; }
  if (del_p_L*del_p_R > 0.0) { del_p_G = 2.0*del_p_L*del_p_R / (del_p_L+del_p_R); }
  else { del_p_G = 0.0; }


  // Step 3 - Project the left, right, centered, and van Leer differences onto the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables, see Stone for notation)
  //          Use the eigenvectors given in Stone 2008, Appendix A

  del_a_0_L = -d_ipo * del_vx_L / (2*a_ipo) + del_p_L / (2*a_ipo*a_ipo);
  del_a_1_L = del_d_L - del_p_L / (a_ipo*a_ipo);
  del_a_2_L = del_vy_L;
  del_a_3_L = del_vz_L;
  del_a_4_L = d_ipo * del_vx_L / (2*a_ipo) + del_p_L / (2*a_ipo*a_ipo);

  del_a_0_R = -d_ipo * del_vx_R / (2*a_ipo) + del_p_R / (2*a_ipo*a_ipo);
  del_a_1_R = del_d_R - del_p_R / (a_ipo*a_ipo);
  del_a_2_R = del_vy_R;
  del_a_3_R = del_vz_R;
  del_a_4_R = d_ipo * del_vx_R / (2*a_ipo) + del_p_R / (2*a_ipo*a_ipo);

  del_a_0_C = -d_ipo * del_vx_C / (2*a_ipo) + del_p_C / (2*a_ipo*a_ipo);
  del_a_1_C = del_d_C - del_p_C / (a_ipo*a_ipo);
  del_a_2_C = del_vy_C;
  del_a_3_C = del_vz_C;
  del_a_4_C = d_ipo * del_vx_C / (2*a_ipo) + del_p_C / (2*a_ipo*a_ipo);

  del_a_0_G = -d_ipo * del_vx_G / (2*a_ipo) + del_p_G / (2*a_ipo*a_ipo);
  del_a_1_G = del_d_G - del_p_G / (a_ipo*a_ipo);
  del_a_2_G = del_vy_G;
  del_a_3_G = del_vz_G;
  del_a_4_G = d_ipo * del_vx_G / (2*a_ipo) + del_p_G / (2*a_ipo*a_ipo);


  // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
  //          Stone Eqn 38

  del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;

  if (del_a_0_L*del_a_0_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
    lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
    del_a_0_m = sgn(del_a_0_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_1_L*del_a_1_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
    lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
    del_a_1_m = sgn(del_a_1_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_2_L*del_a_2_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
    lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
    del_a_2_m = sgn(del_a_2_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_3_L*del_a_3_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
    lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
    del_a_3_m = sgn(del_a_3_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }
  if (del_a_4_L*del_a_4_R > 0.0) {
    lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
    lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
    del_a_4_m = sgn(del_a_4_C) * fmin(2.0*lim_slope_a, lim_slope_b);
  }


  // Step 5 - Project the monotonized difference in the characteristic variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_ipo  = del_a_0_m + del_a_1_m + del_a_4_m;
  del_vx_m_ipo = -a_ipo*del_a_0_m / d_ipo + a_ipo* del_a_4_m / d_ipo;
  del_vy_m_ipo = del_a_2_m;
  del_vz_m_ipo = del_a_3_m;
  del_p_m_ipo  = a_ipo*a_ipo*del_a_0_m + a_ipo*a_ipo*del_a_4_m;


  // Step 6 - Use parabolic interpolation to compute values at the left and right of each cell center
  //          Here, the subscripts L and R refer to the left and right side of the ith cell center
  //          Stone Eqn 46

  d_L  = 0.5*(d_i + d_imo)   - (del_d_m_i  - del_d_m_imo)  / 6.0;
  vx_L = 0.5*(vx_i + vx_imo) - (del_vx_m_i - del_vx_m_imo) / 6.0;
  vy_L = 0.5*(vy_i + vy_imo) - (del_vy_m_i - del_vy_m_imo) / 6.0;
  vz_L = 0.5*(vz_i + vz_imo) - (del_vz_m_i - del_vz_m_imo) / 6.0;
  p_L  = 0.5*(p_i + p_imo)   - (del_p_m_i  - del_p_m_imo)  / 6.0;

  d_R  = 0.5*(d_ipo + d_i)   - (del_d_m_ipo  - del_d_m_i)  / 6.0;
  vx_R = 0.5*(vx_ipo + vx_i) - (del_vx_m_ipo - del_vx_m_i) / 6.0;
  vy_R = 0.5*(vy_ipo + vy_i) - (del_vy_m_ipo - del_vy_m_i) / 6.0;
  vz_R = 0.5*(vz_ipo + vz_i) - (del_vz_m_ipo - del_vz_m_i) / 6.0;
  p_R  = 0.5*(p_ipo + p_i)   - (del_p_m_ipo  - del_p_m_i)  / 6.0;


  // Step 7 - Apply further monotonicity constraints to ensure the values on the left and right side
  //          of cell center lie between neighboring cell-centered values
  //          Stone Eqns 47 - 53

  if ((d_R  - d_i) *(d_i  - d_L)  <= 0) d_L  = d_R  = d_i;
  if ((vx_R - vx_i)*(vx_i - vx_L) <= 0) vx_L = vx_R = vx_i;
  if ((vy_R - vy_i)*(vy_i - vy_L) <= 0) vy_L = vy_R = vy_i;
  if ((vz_R - vz_i)*(vz_i - vz_L) <= 0) vz_L = vz_R = vz_i;
  if ((p_R  - p_i) *(p_i  - p_L)  <= 0) p_L  = p_R  = p_i;

  if ( 6.0*(d_R  - d_L) *(d_i  - 0.5*(d_L  + d_R))  > (d_R  - d_L) *(d_R  - d_L))  d_L  = 3.0*d_i  - 2.0*d_R;
  if ( 6.0*(vx_R - vx_L)*(vx_i - 0.5*(vx_L + vx_R)) > (vx_R - vx_L)*(vx_R - vx_L)) vx_L = 3.0*vx_i - 2.0*vx_R;
  if ( 6.0*(vy_R - vy_L)*(vy_i - 0.5*(vy_L + vy_R)) > (vy_R - vy_L)*(vy_R - vy_L)) vy_L = 3.0*vy_i - 2.0*vy_R;
  if ( 6.0*(vz_R - vz_L)*(vz_i - 0.5*(vz_L + vz_R)) > (vz_R - vz_L)*(vz_R - vz_L)) vz_L = 3.0*vz_i - 2.0*vz_R;
  if ( 6.0*(p_R  - p_L) *(p_i  - 0.5*(p_L  + p_R))  > (p_R  - p_L) *(p_R  - p_L))  p_L  = 3.0*p_i  - 2.0*p_R;

  if ( 6.0*(d_R  - d_L) *(d_i  - 0.5*(d_L  + d_R))  < -(d_R  - d_L) *(d_R  - d_L))  d_R  = 3.0*d_i  - 2.0*d_L;
  if ( 6.0*(vx_R - vx_L)*(vx_i - 0.5*(vx_L + vx_R)) < -(vx_R - vx_L)*(vx_R - vx_L)) vx_R = 3.0*vx_i - 2.0*vx_L;
  if ( 6.0*(vy_R - vy_L)*(vy_i - 0.5*(vy_L + vy_R)) < -(vy_R - vy_L)*(vy_R - vy_L)) vy_R = 3.0*vy_i - 2.0*vy_L;
  if ( 6.0*(vz_R - vz_L)*(vz_i - 0.5*(vz_L + vz_R)) < -(vz_R - vz_L)*(vz_R - vz_L)) vz_R = 3.0*vz_i - 2.0*vz_L;
  if ( 6.0*(p_R  - p_L) *(p_i  - 0.5*(p_L  + p_R))  < -(p_R  - p_L) *(p_R  - p_L))  p_R  = 3.0*p_i  - 2.0*p_L;

  d_L  = fmax( fmin(d_i,  d_imo), d_L );
  d_L  = fmin( fmax(d_i,  d_imo), d_L );
  d_R  = fmax( fmin(d_i,  d_ipo), d_R );
  d_R  = fmin( fmax(d_i,  d_ipo), d_R );
  vx_L = fmax( fmin(vx_i, vx_imo), vx_L );
  vx_L = fmin( fmax(vx_i, vx_imo), vx_L );
  vx_R = fmax( fmin(vx_i, vx_ipo), vx_R );
  vx_R = fmin( fmax(vx_i, vx_ipo), vx_R );
  vy_L = fmax( fmin(vy_i, vy_imo), vy_L );
  vy_L = fmin( fmax(vy_i, vy_imo), vy_L );
  vy_R = fmax( fmin(vy_i, vy_ipo), vy_R );
  vy_R = fmin( fmax(vy_i, vy_ipo), vy_R );
  vz_L = fmax( fmin(vz_i, vz_imo), vz_L );
  vz_L = fmin( fmax(vz_i, vz_imo), vz_L );
  vz_R = fmax( fmin(vz_i, vz_ipo), vz_R );
  vz_R = fmin( fmax(vz_i, vz_ipo), vz_R );
  p_L  = fmax( fmin(p_i,  p_imo), p_L );
  p_L  = fmin( fmax(p_i,  p_imo), p_L );
  p_R  = fmax( fmin(p_i,  p_ipo), p_R );
  p_R  = fmin( fmax(p_i,  p_ipo), p_R );


  // Step 8 - Compute the coefficients for the monotonized parabolic interpolation function
  //          Stone Eqn 54

  del_d_m_i  = d_R  - d_L;
  del_vx_m_i = vx_R - vx_L;
  del_vy_m_i = vy_R - vy_L;
  del_vz_m_i = vz_R - vz_L;
  del_p_m_i  = p_R  - p_L;

  d_6  = 6.*(d_i  - 0.5*(d_L  + d_R));
  vx_6 = 6.*(vx_i - 0.5*(vx_L + vx_R));
  vy_6 = 6.*(vy_i - 0.5*(vy_L + vy_R));
  vz_6 = 6.*(vz_i - 0.5*(vz_L + vz_R));
  p_6  = 6.*(p_i  - 0.5*(p_L  + p_R));


  // Step 9 - Compute the left and right interface values using monotonized parabolic interpolation
  //          Stone Eqns 55 & 56

  Real qx1, qx2;

  qx1 = 0.5*fmax(lambda_p, 0.0)*dtodx;
  d_R = d_R - qx1 * (del_d_m_i - (1.0-(4.0/3.0)*qx1)*d_6);
  vx_R = vx_R - qx1 * (del_vx_m_i - (1.0-(4.0/3.0)*qx1)*vx_6);
  vy_R = vy_R - qx1 * (del_vy_m_i - (1.0-(4.0/3.0)*qx1)*vy_6);
  vz_R = vz_R - qx1 * (del_vz_m_i - (1.0-(4.0/3.0)*qx1)*vz_6);
  p_R = p_R - qx1 * (del_p_m_i - (1.0-(4.0/3.0)*qx1)*p_6);

  qx2 = -0.5*fmin(lambda_m, 0.0)*dtodx;
  d_L = d_L + qx2 * (del_d_m_i + (1.0-(4.0/3.0)*qx2)*d_6);
  vx_L = vx_L + qx2 * (del_vx_m_i + (1.0-(4.0/3.0)*qx2)*vx_6);
  vy_L = vy_L + qx2 * (del_vy_m_i + (1.0-(4.0/3.0)*qx2)*vy_6);
  vz_L = vz_L + qx2 * (del_vz_m_i + (1.0-(4.0/3.0)*qx2)*vz_6);
  p_L = p_L + qx2 * (del_p_m_i + (1.0-(4.0/3.0)*qx2)*p_6);


  // Step 10 - Perform the characteristic tracing
  //           Stone Eqns 57 - 60

  // left-hand interface value, i+1/2
  sum_1 = 0;
  sum_2 = 0;
  sum_3 = 0;
  sum_4 = 0;
  sum_5 = 0;
  if (lambda_m >= 0)
  {
    A = (0.5*dtodx) * (lambda_p - lambda_m);
    B = (1.0/3.0)*(dtodx)*(dtodx)*(lambda_p*lambda_p - lambda_m*lambda_m);

    chi_1 = A*(del_d_m_i - d_6) + B*d_6;
    chi_2 = A*(del_vx_m_i - vx_6) + B*vx_6;
    chi_3 = A*(del_vy_m_i - vy_6) + B*vy_6;
    chi_4 = A*(del_vz_m_i - vz_6) + B*vz_6;
    chi_5 = A*(del_p_m_i - p_6) + B*p_6;

    sum_1 += -d_i*chi_2/(2*a_i) + chi_5/(2*a_i*a_i);
    sum_2 += chi_2/2.0 - chi_5/(2*a_i*d_i);
    sum_5 += -d_i*chi_2*a_i/2.0 + chi_5/2.0;
  }
  if (lambda_0 >= 0)
  {
    A = (0.5*dtodx) * (lambda_p - lambda_0);
    B = (1.0/3.0)*(dtodx)*(dtodx)*(lambda_p*lambda_p - lambda_0*lambda_0);

    chi_1 = A*(del_d_m_i - d_6) + B*d_6;
    chi_2 = A*(del_vx_m_i - vx_6) + B*vx_6;
    chi_3 = A*(del_vy_m_i - vy_6) + B*vy_6;
    chi_4 = A*(del_vz_m_i - vz_6) + B*vz_6;
    chi_5 = A*(del_p_m_i - p_6) + B*p_6;

    sum_1 += chi_1 - chi_5/(a_i*a_i);
    sum_3 += chi_3;
    sum_4 += chi_4;
  }
  if (lambda_p >= 0)
  {
    A = (0.5*dtodx) * (lambda_p - lambda_p);
    B = (1.0/3.0)*(dtodx)*(dtodx)*(lambda_p*lambda_p - lambda_p*lambda_p);

    chi_1 = A*(del_d_m_i - d_6) + B*d_6;
    chi_2 = A*(del_vx_m_i - vx_6) + B*vx_6;
    chi_3 = A*(del_vy_m_i - vy_6) + B*vy_6;
    chi_4 = A*(del_vz_m_i - vz_6) + B*vz_6;
    chi_5 = A*(del_p_m_i - p_6) + B*p_6;

    sum_1 += d_i*chi_2/(2*a_i) + chi_5/(2*a_i*a_i);
    sum_2 += chi_2/2.0 + chi_5/(2*a_i*d_i);
    sum_5 += d_i*chi_2*a_i/2.0 + chi_5/2.0;
  }

  // add the corrections to the initial guesses for the interface values
  d_R += sum_1;
  vx_R += sum_2;
  vy_R += sum_3;
  vz_R += sum_4;
  p_R += sum_5;


  // right-hand interface value, i-1/2
  sum_1 = 0;
  sum_2 = 0;
  sum_3 = 0;
  sum_4 = 0;
  sum_5 = 0;
  if (lambda_m <= 0)
  {
    C = (0.5*dtodx) * (lambda_m - lambda_m);
    D = (1.0/3.0)*(dtodx)*(dtodx)*(lambda_m*lambda_m - lambda_m*lambda_m);

    chi_1 = C*(del_d_m_i + d_6) + D*d_6;
    chi_2 = C*(del_vx_m_i + vx_6) + D*vx_6;
    chi_3 = C*(del_vy_m_i + vy_6) + D*vy_6;
    chi_4 = C*(del_vz_m_i + vz_6) + D*vz_6;
    chi_5 = C*(del_p_m_i + p_6) + D*p_6;

    sum_1 += -d_i*chi_2/(2*a_i) + chi_5/(2*a_i*a_i);
    sum_2 += chi_2/2.0 - chi_5/(2*a_i*d_i);
    sum_5 += -d_i*chi_2*a_i/2.0 + chi_5/2.0;
  }
  if (lambda_0 <= 0)
  {
    C = (0.5*dtodx) * (lambda_m - lambda_0);
    D = (1.0/3.0)*(dtodx)*(dtodx)*(lambda_m*lambda_m - lambda_0*lambda_0);

    chi_1 = C*(del_d_m_i + d_6) + D*d_6;
    chi_2 = C*(del_vx_m_i + vx_6) + D*vx_6;
    chi_3 = C*(del_vy_m_i + vy_6) + D*vy_6;
    chi_4 = C*(del_vz_m_i + vz_6) + D*vz_6;
    chi_5 = C*(del_p_m_i + p_6) + D*p_6;

    sum_1 += chi_1 - chi_5/(a_i*a_i);
    sum_3 += chi_3;
    sum_4 += chi_4;
  }
  if (lambda_p <= 0)
  {
    C = (0.5*dtodx) * (lambda_m - lambda_p);
    D = (1.0/3.0)*(dtodx)*(dtodx)*(lambda_m*lambda_m - lambda_p*lambda_p);

    chi_1 = C*(del_d_m_i + d_6) + D*d_6;
    chi_2 = C*(del_vx_m_i + vx_6) + D*vx_6;
    chi_3 = C*(del_vy_m_i + vy_6) + D*vy_6;
    chi_4 = C*(del_vz_m_i + vz_6) + D*vz_6;
    chi_5 = C*(del_p_m_i + p_6) + D*p_6;

    sum_1 += d_i*chi_2/(2*a_i) + chi_5/(2*a_i*a_i);
    sum_2 += chi_2/2.0 + chi_5/(2*a_i*d_i);
    sum_5 += d_i*chi_2*a_i/2.0 + chi_5/2.0;
  }

  // add the corrections
  d_L += sum_1;
  vx_L += sum_2;
  vy_L += sum_3;
  vz_L += sum_4;
  p_L += sum_5;


  // apply minimum constraints
  d_L = fmax(d_L, TINY_NUMBER);
  d_R = fmax(d_R, TINY_NUMBER);
  p_L = fmax(p_L, TINY_NUMBER);
  p_R = fmax(p_R, TINY_NUMBER);

  // Step 11 - Convert the left and right states in the primitive to the conserved variables
  bounds[0] = d_L;
  bounds[1] = d_L*vx_L;
  bounds[2] = d_L*vy_L;
  bounds[3] = d_L*vz_L;
  bounds[4] = (p_L/(gamma-1.0)) + 0.5*d_L*(vx_L*vx_L + vy_L*vy_L + vz_L*vz_L);
  bounds[5] = d_R;
  bounds[6] = d_R*vx_R;
  bounds[7] = d_R*vy_R;
  bounds[8] = d_R*vz_R;
  bounds[9] = (p_R/(gamma-1.0)) + 0.5*d_R*(vx_R*vx_R + vy_R*vy_R + vz_R*vz_R);

}




#endif //PPMC
#endif //CUDA


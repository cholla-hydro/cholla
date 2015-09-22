/*! \file plm.cpp
 *  \brief Definitions of the piecewise linear reconstruction functions */
#ifndef CUDA
#ifdef PLMP

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"plmp.h"
#include"global.h"


/*! \fn plmp(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
*  \brief Use a piece-wise linear method to calculate boundary values for each cell. */
void plmp(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
{
  // declare the primative and conserved variables we are calculating
  Real dl, dr, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr;
  Real mxl, mxr, myl, myr, mzl, mzr, El, Er;
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

  // calculate the slope (in each primative variable) across cell i
  Real del_d_L, del_vx_L, del_vy_L, del_vz_L, del_p_L;
  Real del_d_R, del_vx_R, del_vy_R, del_vz_R, del_p_R;
  Real d_slope, vx_slope, vy_slope, vz_slope, p_slope;

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
  
  // limit the slopes (B=1 is minmod)
  Real B = 1;
  if (d_slope_right>=0) d_slope = maxof3(0, fmin(B*d_slope_left, d_slope_right), fmin(d_slope_left, B*d_slope_right));
  if (d_slope_right<0)  d_slope = minof3(0, fmax(B*d_slope_left, d_slope_right), fmax(d_slope_left, B*d_slope_right));
  if (vx_slope_right>=0) vx_slope = maxof3(0, fmin(B*vx_slope_left, vx_slope_right), fmin(vx_slope_left, B*vx_slope_right));
  if (vx_slope_right<0)  vx_slope = minof3(0, fmax(B*vx_slope_left, vx_slope_right), fmax(vx_slope_left, B*vx_slope_right));
  if (vy_slope_right>=0) vy_slope = maxof3(0, fmin(B*vy_slope_left, vy_slope_right), fmin(vy_slope_left, B*vy_slope_right));
  if (vy_slope_right<0)  vy_slope = minof3(0, fmax(B*vy_slope_left, vy_slope_right), fmax(vy_slope_left, B*vy_slope_right));
  if (vz_slope_right>=0) vz_slope = maxof3(0, fmin(B*vz_slope_left, vz_slope_right), fmin(vz_slope_left, B*vz_slope_right));
  if (vz_slope_right<0)  vz_slope = minof3(0, fmax(B*vz_slope_left, vz_slope_right), fmax(vz_slope_left, B*vz_slope_right));
  if (p_slope_right>=0) p_slope = maxof3(0, fmin(B*p_slope_left, p_slope_right), fmin(p_slope_left, B*p_slope_right));
  if (p_slope_right<0)  p_slope = minof3(0, fmax(B*p_slope_left, p_slope_right), fmax(p_slope_left, B*p_slope_right));
  
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

  // apply mimimum constraints
  dl = fmax(dl, TINY_NUMBER);
  dr = fmax(dr, TINY_NUMBER);
  pl = fmax(pl, TINY_NUMBER);
  pr = fmax(pr, TINY_NUMBER);


 // calculate the conserved variables and fluxes at each interface
  mxl = dl*vxl;
  mxr = dr*vxr;
  myl = dl*vyl;
  myr = dr*vyr;
  mzl = dl*vzl;
  mzr = dr*vzr;
  El = pl/(gamma-1.0) + 0.5*dl*(vxl*vxl + vyl*vyl + vzl*vzl);
  Er = pr/(gamma-1.0) + 0.5*dr*(vxr*vxr + vyr*vyr + vzr*vzr);

  Real dfl, dfr, mxfl, mxfr, myfl, myfr, mzfl, mzfr, Efl, Efr;

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

  bounds[0] = dl;
  bounds[1] = mxl;
  bounds[2] = myl;
  bounds[3] = mzl;
  bounds[4] = El;
  bounds[5] = dr;
  bounds[6] = mxr;
  bounds[7] = myr;
  bounds[8] = mzr;
  bounds[9] = Er;

}


#endif //PLMP
#endif //CUDA


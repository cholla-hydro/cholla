/*! \file ppmp_ctu_cuda.cu
 *  \brief Function definitions for the ppm kernels, written following Fryxell et al., 2000.*/

#ifdef CUDA
#ifdef PPMP

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"ppmp_ctu_cuda.h"


#define STEEPENING
#define FLATTENING

/*! \fn  PPMP_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
 *  \brief Use the piecewise parabolic method to calculate boundary values for each cell. */
__global__ void PPMP_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
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
  Real d_imt, vx_imt, vy_imt, vz_imt, p_imt;
  Real d_ipt, vx_ipt, vy_ipt, vz_ipt, p_ipt;
  Real p_imth, p_ipth;

  // declare the primative variables we are calculating in registers (for cell i)
  // left and right boundary-extrapolated values of d, v, and p
  Real dl, dr, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr;

  // declare other constants
  Real cs, cl, cr; // sound speed in cell i, and at left and right boundaries
  Real del_d, del_vx, del_vy, del_vz, del_p; // "slope" accross cell i
  Real d_6, vx_6, vy_6, vz_6, p_6;
  Real beta_m, beta_0, beta_p;
  Real alpha_m, alpha_0, alpha_p;
  Real lambda_m, lambda_0, lambda_p; // speed of characteristics
  Real dL_m, vxL_m, pL_m;
  Real dL_0, vyL_0, vzL_0, pL_0;
  Real vxL_p, pL_p;
  Real vxR_m, pR_m;
  Real dR_0, vyR_0, vzR_0, pR_0;
  Real dR_p, vxR_p, pR_p;
  Real chi_L_m, chi_L_0, chi_L_p;
  Real chi_R_m, chi_R_0, chi_R_p;
  // declare constants to use in contact discontinuity detection
  Real d2_rho_imo, d2_rho_ipo, eta_i, delta_m_imo, delta_m_ipo;
  Real F_imo, F_i, F_ipo; //flattening coefficients (Fryxell Eqn 48)

  #ifdef DE
  Real ge_i, ge_imo, ge_ipo, ge_imt, ge_ipt;
  Real gel, ger, del_ge, ge_6, geL_0, geR_0;
  #endif


  // get a thread ID
  // global thread id in the gpu grid
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId*blockDim.x;
  // global id mapped to the real grid
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;
  // id within the thread block
  int id;


  if (xid < nx && yid < ny && zid < nz)
  {
    // load the 7-cell stencil into registers
    // cell i
    id = xid + yid*nx + zid*nx*ny;
    d_i  =  dev_conserved[            id];
    vx_i =  dev_conserved[o1*n_cells + id] / d_i;
    vy_i =  dev_conserved[o2*n_cells + id] / d_i;
    vz_i =  dev_conserved[o3*n_cells + id] / d_i;
    p_i  = (dev_conserved[4*n_cells + id] - 0.5*d_i*(vx_i*vx_i + vy_i*vy_i + vz_i*vz_i)) * (gamma - 1.0);
    p_i  = fmax(p_i, (Real) TINY_NUMBER);
    #ifdef DE
    ge_i =  dev_conserved[5*n_cells + id] / d_i;
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
    ge_imo =  dev_conserved[5*n_cells + id] / d_imo;
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
    ge_ipo =  dev_conserved[5*n_cells + id] / d_ipo;
    #endif
    // cell i-2
    if (dir == 0) id = xid-2 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-2)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-2)*nx*ny;
    d_imt  =  dev_conserved[            id];
    vx_imt =  dev_conserved[o1*n_cells + id] / d_imt;
    vy_imt =  dev_conserved[o2*n_cells + id] / d_imt;
    vz_imt =  dev_conserved[o3*n_cells + id] / d_imt;
    p_imt  = (dev_conserved[4*n_cells + id] - 0.5*d_imt*(vx_imt*vx_imt + vy_imt*vy_imt + vz_imt*vz_imt)) * (gamma - 1.0);
    p_imt  = fmax(p_imt, (Real) TINY_NUMBER);
    #ifdef DE
    ge_imt =  dev_conserved[5*n_cells + id] / d_imt;
    #endif
    // cell i+2
    if (dir == 0) id = xid+2 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid+2)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid+2)*nx*ny;
    d_ipt  =  dev_conserved[            id];
    vx_ipt =  dev_conserved[o1*n_cells + id] / d_ipt;
    vy_ipt =  dev_conserved[o2*n_cells + id] / d_ipt;
    vz_ipt =  dev_conserved[o3*n_cells + id] / d_ipt;
    p_ipt  = (dev_conserved[4*n_cells + id] - 0.5*d_ipt*(vx_ipt*vx_ipt + vy_ipt*vy_ipt + vz_ipt*vz_ipt)) * (gamma - 1.0);
    p_ipt  = fmax(p_ipt, (Real) TINY_NUMBER);
    #ifdef DE
    ge_ipt =  dev_conserved[5*n_cells + id] / d_ipt;
    #endif
    // cell i-3
    if (dir == 0) id = xid-3 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-3)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-3)*nx*ny;
    p_imth = (dev_conserved[4*n_cells + id] - 0.5*
             (dev_conserved[o1*n_cells + id]*dev_conserved[o1*n_cells + id] +
              dev_conserved[o2*n_cells + id]*dev_conserved[o2*n_cells + id] +
              dev_conserved[o3*n_cells + id]*dev_conserved[o3*n_cells + id]) / dev_conserved[id]) * (gamma - 1.0);
    p_imth = fmax(p_imth, (Real) TINY_NUMBER);
    // cell i+3
    if (dir == 0) id = xid+3 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid+3)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid+3)*nx*ny;
    p_ipth = (dev_conserved[4*n_cells + id] - 0.5*
             (dev_conserved[o1*n_cells + id]*dev_conserved[o1*n_cells + id] +
              dev_conserved[o2*n_cells + id]*dev_conserved[o2*n_cells + id] +
              dev_conserved[o3*n_cells + id]*dev_conserved[o3*n_cells + id]) / dev_conserved[id]) * (gamma - 1.0);
    p_ipth = fmax(p_imth, (Real) TINY_NUMBER);  
  

    //use ppm routines to set cell boundary values (see Fryxell Sec. 3.1.1)
    // left
    dl  = interface_value(d_imt,  d_imo,  d_i,  d_ipo,  dx);
    vxl = interface_value(vx_imt, vx_imo, vx_i, vx_ipo, dx);
    vyl = interface_value(vy_imt, vy_imo, vy_i, vy_ipo, dx);
    vzl = interface_value(vz_imt, vz_imo, vz_i, vz_ipo, dx);
    pl  = interface_value(p_imt,  p_imo,  p_i,  p_ipo,  dx);
    #ifdef DE
    gel = interface_value(ge_imt, ge_imo, ge_i, ge_ipo, dx);
    #endif
    // right
    dr  = interface_value(d_imo,  d_i,  d_ipo,  d_ipt,  dx);
    vxr = interface_value(vx_imo, vx_i, vx_ipo, vx_ipt, dx);
    vyr = interface_value(vy_imo, vy_i, vy_ipo, vy_ipt, dx);
    vzr = interface_value(vz_imo, vz_i, vz_ipo, vz_ipt, dx);
    pr  = interface_value(p_imo,  p_i,  p_ipo,  p_ipt,  dx);
    #ifdef DE
    ger = interface_value(ge_imo, ge_i, ge_ipo, ge_ipt, dx);
    #endif


#ifdef STEEPENING
    //check for contact discontinuities & steepen if necessary (see Fryxell Sec 3.1.2)
    //if condition 4 (Fryxell Eqn 37) (Colella Eqn 1.16.5) is true, check further conditions, otherwise do nothing
    if ((fabs(d_ipo - d_imo) / fmin(d_ipo, d_imo)) > 0.01)
    {
      //calculate the second derivative of the density in the imo and ipo cells
      d2_rho_imo = calc_d2_rho(d_imt, d_imo, d_i, dx);
      d2_rho_ipo = calc_d2_rho(d_i, d_ipo, d_ipt, dx);
      //if condition 1 (Fryxell Eqn 38) (Colella Eqn 1.16.5) is true, check further conditions, otherwise do nothing
      if ((d2_rho_imo * d2_rho_ipo) < 0)
      {
        //calculate condition 5, pressure vs density jumps (Fryxell Eqn 39) (Colella Eqn 3.2)
        //if c5 is true, set value of eta for discontinuity steepening
        if ((fabs(p_ipo - p_imo) / fmin(p_ipo, p_imo)) < 0.1 * gamma * (fabs(d_ipo - d_imo) / fmin(d_ipo, d_imo)))
        { 
          //calculate first eta value (Fryxell Eqn 36) (Colella Eqn 1.16.5)
          eta_i = calc_eta(d2_rho_imo, d2_rho_ipo, dx, d_imo, d_ipo);
          //calculate steepening coefficient (Fryxell Eqn 40) (Colella Eqn 1.16)
          eta_i = fmax(0, fmin(20*(eta_i-0.05), 1) );

          //calculate new left and right interface variables using monotonized slopes
          delta_m_imo = calc_delta_q(d_imt, d_imo, d_i, dx);
          delta_m_imo = limit_delta_q(delta_m_imo, d_imt, d_imo, d_i);
          delta_m_ipo = calc_delta_q(d_i, d_ipo, d_ipt, dx);
          delta_m_ipo = limit_delta_q(delta_m_ipo, d_i, d_ipo, d_ipt);

          //replace left and right interface values of density (Colella Eqn 1.14, 1.15)
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
    F_imo = fmax( 0, fmin(1, 10*(( (p_i -   p_imt) / (p_ipo - p_imth)) - 0.75)) );
    F_i   = fmax( 0, fmin(1, 10*(( (p_ipo - p_imo) / (p_ipt - p_imt))  - 0.75)) );
    F_ipo = fmax( 0, fmin(1, 10*(( (p_ipt - p_i)   / (p_ipth - p_imo)) - 0.75)) );
    //ensure that we are encountering a shock (Fryxell Eqns 46 & 47)
    if (fabs(p_i - p_imt) / fmin(p_i, p_imt) < 1./3.)  {F_imo = 0;}
    if (fabs(p_ipo - p_imo) / fmin(p_ipo, p_imo) < 1./3.)  {F_i = 0;}
    if (fabs(p_ipt - p_i) / fmin(p_ipt, p_i) < 1./3.)  {F_ipo = 0;}
    if (vx_i   - vx_imt > 0) {F_imo = 0;}
    if (vx_ipo - vx_imo > 0) {F_i   = 0;}
    if (vx_ipt - vx_i   > 0) {F_ipo = 0;}
    //set the flattening coefficient (Fryxell Eqn 48)
    if (p_ipo - p_imo < 0) {F_i = fmax(F_i, F_ipo);}
    else {F_i = fmax(F_i, F_imo);}
    //modify the interface values
    dl  = F_i * d_i  + (1 - F_i) * dl;
    vxl = F_i * vx_i + (1 - F_i) * vxl;
    vyl = F_i * vy_i + (1 - F_i) * vyl;
    vzl = F_i * vz_i + (1 - F_i) * vzl;
    pl  = F_i * p_i  + (1 - F_i) * pl;
    #ifdef DE
    gel = F_i * ge_i + (1 - F_i) * gel;
    #endif
    dr  = F_i * d_i  + (1 - F_i) * dr;
    vxr = F_i * vx_i + (1 - F_i) * vxr;
    vyr = F_i * vy_i + (1 - F_i) * vyr;
    vzr = F_i * vz_i + (1 - F_i) * vzr;
    pr  = F_i * p_i  + (1 - F_i) * pr;
    #ifdef DE
    ger = F_i * ge_i + (1 - F_i) * ger;
    #endif
#endif  


    // ensure that the parabolic distribution of each of the primative variables is monotonic
    // local maximum or minimum criterion (Fryxell Eqn 52, Fig 11)
    if ( (dr  - d_i)  * (d_i  - dl)  <= 0)  { dl  = dr  = d_i; }
    if ( (vxr - vx_i) * (vx_i - vxl) <= 0)  { vxl = vxr = vx_i; }
    if ( (vyr - vy_i) * (vy_i - vyl) <= 0)  { vyl = vyr = vy_i; }
    if ( (vzr - vz_i) * (vz_i - vzl) <= 0)  { vzl = vzr = vz_i; }
    if ( (pr  - p_i)  * (p_i  - pl)  <= 0)  { pl  = pr  = p_i; }
    #ifdef DE
    if ( (ger - ge_i) * (ge_i - gel) <= 0)  { gel = ger = ge_i; }
    #endif
    // steep gradient criterion (Fryxell Eqn 53, Fig 12)
    if ( (dr  - dl)  * (dl -  (3*d_i  - 2*dr))  < 0)  { dl  = 3*d_i  - 2*dr;  }
    if ( (vxr - vxl) * (vxl - (3*vx_i - 2*vxr)) < 0)  { vxl = 3*vx_i - 2*vxr; }
    if ( (vyr - vyl) * (vyl - (3*vy_i - 2*vyr)) < 0)  { vyl = 3*vy_i - 2*vyr; }
    if ( (vzr - vzl) * (vzl - (3*vz_i - 2*vzr)) < 0)  { vzl = 3*vz_i - 2*vzr; }
    if ( (pr  - pl)  * (pl -  (3*p_i  - 2*pr))  < 0)  { pl  = 3*p_i  - 2*pr;  }
    #ifdef DE
    if ( (ger - gel) * (gel - (3*ge_i - 2*ger)) < 0)  { gel = 3*ge_i - 2*ger; }
    #endif
    if ( (dr  - dl)  * ((3*d_i  - 2*dl)  - dr)  < 0)  { dr  = 3*d_i  - 2*dl;  }
    if ( (vxr - vxl) * ((3*vx_i - 2*vxl) - vxr) < 0)  { vxr = 3*vx_i - 2*vxl; }
    if ( (vyr - vyl) * ((3*vy_i - 2*vyl) - vyr) < 0)  { vyr = 3*vy_i - 2*vyl; }
    if ( (vzr - vzl) * ((3*vz_i - 2*vzl) - vzr) < 0)  { vzr = 3*vz_i - 2*vzl; }
    if ( (pr  - pl)  * ((3*p_i  - 2*pl)  - pr)  < 0)  { pr  = 3*p_i  - 2*pl;  }
    #ifdef DE
    if ( (ger - gel) * ((3*ge_i - 2*gel) - ger) < 0)  { ger = 3*ge_i - 2*gel; }
    #endif


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
    #ifdef DE
    del_ge = ger - gel; // Fryxell Eqn 29
    #endif

    d_6  = 6.0 * (d_i  - 0.5*(dl  + dr));  // Fryxell Eqn 30
    vx_6 = 6.0 * (vx_i - 0.5*(vxl + vxr)); // Fryxell Eqn 30
    vy_6 = 6.0 * (vy_i - 0.5*(vyl + vyr)); // Fryxell Eqn 30
    vz_6 = 6.0 * (vz_i - 0.5*(vzl + vzr)); // Fryxell Eqn 30
    p_6  = 6.0 * (p_i  - 0.5*(pl  + pr));  // Fryxell Eqn 30
    #ifdef DE
    ge_6 = 6.0 * (ge_i - 0.5*(gel + ger)); // Fryxell Eqn 30
    #endif

    // set speed of characteristics (v-c, v, v+c) using average values of v and c
    lambda_m = vx_i - cs;
    lambda_0 = vx_i;
    lambda_p = vx_i + cs;

    // calculate betas (for left state guesses)
    beta_m = fmax( (lambda_m * dt / dx) , 0 ); // Fryxell Eqn 59
    beta_0 = fmax( (lambda_0 * dt / dx) , 0 ); // Fryxell Eqn 59
    beta_p = fmax( (lambda_p * dt / dx) , 0 ); // Fryxell Eqn 59
 
    //calculate alphas (for right state guesses)
    alpha_m = fmax( (-lambda_m * dt / dx), 0); // Fryxell Eqn 61
    alpha_0 = fmax( (-lambda_0 * dt / dx), 0); // Fryxell Eqn 61
    alpha_p = fmax( (-lambda_p * dt / dx), 0); // Fryxell Eqn 61

    // average values under characteristics for left interface (Fryxell Eqn 60)
    dL_m  = dl  + 0.5 * alpha_m * (del_d  + d_6  * (1 - (2./3.) * alpha_m));
    vxL_m = vxl + 0.5 * alpha_m * (del_vx + vx_6 * (1 - (2./3.) * alpha_m));
    pL_m  = pl  + 0.5 * alpha_m * (del_p  + p_6  * (1 - (2./3.) * alpha_m));
    dL_0  = dl  + 0.5 * alpha_0 * (del_d  + d_6  * (1 - (2./3.) * alpha_0));
    vyL_0 = vyl + 0.5 * alpha_0 * (del_vy + vy_6 * (1 - (2./3.) * alpha_0));
    vzL_0 = vzl + 0.5 * alpha_0 * (del_vz + vz_6 * (1 - (2./3.) * alpha_0));
    #ifdef DE
    geL_0 = gel + 0.5 * alpha_0 * (del_ge + ge_6 * (1 - (2./3.) * alpha_0));
    #endif
    pL_0  = pl  + 0.5 * alpha_0 * (del_p  + p_6  * (1 - (2./3.) * alpha_0));
    vxL_p = vxl + 0.5 * alpha_p * (del_vx + vx_6 * (1 - (2./3.) * alpha_p));
    pL_p  = pl  + 0.5 * alpha_p * (del_p  + p_6  * (1 - (2./3.) * alpha_p));

    // average values under characteristics for right interface (Fryxell Eqn 58)
    vxR_m = vxr - 0.5 * beta_m * (del_vx - vx_6 * (1 - (2./3.) * beta_m));
    pR_m  = pr  - 0.5 * beta_m * (del_p  - p_6  * (1 - (2./3.) * beta_m));
    dR_0  = dr  - 0.5 * beta_0 * (del_d  - d_6  * (1 - (2./3.) * beta_0));
    vyR_0 = vyr - 0.5 * beta_0 * (del_vy - vy_6 * (1 - (2./3.) * beta_0));
    vzR_0 = vzr - 0.5 * beta_0 * (del_vz - vz_6 * (1 - (2./3.) * beta_0));
    #ifdef DE
    geR_0 = ger - 0.5 * beta_0 * (del_ge - ge_6 * (1 - (2./3.) * beta_0));
    #endif
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
    #ifdef DE
    gel = geL_0;
    #endif
    pl  = pL_m;
    // right
    dr  = dR_p;
    vxr = vxR_p;
    vyr = vyR_0;
    vzr = vzR_0;
    #ifdef DE
    ger = geR_0;
    #endif
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
    pl = pl + (dl*dl*cl*cl) * (chi_L_p + chi_L_m);
    vxl = vxl + dl*cl * (chi_L_p - chi_L_m);
    dl = pow( ((1.0/dl) - (chi_L_m + chi_L_0 + chi_L_p)) , -1);
    pr = pr + (dr*dr*cr*cr) * (chi_R_p + chi_R_m);
    vxr = vxr + dr*cr * (chi_R_p - chi_R_m);
    dr = pow( ((1.0/dr) - (chi_R_m + chi_R_0 + chi_R_p)) , -1);

    // enforce minimum values
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
    dev_bounds_R[o1*n_cells + id] = dl*vxl;
    dev_bounds_R[o2*n_cells + id] = dl*vyl;
    dev_bounds_R[o3*n_cells + id] = dl*vzl;
    dev_bounds_R[4*n_cells + id] = pl/(gamma-1.0) + 0.5*dl*(vxl*vxl + vyl*vyl + vzl*vzl);    
    #ifdef DE
    dev_bounds_R[5*n_cells + id] = dl*gel;    
    #endif
    // bounds_L refers to the left side of the i+1/2 interface
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_L[            id] = dr;
    dev_bounds_L[o1*n_cells + id] = dr*vxr;
    dev_bounds_L[o2*n_cells + id] = dr*vyr;
    dev_bounds_L[o3*n_cells + id] = dr*vzr;
    dev_bounds_L[4*n_cells + id] = pr/(gamma-1.0) + 0.5*dr*(vxr*vxr + vyr*vyr + vzr*vzr);
    #ifdef DE
    dev_bounds_L[5*n_cells + id] = dr*ger;
    #endif

  }
}



/*! \fn interface_value 
 *  \brief Returns the interpolated value at i | ipo cell interface.*/
__device__ Real interface_value(Real q_imo, Real q_i, Real q_ipo, Real q_ipt, Real dx)
{
  Real dq_i;
  Real dq_ipo;

  dq_i = calc_delta_q(q_imo, q_i, q_ipo, dx);
  dq_ipo = calc_delta_q(q_i, q_ipo, q_ipt, dx);

  dq_i = limit_delta_q(dq_i, q_imo, q_i, q_ipo);
  dq_ipo = limit_delta_q(dq_ipo, q_i, q_ipo, q_ipt);

  return q_i + 0.5*(q_ipo - q_i) - (1./6.)*(dq_ipo - dq_i); 

}


/*! \fn calc_delta_q
 *  \brief Returns the average slope in zone i of the parabola with zone averages
     of imo, i, and ipo. See Fryxell Eqn 24. */
__device__ Real calc_delta_q(Real q_imo, Real q_i, Real q_ipo, Real dx)
{
  return 0.5*(q_ipo - q_imo);
}


/*! \fn limit_delta_q
 *  \brief Limits the value of delta_rho according to Fryxell Eqn 26
     to ensure monotonic interface values. van Leer limiters. */
__device__ Real limit_delta_q(Real del_in, Real q_imo, Real q_i, Real q_ipo)
{
  if ( (q_ipo-q_i)*(q_i-q_imo) > 0)
  {
    return minof3(fabs(del_in), 2*fabs(q_i-q_imo), 2*fabs(q_i - q_ipo)) * sgn_CUDA(del_in);
  }
  else return 0;
}


/*! \fn calc_d2_rho
 *  \brief Returns the second derivative of rho across zone i. (Fryxell Eqn 35) */
__device__ Real calc_d2_rho(Real rho_imo, Real rho_i, Real rho_ipo, Real dx)
{
  return (1. / (6*dx*dx)) * (rho_ipo - 2*rho_i + rho_imo);
} 


/*! \fn calc_eta
 *  \brief Returns a dimensionless quantity relating the 1st and 3rd derivatives
    See Fryxell Eqn 36. */
__device__ Real calc_eta(Real d2rho_imo, Real d2rho_ipo, Real dx, Real rho_imo, Real rho_ipo)
{
  Real A, B;

  A = (d2rho_ipo - d2rho_imo)*dx*dx;
  B = 1.0 / (rho_ipo - rho_imo);

  return -A * B;
}




#endif //PPMP
#endif //CUDA

/*! \file ppmc_ctu_cuda.cu
 *  \brief Functions definitions for the ppm kernels, using characteristic tracing. 
           Written following Stone et al. 2008. */
#ifdef CUDA
#ifdef PPMC

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"ppmc_ctu_cuda.h"



/*! \fn void PPMC_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using ppm. */
__global__ void PPMC_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
{
  int n_cells = nx*ny*nz;
  int o1, o2, o3;
  if (dir == 0 ) {
    o1 = 1; o2 = 2; o3 = 3;
  }
  if (dir == 1 ) {
    o1 = 2; o2 = 3; o3 = 1;
  }
  if (dir == 2 ) {
    o1 = 3; o2 = 1; o3 = 2;
  }

  // declare primative variables for each stencil
  // these will be placed into registers for each thread
  Real d_i, vx_i, vy_i, vz_i, p_i;
  Real d_imo, vx_imo, vy_imo, vz_imo, p_imo; 
  Real d_ipo, vx_ipo, vy_ipo, vz_ipo, p_ipo;
  Real d_imt, vx_imt, vy_imt, vz_imt, p_imt;
  Real d_ipt, vx_ipt, vy_ipt, vz_ipt, p_ipt;

  Real dtodx = dt/dx;
 
  // declare other variables to be used
  Real a;
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
  #ifdef DE
  Real ge_i, ge_imo, ge_ipo, ge_imt, ge_ipt;
  Real del_ge_L, del_ge_R, del_ge_C, del_ge_G;
  Real del_ge_m_imo, del_ge_m_i, del_ge_m_ipo;
  Real ge_L, ge_R, ge_6;
  Real chi_6, sum_6;
  #endif


  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId * blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;

  if (xid < nx && yid < ny && zid < nz)
  {
    // load the 5-cell stencil into registers
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
    
    //printf("%d %d %d %f %f %f %f %f\n", xid, yid, zid, d_i, vx_i, vy_i, vz_i, p_i);

    // Steps 2 - 5 are repeated for cell i-1, i, and i+1
    // Step 2 - Compute the left, right, centered, and van Leer differences of the primative variables
    //          Note that here L and R refer to locations relative to the cell center
    //          Stone Eqn 36

    // calculate the adiabatic sound speed in cell imo
    a = sqrt(gamma*p_imo/d_imo);

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

    #ifdef DE
    del_ge_L  = ge_imo  - ge_imt;
    del_ge_R  = ge_i  - ge_imo;
    del_ge_C  = 0.5*(ge_i - ge_imt);
    if (del_ge_L*del_ge_R > 0.0) { del_ge_G = 2.0*del_ge_L*del_ge_R / (del_ge_L+del_ge_R); }
    else { del_ge_G = 0.0; } 
    #endif


    // Step 3 - Project the left, right, centered and van Leer differences onto the characteristic variables
    //          Stone Eqn 37 (del_a are differences in characteristic variables, see Stone for notation)
    //          Use the eigenvectors given in Stone 2008, Appendix A

    del_a_0_L = -0.5*d_imo*del_vx_L/a + 0.5*del_p_L/(a*a);
    del_a_1_L = del_d_L - del_p_L/(a*a);
    del_a_2_L = del_vy_L;
    del_a_3_L = del_vz_L;
    del_a_4_L = 0.5*d_imo*del_vx_L/a + 0.5*del_p_L/(a*a);

    del_a_0_R = -0.5*d_imo*del_vx_R/a + 0.5*del_p_R/(a*a);
    del_a_1_R = del_d_R - del_p_R/(a*a);
    del_a_2_R = del_vy_R;
    del_a_3_R = del_vz_R;
    del_a_4_R = 0.5*d_imo*del_vx_R/a + 0.5*del_p_R/(a*a);

    del_a_0_C = -0.5*d_imo*del_vx_C/a + 0.5*del_p_C/(a*a);
    del_a_1_C = del_d_C - del_p_C/(a*a);
    del_a_2_C = del_vy_C;
    del_a_3_C = del_vz_C;
    del_a_4_C = 0.5*d_imo*del_vx_C/a + 0.5*del_p_C/(a*a);

    del_a_0_G = -0.5*d_imo*del_vx_G/a + 0.5*del_p_G/(a*a);
    del_a_1_G = del_d_G - del_p_G/(a*a);
    del_a_2_G = del_vy_G;
    del_a_3_G = del_vz_G;
    del_a_4_G = 0.5*d_imo*del_vx_G/a + 0.5*del_p_G/(a*a); 


    // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
    //          Stone Eqn 38

    del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;
    
    if (del_a_0_L*del_a_0_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
      lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
      del_a_0_m = sgn(del_a_0_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_1_L*del_a_1_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
      lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
      del_a_1_m = sgn(del_a_1_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_2_L*del_a_2_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
      lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
      del_a_2_m = sgn(del_a_2_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_3_L*del_a_3_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
      lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
      del_a_3_m = sgn(del_a_3_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_4_L*del_a_4_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
      lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
      del_a_4_m = sgn(del_a_4_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b); 
    }
    #ifdef DE
    if (del_ge_L*del_ge_R > 0.0) {
      lim_slope_a = fmin(fabs(del_ge_L), fabs(del_ge_R));
      lim_slope_b = fmin(fabs(del_ge_C), fabs(del_ge_G));
      del_ge_m_imo = sgn(del_ge_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    else del_ge_m_imo = 0.0;
    #endif


    // Step 5 - Project the monotonized difference in the characteristic variables back onto the 
    //          primative variables
    //          Stone Eqn 39

    del_d_m_imo  = del_a_0_m + del_a_1_m + del_a_4_m;
    del_vx_m_imo = -a*del_a_0_m/d_imo + a*del_a_4_m/d_imo;
    del_vy_m_imo = del_a_2_m;
    del_vz_m_imo = del_a_3_m;
    del_p_m_imo  = a*a*del_a_0_m + a*a*del_a_4_m;  


    // Step 2 - Compute the left, right, centered, and van Leer differences of the primative variables
    //          Note that here L and R refer to locations relative to the cell center
    //          Stone Eqn 36
    
    // calculate the adiabatic sound speed in cell i
    a = sqrt(gamma*p_i/d_i);

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

    #ifdef DE
    del_ge_L = ge_i - ge_imo;
    del_ge_R = ge_ipo - ge_i;
    del_ge_C = 0.5*(ge_ipo - ge_imo);
    if (del_ge_L*del_ge_R > 0.0) { del_ge_G = 2.0*del_ge_L*del_ge_R / (del_ge_L+del_ge_R); }
    else { del_ge_G = 0.0; } 
    #endif


    // Step 3 - Project the left, right, centered, and van Leer differences onto the characteristic variables
    //          Stone Eqn 37 (del_a are differences in characteristic variables, see Stone for notation)
    //          Use the eigenvectors given in Stone 2008, Appendix A

    del_a_0_L = -0.5*d_i*del_vx_L/a + 0.5*del_p_L/(a*a);
    del_a_1_L = del_d_L - del_p_L/(a*a);
    del_a_2_L = del_vy_L;
    del_a_3_L = del_vz_L;
    del_a_4_L = 0.5*d_i*del_vx_L/a + 0.5*del_p_L/(a*a);

    del_a_0_R = -0.5*d_i*del_vx_R/a + 0.5*del_p_R/(a*a);
    del_a_1_R = del_d_R - del_p_R/(a*a);
    del_a_2_R = del_vy_R;
    del_a_3_R = del_vz_R;
    del_a_4_R = 0.5*d_i*del_vx_R/a + 0.5*del_p_R/(a*a);

    del_a_0_C = -0.5*d_i*del_vx_C/a + 0.5*del_p_C/(a*a);
    del_a_1_C = del_d_C - del_p_C/(a*a);
    del_a_2_C = del_vy_C;
    del_a_3_C = del_vz_C;
    del_a_4_C = 0.5*d_i*del_vx_C/a + 0.5*del_p_C/(a*a);

    del_a_0_G = -0.5*d_i*del_vx_G/a + 0.5*del_p_G/(a*a);
    del_a_1_G = del_d_G - del_p_G/(a*a);
    del_a_2_G = del_vy_G;
    del_a_3_G = del_vz_G;
    del_a_4_G = 0.5*d_i*del_vx_G/a + 0.5*del_p_G/(a*a);


    // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
    //          Stone Eqn 38

    del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;
    
    if (del_a_0_L*del_a_0_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
      lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
      del_a_0_m = sgn(del_a_0_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_1_L*del_a_1_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
      lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
      del_a_1_m = sgn(del_a_1_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_2_L*del_a_2_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
      lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
      del_a_2_m = sgn(del_a_2_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_3_L*del_a_3_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
      lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
      del_a_3_m = sgn(del_a_3_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_4_L*del_a_4_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
      lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
      del_a_4_m = sgn(del_a_4_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b); 
    }
    #ifdef DE
    if (del_ge_L*del_ge_R > 0.0) {
      lim_slope_a = fmin(fabs(del_ge_L), fabs(del_ge_R));
      lim_slope_b = fmin(fabs(del_ge_C), fabs(del_ge_G));
      del_ge_m_i = sgn(del_ge_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b); 
    }
    else del_ge_m_i = 0.0;
    #endif


    // Step 5 - Project the monotonized difference in the characteristic variables back onto the 
    //          primative variables
    //          Stone Eqn 39

    del_d_m_i  = del_a_0_m + del_a_1_m + del_a_4_m;
    del_vx_m_i = -a*del_a_0_m/d_i + a*del_a_4_m/d_i;
    del_vy_m_i = del_a_2_m;
    del_vz_m_i = del_a_3_m;
    del_p_m_i  = a*a*del_a_0_m + a*a*del_a_4_m;  


    // Step 2 - Compute the left, right, centered, and van Leer differences of the primative variables
    //          Note that here L and R refer to locations relative to the cell center
    //          Stone Eqn 36


    // calculate the adiabatic sound speed in cell ipo
    a = sqrt(gamma*p_ipo/d_ipo);

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

    #ifdef DE
    del_ge_L = ge_ipo - ge_i;
    del_ge_R = ge_ipt - ge_ipo;
    del_ge_C = 0.5*(ge_ipt- ge_i);
    if (del_ge_L*del_ge_R > 0.0) { del_ge_G = 2.0*del_ge_L*del_ge_R / (del_ge_L+del_ge_R); }
    else { del_ge_G = 0.0; } 
    #endif

    // Step 3 - Project the left, right, centered, and van Leer differences onto the characteristic variables
    //          Stone Eqn 37 (del_a are differences in characteristic variables, see Stone for notation)
    //          Use the eigenvectors given in Stone 2008, Appendix A
    
    del_a_0_L = -0.5*d_ipo*del_vx_L/a + 0.5*del_p_L/(a*a);
    del_a_1_L = del_d_L - del_p_L/(a*a);
    del_a_2_L = del_vy_L;
    del_a_3_L = del_vz_L;
    del_a_4_L = 0.5*d_ipo*del_vx_L/a + 0.5*del_p_L/(a*a);

    del_a_0_R = -0.5*d_ipo*del_vx_R/a + 0.5*del_p_R/(a*a);
    del_a_1_R = del_d_R - del_p_R/(a*a);
    del_a_2_R = del_vy_R;
    del_a_3_R = del_vz_R;
    del_a_4_R = 0.5*d_ipo*del_vx_R/a + 0.5*del_p_R/(a*a);

    del_a_0_C = -0.5*d_ipo*del_vx_C/a + 0.5*del_p_C/(a*a);
    del_a_1_C = del_d_C - del_p_C/(a*a);
    del_a_2_C = del_vy_C;
    del_a_3_C = del_vz_C;
    del_a_4_C = 0.5*d_ipo*del_vx_C/a + 0.5*del_p_C/(a*a);

    del_a_0_G = -0.5*d_ipo*del_vx_G/a + 0.5*del_p_G/(a*a);
    del_a_1_G = del_d_G - del_p_G/(a*a);
    del_a_2_G = del_vy_G;
    del_a_3_G = del_vz_G;
    del_a_4_G = 0.5*d_ipo*del_vx_G/a + 0.5*del_p_G/(a*a); 


    // Step 4 - Apply monotonicity constraints to the differences in the characteristic variables
    //          Stone Eqn 38

    del_a_0_m = del_a_1_m = del_a_2_m = del_a_3_m = del_a_4_m = 0.0;
    
    if (del_a_0_L*del_a_0_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_0_L), fabs(del_a_0_R));
      lim_slope_b = fmin(fabs(del_a_0_C), fabs(del_a_0_G));
      del_a_0_m = sgn(del_a_0_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_1_L*del_a_1_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_1_L), fabs(del_a_1_R));
      lim_slope_b = fmin(fabs(del_a_1_C), fabs(del_a_1_G));
      del_a_1_m = sgn(del_a_1_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_2_L*del_a_2_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_2_L), fabs(del_a_2_R));
      lim_slope_b = fmin(fabs(del_a_2_C), fabs(del_a_2_G));
      del_a_2_m = sgn(del_a_2_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_3_L*del_a_3_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_3_L), fabs(del_a_3_R));
      lim_slope_b = fmin(fabs(del_a_3_C), fabs(del_a_3_G));
      del_a_3_m = sgn(del_a_3_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);
    }
    if (del_a_4_L*del_a_4_R > 0.0) {
      lim_slope_a = fmin(fabs(del_a_4_L), fabs(del_a_4_R));
      lim_slope_b = fmin(fabs(del_a_4_C), fabs(del_a_4_G));
      del_a_4_m = sgn(del_a_4_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b); 
    }
    #ifdef DE
    if (del_ge_L*del_ge_R > 0.0) {
      lim_slope_a = fmin(fabs(del_ge_L), fabs(del_ge_R));
      lim_slope_b = fmin(fabs(del_ge_C), fabs(del_ge_G));
      del_ge_m_ipo = sgn(del_ge_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b); 
    }
    else del_ge_m_ipo = 0.0;
    #endif


    // Step 5 - Project the monotonized difference in the characteristic variables back onto the 
    //          primative variables
    //          Stone Eqn 39

    del_d_m_ipo  = del_a_0_m + del_a_1_m + del_a_4_m;
    del_vx_m_ipo = -a*del_a_0_m / d_ipo + a* del_a_4_m / d_ipo;
    del_vy_m_ipo = del_a_2_m;
    del_vz_m_ipo = del_a_3_m;
    del_p_m_ipo  = a*a*del_a_0_m + a*a*del_a_4_m;  


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

    #ifdef DE
    ge_L  = 0.5*(ge_i + ge_imo)   - (del_ge_m_i  - del_ge_m_imo)  / 6.0;
    ge_R  = 0.5*(ge_ipo + ge_i)   - (del_ge_m_ipo  - del_ge_m_i)  / 6.0;
    #endif


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

    #ifdef DE
    if ((ge_R  - ge_i) *(ge_i  - ge_L)  <= 0) ge_L  = ge_R  = ge_i;
    if ( 6.0*(ge_R  - ge_L) *(ge_i  - 0.5*(ge_L  + ge_R))  > (ge_R  - ge_L) *(ge_R  - ge_L))  ge_L  = 3.0*ge_i  - 2.0*ge_R;
    if ( 6.0*(ge_R  - ge_L) *(ge_i  - 0.5*(ge_L  + ge_R))  < -(ge_R  - ge_L) *(ge_R  - ge_L))  ge_R  = 3.0*ge_i  - 2.0*ge_L;
    ge_L  = fmax( fmin(ge_i,  ge_imo), ge_L );
    ge_L  = fmin( fmax(ge_i,  ge_imo), ge_L );
    ge_R  = fmax( fmin(ge_i,  ge_ipo), ge_R );
    ge_R  = fmin( fmax(ge_i,  ge_ipo), ge_R );
    #endif

    // Step 8 - Compute the coefficients for the monotonized parabolic interpolation function
    //          Stone Eqn 54

    del_d_m_i  = d_R  - d_L;
    del_vx_m_i = vx_R - vx_L;
    del_vy_m_i = vy_R - vy_L;
    del_vz_m_i = vz_R - vz_L;
    del_p_m_i  = p_R  - p_L;

    d_6  = 6.0*(d_i  - 0.5*(d_L  + d_R));
    vx_6 = 6.0*(vx_i - 0.5*(vx_L + vx_R));
    vy_6 = 6.0*(vy_i - 0.5*(vy_L + vy_R));
    vz_6 = 6.0*(vz_i - 0.5*(vz_L + vz_R));
    p_6  = 6.0*(p_i  - 0.5*(p_L  + p_R));

    #ifdef DE
    del_ge_m_i = ge_R - ge_L;
    ge_6 = 6.0*(ge_i - 0.5*(ge_L + ge_R));
    #endif


    // Step 1 - Compute the eigenvalues of the linearized equations in the
    //          primative variables using the cell-centered primative variables

    // recalculate the adiabatic sound speed in cell i
    a = sqrt(gamma*p_i/d_i);

    lambda_m = vx_i-a;
    lambda_0 = vx_i;
    lambda_p = vx_i+a; 


    // Step 9 - Compute the left and right interface values using monotonized parabolic interpolation
    //          Stone Eqns 55 & 56

    // largest eigenvalue
    lambda_max = fmax(lambda_p, 0);
    // smallest eigenvalue
    lambda_min = fmin(lambda_m, 0);


    // left interface value, i+1/2
    d_R  = d_R  - lambda_max * (0.5*dtodx)*(del_d_m_i  - (1.0 - (2.0/3.0)*lambda_max*dtodx)*d_6);
    vx_R = vx_R - lambda_max * (0.5*dtodx)*(del_vx_m_i - (1.0 - (2.0/3.0)*lambda_max*dtodx)*vx_6);
    vy_R = vy_R - lambda_max * (0.5*dtodx)*(del_vy_m_i - (1.0 - (2.0/3.0)*lambda_max*dtodx)*vy_6);
    vz_R = vz_R - lambda_max * (0.5*dtodx)*(del_vz_m_i - (1.0 - (2.0/3.0)*lambda_max*dtodx)*vz_6);
    p_R  = p_R  - lambda_max * (0.5*dtodx)*(del_p_m_i  - (1.0 - (2.0/3.0)*lambda_max*dtodx)*p_6);
    
    // right interface value, i-1/2
    d_L  = d_L  - lambda_min * (0.5*dtodx)*(del_d_m_i  + (1.0 + (2.0/3.0)*lambda_min*dtodx)*d_6);
    vx_L = vx_L - lambda_min * (0.5*dtodx)*(del_vx_m_i + (1.0 + (2.0/3.0)*lambda_min*dtodx)*vx_6);
    vy_L = vy_L - lambda_min * (0.5*dtodx)*(del_vy_m_i + (1.0 + (2.0/3.0)*lambda_min*dtodx)*vy_6);
    vz_L = vz_L - lambda_min * (0.5*dtodx)*(del_vz_m_i + (1.0 + (2.0/3.0)*lambda_min*dtodx)*vz_6);
    p_L  = p_L  - lambda_min * (0.5*dtodx)*(del_p_m_i  + (1.0 + (2.0/3.0)*lambda_min*dtodx)*p_6);

    #ifdef DE
    ge_R = ge_R - lambda_max * (0.5*dtodx)*(del_ge_m_i - (1.0 - (2.0/3.0)*lambda_max*dtodx)*ge_6);
    ge_L = ge_L - lambda_min * (0.5*dtodx)*(del_ge_m_i + (1.0 + (2.0/3.0)*lambda_min*dtodx)*ge_6);
    #endif

    // Step 10 - Perform the characteristic tracing
    //           Stone Eqns 57 - 60 

    // left-hand interface value, i+1/2
    sum_1 = 0;
    sum_2 = 0;
    sum_3 = 0;
    sum_4 = 0;
    sum_5 = 0;
    #ifdef DE
    sum_6 = 0;
    #endif
    if (lambda_m >= 0)
    {
      A = (0.5*dtodx) * (lambda_p - lambda_m); 
      B = (1.0/3.0)*(dtodx)*(dtodx)*(lambda_p*lambda_p - lambda_m*lambda_m);

      chi_1 = A*(del_d_m_i - d_6) + B*d_6; 
      chi_2 = A*(del_vx_m_i - vx_6) + B*vx_6; 
      chi_3 = A*(del_vy_m_i - vy_6) + B*vy_6; 
      chi_4 = A*(del_vz_m_i - vz_6) + B*vz_6; 
      chi_5 = A*(del_p_m_i - p_6) + B*p_6;

      sum_1 += -0.5*(d_i*chi_2/a - chi_5/(a*a));
      sum_2 += 0.5*(chi_2 - chi_5/(a*d_i));
      sum_5 += -0.5*(d_i*chi_2*a - chi_5);
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
      #ifdef DE
      chi_6 = A*(del_ge_m_i - ge_6) + B*ge_6;
      #endif
    
      sum_1 += chi_1 - chi_5/(a*a);
      sum_3 += chi_3;
      sum_4 += chi_4;
      #ifdef DE
      sum_6 += chi_6;
      #endif
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

      sum_1 += 0.5*(d_i*chi_2/a + chi_5/(a*a));
      sum_2 += 0.5*(chi_2 + chi_5/(a*d_i));
      sum_5 += 0.5*(d_i*chi_2*a + chi_5);
    }

    // add the corrections to the initial guesses for the interface values
    d_R += sum_1;
    vx_R += sum_2;
    vy_R += sum_3;
    vz_R += sum_4;
    p_R += sum_5;
    #ifdef DE
    ge_R += sum_6;
    #endif


    // right-hand interface value, i-1/2
    sum_1 = 0;
    sum_2 = 0;
    sum_3 = 0;
    sum_4 = 0;
    sum_5 = 0;
    #ifdef DE
    sum_6 = 0;
    #endif
    if (lambda_m <= 0)
    {
      C = (0.5*dtodx) * (lambda_m - lambda_m);
      D = (1.0/3.0)*(dtodx)*(dtodx)*(lambda_m*lambda_m - lambda_m*lambda_m);
   
      chi_1 = C*(del_d_m_i + d_6) + D*d_6; 
      chi_2 = C*(del_vx_m_i + vx_6) + D*vx_6; 
      chi_3 = C*(del_vy_m_i + vy_6) + D*vy_6; 
      chi_4 = C*(del_vz_m_i + vz_6) + D*vz_6; 
      chi_5 = C*(del_p_m_i + p_6) + D*p_6;

      sum_1 += -0.5*(d_i*chi_2/a - chi_5/(a*a));
      sum_2 += 0.5*(chi_2 - chi_5/(a*d_i));
      sum_5 += -0.5*(d_i*chi_2*a - chi_5);
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
      #ifdef DE
      chi_6 = C*(del_ge_m_i + ge_6) + D*ge_6; 
      #endif
    
      sum_1 += chi_1 - chi_5/(a*a);
      sum_3 += chi_3;
      sum_4 += chi_4;
      #ifdef DE
      sum_6 += chi_6;
      #endif
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

      sum_1 += 0.5*(d_i*chi_2/a + chi_5/(a*a));
      sum_2 += 0.5*(chi_2 + chi_5/(a*d_i));
      sum_5 += 0.5*(d_i*chi_2*a + chi_5);
    }

    // add the corrections
    d_L += sum_1;
    vx_L += sum_2;
    vy_L += sum_3;
    vz_L += sum_4;
    p_L += sum_5;
    #ifdef DE
    ge_L += sum_6;
    #endif


    // enfore minimum values
    d_L = fmax(d_L, (Real) TINY_NUMBER);
    d_R = fmax(d_R, (Real) TINY_NUMBER);
    p_L = fmax(p_L, (Real) TINY_NUMBER);
    p_R = fmax(p_R, (Real) TINY_NUMBER);

    // Step 11 - Send final values back from kernel 

    // bounds_R refers to the right side of the i-1/2 interface
    if (dir == 0) id = xid-1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-1)*nx*ny;
    dev_bounds_R[            id] = d_L;
    dev_bounds_R[o1*n_cells + id] = d_L*vx_L;
    dev_bounds_R[o2*n_cells + id] = d_L*vy_L;
    dev_bounds_R[o3*n_cells + id] = d_L*vz_L;
    dev_bounds_R[4*n_cells + id] = p_L/(gamma-1.0) + 0.5*d_L*(vx_L*vx_L + vy_L*vy_L + vz_L*vz_L);    
    #ifdef DE
    dev_bounds_R[5*n_cells + id] = d_L*ge_L;
    #endif
    // bounds_L refers to the left side of the i+1/2 interface
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_L[            id] = d_R;
    dev_bounds_L[o1*n_cells + id] = d_R*vx_R;
    dev_bounds_L[o2*n_cells + id] = d_R*vy_R;
    dev_bounds_L[o3*n_cells + id] = d_R*vz_R;
    dev_bounds_L[4*n_cells + id] = p_R/(gamma-1.0) + 0.5*d_R*(vx_R*vx_R + vy_R*vy_R + vz_R*vz_R);
    #ifdef DE
    dev_bounds_L[5*n_cells + id] = d_R*ge_R;
    #endif

  }
}


#endif //PPMC
#endif //CUDA


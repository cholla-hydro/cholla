/*! \file ppmp_cuda.cu
 *  \brief Definitions of the piecewise parabolic reconstruction (Fryxell 2000) functions 
           with limiting in the primative variables. */
#ifdef CUDA
#ifdef PPMP

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"ppmp_cuda.h"

#ifdef DE //PRESSURE_DE
#include"hydro_cuda.h"
#endif

// #define STEEPENING
// #define FLATTENING
//Note: Errors when using FLATTENING, need to check the ghost cells

/*! \fn __global__ void PPMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir, int n_fields)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using ppm with limiting in the primative variables. */
__global__ void PPMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir, int n_fields)
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

  // declare primative variables in the stencil
  Real d_i, vx_i, vy_i, vz_i, p_i;
  Real d_imo, vx_imo, vy_imo, vz_imo, p_imo; 
  Real d_ipo, vx_ipo, vy_ipo, vz_ipo, p_ipo;
  Real d_imt, vx_imt, vy_imt, vz_imt, p_imt; 
  Real d_ipt, vx_ipt, vy_ipt, vz_ipt, p_ipt;
  #ifdef FLATTENING
  Real p_imth, p_ipth;
  #endif

  // declare left and right interface values
  Real d_L, vx_L, vy_L, vz_L, p_L;
  Real d_R, vx_R, vy_R, vz_R, p_R;

  // declare other variables
  Real del_q_imo, del_q_i, del_q_ipo;

  #ifdef CTU
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
  #endif  

  #ifdef DE
  Real ge_i, ge_imo, ge_ipo, ge_imt, ge_ipt, ge_L, ge_R, E_kin, E, dge;
  #ifdef CTU
  Real del_ge, ge_6, geL_0, geR_0;
  #endif
  #endif
  
  #ifdef SCALAR
  Real scalar_i[NSCALARS], scalar_imo[NSCALARS], scalar_ipo[NSCALARS], scalar_imt[NSCALARS], scalar_ipt[NSCALARS];
  Real scalar_L[NSCALARS], scalar_R[NSCALARS];
  #ifdef CTU
  Real del_scalar[NSCALARS], scalar_6[NSCALARS], scalarL_0[NSCALARS], scalarR_0[NSCALARS];
  #endif
  #endif



  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId*blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;

  int xs, xe, ys, ye, zs, ze;
  
  // 
  // if (dir == 0) {
  //   xs = 3; xe = nx-4;
  //   ys = 0; ye = ny;
  //   zs = 0; ze = nz;
  // }
  // if (dir == 1) {
  //   xs = 0; xe = nx;
  //   ys = 3; ye = ny-4;
  //   zs = 0; ze = nz;
  // }
  // if (dir == 2) {
  //   xs = 0; xe = nx;
  //   ys = 0; ye = ny;
  //   zs = 3; ze = nz-4;
  // }
  
  //Ignore only the 2 ghost cells on each side ( intead of ignoring 3 ghost cells on each side )
  if (dir == 0) {
    xs = 2; xe = nx-3;
    ys = 0; ye = ny;
    zs = 0; ze = nz;
  }
  if (dir == 1) {
    xs = 0; xe = nx;
    ys = 2; ye = ny-3;
    zs = 0; ze = nz;
  }
  if (dir == 2) {
    xs = 0; xe = nx;
    ys = 0; ye = ny;
    zs = 2; ze = nz-3;
  }

  if (xid >= xs && xid < xe && yid >= ys && yid < ye && zid >= zs && zid < ze)
  {
    // load the 5-cell stencil into registers
    // cell i
    id = xid + yid*nx + zid*nx*ny;
    d_i  =  dev_conserved[            id];
    vx_i =  dev_conserved[o1*n_cells + id] / d_i;
    vy_i =  dev_conserved[o2*n_cells + id] / d_i;
    vz_i =  dev_conserved[o3*n_cells + id] / d_i;
    #ifdef DE //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    E_kin = 0.5 * d_i * ( vx_i*vx_i + vy_i*vy_i + vz_i*vz_i );
    dge = dev_conserved[(n_fields-1)*n_cells + id];
    p_i = Get_Pressure_From_DE( E, E - E_kin, dge, gamma ); 
    #else 
    p_i  = (dev_conserved[4*n_cells + id] - 0.5*d_i*(vx_i*vx_i + vy_i*vy_i + vz_i*vz_i)) * (gamma - 1.0);
    #endif //PRESSURE_DE
    p_i  = fmax(p_i, (Real) TINY_NUMBER);
    #ifdef DE
    ge_i = dge / d_i;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_i[i] =  dev_conserved[(5+i)*n_cells + id] / d_i;
    }    
    #endif
    // cell i-1
    if (dir == 0) id = xid-1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-1)*nx*ny;
    d_imo  =  dev_conserved[            id];
    vx_imo =  dev_conserved[o1*n_cells + id] / d_imo;
    vy_imo =  dev_conserved[o2*n_cells + id] / d_imo;
    vz_imo =  dev_conserved[o3*n_cells + id] / d_imo;
    #ifdef DE //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    E_kin = 0.5 * d_imo * ( vx_imo*vx_imo + vy_imo*vy_imo + vz_imo*vz_imo );
    dge = dev_conserved[(n_fields-1)*n_cells + id];
    p_imo = Get_Pressure_From_DE( E, E - E_kin, dge, gamma ); 
    #else    
    p_imo  = (dev_conserved[4*n_cells + id] - 0.5*d_imo*(vx_imo*vx_imo + vy_imo*vy_imo + vz_imo*vz_imo)) * (gamma - 1.0);
    #endif //PRESSURE_DE
    p_imo  = fmax(p_imo, (Real) TINY_NUMBER);
    #ifdef DE
    ge_imo = dge / d_imo;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_imo[i]  =  dev_conserved[(5+i)*n_cells + id] / d_imo;
    }
    #endif    
    // cell i+1
    if (dir == 0) id = xid+1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid+1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid+1)*nx*ny;
    d_ipo  =  dev_conserved[            id];
    vx_ipo =  dev_conserved[o1*n_cells + id] / d_ipo;
    vy_ipo =  dev_conserved[o2*n_cells + id] / d_ipo;
    vz_ipo =  dev_conserved[o3*n_cells + id] / d_ipo;
    #ifdef DE //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    E_kin = 0.5 * d_ipo * ( vx_ipo*vx_ipo + vy_ipo*vy_ipo + vz_ipo*vz_ipo );
    dge = dev_conserved[(n_fields-1)*n_cells + id];
    p_ipo = Get_Pressure_From_DE( E, E - E_kin, dge, gamma ); 
    #else
    p_ipo  = (dev_conserved[4*n_cells + id] - 0.5*d_ipo*(vx_ipo*vx_ipo + vy_ipo*vy_ipo + vz_ipo*vz_ipo)) * (gamma - 1.0);
    #endif //PRESSURE_DE
    p_ipo  = fmax(p_ipo, (Real) TINY_NUMBER);
    #ifdef DE
    ge_ipo = dge / d_ipo;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_ipo[i]  =  dev_conserved[(5+i)*n_cells + id] / d_ipo;
    }
    #endif    
    // cell i-2
    if (dir == 0) id = xid-2 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-2)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-2)*nx*ny;
    d_imt  =  dev_conserved[            id];
    vx_imt =  dev_conserved[o1*n_cells + id] / d_imt;
    vy_imt =  dev_conserved[o2*n_cells + id] / d_imt;
    vz_imt =  dev_conserved[o3*n_cells + id] / d_imt;
    #ifdef DE //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    E_kin = 0.5 * d_imt * ( vx_imt*vx_imt + vy_imt*vy_imt + vz_imt*vz_imt );
    dge = dev_conserved[(n_fields-1)*n_cells + id];
    p_imt = Get_Pressure_From_DE( E, E - E_kin, dge, gamma ); 
    #else
    p_imt  = (dev_conserved[4*n_cells + id] - 0.5*d_imt*(vx_imt*vx_imt + vy_imt*vy_imt + vz_imt*vz_imt)) * (gamma - 1.0);
    #endif //PRESSURE_DE
    p_imt  = fmax(p_imt, (Real) TINY_NUMBER);
    #ifdef DE
    ge_imt = dge / d_imt;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_imt[i]  =  dev_conserved[(5+i)*n_cells + id] / d_imt;
    }
    #endif    
    // cell i+2
    if (dir == 0) id = xid+2 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid+2)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid+2)*nx*ny;
    d_ipt  =  dev_conserved[            id];
    vx_ipt =  dev_conserved[o1*n_cells + id] / d_ipt;
    vy_ipt =  dev_conserved[o2*n_cells + id] / d_ipt;
    vz_ipt =  dev_conserved[o3*n_cells + id] / d_ipt;
    #ifdef DE //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    E_kin = 0.5 * d_ipt * ( vx_ipt*vx_ipt + vy_ipt*vy_ipt + vz_ipt*vz_ipt );
    dge = dev_conserved[(n_fields-1)*n_cells + id];
    p_ipt = Get_Pressure_From_DE( E, E - E_kin, dge, gamma ); 
    #else
    p_ipt  = (dev_conserved[4*n_cells + id] - 0.5*d_ipt*(vx_ipt*vx_ipt + vy_ipt*vy_ipt + vz_ipt*vz_ipt)) * (gamma - 1.0);
    #endif //PRESSURE_DE
    p_ipt  = fmax(p_ipt, (Real) TINY_NUMBER);
    #ifdef DE
    ge_ipt = dge / d_ipt;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_ipt[i]  =  dev_conserved[(5+i)*n_cells + id] / d_ipt;
    }
    #endif    
    #ifdef FLATTENING
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
    #endif //FLATTENING
  
    //use ppm routines to set cell boundary values (see Fryxell Sec. 3.1.1)

    // Calculate the monotonized slopes for cells imo, i, ipo (density)
    del_q_imo = Calculate_Slope(d_imt, d_imo, d_i);
    del_q_i   = Calculate_Slope(d_imo, d_i,   d_ipo);
    del_q_ipo = Calculate_Slope(d_i,   d_ipo, d_ipt);

    // Calculate the interface values for density
    Interface_Values_PPM(d_imo,  d_i,  d_ipo,  del_q_imo, del_q_i, del_q_ipo, &d_L,  &d_R); 

    // Calculate the monotonized slopes for cells imo, i, ipo (x-velocity)
    del_q_imo = Calculate_Slope(vx_imt, vx_imo, vx_i);
    del_q_i   = Calculate_Slope(vx_imo, vx_i,   vx_ipo);
    del_q_ipo = Calculate_Slope(vx_i,   vx_ipo, vx_ipt);

    // Calculate the interface values for x-velocity
    Interface_Values_PPM(vx_imo, vx_i, vx_ipo, del_q_imo, del_q_i, del_q_ipo, &vx_L, &vx_R); 

    // Calculate the monotonized slopes for cells imo, i, ipo (y-velocity)
    del_q_imo = Calculate_Slope(vy_imt, vy_imo, vy_i);
    del_q_i   = Calculate_Slope(vy_imo, vy_i,   vy_ipo);
    del_q_ipo = Calculate_Slope(vy_i,   vy_ipo, vy_ipt);

    // Calculate the interface values for y-velocity
    Interface_Values_PPM(vy_imo, vy_i, vy_ipo, del_q_imo, del_q_i, del_q_ipo, &vy_L, &vy_R); 

    // Calculate the monotonized slopes for cells imo, i, ipo (z-velocity)
    del_q_imo = Calculate_Slope(vz_imt, vz_imo, vz_i);
    del_q_i   = Calculate_Slope(vz_imo, vz_i,   vz_ipo);
    del_q_ipo = Calculate_Slope(vz_i,   vz_ipo, vz_ipt);

    // Calculate the interface values for z-velocity
    Interface_Values_PPM(vz_imo, vz_i, vz_ipo, del_q_imo, del_q_i, del_q_ipo, &vz_L, &vz_R); 

    // Calculate the monotonized slopes for cells imo, i, ipo (pressure)
    del_q_imo = Calculate_Slope(p_imt, p_imo, p_i);
    del_q_i   = Calculate_Slope(p_imo, p_i,   p_ipo);
    del_q_ipo = Calculate_Slope(p_i,   p_ipo, p_ipt);

    // Calculate the interface values for pressure
    Interface_Values_PPM(p_imo,  p_i,  p_ipo,  del_q_imo, del_q_i, del_q_ipo, &p_L,  &p_R); 

    #ifdef DE
    // Calculate the monotonized slopes for cells imo, i, ipo (internal energy)
    del_q_imo = Calculate_Slope(ge_imt, ge_imo, ge_i);
    del_q_i   = Calculate_Slope(ge_imo, ge_i,   ge_ipo);
    del_q_ipo = Calculate_Slope(ge_i,   ge_ipo, ge_ipt);

    // Calculate the interface values for internal energy
    Interface_Values_PPM(ge_imo,  ge_i,  ge_ipo,  del_q_imo, del_q_i, del_q_ipo, &ge_L,  &ge_R); 
    #endif

    #ifdef SCALAR
    // Calculate the monotonized slopes for cells imo, i, ipo (passive scalars)
    for (int i=0; i<NSCALARS; i++) {
      del_q_imo = Calculate_Slope(scalar_imt[i], scalar_imo[i], scalar_i[i]);
      del_q_i   = Calculate_Slope(scalar_imo[i], scalar_i[i],   scalar_ipo[i]);
      del_q_ipo = Calculate_Slope(scalar_i[i],   scalar_ipo[i], scalar_ipt[i]);

      // Calculate the interface values for the passive scalars 
      Interface_Values_PPM(scalar_imo[i],  scalar_i[i],  scalar_ipo[i],  del_q_imo, del_q_i, del_q_ipo, &scalar_L[i],  &scalar_R[i]); 
    }
    #endif

#ifdef STEEPENING
    Real d2_rho_imo, d2_rho_ipo, eta_i;
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
          del_q_imo = Calculate_Slope(d_imt, d_imo, d_i);
          del_q_ipo = Calculate_Slope(d_i, d_ipo, d_ipt);

          //replace left and right interface values of density (Colella Eqn 1.14, 1.15)
          d_L = d_L*(1-eta_i) + (d_imo + 0.5 * del_q_imo) * eta_i;
          d_R = d_R*(1-eta_i) + (d_ipo - 0.5 * del_q_ipo) * eta_i;
        }
      }
    }
#endif

#ifdef FLATTENING
    Real F_imo, F_i, F_ipo;
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
    d_L  = F_i * d_i  + (1 - F_i) * d_L;
    vx_L = F_i * vx_i + (1 - F_i) * vx_L;
    vy_L = F_i * vy_i + (1 - F_i) * vy_L;
    vz_L = F_i * vz_i + (1 - F_i) * vz_L;
    p_L  = F_i * p_i  + (1 - F_i) * p_L;
    #ifdef DE
    ge_L = F_i * ge_i + (1 - F_i) * ge_L;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_L[i] = F_i * scalar_i[i] + (1 - F_i) * scalar_L[i];
    }
    #endif
    d_R  = F_i * d_i  + (1 - F_i) * d_R;
    vx_R = F_i * vx_i + (1 - F_i) * vx_R;
    vy_R = F_i * vy_i + (1 - F_i) * vy_R;
    vz_R = F_i * vz_i + (1 - F_i) * vz_R;
    p_R  = F_i * p_i  + (1 - F_i) * p_R;
    #ifdef DE
    ge_R = F_i * ge_i + (1 - F_i) * ge_R;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_R[i] = F_i * scalar_i[i] + (1 - F_i) * scalar_R[i];
    }
    #endif
#endif     


#ifdef CTU
    // compute sound speed in cell i
    cs = sqrt(gamma * p_i / d_i);

    // compute a first guess at the left and right states by taking the average
    // under the characteristic on each side that has the largest speed

    // recompute slope across cell for each variable Fryxell Eqn 29
    del_d  = d_R  - d_L; 
    del_vx = vx_R - vx_L;
    del_vy = vy_R - vy_L;
    del_vz = vz_R - vz_L;
    del_p  = p_R  - p_L;
    #ifdef DE
    del_ge = ge_R - ge_L;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      del_scalar[i] = scalar_R[i] - scalar_L[i];
    }
    #endif

    d_6  = 6.0 * (d_i  - 0.5*(d_L  + d_R));  // Fryxell Eqn 30
    vx_6 = 6.0 * (vx_i - 0.5*(vx_L + vx_R)); // Fryxell Eqn 30
    vy_6 = 6.0 * (vy_i - 0.5*(vy_L + vy_R)); // Fryxell Eqn 30
    vz_6 = 6.0 * (vz_i - 0.5*(vz_L + vz_R)); // Fryxell Eqn 30
    p_6  = 6.0 * (p_i  - 0.5*(p_L  + p_R));  // Fryxell Eqn 30
    #ifdef DE
    ge_6 = 6.0 * (ge_i - 0.5*(ge_L + ge_R)); // Fryxell Eqn 30
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_6[i] = 6.0 * (scalar_i[i] - 0.5*(scalar_L[i] + scalar_R[i])); // Fryxell Eqn 30
    }
    #endif

    // set speed of characteristics (v-c, v, v+c) using average values of v and c
    lambda_m = vx_i - cs;
    lambda_0 = vx_i;
    lambda_p = vx_i + cs;

    // calculate betas (for left state guesses)
    beta_m = fmax( (lambda_m * dt / dx) , 0.0 ); // Fryxell Eqn 59
    beta_0 = fmax( (lambda_0 * dt / dx) , 0.0); // Fryxell Eqn 59
    beta_p = fmax( (lambda_p * dt / dx) , 0.0 ); // Fryxell Eqn 59
 
    //calculate alphas (for right state guesses)
    alpha_m = fmax( (-lambda_m * dt / dx), 0.0); // Fryxell Eqn 61
    alpha_0 = fmax( (-lambda_0 * dt / dx), 0.0); // Fryxell Eqn 61
    alpha_p = fmax( (-lambda_p * dt / dx), 0.0); // Fryxell Eqn 61

    // average values under characteristics for left interface (Fryxell Eqn 60)
    dL_m  = d_L  + 0.5 * alpha_m * (del_d  + d_6  * (1 - (2./3.) * alpha_m));
    vxL_m = vx_L + 0.5 * alpha_m * (del_vx + vx_6 * (1 - (2./3.) * alpha_m));
    pL_m  = p_L  + 0.5 * alpha_m * (del_p  + p_6  * (1 - (2./3.) * alpha_m));
    dL_0  = d_L  + 0.5 * alpha_0 * (del_d  + d_6  * (1 - (2./3.) * alpha_0));
    vyL_0 = vy_L + 0.5 * alpha_0 * (del_vy + vy_6 * (1 - (2./3.) * alpha_0));
    vzL_0 = vz_L + 0.5 * alpha_0 * (del_vz + vz_6 * (1 - (2./3.) * alpha_0));
    #ifdef DE
    geL_0 = ge_L + 0.5 * alpha_0 * (del_ge + ge_6 * (1 - (2./3.) * alpha_0));
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalarL_0[i] = scalar_L[i] + 0.5 * alpha_0 * (del_scalar[i] + scalar_6[i] * (1 - (2./3.) * alpha_0));
    }
    #endif
    pL_0  = p_L  + 0.5 * alpha_0 * (del_p  + p_6  * (1 - (2./3.) * alpha_0));
    vxL_p = vx_L + 0.5 * alpha_p * (del_vx + vx_6 * (1 - (2./3.) * alpha_p));
    pL_p  = p_L  + 0.5 * alpha_p * (del_p  + p_6  * (1 - (2./3.) * alpha_p));

    // average values under characteristics for right interface (Fryxell Eqn 58)
    vxR_m = vx_R - 0.5 * beta_m * (del_vx - vx_6 * (1 - (2./3.) * beta_m));
    pR_m  = p_R  - 0.5 * beta_m * (del_p  - p_6  * (1 - (2./3.) * beta_m));
    dR_0  = d_R  - 0.5 * beta_0 * (del_d  - d_6  * (1 - (2./3.) * beta_0));
    vyR_0 = vy_R - 0.5 * beta_0 * (del_vy - vy_6 * (1 - (2./3.) * beta_0));
    vzR_0 = vz_R - 0.5 * beta_0 * (del_vz - vz_6 * (1 - (2./3.) * beta_0));
    #ifdef DE
    geR_0 = ge_R - 0.5 * beta_0 * (del_ge - ge_6 * (1 - (2./3.) * beta_0));
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalarR_0[i] = scalar_R[i] - 0.5 * beta_0 * (del_scalar[i] - scalar_6[i] * (1 - (2./3.) * beta_0));
    }
    #endif
    pR_0  = p_R  - 0.5 * beta_0 * (del_p  - p_6  * (1 - (2./3.) * beta_0));
    dR_p  = d_R  - 0.5 * beta_p * (del_d  - d_6  * (1 - (2./3.) * beta_p));
    vxR_p = vx_R - 0.5 * beta_p * (del_vx - vx_6 * (1 - (2./3.) * beta_p));
    pR_p  = p_R  - 0.5 * beta_p * (del_p  - p_6  * (1 - (2./3.) * beta_p));

    // as a first guess, use characteristics with the largest speeds
    // for transverse velocities, use the 0 characteristic
    // left
    d_L  = dL_m;
    vx_L = vxL_m;
    vy_L = vyL_0;
    vz_L = vzL_0;
    p_L  = pL_m;
    #ifdef DE
    ge_L = geL_0;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_L[i] = scalarL_0[i];
    }
    #endif
    // right
    d_R  = dR_p;
    vx_R = vxR_p;
    vy_R = vyR_0;
    vz_R = vzR_0;
    p_R  = pR_p;
    #ifdef DE
    ge_R = geR_0;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_R[i] = scalarR_0[i];
    }
    #endif

    // correct these initial guesses by taking into account the number of 
    // characteristics on each side of the interface

    // calculate the 'guess' sound speeds 
    cl = sqrt(gamma * p_L / d_L);
    cr = sqrt(gamma * p_R / d_L);

    // calculate the chi values (Fryxell Eqns 62 & 63)
    chi_L_m =  1./(2*d_L*cl) * (vx_L - vxL_m - (p_L - pL_m)/(d_L*cl));
    chi_L_p = -1./(2*d_L*cl) * (vx_L - vxL_p + (p_L - pL_p)/(d_L*cl));
    chi_L_0 = (p_L - pL_0)/(d_L*d_L*cl*cl) + 1./d_L - 1./dL_0;
    chi_R_m =  1./(2*d_R*cr) * (vx_R - vxR_m - (p_R - pR_m)/(d_R*cr));
    chi_R_p = -1./(2*d_R*cr) * (vx_R - vxR_p + (p_R - pR_p)/(d_R*cr));
    chi_R_0 = (p_R - pR_0)/(d_R*d_R*cr*cr) + 1./d_R - 1./dR_0;

    // set chi to 0 if characteristic velocity has the wrong sign (Fryxell Eqn 64)
    if (lambda_m >= 0) { chi_L_m = 0; }
    if (lambda_0 >= 0) { chi_L_0 = 0; }
    if (lambda_p >= 0) { chi_L_p = 0; }
    if (lambda_m <= 0) { chi_R_m = 0; }
    if (lambda_0 <= 0) { chi_R_0 = 0; }
    if (lambda_p <= 0) { chi_R_p = 0; }

    // use the chi values to correct the initial guesses and calculate final input states
    p_L = p_L + (d_L*d_L*cl*cl) * (chi_L_p + chi_L_m);
    vx_L = vx_L + d_L*cl * (chi_L_p - chi_L_m);
    d_L = pow( ((1.0/d_L) - (chi_L_m + chi_L_0 + chi_L_p)) , -1);
    p_R = p_L + (d_R*d_R*cr*cr) * (chi_R_p + chi_R_m);
    vx_R = vx_R + d_R*cr * (chi_R_p - chi_R_m);
    d_R = pow( ((1.0/d_R) - (chi_R_m + chi_R_0 + chi_R_p)) , -1);
#endif //CTU


    // Apply mimimum constraints
    d_L = fmax(d_L, (Real) TINY_NUMBER);
    d_R = fmax(d_R, (Real) TINY_NUMBER);
    p_L = fmax(p_L, (Real) TINY_NUMBER);
    p_R = fmax(p_R, (Real) TINY_NUMBER);

    // Convert the left and right states in the primitive to the conserved variables
    // send final values back from kernel
    // bounds_R refers to the right side of the i-1/2 interface
    if (dir == 0) id = xid-1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-1)*nx*ny;
    dev_bounds_R[            id] = d_L;
    dev_bounds_R[o1*n_cells + id] = d_L*vx_L;
    dev_bounds_R[o2*n_cells + id] = d_L*vy_L;
    dev_bounds_R[o3*n_cells + id] = d_L*vz_L;
    dev_bounds_R[4*n_cells + id] = p_L/(gamma-1.0) + 0.5*d_L*(vx_L*vx_L + vy_L*vy_L + vz_L*vz_L);  
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_R[(5+i)*n_cells + id] = d_L*scalar_L[i];
    }
    #endif    
    #ifdef DE
    dev_bounds_R[(n_fields-1)*n_cells + id] = d_L*ge_L;
    #endif
    // bounds_L refers to the left side of the i+1/2 interface
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_L[            id] = d_R;
    dev_bounds_L[o1*n_cells + id] = d_R*vx_R;
    dev_bounds_L[o2*n_cells + id] = d_R*vy_R;
    dev_bounds_L[o3*n_cells + id] = d_R*vz_R;
    dev_bounds_L[4*n_cells + id] = p_R/(gamma-1.0) + 0.5*d_R*(vx_R*vx_R + vy_R*vy_R + vz_R*vz_R);      
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_L[(5+i)*n_cells + id] = d_R*scalar_R[i];
    }
    #endif    
    #ifdef DE
    dev_bounds_L[(n_fields-1)*n_cells + id] = d_R*ge_R;
    #endif
  
  }
}
    


/*! \fn __device__ Real Calculate_Slope(Real q_imo, Real q_i, Real q_ipo)
 *  \brief Calculates the limited slope across a cell.*/
__device__ Real Calculate_Slope(Real q_imo, Real q_i, Real q_ipo)
{
  Real del_q_L, del_q_R, del_q_C, del_q_G; 
  Real lim_slope_a, lim_slope_b, del_q_m;

  // Compute the left, right, and centered differences of the primative variables
  // Note that here L and R refer to locations relative to the cell center
  
  // left
  del_q_L  = q_i - q_imo;
  // right
  del_q_R  = q_ipo - q_i;
  // centered
  del_q_C  = 0.5*(q_ipo - q_imo);
  // Van Leer
  if (del_q_L*del_q_R > 0.0) { del_q_G = 2.0*del_q_L*del_q_R / (del_q_L+del_q_R); }
  else { del_q_G = 0.0; }


  // Monotonize the differences
  lim_slope_a = fmin(fabs(del_q_L), fabs(del_q_R));
  lim_slope_b = fmin(fabs(del_q_C), fabs(del_q_G));

  // Minmod limiter
  //del_q_m = sgn_CUDA(del_q_C)*fmin(2.0*lim_slope_a, fabs(del_q_C));

  // Van Leer limiter
  del_q_m = sgn_CUDA(del_q_C) * fmin((Real) 2.0*lim_slope_a, lim_slope_b);

  return del_q_m;

}


/*! \fn __device__ void Interface_Values_PPM(Real q_imo, Real q_i, Real q_ipo, Real del_q_imo, Real del_q_i, Real del_q_ipo, Real *q_L, Real *q_R)
 *  \brief Calculates the left and right interface values for a cell using parabolic reconstruction
           in the primitive variables with limited slopes provided. Applies further monotonicity constraints.*/
__device__ void Interface_Values_PPM(Real q_imo, Real q_i, Real q_ipo, Real del_q_imo, Real del_q_i, Real del_q_ipo, Real *q_L, Real *q_R)
{
  // Calculate the left and right interface values using the limited slopes
  *q_L = 0.5*(q_i + q_imo) - (1.0/6.0)*(del_q_i - del_q_imo);
  *q_R = 0.5*(q_ipo + q_i) - (1.0/6.0)*(del_q_ipo - del_q_i);

  // Apply further monotonicity constraints to ensure interface values lie between
  // neighboring cell-centered values

  // local maximum or minimum criterion (Fryxell Eqn 52, Fig 11)
  if ((*q_R - q_i)*(q_i - *q_L) <= 0) *q_L = *q_R = q_i;

  // steep gradient criterion (Fryxell Eqn 53, Fig 12)
  if (6.0*(*q_R - *q_L)*(q_i - 0.5*(*q_L + *q_R)) > (*q_R - *q_L)*(*q_R - *q_L))  *q_L = 3.0*q_i - 2.0*(*q_R);
  if (6.0*(*q_R - *q_L)*(q_i - 0.5*(*q_L + *q_R)) < -(*q_R - *q_L)*(*q_R - *q_L)) *q_R = 3.0*q_i - 2.0*(*q_L);
  
  *q_L  = fmax( fmin(q_i, q_imo), *q_L );
  *q_L  = fmin( fmax(q_i, q_imo), *q_L );
  *q_R  = fmax( fmin(q_i, q_ipo), *q_R );
  *q_R  = fmin( fmax(q_i, q_ipo), *q_R );

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

/*! \file plmp_cuda.cu
 *  \brief Definitions of the piecewise linear reconstruction functions for  
           with limiting in the primative variables. */
#ifdef CUDA

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"plmp_cuda.h"

#ifdef DE //PRESSURE_DE
#include"hydro_cuda.h"
#endif


/*! \fn __global__ void PLMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir, int n_fields)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using plm. */
__global__ void PLMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir, int n_fields)
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

  // declare left and right interface values
  Real d_L, vx_L, vy_L, vz_L, p_L;
  Real d_R, vx_R, vy_R, vz_R, p_R;

  // declare conserved variables;
  Real mx_L, my_L, mz_L, E_L;
  Real mx_R, my_R, mz_R, E_R;

  #ifdef DE
  Real ge_i, ge_imo, ge_ipo, ge_L, ge_R, dge_L, dge_R, E_kin, E, dge;
  #endif
  #ifdef SCALAR
  Real scalar_i[NSCALARS], scalar_imo[NSCALARS], scalar_ipo[NSCALARS];
  Real scalar_L[NSCALARS], scalar_R[NSCALARS], dscalar_L[NSCALARS], dscalar_R[NSCALARS];
  #endif

  #ifndef VL
  Real dtodx = dt/dx;  
  Real dfl, dfr, mxfl, mxfr, myfl, myfr, mzfl, mzfr, Efl, Efr;
  #ifdef DE
  Real gefl, gefr;
  #endif
  #ifdef SCALAR
  Real scalarfl[NSCALARS], scalarfr[NSCALARS];
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
  if (dir == 0) {
    xs = 1; xe = nx-2;
    ys = 0; ye = ny;
    zs = 0; ze = nz;
  }
  if (dir == 1) {
    xs = 0; xe = nx;
    ys = 1; ye = ny-2;
    zs = 0; ze = nz;
  }
  if (dir == 2) {
    xs = 0; xe = nx;
    ys = 0; ye = ny;
    zs = 1; ze = nz-2;
  }


  if (xid >= xs && xid < xe && yid >= ys && yid < ye && zid >= zs && zid < ze)
  {
    // load the 3-cell stencil into registers
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
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_i[i] = dev_conserved[(5+i)*n_cells + id] / d_i;
    }
    #endif
    #ifdef DE
    ge_i = dge / d_i;
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
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_imo[i] = dev_conserved[(5+i)*n_cells + id] / d_imo;
    }
    #endif
    #ifdef DE
    ge_imo = dge / d_imo;
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
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalar_ipo[i] = dev_conserved[(5+i)*n_cells + id] / d_ipo;
    }
    #endif
    #ifdef DE
    ge_ipo =  dge / d_ipo;
    #endif


    // Calculate the interface values for each primitive variable
    Interface_Values_PLM(d_imo,  d_i,  d_ipo,  &d_L,  &d_R); 
    Interface_Values_PLM(vx_imo, vx_i, vx_ipo, &vx_L, &vx_R); 
    Interface_Values_PLM(vy_imo, vy_i, vy_ipo, &vy_L, &vy_R); 
    Interface_Values_PLM(vz_imo, vz_i, vz_ipo, &vz_L, &vz_R); 
    Interface_Values_PLM(p_imo,  p_i,  p_ipo,  &p_L,  &p_R); 
    #ifdef DE
    Interface_Values_PLM(ge_imo,  ge_i,  ge_ipo,  &ge_L,  &ge_R); 
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      Interface_Values_PLM(scalar_imo[i],  scalar_i[i],  scalar_ipo[i],  &scalar_L[i],  &scalar_R[i]); 
    }
    #endif

    // Apply mimimum constraints
    d_L = fmax(d_L, (Real) TINY_NUMBER);
    d_R = fmax(d_R, (Real) TINY_NUMBER);
    p_L = fmax(p_L, (Real) TINY_NUMBER);
    p_R = fmax(p_R, (Real) TINY_NUMBER);

    // calculate the conserved variables at each interface
    mx_L = d_L*vx_L;
    mx_R = d_R*vx_R;
    my_L = d_L*vy_L;
    my_R = d_R*vy_R;
    mz_L = d_L*vz_L; 
    mz_R = d_R*vz_R;
    E_L = p_L/(gamma-1.0) + 0.5*d_L*(vx_L*vx_L + vy_L*vy_L + vz_L*vz_L);
    E_R = p_R/(gamma-1.0) + 0.5*d_R*(vx_R*vx_R + vy_R*vy_R + vz_R*vz_R);
    #ifdef DE
    dge_L = d_L*ge_L;
    dge_R = d_R*ge_R;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dscalar_L[i] = d_L*scalar_L[i];
      dscalar_R[i] = d_R*scalar_R[i];
    }
    #endif


    #ifndef VL 
    // calculate fluxes for each variable
    dfl = mx_L;
    dfr = mx_R;
    mxfl = mx_L*vx_L + p_L;
    mxfr = mx_R*vx_R + p_R;
    myfl = mx_L*vy_L;
    myfr = mx_R*vy_R;
    mzfl = mx_L*vz_L;
    mzfr = mx_R*vz_R;
    Efl = (E_L + p_L) * vx_L;
    Efr = (E_R + p_R) * vx_R;
    #ifdef DE
    gefl = dge_L*vx_L;
    gefr = dge_R*vx_R;
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      scalarfl[i] = dscalar_L[i]*vx_L;
      scalarfr[i] = dscalar_R[i]*vx_R;
    }
    #endif

    // Evolve the boundary extrapolated values half a timestep.
    d_L += 0.5 * (dtodx) * (dfl - dfr);
    d_R += 0.5 * (dtodx) * (dfl - dfr);
    mx_L += 0.5 * (dtodx) * (mxfl - mxfr);
    mx_R += 0.5 * (dtodx) * (mxfl - mxfr);
    my_L += 0.5 * (dtodx) * (myfl - myfr);
    my_R += 0.5 * (dtodx) * (myfl - myfr);
    mz_L += 0.5 * (dtodx) * (mzfl - mzfr);
    mz_R += 0.5 * (dtodx) * (mzfl - mzfr);
    E_L += 0.5 * (dtodx) * (Efl - Efr);
    E_R += 0.5 * (dtodx) * (Efl - Efr);	
    #ifdef DE
    dge_L += 0.5 * (dtodx) * (gefl - gefr);
    dge_R += 0.5 * (dtodx) * (gefl - gefr);
    #endif
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dscalar_L[i] += 0.5 * (dtodx) * (scalarfl[i] - scalarfr[i]);
      dscalar_R[i] += 0.5 * (dtodx) * (scalarfl[i] - scalarfr[i]);
    }
    #endif
    
    #endif //CTU

    // Convert the left and right states in the primitive to the conserved variables
    // send final values back from kernel
    // bounds_R refers to the right side of the i-1/2 interface
    if (dir == 0) id = xid-1 + yid*nx + zid*nx*ny;
    if (dir == 1) id = xid + (yid-1)*nx + zid*nx*ny;
    if (dir == 2) id = xid + yid*nx + (zid-1)*nx*ny;
    dev_bounds_R[            id] = d_L;
    dev_bounds_R[o1*n_cells + id] = mx_L;
    dev_bounds_R[o2*n_cells + id] = my_L;
    dev_bounds_R[o3*n_cells + id] = mz_L;
    dev_bounds_R[4*n_cells + id] = E_L;  
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_R[(5+i)*n_cells + id] = dscalar_L[i];
    }
    #endif
    #ifdef DE
    dev_bounds_R[(n_fields-1)*n_cells + id] = dge_L;
    #endif    
    // bounds_L refers to the left side of the i+1/2 interface
    id = xid + yid*nx + zid*nx*ny;
    dev_bounds_L[            id] = d_R;
    dev_bounds_L[o1*n_cells + id] = mx_R;
    dev_bounds_L[o2*n_cells + id] = my_R;
    dev_bounds_L[o3*n_cells + id] = mz_R;
    dev_bounds_L[4*n_cells + id] = E_R;      
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_bounds_L[(5+i)*n_cells + id] = dscalar_R[i];
    }
    #endif
    #ifdef DE
    dev_bounds_L[(n_fields-1)*n_cells + id] = dge_R;
    #endif    

  }
}
    


__device__ void Interface_Values_PLM(Real q_imo, Real q_i, Real q_ipo, Real *q_L, Real *q_R)
{
  Real del_q_L, del_q_R, del_q_C, del_q_G; 
  Real lim_slope_a, lim_slope_b, del_q_m;  

  // Compute the left, right, centered, and Van Leer differences of the primative variables
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


  // Calculate the left and right interface values using the limited slopes
  *q_L = q_i - 0.5*del_q_m;
  *q_R = q_i + 0.5*del_q_m;

}


#endif //CUDA

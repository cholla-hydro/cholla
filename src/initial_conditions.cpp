/*! \file initial_conditions.cpp
 *  \brief Definitions of initial conditions for different tests.
           Note that the grid is mapped to 1D as i + (x_dim)*j + (x_dim*y_dim)*k.
           Functions are members of the Grid3D class. */


#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "global.h"
#include "grid3D.h"
#include "mpi_routines.h"
#include "io.h"
#include "error_handling.h"
#include <stdio.h>
#include <cmath>


/*! \fn void Set_Initial_Conditions(parameters P)
 *  \brief Set the initial conditions based on info in the parameters structure. */
void Grid3D::Set_Initial_Conditions(parameters P, Real C_cfl) {

  Set_Domain_Properties(P);
  Set_Gammas(P.gamma);

  if (strcmp(P.init, "Constant")==0) {
    Constant(P.rho, P.vx, P.vy, P.vz, P.P);    
  } else if (strcmp(P.init, "Sound_Wave")==0) {
    Sound_Wave(P.rho, P.vx, P.vy, P.vz, P.P, P.A);
  } else if (strcmp(P.init, "Square_Wave")==0) {
    Square_Wave(P.rho, P.vx, P.vy, P.vz, P.P, P.A);    
  } else if (strcmp(P.init, "Riemann")==0) {
    Riemann(P.rho_l, P.v_l, P.P_l, P.rho_r, P.v_r, P.P_r, P.diaph);
  } else if (strcmp(P.init, "Shu_Osher")==0) {
    Shu_Osher();
  } else if (strcmp(P.init, "Blast_1D")==0) {
    Blast_1D();
  } else if (strcmp(P.init, "KH_discontinuous_2D")==0) {
    KH_discontinuous_2D();
  } else if (strcmp(P.init, "KH_res_ind")==0) {
    KH_res_ind();
  } else if (strcmp(P.init, "Rayleigh_Taylor")==0) {
    Rayleigh_Taylor();
  } else if (strcmp(P.init, "Implosion_2D")==0) {
    Implosion_2D();
  } else if (strcmp(P.init, "Gresho")==0) {
    Gresho();
  } else if (strcmp(P.init, "Noh_2D")==0) {
    Noh_2D();
  } else if (strcmp(P.init, "Noh_3D")==0) {
    Noh_3D();    
  } else if (strcmp(P.init, "Disk_2D")==0) {
    Disk_2D();    
  } else if (strcmp(P.init, "Disk_3D")==0) {
    Disk_3D(P);    
  } else if (strcmp(P.init, "Read_Grid")==0) {
    Read_Grid(P);    
  } else {
    chprintf ("ABORT: %s: Unknown initial conditions!\n", P.init);
    chexit(0);
  }

}

/*! \fn void Set_Domain_Properties(struct parameters P)
 *  \brief Set local domain properties */
void Grid3D::Set_Domain_Properties(struct parameters P)
{
#ifndef  MPI_CHOLLA
  H.xbound = P.xmin;
  H.ybound = P.ymin;
  H.zbound = P.zmin;

  /*perform 1-D first*/
  if(H.nx > 1 && H.ny==1 && H.nz==1)
  {
    H.domlen_x =  P.xlen;
    H.domlen_y =  P.ylen / (H.nx - 2*H.n_ghost);
    H.domlen_z =  P.zlen / (H.nx - 2*H.n_ghost);
    H.dx = H.domlen_x / (H.nx - 2*H.n_ghost);
    H.dy = H.domlen_y;
    H.dz = H.domlen_z;
  }

  /*perform 2-D next*/
  if(H.nx > 1 && H.ny>1 && H.nz==1)
  {
    H.domlen_x =  P.xlen;
    H.domlen_y =  P.ylen;
    H.domlen_z =  P.zlen / (H.nx - 2*H.n_ghost);
    H.dx = H.domlen_x / (H.nx - 2*H.n_ghost);
    H.dy = H.domlen_y / (H.ny - 2*H.n_ghost);
    H.dz = H.domlen_z;
  }

  /*perform 3-D last*/
  if(H.nx>1 && H.ny>1 && H.nz>1)
  {
    H.domlen_x = P.xlen;
    H.domlen_y = P.ylen;
    H.domlen_z = P.zlen;
    H.dx = H.domlen_x / (H.nx - 2*H.n_ghost);
    H.dy = H.domlen_y / (H.ny - 2*H.n_ghost);
    H.dz = H.domlen_z / (H.nz - 2*H.n_ghost);
  }

  /*set MPI variables (same as local for non-MPI)*/
  H.xblocal = H.xbound;
  H.yblocal = H.ybound;
  H.zblocal = H.zbound;
  H.xdglobal = H.domlen_x;
  H.ydglobal = H.domlen_y;
  H.zdglobal = H.domlen_z;

#else  /*MPI_CHOLLA*/

  /* set the local domains on each process */
  Set_Parallel_Domain(P.xmin, P.ymin, P.zmin, P.xlen, P.ylen, P.zlen, &H);

#endif /*MPI_CHOLLA*/
}



/*! \fn void Constant(Real rho, Real vx, Real vy, Real vz, Real P)
 *  \brief Constant gas properties. */
void Grid3D::Constant(Real rho, Real vx, Real vy, Real vz, Real P)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;

  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny-H.n_ghost;
  }
  else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // set constant initial states
        C.density[id]    = rho;
        C.momentum_x[id] = rho*vx;
        C.momentum_y[id] = rho*vy;
        C.momentum_z[id] = rho*vz;
        C.Energy[id]     = P/(gama-1.0) + 0.5*rho*(vx*vx + vy*vy + vz*vz);
        #ifdef DE
        C.GasEnergy[id]  = P/(gama-1.0);
        #endif
      }
    }
  }

}


/*! \fn void Sound_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
 *  \brief Sine wave perturbation. */
void Grid3D::Sound_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;

  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny-H.n_ghost;
  }
  else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // set constant initial states
        C.density[id]    = rho;
        C.momentum_x[id] = rho*vx;
        C.momentum_y[id] = rho*vy;
        C.momentum_z[id] = rho*vz;
        C.Energy[id]     = P/(gama-1.0) + 0.5*rho*(vx*vx + vy*vy + vz*vz);
        // add small-amplitude perturbations
        C.density[id]    = C.density[id]    + A * sin(2.0*PI*x_pos);
        C.momentum_x[id] = C.momentum_x[id] + A * sin(2.0*PI*x_pos);
        C.momentum_y[id] = C.momentum_y[id] + A * sin(2.0*PI*x_pos);
        C.momentum_z[id] = C.momentum_z[id] + A * sin(2.0*PI*x_pos);
        C.Energy[id]     = C.Energy[id]     + A * (1.5) * sin(2*PI*x_pos);
      }
    }
  }

}


/*! \fn void Square_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
 *  \brief Square wave density perturbation with amplitude A*rho in pressure equilibrium. */
void Grid3D::Square_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;

  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny-H.n_ghost;
  }
  else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        C.density[id]    = rho;
        //C.momentum_x[id] = 0.0;
        C.momentum_x[id] = rho * vx;
        C.momentum_y[id] = rho * vy;
        C.momentum_z[id] = rho * vz;
        //C.momentum_z[id] = rho_l * v_l;
        C.Energy[id]     = P/(gama-1.0) + 0.5*rho*(vx*vx + vy*vy + vz*vz);
        #ifdef DE
        C.GasEnergy[id]  = P/(gama-1.0);
        #endif
        if (x_pos > 0.25*H.domlen_x && x_pos < 0.75*H.domlen_x)
        {
          C.density[id]    = rho*A;
          C.momentum_x[id] = rho*A * vx;
          C.momentum_y[id] = rho*A * vy;
          C.momentum_z[id] = rho*A * vz;
          C.Energy[id]     = P/(gama-1.0) + 0.5*rho*A*(vx*vx + vy*vy + vz*vz);        
          #ifdef DE
          C.GasEnergy[id]  = P/(gama-1.0);
          #endif
        }
      }
    }
  }
}


/*! \fn void Riemann(Real rho_l, Real v_l, Real P_l, Real rho_r, Real v_r, Real P_r, Real diaph)
 *  \brief Initialize the grid with a Riemann problem. */
void Grid3D::Riemann(Real rho_l, Real v_l, Real P_l, Real rho_r, Real v_r, Real P_r, Real diaph)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;
  Real v, P, cs;

  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny-H.n_ghost;
  }
  else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        if (x_pos < diaph)
        {
          C.density[id]    = rho_l;
          C.momentum_x[id] = rho_l * v_l;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          C.Energy[id]     = P_l/(gama-1.0) + 0.5*rho_l*v_l*v_l;
          #ifdef DE
          C.GasEnergy[id]  = P_l/(gama-1.0);
          #endif
        }
        else
        {
          C.density[id]    = rho_r;
          C.momentum_x[id] = rho_r * v_r;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          C.Energy[id]     = P_r/(gama-1.0) + 0.5*rho_r*v_r*v_r;        
          #ifdef DE
          C.GasEnergy[id]  = P_r/(gama-1.0);
          #endif
        }
      }
    }
  }
}


/*! \fn void Shu_Osher()
 *  \brief Initialize the grid with the Shu-Osher shock tube problem. See Stone 2008, Section 8.1 */
void Grid3D::Shu_Osher()
{
  int i, id;
  Real x_pos, y_pos, z_pos;
  Real vx, P;

  // set initial values of conserved variables
  for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
    id = i;
    // get centered x position
    Get_Position(i, H.n_ghost, H.n_ghost, &x_pos, &y_pos, &z_pos);

    if (x_pos < -0.8)
    {
      C.density[id] = 3.857143;
      vx = 2.629369;
      C.momentum_x[id] = C.density[id]*vx;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      P = 10.33333;
      C.Energy[id] = P/(gama-1.0) + 0.5*C.density[id]*vx*vx;
    }
    else
    {
      C.density[id] = 1.0 + 0.2*sin(5.0*PI*x_pos);
      Real vx = 0.0;
      C.momentum_x[id] = C.density[id]*vx;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      Real P = 1.0;
      C.Energy[id] = P/(gama-1.0) + 0.5*C.density[id]*vx*vx;
    }
  }
}


/*! \fn void Blast_1D()
 *  \brief Initialize the grid with two interacting blast waves. See Stone 2008, Section 8.1.*/
void Grid3D::Blast_1D()
{
  int i, id;
  Real x_pos, y_pos, z_pos;
  Real vx, P;

  // set initial values of conserved variables
  for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
    id = i;
    // get the centered x position
    Get_Position(i, H.n_ghost, H.n_ghost, &x_pos, &y_pos, &z_pos);

    if (x_pos < 0.1)
    {
      C.density[id] = 1.0;
      C.momentum_x[id] = 0.0;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      P = 1000.0;
      C.Energy[id] = P/(gama-1.0);
    }
    else if (x_pos > 0.9)
    {
      C.density[id] = 1.0;
      C.momentum_x[id] = 0.0;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      P = 100;
      C.Energy[id] = P/(gama-1.0);
    }        
    else
    {
      C.density[id] = 1.0;
      C.momentum_x[id] = 0.0;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      P = 0.01;
      C.Energy[id] = P/(gama-1.0);
    }
  }
}


/*! \fn void KH_discontinuous_2D()
 *  \brief Initialize the grid with a 2D Kelvin-Helmholtz instability. 
           This version of KH test has a discontinuous boundary.
           Use KH_res_ind_2D for a version that is resolution independent. */
void Grid3D::KH_discontinuous_2D()
{
  int i, j, k, id;
  int istart, iend, jstart, jend, kstart, kend;
  Real x_pos, y_pos, z_pos;
  Real vx, vy, vz;

  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  jstart = H.n_ghost;
  jend   = H.ny-H.n_ghost;
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  // set the initial values of the conserved variables
  for (k=kstart; k<kend; k++) {
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;
        // get the centered x and y positions
        Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

        // outer thirds of slab
        if (y_pos <= 1.0*H.ydglobal/3.0) 
        {
          C.density[id] = 1.0;
          C.momentum_x[id] = 0.5 + 0.01*sin(2*PI*x_pos);
          C.momentum_y[id] = 0.0 + 0.01*sin(2*PI*x_pos);
          C.momentum_z[id] = 0.0;
          C.Energy[id] = 2.5/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
        }
        else if (y_pos >= 2.0*H.ydglobal/3.0)
        {
          C.density[id] = 1.0;
          C.momentum_x[id] = 0.5 + 0.01*sin(2*PI*x_pos);
          C.momentum_y[id] = 0.0 + 0.01*sin(2*PI*x_pos);
          C.momentum_z[id] = 0.0;
          C.Energy[id] = 2.5/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
        }
        // inner third of slab
        else
        {
          C.density[id] = 2.0;
          C.momentum_x[id] = -1.0 + 0.02*sin(2*PI*x_pos);
          C.momentum_y[id] = 0.0  + 0.02*sin(2*PI*x_pos);
          C.momentum_z[id] = 0.0;
          C.Energy[id] = 2.5/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
        }
      }
    }
  }

}


/*! \fn void KH_res_ind()
 *  \brief Initialize the grid with a Kelvin-Helmholtz instability whose modes are resolution independent. */
void Grid3D::KH_res_ind()
{
  int i, j, k, id;
  int istart, iend, jstart, jend, kstart, kend;
  Real x_pos, y_pos, z_pos;
  Real mx, my, mz;
  Real r, yc, zc, phi;
  Real d1, d2, v1, v2, P, dy, A;

  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  jstart = H.n_ghost;
  jend   = H.ny-H.n_ghost;
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  // y, z center of cylinder (assuming x is long direction)
  yc = 0.0;
  zc = 0.0;

  d1 = 100.0; // inner density
  d2 = 1.0; // outer density
  v1 = 0.5; // inner velocity
  v2 = -0.5; // outer velocity
  P = 2.5; // pressure
  dy = 0.05; // width of ramp function (see Robertson 2009)
  A = 0.1; // amplitude of the perturbation

  // set the initial values of the conserved variables
  for (k=kstart; k<kend; k++) {
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;
        // get the centered x and y positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);


        // inner fluid
        if (fabs(y_pos-0.5) < 0.25)
        {
          if (y_pos > 0.5)
          {
            C.density[id] = d1 - (d1-d2)*exp( -0.5*pow(y_pos-0.75 - sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
            C.momentum_x[id] = v1*C.density[id] - C.density[id] * (v1-v2) * exp( -0.5*pow(y_pos-0.75 - sqrt(-2.0*dy*dy*log(0.5)),2) /(dy*dy) );
            C.momentum_y[id] = C.density[id] * A*sin(4*PI*x_pos) * exp( -0.5*pow(y_pos-0.75 - sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) ) ;
          }
          else
          {
            C.density[id] = d1 - (d1-d2)*exp( -0.5*pow(y_pos-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
            C.momentum_x[id] = v1*C.density[id] - C.density[id] * (v1 - v2) * exp( -0.5*pow(y_pos-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2) /(dy*dy) );
            C.momentum_y[id] = C.density[id] * A*sin(4*PI*x_pos) * exp( -0.5*pow(y_pos-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) ); 
          }
        }
        // outer fluid
        else
        {
          if (y_pos > 0.5)
          {
            C.density[id] = d2 + (d1-d2)*exp( -0.5*pow(y_pos-0.75 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
            C.momentum_x[id] = v2*C.density[id] + C.density[id] * (v1 - v2) * exp( -0.5*pow(y_pos-0.75 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
            C.momentum_y[id] = C.density[id] * A*sin(4*PI*x_pos) * exp( -0.5*pow(y_pos-0.75 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
          }
          else
          {
            C.density[id] = d2 + (d1-d2)*exp( -0.5*pow(y_pos-0.25 - sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
            C.momentum_x[id] = v2*C.density[id] + C.density[id] * (v1 - v2) * exp( -0.5*pow(y_pos-0.25 - sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
            C.momentum_y[id] = C.density[id] * A*sin(4*PI*x_pos) * exp( -0.5*pow(y_pos-0.25 - sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
          }

        }
        //C.momentum_y[id] = C.density[id] * A*sin(4*PI*x_pos);
        C.momentum_z[id] = 0.0;
        mx = C.momentum_x[id];
        my = C.momentum_y[id];
        mz = C.momentum_z[id];
        C.Energy[id] = P/(gama-1.0) + 0.5*(mx*mx + my*my + mz*mz)/C.density[id];        

/*
        // cylindrical version (3D only)
        r = sqrt((z_pos-zc)*(z_pos-zc) + (y_pos-yc)*(y_pos-yc)); // center the cylinder at yc, zc
        phi = atan2((z_pos-zc), (y_pos-yc));

        if (r < 0.25) // inside the cylinder
        {
          C.density[id] = d1 - (d1-d2)*exp( -0.5*pow(r-0.25 - sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
          C.momentum_x[id] = v1*C.density[id] - C.density[id] * exp( -0.5*pow(r-0.25 - sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
          C.momentum_y[id] = cos(phi) * C.density[id] * A*sin(4*PI*x_pos) * exp( -0.5*pow(r-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
          C.momentum_z[id] = sin(phi) * C.density[id] * A*sin(4*PI*x_pos) * exp( -0.5*pow(r-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
          mx = C.momentum_x[id];
          my = C.momentum_y[id];
          mz = C.momentum_z[id];
          C.Energy[id] = P/(gama-1.0) + 0.5*(mx*mx + my*my + mz*mz)/C.density[id];
        }
        else // outside the cylinder
        {
          C.density[id] = d2 + (d1-d2)*exp( -0.5*pow(r-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
          C.momentum_x[id] = v2*C.density[id] + C.density[id] * exp( -0.5*pow(r-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) );
          C.momentum_y[id] = cos(phi) * C.density[id] * A*sin(4*PI*x_pos) * (1.0 - exp( -0.5*pow(r-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) ));
          C.momentum_z[id] = sin(phi) * C.density[id] * A*sin(4*PI*x_pos) * (1.0 - exp( -0.5*pow(r-0.25 + sqrt(-2.0*dy*dy*log(0.5)),2)/(dy*dy) ));
          mx = C.momentum_x[id];
          my = C.momentum_y[id];
          mz = C.momentum_z[id];
          C.Energy[id] = P/(gama-1.0) + 0.5*(mx*mx + my*my + mz*mz)/C.density[id];
        }
*/      
      }
    }
  }


}



/*! \fn void Rayleigh_Taylor()
 *  \brief Initialize the grid with a 2D Rayleigh-Taylor instability. */
void Grid3D::Rayleigh_Taylor()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real dl, du, vy, g, P, P_0;
  dl = 1.0;
  du = 2.0;
  g = -0.1;

  // set the initial values of the conserved variables
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      // set the y velocities (small perturbation tapering off from center)
      vy = 0.01*cos(6*PI*x_pos+PI)*exp(-(y_pos-0.5*H.ydglobal)*(y_pos-0.5*H.ydglobal)/0.1);
      //vy = 0.0;

      // lower half of slab
      if (y_pos <= 0.5*H.ydglobal) 
      {
        P_0 = 1.0/gama - dl*g*0.5;
        P = P_0 + dl*g*y_pos;
        C.density[id] = dl;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = dl*vy;
        C.momentum_z[id] = 0.0;
        C.Energy[id] = P/(gama-1.0) + 0.5*(C.momentum_y[id]*C.momentum_y[id])/C.density[id];
      }
      // upper half of slab
      else
      {
        P_0 = 1.0/gama - du*g*0.5;
        P = P_0 + du*g*y_pos;
        C.density[id] = du;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = du*vy;
        C.momentum_z[id] = 0.0;
        C.Energy[id] = P/(gama-1.0) + 0.5*(C.momentum_y[id]*C.momentum_y[id])/C.density[id];
      }
    }
  }

}


/*! \fn void Gresho()
 *  \brief Initialize the grid with the 2D Gresho problem described in LW03. */
void Grid3D::Gresho()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos, xc, yc, r, phi;
  Real d, vx, vy, P, v_boost;
  Real x, y, dx, dy;
  int ran, N;
  N = 100000;
  d = 1.0;
  v_boost = 0.0;

  // center the vortex at (0.0,0.0)
  xc = 0.0;
  yc = 0.0;

  // seed the random number generator
  srand(0);

  // set the initial values of the conserved variables
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);
      
      // calculate centered radial position and phi
      r = sqrt((x_pos-xc)*(x_pos-xc) + (y_pos-yc)*(y_pos-yc));
      phi = atan2((y_pos-yc), (x_pos-xc));

/*
      // set vx, vy, P to zero before integrating 
      vx = 0.0;
      vy = 0.0;
      P = 0.0;

      // monte carlo sample to get an integrated value for vx, vy, P
      for (int ii = 0; ii<N; ii++) {
        // get a random dx and dy to sample within the cell
        ran = rand() % 1000;
        dx = H.dx*(ran/1000.0 - 0.5); 
        ran = rand() % 1000;
        dy = H.dy*(ran/1000.0 - 0.5); 
        x = x_pos + dx;
        y = y_pos + dy;
        // calculate r and phi using the new x & y positions
        r = sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc));
        phi = atan2((y-yc), (x-xc));
        if (r < 0.2) {
          vx += -sin(phi)*5.0*r + v_boost;
          vy += cos(phi)*5.0*r;
          P += 5.0 + 0.5*25.0*r*r;
        }
        else if (r >= 0.2 && r < 0.4) {
          vx += -sin(phi)*(2.0-5.0*r) + v_boost;
          vy += cos(phi)*(2.0-5.0*r);
          P += 9.0 - 4.0*log(0.2) + 0.5*25.0*r*r - 20.0*r + 4.0*log(r);
        }
        else {
          vx += 0.0;
          vy += 0.0;
          P += 3.0 + 4.0*log(2.0);
        }
      }
      vx = vx/N;
      vy = vy/N;
      P = P/N;
*/
      if (r < 0.2) {
        vx = -sin(phi)*5.0*r + v_boost;
        vy = cos(phi)*5.0*r;
        P = 5.0 + 0.5*25.0*r*r;
      }
      else if (r >= 0.2 && r < 0.4) {
        vx = -sin(phi)*(2.0-5.0*r) + v_boost;
        vy = cos(phi)*(2.0-5.0*r);
        P = 9.0 - 4.0*log(0.2) + 0.5*25.0*r*r - 20.0*r + 4.0*log(r);
      }
      else {
        vx = 0.0;
        vy = 0.0;
        P = 3.0 + 4.0*log(2.0);
      }
      // set P constant for modified Gresho problem
      //P = 5.5;

      // set values of conserved variables   
      C.density[id] = d;
      C.momentum_x[id] = d*vx;
      C.momentum_y[id] = d*vy;
      C.momentum_z[id] = 0.0;
      C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy);
      //r = sqrt((x_pos-xc)*(x_pos-xc) + (y_pos-yc)*(y_pos-yc));
      //printf("%f %f %f %f %f\n", x_pos, y_pos, r, vx, vy);
    }
  }
        

}



/*! \fn void Implosion_2D()
 *  \brief Implosion test described in Liska, 2003. */
void Grid3D::Implosion_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real P;


  // set the initial values of the conserved variables
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      // inner corner of box
      if (y_pos < (0.1500001 - x_pos)) {
        C.density[id] = 0.125;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        P = 0.14;
        C.Energy[id] = P/(gama-1.0);
        #ifdef DE
        C.GasEnergy[id] = P/(gama-1.0);
        #endif
      }
      // everywhere else
      else {
        C.density[id] = 1.0;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        P = 1.0;
        C.Energy[id] = P/(gama-1.0);
        #ifdef DE
        C.GasEnergy[id] = P/(gama-1.0);
        #endif
      }
    }
  }

}


/*! \fn void Noh_2D()
 *  \brief Noh test described in Liska, 2003. */
void Grid3D::Noh_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real vx, vy, P, r;


  // set the initial values of the conserved variables
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i + j*H.nx;
      // get the centered x and y positions at (x,y,z)
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      C.density[id] = 1.0;
      r = sqrt(x_pos*x_pos + y_pos*y_pos);
      vx = x_pos / r;
      vy = y_pos / r;
      C.momentum_x[id] = - x_pos / r;
      C.momentum_y[id] = - y_pos / r;
      C.momentum_z[id] = 0.0;
      C.Energy[id] = 1.0e-6/(gama-1.0) + 0.5;
    }
  }

}



/*! \fn void Noh_3D()
 *  \brief Noh test described in Stone, 2008. */
void Grid3D::Noh_3D()
{
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r;


  // set the initial values of the conserved variables
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        C.density[id] = 1.0;
        r = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);
        C.momentum_x[id] = - x_pos / r;
        C.momentum_y[id] = - y_pos / r;
        C.momentum_z[id] = - z_pos / r;
        C.Energy[id] = 1.0e-6/(gama-1.0) + 0.5;
      }
    }
  }



}



/*! \fn void Disk_2D()
 *  \brief Initialize the grid with a 2D disk following a Kuzmin profile. */
void Grid3D::Disk_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos, r, phi;
  Real d, n, a, a_d, a_h, v, vx, vy, P, T_d, x;
  Real M_vir, M_h, M_d, c_vir, R_vir, R_h, R_d, Sigma;

  M_vir = 1.0e12; // viral mass of MW in M_sun
  M_d = 6.5e10; // mass of disk in M_sun
  M_h = M_vir - M_d; // halo mass in M_sun
  R_vir = 261; // viral radius in kpc
  c_vir = 20; // halo concentration
  R_h = R_vir / c_vir; // halo scale length in kpc
  R_d = 3.5; // disk scale length in kpc
  T_d = 10000; // disk temperature, 10^4K


  // set the initial values of the conserved variables
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);
      
      // calculate centered radial position and phi
      r = sqrt(x_pos*x_pos + y_pos*y_pos);
      phi = atan2(y_pos, x_pos);

      // Disk surface density [M_sun / kpc^2]
      // Assume gas surface density is exponential with scale length 2*R_d and
      // mass 0.25*M_d
      Sigma = 0.25*M_d * exp(-r/(2*R_d)) / (8*PI*R_d*R_d) ;
      d = Sigma; // just use sigma for mass density since height is arbitrary
      n = d * DENSITY_UNIT / MP; // number density, cgs
      P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

      // radial acceleration due to Kuzmin disk + NFW halo
      x = r / R_h;
      a_d = GN * M_d * r * pow(r*r + R_d*R_d, -1.5);
      a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
      a = a_d + a_h;

      // circular velocity 
      v = sqrt(r*a);
      vx = -sin(phi)*v;
      vy = cos(phi)*v;

      // set values of conserved variables   
      C.density[id] = d;
      C.momentum_x[id] = d*vx;
      C.momentum_y[id] = d*vy;
      C.momentum_z[id] = 0.0;
      C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy);
      //printf("%e %e %f %f %f %f %f\n", x_pos, y_pos, d, Sigma, vx, vy, P);
    }
  }
        

}


//functions for Disk_3d

//disk radial surface density profile
Real Sigma_disk_D3D(Real r, Real *hdp)
{
  //return the exponential surface density
  Real Sigma_0 = hdp[9];
  Real R_g     = hdp[10];
  return Sigma_0 * exp(-r/R_g);
}

//vertical acceleration in miyamoto nagai
Real gz_disk_D3D(Real R, Real z, Real *hdp)
{
  Real M_d = hdp[1]; //disk mass
  Real R_d = hdp[6]; //MN disk length
  Real Z_d = hdp[7]; //MN disk height
  Real a = R_d;
  Real b = Z_d;
  Real A = sqrt(b*b + z*z);
  Real B = a + A;
  Real C = pow(B*B + R*R, 1.5);

  //checked with wolfram alpha
  return -GN*M_d*z*B/(A*C);
}


//radial acceleration in miyamoto nagai
Real gr_disk_D3D(Real R, Real z, Real *hdp)
{
  Real M_d = hdp[1]; //disk mass 
  Real R_d = hdp[6]; //MN disk length
  Real Z_d = hdp[7]; //MN disk height
  Real A = sqrt(Z_d*Z_d + z*z);
  Real B = R_d + A;
  Real C = pow(B*B + R*R, 1.5);

  //checked with wolfram alpha
  return -GN*M_d*R/C;
}

// function with logarithms used in NFW definitions
Real log_func(Real y)
{
  return log(1+y) - y/(1+y);
}

//vertical acceleration in NFW halo
Real gz_halo_D3D(Real R, Real z, Real *hdp)
{
  Real M_h = hdp[2]; //halo mass
  Real R_h = hdp[5]; //halo scale length
  Real c_vir = hdp[4]; //halo concentration parameter
  Real r = sqrt(R*R + z*z); //spherical radius
  Real x = r / R_h;
  Real z_comp = z/r;

  Real A = log_func(x);
  Real B = 1.0 / (r*r);
  Real C = GN*M_h/log_func(c_vir);

  //checked with wolfram alpha
  return -C*A*B*z_comp;
}


//radial acceleration in NFW halo
Real gr_halo_D3D(Real R, Real z, Real *hdp)
{
  Real M_h = hdp[2]; //halo mass
  Real R_h = hdp[5]; //halo scale length
  Real c_vir = hdp[4]; //halo concentration parameter
  Real r = sqrt(R*R + z*z); //spherical radius
  Real x = r / R_h;
  Real r_comp = R/r;

  Real A = log_func(x);
  Real B = 1.0 / (r*r);
  Real C = GN*M_h/log_func(c_vir);

  //checked with wolfram alpha
  return -C*A*B*r_comp;
}




//exponential integral
Real edpi_D3D(Real z_min, Real z_max, Real *hdp)
{
  //int_a^b exp(-z/h) dz = h*(exp(-a/h) - exp(-b/h))
  Real h = hdp[11]; //initial guess for scale height
  return h*( exp(-1.*z_min/h) - exp(-1.*z_max/h) );
}


//returns the cell-centered vertical
//location of the cell with index k
//k is indexed at 0 at the lowest ghost cell
Real z_hc_D3D(int k, Real dz, int nz, int ng)
{
  //checked that this works, such that the
  //if dz = L_z/nz for the real domain, then the z positions
  //are set correctly for cell centers with nz spanning
  //the real domain, and nz + 2*ng spanning the real + ghost domains
  if(!(nz%2))
  {
    //even # of cells
    return 0.5*dz + ((Real) (k-ng-nz/2))*dz;
  }else{
    //odd # of cells
    return ((Real) (k-ng-(nz-1)/2))*dz;
  }
}



//returns an array containing a 
//column with the hydrostatic density profile
//at locations (x,y,*)
void hydrostatic_column_D3D(Real *rho, Real r, Real *hdp, Real dz, int nz, int ng)
{
  //x is cell center in x direction
  //y is cell center in y direction
  //dz is cell width in z direction
  //nz is number of real cells
  //ng is number of ghost cells
  //total number of cells in column is nz * 2*ng
  //hdp[0] = M_vir; 
  //hdp[1] = M_d; 
  //hdp[2] = M_h; 
  //hdp[3] = R_vir; 
  //hdp[4] = c_vir; 
  //hdp[5] = R_s; 
  //hdp[6] = R_d; 
  //hdp[7] = z_d; 
  //hdp[8] = T_d; 
  //hdp[9] = Sigma_0; 
  //hdp[10] = R_g;
  //hdp[11] = H_g;
  //hdp[12] = K_eos;
  //hdp[13] = gamma;
  //hdp[14] = rho_floor;

  int k;        //index along z axis
  //Real *rho;  //density array in column
  Real *gz;   //vertical acceleration in column
  Real drho;  //change to density

  Real z;     //cell center in z direction
  int nzt;      //total number of cells in z-direction
  Real Sigma; //surface density in column
  Real Sigma_n;
  Real z_min; //bottom of the cell
  Real z_max; //top of the cell
  Real Sigma_r; //surface density expected at r
  Real K = hdp[12]; //K coefficient in EOS; P = K \rho^gamma
  Real gamma = hdp[13];
  Real rho_floor = hdp[14]; //density floor

  int iter = 0; //number if iterations
  int ks; //start of integrals above disk plane
  if(nz%2)
  {
    ks = ng+(nz-1)/2;
  }else{
    ks = ng + nz/2;
  }

  //printf("In hydrostatic_column.\n");

  // get the disk surface density
  // have verified that at this point, Sigma_r is correct
  Sigma_r = Sigma_disk_D3D(r, hdp);

  //set the z-column size, including ghost cells
  nzt = nz + 2*ng;

  //allocate vertical acceleration 
  if(!(gz=(Real *) calloc(nzt,sizeof(Real))))
  {
    printf("Error allocating gz array of size %d.\n",nzt);
    printf("Aborting...\n");
    fflush(stdout);
    exit(-1);
  }

  //compute vertical and radial
  //gravitational accelerations
  for(k=0;k<nzt;k++)
  {
    z     = z_hc_D3D(k,dz,nz,ng);
    gz[k] = gz_disk_D3D(r,z,hdp) + gz_halo_D3D(r,z,hdp);
  }

  //set initial guess for disk properties
  //assume the disk is an exponential vertically to start
  Sigma = 0;
  for(k=0;k<nzt;k++)
  {
    z_min  = z_hc_D3D(k,dz,nz,ng) - 0.5*dz;
    z_max  = z_hc_D3D(k,dz,nz,ng) + 0.5*dz;
    if(z_max>0)
    {
      if(z_min<0)
      {
        //in disk plane centered at z=0
        rho[k] = 2.*edpi_D3D(0, z_max, hdp);
      }else{
        //above disk plane
        rho[k] = edpi_D3D(z_min, z_max, hdp);        
      }
    }else{

      //below disk plane
      rho[k] = edpi_D3D(fabs(z_max), fabs(z_min), hdp);
    }
    Sigma += rho[k];
  }

  //renormalize density to match surface density
  for(k=0;k<nzt;k++)
  {
    rho[k] *= Sigma_r/(Sigma*dz);
  }
  //verified this works as intended until here

  //OK, rho is set initially to an exponential
  //let's adjust to make it hydrostatic

  //begin iterative process to set the density
  int flag = 1;
  Real rho_new;
  Real mass_loss;
  while(flag)
  {
    z = z_hc_D3D(k,dz,nz,ng);

    //adjust density (with zeros on first iteration)
    for(k=ks;k<nzt-1;k++)
    {
      //z position
      z     = z_hc_D3D(k,dz,nz,ng);

      //change in density between vertically adjacent
      //cells that would result in hydrostatic balance
      drho = -1.*pow(rho[k],2-gamma)*(fabs(gz[k])*dz/(gamma*K));

      //prevent driving the density to zero
      if(drho<-0.95*rho[k])
        drho = -0.95*rho[k];

      //set the new density immediately above
      //this cell
      rho_new = rho[k]+drho;

      //track any mass we might lose
      //mass_loss += (rho[k+1]-rho_new);

      //set the revised density in the cell
      //immediately above the current one
      rho[k+1] = rho_new;
      //if(rho[k+1]<rho_floor)
        //rho[k+1] = rho_floor;
    }

    //set the upper most ghost cell
    rho[nzt-1] = rho[nzt-2];
    //mass_loss -= rho[nzt-1];

    //check for mass conservation
    Sigma = 0;
    for(k=0;k<nzt;k++)
    {
      Sigma  += rho[k]*dz;
    }
    mass_loss = Sigma_r - Sigma;

    //printf("mass_loss = %e\n",mass_loss*dz/Sigma);
    int km;
    for(k=ks;k<nzt;k++)
    {
      //spread the lost mass over all the cells
      //if (r < 7.046858)
      if(mass_loss<0)
      {
        rho[k] -= mass_loss/((float) (nzt-1-ks+1));
      }else{
        rho[k] += mass_loss/((float) (nzt-1-ks+1));
      }
      /*
      else{
      if(mass_loss<0)
      {
        rho[k] += mass_loss/((float) (nzt-1-ks+1));
      }else{
        rho[k] -= mass_loss/((float) (nzt-1-ks+1));
      }
      }
      */
      //if(rho[k]<rho_floor)
        //rho[k] = rho_floor;

      //mirror densities
      //above and below disk plane
      if(nz%2)
      {
        km = (ng+(nz-1)/2) - (k-ks);
      }else{
        km = ng + nz/2 - (k-ks) -1;
      }
      rho[km] = rho[k];
    }

    //printf("*****\n");
    iter++;

    //stop once we've converged to 0.1% 
    //of the expected surface density
    //in this column
    if(fabs(mass_loss*dz)/Sigma_r<1.0e-3)
      flag=0;

    if(iter>10000)
    //if (r < 7.046858)
      printf("Iter = %d, r = %f, mass_loss = %e, mass_loss*dz = %e, Sigma = %e, Sigma_r = %e\n", iter, r, mass_loss, fabs(mass_loss*dz), Sigma, Sigma_r);
      //printf("Error converging with iter = %d, mass_loss = %e\n",iter,mass_loss);
  }

  //free ancillary arrays
  free(gz);

}

/*! \fn void Disk_3D(parameters P)
 *  \brief Initialize the grid with a 3D disk. */
void Grid3D::Disk_3D(parameters p)
{
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r, phi;
  Real d, n, a, a_d, a_h, v, vx, vy, vz, P, T_d, T_h, x;
  Real M_vir, M_h, M_d, c_vir, R_vir, R_s, R_d, z_d, Sigma;
  Real K_eos, rho_eos, cs, K_eos_h, rho_eos_h, cs_h;
  Real Sigma_0, R_g, H_g;
  Real rho_floor;

  M_vir = 1.0e12; // viral mass of MW in M_sun
  M_d = 6.5e10; // mass of disk in M_sun (assume all gas)
  M_h = M_vir - M_d; // halo mass in M_sun
  R_vir = 261; // viral radius in kpc
  c_vir = 20; // halo concentration
  R_s = R_vir / c_vir; // halo scale length in kpc
  R_d = 3.5; // disk scale length in kpc
  z_d = 3.5/5.0; // disk scale height in kpc
  T_d = 1.0e4; // disk temperature, at normalized density rho_eos
  T_h = 1.0e6; // halo temperature, at density floor 
  rho_eos = 1.0e7; //gas eos normalized at 1e7 Msun/kpc^3
  rho_eos_h = 3.0e4; //gas eos normalized at 3e4 Msun/kpc^3 (about n_h = 10^-2.5)
  R_g = 2.0*R_d;   //gas scale length in kpc
  Sigma_0 = 0.25*M_d/(2*M_PI*R_g*R_g); //central surface density in Msun/kpc^2
  H_g = z_d; //initial guess for gas scale height
  rho_floor = 1.0e3; //ICs minimum density in Msun/kpc^3


  //EOS info
  cs = sqrt(KB*T_d/(0.6*MP))*TIME_UNIT/LENGTH_UNIT; //sound speed in kpc/kyr
  cs_h = sqrt(KB*T_h/(0.6*MP))*TIME_UNIT/LENGTH_UNIT; //sound speed in kpc/kyr
  K_eos = cs*cs*pow(rho_eos,1.0-p.gamma)/p.gamma; //P = K\rho^gamma
  K_eos_h = cs_h*cs_h*pow(rho_eos_h,1.0-gama)/gama;

  int nhdp = 15;  //number of parameters to pass hydrostatic column
  Real *hdp = (Real *) calloc(nhdp,sizeof(Real));  //parameters
  
  hdp[0] = M_vir; 
  hdp[1] = M_d; 
  hdp[2] = M_h; 
  hdp[3] = R_vir; 
  hdp[4] = c_vir; 
  hdp[5] = R_s; 
  hdp[6] = R_d; 
  hdp[7] = z_d; 
  hdp[8] = T_d; 
  hdp[9] = Sigma_0; 
  hdp[10] = R_g;
  hdp[11] = H_g;
  hdp[12] = K_eos;
  hdp[13] = p.gamma;
  //hdp[14] = rho_floor;
  hdp[14] = 0.0;
  //for(k=0;k<nhdp;k++)
  //  printf("k %d hdp %e procID %d\n",k,hdp[k],procID);
  //printf("cs %e procID %d\n",cs,procID);
  //MPI_Finalize();
  //exit(0);

  //printf("proIC %d Parameter nz in Disk_3D = %d\n",procID,p.nz);

  //printf("procID %d is in Disk_3D\n",procID);
  //printf("procID %d nx %d ny %d nz %d n_ghost %d\n",procID,H.nx,H.ny,H.nz,H.n_ghost);
  //printf("procID %d H.dz %e\n",procID, H.dz);
  //printf("procID %d zmin_global %e zmax_global %e\n",procID, p.zmin,p.zlen);

  //Now we can start the density calculation
  //we will loop over each column and compute
  //the density distribution
  int nz  = p.nz;
  int nzt = 2*H.n_ghost + nz;
  Real dz = p.zlen / ((Real) nz);
  Real *rho = (Real *) calloc(nzt,sizeof(Real));
  Real dPdx, dPdy, dPdr;


  // uses cylindrical coordinates (r, phi, z)
  // r, phi will be converted to x, y
  // hydrostatic column for the disk 
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      // get the centered x, y, and z positions
      k = H.n_ghost + H.ny;
      Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

      //cylindrical radius
      r = sqrt(x_pos*x_pos + y_pos*y_pos);

      hydrostatic_column_D3D(rho, r, hdp, dz, nz, H.n_ghost);

      //store densities
      for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
        id = i + j*H.nx + k*H.nx*H.ny;

        //get density from hydrostatic column computation
        d = rho[nz_local_start + H.n_ghost + (k-H.n_ghost)];
        //d = rho[nz_local_start + (k-H.n_ghost)];
        //if(d<rho_floor) d = rho_floor;

        // set pressure adiabatically
        P = K_eos*pow(d,p.gamma);

        // store density in density
        C.density[id]    = d;

        // store internal energy in Energy array
        C.Energy[id] = P/(gama-1.0);
      }
    }
  }

  //free density profile
  free(rho);
  free(hdp);

//  printf("procID %d finished with hydrostatic column computation.\n",procID);
  // add in hot halo
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      
        id = i + j*H.nx + k*H.nx*H.ny;
        
        // add density floor of 1e3,
        // adjust the internal energy of 
        // all the gas such that the halo will be hot
        C.density[id] += rho_floor; //brant added
        C.Energy[id] += rho_floor*KB*T_h / (0.6*MP*(gama-1.0)) * DENSITY_UNIT / ENERGY_UNIT; 

      }
    }
  }  

  int idm, idp;
  Real xpm, xpp;
  Real ypm, ypp;
  Real zpm, zpp;
  Real Pm, Pp;


  //compute radial pressure gradients, adjust circular velocities
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;

        //get density
        d = C.density[id];

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate radial position and phi (assumes disk is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos);
        phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)

        // radial acceleration from disk
        a_d = fabs(gr_disk_D3D(r, z_pos, hdp));
        // radial acceleration from halo 
        a_h = fabs(gr_halo_D3D(r, z_pos, hdp));

        //  pressure gradient along x direction
        // gradient calc is first order at boundaries
        if (i == H.n_ghost) idm = i + j*H.nx + k*H.nx*H.ny;
        else idm  = (i-1) + j*H.nx + k*H.nx*H.ny; 
        if (i == H.nx-H.n_ghost-1) idp  = i + j*H.nx + k*H.nx*H.ny;
        else idp  = (i+1) + j*H.nx + k*H.nx*H.ny; 
        Get_Position(i-1, j, k, &xpm, &ypm, &zpm);
        Get_Position(i+1, j, k, &xpp, &ypp, &zpm);
        Pm = C.Energy[idm]*(gama-1.0); // only internal energy stored in energy currently
        Pp = C.Energy[idp]*(gama-1.0); // only internal energy stored in energy currently
        dPdx =  (Pp-Pm)/(xpp-xpm);

        //pressure gradient along y direction
        if (j == H.n_ghost) idm = i + j*H.nx + k*H.nx*H.ny;
        else idm  = i + (j-1)*H.nx + k*H.nx*H.ny; 
        if (j == H.ny-H.n_ghost-1) idp  = i + j*H.nx + k*H.nx*H.ny; 
        else idp  = i + (j+1)*H.nx + k*H.nx*H.ny; 
        Get_Position(i, j-1, k, &xpm, &ypm, &zpm);
        Get_Position(i, j+1, k, &xpp, &ypp, &zpm);
        Pm = C.Energy[idm]*(gama-1.0); // only internal energy stored in energy currently
        Pp = C.Energy[idp]*(gama-1.0); // only internal energy stored in energy currently
        dPdy =  (Pp-Pm)/(ypp-ypm);

        //radial pressure gradient
        dPdr = x_pos*dPdx/r + y_pos*dPdy/r;

        //radial acceleration
        a = a_d + a_h + dPdr/d;
        if(a<0) //brant added
          a=0;
        /*
        if(isnan(a)||(a!=a)||(r*a<0))
        {
          printf("i %d j %d k %d a %e a_d %e dPdr %e d %e\n",i,j,k,a,a_d,dPdr,d);
          printf("i %d j %d k %d x_pos %e y_pos %e z_pos %e dPdx %e dPdy %e\n",i,j,k,x_pos,y_pos,z_pos,dPdx,dPdy);
          printf("i %d j %d k %d Pm %e Pp %e\n",i,j,k,Pm,Pp);
        }
        */
        

        // radial velocity 
        v = sqrt(r*a);
        vx = -sin(phi)*v;
        vy = cos(phi)*v;
        vz = 0;

        // set the momenta 
        C.momentum_x[id] = d*vx;
        C.momentum_y[id] = d*vy;
        C.momentum_z[id] = d*vz;

        //sheepishly check for NaN's!
        /*
        if((d<0)||(P<0)||(isnan(d))||(isnan(P))||(d!=d)||(P!=P))
          printf("d %e P %e i %d j %d k %d id %d\n",d,P,i,j,k,id);

        if((isnan(vx))||(isnan(vy))||(isnan(vz))||(vx!=vx)||(vy!=vy)||(vz!=vz))
          printf("vx %e vy %e vz %e i %d j %d k %d id %d\n",vx,vy,vz,i,j,k,id);
        */ 
        
      }
    }
  }

  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;
        
        // add kinetic contribution to total energy
        C.Energy[id] += 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id] + C.momentum_z[id]*C.momentum_z[id])/C.density[id];

      }
    } 
  }
  //MPI_Finalize();
  //exit(0);

/*
  // uses cylindrical coordinates (r, phi, z)
  // r, phi will be converted to x, y
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate radial position and phi (assumes disk is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos);
        phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)
        //theta = atan2(r, z_pos); // polar angle (from z to x-y plane)
        //rho = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos); // radius in spherical coordinates

        // mass density [M_sun / kpc^3]
        d = ((z_d*z_d*M_d)/(4*PI)) * (R_d*r*r + (R_d + 3*sqrt(z_pos*z_pos+z_d*z_d))*(R_d + sqrt(z_pos*z_pos+z_d*z_d))) / (pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d), 2),2.5) * pow(z_pos*z_pos + z_d*z_d, 1.5));
        n = d * DENSITY_UNIT / MP; // number density, cgs
        P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

        // radial acceleration due to Kuzmin disk + NFW halo
        x = r / R_s;
        a_d = GN * M_d * r * pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
        a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
        a = a_d + a_h;

        // radial velocity 
        v = sqrt(r*a);
        vx = -sin(phi)*v;
        vy = cos(phi)*v;

        // vertical acceleration due to Kuzmin disk + NFW halo
        //x = z_pos / R_s;
        //a_d = GN * M_d * z_pos * (R_d + sqrt(z_pos*z_pos + z_d*z_d)) / ( pow(r*r + pow(R_d + sqrt(z*z + z_d*z_d), 2), 1.5) * sqrt(z_pos*z_pos + z_d*z_d) );
        //a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * z_pos*z_pos);
        vz =  0.0;

        // set values of conserved variables   
        C.density[id] = d;
        C.momentum_x[id] = d*vx;
        C.momentum_y[id] = d*vy;
        C.momentum_z[id] = d*vz;
        C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy + vz*vz);
        //printf("%e %e %f %f %f %f %f\n", x_pos, y_pos, d, Sigma, vx, vy, P);
      }
    }
  }
*/

}




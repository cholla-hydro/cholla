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
#include "ran.h"
#include "rng.h"
#include "mpi_routines.h"
#include "io.h"
#include "error_handling.h"
#include <stdio.h>

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
  } else if (strcmp(P.init, "KH_res_ind_2D")==0) {
    KH_res_ind_2D();
  } else if (strcmp(P.init, "Implosion_2D")==0) {
    Implosion_2D();
  } else if (strcmp(P.init, "Explosion_2D")==0) {
    Explosion_2D();
  } else if (strcmp(P.init, "Noh_2D")==0) {
    Noh_2D();
  } else if (strcmp(P.init, "Cloud_2D")==0) {
    Cloud_2D();
  } else if (strcmp(P.init, "Noh_3D")==0) {
    Noh_3D();    
  } else if (strcmp(P.init, "Sedov_Taylor")==0) {
    Sedov_Taylor(P.rho_l, P.P_l, P.rho_r, P.P_r);
  } else if (strcmp(P.init, "Turbulent_Slab")==0) {
    Turbulent_Slab();
  } else if (strcmp(P.init, "Cloud_3D")==0) {
    Cloud_3D();
  } else if (strcmp(P.init, "Turbulence")==0) {
    Turbulence(P.rho, P.vx, P.vy, P.vz, P.P);
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
  Real T = 1e6;
  P = rho*KB*T / PRESSURE_UNIT;

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
        C.GasEnergy[id] = P/(gama-1.0);
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

  Real d_wind = 1.285209e-27 / DENSITY_UNIT;
  Real v_wind = 1.229560e8 / VELOCITY_UNIT;
  Real P_wind = 4.232212e-13 / PRESSURE_UNIT;
  Real d_cloud = rho_r;

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
          //C.density[id]    = d_wind;
          //C.momentum_x[id] = 0.0;
          C.momentum_x[id] = rho_l * v_l;
          //C.momentum_x[id] = d_wind * v_wind;
          C.momentum_y[id] = 0.0;
          //C.momentum_y[id] = rho_l * v_l;
          C.momentum_z[id] = 0.0;
          //C.momentum_z[id] = rho_l * v_l;
          C.Energy[id]     = P_l/(gama-1.0) + 0.5*rho_l*v_l*v_l;
          //C.Energy[id]     = P_wind/(gama-1.0) + 0.5*d_wind*v_wind*v_wind;
          #ifdef DE
          C.GasEnergy[id]  = P_l/(gama-1.0);
          //C.GasEnergy[id]  = P_wind/(gama-1.0);
          #endif
        }
        else
        {
          C.density[id]    = rho_r;
          //C.density[id]    = d_cloud;
          //C.momentum_x[id] = 0.0;
          C.momentum_x[id] = rho_r * v_r;
          C.momentum_y[id] = 0.0;
          //C.momentum_y[id] = rho_r * v_r;
          C.momentum_z[id] = 0.0;
          //C.momentum_z[id] = rho_r * v_r;
          C.Energy[id]     = P_r/(gama-1.0) + 0.5*rho_r*v_r*v_r;        
          //C.Energy[id]     = P_wind/(gama-1.0);        
          #ifdef DE
          C.GasEnergy[id]  = P_r/(gama-1.0);
          //C.GasEnergy[id]  = P_wind/(gama-1.0);
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
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real vx, vy, vz;


  // set the initial values of the conserved variables
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      // outer thirds of slab
      if (y_pos <= 1.0*H.ydglobal/3.0) 
      {
        C.density[id] = 1.0;
        C.momentum_x[id] = 0.5;
        //C.momentum_x[id] = 0.5 + 0.01*sin(2*PI*x_pos);
        C.momentum_y[id] = 0.0;
        //C.momentum_y[id] = 0.01*sin(2*PI*x_pos);
        C.momentum_z[id] = 0.0;
        C.Energy[id] = 2.5/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
      }
      else if (y_pos >= 2.0*H.ydglobal/3.0)
      {
        C.density[id] = 1.0;
        C.momentum_x[id] = 0.5;
        //C.momentum_x[id] = 0.5 + 0.01*sin(2*PI*x_pos);
        C.momentum_y[id] = 0.0;
        //C.momentum_y[id] = 0.0 + 0.01*sin(2*PI*x_pos);
        //C.momentum_y[id] = C.density[id]*(0.1*sin(4*PI*x_pos)*exp(-pow(y_pos-0.25,2)/(2*0.05*0.05) + pow(y_pos-0.75,2)/(2*0.05*0.05)));
        C.momentum_z[id] = 0.0;
        C.Energy[id] = 2.5/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
      }
      // inner third of slab
      else
      {
        C.density[id] = 2.0;
        //C.momentum_x[id] = -1.0 + 0.02*sin(2*PI*x_pos);
        C.momentum_x[id] = -1.0;
        C.momentum_y[id] = 0.0  + 0.02*sin(2*PI*x_pos);
        //C.momentum_y[id] = 0.0;
        //C.momentum_y[id] = C.density[id]*(0.1*sin(4*PI*x_pos)*exp(-pow(y_pos-0.25,2)/(2*0.05*0.05) + pow(y_pos-0.75,2)/(2*0.05*0.05)));
        C.momentum_z[id] = 0.0;
        C.Energy[id] = 2.5/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
      }
    }
  }

}


/*! \fn void KH_res_ind_2D()
 *  \brief Initialize the grid with a 2D Kelvin-Helmholtz instability whose modes are resolution independent. */
void Grid3D::KH_res_ind_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real mx, my, mz;


  // set the initial values of the conserved variables
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);
      
      // inner half of slab
      if (fabs(y_pos-0.5) < 0.25)
      {
        if (y_pos > 0.5)
        {
          C.density[id] = 2.0 - exp( -0.5*pow(y_pos-0.75 - sqrt(-2.0*0.05*0.05*log(0.5)),2)/pow(0.05,2) );
          C.momentum_x[id] = 0.5*C.density[id] - C.density[id] * exp( -0.5*pow(y_pos-0.75 - sqrt(-2.0*0.05*0.05*log(0.5)),2) / pow(0.05,2) );
        }
        else
        {
          C.density[id] = 2.0 - exp( -0.5*pow(y_pos-0.25 + sqrt(-2.0*0.05*0.05*log(0.5)),2)/pow(0.05,2) );
          C.momentum_x[id] = 0.5*C.density[id] - C.density[id] * exp( -0.5*pow(y_pos-0.25 + sqrt(-2.0*0.05*0.05*log(0.5)),2) / pow(0.05,2) );
        }
        C.momentum_y[id] = 0.0 + C.density[id] * 0.1*sin(4*PI*x_pos);
        C.momentum_z[id] = 0.0;
        mx = C.momentum_x[id];
        my = C.momentum_y[id];
        mz = C.momentum_z[id];
        C.Energy[id] = 2.5/(gama-1.0) + 0.5*(mx*my + my*my + mz*mz)/C.density[id];
      }
      // outer quarters of slab
      else
      {
        if (y_pos > 0.5)
        {
          C.density[id] = 1.0 + exp( -0.5*pow(y_pos-0.75 + sqrt(-2.0*0.05*0.05*log(0.5)),2)/pow(0.05,2) );
          C.momentum_x[id] = -0.5*C.density[id] + C.density[id] * exp( -0.5*pow(y_pos-0.75 + sqrt(-2.0*0.05*0.05*log(0.5)),2)/pow(0.05,2) );
        }
        else
        {
          C.density[id] = 1.0 + exp( -0.5*pow(y_pos-0.25 - sqrt(-2.0*0.05*0.05*log(0.5)),2)/pow(0.05,2) );
          C.momentum_x[id] = -0.5*C.density[id] + C.density[id] * exp( -0.5*pow(y_pos-0.25 - sqrt(-2.0*0.05*0.05*log(0.5)),2)/pow(0.05,2) );
        }
        C.momentum_y[id] = 0.0 + C.density[id] * 0.1*sin(4*PI*x_pos);
        C.momentum_z[id] = 0.0;
        mx = C.momentum_x[id];
        my = C.momentum_y[id];
        mz = C.momentum_z[id];
        C.Energy[id] = 2.5/(gama-1.0) + 0.5*(mx*mx + my*my + mz*mz)/C.density[id];
      }
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
        //printf("%f %f %f\n", x_pos, y_pos, P);
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
        //printf("%f %f %f\n", x_pos, y_pos, P);
      }
    }
  }

}


/*! \fn void Explosion_2D()
 *  \brief Explosion test described in Liska, 2003. */
void Grid3D::Explosion_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real P;
  Real weight, xpoint, ypoint;
  int incount, ii;


  // set the initial values of the conserved variables
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i + j*H.nx;
      // get the centered x and y positions at (x,y,z)
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      // inside the circle
      if (x_pos*x_pos + y_pos*y_pos < 0.159999 ) {
        C.density[id] = 1.0;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        P = 1.0;
        C.Energy[id] = P/(gama-1.0);
      }
      // outside the circle 
      else {
        C.density[id] = 0.125;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        P = 0.1;
        C.Energy[id] = P/(gama-1.0);
      }        
      // on the circle 
      if ((fabs(x_pos)-0.5*H.dx)*(fabs(x_pos)-0.5*H.dx) + (fabs(y_pos)-0.5*H.dy)*(fabs(y_pos)-0.5*H.dy) < 0.16 && (fabs(x_pos)+0.5*H.dx)*(fabs(x_pos)+0.5*H.dx) + (fabs(y_pos)+0.5*H.dy)*(fabs(y_pos)+0.5*H.dy) > 0.16) {
        // quick Monte Carlo to determine weighting
        Ran quickran(time(NULL));
        incount = 0;
        for (ii=0; ii<1000000; ii++) {
          // generate a random number between x_pos and dx
          xpoint = fabs(x_pos)-0.5*H.dx + H.dx*quickran.doub();
          // generate a random number between y_pos and dy
          ypoint = fabs(y_pos)-0.5*H.dy + H.dy*quickran.doub();
          // check to see whether the point is within the circle
          if (xpoint*xpoint + ypoint*ypoint < 0.159999) incount++;
        }
        weight = incount / 1000000.0;
        C.density[id] = weight+(1-weight)*0.125;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        P = weight+(1-weight)*0.1;
        C.Energy[id] = P/(gama-1.0);
      }
    }
  }

#ifndef  MPI_CHOLLA
  // enforce symmetry in initial conditions
  for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      id = i + j*H.nx;
      if (i < j) {
        C.density[j + i*H.nx] = C.density[i + j*H.nx];
        C.Energy[j + i*H.nx] = C.Energy[i + j*H.nx];
      }
    }
  }
#endif /*MPI_CHOLLA*/

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

/*! \fn void Cloud_2D()
 *  \brief Circular cloud in a hot diffuse wind. */
void Grid3D::Cloud_2D() {

  int i, j, id;
  Real x_pos, y_pos, z_pos, d_wind, v_wind, d_cloud;
  Real P, P_cloud, P_wind;
  Real xcen, ycen, r, R_c, R_max;
  Real vx, vy;

  d_wind = 1.285209e-27 / DENSITY_UNIT;
  v_wind = 1.229560e8 / VELOCITY_UNIT;
  P_wind = 4.232212e-13 / PRESSURE_UNIT;
  
  // number density of cloud in code units (hydrogen atom/cc)
  d_cloud = 1.0;
  P_cloud = P_wind;  // cloud in pressure equilibrium with hot wind
  R_max = 5.0; // radius of the edge of the cloud in code units (5pc)
  R_c = R_max/1.28; // radius at which cloud begins to taper

  // cloud center in code units
  xcen = 10.0;
  ycen = 60.0;

  // hot wind
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

      // get cell-centered position
      id = i + j*H.nx;
      Get_Position(i, j, 0, &x_pos, &y_pos, &z_pos);

      C.density[id] = d_wind;
      vx = v_wind;
      C.momentum_x[id] = C.density[id]*vx;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      C.Energy[id] = (P_wind)/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id] + C.momentum_z[id]*C.momentum_z[id])/C.density[id];
      #ifdef DE
      C.GasEnergy[id] = P_wind / (gama-1.0);
      #endif
      }
    }


  // circular cloud 
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

      id = i + j*H.nx;

      // get cell-centered position
      Get_Position(i, j, 0, &x_pos, &y_pos, &z_pos);

      // circular cloud with tapered edge
      // radial position relative to cloud ceneter
      r = sqrt((x_pos-xcen)*(x_pos-xcen) + (y_pos-ycen)*(y_pos-ycen));

      if (r < R_c ) {
        C.density[id] = d_cloud;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        C.Energy[id] = P_cloud/(gama-1.0);
        #ifdef DE
        C.GasEnergy[id] = P_cloud/(gama-1.0);
        #endif
      }
      if (r > R_c && r < R_max) {
        C.density[id] = d_cloud*exp(-1.0 *fabs(r - R_c)/4.0);
        if (C.density[id] < d_wind) C.density[id] = d_wind;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        C.Energy[id] = P_cloud/(gama-1.0);
        #ifdef DE
        C.GasEnergy[id] = P_cloud/(gama-1.0);
        #endif
      }
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



/*! \fn void Sedov_Taylor(Real rho_l, Real P_l, Real rho_r, Real P_r)
 *  \brief Sedov_Taylor blast wave test. */
void Grid3D::Sedov_Taylor(Real rho_l, Real P_l, Real rho_r, Real P_r)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;
  Real rsq, R, P, P_ambient, P_sedov;
  Real weight, xpoint, ypoint, zpoint;
  int incount, ii;
  R = H.dx; //radius of the explosion - one cell width
  P_ambient = P_r;
  P_sedov = P_l;

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

        // get radius
        rsq = x_pos*x_pos;
        if (H.ny > 1) rsq += y_pos*y_pos;
        if (H.nz > 1) rsq += z_pos*z_pos;
        

        // inside the sphere 
        if (rsq < R*R) {
          C.density[id] = rho_l;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          C.Energy[id] = P_sedov/(gama-1.0);
        }
        // outside the sphere 
        else {
          C.density[id] = rho_r;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          C.Energy[id] = P_ambient/(gama-1.0);
        }
        // on the sphere 
        
        if ((x_pos-0.5*H.dx)*(x_pos-0.5*H.dx) + (y_pos-0.5*H.dy)*(y_pos-0.5*H.dy) + (z_pos-0.5*H.dz)*(z_pos-0.5*H.dz) < R*R && (x_pos+0.5*H.dx)*(x_pos+0.5*H.dx) + (y_pos+0.5*H.dy)*(y_pos+0.5*H.dy) + (z_pos+0.5*H.dz)*(z_pos+0.5*H.dz) > R*R) {
          // quick Monte Carlo to determine weighting
          Ran quickran(time(NULL));
          incount = 0;
          for (ii=0; ii<1000; ii++) {
            // generate a random number between x_pos and dx
            xpoint = x_pos-0.5*H.dx + H.dx*quickran.doub();
            // generate a random number between y_pos and dy
            ypoint = y_pos-0.5*H.dy + H.dy*quickran.doub();
            // generate a random number between z_pos and dz
            zpoint = z_pos-0.5*H.dz + H.dz*quickran.doub();
            // check to see whether the point is within the sphere 
            if (xpoint*xpoint + ypoint*ypoint + zpoint*zpoint < R*R) incount++;
          }
          weight = incount / 1000.0;
          C.density[id] = rho_l;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          P = weight*P_sedov + (1-weight)*P_ambient;
          C.Energy[id] = P/(gama-1.0);
        }
        
      }
    }
  }



}


/*! \fn void Turbulent_Slab()
 *  \brief Turbulent slab in a hot diffuse wind. */
void Grid3D::Turbulent_Slab() {

  int i, j, k, id;
  Real x_pos, y_pos, z_pos, d_0;
  Real P, P_shock, P_sedov, P_cloud;
  Real xcen, ycen, zcen, r, R_c, R_max;
  Real weight, xpoint, ypoint, zpoint;
  int incount, iii;
  Real velocity_unit = LENGTH_UNIT / TIME_UNIT;
  Real pressure_unit = DENSITY_UNIT * LENGTH_UNIT * LENGTH_UNIT / (TIME_UNIT * TIME_UNIT);
  Real vx, vz;

  // mean density of slab (2e-24/cc)
  d_0 = 2.0;

  // ambient medium 
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        // CODE UNITS:
        // ambient medium is 0.1 hydrogen atom / cc
        C.density[id] = 0.1;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        C.momentum_z[id] = C.density[id]*vz;
        // scale the pressure such that the ambient medium has 
        // a temperature of 10000 Kelvin
        P_cloud = 0.1*KB*1e4 / pressure_unit;
        C.Energy[id] = (P_cloud)/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id] + C.momentum_z[id]*C.momentum_z[id])/C.density[id];
        #ifdef DE
        C.GasEnergy[id] = P_cloud/(gama-1.0);
        #endif
      }
    }
  }


  // turbulent slab
  FILE *fp;
  fp = fopen("/gsfs1/rsgrps/brant/evan/data/slab/slab.128.dat", "r");
  //fp = fopen("/gsfs1/rsgrps/brant/evan/data/slab/slab.256.dat", "r");

  if (fp == NULL) {
    chprintf("Can't open input file.\n");
    chexit(1);
  }
  
  // read in the slab data
  int nx, ny, nz;
  fread(&nx, 1, sizeof(int), fp);
  fread(&ny, 1, sizeof(int), fp);
  fread(&nz, 1, sizeof(int), fp);

  int ii, jj, kk;
  float d, mx, my, mz;
  for(i=0;i<nx;i++) {
    for(j=0;j<ny;j++) {
      for(k=0;k<nz;k++) {
        fread(&ii, 1, sizeof(int), fp);
        fread(&jj, 1, sizeof(int), fp);
        fread(&kk, 1, sizeof(int), fp);
        fread(&d, 1, sizeof(float), fp);
        fread(&mx, 1, sizeof(float), fp);
        fread(&my, 1, sizeof(float), fp);
        fread(&mz, 1, sizeof(float), fp);
        // only place in cells that are in your domain
        #ifdef MPI_CHOLLA
        //if (ii >= nx_local_start && ii < nx_local_start+nx_local) {
        if (kk+1*nx_global/32 >= nx_local_start && kk+1*nx_global/32 < nx_local_start+nx_local) {
        if (jj >= ny_local_start && jj < ny_local_start+ny_local) {
        //if (kk+1*nz_global/16 >= nz_local_start && kk+1*nz_global/16 < nz_local_start+nz_local) {
        if (ii >= nz_local_start && ii < nz_local_start+nz_local) {
          //id = ii+H.n_ghost-nx_local_start + (jj+H.n_ghost-ny_local_start)*H.nx + (kk+1*nz_global/8+H.n_ghost-nz_local_start)*H.nx*H.ny;
          id = kk+1*nx_global/32+H.n_ghost-nx_local_start + (jj+H.n_ghost-ny_local_start)*H.nx + (ii+H.n_ghost-nz_local_start)*H.nx*H.ny;
        #endif
        #ifndef MPI_CHOLLA
          id = (ii+H.n_ghost) + (jj+H.n_ghost)*H.nx + (kk+nz/8+H.n_ghost)*H.nx*H.ny;
        #endif
          //scale the slab density such that the average density matches
          C.density[id] = d * d_0;
          if (C.density[id] < 0.1) C.density[id] = 0.1;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          //C.Energy[id] = (P_cloud)/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id] + C.momentum_z[id]*C.momentum_z[id])/C.density[id];
          C.Energy[id] = (P_cloud)/(gama-1.0);
          #ifdef DE
          C.GasEnergy[id] = P_cloud/(gama-1.0);
          #endif
        #ifdef MPI_CHOLLA
        }
	}
	}
        #endif
      }
    }
  }
  fclose(fp);

/*
  // constant density slab 
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        if (z_pos > 0.125*20 && z_pos < 0.25*20 ) {
          C.density[id] = d_0;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          C.Energy[id] = P_cloud/(gama-1.0);
        }
      }
    }
  }
*/
}


/*! \fn void Cloud_3D()
 *  \brief Turbulent cloud in a hot diffuse wind. */
void Grid3D::Cloud_3D() {

  int i, j, k, id;
  Real x_pos, y_pos, z_pos, d_wind, v_wind, d_cloud;
  Real P, P_shock, P_sedov, P_cloud, P_wind;
  Real xcen, ycen, zcen, r, R_c, R_max;
  Real weight, xpoint, ypoint, zpoint;
  int incount, iii;
  //Real velocity_unit = LENGTH_UNIT / TIME_UNIT;
  //Real pressure_unit = DENSITY_UNIT * LENGTH_UNIT * LENGTH_UNIT / (TIME_UNIT * TIME_UNIT);
  Real vx, vy, vz;

  // CODE UNITS:
  // density: 1.67e-24  # 1.0 hydrogen atom/cc (g/cc)
  // length:  3.0857e18 # 1 parsec in cm
  // time:    3.1557e10 # 1 kyr in seconds

  // Cooper 2009
  //d_wind = 0.1;
  //v_wind = 1.20e8 / VELOCITY_UNIT;  // 1200km/s (from Cooper 2009)
  //P_wind = d_ism*KB*5e6 / PRESSURE_UNIT;  // wind temp of 5e6 K (from Cooper 2009)
  // Mach 1 CC85
  //d_wind = 2.712148e-26 / DENSITY_UNIT;
  //v_wind = 6.473926e7 / VELOCITY_UNIT;
  //P_wind = 6.820245e-11 / PRESSURE_UNIT;
  // Mach 5.25 (R = 1000pc) (old parameters)
  //d_wind = 1.285209e-27 / DENSITY_UNIT;
  //v_wind = 1.229560e8 / VELOCITY_UNIT;
  //P_wind = 4.232212e-13 / PRESSURE_UNIT;
  // Mach 5.25 (R = 1000pc) (new parameters)
  d_wind = 8.807181e-27 / DENSITY_UNIT;
  v_wind = 1.196177e8 / VELOCITY_UNIT;
  P_wind = 2.744870e-12 / PRESSURE_UNIT;
  
  // number density of cloud in code units (hydrogen atom/cc)
  d_cloud = 1.0;
  P_cloud = P_wind;  // cloud in pressure equilibrium with hot wind
  R_max = 5.0; // radius of the edge of the cloud in code units (5pc)
  R_c = R_max/1.28; // radius at which cloud begins to taper

  // Set initial conditions to be pre-shock CGM
  // in pressure equilibrium with cloud
  d_wind = 1.0e-3;
  v_wind = 0.0;
  P_wind = d_cloud * KB * 1e3;
  

  // cloud center in code units
  xcen = 10.0;
  ycen = 30.0;
  zcen = 30.0;

  // hot wind
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        // get cell-centered position
        id = i + j*H.nx + k*H.nx*H.ny;
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        C.density[id] = d_wind;
        vx = v_wind;
        C.momentum_x[id] = C.density[id]*vx;
        C.momentum_y[id] = 0.0;
        vz = 0.0;
        C.momentum_z[id] = C.density[id]*vz;
        C.Energy[id] = (P_wind)/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id] + C.momentum_z[id]*C.momentum_z[id])/C.density[id];
        #ifdef DE
        C.GasEnergy[id] = P_wind / (gama-1.0);
        #endif
      }
    }
  }


/*
  // sedov explosion in ambient medium
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        R = 2*H.dx; // radius of the explosion in code units (2/256)

        // CODE UNITS:
        // density: 1.67e-25  # 0.1 hydrogen atom/cc (g/cc)
        // length:  3.0857e19 # 10 parsec in cm
        // time:    3.1557e10 # 1 kyr in seconds
        // inside the sphere, sn explosion, 10^51 ergs  
        if (x_pos*x_pos + y_pos*y_pos + z_pos*z_pos < R*R ) {
          C.density[id] = 1.0;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          P_sedov = 71135;
          C.Energy[id] = P_sedov/(gama-1.0);
        }
        // outside the sphere 
        else {
          C.density[id] = 1.0;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          // scale the pressure such that the ambient medium has
          // a temperature of 10000 Kelvin
          P_cloud = 8.641e-7;
          C.Energy[id] = P_cloud/(gama-1.0);
        }
        // on the sphere 
        if ((x_pos-0.5*H.dx)*(x_pos-0.5*H.dx) + (y_pos-0.5*H.dy)*(y_pos-0.5*H.dy) + (z_pos-0.5*H.dz)*(z_pos-0.5*H.dz) < R*R && (x_pos+0.5*H.dx)*(x_pos+0.5*H.dx) + (y_pos+0.5*H.dy)*(y_pos+0.5*H.dy) + (z_pos+0.5*H.dz)*(z_pos+0.5*H.dz) > R*R) {
          // quick Monte Carlo to determine weighting
          Ran quickran(time(NULL));
          incount = 0;
          for (iii=0; iii<1000; iii++) {
            // generate a random number between x_pos and dx
            xpoint = x_pos-0.5*H.dx + H.dx*quickran.doub();
            // generate a random number between y_pos and dy
            ypoint = y_pos-0.5*H.dy + H.dy*quickran.doub();
            // generate a random number between z_pos and dz
            zpoint = z_pos-0.5*H.dz + H.dz*quickran.doub();
            // check to see whether the point is within the sphere 
            if (xpoint*xpoint + ypoint*ypoint + zpoint*zpoint < R*R) incount++;
          }
          weight = incount / 1000.0;
          C.density[id] = 1.0;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          P = weight*P_sedov + (1-weight)*P_cloud;
          C.Energy[id] = P/(gama-1.0);
        }        
      }
    }
  }
*/      
/*
  // turbulent cloud
  FILE *fp;
  //fp = fopen("/gsfs1/rsgrps/brant/evan/data/cloud_3D/cloud.64.dat", "r");
  fp = fopen("/gsfs1/rsgrps/brant/evan/data/cloud_3D/cloud.128.dat", "r");

  if (fp == NULL) {
    chprintf("Can't open input file.\n");
    chexit(1);
  }
  
  // read in the cloud data
  int nx, ny, nz;
  fread(&nx, 1, sizeof(int), fp);
  fread(&ny, 1, sizeof(int), fp);
  fread(&nz, 1, sizeof(int), fp);
  //printf("%d %d %d\n", nx_local_start, nx_local, nx_global);

  int ii, jj, kk;
  float d, mx, my, mz;
  int ioff, joff, koff;
  for(i=0;i<nx;i++) {
    for(j=0;j<ny;j++) {
      for(k=0;k<nz;k++) {

        // read in cloud data
        fread(&ii, 1, sizeof(int), fp);
        fread(&jj, 1, sizeof(int), fp);
        fread(&kk, 1, sizeof(int), fp);
        fread(&d, 1, sizeof(float), fp);
        fread(&mx, 1, sizeof(float), fp);
        fread(&my, 1, sizeof(float), fp);
        fread(&mz, 1, sizeof(float), fp);
        // only place in cells that are in your domain
        #ifdef MPI_CHOLLA
        ioff = 1*nx_global/30;
        joff = 5*ny_global/12;
        koff = 5*nz_global/12;
        if (ii+ioff >= nx_local_start && ii+ioff < nx_local_start+nx_local) {
        if (jj+joff >= ny_local_start && jj+joff < ny_local_start+ny_local) {
        if (kk+koff >= nz_local_start && kk+koff < nz_local_start+nz_local) {
          id = ii+ioff+H.n_ghost-nx_local_start + (jj+joff+H.n_ghost-ny_local_start)*H.nx + (kk+koff+H.n_ghost-nz_local_start)*H.nx*H.ny;
          // get cell-centered position
          Get_Position(ii+ioff+H.n_ghost-nx_local_start, jj+joff+H.n_ghost-ny_local_start, kk+koff+H.n_ghost-nz_local_start, &x_pos, &y_pos, &z_pos);
        #endif
        #ifndef MPI_CHOLLA
          id = (ii+nx/2+H.n_ghost) + (jj+ny/2+H.n_ghost)*H.nx + (kk+nz/2+H.n_ghost)*H.nx*H.ny;
          Get_Position(ii+ioff+H.n_ghost, jj+joff+H.n_ghost, kk+koff+H.n_ghost, &x_pos, &y_pos, &z_pos);
        #endif

	  // radial position relative to cloud ceneter
          r = sqrt((x_pos-xcen)*(x_pos-xcen) + (y_pos-ycen)*(y_pos-ycen) + (z_pos-zcen)*(z_pos-zcen));

          //scale the cloud density such that the ambient density matches (20*0.005)
          d = d_cloud*d;

          //only place cells within the region defined by the cloud radius
          if (r < R_c) {
            C.density[id] = d;
            if (C.density[id] < d_wind) C.density[id] = d_wind;
            C.momentum_x[id] = 0.0;
            C.momentum_y[id] = 0.0;
            C.momentum_z[id] = 0.0;
            C.Energy[id] = (P_cloud)/(gama-1.0);
            #ifdef DE
            C.GasEnergy[id] = P_cloud/(gama-1.0);
            #endif
          }
          if (r > R_c && r < R_max) {
            C.density[id] = d*exp(-5.0*fabs(r - R_c)/(R_max-R_c));
            if (C.density[id] < d_wind) C.density[id] = d_wind;
            C.momentum_x[id] = 0.0;
            C.momentum_y[id] = 0.0;
            C.momentum_z[id] = 0.0;
            C.Energy[id] = P_cloud/(gama-1.0);
            #ifdef DE
            C.GasEnergy[id] = P_cloud/(gama-1.0);
            #endif
          }
        #ifdef MPI_CHOLLA
        }
        }
        }
        #endif
      }
    }
  }
  fclose(fp);
*/
/*
  // spherical cloud 
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

      	// spherical cloud with tapered edge
      	// radial position relative to cloud ceneter
        r = sqrt((x_pos-xcen)*(x_pos-xcen) + (y_pos-ycen)*(y_pos-ycen) + (z_pos-zcen)*(z_pos-zcen));
	
        if (r < R_c ) {
          C.density[id] = d_cloud;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          C.Energy[id] = P_cloud/(gama-1.0);
          #ifdef DE
          C.GasEnergy[id] = P_cloud/(gama-1.0);
          #endif
        }
        if (r > R_c && r < R_max) {
          C.density[id] = d_cloud*exp(-1.0 *fabs(r - R_c)/4.0);
          if (C.density[id] < d_wind) C.density[id] = d_wind;
          C.momentum_x[id] = 0.0;
          C.momentum_y[id] = 0.0;
          C.momentum_z[id] = 0.0;
          C.Energy[id] = P_cloud/(gama-1.0);
          #ifdef DE
          C.GasEnergy[id] = P_cloud/(gama-1.0);
          #endif
        }
      }
    }
  }
*/

}


/*! \fn void Turbulence(Real rho, Real vx, Real vy, Real vz, Real P)
 *  \brief Generate a 3D turbulent forcing field. */
void Grid3D::Turbulence(Real rho, Real vx, Real vy, Real vz, Real P)
{
/*
  int i, j, k, id, fid;
  int n_cells = H.nx_real * H.ny_real * H.nz_real;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;
  Real *A1, *A2, *A3, *A4, *B1, *B2, *B3, *B4;
  Real *vxp, *vyp, *vzp;
  Real mxp, myp, mzp;
  Real vx_av, vy_av, vz_av, v_av;
  Real vxsq, vysq, vzsq;
  Real mxsq, mysq, mzsq;
  Real mx, my, mz;
  Real M, d_tot;
  A1 = rng_direction(3);
  A2 = rng_direction(3);
  A3 = rng_direction(3);
  A4 = rng_direction(3);
  B1 = rng_direction(3);
  B2 = rng_direction(3);
  B3 = rng_direction(3);
  B4 = rng_direction(3);
  vxp = (Real *) malloc(n_cells*sizeof(Real));
  vyp = (Real *) malloc(n_cells*sizeof(Real));
  vzp = (Real *) malloc(n_cells*sizeof(Real));
  //printf("%f %f %f\n", A[0], A[1], A[2]);
  //printf("%f %f %f\n", B[0], B[1], B[2]);
  //printf("%f %f %f\n", A2[0], A2[1], A2[2]);
  //printf("%f %f %f\n", B2[0], B2[1], B2[2]);

  //set the desired Mach number
  M = 2.0;
  d_tot = 0;

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

  mxp = myp = mzp = 0.0;
  mx = my = mz = 0.0;
  vx_av = vy_av = vz_av = 0.0;
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;
        fid = (i-H.n_ghost) + (j-H.n_ghost)*H.nx_real + (k-H.n_ghost)*H.nx_real*H.ny_real;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // set constant initial states
        C.density[id]    = rho;
        C.momentum_x[id] = rho*vx;
        C.momentum_y[id] = rho*vy;
        C.momentum_z[id] = rho*vz;
        C.Energy[id]     = P/(gama-1.0);

        // calculate velocity perturbations
        vxp[fid] = B1[0]*sin( (2*PI/sqrt(A1[0]*A1[0] + A1[1]*A1[1] + A1[2]*A1[2])) * (A1[0]*x_pos + A1[1]*y_pos + A1[2]*z_pos));
        vyp[fid] = B1[1]*sin( (2*PI/sqrt(A1[0]*A1[0] + A1[1]*A1[1] + A1[2]*A1[2])) * (A1[0]*x_pos + A1[1]*y_pos + A1[2]*z_pos));
        vzp[fid] = B1[2]*sin( (2*PI/sqrt(A1[0]*A1[0] + A1[1]*A1[1] + A1[2]*A1[2])) * (A1[0]*x_pos + A1[1]*y_pos + A1[2]*z_pos));
        vxp[fid] += B2[0]*sin( (4*PI/sqrt(A2[0]*A2[0] + A2[1]*A2[1] + A2[2]*A2[2])) * (A2[0]*x_pos + A2[1]*y_pos + A2[2]*z_pos));
        vyp[fid] += B2[1]*sin( (4*PI/sqrt(A2[0]*A2[0] + A2[1]*A2[1] + A2[2]*A2[2])) * (A2[0]*x_pos + A2[1]*y_pos + A2[2]*z_pos));
        vzp[fid] += B2[2]*sin( (4*PI/sqrt(A2[0]*A2[0] + A2[1]*A2[1] + A2[2]*A2[2])) * (A2[0]*x_pos + A2[1]*y_pos + A2[2]*z_pos));
         
        vxp[fid] += B3[0]*cos( (2*PI/sqrt(A3[0]*A3[0] + A3[1]*A3[1] + A3[2]*A3[2])) * (A3[0]*x_pos + A3[1]*y_pos + A3[2]*z_pos));
        vyp[fid] += B3[1]*cos( (2*PI/sqrt(A3[0]*A3[0] + A3[1]*A3[1] + A3[2]*A3[2])) * (A3[0]*x_pos + A3[1]*y_pos + A3[2]*z_pos));
        vzp[fid] += B3[2]*cos( (2*PI/sqrt(A3[0]*A3[0] + A3[1]*A3[1] + A3[2]*A3[2])) * (A3[0]*x_pos + A3[1]*y_pos + A3[2]*z_pos));
        vxp[fid] += B4[0]*cos( (4*PI/sqrt(A4[0]*A4[0] + A4[1]*A4[1] + A4[2]*A4[2])) * (A4[0]*x_pos + A4[1]*y_pos + A4[2]*z_pos));
        vyp[fid] += B4[1]*cos( (4*PI/sqrt(A4[0]*A4[0] + A4[1]*A4[1] + A4[2]*A4[2])) * (A4[0]*x_pos + A4[1]*y_pos + A4[2]*z_pos));
        vzp[fid] += B4[2]*cos( (4*PI/sqrt(A4[0]*A4[0] + A4[1]*A4[1] + A4[2]*A4[2])) * (A4[0]*x_pos + A4[1]*y_pos + A4[2]*z_pos));
        
       
        // calculate momentum of forcing field
        mxp = C.density[id]*vxp[fid]; 
        myp = C.density[id]*vyp[fid]; 
        mzp = C.density[id]*vzp[fid]; 

        // track total momentum
        mx += mxp;
        my += myp;
        mz += mzp;

        d_tot += C.density[id];
      }
    }
  }
  // calculate density weighted average velocity of forcing field
  vx_av = mx / d_tot;
  vy_av = my / d_tot;
  vz_av = mz / d_tot;
  //printf("%f %f %f\n", vx_av, vy_av, vz_av);

  mxsq = mysq = mzsq = 0.0;
  mx = my = mz = 0.0;
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;
        fid = (i-H.n_ghost) + (j-H.n_ghost)*H.nx_real + (k-H.n_ghost)*H.nx_real*H.ny_real;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // subtract off average velocity to create a field with net zero momentum
        vxp[fid] -= vx_av;
        vyp[fid] -= vy_av;
        vzp[fid] -= vz_av;
        
        // calculate momentum of forcing field
        mxp = C.density[id]*vxp[fid]; 
        myp = C.density[id]*vyp[fid]; 
        mzp = C.density[id]*vzp[fid]; 

        // track total momentum
        mx += mxp; 
        my += myp; 
        mz += mzp; 

        // calculate <v^2> for each direction
        mxsq += mxp*mxp/C.density[id];
        mysq += myp*myp/C.density[id];
        mzsq += mzp*mzp/C.density[id];
        
      }
    }
  }
  vx_av = sqrt(mxsq / d_tot);
  vy_av = sqrt(mysq / d_tot);
  vz_av = sqrt(mzsq / d_tot);
  v_av = sqrt(vx_av*vx_av + vy_av*vy_av + vz_av*vz_av);
  //printf("%f %f %f %f\n", vx_av, vy_av, vz_av, v_av);
  //printf("%f %f %f\n", mx, my, mz); 

  mx = my = mz = 0.0;
  mxp = myp = mzp = 0.0;
  mxsq = mysq = mzsq = 0.0;
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        // get cell index
        id = i + j*H.nx + k*H.nx*H.ny;
        fid = (i-H.n_ghost) + (j-H.n_ghost)*H.nx_real + (k-H.n_ghost)*H.nx_real*H.ny_real;

        // rescale velocities to get desired Mach number
        vxp[fid] *= sqrt(M*M/3.0) / vx_av;
        vyp[fid] *= sqrt(M*M/3.0) / vy_av;
        vzp[fid] *= sqrt(M*M/3.0) / vz_av;
        
        // calculate momentum perturbations and apply
        mxp = C.density[id]*vxp[fid];
        myp = C.density[id]*vyp[fid];
        mzp = C.density[id]*vzp[fid];
        C.momentum_x[id] += mxp;
        C.momentum_y[id] += myp;
        C.momentum_z[id] += mzp;
        C.Energy[id] += 0.5*(mxp*mxp + myp*myp + mzp*mzp)/C.density[id];

        // track total momentum
        mx += C.momentum_x[id];
        my += C.momentum_y[id];
        mz += C.momentum_z[id];

        // calculate <v^2> for each direction
        mxsq += C.momentum_x[id]*C.momentum_x[id]/C.density[id];
        mysq += C.momentum_y[id]*C.momentum_y[id]/C.density[id];
        mzsq += C.momentum_z[id]*C.momentum_z[id]/C.density[id];

      }
    }
  }
  vx_av = sqrt(mxsq / d_tot);
  vy_av = sqrt(mysq / d_tot);
  vz_av = sqrt(mzsq / d_tot);
  v_av = sqrt(vx_av*vx_av + vy_av*vy_av + vz_av*vz_av);
  printf("%f %f %f %f\n", vx_av, vy_av, vz_av, v_av);
  //printf("%f %f %f\n", mx, my, mz);  

  free(vxp);
  free(vyp);
  free(vzp);
  free(A1);
  free(A2);
  free(A3);
  free(A4);
  free(B1);
  free(B2);
  free(B3);
  free(B4);
*/
}



/*! \fn void Read_Grid()
 *  \brief Read in grid data from an output file. */
void Grid3D::Read_Grid(struct parameters P) {

  int id, i, j, k;
  Real buf;

  FILE *fp;
  char filename[80];
  char timestep[20];
  int nfile = P.nfile; //output step you want to read from

  // create the filename to read from
  // assumes your data is in the outdir specified in the input file
  strcpy(filename, P.outdir);
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);
  strcat(filename,".bin");
  // for now assumes you will run on the same number of processors
  #ifdef MPI_CHOLLA
  sprintf(filename,"%s.%d",filename,procID);
  #endif

  fp = fopen(filename, "r");
  if (!fp) {
    printf("Unable to open input file.\n");
  }

  // Read in the header data
  fread(&H.n_cells, sizeof(int), 1, fp); 
  fread(&H.n_ghost, sizeof(int), 1, fp); 
  fread(&H.nx, sizeof(int), 1, fp); 
  fread(&H.ny, sizeof(int), 1, fp); 
  fread(&H.nz, sizeof(int), 1, fp); 
  fread(&H.nx_real, sizeof(int), 1, fp); 
  fread(&H.ny_real, sizeof(int), 1, fp); 
  fread(&H.nz_real, sizeof(int), 1, fp); 
  fread(&H.xbound, sizeof(Real), 1, fp); 
  fread(&H.ybound, sizeof(Real), 1, fp); 
  fread(&H.zbound, sizeof(Real), 1, fp); 
  fread(&H.domlen_x, sizeof(Real), 1, fp); 
  fread(&H.domlen_y, sizeof(Real), 1, fp); 
  fread(&H.domlen_z, sizeof(Real), 1, fp); 
  fread(&H.xblocal, sizeof(Real), 1, fp); 
  fread(&H.yblocal, sizeof(Real), 1, fp); 
  fread(&H.zblocal, sizeof(Real), 1, fp); 
  fread(&H.xdglobal, sizeof(Real), 1, fp); 
  fread(&H.ydglobal, sizeof(Real), 1, fp); 
  fread(&H.zdglobal, sizeof(Real), 1, fp); 
  fread(&H.dx, sizeof(Real), 1, fp); 
  fread(&H.dy, sizeof(Real), 1, fp); 
  fread(&H.dz, sizeof(Real), 1, fp); 
  fread(&H.t, sizeof(Real), 1, fp); 
  fread(&H.dt, sizeof(Real), 1, fp); 
  fread(&H.t_wall, sizeof(Real), 1, fp); 
  fread(&H.n_step, sizeof(int), 1, fp); 
  //fread(&H, 1, 184, fp);


  // Read in the conserved quantities from the input file
  #ifdef WITH_GHOST
  fread(&(C.density[id]),    sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_x[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_y[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_z[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.Energy[id]),     sizeof(Real), H.n_cells, fp); 
  #endif //WITH_GHOST

  #ifdef NO_GHOST
  // 1D case
  if (H.nx>1 && H.ny==1 && H.nz==1) {

    id = H.n_ghost;

    fread(&(C.density[id]),    sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.Energy[id]),     sizeof(Real), H.nx_real, fp);
    #ifdef DE
    fread(&(C.GasEnergy[id]),  sizeof(Real), H.nx_real, fp);
    #endif
  }

  // 2D case
  else if (H.nx>1 && H.ny>1 && H.nz==1) {
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.density[id]), sizeof(Real), H.nx_real, fp);
    }    
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
    }     
    #ifdef DE
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
    }     
    #endif
  }

  // 3D case
  else {
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.density[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    #ifdef DE
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    #endif

  }    
  #endif

  fclose(fp);
}

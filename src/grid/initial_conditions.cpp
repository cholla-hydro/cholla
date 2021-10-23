/*! \file initial_conditions.cpp
 *  \brief Definitions of initial conditions for different tests.
           Note that the grid is mapped to 1D as i + (x_dim)*j + (x_dim*y_dim)*k.
           Functions are members of the Grid3D class. */


#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../mpi/mpi_routines.h"
#include "../io/io.h"
#include "../utils/error_handling.h"
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;

/*! \fn void Set_Initial_Conditions(parameters P)
 *  \brief Set the initial conditions based on info in the parameters structure. */
void Grid3D::Set_Initial_Conditions(parameters P) {

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
  } else if (strcmp(P.init, "KH")==0) {
    KH();
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
  } else if (strcmp(P.init, "Disk_3D_particles")==0) {
    #ifndef ONLY_PARTICLES
    Disk_3D(P);
    #else
    // Initialize a m hydro grid when only integrating particles
    Uniform_Grid();
    #endif
  } else if (strcmp(P.init, "Spherical_Overpressure_3D")==0) {
    Spherical_Overpressure_3D();
  } else if (strcmp(P.init, "Spherical_Overdensity_3D")==0) {
    Spherical_Overdensity_3D();
  } else if (strcmp(P.init, "Read_Grid")==0) {
    #ifndef ONLY_PARTICLES
    Read_Grid(P);
    #else
    // Initialize a uniform hydro grid when only integrating particles
    Uniform_Grid();
    #endif
  } else if (strcmp(P.init, "Uniform")==0) {
    Uniform_Grid();
  } else if (strcmp(P.init, "Zeldovich_Pancake")==0) {
    Zeldovich_Pancake(P);
  } else {
    chprintf ("ABORT: %s: Unknown initial conditions!\n", P.init);
    chexit(-1);
  }

  if ( C.device != NULL )
    {
    CudaSafeCall(
      cudaMemcpy(C.device, C.density, H.n_fields*H.n_cells*sizeof(Real),
                 cudaMemcpyHostToDevice) );
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
  Real mu = 0.6;
  Real n, T;

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
/*
        if (i==istart && j==jstart && k==kstart) {
          n = rho*DENSITY_UNIT / (mu*MP);
          T = P*PRESSURE_UNIT / (n*KB);
          printf("Initial n = %e, T = %e\n", n, T);
        }
*/
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
        #ifdef SCALAR
        C.scalar[id] = C.density[id]*0.0;
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
          #ifdef SCALAR
          C.scalar[id] = C.density[id]*1.0;
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
          #ifdef SCALAR
          C.scalar[id] = 1.0*rho_l;
          #endif
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
          #ifdef SCALAR
          C.scalar[id] = 0.0*rho_r;
          #endif
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


/*! \fn void KH()
 *  \brief Initialize the grid with a Kelvin-Helmholtz instability.
           This version of KH test has a discontinuous boundary.
           Use KH_res_ind for a version that is resolution independent. */
void Grid3D::KH()
{
  int i, j, k, id;
  int istart, iend, jstart, jend, kstart, kend;
  Real x_pos, y_pos, z_pos;
  Real vx, vy, vz;
  Real d1, d2, v1, v2, P, A;

  d1 = 2.0;
  d2 = 1.0;
  v1 = 0.5;
  v2 = -0.5;
  P = 2.5;
  A = 0.1;

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

        // outer quarters of slab
        if (y_pos <= 1.0*H.ydglobal/4.0)
        {
          C.density[id] = d2;
          C.momentum_x[id] = v2*C.density[id];
          C.momentum_y[id] = C.density[id]*A*sin(4*PI*x_pos);
          C.momentum_z[id] = 0.0;
          C.Energy[id] = P/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
          #ifdef SCALAR
          C.scalar[id] = 0.0;
          #endif
        }
        else if (y_pos >= 3.0*H.ydglobal/4.0)
        {
          C.density[id] = d2;
          C.momentum_x[id] = v2*C.density[id];
          C.momentum_y[id] = C.density[id]*A*sin(4*PI*x_pos);
          C.momentum_z[id] = 0.0;
          C.Energy[id] = P/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
          #ifdef SCALAR
          C.scalar[id] = 0.0;
          #endif
        }
        // inner half of slab
        else
        {
          C.density[id] = d1;
          C.momentum_x[id] = v1*C.density[id];
          C.momentum_y[id] = C.density[id]*A*sin(4*PI*x_pos);
          C.momentum_z[id] = 0.0;
          C.Energy[id] = P/(gama-1.0) + 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id])/C.density[id];
          #ifdef SCALAR
          C.scalar[id] = 1.0*d1;
          #endif
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
  v1 = 10.5; // inner velocity
  v2 = 9.5; // outer velocity
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

/*! \fn void Spherical_Overpressure_3D()
 *  \brief Spherical overdensity and overpressure causing an spherical explosion */
void Grid3D::Spherical_Overpressure_3D()
{
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r, center_x, center_y, center_z;
  Real density, pressure, overDensity, overPressure, energy;
  Real vx, vy, vz, v2;
  center_x = 0.5;
  center_y = 0.5;
  center_z = 0.5;
  overDensity = 1;
  overPressure = 10;
  vx = 0;
  vy = 0;
  vz = 0;

  // set the initial values of the conserved variables
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;

        // // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        density = 0.1;
        pressure = 1;

        r = sqrt( (x_pos-center_x)*(x_pos-center_x) + (y_pos-center_y)*(y_pos-center_y) + (z_pos-center_z)*(z_pos-center_z) );
        if ( r < 0.2 ){
          density = overDensity;
          pressure += overPressure;
        }
        v2 = vx*vx + vy*vy + vz*vz;
        energy = pressure/(gama-1) + 0.5*density*v2;
        C.density[id] = density;
        C.momentum_x[id] = density*vx;
        C.momentum_y[id] = density*vy;
        C.momentum_z[id] = density*vz;
        C.Energy[id] = energy;

        #ifdef DE
        C.GasEnergy[id] = pressure/(gama-1);
        #endif
      }
    }
  }
}

/*! \fn void Spherical_Overdensity_3D()
 *  \brief Spherical overdensity for gravitational colapse */
void Grid3D::Spherical_Overdensity_3D()
{
 int i, j, k, id;
 Real x_pos, y_pos, z_pos, r, center_x, center_y, center_z;
 Real density, pressure, overDensity, overPressure, energy, radius, background_density;
 Real vx, vy, vz, v2;
 center_x = 0.5;
 center_y = 0.5;
 center_z = 0.5;
 overDensity = 1;
 overPressure = 0;
 vx = 0;
 vy = 0;
 vz = 0;
 radius = 0.2;
 background_density = 0.0005;
 H.sphere_density = overDensity;
 H.sphere_radius = radius;
 H.sphere_background_density = background_density;
 H.sphere_center_x = center_x;
 H.sphere_center_y = center_y;
 H.sphere_center_z = center_z;

 // set the initial values of the conserved variables
 for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
   for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
     for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
       id = i + j*H.nx + k*H.nx*H.ny;

       // // get the centered cell positions at (i,j,k)
       Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
       density = background_density;
       pressure = 0.0005;

       r = sqrt( (x_pos-center_x)*(x_pos-center_x) + (y_pos-center_y)*(y_pos-center_y) + (z_pos-center_z)*(z_pos-center_z) );
       if ( r < radius ){
         density = overDensity;
         pressure += overPressure;
       }
       v2 = vx*vx + vy*vy + vz*vz;
       energy = pressure/(gama-1) + 0.5*density*v2;
       C.density[id] = density;
       C.momentum_x[id] = density*vx;
       C.momentum_y[id] = density*vy;
       C.momentum_z[id] = density*vz;
       C.Energy[id] = energy;

       #ifdef DE
       C.GasEnergy[id] = pressure/(gama-1);
       #endif
     }
   }
 }
}


void Grid3D::Uniform_Grid()
{
  chprintf( " Initializing Uniform Grid\n");
  int i, j, k, id;
  // set the initial values of the conserved variables
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;

        C.density[id] = 0;
        C.momentum_x[id] = 0;
        C.momentum_y[id] = 0;
        C.momentum_z[id] = 0;
        C.Energy[id] = 0;

        #ifdef DE
        C.GasEnergy[id] = 0;
        #endif
      }
    }
  }
}

void Grid3D::Zeldovich_Pancake( struct parameters P ){

  #ifndef COSMOLOGY
  chprintf( "To run a Zeldovich Pancake COSMOLOGY has to be turned ON \n" );
  exit(-1);
  #else


  int i, j, k, id;
  Real x_pos, y_pos, z_pos;
  Real H0, h, Omega_M, rho_0, G, z_zeldovich, z_init, x_center, T_init, k_x;

  chprintf("Setting Zeldovich Pancake initial conditions...\n");
  H0 = P.H0;
  h = H0 / 100;
  Omega_M = P.Omega_M;

  chprintf( " h = %f \n", h );
  chprintf( " Omega_M = %f \n", Omega_M );

  H0 /= 1000;               //[km/s / kpc]
  G = G_COSMO;
  rho_0 = 3*H0*H0 / ( 8*M_PI*G ) * Omega_M /h / h;
  z_zeldovich = 1;
  z_init = P.Init_redshift;
  chprintf( " rho_0 = %f \n", rho_0 );
  chprintf( " z_init = %f \n", z_init );
  chprintf( " z_zeldovich = %f \n", z_zeldovich );

  x_center = H.xdglobal / 2;
  chprintf( " Peak Center = %f \n", x_center );

  T_init = 100;
  chprintf( " T initial = %f \n", T_init );

  k_x = 2 * M_PI /  H.xdglobal;


  char filename[100];
  // create the filename to read from
  strcpy(filename, P.indir);
  strcat(filename, "ics_zeldovich.dat");
  chprintf( " Loading ICs File: %s\n", filename);

  real_vector_t ics_values;

  ifstream file_in( filename );
  string line;
  Real ic_val;
  if (file_in.is_open()){
    while ( getline (file_in, line) ){
      ic_val = atof( line.c_str() );
      ics_values.push_back( ic_val );
      // chprintf("%f\n", ic_val);
    }
    file_in.close();
  }
  else{
    chprintf("  Error: Unable to open ics zeldovich file\n");
    exit(1);
  }
  int nPoints = 256;



  Real dens, vel, temp, U, E, gamma;
  gamma = P.gamma;

  int index;
  // set the initial values of the conserved variables
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;

        // // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        //Analytical Initial Conditions
        // dens = rho_0 / ( 1 - ( 1 + z_zeldovich ) / ( 1 + z_init ) * cos( k_x*( x_pos - x_center )) );
        // vel = - H0 * ( 1 + z_zeldovich ) / sqrt( 1 + z_init ) * sin( k_x*( x_pos - x_center )) / k_x;
        // temp = T_init * pow( dens / rho_0, 2./3 );
        // U = temp / (gamma - 1) / MP * KB * 1e-10 * dens;
        // E = 0.5 * dens * vel * vel + U;


        index = (int( x_pos / H.dx ) + 0 ) %256;
        // index = ( index + 16 ) % 256;
        dens = ics_values[ 0*nPoints + index];
        vel = ics_values[ 1*nPoints + index];
        E = ics_values[ 2*nPoints + index];
        U = ics_values[ 3*nPoints + index];
        // //

        // chprintf( "%f \n", vel );
        C.density[id] = dens;
        C.momentum_x[id] = dens * vel;
        C.momentum_y[id] = 0;
        C.momentum_z[id] = 0;
        C.Energy[id] = E;

        #ifdef DE
        C.GasEnergy[id] = U ;
        #endif

      }
    }
  }

  #endif //COSMOLOGY

}










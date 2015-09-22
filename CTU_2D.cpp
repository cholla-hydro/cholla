/*! \file CTU_2D.cpp
 *  \brief Definitions of the 2D CTU algorithm. See Sec. 4.2 of Gardiner & Stone, 2008. */

#ifndef CUDA

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"global.h"
#include"CTU_2D.h"
#include"plmp.h"
#include"plmc.h"
#include"ppmp.h"
#include"ppmc.h"
#include"exact.h"
#include"roe.h"


/*! \fn CTU_Algorithm_2D(Real *C, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt)
 *! \brief The corner transport upwind algorithm of Gardiner & Stone, 2008. */
void CTU_Algorithm_2D(Real *C, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt)
{
  int n_cells = nx*ny;
  // Create structures to hold the initial input states and associated interface fluxes (Q* and F* from Stone, 2008)
  States_2D Q1(n_cells);
  Fluxes_2D F1(n_cells);
  // and the states and fluxes after half a timestep of evolution (Q n+1/2 and F n+1/2)
  States_2D Q2(n_cells);
  Fluxes_2D F2(n_cells);

  // Create arrays to hold the eta values for the H correction
  Real *eta_x = (Real *) malloc(n_cells*sizeof(Real));
  Real *eta_y = (Real *) malloc(n_cells*sizeof(Real));
  // Set H correction to zero
  Real etah = 0.0;


  // Set up pointers to the appropriate locations in C to read and modify values of conserved variables
  Real *density    = &C[0];
  Real *momentum_x = &C[n_cells];
  Real *momentum_y = &C[2*n_cells];
  Real *momentum_z = &C[3*n_cells];
  Real *Energy     = &C[4*n_cells];

  // Declare iteration variables
  int i, j;
  int istart, istop, jstart, jstop;
  // index of each cell
  int id;
  
  Real dtodx = dt/dx;
  Real dtody = dt/dy;


  // Step 1: Calculate the left and right states at each cell interface

  #ifdef PCM
  // sweep through cells and use cell averages to set input states
  // do the calculation for all interfaces
  // the new left and right states for each i+1/2 interface are assigned to cell i

  istart = 0; istop = nx-1;
  jstart = 0; jstop = ny;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {
      id = i + j*nx;
      Q1.d_Lx[i + j*nx]  = density[id];
      Q1.mx_Lx[i + j*nx] = momentum_x[id];
      Q1.my_Lx[i + j*nx] = momentum_y[id];
      Q1.mz_Lx[i + j*nx] = momentum_z[id];
      Q1.E_Lx[i + j*nx]  = Energy[id];
      id = (i+1) + j*nx;
      Q1.d_Rx[i + j*nx]  = density[id];
      Q1.mx_Rx[i + j*nx] = momentum_x[id];
      Q1.my_Rx[i + j*nx] = momentum_y[id];
      Q1.mz_Rx[i + j*nx] = momentum_z[id];
      Q1.E_Rx[i + j*nx]  = Energy[id];
    }
  }
  istart = 0; istop = nx;
  jstart = 0; jstop = ny-1;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {
      id = i + j*nx;
      Q1.d_Ly[i + j*nx]  = density[id];
      Q1.mx_Ly[i + j*nx] = momentum_x[id];
      Q1.my_Ly[i + j*nx] = momentum_y[id];
      Q1.mz_Ly[i + j*nx] = momentum_z[id];
      Q1.E_Ly[i + j*nx]  = Energy[id];
      id = i + (j+1)*nx;
      Q1.d_Ry[i + j*nx]  = density[id];
      Q1.mx_Ry[i + j*nx] = momentum_x[id];
      Q1.my_Ry[i + j*nx] = momentum_y[id];
      Q1.mz_Ry[i + j*nx] = momentum_z[id];        
      Q1.E_Ry[i + j*nx]  = Energy[id];
    }
  }
  #endif //PCM


  #if defined (PLMP) || defined (PLMC)
  // sweep through cells, use the piecewise linear method to calculate reconstructed boundary values

  // create the stencil of conserved variables needed to calculate the boundary values 
  // on either side of the cell interface 
  Real stencil[15];

  // create array to hold the boundary values 
  // returned from the linear reconstruction function (conserved variables)
  Real bounds[10];

  // x direction
  istart = n_ghost-1; istop = nx-n_ghost+1;
  for (j=0; j<ny; j++) {
    for (i=istart; i<istop; i++) {

      // fill the stencil for the x-direction
      id = i + j*nx;
      stencil[0] = density[id]; 
      stencil[1] = momentum_x[id];
      stencil[2] = momentum_y[id];
      stencil[3] = momentum_z[id];
      stencil[4] = Energy[id];
      id = (i-1) + j*nx;
      stencil[5] = density[id];
      stencil[6] = momentum_x[id];
      stencil[7] = momentum_y[id];
      stencil[8] = momentum_z[id];
      stencil[9] = Energy[id];
      id = (i+1) + j*nx;
      stencil[10] = density[id];
      stencil[11] = momentum_x[id];
      stencil[12] = momentum_y[id];
      stencil[13] = momentum_z[id];
      stencil[14] = Energy[id];

      // pass the stencil to the linear reconstruction function - returns the reconstructed left
      // and right boundary values for the cell (conserved variables)
      #ifdef PLMP
      plmp(stencil, bounds, dx, dt, gama);
      #endif
      #ifdef PLMC
      plmc(stencil, bounds, dx, dt, gama);
      //lr_states(stencil, bounds, dx, dt, gama);
      #endif


      // place the boundary values in the relevant array
      // the reconstruction function returns l & r for cell i, i.e. |L i R|
      // for compatibility with ppm, switch this so that the input states for the i+1/2 interface
      // are associated with the ith cell
      id = (i-1) + j*nx;
      Q1.d_Rx[id]  = bounds[0];
      Q1.mx_Rx[id] = bounds[1];
      Q1.my_Rx[id] = bounds[2];
      Q1.mz_Rx[id] = bounds[3];
      Q1.E_Rx[id]  = bounds[4];
      id = i + j*nx;
      Q1.d_Lx[id]  = bounds[5];
      Q1.mx_Lx[id] = bounds[6];
      Q1.my_Lx[id] = bounds[7];
      Q1.mz_Lx[id] = bounds[8];
      Q1.E_Lx[id]  = bounds[9];
      // now L&R correspond to left and right of the interface, i.e. L | R

    }
  }


  // y-direction
  jstart = n_ghost-1; jstop = ny-n_ghost+1;
  for (j=jstart; j<jstop; j++) {
    for (i=0; i<nx; i++) {

      // fill the stencil for the y direction
      id = i + j*nx;
      stencil[0] = density[id]; 
      stencil[1] = momentum_y[id];
      stencil[2] = momentum_z[id];
      stencil[3] = momentum_x[id];
      stencil[4] = Energy[id];
      id = i + (j-1)*nx;
      stencil[5] = density[id];
      stencil[6] = momentum_y[id];
      stencil[7] = momentum_z[id];
      stencil[8] = momentum_x[id];
      stencil[9] = Energy[id];
      id = i + (j+1)*nx;
      stencil[10] = density[id];
      stencil[11] = momentum_y[id];
      stencil[12] = momentum_z[id];
      stencil[13] = momentum_x[id];
      stencil[14] = Energy[id];

      // pass the stencil to the linear reconstruction function - returns the reconstructed left
      // and right boundary values for the cell (conserved variables)
      #ifdef PLMP
      plmp(stencil, bounds, dy, dt, gama);
      #endif
      #ifdef PLMC
      plmc(stencil, bounds, dy, dt, gama);
      #endif

      // place the boundary values in the relevant array
      id = i + (j-1)*nx;
      Q1.d_Ry[id]  = bounds[0];
      Q1.my_Ry[id] = bounds[1];
      Q1.mz_Ry[id] = bounds[2];
      Q1.mx_Ry[id] = bounds[3];
      Q1.E_Ry[id]  = bounds[4];
      id = i + j*nx;
      Q1.d_Ly[id]  = bounds[5];
      Q1.my_Ly[id] = bounds[6];
      Q1.mz_Ly[id] = bounds[7];
      Q1.mx_Ly[id] = bounds[8];
      Q1.E_Ly[id]  = bounds[9];
    }
  }

  #endif //PLM or PLM_ATHENA


  #if defined (PPMP) || defined (PPMC)
  // sweep through cells, use PPM to calculate reconstructed boundary values

  // get start time
  //double start_ppm, stop_ppm;
  //start_ppm = get_time();

  // create the stencil of conserved variables needed to calculate the boundary values 
  // on either side of the cell interface 
  #ifdef PPMP
  Real stencil[35];
  #endif
  #ifdef PPMC
  Real stencil[25];
  #endif

  // create an array to hold the intermediate boundary values 
  // returned from the ppm function (left and right, for d, mx, my, mz, E)
  Real bounds[10];


  // x-direction
  istart = n_ghost-2; istop = nx-n_ghost+2;
  jstart = 0; jstop = ny;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      // fill the stencil for the x-direction
      id = i + j*nx;
      stencil[0]  = density[id];
      stencil[1]  = momentum_x[id];
      stencil[2]  = momentum_y[id];
      stencil[3]  = momentum_z[id];
      stencil[4]  = Energy[id];
      id = (i-1) + j*nx;
      stencil[5]  = density[id];
      stencil[6]  = momentum_x[id];
      stencil[7]  = momentum_y[id];
      stencil[8]  = momentum_z[id];
      stencil[9]  = Energy[id];
      id = (i+1) + j*nx;
      stencil[10] = density[id];
      stencil[11] = momentum_x[id];
      stencil[12] = momentum_y[id];
      stencil[13] = momentum_z[id];
      stencil[14] = Energy[id];
      id = (i-2) + j*nx;
      stencil[15] = density[id];
      stencil[16] = momentum_x[id];
      stencil[17] = momentum_y[id];
      stencil[18] = momentum_z[id];
      stencil[19] = Energy[id];
      id = (i+2) + j*nx;
      stencil[20] = density[id];
      stencil[21] = momentum_x[id];
      stencil[22] = momentum_y[id];
      stencil[23] = momentum_z[id];
      stencil[24] = Energy[id];
      #ifdef PPMP
      id = (i-3) + j*nx;
      stencil[25] = density[id];
      stencil[26] = momentum_x[id];
      stencil[27] = momentum_y[id];
      stencil[28] = momentum_z[id];
      stencil[29] = Energy[id];
      id = (i+3) + j*nx;
      stencil[30] = density[id];
      stencil[31] = momentum_x[id];
      stencil[32] = momentum_y[id];
      stencil[33] = momentum_z[id];
      stencil[34] = Energy[id];
      #endif

      // pass the stencil to the ppm reconstruction function - returns the reconstructed left
      // and right boundary values (conserved variables)
      #ifdef PPMP
      ppmp(stencil, bounds, dx, dt, gama);
      #endif
      #ifdef PPMC
      ppmc(stencil, bounds, dx, dt, gama);
      #endif

      // place the boundary values in the relevant array
      // at this point, L&R correspond to left and right of the interface i.e. L|R
      id = i-1 + j*nx;
      Q1.d_Rx[id]  = bounds[0];
      Q1.mx_Rx[id] = bounds[1];
      Q1.my_Rx[id] = bounds[2];
      Q1.mz_Rx[id] = bounds[3];
      Q1.E_Rx[id]  = bounds[4];
      id = i + j*nx;
      Q1.d_Lx[id]  = bounds[5];
      Q1.mx_Lx[id] = bounds[6];
      Q1.my_Lx[id] = bounds[7];
      Q1.mz_Lx[id] = bounds[8];
      Q1.E_Lx[id]  = bounds[9];

    }
  }

  // y-direction
  istart = 0; istop = nx;
  jstart = n_ghost-2; jstop = ny-n_ghost+2;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      // fill the stencil for the y direction
      id = i + j*nx;
      stencil[0]  = density[id];
      stencil[1]  = momentum_y[id];
      stencil[2]  = momentum_z[id];
      stencil[3]  = momentum_x[id];
      stencil[4]  = Energy[id];
      id = i + (j-1)*nx;
      stencil[5]  = density[id];
      stencil[6]  = momentum_y[id];
      stencil[7]  = momentum_z[id];
      stencil[8]  = momentum_x[id];
      stencil[9]  = Energy[id];
      id = i + (j+1)*nx;
      stencil[10] = density[id];
      stencil[11] = momentum_y[id];
      stencil[12] = momentum_z[id];
      stencil[13] = momentum_x[id];
      stencil[14] = Energy[id];
      id = i + (j-2)*nx;
      stencil[15] = density[id];
      stencil[16] = momentum_y[id];
      stencil[17] = momentum_z[id];
      stencil[18] = momentum_x[id];
      stencil[19] = Energy[id];
      id = i + (j+2)*nx;
      stencil[20] = density[id];
      stencil[21] = momentum_y[id];
      stencil[22] = momentum_z[id];
      stencil[23] = momentum_x[id];
      stencil[24] = Energy[id];
      #ifdef PPMP
      id = i + (j-3)*nx;
      stencil[25] = density[id];
      stencil[26] = momentum_y[id];
      stencil[27] = momentum_z[id];
      stencil[28] = momentum_x[id];
      stencil[29] = Energy[id];
      id = i + (j+3)*nx;
      stencil[30] = density[id];
      stencil[31] = momentum_y[id];
      stencil[32] = momentum_z[id];
      stencil[33] = momentum_x[id];
      stencil[34] = Energy[id];
      #endif
        
      // pass the stencil to the ppm reconstruction function - returns the reconstructed left
      // and right boundary values (conserved variables)
      #ifdef PPMP
      ppmp(stencil, bounds, dy, dt, gama);
      #endif
      #ifdef PPMC
      ppmc(stencil, bounds, dy, dt, gama);
      #endif

      // place the boundary values in the relevant array
      id = i + (j-1)*nx;
      Q1.d_Ry[id]  = bounds[0];
      Q1.my_Ry[id] = bounds[1];
      Q1.mz_Ry[id] = bounds[2];
      Q1.mx_Ry[id] = bounds[3];
      Q1.E_Ry[id]  = bounds[4];
      id = i + j*nx;
      Q1.d_Ly[id]  = bounds[5];
      Q1.my_Ly[id] = bounds[6];
      Q1.mz_Ly[id] = bounds[7];
      Q1.mx_Ly[id] = bounds[8];
      Q1.E_Ly[id]  = bounds[9];

    }
  }

  // get stop time
  //stop_ppm = get_time();
  //printf("ppm time = %9.3f ms\n", (stop_ppm-start_ppm)*1000);

  #endif //PPMP or PPMC


  // Step 2: Using the input states, compute the 1D fluxes at each interface.
  // Only do this for interfaces touching real cells (start at i = n_ghost-1 since
  // flux for the i+1/2 interface is stored by cell i)

  // get start time
  //double start_riemann1, stop_riemann1;
  //start_riemann1 = get_time();

  // Create arrays to hold the input states for the Riemann solver and the returned fluxes
  Real cW[10];
  Real flux[5];

  // Solve the Riemann problem at each x-interface
  // do the calculation for all the real interfaces in the x direction
  // and one ghost cell on either side in the y direction
  istart = n_ghost-2; istop = nx-n_ghost+1;
  jstart = n_ghost-2; jstop = ny-n_ghost+2; 
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      // set input variables for the x interfaces
      // exact Riemann solver takes conserved variables
      id = i + j*nx;
      cW[0] = Q1.d_Lx[id];
      cW[1] = Q1.d_Rx[id];
      cW[2] = Q1.mx_Lx[id]; 
      cW[3] = Q1.mx_Rx[id];
      cW[4] = Q1.my_Lx[id];
      cW[5] = Q1.my_Rx[id];
      cW[6] = Q1.mz_Lx[id];
      cW[7] = Q1.mz_Rx[id];
      cW[8] = Q1.E_Lx[id];
      cW[9] = Q1.E_Rx[id];

      // call a Riemann solver to evaluate fluxes at the cell interface
      #ifdef EXACT
      Calculate_Exact_Fluxes(cW, flux, gama);
      #endif
      #ifdef ROE
      Calculate_Roe_Fluxes(cW, flux, gama, etah);
      #endif

      // update the fluxes in the x-direction
      F1.dflux_x[id] = flux[0];
      F1.xmflux_x[id] = flux[1];
      F1.ymflux_x[id] = flux[2];
      F1.zmflux_x[id] = flux[3];
      F1.Eflux_x[id] = flux[4];

    }
  }

  // Solve the Riemann problem at each y-interface
  // do the calculation for all the real interfaces in the y direction
  // and one ghost cell on either side in the x direction
  istart = n_ghost-2; istop = nx-n_ghost+2;
  jstart = n_ghost-2; jstop = ny-n_ghost+1;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      // set input variables for the y interfaces
      id = i + j*nx;
      cW[0] = Q1.d_Ly[id];
      cW[1] = Q1.d_Ry[id];
      cW[2] = Q1.my_Ly[id]; 
      cW[3] = Q1.my_Ry[id];
      cW[4] = Q1.mz_Ly[id];
      cW[5] = Q1.mz_Ry[id];
      cW[6] = Q1.mx_Ly[id];
      cW[7] = Q1.mx_Ry[id];
      cW[8] = Q1.E_Ly[id];
      cW[9] = Q1.E_Ry[id];

      // call a Riemann solver to evaluate fluxes at the cell interface
      #ifdef EXACT
      Calculate_Exact_Fluxes(cW, flux, gama);
      #endif
      #ifdef ROE
      Calculate_Roe_Fluxes(cW, flux, gama, etah);
      #endif

      // update the fluxes in the y-direction
      F1.dflux_y[id] = flux[0];
      F1.ymflux_y[id] = flux[1];
      F1.zmflux_y[id] = flux[2];
      F1.xmflux_y[id] = flux[3];
      F1.Eflux_y[id] = flux[4];

    }
  }

  // get stop time
  //stop_riemann1 = get_time();
  //printf("riemann 1 time = %9.3f ms\n", (stop_riemann1-start_riemann1)*1000);


  // Step 3: Apply the CT algorithm to calculate the CT electric fields at cell-corners
  // (not applicable if not doing MHD)

  // Step 4: Update the face-centered magnetic field by dt/2 using EMFs computed in step 3
  // (not applicable if not doing MHD)

  // Step 5: Evolve the left and right states at each interface by dt/2 using transverse flux gradients
  // Only do this for interfaces bordering real cells (start at i = n_ghost-1 since
  // flux for the i+1/2 interface is stored by cell i)

  // get start time
  //double start_evolution, stop_evolution;
  //start_evolution = get_time();

  // Evolve the x-interface states
  // do the calculation for all the real interfaces in the x direction
  istart = n_ghost-2; istop = nx-n_ghost+1;
  jstart = n_ghost-1; jstop = ny-n_ghost+1;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      // density
      Q2.d_Lx[i+j*nx] = Q1.d_Lx[i+j*nx] + 0.5*dtody*(F1.dflux_y[ i   +(j-1)*nx] - F1.dflux_y[ i   +j*nx]);
      Q2.d_Rx[i+j*nx] = Q1.d_Rx[i+j*nx] + 0.5*dtody*(F1.dflux_y[(i+1)+(j-1)*nx] - F1.dflux_y[(i+1)+j*nx]);

      // x-momentum
      Q2.mx_Lx[i+j*nx] = Q1.mx_Lx[i+j*nx] + 0.5*dtody*(F1.xmflux_y[ i   +(j-1)*nx] - F1.xmflux_y[ i   +j*nx]);
      Q2.mx_Rx[i+j*nx] = Q1.mx_Rx[i+j*nx] + 0.5*dtody*(F1.xmflux_y[(i+1)+(j-1)*nx] - F1.xmflux_y[(i+1)+j*nx]);

      // y-momentum
      Q2.my_Lx[i+j*nx] = Q1.my_Lx[i+j*nx] + 0.5*dtody*(F1.ymflux_y[ i   +(j-1)*nx] - F1.ymflux_y[ i   +j*nx]);
      Q2.my_Rx[i+j*nx] = Q1.my_Rx[i+j*nx] + 0.5*dtody*(F1.ymflux_y[(i+1)+(j-1)*nx] - F1.ymflux_y[(i+1)+j*nx]);
        
      // z-momentum
      Q2.mz_Lx[i+j*nx] = Q1.mz_Lx[i+j*nx] + 0.5*dtody*(F1.zmflux_y[ i   +(j-1)*nx] - F1.zmflux_y[ i   +j*nx]);
      Q2.mz_Rx[i+j*nx] = Q1.mz_Rx[i+j*nx] + 0.5*dtody*(F1.zmflux_y[(i+1)+(j-1)*nx] - F1.zmflux_y[(i+1)+j*nx]);

      // Energy
      Q2.E_Lx[i+j*nx] = Q1.E_Lx[i+j*nx] + 0.5*dtody*(F1.Eflux_y[ i   +(j-1)*nx] - F1.Eflux_y[ i   +j*nx]);
      Q2.E_Rx[i+j*nx] = Q1.E_Rx[i+j*nx] + 0.5*dtody*(F1.Eflux_y[(i+1)+(j-1)*nx] - F1.Eflux_y[(i+1)+j*nx]);

    }
  }

  // Evolve the y-interface states
  // do the calculation for all the real interfaces in the y direction
  istart = n_ghost-1; istop = nx-n_ghost+1;
  jstart = n_ghost-2; jstop = ny-n_ghost+1;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      // density
      Q2.d_Ly[i+j*nx] = Q1.d_Ly[i+j*nx] + 0.5*dtodx*(F1.dflux_x[(i-1)+ j   *nx] - F1.dflux_x[i+ j   *nx]);
      Q2.d_Ry[i+j*nx] = Q1.d_Ry[i+j*nx] + 0.5*dtodx*(F1.dflux_x[(i-1)+(j+1)*nx] - F1.dflux_x[i+(j+1)*nx]);

      // x-momentum
      Q2.mx_Ly[i+j*nx] = Q1.mx_Ly[i+j*nx] + 0.5*dtodx*(F1.xmflux_x[(i-1)+ j   *nx] - F1.xmflux_x[i+ j   *nx]);
      Q2.mx_Ry[i+j*nx] = Q1.mx_Ry[i+j*nx] + 0.5*dtodx*(F1.xmflux_x[(i-1)+(j+1)*nx] - F1.xmflux_x[i+(j+1)*nx]);

      // y-momentum
      Q2.my_Ly[i+j*nx] = Q1.my_Ly[i+j*nx] + 0.5*dtodx*(F1.ymflux_x[(i-1)+ j   *nx] - F1.ymflux_x[i+ j   *nx]);
      Q2.my_Ry[i+j*nx] = Q1.my_Ry[i+j*nx] + 0.5*dtodx*(F1.ymflux_x[(i-1)+(j+1)*nx] - F1.ymflux_x[i+(j+1)*nx]);

      // z-momentum
      Q2.mz_Ly[i+j*nx] = Q1.mz_Ly[i+j*nx] + 0.5*dtodx*(F1.zmflux_x[(i-1)+ j   *nx] - F1.zmflux_x[i+ j   *nx]);
      Q2.mz_Ry[i+j*nx] = Q1.mz_Ry[i+j*nx] + 0.5*dtodx*(F1.zmflux_x[(i-1)+(j+1)*nx] - F1.zmflux_x[i+(j+1)*nx]);

      // Energy
      Q2.E_Ly[i+j*nx] = Q1.E_Ly[i+j*nx] + 0.5*dtodx*(F1.Eflux_x[(i-1)+ j   *nx] - F1.Eflux_x[i+ j   *nx]);
      Q2.E_Ry[i+j*nx] = Q1.E_Ry[i+j*nx] + 0.5*dtodx*(F1.Eflux_x[(i-1)+(j+1)*nx] - F1.Eflux_x[i+(j+1)*nx]);  

    }
  }
  // get end time
  //stop_evolution = get_time();
  //printf("evolution time = %9.3f ms\n", (stop_evolution-start_evolution)*1000);  


  // Step 6: Calculate a cell-centered electric field at t^n+1/2 
  // (not applicable if not doing MHD)


  #ifdef H_CORRECTION
  // Step 6 1/2: Calculate the eta values for the H correction of Sanders et al., 1998
  // do the calculation for all the real interfaces in the x direction
  istart = n_ghost-2; istop = nx-n_ghost+1;
  jstart = n_ghost-1; jstop = ny-n_ghost+1;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      // set input variables for the x interfaces
      id = i + j*nx;
      cW[0] = Q2.d_Lx[id];
      cW[1] = Q2.d_Rx[id];
      cW[2] = Q2.mx_Lx[id]; 
      cW[3] = Q2.mx_Rx[id];
      cW[4] = Q2.my_Lx[id];
      cW[5] = Q2.my_Rx[id];
      cW[6] = Q2.mz_Lx[id];
      cW[7] = Q2.mz_Rx[id];
      cW[8] = Q2.E_Lx[id];
      cW[9] = Q2.E_Rx[id];
     
      eta_x[id] = calc_eta(cW, gama);

    }
  }
  // do the calculation for all the real y interfaces 
  istart = n_ghost-1; istop = nx-n_ghost+1;
  jstart = n_ghost-2; jstop = ny-n_ghost+1;
  for (i=istart; i<istop; i++) {
    for (j=jstart; j<jstop; j++) {

      // set input variables for the y interfaces
      id = i + j*nx;
      cW[0] = Q2.d_Ly[id];
      cW[1] = Q2.d_Ry[id];
      cW[2] = Q2.my_Ly[id]; 
      cW[3] = Q2.my_Ry[id];
      cW[4] = Q2.mz_Ly[id];
      cW[5] = Q2.mz_Ry[id];
      cW[6] = Q2.mx_Ly[id];
      cW[7] = Q2.mx_Ry[id];
      cW[8] = Q2.E_Ly[id];
      cW[9] = Q2.E_Ry[id];

      eta_y[id] = calc_eta(cW, gama);

    }
  }
  #endif // H_CORRECTION


  // Step 7: Compute new fluxes at cell interfaces using the corrected left and right
  // states from step 5, and the interface magnetic fields computed in step 4.
  // Again, only do this for interfaces bordering real cells (start at i = n_ghost-1 since
  // flux for the i+1/2 interface is stored by cell i).

  // get start time
  //double start_riemann2, stop_riemann2;
  //start_riemann2 = get_time();

  // Solve the Riemann problem at each x-interface
  // do the calculation for all the real x interfaces
  istart = n_ghost-1; istop = nx-n_ghost;
  jstart = n_ghost-1; jstop = ny-n_ghost+1;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      // set input variables for the x interfaces
      // exact Riemann solver takes conserved variables
      id = i + j*nx;
      cW[0] = Q2.d_Lx[id];
      cW[1] = Q2.d_Rx[id];
      cW[2] = Q2.mx_Lx[id]; 
      cW[3] = Q2.mx_Rx[id];
      cW[4] = Q2.my_Lx[id];
      cW[5] = Q2.my_Rx[id];
      cW[6] = Q2.mz_Lx[id];
      cW[7] = Q2.mz_Rx[id];
      cW[8] = Q2.E_Lx[id];
      cW[9] = Q2.E_Rx[id];

      #ifdef H_CORRECTION
      etah = fmax(eta_y[i + (j-1)*nx], eta_y[i + j*nx]);
      etah = fmax(etah, eta_x[id]);
      etah = fmax(etah, eta_y[i+1 + (j-1)*nx]);
      etah = fmax(etah, eta_y[i+1 + j*nx]);
      #endif

      // call a Riemann solver to evaluate fluxes at the cell interface
      #ifdef EXACT
      Calculate_Exact_Fluxes(cW, flux, gama);
      #endif
      #ifdef ROE
      Calculate_Roe_Fluxes(cW, flux, gama, etah);
      #endif

      // update the fluxes in the x-direction
      F2.dflux_x[id] = flux[0];
      F2.xmflux_x[id] = flux[1];
      F2.ymflux_x[id] = flux[2];
      F2.zmflux_x[id] = flux[3];
      F2.Eflux_x[id] = flux[4];
    }
  }

  // Solve the Riemann problem at each y-interface
  // do the calculation for all the real y interfaces 
  istart = n_ghost-1; istop = nx-n_ghost+1;
  jstart = n_ghost-1; jstop = ny-n_ghost;
  for (i=istart; i<istop; i++) {
    for (j=jstart; j<jstop; j++) {

      // set input variables for the y interfaces
      id = i + j*nx;
      cW[0] = Q2.d_Ly[id];
      cW[1] = Q2.d_Ry[id];
      cW[2] = Q2.my_Ly[id]; 
      cW[3] = Q2.my_Ry[id];
      cW[4] = Q2.mz_Ly[id];
      cW[5] = Q2.mz_Ry[id];
      cW[6] = Q2.mx_Ly[id];
      cW[7] = Q2.mx_Ry[id];
      cW[8] = Q2.E_Ly[id];
      cW[9] = Q2.E_Ry[id];

      #ifdef H_CORRECTION
      etah = fmax(eta_x[i-1 + j*nx], eta_x[i + j*nx]);
      etah = fmax(etah, eta_y[id]);
      etah = fmax(etah, eta_x[i-1 + (j+1)*nx]);
      etah = fmax(etah, eta_x[i + (j+1)*nx]);
      #endif

      // call a Riemann solver to evaluate fluxes at the cell interface
      #ifdef EXACT
      Calculate_Exact_Fluxes(cW, flux, gama);
      #endif
      #ifdef ROE
      Calculate_Roe_Fluxes(cW, flux, gama, etah);
      #endif

      // update the fluxes in the y-direction
      F2.dflux_y[id] = flux[0];
      F2.ymflux_y[id] = flux[1];
      F2.zmflux_y[id] = flux[2];
      F2.xmflux_y[id] = flux[3];
      F2.Eflux_y[id] = flux[4];

    } 
  }

  // get stop time
  //stop_riemann2 = get_time();
  //printf("riemann 2 time = %9.3f ms\n", (stop_riemann2-start_riemann2)*1000);  


  // Step 8: Apply the CT algorithm to calculate the CT electric fields
  // Not applicable if not doing MHD


  // Step 9: Update the solution from time level n to n+1

  // get start time
  //double start_update, stop_update;
  //start_update = get_time();

  // Only update real cells
  istart = n_ghost; istop = nx-n_ghost;
  jstart = n_ghost; jstop = ny-n_ghost;
  for (j=jstart; j<jstop; j++) {
    for (i=istart; i<istop; i++) {

      density[   i+j*nx] += dtodx*(F2.dflux_x[ (i-1)+j*nx] - F2.dflux_x[ i+j*nx]) + dtody*(F2.dflux_y[ i+(j-1)*nx] - F2.dflux_y[ i+j*nx]); 
      momentum_x[i+j*nx] += dtodx*(F2.xmflux_x[(i-1)+j*nx] - F2.xmflux_x[i+j*nx]) + dtody*(F2.xmflux_y[i+(j-1)*nx] - F2.xmflux_y[i+j*nx]);
      momentum_y[i+j*nx] += dtodx*(F2.ymflux_x[(i-1)+j*nx] - F2.ymflux_x[i+j*nx]) + dtody*(F2.ymflux_y[i+(j-1)*nx] - F2.ymflux_y[i+j*nx]);
      momentum_z[i+j*nx] += dtodx*(F2.zmflux_x[(i-1)+j*nx] - F2.zmflux_x[i+j*nx]) + dtody*(F2.zmflux_y[i+(j-1)*nx] - F2.zmflux_y[i+j*nx]); 
      Energy[    i+j*nx] += dtodx*(F2.Eflux_x[ (i-1)+j*nx] - F2.Eflux_x[ i+j*nx]) + dtody*(F2.Eflux_y[ i+(j-1)*nx] - F2.Eflux_y[ i+j*nx]);
    }
  }

  // get stop time
  //stop_update = get_time();
  //printf("update time = %9.3f ms\n", (stop_update-start_update)*1000);  

  // free the interface states and flux structures
  free(Q1.d_Lx);
  free(Q2.d_Lx);
  free(F1.dflux_x);
  free(F2.dflux_x);

  free(eta_x);
  free(eta_y);

}




States_2D::States_2D(int n_cells)
{
  // allocate memory for the interface state arrays (left and right for each interface)
  d_Lx = (Real *) malloc(20*n_cells*sizeof(Real));
  d_Rx = &(d_Lx[n_cells]);
  d_Ly = &(d_Lx[2*n_cells]);
  d_Ry = &(d_Lx[3*n_cells]);
  mx_Lx = &(d_Lx[4*n_cells]);
  mx_Rx = &(d_Lx[5*n_cells]);
  mx_Ly = &(d_Lx[6*n_cells]);
  mx_Ry = &(d_Lx[7*n_cells]);
  my_Lx = &(d_Lx[8*n_cells]);
  my_Rx = &(d_Lx[9*n_cells]);
  my_Ly = &(d_Lx[10*n_cells]);
  my_Ry = &(d_Lx[11*n_cells]);
  mz_Lx = &(d_Lx[12*n_cells]);
  mz_Rx = &(d_Lx[13*n_cells]);
  mz_Ly = &(d_Lx[14*n_cells]);
  mz_Ry = &(d_Lx[15*n_cells]);
  E_Lx = &(d_Lx[16*n_cells]);
  E_Rx = &(d_Lx[17*n_cells]);
  E_Ly = &(d_Lx[18*n_cells]);
  E_Ry = &(d_Lx[19*n_cells]);

  // initialize array
  for (int i=0; i<20*n_cells; i++)
  {
    d_Lx[i] = 0.0;
  }

}


Fluxes_2D::Fluxes_2D(int n_cells)
{
  // allocate memory for flux arrays (x & y for density, momentum, energy)
  dflux_x = (Real *) malloc(10*n_cells*sizeof(Real));
  dflux_y = &(dflux_x[n_cells]);
  xmflux_x = &(dflux_x[2*n_cells]);
  xmflux_y = &(dflux_x[3*n_cells]);
  ymflux_x = &(dflux_x[4*n_cells]);
  ymflux_y = &(dflux_x[5*n_cells]);
  zmflux_x = &(dflux_x[6*n_cells]);
  zmflux_y = &(dflux_x[7*n_cells]);
  Eflux_x = &(dflux_x[8*n_cells]);
  Eflux_y = &(dflux_x[9*n_cells]);

  // initialize array
  for (int i=0; i<10*n_cells; i++)
  {
    dflux_x[i] = 0.0;
  }

}




#endif //no CUDA

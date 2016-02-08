/*! \file CTU_1D.cpp
 *  \brief Definitions of the CTU algorithm. See Sec. 4.2 of Gardiner & Stone, 2008. */
#ifndef CUDA

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"global.h"
#include"CTU_1D.h"
#include"plmp.h"
#include"plmc.h"
#include"ppmp.h"
#include"ppmc.h"
#include"exact.h"
#include"roe.h"


/*! \fn CTU_Algorithm_1D(Real *C, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt)
 *! \brief The corner transport upwind algorithm of Gardiner & Stone, 2008. */
void CTU_Algorithm_1D(Real *C, int nx, int n_ghost, Real dx, Real dt)
{
  int n_cells = nx;
  // Create structures to hold the initial input states and associated interface fluxes (Q* and F* from Stone, 2008)
  States_1D Q1(n_cells);
  Fluxes_1D F1(n_cells);

  // Set up pointers to the appropriate locations in C to read and modify values of conserved variables
  Real *density    = &C[0];
  Real *momentum_x = &C[n_cells];
  Real *momentum_y = &C[2*n_cells];
  Real *momentum_z = &C[3*n_cells];
  Real *Energy     = &C[4*n_cells];

  // set H correction to zero
  Real etah = 0.0;

  // Declare iteration variables
  int i;
  int istart, istop;
  // index of each cell
  int id;
  
  Real dtodx = dt/dx;


  // Step 1: Calculate the left and right states at each cell interface

  #ifdef PCM
  // sweep through cells and use cell averages to set input states
  // do the calculation for all real cells and one ghost cell on either side
  // the new left and right states for each i+1/2 interface are assigned to cell i

  istart = 0; istop = nx-1;
  for (i=istart; i<istop; i++) 
  {
    // piecewise constant reconstruction 
    Q1.d_L[i]  = density[i];
    Q1.mx_L[i] = momentum_x[i];
    Q1.my_L[i] = momentum_y[i];
    Q1.mz_L[i] = momentum_z[i];
    Q1.E_L[i]  = Energy[i];
    Q1.d_R[i]  = density[(i+1)];
    Q1.mx_R[i] = momentum_x[(i+1)];
    Q1.my_R[i] = momentum_y[(i+1)];
    Q1.mz_R[i] = momentum_z[(i+1)];
    Q1.E_R[i]  = Energy[(i+1)];
  }
  #endif //PCM


  #if defined (PLMP) || defined (PLMC)
  // sweep through cells, use the piecewise linear method to calculate reconstructed boundary values

  // create the stencil of conserved variables needed to calculate the boundary values 
  // on either sie of the cell interface 
  Real stencil[15];

  // create array to hold the boundary values 
  // returned from the linear reconstruction function (conserved variables)
  Real bounds[10];

  // x direction
  istart = n_ghost-1; istop = nx-n_ghost+1;
  for (i=istart; i<istop; i++)
  {
    // fill the stencil for the x-direction
    id = i;
    stencil[0] = density[id]; 
    stencil[1] = momentum_x[id];
    stencil[2] = momentum_y[id];
    stencil[3] = momentum_z[id];
    stencil[4] = Energy[id];
    id = i-1;
    stencil[5] = density[id];
    stencil[6] = momentum_x[id];
    stencil[7] = momentum_y[id];
    stencil[8] = momentum_z[id];
    stencil[9] = Energy[id];
    id = i+1;
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
    id = i-1;
    Q1.d_R[id]  = bounds[0];
    Q1.mx_R[id] = bounds[1];
    Q1.my_R[id] = bounds[2];
    Q1.mz_R[id] = bounds[3];
    Q1.E_R[id]  = bounds[4];
    id = i;
    Q1.d_L[id]    = bounds[5];
    Q1.mx_L[id]   = bounds[6];
    Q1.my_L[id]   = bounds[7];
    Q1.mz_L[id]   = bounds[8];
    Q1.E_L[id]    = bounds[9];
    // now L&R correspond to left and right of the interface, i.e. L | R

  }

  #endif //PLMP or PLMC


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
  istart = n_ghost-1; istop = nx-n_ghost+1;
  for (i=istart; i<istop; i++)
  {
    // fill the stencil for the x-direction
    id = i;
    stencil[0]  = density[id];
    stencil[1]  = momentum_x[id];
    stencil[2]  = momentum_y[id];
    stencil[3]  = momentum_z[id];
    stencil[4]  = Energy[id];
    id = i-1;
    stencil[5]  = density[id];
    stencil[6]  = momentum_x[id];
    stencil[7]  = momentum_y[id];
    stencil[8]  = momentum_z[id];
    stencil[9]  = Energy[id];
    id = i+1;
    stencil[10] = density[id];
    stencil[11] = momentum_x[id];
    stencil[12] = momentum_y[id];
    stencil[13] = momentum_z[id];
    stencil[14] = Energy[id];
    id = i-2;
    stencil[15] = density[id];
    stencil[16] = momentum_x[id];
    stencil[17] = momentum_y[id];
    stencil[18] = momentum_z[id];
    stencil[19] = Energy[id];
    id = i+2;
    stencil[20] = density[id];
    stencil[21] = momentum_x[id];
    stencil[22] = momentum_y[id];
    stencil[23] = momentum_z[id];
    stencil[24] = Energy[id];
    #ifdef PPMP
    id = i-3;
    stencil[25] = density[id];
    stencil[26] = momentum_x[id];
    stencil[27] = momentum_y[id];
    stencil[28] = momentum_z[id];
    stencil[29] = Energy[id];
    id = i+3;
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
    //lr_states(stencil, bounds, dx, dt, gama);
    #endif


    // place the boundary values in the relevant array
    // at this point, L&R correspond to left and right of that cell, i.e. |L i R|
    id = i-1;
    Q1.d_R[id]  = bounds[0];
    Q1.mx_R[id] = bounds[1];
    Q1.my_R[id] = bounds[2];
    Q1.mz_R[id] = bounds[3];
    Q1.E_R[id]  = bounds[4];
    id = i;
    Q1.d_L[id]  = bounds[5];
    Q1.mx_L[id] = bounds[6];
    Q1.my_L[id] = bounds[7];
    Q1.mz_L[id] = bounds[8];
    Q1.E_L[id]  = bounds[9];

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
  istart = n_ghost-1; istop = nx-n_ghost;
  for (i=istart; i<istop; i++)
  {
    // set input variables for the x interfaces
    // exact Riemann solver takes conserved variables
    cW[0] = Q1.d_L[i];
    cW[1] = Q1.d_R[i];
    cW[2] = Q1.mx_L[i]; 
    cW[3] = Q1.mx_R[i];
    cW[4] = Q1.my_L[i];
    cW[5] = Q1.my_R[i];
    cW[6] = Q1.mz_L[i];
    cW[7] = Q1.mz_R[i];
    cW[8] = Q1.E_L[i];
    cW[9] = Q1.E_R[i];

    // call a Riemann solver to evaluate fluxes at the cell interface
    #ifdef EXACT
    Calculate_Exact_Fluxes(cW, flux, gama);
    #endif
    #ifdef ROE
    Calculate_Roe_Fluxes(cW, flux, gama, etah);
    #endif

    // update the fluxes in the x-direction
    F1.dflux[i]  = flux[0];
    F1.xmflux[i] = flux[1];
    F1.ymflux[i] = flux[2];
    F1.zmflux[i] = flux[3];
    F1.Eflux[i]  = flux[4];

  }

  


  // Step 3: Apply the CT algorithm to calculate the CT electric fields at cell-corners
  // (not applicable if not doing MHD)

  // Step 4: Update the face-centered magnetic field by dt/2 using EMFs computed in step 3
  // (not applicable if not doing MHD)

  // Step 5: Evolve the left and right states at each interface by dt/2 using transverse flux gradients
  // (not necessary for 1D; included for completeness)

  // Step 6: Calculate a cell-centered electric field at t^n+1/2 
  // (not applicable if not doing MHD)

  // Step 7: Compute new fluxes at cell interfaces using the corrected left and right
  // states from step 5, and the interface magnetic fields computed in step 4.
  // (not necessary for 1D)

  // Step 8: Apply the CT algorithm to calculate the CT electric fields
  // Not applicable if not doing MHD


  // Step 9: Update the solution from time level n to n+1

  // get start time
  //double start_update, stop_update;
  //start_update = get_time();

  // Only update real cells
  istart = n_ghost; istop = nx-n_ghost;
  for (i=istart; i<istop; i++) 
  { 
    density[i]    += dtodx * (F1.dflux[i-1]  - F1.dflux[i]);
    momentum_x[i] += dtodx * (F1.xmflux[i-1] - F1.xmflux[i]); 
    momentum_y[i] += dtodx * (F1.ymflux[i-1] - F1.ymflux[i]); 
    momentum_z[i] += dtodx * (F1.zmflux[i-1] - F1.zmflux[i]);
    Energy[i]     += dtodx * (F1.Eflux[i-1]  - F1.Eflux[i]);
  }

  // get stop time
  //stop_update = get_time();
  //printf("update time = %9.3f ms\n", (stop_update-start_update)*1000);  
   

  // free the interface states and flux structures
  free(Q1.d_L);
  free(F1.dflux);

}


States_1D::States_1D(int n_cells)
{
  // allocate memory for the interface state arrays (left and right for each interface)
  d_L  = (Real *) malloc(10*n_cells*sizeof(Real));
  d_R  = &(d_L[  n_cells]);
  mx_L = &(d_L[2*n_cells]);
  mx_R = &(d_L[3*n_cells]);
  my_L = &(d_L[4*n_cells]);
  my_R = &(d_L[5*n_cells]);
  mz_L = &(d_L[6*n_cells]);
  mz_R = &(d_L[7*n_cells]);
  E_L  = &(d_L[8*n_cells]);
  E_R  = &(d_L[9*n_cells]);

  // initialize array
  for (int i=0; i<10*n_cells; i++)
  {
    d_L[i] = 0.0;
  }

}


Fluxes_1D::Fluxes_1D(int n_cells)
{
  // allocate memory for flux arrays (density, momentum, energy)
  dflux  = (Real *) malloc(5*n_cells*sizeof(Real));
  xmflux = &(dflux[  n_cells]);
  ymflux = &(dflux[2*n_cells]);
  zmflux = &(dflux[3*n_cells]);
  Eflux  = &(dflux[4*n_cells]);

  // initialize array
  for (int i=0; i<5*n_cells; i++)
  {
    dflux[i] = 0.0;
  }

}

#endif //No CUDA

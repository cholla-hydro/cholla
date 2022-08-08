/*! \file radiation.cpp
 *  \brief Definitions for the radiative transfer wrapper */


#ifdef RT

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include"radiation.h"
#include"RT_functions.h"
#include"../global/global.h"
#include "../grid/grid3D.h"
#include"../io/io.h"



// function to do various initialization tasks, i.e. allocating memory, etc.
void Grid3D::Initialize_RT(void) {

  chprintf( "Initializing Radiative Transfer...\n");

  // Set radition grid parameters
  Rad.nx = H.nx_real + 2*Rad.n_ghost;
  Rad.ny = H.ny_real + 2*Rad.n_ghost;
  Rad.nz = H.nz_real + 2*Rad.n_ghost;
  Rad.n_cells = Rad.nx*Rad.ny*Rad.nz;

  // Allocate memory for abundances (passive scalars added to the hydro grid)
  // This is done in grid3D::Allocate_Memory
  chprintf( " N scalar fields: %d \n", NSCALARS );

  // Allocate memory for radiation fields (non-advecting, 2 per frequency plus 1 optically thin field)
  chprintf( "Allocating memory for radiation fields. \n");
  // allocate memory on the host
  Rad.rtFields.rfn = (Real *) malloc(Rad.n_freq*Rad.n_cells * sizeof(Real));  
  Rad.rtFields.rff = (Real *) malloc(Rad.n_freq*Rad.n_cells * sizeof(Real)); 
  Rad.rtFields.ot = (Real *) malloc(Rad.n_cells * sizeof(Real));
  // allocate memory on the device
  CudaSafeCall( cudaMalloc((void**)&Rad.rtFields.dev_rfn, Rad.n_freq*Rad.n_cells*sizeof(Real)) );  
  CudaSafeCall( cudaMalloc((void**)&Rad.rtFields.dev_rff, Rad.n_freq*Rad.n_cells*sizeof(Real)) );  
  CudaSafeCall( cudaMalloc((void**)&Rad.rtFields.dev_ot, Rad.n_cells*sizeof(Real)) );

  // Allocate memory for Eddington tensor only on device?



}

// function to call the radiation solver from main
void Grid3D::Update_RT() {

  // call the OTVET iteration
  // passes d_scalar as that is the pointer to the first abundance array, HI
  rtSolve(C.d_scalar);

  // pass boundaries
  rtBoundaries(C.d_scalar, Rad.rtFields);

}

void Rad3D::Free_Memory_RT(void){
  
  free( rtFields.rfn);
  free( rtFields.rff);
  free( rtFields.ot);
  cudaFree(rtFields.dev_rfn);  
  cudaFree(rtFields.dev_rff);  
  cudaFree(rtFields.dev_ot);  
  
}


#endif // RT

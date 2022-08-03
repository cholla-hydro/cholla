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

  // allocate memory for abundances (these can be passive scalars added to the hydro grid)
  // This is done in grid3D::Allocate_Memory
  //Allocate_Abundances();
  chprintf( " N scalar fields: %d \n", NSCALARS );

  // allocate memory for radiation fields (non-advecting, 2 per frequency plus 1 optically thin field)
  chprintf( "Allocating memory for radiation fields. \n");
  // allocate memory on the host
  Rad.RT_Fields.rfn = (Real *) malloc(Rad.n_freq*H.n_cells * sizeof(Real));  
  Rad.RT_Fields.rff = (Real *) malloc(Rad.n_freq*H.n_cells * sizeof(Real)); 
  Rad.RT_Fields.ot = (Real *) malloc(H.n_cells * sizeof(Real));
  // allocate memory on the device
  CudaSafeCall( cudaMalloc((void**)&Rad.RT_Fields.dev_rfn, Rad.n_freq*H.n_cells*sizeof(Real)) );  
  CudaSafeCall( cudaMalloc((void**)&Rad.RT_Fields.dev_rff, Rad.n_freq*H.n_cells*sizeof(Real)) );  
  CudaSafeCall( cudaMalloc((void**)&Rad.RT_Fields.dev_ot, H.n_cells*sizeof(Real)) );

  // allocate memory for Eddington tensor?



}

// function to call the radiation solver from main
void Grid3D::Update_RT() {

  // call the OTVET iteration
  // passes d_scalar as that is the pointer to the first abundance array, HI
  rtSolve(C.d_scalar);

  // pass boundaries
  rtBoundaries(C.d_scalar, Rad.RT_Fields.dev_rfn);

}

void Rad3D::Free_Memory_RT(void){
  
  free( RT_Fields.rfn);
  free( RT_Fields.rff);
  free( RT_Fields.ot);
  cudaFree(RT_Fields.dev_rfn);  
  cudaFree(RT_Fields.dev_rff);  
  cudaFree(RT_Fields.dev_ot);  
  
}


#endif // RT

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

  // Set radiation grid parameters
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

  // Allocate memory for Eddington tensor only on device
  CudaSafeCall( cudaMalloc((void**)&Rad.rtFields.dev_et, 6*Rad.n_cells*sizeof(Real)) );

  // Allocate memory for radiation source field only on device
  CudaSafeCall( cudaMalloc((void**)&Rad.rtFields.dev_rs, Rad.n_cells*sizeof(Real)) );

  // Initialize Field values (for now)
  Rad.Initialize_RT_Fields();

}


// function to call the radiation solver from main
void Grid3D::Update_RT() {

  // passes d_scalar as that is the pointer to the first abundance array, rho_HI
  Rad.rtSolve(C.d_scalar);

}

// function to initialize radiation fields (temporary)
void Rad3D::Initialize_RT_Fields(void) {

  // set values for frequency fields
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;

  istart = n_ghost;
  iend   = nx-n_ghost;
  jstart = n_ghost;
  jend   = ny-n_ghost;
  kstart = n_ghost;
  kend   = nz-n_ghost;

  // set initial states for CPU fields
  for(k=kstart-1; k<kend; k++) {
    for(j=jstart-1; j<jend; j++) {
      for(i=istart-1; i<iend; i++) {

        //get cell index
        id = i + j*nx + k*nx*ny;

        for (int ii=0; ii<n_freq; ii++) {
          rtFields.rfn[id + ii*n_cells] = ii;
          rtFields.rff[id + ii*n_cells] = n_freq+ii;
        }
        rtFields.ot[id] = 1;
      }
    }
  }  
  // set GPU fields (in RT_functions.cu)
  Initialize_RT_Fields_GPU();

}

void Rad3D::Free_Memory_RT(void) {
  
  free( rtFields.rfn);
  free( rtFields.rff);
  free( rtFields.ot);
  cudaFree(rtFields.dev_rfn);  
  cudaFree(rtFields.dev_rff);  
  cudaFree(rtFields.dev_ot); 
  cudaFree(rtFields.dev_et); 
  cudaFree(rtFields.dev_rs); 
  
}


#endif // RT

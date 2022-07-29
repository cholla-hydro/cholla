/*! \file radiation.cpp
 *  \brief Definitions for the radiative transfer wrapper */


#ifdef RT

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include"radiation.h"
#include"../global/global.h"
#include "../grid/grid3D.h"
#include"../io/io.h"



// function to do various initialization tasks, i.e. allocating memory, etc.
void Grid3D::Initialize_RT( struct parameters *P ) {

  chprintf( "Initializing Radiative Transfer...\n");

  Rad.Initialize( P );

  // allocate memory for abundances (these can be passive scalars added to the hydro grid)
  // This is done in grid3D::Allocate_Memory
  //Allocate_Abundances();

  // allocate memory for radiation fields (non-advecting, 2 per frequency plus 1 optically thin field)
  Rad.Allocate_Memory_RT();

  // allocate memory for Eddington tensor?



}


void Rad3D::Initialize( struct parameters *P) {

  chprintf( " N scalar fields: %d \n", NSCALARS );

}


// Sets pointers for abundances (already allocated in Grid3D.cpp)
//void Grid3D::Allocate_Abundances() {
//
//  chprintf( " Setting pointers for: HI, HII, HeI, HeII, HeIII, densities\n");
//  RT.HI_density      = &C.scalar[ 0*H.n_cells ];
//  RT.HII_density     = &C.scalar[ 1*H.n_cells ];
//  RT.HeI_density     = &C.scalar[ 2*H.n_cells ];
//  RT.HeII_density    = &C.scalar[ 3*H.n_cells ];
//  RhT.HeIII_density   = &C.scalar[ 4*H.n_cells ];

//}


// function to allocate memory for radiation fields
void Rad3D::Allocate_Memory_RT(void) {


}


void rtStart(void) {

}

void rtFinish(void) {

}




#endif // RT

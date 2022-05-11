/*! \file RT.cpp
 *  \brief Definitions for the radiative transfer wrapper */


#ifdef RT

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include"../global.h"
#include"../io.h"

#include"RT.h"



Rad3D::Rad3D( void ){}



// function to do various initialization tasks, i.e. allocating memory, etc.
void Grid3D::Initialize_RT( struct parameters *P ) {

  chprintf( "Initializing Radiative Transfer...\n");

  RT.Initialize( P );

  // allocate memory for abundances (these are passive scalars already allocated in the hydro grid)

  // allocate memory for radiation fields (non-advecting, 2 per frequency plus 1 optically thin field)
  RT.Allocate_Memory_RT();

  // allocate memory for Eddington tensor?



}


void Rad3D::Initialize( struct parameters *P) {

  chprintf( " N scalar fields: %d \n", NSCALARS );

}



// function to allocate memory for radiation fields
void Rad3D::Allocate_Memory_RT(void) {


}


void rtStart(void) {

}

void rtFinish(void) {

}




#endif // RT

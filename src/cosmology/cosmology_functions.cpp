#ifdef COSMOLOGY


#include"../grid3D.h"
#include"../global.h"
#include "../io.h"



void Grid3D::Initialize_Cosmology( struct parameters *P ){
  
  chprintf( "\nInitializing Cosmology... \n");
  
  Cosmo.Initialize( P, Grav, Particles );
  chprintf( "Cosmology Successfully Initialized. \n\n");
  
  
  
}

#endif
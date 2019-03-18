#ifdef PARTICLES

#include <unistd.h>
#include "../io.h"
#include "../grid3D.h"
#include "particles_3D.h"

#ifdef MPI_CHOLLA
#include"../mpi_routines.h"
#endif

#ifdef PARALLEL_OMP
#include "../parallel_omp.h"
#endif

Particles_3D::Particles_3D( void ){}

void Grid3D::Initialize_Particles( struct parameters *P ){
  
  chprintf( "\nInitializing Particles...\n");
  
  
  #ifdef MPI_CHOLLA
  MPI_Barrier( world );
  #endif
  chprintf( "Particles Initialized Successfully. \n\n");
  
  
} 

void Particles_3D::Initialize( struct parameters *P, Grav3D &Grav){

  n_local = 0;
  n_total = 0;
  
}


















#endif//PARTICLES
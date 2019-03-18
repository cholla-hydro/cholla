#ifdef PARTICLES

#include "../io.h"
#include "../grid3D.h"
#include "particles_3D.h"


#ifdef MPI_CHOLLA
void Grid3D::Transfer_Particles_Density_Boundaries_MPI( struct parameters P ){
  
  Particles.TRANSFER_DENSITY_BOUNDARIES = true;
  Set_Boundary_Conditions(P);
  Particles.TRANSFER_DENSITY_BOUNDARIES = false;
  
}
#endif

void Grid3D::Transfer_Particles_Density_Boundaries( struct parameters P ){
  
  #ifdef MPI_CHOLLA
  Transfer_Particles_Density_Boundaries_MPI(P);
  #else
  chprintf( " Error Partcles not implemented for non MPI");
  exit(-1);
  #endif
  
  
  
  
  
}




#endif
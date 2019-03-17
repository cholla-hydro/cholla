#ifdef GRAVITY

#include"../grid3D.h"
#include"../global.h"
#include "../io.h"


void Grid3D::Initialize_Gravity( struct parameters *P ){
  
  chprintf( "\nInitializing Gravity... \n");
  
  Grav.Initialize( H.xblocal, H.yblocal, H.zblocal, H.xdglobal, H.ydglobal, H.zdglobal, P->nx, P->ny, P->nz, H.nx_real, H.ny_real, H.nz_real, H.dx, H.dy, H.dz, H.n_ghost_potential_offset  );
  






  chprintf( "Gravity Successfully Initialized \n\n");

  
}


#endif //GRAVITY
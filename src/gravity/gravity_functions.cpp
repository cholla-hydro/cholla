#ifdef GRAVITY

#include"../grid3D.h"
#include"../global.h"
#include "../io.h"


void Grid3D::Initialize_Gravity( struct parameters *P ){
  chprintf( "\nInitializing Gravity... \n");
  Grav.Initialize( H.xblocal, H.yblocal, H.zblocal, H.xdglobal, H.ydglobal, H.zdglobal, P->nx, P->ny, P->nz, H.nx_real, H.ny_real, H.nz_real, H.dx, H.dy, H.dz, H.n_ghost_potential_offset  );
  chprintf( "Gravity Successfully Initialized \n\n");
}

void Grid3D::Compute_Gravitational_Potential( struct parameters *P ){
  
  Real Grav_Constant = 1;
  
  Real dens_avrg, current_a;
  
  dens_avrg = 0;
  current_a = 1;
  
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif
  
  Grav.Poisson_solver.Get_Potential( Grav.F.density_h, Grav.F.potential_h, Grav_Constant, dens_avrg, current_a);
  
  
  #ifdef CPU_TIME
  Timer.End_and_Record_Time( 3 );
  #endif
  
}


#endif //GRAVITY
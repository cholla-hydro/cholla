#ifdef ANALYSIS

#include<stdio.h>
#include"analysis.h"
#include"../io.h"

Analysis_Module::Analysis_Module( void ){}

void Grid3D::Initialize_Analysis_Module( struct parameters *P ){
  
  chprintf( "\nInitializng Analysis Module...\n");
  
  Real z_now;
  #ifdef COSMOLOGY
  z_now = Cosmo.current_z;
  #else 
  z_now = NULL;
  #endif
  
  Analysis.Initialize( H.xdglobal, H.ydglobal, H.zdglobal, H.xblocal, H.yblocal, H.zblocal, P->nx, P->ny, P->nz, H.nx_real, H.ny_real, H.nz_real, H.dx, H.dy, H.dz, H.n_ghost, z_now, P );
  
}

void Analysis_Module::Initialize( Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx_real, Real dy_real, Real dz_real, int n_ghost_hydro, Real z_now, struct parameters *P ){
  
  //Domain Length
  Lbox_x = Lx;
  Lbox_y = Ly;
  Lbox_z = Lz;
  
  //Left Boundaries of Local domain
  xMin = x_min;
  yMin = y_min;
  zMin = z_min;
  
  //Cell sizes
  dx = dx_real;
  dy = dy_real;
  dz = dz_real;
  
  //Size of Global Domain
  nx_total = nx;
  ny_total = ny;
  nz_total = nz;
  
  //Size of Local Domain
  nx_local = nx_real;
  ny_local = ny_real;
  nz_local = nz_real;
  
  //Number of ghost cells in the conserved arrays
  n_ghost = n_ghost_hydro;
  
  #ifdef COSMOLOGY
  current_z = z_now;
  #endif
    
  //Load values of scale factor for analysis outputs
  Load_Scale_Outputs(P);
  
  #ifdef PHASE_DIAGRAM
  Initialize_Phase_Diagram(P);
  #endif
  
  #ifdef LYA_STATISTICS
  Initialize_Lya_Statistics(P);
  #endif
    
  chprintf( "Analysis Module Sucessfully Initialized.\n\n");
  
}



void Grid3D::Compute_and_Output_Analysis( struct parameters *P ){
  
  chprintf("\nComputing Analysis  current_z: %f\n", Analysis.current_z );
  
  Compute_Phase_Diagram();

  
  //Write to HDF5 file
  #ifdef MPI_CHOLLA
  if ( procID == 0 ) Output_Analysis(P);
  #else
  Output_Analysis(P);
  #endif
  
  Analysis.Set_Next_Scale_Output();
  Analysis.output_now = false;
  
}

void Analysis_Module::Reset(){
  
  #ifdef PHASE_DIAGRAM
  free(phase_diagram);
  #endif
  
}



#endif
#ifdef ANALYSIS

#include<stdio.h>
#include"analysis.h"
#include"../io.h"


Analysis_Module::Analysis_Module( void ){}

void Grid3D::Compute_Lya_Statistics( ){
  
  // Copmpute Lya Statitics
  chprintf( "Computing Lya Statistics \n");
  
  int axis, n_skewers;
  
  for ( axis=0; axis<3; axis++ ){
    
    if ( axis == 0 ){
      n_skewers = Analysis.n_skewers_local_x;
    }
    
    if ( axis == 1 ){
      n_skewers = Analysis.n_skewers_local_y;
    }
    
    if ( axis == 2 ){
      n_skewers = Analysis.n_skewers_local_z;
    }
    
    if ( axis == 0 ) chprintf( " Computing Along X axis:\n");
    if ( axis == 1 ) chprintf( " Computing Along Y axis:\n");
    if ( axis == 2 ) chprintf( " Computing Along Z axis:\n");
    
    
    Populate_Lya_Skewers_Local( axis );
    Analysis.Transfer_Skewers_Data( axis );
    Analysis.Initialize_Lya_Statistics_Measurements( axis );
    
    for ( int skewer_id=0; skewer_id< n_skewers; skewer_id++ ){
      Compute_Transmitted_Flux_Skewer( skewer_id, axis );
      Analysis.Compute_Lya_Statistics_Skewer( skewer_id, axis );
    }
    
    Analysis.Reduce_Lya_Statists_Axis( axis );
    
  }  
  
  Analysis.Reduce_Lya_Statists_Global();
  
  
  chprintf( "Completed Lya Statistics \n" );
  
}

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
  
  //Domain Global left Boundaty 
  xMin_global = P->xmin;
  yMin_global = P->ymin;
  zMin_global = P->zmin;
  
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
  
  #ifdef PHASE_DIAGRAM
  Compute_Phase_Diagram();
  #endif
  
  #ifdef LYA_STATISTICS
  Compute_Lya_Statistics();
  #endif
  
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
  
  #ifdef LYA_STATISTICS
  free( skewers_HI_density_local_x );
  free( skewers_HI_density_local_y );
  free( skewers_HI_density_local_z );  
  free( skewers_velocity_local_x );
  free( skewers_velocity_local_y );
  free( skewers_velocity_local_z );
  free( skewers_temperature_local_x );
  free( skewers_temperature_local_y );
  free( skewers_temperature_local_z );
  #ifdef MPI_CHOLLA
  free( skewers_HI_density_root_x );
  free( skewers_HI_density_root_y );
  free( skewers_HI_density_root_z );  
  free( skewers_velocity_root_x );
  free( skewers_velocity_root_y );
  free( skewers_velocity_root_z );
  free( skewers_temperature_root_x );
  free( skewers_temperature_root_y );
  free( skewers_temperature_root_z );
  #endif
  #endif
  
  
}



#endif
#ifdef ANALYSIS

#include"analysis.h"
#include"../io.h"
#include <unistd.h>

#ifdef MPI_CHOLLA
#include"../mpi_routines.h"
#endif

void Grid3D::Populate_Lya_Skewers_Local( int axis ){
  
  int nx_local, ny_local, nz_local, n_ghost;
  int nx_grid, ny_grid, nz_grid;
  int ni, nj, n_los, stride, n_skewers_local;
  Real *momentum_los;
  Real *HI_density_los;
  Real *velocity_los;
  Real *temperature_los;
  
  nx_local = Analysis.nx_local;
  ny_local = Analysis.ny_local;
  nz_local = Analysis.nz_local;
  n_ghost = Analysis.n_ghost;
  nx_grid = nx_local + 2*n_ghost;
  ny_grid = ny_local + 2*n_ghost;
  nz_grid = nz_local + 2*n_ghost;
  stride = Analysis.n_stride;
  
  // X axis
  if ( axis == 0 ){
    n_los = nx_local;
    ni    = ny_local;
    nj    = nz_local;
    n_skewers_local = Analysis.n_skewers_local_x;
    momentum_los = C.momentum_x;
    HI_density_los  = Analysis.skewers_HI_density_local_x;
    velocity_los    = Analysis.skewers_velocity_local_x;
    temperature_los = Analysis.skewers_velocity_local_x; 
  } 
  
  // Y axis
  if ( axis == 1 ){
    n_los = ny_local;
    ni    = nx_local;
    nj    = nz_local;
    n_skewers_local = Analysis.n_skewers_local_y;
    momentum_los = C.momentum_y;
    HI_density_los  = Analysis.skewers_HI_density_local_y;
    velocity_los    = Analysis.skewers_velocity_local_y;
    temperature_los = Analysis.skewers_velocity_local_y;
  }
  
  // Z axis
  if ( axis == 2 ){
    n_los = nz_local;
    ni    = nx_local;
    nj    = ny_local;
    n_skewers_local = Analysis.n_skewers_local_z;
    momentum_los = C.momentum_z;
    HI_density_los  = Analysis.skewers_HI_density_local_z;
    velocity_los    = Analysis.skewers_velocity_local_z;
    temperature_los = Analysis.skewers_velocity_local_z;
  } 
    
  int n_iter_i, n_iter_j, id_grid;   
  n_iter_i = ni / stride;
  n_iter_j = nj / stride;
  int id_i, id_j, skewer_id;
  Real density, HI_density, velocity, temperature ;
  for ( int i=0; i<n_iter_i; i++ ){
    for ( int j=0; j<n_iter_j; j++ ){
      for ( int id_los=0; id_los<n_los; id_los++ ){
        id_i = i * stride;
        id_j = j * stride;
        if ( axis == 0 ) id_grid = ( id_los + n_ghost ) + ( id_i + n_ghost )*nx_grid   + ( id_j + n_ghost )*nx_grid*nz_grid;
        if ( axis == 1 ) id_grid = ( id_i + n_ghost )   + ( id_los + n_ghost )*nx_grid + ( id_j + n_ghost )*nx_grid*nz_grid;
        if ( axis == 2 ) id_grid = ( id_i + n_ghost )   + ( id_j + n_ghost )*nx_grid   + ( id_los + n_ghost )*nx_grid*nz_grid;   
        density = C.density[id_grid] * Cosmo.rho_0_gas;
        velocity = momentum_los[id_grid] * Cosmo.rho_0_gas * Cosmo.v_0_gas / Cosmo.current_a / density;
        #ifdef COOLING_GRACKLE
        HI_density  = Cool.fields.HI_density[id_grid] * Cosmo.rho_0_gas;
        temperature = Cool.temperature[id_grid];
        #else
        chprintf( "ERROR: Lya Statistics only supported for Grackle Cooling \n");
        exit(-1);
        #endif
        HI_density_los[skewer_id*n_los + id_los] = HI_density;
        velocity_los[skewer_id*n_los + id_los] = velocity;
        temperature_los[skewer_id*n_los + id_los] = temperature;
      }
      skewer_id += 1;
    }
  }  
  
  if ( (skewer_id + 1) != n_skewers_local ){
    chprintf( "ERROR: Skewers numbers don't match\n" );
    exit(-1);
  }

}


void Analysis_Module::Initialize_Lya_Statistics( struct parameters *P ){
  
  
  chprintf(" Initializing Lya Statistics,,,\n");
  
  n_stride = 4;
  n_skewers_local_x = ( ny_local / n_stride ) * ( nz_local / n_stride );
  n_skewers_local_y = ( nx_local / n_stride ) * ( nz_local / n_stride );
  n_skewers_local_z = ( nx_local / n_stride ) * ( ny_local / n_stride );
  
  #ifdef MPI_CHOLLA
  n_skewers_total_x = ( ny_total / n_stride ) * ( nz_total / n_stride );
  n_skewers_total_y = ( nx_total / n_stride ) * ( nz_total / n_stride );
  n_skewers_total_z = ( nx_total / n_stride ) * ( ny_total / n_stride );
  #else
  n_skewers_total_x = n_skewers_local_x;
  n_skewers_total_y = n_skewers_local_y;
  n_skewers_total_z = n_skewers_local_z;
  #endif
  
  
  // Alocate Memory For Properties of Local Skewers
  skewers_HI_density_local_x = (Real *) malloc(n_skewers_local_x*nx_local*sizeof(Real));
  skewers_HI_density_local_y = (Real *) malloc(n_skewers_local_y*ny_local*sizeof(Real));
  skewers_HI_density_local_z = (Real *) malloc(n_skewers_local_z*nz_local*sizeof(Real));
  
  skewers_velocity_local_x = (Real *) malloc(n_skewers_local_x*nx_local*sizeof(Real));
  skewers_velocity_local_y = (Real *) malloc(n_skewers_local_y*ny_local*sizeof(Real));
  skewers_velocity_local_z = (Real *) malloc(n_skewers_local_z*nz_local*sizeof(Real));
  
  skewers_temperature_local_x = (Real *) malloc(n_skewers_local_x*nx_local*sizeof(Real));
  skewers_temperature_local_y = (Real *) malloc(n_skewers_local_y*ny_local*sizeof(Real));
  skewers_temperature_local_z = (Real *) malloc(n_skewers_local_z*nz_local*sizeof(Real));
  
  
  // for (int i=0; i<nproc; i++ ){
  //   if ( procID == i  )  printf( " pID: %d    n_x: %d   n_y:%d   n_z:%d \n", procID, n_skewers_total_x, n_skewers_total_y, n_skewers_total_z ); 
  //   MPI_Barrier(world);  
  // }
    
  #ifdef MPI_CHOLLA
  mpi_domain_boundary_x = (Real *) malloc(nproc*sizeof(Real));
  mpi_domain_boundary_y = (Real *) malloc(nproc*sizeof(Real));
  mpi_domain_boundary_z = (Real *) malloc(nproc*sizeof(Real)); 
  
  
  MPI_Allgather(&xMin, 1, MPI_CHREAL, mpi_domain_boundary_x, 1, MPI_CHREAL, world );
  MPI_Allgather(&yMin, 1, MPI_CHREAL, mpi_domain_boundary_y, 1, MPI_CHREAL, world );
  MPI_Allgather(&zMin, 1, MPI_CHREAL, mpi_domain_boundary_z, 1, MPI_CHREAL, world );
  
  
  root_id_x = -1;
  root_id_y = -1;
  root_id_z = -1;
  
  // Find root_id
  for (int i=0; i<nproc; i++ ){
    if ( mpi_domain_boundary_x[i] == xMin_global && mpi_domain_boundary_y[i] == yMin && mpi_domain_boundary_z[i] == zMin ) root_id_x = i;
    if ( mpi_domain_boundary_y[i] == yMin_global && mpi_domain_boundary_x[i] == xMin && mpi_domain_boundary_z[i] == zMin ) root_id_y = i;
    if ( mpi_domain_boundary_z[i] == zMin_global && mpi_domain_boundary_x[i] == xMin && mpi_domain_boundary_y[i] == yMin ) root_id_z = i;
  }
  
  // for (int i=0; i<nproc; i++ ){
  //   if ( procID == i  )  printf( " pID: %d    root_x: %d   root_y:%d   root_z:%d \n", procID, root_id_x, root_id_y, root_id_z ); 
  //   MPI_Barrier(world);  
  // }
  
  //Construct the procIDs for the processors skewers
  for (int i=0; i<nproc; i++ ){
    if ( mpi_domain_boundary_y[i] == yMin && mpi_domain_boundary_z[i] == zMin ) mpi_indices_x.push_back(i);
    if ( mpi_domain_boundary_x[i] == xMin && mpi_domain_boundary_z[i] == zMin ) mpi_indices_y.push_back(i);
    if ( mpi_domain_boundary_x[i] == xMin && mpi_domain_boundary_y[i] == yMin ) mpi_indices_z.push_back(i);
  }
  
  int n_mpi_x = mpi_indices_x.size();
  int n_mpi_y = mpi_indices_y.size();
  int n_mpi_z = mpi_indices_z.size();
  
  int temp_indx;
  bool sorted; 
  if ( n_mpi_x > 0 ){
    sorted = true;
    while ( !sorted ){
      sorted = true; 
      for (int i=0; i<n_mpi_x-1; i++ ){
        if ( mpi_domain_boundary_x[mpi_indices_x[i]] > mpi_domain_boundary_x[mpi_indices_x[i+1]] ){
          temp_indx = mpi_indices_x[i];
          mpi_indices_x[i] = mpi_indices_x[i+1];
          mpi_indices_x[i+1] = temp_indx;
          sorted = false;
        }
      }
    }
  }
  
  if ( n_mpi_y > 0 ){
    sorted = true;
    while ( !sorted ){
      sorted = true; 
      for (int i=0; i<n_mpi_y-1; i++ ){
        if ( mpi_domain_boundary_y[mpi_indices_y[i]] > mpi_domain_boundary_y[mpi_indices_y[i+1]] ){
          temp_indx = mpi_indices_y[i];
          mpi_indices_y[i] = mpi_indices_y[i+1];
          mpi_indices_y[i+1] = temp_indx;
          sorted = false;
        }
      }
    }
  }
  
  if ( n_mpi_z > 0 ){
    sorted = true;
    while ( !sorted ){
      sorted = true; 
      for (int i=0; i<n_mpi_z-1; i++ ){
        if ( mpi_domain_boundary_z[mpi_indices_z[i]] > mpi_domain_boundary_z[mpi_indices_z[i+1]] ){
          temp_indx = mpi_indices_z[i];
          mpi_indices_z[i] = mpi_indices_z[i+1];
          mpi_indices_z[i+1] = temp_indx;
          sorted = false;
        }
      }
    }
  }
  
  
  
  // for (int i=0; i<nproc; i++ ){
  //   if ( procID == i  ){
  //     printf( " pID: %d  \n", procID );
  //     printf("  mpi_indx_x: " );
  //     for (int i=0; i<n_mpi_x; i++ ) printf("  %d", mpi_indices_x[i] );
  //     printf("\n" ); 
  //     printf("  mpi_indx_y: " );
  //     for (int i=0; i<n_mpi_y; i++ ) printf("  %d", mpi_indices_y[i] );
  //     printf("\n" ); 
  //     printf("  mpi_indx_z: " );
  //     for (int i=0; i<n_mpi_z; i++ ) printf("  %d", mpi_indices_z[i] );
  //     printf("\n" ); 
  //     MPI_Barrier(world);  
  // 
  //   }
  // }
  // 
  // 
  if ( mpi_indices_x[0] != root_id_x ){
    printf(" ERROR: Root id doesn't match mpi_indx list for x axis\n" );
    exit(-1);
  }     
  
  if ( mpi_indices_y[0] != root_id_y ){
    printf(" ERROR: Root id doesn't match mpi_indx list for y axis\n" );
    exit(-1);
  }     
  
  if ( mpi_indices_z[0] != root_id_z ){
    printf(" ERROR: Root id doesn't match mpi_indx list for z axis\n" );
    exit(-1);
  }     
  
  
  if ( procID == root_id_x ) am_I_root_x = true;
  else am_I_root_x = false;
  
  
  if ( procID == root_id_y ) am_I_root_y = true;
  else am_I_root_y = false;
  
  
  
  if ( procID == root_id_z ) am_I_root_z = true;
  else am_I_root_z = false;
  
  
  chprintf( " Root Ids X:  \n");
  MPI_Barrier(world);
  sleep(1);  
  if ( am_I_root_x  ){
    printf( "  pID: %d  \n", procID );
  }
  MPI_Barrier(world);
  sleep(1);  
  
  chprintf( " Root Ids Y:  \n");
  MPI_Barrier(world);
  sleep(1);  
  if ( am_I_root_y  ){
    printf( "  pID: %d  \n", procID );
  }
  MPI_Barrier(world);
  sleep(1);  
  
  
  chprintf( " Root Ids Z:  \n");
  MPI_Barrier(world);  
  sleep(1);
  if ( am_I_root_z  ){
    printf( "  pID: %d  \n", procID );
  }
  MPI_Barrier(world);  
  sleep(1);
  
  
  if ( am_I_root_x ){
    skewers_HI_density_root_x  = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
    skewers_velocity_root_x    = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
    skewers_temperature_root_x = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
  }
  
  if ( am_I_root_y ){
    skewers_HI_density_root_y  = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    skewers_velocity_root_y    = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    skewers_temperature_root_y = (Real *) malloc(n_skewers_local_x*ny_total*sizeof(Real));
  }
  
  if ( am_I_root_z ){
    skewers_HI_density_root_z  = (Real *) malloc(n_skewers_local_z*nx_total*sizeof(Real));
    skewers_velocity_root_z    = (Real *) malloc(n_skewers_local_z*nx_total*sizeof(Real));
    skewers_temperature_root_z = (Real *) malloc(n_skewers_local_z*nx_total*sizeof(Real));
  }
  
  #else 
  
  skewers_HI_density_root_x = skewers_HI_density_local_x;
  skewers_HI_density_root_y = skewers_HI_density_local_y;
  skewers_HI_density_root_z = skewers_HI_density_local_z;

  skewers_velocity_root_x = skewers_velocity_local_x;
  skewers_velocity_root_y = skewers_velocity_local_y;
  skewers_velocity_root_z = skewers_velocity_local_z;
  
  skewers_temperature_root_x = skewers_temperature_local_x;
  skewers_temperature_root_y = skewers_temperature_local_y;
  skewers_temperature_root_z = skewers_temperature_local_z;


  
  #endif
  
  
  chprintf(" Lya Statistics Initialized.\n");
  
  
  
}














#endif
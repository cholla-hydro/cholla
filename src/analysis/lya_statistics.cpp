#ifdef ANALYSIS

#include"analysis.h"
#include"../io.h"
#include <unistd.h>

#ifdef MPI_CHOLLA
#include"../mpi_routines.h"
#endif



void Analysis_Module::Reduce_Lya_Statists_Global( void ){
    
  n_skewers_processed = n_skewers_processed_x + n_skewers_processed_y + n_skewers_processed_z;
  Flux_mean = ( Flux_mean_x*n_skewers_processed_x + Flux_mean_y*n_skewers_processed_y + Flux_mean_z*n_skewers_processed_z  ) / n_skewers_processed;
  
  chprintf( " Flux Mean Global: %e      N_Skewers_Processed: %d\n", Flux_mean, n_skewers_processed );
}

void Analysis_Module::Reduce_Lya_Statists_Axis( int axis ){
  
  int  *n_skewers_processed;
  int  *n_skewers_processed_root;
  Real *Flux_mean;
  Real *Flux_mean_root;
  
  if ( axis == 0 ){
    n_skewers_processed      = &n_skewers_processed_x;
    n_skewers_processed_root = &n_skewers_processed_root_x;
    Flux_mean                = &Flux_mean_x;
    Flux_mean_root           = &Flux_mean_root_x;
  }

  if ( axis == 1 ){
    n_skewers_processed      = &n_skewers_processed_y;
    n_skewers_processed_root = &n_skewers_processed_root_y;
    Flux_mean                = &Flux_mean_y;
    Flux_mean_root           = &Flux_mean_root_y;
  }

  if ( axis == 2 ){
    n_skewers_processed      = &n_skewers_processed_z;
    n_skewers_processed_root = &n_skewers_processed_root_z;
    Flux_mean                = &Flux_mean_z;
    Flux_mean_root           = &Flux_mean_root_z;
  }


  
  #ifdef MPI_CHOLLA

  for ( int i=0; i<nproc; i++ ){
    if (procID == i) printf("   procID:%d   Flux Sum: %e     N_Skewers_Processed: %d \n", procID, (*Flux_mean_root), *n_skewers_processed_root );
    MPI_Barrier(world);
    sleep(1);
  }
  
  MPI_Allreduce( Flux_mean_root, Flux_mean, 1, MPI_CHREAL, MPI_SUM, world );
  MPI_Allreduce( n_skewers_processed_root, n_skewers_processed, 1, MPI_INT, MPI_SUM, world );

  #else

  *Flux_mean = *Flux_mean_root;
  *n_skewers_processed = *n_skewers_processed_root;

  #endif

  *Flux_mean = *Flux_mean / *n_skewers_processed;
  chprintf( "  Flux Mean: %e     N_Skewers_Processed: %d \n", *Flux_mean, *n_skewers_processed );

  
}


void Analysis_Module::Compute_Lya_Statistics_Skewer( int skewer_id, int axis ){
  
  bool am_I_root;
  int  n_los;
  int  *n_skewers_processed_root;
  Real *F_mean_root;
  Real *transmitted_flux;
  
  if ( axis == 0 ){
    am_I_root = am_I_root_x;
    n_los = nx_total;
    F_mean_root = &Flux_mean_root_x;
    transmitted_flux = transmitted_flux_x;
    n_skewers_processed_root = &n_skewers_processed_root_x; 
  }
  
  if ( axis == 1 ){
    am_I_root = am_I_root_y;
    n_los = ny_total;
    F_mean_root = &Flux_mean_root_y;
    transmitted_flux = transmitted_flux_y;
    n_skewers_processed_root = &n_skewers_processed_root_y;
  }
  
  if ( axis == 2 ){
    am_I_root = am_I_root_z;
    n_los = nz_total;
    F_mean_root = &Flux_mean_root_z;
    transmitted_flux = transmitted_flux_z;
    n_skewers_processed_root = &n_skewers_processed_root_z;
  }
  
  if ( !am_I_root ) return;
  
  Real F_mean = 0;
  for ( int los_id=0; los_id<n_los; los_id++ ){
    F_mean += transmitted_flux[los_id] / n_los;
  }
  
  // *F_mean_root += 1;
  *F_mean_root += F_mean;
  *n_skewers_processed_root += 1;
  
}
  
void Analysis_Module::Initialize_Lya_Statistics_Measurements( int axis ){
  
  if ( axis == 0 ){
    n_skewers_processed_root_x = 0;
    Flux_mean_root_x = 0;
  }
  
  if ( axis == 1 ){
    n_skewers_processed_root_y = 0;
    Flux_mean_root_y = 0;
  }
  
  if ( axis == 2 ){
    n_skewers_processed_root_z = 0;
    Flux_mean_root_z = 0;
  }
  
}

void Grid3D::Compute_Transmitted_Flux_Skewer( int skewer_id, int axis ){
  
  int n_los_full, n_los_total, n_ghost;
  bool am_I_root;
  Real *full_HI_density;
  Real *full_velocity;
  Real *full_temperature;
  Real *full_optical_depth;
  Real *full_vel_Hubble;
  Real *transmitted_flux;
  Real *skewers_HI_density_root;
  Real *skewers_velocity_root;
  Real *skewers_temperature_root;
  Real Lbox, delta_x;
  
  n_ghost = Analysis.n_ghost_skewer;
  
  
  if ( axis == 0 ){
    Lbox    = Analysis.Lbox_x;
    delta_x = Analysis.dx;
    am_I_root   = Analysis.am_I_root_x;
    n_los_full  = Analysis.n_los_full_x;
    n_los_total = Analysis.nx_total;
    full_HI_density  = Analysis.full_HI_density_x;
    full_velocity    = Analysis.full_velocity_x;
    full_temperature = Analysis.full_temperature_x;
    full_optical_depth = Analysis.full_optical_depth_x;
    full_vel_Hubble = Analysis.full_vel_Hubble_x;
    transmitted_flux = Analysis.transmitted_flux_x;
    skewers_HI_density_root  = Analysis.skewers_HI_density_root_x;
    skewers_velocity_root    = Analysis.skewers_velocity_root_x; 
    skewers_temperature_root = Analysis.skewers_temperature_root_x; 
  }
  
  if ( axis == 1 ){
    Lbox    = Analysis.Lbox_y;
    delta_x = Analysis.dy;  
    am_I_root   = Analysis.am_I_root_y;
    n_los_full  = Analysis.n_los_full_y;
    n_los_total = Analysis.ny_total;
    full_HI_density  = Analysis.full_HI_density_y;
    full_velocity    = Analysis.full_velocity_y;
    full_temperature = Analysis.full_temperature_y;
    full_optical_depth = Analysis.full_optical_depth_y;
    full_vel_Hubble = Analysis.full_vel_Hubble_y;
    transmitted_flux = Analysis.transmitted_flux_y; 
    skewers_HI_density_root  = Analysis.skewers_HI_density_root_y;
    skewers_velocity_root    = Analysis.skewers_velocity_root_y; 
    skewers_temperature_root = Analysis.skewers_temperature_root_y; 
  }
  
  if ( axis == 2 ){
    Lbox    = Analysis.Lbox_z;
    delta_x = Analysis.dz;
    am_I_root   = Analysis.am_I_root_z;
    n_los_full  = Analysis.n_los_full_z;
    n_los_total = Analysis.nz_total;
    full_HI_density  = Analysis.full_HI_density_z;
    full_velocity    = Analysis.full_velocity_z;
    full_temperature = Analysis.full_temperature_z;
    full_optical_depth = Analysis.full_optical_depth_z;
    full_vel_Hubble = Analysis.full_vel_Hubble_z;
    transmitted_flux = Analysis.transmitted_flux_z; 
    skewers_HI_density_root  = Analysis.skewers_HI_density_root_z;
    skewers_velocity_root    = Analysis.skewers_velocity_root_z; 
    skewers_temperature_root = Analysis.skewers_temperature_root_z; 
  }
  
  if ( !am_I_root ) return;
  
  // printf( "  Computing Skewer ID: %d \n", skewer_id );
  
  Real HI_density, velocity, temperature, Msun, kpc, Mp, kpc3, Me, e_charge, c, Kb;
  
  // Constants in CGS
  Kb   = 1.38064852e-16; //g (cm/s)^2 K-1
  Msun = 1.98847e33;     //g 
  Mp   = 1.6726219e-24;  //g 
  Me   = 9.10938356e-28; //g
  c    = 2.99792458e10;    //cm/s
  kpc  = 3.0857e21;        //cm
  kpc3 = kpc * kpc * kpc;
  e_charge =  4.8032e-10;  // cm^3/2 g^1/2 s^-1 
  
  Real dens_factor, vel_factor;

  dens_factor = 1. / ( Cosmo.current_a * Cosmo.current_a * Cosmo.current_a ) * Cosmo.cosmo_h * Cosmo.cosmo_h; 
  dens_factor *= Msun / ( kpc3 ) / Mp;
  vel_factor = 1e5; //cm/s
   
  // Fill the Real cells first
  for (int los_id=0; los_id<n_los_total; los_id++ ){
    HI_density  = skewers_HI_density_root [ skewer_id*n_los_total + los_id ];
    velocity    = skewers_velocity_root   [ skewer_id*n_los_total + los_id ];
    temperature = skewers_temperature_root[ skewer_id*n_los_total + los_id ];
    HI_density *= dens_factor;
    velocity   *= vel_factor;
    full_HI_density [los_id + n_ghost] = HI_density;
    full_velocity   [los_id + n_ghost] = velocity;
    full_temperature[los_id + n_ghost] = temperature;
  }
  
  // Fill the ghost cells
  for ( int los_id=0; los_id<n_ghost; los_id++ ){
    full_HI_density [los_id] = full_HI_density [n_los_total+los_id];
    full_velocity   [los_id] = full_velocity   [n_los_total+los_id];
    full_temperature[los_id] = full_temperature[n_los_total+los_id];
    full_HI_density [n_los_total+n_ghost+los_id] = full_HI_density [n_ghost+los_id];
    full_velocity   [n_los_total+n_ghost+los_id] = full_velocity   [n_ghost+los_id];
    full_temperature[n_los_total+n_ghost+los_id] = full_temperature[n_ghost+los_id];
  } 
  
  
  // for (int los_id=0; los_id<n_los_full; los_id++ ){
  //   HI_density  = full_HI_density [ los_id ];
  //   velocity    = full_velocity   [ los_id ];
  //   temperature = full_temperature[ los_id ];
  //   printf("id: %d   dens: %f   vel:%f   temp:%f \n", los_id, HI_density, velocity, temperature );    
  // }
  
  // for (int los_id=0; los_id<n_los_full; los_id++ ){
  //   full_HI_density[ los_id ] = 10*dens_factor;
  // }
  
  // Get Cosmological variables
  Real H, current_a, L_proper, dx_proper, dv_Hubble; 
  Real H_cgs, Lya_lambda, f_12, Lya_sigma;
  current_a = Cosmo.current_a;
  L_proper = Lbox * current_a / Cosmo.cosmo_h;
  dx_proper = delta_x * current_a / Cosmo.cosmo_h;
  H = Cosmo.Get_Hubble_Parameter( current_a );
  dv_Hubble = H * dx_proper * vel_factor; // cm/s
  
  
  // Fill the Hubble velocity with ghost cells
  for ( int los_id=0; los_id<n_los_full; los_id++ ){
    full_vel_Hubble[los_id] = ( los_id - n_ghost + 0.5 ) * dv_Hubble;
  }
  
  H_cgs = H * 1e5 / kpc;
  Lya_lambda = 1.21567e-5; // cm  Rest wave length of the Lyman Alpha Transition
  f_12 =  0.416;           // Oscillator strength
  Lya_sigma = M_PI * e_charge * e_charge / Me / c * f_12;
  
  
  // printf("H0: %f    H:%f\n", Cosmo.H0, H );
  // printf("dv_Hubble: %f    \n", dv_Hubble);
  // printf("L: %f    dx:%f\n", Lbox, delta_x );
  // printf( "%f \n", Lya_sigma * Lya_lambda / H_cgs);
  
  //Compute the optical depth
  Real vel_i, tau_i, vel_j, b_j, n_HI_j, y_l, y_r; 
  for ( int i=0; i<n_los_full; i++ ){
    vel_i = full_vel_Hubble[i];
    tau_i = 0;
    for ( int j=0; j<n_los_full; j++ ){
      n_HI_j = full_HI_density[j];
      // if ( i==0 ) printf( "%f \n", full_velocity[j] * 1e-5);
      vel_j = full_vel_Hubble[j] + full_velocity[j];
      b_j = sqrt( 2 * Kb / Mp * full_temperature[j] );
      y_l = ( vel_i - 0.5*dv_Hubble - vel_j ) / b_j;
      y_r = ( vel_i + 0.5*dv_Hubble - vel_j ) / b_j;
      tau_i += n_HI_j * ( erf(y_r) - erf(y_l) ) / 2;
    }
    tau_i *= Lya_sigma * Lya_lambda / H_cgs;
    full_optical_depth[i] = tau_i;    
  }
  
  // Compute the transmitted_flux
  for ( int los_id=0; los_id<n_los_total; los_id++ ){
    transmitted_flux[los_id] = exp( -full_optical_depth[los_id + n_ghost]);
  }
  
  
  
}

void Analysis_Module::Transfer_Skewers_Data( int axis ){
  
  bool am_I_root;
  int n_skewers, n_los_local, n_los_total, root_id;
  Real *skewers_HI_density_local;
  Real *skewers_velocity_local;
  Real *skewers_temperature_local;
  Real *skewers_HI_density_root;
  Real *skewers_velocity_root;
  Real *skewers_temperature_root;
  
  #ifdef MPI_CHOLLA
  vector<int> mpi_indices;  
  MPI_Status mpi_status;

  #endif
  
  if ( axis == 0 ){
    root_id = root_id_x;
    am_I_root = am_I_root_x;
    n_los_local = nx_local;
    n_los_total = nx_total;
    n_skewers = n_skewers_local_x;
    skewers_HI_density_local = skewers_HI_density_local_x;
    skewers_velocity_local = skewers_velocity_local_x;
    skewers_temperature_local = skewers_temperature_local_x;
    skewers_HI_density_root = skewers_HI_density_root_x;
    skewers_velocity_root = skewers_velocity_root_x;
    skewers_temperature_root = skewers_temperature_root_x;
    #ifdef MPI_CHOLLA
    mpi_indices = mpi_indices_x;
    #endif
  }
  
  
  if ( axis == 1 ){
    root_id = root_id_y;
    am_I_root = am_I_root_y;
    n_los_local = ny_local;
    n_los_total = ny_total;
    n_skewers = n_skewers_local_y;
    skewers_HI_density_local = skewers_HI_density_local_y;
    skewers_velocity_local = skewers_velocity_local_y;
    skewers_temperature_local = skewers_temperature_local_y;
    skewers_HI_density_root = skewers_HI_density_root_y;
    skewers_velocity_root = skewers_velocity_root_y;
    skewers_temperature_root = skewers_temperature_root_y;
    #ifdef MPI_CHOLLA
    mpi_indices = mpi_indices_y;
    #endif
  }
  
  
  if ( axis == 2 ){
    root_id = root_id_z;
    am_I_root = am_I_root_z;
    n_los_local = nz_local;
    n_los_total = nz_total;
    n_skewers = n_skewers_local_z;
    skewers_HI_density_local = skewers_HI_density_local_z;
    skewers_velocity_local = skewers_velocity_local_z;
    skewers_temperature_local = skewers_temperature_local_z;
    skewers_HI_density_root = skewers_HI_density_root_z;
    skewers_velocity_root = skewers_velocity_root_z;
    skewers_temperature_root = skewers_temperature_root_z;
    #ifdef MPI_CHOLLA
    mpi_indices = mpi_indices_z;
    #endif
  }
  
  
  // Copy Skewers Local Data to Root data
  
  Real HI_density, velocity, temperature;
  
  #ifdef MPI_CHOLLA
  if ( am_I_root ){
    
    if ( root_id != procID ){
      printf("ERROR: Root ID doesn't match procID\n" );
      exit(-1);
    }
    
    for ( int skewer_id=0; skewer_id<n_skewers; skewer_id++){
      for ( int los_id=0; los_id<n_los_local; los_id++){
        HI_density = skewers_HI_density_local[skewer_id*n_los_local + los_id];
        velocity = skewers_velocity_local[skewer_id*n_los_local + los_id];
        temperature = skewers_temperature_local[skewer_id*n_los_local + los_id];
        skewers_HI_density_root[skewer_id*n_los_total + los_id] = HI_density;
        skewers_velocity_root[skewer_id*n_los_total + los_id] = velocity;
        skewers_temperature_root[skewer_id*n_los_total + los_id] = temperature;
        // if ( skewer_id == 0 ) printf("Dens: %f   vel:%f   temp:%f \n", HI_density, velocity, temperature ); 
      }
    }
  
    int n_indices = mpi_indices.size();
    printf( "  N MPI indices: %d \n", n_indices ); 
    
    int mpi_id; 
    for ( int indx=0; indx<n_indices; indx++ ){
      mpi_id = mpi_indices[indx];
      if ( indx == 0 ){
        if ( mpi_id != procID ){
          printf( "ERROR: Fist MPI indx doesn't match root indx \n");
          exit(-1);
        }
        continue; 
      } 
      
      printf("  Recieving Skewers From pID: %d\n", mpi_id );
      MPI_Recv( skewers_HI_density_local, n_skewers*n_los_local, MPI_CHREAL, mpi_id, 0, world, &mpi_status  );
      MPI_Recv( skewers_velocity_local, n_skewers*n_los_local, MPI_CHREAL, mpi_id, 1, world, &mpi_status  );
      MPI_Recv( skewers_temperature_local, n_skewers*n_los_local, MPI_CHREAL, mpi_id, 2, world, &mpi_status  );
      
      
      
      for ( int skewer_id=0; skewer_id<n_skewers; skewer_id++){
        for ( int los_id=0; los_id<n_los_local; los_id++){
          skewers_HI_density_root [skewer_id*n_los_total + indx*n_los_local + los_id] = skewers_HI_density_local [skewer_id*n_los_local + los_id];
          skewers_velocity_root   [skewer_id*n_los_total + indx*n_los_local + los_id] = skewers_velocity_local   [skewer_id*n_los_local + los_id];
          skewers_temperature_root[skewer_id*n_los_total + indx*n_los_local + los_id] = skewers_temperature_local[skewer_id*n_los_local + los_id];
        }
      }
      
    }
  }
    
  else{
    
    MPI_Send( skewers_HI_density_local, n_skewers*n_los_local, MPI_CHREAL, root_id, 0, world  );
    MPI_Send( skewers_velocity_local, n_skewers*n_los_local, MPI_CHREAL, root_id, 1, world  );
    MPI_Send( skewers_temperature_local, n_skewers*n_los_local, MPI_CHREAL, root_id, 2, world  );
  } 
  
  
  MPI_Barrier( world );
  #endif
    
    

  chprintf("  Skewers Data Transfered\n" );
}



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
    temperature_los = Analysis.skewers_temperature_local_x; 
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
    temperature_los = Analysis.skewers_temperature_local_y;
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
    temperature_los = Analysis.skewers_temperature_local_z;
  } 
    
  int n_iter_i, n_iter_j, id_grid;   
  n_iter_i = ni / stride;
  n_iter_j = nj / stride;
  int id_i, id_j, skewer_id;
  Real density, HI_density, velocity, temperature ;
  skewer_id = 0;
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
        // HI_density = 10;
        // velocity /= 1000;
        // temperature = 100 + id_los/2;
        HI_density_los[skewer_id*n_los + id_los] = HI_density;
        velocity_los[skewer_id*n_los + id_los] = velocity;
        temperature_los[skewer_id*n_los + id_los] = temperature;
        // if ( skewer_id == 0 ) printf("Dens: %f   vel:%f   temp:%f \n", HI_density, velocity, temperature );      
      }
      skewer_id += 1;
    }
  }  
  
  if ( skewer_id != n_skewers_local ){
    printf( "ERROR: Skewers numbers don't match.  ID: %d   N_skewers: %d \n ", skewer_id, n_skewers_local );
    exit(-1);
  }
  
  // printf( " Local Skewers Data Copied \n");

}


void Analysis_Module::Initialize_Lya_Statistics( struct parameters *P ){
  
  
  chprintf(" Initializing Lya Statistics...\n");
  
  
  n_ghost_skewer = max( nx_total, ny_total );
  n_ghost_skewer = max( nz_total, n_ghost_skewer );
  n_ghost_skewer = 0.1 * n_ghost_skewer;
  n_los_full_x = nx_total + 2 * n_ghost_skewer;
  n_los_full_y = ny_total + 2 * n_ghost_skewer;
  n_los_full_z = nz_total + 2 * n_ghost_skewer;
   
  
  
  n_stride = P->lya_skewers_stride;
  chprintf("  Lya Skewers Stride: %d\n", n_stride );
  
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
  //   // if ( procID == i  )  printf( " pID: %d    n_x: %d   n_y:%d   n_z:%d \n", procID, n_skewers_total_x, n_skewers_total_y, n_skewers_total_z ); 
  //   if ( procID == i  )  printf( " pID: %d    n_x: %d   n_y:%d   n_z:%d \n", procID, n_skewers_local_x, n_skewers_local_y, n_skewers_local_z ); 
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
    full_HI_density_x  = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_velocity_x    = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_temperature_x = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_optical_depth_x = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_vel_Hubble_x  = (Real *) malloc(n_los_full_x*sizeof(Real));
    transmitted_flux_x = (Real *) malloc(nx_total*sizeof(Real));
  }
  
  if ( am_I_root_y ){
    skewers_HI_density_root_y  = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    skewers_velocity_root_y    = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    skewers_temperature_root_y = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    full_HI_density_y  = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_velocity_y    = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_temperature_y = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_optical_depth_y = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_vel_Hubble_y  = (Real *) malloc(n_los_full_y*sizeof(Real));
    transmitted_flux_y = (Real *) malloc(ny_total*sizeof(Real));
  }
  
  if ( am_I_root_z ){
    skewers_HI_density_root_z  = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    skewers_velocity_root_z    = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    skewers_temperature_root_z = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    full_HI_density_z  = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_velocity_z    = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_temperature_z = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_optical_depth_z = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_vel_Hubble_z  = (Real *) malloc(n_los_full_z*sizeof(Real));
    transmitted_flux_z = (Real *) malloc(nz_total*sizeof(Real));
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
  
  full_HI_density_x  = (Real *) malloc(n_los_full_x*sizeof(Real));
  full_velocity_x    = (Real *) malloc(n_los_full_x*sizeof(Real));
  full_temperature_x = (Real *) malloc(n_los_full_x*sizeof(Real));
  full_optical_depth_x = (Real *) malloc(n_los_full_x*sizeof(Real));
  full_vel_Hubble_x  = (Real *) malloc(n_los_full_x*sizeof(Real));
  transmitted_flux_x = (Real *) malloc(nx_total*sizeof(Real));
  
  full_HI_density_y  = (Real *) malloc(n_los_full_y*sizeof(Real));
  full_velocity_y    = (Real *) malloc(n_los_full_y*sizeof(Real));
  full_temperature_y = (Real *) malloc(n_los_full_y*sizeof(Real));
  full_optical_depth_y = (Real *) malloc(n_los_full_y*sizeof(Real));
  full_vel_Hubble_y  = (Real *) malloc(n_los_full_y*sizeof(Real));
  transmitted_flux_y = (Real *) malloc(ny_total*sizeof(Real));
  
  full_HI_density_z  = (Real *) malloc(n_los_full_z*sizeof(Real));
  full_velocity_z    = (Real *) malloc(n_los_full_z*sizeof(Real));
  full_temperature_z = (Real *) malloc(n_los_full_z*sizeof(Real));
  full_optical_depth_z = (Real *) malloc(n_los_full_z*sizeof(Real));
  full_vel_Hubble_z  = (Real *) malloc(n_los_full_z*sizeof(Real));  
  transmitted_flux_z = (Real *) malloc(nz_total*sizeof(Real));


  
  #endif
  
   
   
  
  chprintf(" Lya Statistics Initialized.\n");
  
  
  
}














#endif
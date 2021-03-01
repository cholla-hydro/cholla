#ifdef ANALYSIS
#ifdef LYA_STATISTICS

#include"analysis.h"
#include"../io.h"
#include <unistd.h>
#include <complex.h>

#ifdef MPI_CHOLLA
#include"../mpi_routines.h"
#endif

// #define PRINT_ANALYSIS_LOG

int Locate_Index( Real val, Real *values, int N ){
  // Find the index such that  values[index] < val < values[index+1]
  // Values has to be sorted 
  if ( val < values[0] )   return -2;
  if ( val > values[N-1] ) return -1;
  
  int index = 0;
  while ( index < N ){
    if ( val < values[index] ) break;
    index += 1;
  }
  
  if ( val < values[index-1] ){
    chprintf( "ERROR; Value less than left edge:  val=%f    left=%f \n", val, values[index-1] );
    exit(-1);
  }
  if ( val > values[index] ){
    chprintf( "ERROR; Value grater than right edge:  val=%f    right=%f \n", val, values[index] );
    exit(-1);
  }
  
  // chprintf( " %d:    %e   %e   %e \n ", index, values[index-1], val, values[index]);
  return index-1;
  
}

void Analysis_Module::Clear_Power_Spectrum_Measurements( void ){
  
  MPI_Barrier( world );
  
  // chprintf( "Cleared Power Spectrum cache \n ");
  free( hist_k_edges_x );
  free( hist_PS_x );
  free( hist_n_x );  
  free( ps_root_x ); 
  free( ps_global_x );

  free( hist_k_edges_y );
  free( hist_PS_y );
  free( hist_n_y );  
  free( ps_root_y ); 
  free( ps_global_y );  
  
  free( hist_k_edges_z );
  free( hist_PS_z );
  free( hist_n_z );    
  free( ps_root_z ); 
  free( ps_global_z );
  
  free( k_ceters );
  free( ps_mean );
  
}

void Grid3D::Initialize_Power_Spectrum_Measurements( int axis ){
  
  int n_los, n_fft; 
  Real Lbox, delta_x;
  Real *k_vals;
  
  if ( axis == 0 ){
    Analysis.n_PS_processed_x = 0;
    n_los               = Analysis.nx_total;
    n_fft               = Analysis.n_fft_x;
    Lbox                = Analysis.Lbox_x;
    delta_x             = Analysis.dx;
    k_vals              = Analysis.k_vals_x;
  }
  
  if ( axis == 1 ){
    Analysis.n_PS_processed_y = 0;
    n_los               = Analysis.ny_total;
    n_fft               = Analysis.n_fft_y;
    Lbox                = Analysis.Lbox_y;
    delta_x             = Analysis.dy;
    k_vals              = Analysis.k_vals_y;
  }
  
  if ( axis == 2 ){
    Analysis.n_PS_processed_z = 0;
    n_los               = Analysis.nz_total;
    n_fft               = Analysis.n_fft_z;
    Lbox                = Analysis.Lbox_z;
    delta_x             = Analysis.dz;
    k_vals              = Analysis.k_vals_z;
  }
  
  
  // Get Cosmological variables
  Real H, current_a, L_proper, dx_proper, dv_Hubble; 
  current_a = Cosmo.current_a;
  L_proper = Lbox * current_a / Cosmo.cosmo_h;
  dx_proper = delta_x * current_a / Cosmo.cosmo_h;
  H = Cosmo.Get_Hubble_Parameter( current_a );
  dv_Hubble = H * dx_proper; // km/s
  
  
  // Compute the K values 
  for ( int i=0; i<n_fft; i++ ){
    k_vals[i] = 2 * M_PI * i / ( n_los * dv_Hubble );
    // if ( axis == 0 ) chprintf( "k: %f \n", k_vals[i]  );
  }
  
  Real k_val, k_min,  k_max, d_log_k, k_start;
  d_log_k = Analysis.d_log_k;
  k_min = log10( k_vals[1] );
  k_max = log10( k_vals[n_fft-1] );
  k_start = log10( 0.99 * k_vals[1] );
  
  
  if ( d_log_k == 0 ){
    chprintf( "ERROR: d_log_k = 0    Set  lya_Pk_d_log_k in the parameter file \n"  );
    exit(-1);
  } 
  
  // if ( axis == 0 ) chprintf( "dv_Hubble: %f \n", dv_Hubble  );
  // if ( axis == 0 ) chprintf( "k min : %f \n", k_min  );
  // if ( axis == 0 ) chprintf( "k max : %f \n", k_max  );
  
  k_val = k_start;
  int n_hist_edges = 1;
  while ( k_val < k_max ){
    n_hist_edges += 1;
    k_val += d_log_k ;
  }
    
  Real *hist_k_edges;
  Real *hist_PS;
  Real *hist_n;
  Real *ps_root;
  Real *ps_global;
  
  int n_bins = n_hist_edges - 1; 
  // chprintf( " n bins : %d \n", n_bins  );
  hist_k_edges = (Real *) malloc(n_hist_edges*sizeof(Real));
  hist_PS     = (Real *) malloc(n_bins*sizeof(Real));
  hist_n      = (Real *) malloc(n_bins*sizeof(Real));
  ps_root     = (Real *) malloc(n_bins*sizeof(Real));
  ps_global   = (Real *) malloc(n_bins*sizeof(Real));
  
  k_val = k_start;
  for ( int bin_id=0; bin_id<n_hist_edges; bin_id++ ){
    hist_k_edges[bin_id] = pow( 10, k_val );
    k_val += d_log_k; 
  }
  
  for ( int bin_id=0; bin_id<n_bins; bin_id++ ){
    ps_root[bin_id] = 0;
    ps_global[bin_id] = 0;
  }
  
  if ( axis == 0 ){
    Analysis.n_hist_edges_x = n_hist_edges;
    Analysis.hist_k_edges_x = hist_k_edges;
    Analysis.hist_PS_x     = hist_PS;
    Analysis.hist_n_x      = hist_n;
    Analysis.ps_root_x     = ps_root;
    Analysis.ps_global_x   = ps_global;    
  } 
  
  if ( axis == 1 ){
    Analysis.n_hist_edges_y = n_hist_edges;
    Analysis.hist_k_edges_y = hist_k_edges;
    Analysis.hist_PS_y     = hist_PS;
    Analysis.hist_n_y      = hist_n;    
    Analysis.ps_root_y     = ps_root;
    Analysis.ps_global_y   = ps_global;    
  } 
  
  if ( axis == 2 ){
    Analysis.n_hist_edges_z = n_hist_edges;
    Analysis.hist_k_edges_z = hist_k_edges;
    Analysis.hist_PS_z     = hist_PS;
    Analysis.hist_n_z      = hist_n;
    Analysis.ps_root_z     = ps_root;
    Analysis.ps_global_z   = ps_global;    
  } 
  
  
  // Create array  for global PS
  if ( axis == 2 ){
    if ( Analysis.n_hist_edges_x != Analysis.n_hist_edges_y || Analysis.n_hist_edges_x != Analysis.n_hist_edges_z ){
      chprintf( "ERROR: PS Histogram sizes dont match \n");
      exit(-1);
    } 
    else{
      Analysis.ps_mean  = (Real *) malloc(n_bins*sizeof(Real));
      Analysis.k_ceters = (Real *) malloc(n_bins*sizeof(Real));
      
      for (int bin_id=0; bin_id<n_bins; bin_id++ ){
        Analysis.k_ceters[bin_id] = sqrt( Analysis.hist_k_edges_x[bin_id] * Analysis.hist_k_edges_x[bin_id+1]  );
      }
    }
    
  }
    
  // if ( axis == 0 ){
  //   for ( int bin_id=0; bin_id<n_hist_edges; bin_id++ ){ 
  //     chprintf( "%f \n", hist_k_edges[bin_id]);
  //   }
  // }

  
}



void Grid3D::Compute_Flux_Power_Spectrum_Skewer( int skewer_id, int axis ){
  
  bool am_I_root;
  int n_los, n_fft, n_hist_edges;
  Real Lbox, delta_x;
  Real *delta_F;
  fftw_complex *fft_delta_F; 
  Real *fft2_delta_F;
  fftw_plan fftw_plan;
  Real *k_vals;
  Real *hist_k_edges;
  Real *hist_PS;
  Real *hist_n;
  Real *ps_root;
  Real *skewers_transmitted_flux;
  
  if ( axis == 0 ){
    am_I_root           = Analysis.am_I_root_x;
    n_los               = Analysis.nx_total;
    n_fft               = Analysis.n_fft_x;
    n_hist_edges        = Analysis.n_hist_edges_x;
    Lbox                = Analysis.Lbox_x;
    delta_x             = Analysis.dx;
    delta_F             = Analysis.delta_F_x;
    fft_delta_F         = Analysis.fft_delta_F_x;
    fft2_delta_F        = Analysis.fft2_delta_F_x;
    k_vals              = Analysis.k_vals_x;
    fftw_plan           = Analysis.fftw_plan_x;
    hist_k_edges        = Analysis.hist_k_edges_x;
    hist_PS             = Analysis.hist_PS_x;
    hist_n              = Analysis.hist_n_x;
    ps_root             = Analysis.ps_root_x;
    skewers_transmitted_flux    = Analysis.skewers_transmitted_flux_HI_x;
  }
  
  if ( axis == 1 ){
    am_I_root           = Analysis.am_I_root_y;
    n_los               = Analysis.ny_total;
    n_fft               = Analysis.n_fft_y;
    n_hist_edges        = Analysis.n_hist_edges_y;
    Lbox                = Analysis.Lbox_y;
    delta_x             = Analysis.dy;
    delta_F             = Analysis.delta_F_y;
    fft_delta_F         = Analysis.fft_delta_F_y;
    fft2_delta_F        = Analysis.fft2_delta_F_y;
    k_vals              = Analysis.k_vals_y;
    fftw_plan           = Analysis.fftw_plan_y;
    hist_k_edges        = Analysis.hist_k_edges_y;
    hist_PS             = Analysis.hist_PS_y;
    hist_n              = Analysis.hist_n_y;
    ps_root             = Analysis.ps_root_y;
    skewers_transmitted_flux    = Analysis.skewers_transmitted_flux_HI_y;
  }
  
  if ( axis == 2 ){
    am_I_root           = Analysis.am_I_root_z;
    n_los               = Analysis.nz_total;
    n_fft               = Analysis.n_fft_z;
    n_hist_edges        = Analysis.n_hist_edges_z;
    Lbox                = Analysis.Lbox_z;
    delta_x             = Analysis.dz;
    delta_F             = Analysis.delta_F_z;
    fft_delta_F         = Analysis.fft_delta_F_z;
    fft2_delta_F        = Analysis.fft2_delta_F_z;
    k_vals              = Analysis.k_vals_z;
    fftw_plan           = Analysis.fftw_plan_z;
    hist_k_edges        = Analysis.hist_k_edges_z;
    hist_PS             = Analysis.hist_PS_z;
    hist_n              = Analysis.hist_n_z;
    ps_root             = Analysis.ps_root_z;
    skewers_transmitted_flux    = Analysis.skewers_transmitted_flux_HI_z;
  }
  
  int n_bins = n_hist_edges - 1;
  
  if ( !am_I_root ){
    for ( int i=0; i<n_bins; i++ ){
      ps_root[i] = 0;
    }
    return;
  }  
  
  
  for ( int los_id=0; los_id<n_los; los_id++ ){
    delta_F[los_id] = skewers_transmitted_flux[ skewer_id * n_los + los_id] / Analysis.Flux_mean_HI;
  }
  
  // Compute the r2c FFT
  fftw_execute( fftw_plan );
  
  // Get Cosmological variables
  Real H, current_a, L_proper, dx_proper, dv_Hubble; 
  current_a = Cosmo.current_a;
  L_proper = Lbox * current_a / Cosmo.cosmo_h;
  dx_proper = delta_x * current_a / Cosmo.cosmo_h;
  H = Cosmo.Get_Hubble_Parameter( current_a );
  dv_Hubble = H * dx_proper; // km/s
  
  // Compute the amplitude of the FFT
  for ( int i=0; i<n_fft; i++ ){
    fft2_delta_F[i] = (fft_delta_F[i][0]*fft_delta_F[i][0] + fft_delta_F[i][1]*fft_delta_F[i][1]) / n_los / n_los;
  }
  
  // Arrange in k-bins
  for ( int i=0; i<n_bins; i++ ){
    hist_PS[i] = 0;
    hist_n[i] = 0;
  }
  
  
  
  int bin_id;
  Real k_val;
  for ( int i=0; i<n_fft; i++ ){
    k_val = k_vals[i];
    if ( k_val == 0 ) continue;
    bin_id = Locate_Index( k_val, hist_k_edges, n_hist_edges );
    if ( bin_id < 0 ) chprintf( " %d:   %e    %e   %e \n", bin_id, hist_k_edges[0], k_val, hist_k_edges[1] );
    if ( bin_id < 0 || bin_id >= n_bins ) continue;
    hist_PS[bin_id] += fft2_delta_F[i]; 
    hist_n[bin_id]  += 1;
  }
  
  int hist_sum = 0;
  for ( int i=0; i<n_bins; i++ ){
    hist_sum += hist_n[i];
  }
  
  // Add skewer PS to root PS
  Real PS_bin_val;
  for ( int i=0; i<n_bins; i++ ){
    if ( hist_n[i] == 0 ) PS_bin_val = 0;   
    else PS_bin_val = hist_PS[i] / hist_n[i] * ( H*L_proper);
    ps_root[i] += PS_bin_val;
  }
  
  if ( hist_sum != n_fft-1 ){
    printf("ERROR: Histogram sum doesn't match n_pfft:  sum=%d    n_fft=%d \n", hist_sum, n_fft-1);
    exit(-1);
  }
  
  
  if ( axis == 0 ) Analysis.n_PS_processed_x += 1;
  if ( axis == 1 ) Analysis.n_PS_processed_y += 1;
  if ( axis == 2 ) Analysis.n_PS_processed_z += 1;
  
  
}


void Analysis_Module::Reduce_Power_Spectrum_Axis( int axis ){
  
  int n_root, n_bins;
  Real *ps_root;
  Real *ps_global;
  int *n_axis;
  
  if ( axis == 0 ){
    n_bins    = n_hist_edges_x - 1;
    n_root    = n_PS_processed_x;
    ps_root   = ps_root_x;
    ps_global = ps_global_x;
    n_axis    = &n_PS_axis_x;
  }
  
  if ( axis == 1 ){
    n_bins    = n_hist_edges_y - 1;
    n_root    = n_PS_processed_y;
    ps_root   = ps_root_y;
    ps_global = ps_global_y;
    n_axis    = &n_PS_axis_y;
  }
  
  if ( axis == 2 ){
    n_bins    = n_hist_edges_z - 1;
    n_root    = n_PS_processed_z;
    ps_root   = ps_root_z;
    ps_global = ps_global_z;
    n_axis    = &n_PS_axis_z;
  }
  
  MPI_Allreduce( ps_root, ps_global, n_bins, MPI_CHREAL, MPI_SUM, world );
  MPI_Allreduce( &n_root, n_axis, 1, MPI_INT, MPI_SUM, world );
  
  
  chprintf( "  N_Skewers_Processed: %d \n", *n_axis );
  
}


void Analysis_Module::Reduce_Power_Spectrum_Global( ){
  
  int n_PS_total = n_PS_axis_x + n_PS_axis_y + n_PS_axis_z;
  if ( n_hist_edges_x != n_hist_edges_y || n_hist_edges_x != n_hist_edges_z ){
    chprintf( "ERROR: PS Histogram sizes dont match \n");
    exit(-1);
  } 
  
  int n_bins = n_hist_edges_x - 1;
   
  for (int bin_id=0; bin_id<n_bins; bin_id++ ){
    ps_mean[bin_id] = ( ps_global_x[bin_id] + ps_global_y[bin_id] + ps_global_z[bin_id] ) / n_PS_total;
  }
  
  chprintf( " PS Bins: %d     N_Skewers_Processed: %d \n", n_bins, n_PS_total );
  
  // for (int bin_id=0; bin_id<n_bins; bin_id++ ){
  //   chprintf( " %e   %e  \n", k_ceters[bin_id], ps_mean[bin_id] *k_ceters[bin_id] / M_PI);  
  // }
  
}


void Analysis_Module::Reduce_Lya_Mean_Flux_Global(){
   
  n_skewers_processed = n_skewers_processed_x + n_skewers_processed_y + n_skewers_processed_z;
  Flux_mean_HI   = ( Flux_mean_HI_x  *n_skewers_processed_x + Flux_mean_HI_y  *n_skewers_processed_y + Flux_mean_HI_z  *n_skewers_processed_z  ) / n_skewers_processed;;
  Flux_mean_HeII = ( Flux_mean_HeII_x*n_skewers_processed_x + Flux_mean_HeII_y*n_skewers_processed_y + Flux_mean_HeII_z*n_skewers_processed_z  ) / n_skewers_processed;;
  chprintf( " N_Skewers_Processed  Global: %d    F_Mean_HI: %e  F_Mean_HeII: %e\n", n_skewers_processed, Flux_mean_HI, Flux_mean_HeII  );
}

void Analysis_Module::Reduce_Lya_Mean_Flux_Axis( int axis ){
  
  int  *n_skewers_processed;
  int  *n_skewers_processed_root;
  Real *Flux_mean_HI;
  Real *Flux_mean_HeII;
  Real *Flux_mean_root_HI;
  Real *Flux_mean_root_HeII;
  
  if ( axis == 0 ){
    n_skewers_processed      = &n_skewers_processed_x;
    n_skewers_processed_root = &n_skewers_processed_root_x;
    Flux_mean_HI             = &Flux_mean_HI_x;
    Flux_mean_HeII           = &Flux_mean_HeII_x;
    Flux_mean_root_HI        = &Flux_mean_root_HI_x;
    Flux_mean_root_HeII      = &Flux_mean_root_HeII_x;
  }

  if ( axis == 1 ){
    n_skewers_processed      = &n_skewers_processed_y;
    n_skewers_processed_root = &n_skewers_processed_root_y;
    Flux_mean_HI             = &Flux_mean_HI_y;
    Flux_mean_HeII           = &Flux_mean_HeII_y;
    Flux_mean_root_HI        = &Flux_mean_root_HI_y;
    Flux_mean_root_HeII      = &Flux_mean_root_HeII_y;
  }

  if ( axis == 2 ){
    n_skewers_processed      = &n_skewers_processed_z;
    n_skewers_processed_root = &n_skewers_processed_root_z;
    Flux_mean_HI             = &Flux_mean_HI_z;
    Flux_mean_HeII           = &Flux_mean_HeII_z;
    Flux_mean_root_HI        = &Flux_mean_root_HI_z;
    Flux_mean_root_HeII      = &Flux_mean_root_HeII_z;
  }


  
  #ifdef MPI_CHOLLA
  
  #ifdef PRINT_ANALYSIS_LOG
  for ( int i=0; i<nproc; i++ ){
    if (procID == i) printf("   procID:%d   Flux_HI_Sum: %e     N_Skewers_Processed: %d \n", procID, (*Flux_mean_root_HI), *n_skewers_processed_root );
    MPI_Barrier(world);
    sleep(1);
  }
  #endif
  
  MPI_Allreduce( Flux_mean_root_HI,   Flux_mean_HI,   1, MPI_CHREAL, MPI_SUM, world );
  MPI_Allreduce( Flux_mean_root_HeII, Flux_mean_HeII, 1, MPI_CHREAL, MPI_SUM, world );
  MPI_Allreduce( n_skewers_processed_root, n_skewers_processed, 1, MPI_INT, MPI_SUM, world );

  #else
  
  *Flux_mean_HI   = *Flux_mean_root_HI;
  *Flux_mean_HeII = *Flux_mean_root_HeII;
  *n_skewers_processed = *n_skewers_processed_root;

  #endif
  
  *Flux_mean_HI   = *Flux_mean_HI   / *n_skewers_processed;
  *Flux_mean_HeII = *Flux_mean_HeII / *n_skewers_processed;
  chprintf( "  N_Skewers_Processed: %d  Flux Mean HI: %e  Flux_mean_HeII: %e \n", *n_skewers_processed, *Flux_mean_HI, *Flux_mean_HeII );

  
}


void Analysis_Module::Compute_Lya_Mean_Flux_Skewer( int skewer_id, int axis ){
  
  bool am_I_root;
  int  n_los;
  int  *n_skewers_processed_root;
  Real *F_mean_root_HI;
  Real *F_mean_root_HeII;
  Real *skewers_transmitted_flux_HI;
  Real *skewers_transmitted_flux_HeII;
  
  if ( axis == 0 ){
    am_I_root = am_I_root_x;
    n_los = nx_total;
    F_mean_root_HI   = &Flux_mean_root_HI_x;
    F_mean_root_HeII = &Flux_mean_root_HeII_x;
    skewers_transmitted_flux_HI   = skewers_transmitted_flux_HI_x;
    skewers_transmitted_flux_HeII = skewers_transmitted_flux_HeII_x;
    n_skewers_processed_root = &n_skewers_processed_root_x; 
  }
  
  if ( axis == 1 ){
    am_I_root = am_I_root_y;
    n_los = ny_total;
    F_mean_root_HI   = &Flux_mean_root_HI_y;
    F_mean_root_HeII = &Flux_mean_root_HeII_y;
    skewers_transmitted_flux_HI   = skewers_transmitted_flux_HI_y;
    skewers_transmitted_flux_HeII = skewers_transmitted_flux_HeII_y;
    n_skewers_processed_root = &n_skewers_processed_root_y;
  }
  
  if ( axis == 2 ){
    am_I_root = am_I_root_z;
    n_los = nz_total;
    F_mean_root_HI   = &Flux_mean_root_HI_z;
    F_mean_root_HeII = &Flux_mean_root_HeII_z;
    skewers_transmitted_flux_HI   = skewers_transmitted_flux_HI_z;
    skewers_transmitted_flux_HeII = skewers_transmitted_flux_HeII_z;
    n_skewers_processed_root = &n_skewers_processed_root_z;
  }
  
  if ( !am_I_root ) return;
  
  Real F_mean_HI, F_mean_HeII;
  F_mean_HI   = 0;
  F_mean_HeII = 0;
  for ( int los_id=0; los_id<n_los; los_id++ ){
    F_mean_HI += skewers_transmitted_flux_HI  [ skewer_id*n_los + los_id] / n_los;
    F_mean_HeII += skewers_transmitted_flux_HeII[ skewer_id*n_los + los_id] / n_los;
  }
  
  *F_mean_root_HI   += F_mean_HI;
  *F_mean_root_HeII += F_mean_HeII;
  *n_skewers_processed_root += 1;
  
}


  
void Analysis_Module::Initialize_Lya_Statistics_Measurements( int axis ){
  
  if ( axis == 0 ){
    n_skewers_processed_root_x = 0;
    Flux_mean_root_HI_x = 0;
    Flux_mean_root_HeII_x = 0;
  }
  
  if ( axis == 1 ){
    n_skewers_processed_root_y = 0;
    Flux_mean_root_HI_y = 0;
    Flux_mean_root_HeII_y = 0;
  }
  
  if ( axis == 2 ){
    n_skewers_processed_root_z = 0;
    Flux_mean_root_HI_z = 0;
    Flux_mean_root_HeII_z = 0;
  }
  
}

void Grid3D::Compute_Transmitted_Flux_Skewer( int skewer_id, int axis ){
  
  int n_los_full, n_los_total, n_ghost;
  bool am_I_root;
  Real *full_density_HI;
  Real *full_density_HeII;
  Real *full_velocity;
  Real *full_temperature;
  Real *full_optical_depth_HI;
  Real *full_optical_depth_HeII;
  Real *full_vel_Hubble;
  Real *skewers_transmitted_flux_HI;
  Real *skewers_transmitted_flux_HeII;
  Real *skewers_HI_density_root;
  Real *skewers_HeII_density_root;
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
    full_density_HI           = Analysis.full_HI_density_x;
    full_density_HeII         = Analysis.full_HeII_density_x;
    full_velocity             = Analysis.full_velocity_x;
    full_temperature          = Analysis.full_temperature_x;
    full_optical_depth_HI     = Analysis.full_optical_depth_HI_x;
    full_optical_depth_HeII   = Analysis.full_optical_depth_HeII_x;
    full_vel_Hubble           = Analysis.full_vel_Hubble_x;
    skewers_HI_density_root   = Analysis.skewers_HI_density_root_x;
    skewers_HeII_density_root = Analysis.skewers_HeII_density_root_x;
    skewers_velocity_root     = Analysis.skewers_velocity_root_x; 
    skewers_temperature_root  = Analysis.skewers_temperature_root_x; 
    skewers_transmitted_flux_HI   = Analysis.skewers_transmitted_flux_HI_x;
    skewers_transmitted_flux_HeII = Analysis.skewers_transmitted_flux_HeII_x;
  }
  
  if ( axis == 1 ){
    Lbox    = Analysis.Lbox_y;
    delta_x = Analysis.dy;  
    am_I_root   = Analysis.am_I_root_y;
    n_los_full  = Analysis.n_los_full_y;
    n_los_total = Analysis.ny_total;
    full_density_HI           = Analysis.full_HI_density_y;
    full_density_HeII         = Analysis.full_HeII_density_y;
    full_density_HI           = Analysis.full_HI_density_y;
    full_density_HeII         = Analysis.full_HeII_density_y;
    full_velocity             = Analysis.full_velocity_y;
    full_temperature          = Analysis.full_temperature_y;
    full_optical_depth_HI     = Analysis.full_optical_depth_HI_y;
    full_optical_depth_HeII   = Analysis.full_optical_depth_HeII_y;
    full_vel_Hubble           = Analysis.full_vel_Hubble_y;
    skewers_HI_density_root   = Analysis.skewers_HI_density_root_y;
    skewers_HeII_density_root = Analysis.skewers_HeII_density_root_y;
    skewers_velocity_root     = Analysis.skewers_velocity_root_y; 
    skewers_temperature_root  = Analysis.skewers_temperature_root_y;
    skewers_transmitted_flux_HI   = Analysis.skewers_transmitted_flux_HI_y;
    skewers_transmitted_flux_HeII = Analysis.skewers_transmitted_flux_HeII_y;
  }
  
  if ( axis == 2 ){
    Lbox    = Analysis.Lbox_z;
    delta_x = Analysis.dz;
    am_I_root   = Analysis.am_I_root_z;
    n_los_full  = Analysis.n_los_full_z;
    n_los_total = Analysis.nz_total;
    full_density_HI           = Analysis.full_HI_density_z;
    full_density_HeII         = Analysis.full_HeII_density_z;
    full_velocity             = Analysis.full_velocity_z;
    full_temperature          = Analysis.full_temperature_z;
    full_optical_depth_HI     = Analysis.full_optical_depth_HI_z;
    full_optical_depth_HeII   = Analysis.full_optical_depth_HeII_z;
    full_vel_Hubble           = Analysis.full_vel_Hubble_z;
    skewers_HI_density_root   = Analysis.skewers_HI_density_root_z;
    skewers_HeII_density_root = Analysis.skewers_HeII_density_root_z;
    skewers_velocity_root     = Analysis.skewers_velocity_root_z; 
    skewers_temperature_root  = Analysis.skewers_temperature_root_z;
    skewers_transmitted_flux_HI   = Analysis.skewers_transmitted_flux_HI_z;
    skewers_transmitted_flux_HeII = Analysis.skewers_transmitted_flux_HeII_z;
  }
  
  if ( !am_I_root ) return;
  
  // printf( "  Computing Skewer ID: %d \n", skewer_id );
  
  Real density_HI, density_HeII, velocity, temperature, Msun, kpc, Mp, kpc3, Me, e_charge, c, Kb;
  
  // Constants in CGS
  Kb   = 1.38064852e-16; //g (cm/s)^2 K-1
  Msun = 1.98847e33;     //g 
  Mp   = 1.6726219e-24;  //g 
  Me   = 9.10938356e-28; //g
  c    = 2.99792458e10;    //cm/s
  kpc  = 3.0857e21;        //cm
  kpc3 = kpc * kpc * kpc;
  e_charge =  4.8032e-10;  // cm^3/2 g^1/2 s^-1 
  

   
  // Fill the Real cells first
  for (int los_id=0; los_id<n_los_total; los_id++ ){
    density_HI   = skewers_HI_density_root   [ skewer_id*n_los_total + los_id ];
    density_HeII = skewers_HeII_density_root [ skewer_id*n_los_total + los_id ];
    velocity     = skewers_velocity_root     [ skewer_id*n_los_total + los_id ];
    temperature  = skewers_temperature_root  [ skewer_id*n_los_total + los_id ];
    full_density_HI   [los_id + n_ghost] = density_HI;
    full_density_HeII [los_id + n_ghost] = density_HeII;
    full_velocity     [los_id + n_ghost] = velocity;
    full_temperature  [los_id + n_ghost] = temperature;
  }
  
  // Fill the ghost cells
  for ( int los_id=0; los_id<n_ghost; los_id++ ){
    full_density_HI  [los_id] = full_density_HI  [n_los_total+los_id];
    full_density_HeII[los_id] = full_density_HeII[n_los_total+los_id];
    full_velocity    [los_id] = full_velocity    [n_los_total+los_id];
    full_temperature [los_id] = full_temperature [n_los_total+los_id];
    full_density_HI  [n_los_total+n_ghost+los_id] = full_density_HI  [n_ghost+los_id];
    full_density_HeII[n_los_total+n_ghost+los_id] = full_density_HeII[n_ghost+los_id];
    full_velocity    [n_los_total+n_ghost+los_id] = full_velocity    [n_ghost+los_id];
    full_temperature [n_los_total+n_ghost+los_id] = full_temperature [n_ghost+los_id];
  } 
  

  Real dens_factor, dens_factor_HI, dens_factor_HeII, vel_factor;
  dens_factor = 1. / ( Cosmo.current_a * Cosmo.current_a * Cosmo.current_a ) * Cosmo.cosmo_h * Cosmo.cosmo_h; 
  dens_factor_HI   = dens_factor * Msun / ( kpc3 ) / Mp;
  dens_factor_HeII = dens_factor * Msun / ( kpc3 ) / (4*Mp);
  vel_factor = 1e5; //cm/s
  
  // Get Cosmological variables
  Real H, current_a, L_proper, dx_proper, dv_Hubble; 
  Real H_cgs, Lya_lambda_HI, f_12, Lya_sigma_HI;
  Real Lya_lambda_HeII, Lya_sigma_HeII;
  current_a = Cosmo.current_a;
  L_proper = Lbox * current_a / Cosmo.cosmo_h;
  dx_proper = delta_x * current_a / Cosmo.cosmo_h;
  H = Cosmo.Get_Hubble_Parameter( current_a );
  dv_Hubble = H * dx_proper * vel_factor; // cm/s
  
  // Fill the Hubble velocity with ghost cells
  for ( int los_id=0; los_id<n_los_full; los_id++ ){
    full_vel_Hubble[los_id] = ( los_id - n_ghost + 0.5 ) * dv_Hubble;
  }
  
  Lya_lambda_HI   = 1.21567e-5;          // cm  Rest wave length of the Lyman Alpha Transition Hydrogen
  Lya_lambda_HeII = Lya_lambda_HI / 4;   // cm  Rest wave length of the Lyman Alpha Transition Helium II
  f_12 =  0.416;                         // Lya transition Oscillator strength
  H_cgs = H * 1e5 / kpc;
  
  Lya_sigma_HI   = M_PI * e_charge * e_charge / Me / c * Lya_lambda_HI   * f_12 / H_cgs;
  Lya_sigma_HeII = M_PI * e_charge * e_charge / Me / c * Lya_lambda_HeII * f_12 / H_cgs;   
    
  //Compute the optical depth
  Real b_HI_j,   n_HI_j; 
  Real b_HeII_j, n_HeII_j;
  Real tau_HI_i, tau_HeII_i; 
  Real vel_i, vel_j, y_l, y_r;
  for ( int i=0; i<n_los_full; i++ ){
    vel_i = full_vel_Hubble[i];
    tau_HI_i = 0;
    tau_HeII_i = 0;
    for ( int j=0; j<n_los_full; j++ ){
      n_HI_j   = full_density_HI[j]   * dens_factor_HI;
      n_HeII_j = full_density_HeII[j] * dens_factor_HeII;
      vel_j    =  full_vel_Hubble[j] + ( full_velocity[j] * vel_factor );
      b_HI_j   = sqrt( 2 * Kb / Mp     * full_temperature[j] );
      b_HeII_j = sqrt( 2 * Kb / (4*Mp) * full_temperature[j] );
      y_l = ( vel_i - 0.5*dv_Hubble - vel_j ) / b_HI_j;
      y_r = ( vel_i + 0.5*dv_Hubble - vel_j ) / b_HI_j;
      tau_HI_i   += n_HI_j * ( erf(y_r) - erf(y_l) ) / 2;
      y_l = ( vel_i - 0.5*dv_Hubble - vel_j ) / b_HeII_j;
      y_r = ( vel_i + 0.5*dv_Hubble - vel_j ) / b_HeII_j;
      tau_HeII_i += n_HeII_j * ( erf(y_r) - erf(y_l) ) / 2;
    }
    tau_HI_i   *= Lya_sigma_HI;
    tau_HeII_i *= Lya_sigma_HeII;
    full_optical_depth_HI[i]   = tau_HI_i;
    full_optical_depth_HeII[i] = tau_HeII_i;    
  }
  
  // Compute the transmitted_flux
  for ( int los_id=0; los_id<n_los_total; los_id++ ){
    skewers_transmitted_flux_HI  [skewer_id*n_los_total + los_id] = exp( -full_optical_depth_HI  [los_id + n_ghost] );
    skewers_transmitted_flux_HeII[skewer_id*n_los_total + los_id] = exp( -full_optical_depth_HeII[los_id + n_ghost] );
  }
  
  
  
}

void Analysis_Module::Transfer_Skewers_Data( int axis ){
  
  bool am_I_root;
  int n_skewers, n_los_local, n_los_total, root_id;
  Real *skewers_HI_density_local;
  Real *skewers_HeII_density_local;
  Real *skewers_velocity_local;
  Real *skewers_temperature_local;
  Real *skewers_HI_density_root;
  Real *skewers_HeII_density_root;
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
    skewers_HI_density_root = skewers_HI_density_root_x;
    skewers_HeII_density_local = skewers_HeII_density_local_x;
    skewers_HeII_density_root  = skewers_HeII_density_root_x;
    skewers_velocity_local = skewers_velocity_local_x;
    skewers_temperature_local = skewers_temperature_local_x;
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
    skewers_HeII_density_local = skewers_HeII_density_local_y;
    skewers_HeII_density_root  = skewers_HeII_density_root_y;
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
    skewers_HeII_density_local = skewers_HeII_density_local_z;
    skewers_HeII_density_root  = skewers_HeII_density_root_z;
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
  
  Real HI_density, HeII_density, velocity, temperature;
  
  #ifdef MPI_CHOLLA
  if ( am_I_root ){
    
    if ( root_id != procID ){
      printf("ERROR: Root ID doesn't match procID\n" );
      exit(-1);
    }
    
    for ( int skewer_id=0; skewer_id<n_skewers; skewer_id++){
      for ( int los_id=0; los_id<n_los_local; los_id++){
        HI_density   = skewers_HI_density_local[skewer_id*n_los_local + los_id];
        HeII_density = skewers_HeII_density_local[skewer_id*n_los_local + los_id];
        velocity     = skewers_velocity_local[skewer_id*n_los_local + los_id];
        temperature  = skewers_temperature_local[skewer_id*n_los_local + los_id];
        skewers_HI_density_root[skewer_id*n_los_total + los_id]   = HI_density;
        skewers_HeII_density_root[skewer_id*n_los_total + los_id] = HeII_density;
        skewers_velocity_root[skewer_id*n_los_total + los_id]     = velocity;
        skewers_temperature_root[skewer_id*n_los_total + los_id]  = temperature;
        // if ( skewer_id == 0 ) printf("Dens: %f   vel:%f   temp:%f \n", HI_density, velocity, temperature ); 
      }
    }
  
    int n_indices = mpi_indices.size();
    
    
    
    #ifdef PRINT_ANALYSIS_LOG
    printf( "  N MPI indices: %d \n", n_indices ); 
    #endif
    
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
      
      
      #ifdef PRINT_ANALYSIS_LOG
      printf("  Recieving Skewers From pID: %d\n", mpi_id );
      #endif
      
      MPI_Recv( skewers_HI_density_local,   n_skewers*n_los_local, MPI_CHREAL, mpi_id, 0, world, &mpi_status  );
      MPI_Recv( skewers_HeII_density_local, n_skewers*n_los_local, MPI_CHREAL, mpi_id, 0, world, &mpi_status  );
      MPI_Recv( skewers_velocity_local,     n_skewers*n_los_local, MPI_CHREAL, mpi_id, 1, world, &mpi_status  );
      MPI_Recv( skewers_temperature_local,  n_skewers*n_los_local, MPI_CHREAL, mpi_id, 2, world, &mpi_status  );
      
      
      
      for ( int skewer_id=0; skewer_id<n_skewers; skewer_id++){
        for ( int los_id=0; los_id<n_los_local; los_id++){
          skewers_HI_density_root   [skewer_id*n_los_total + indx*n_los_local + los_id] = skewers_HI_density_local   [skewer_id*n_los_local + los_id];
          skewers_HeII_density_root [skewer_id*n_los_total + indx*n_los_local + los_id] = skewers_HeII_density_local [skewer_id*n_los_local + los_id];
          skewers_velocity_root     [skewer_id*n_los_total + indx*n_los_local + los_id] = skewers_velocity_local     [skewer_id*n_los_local + los_id];
          skewers_temperature_root  [skewer_id*n_los_total + indx*n_los_local + los_id] = skewers_temperature_local  [skewer_id*n_los_local + los_id];
        }
      }
      
    }
  }
    
  else{
    
    MPI_Send( skewers_HI_density_local,   n_skewers*n_los_local, MPI_CHREAL, root_id, 0, world  );
    MPI_Send( skewers_HeII_density_local, n_skewers*n_los_local, MPI_CHREAL, root_id, 0, world  );
    MPI_Send( skewers_velocity_local,     n_skewers*n_los_local, MPI_CHREAL, root_id, 1, world  );
    MPI_Send( skewers_temperature_local,  n_skewers*n_los_local, MPI_CHREAL, root_id, 2, world  );
  } 
  
  
  MPI_Barrier( world );
  #endif
    
    
  #ifdef PRINT_ANALYSIS_LOG
  chprintf("  Skewers Data Transfered\n" );
  #endif
}



void Grid3D::Populate_Lya_Skewers_Local( int axis ){
  
  int nx_local, ny_local, nz_local, n_ghost;
  int nx_grid, ny_grid, nz_grid;
  int ni, nj, n_los, stride, n_skewers_local;
  Real *momentum_los;
  Real *HI_density_los;
  Real *HeII_density_los;
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
    HI_density_los    = Analysis.skewers_HI_density_local_x;
    HeII_density_los  = Analysis.skewers_HeII_density_local_x;
    velocity_los      = Analysis.skewers_velocity_local_x;
    temperature_los   = Analysis.skewers_temperature_local_x; 
  } 
  
  // Y axis
  if ( axis == 1 ){
    n_los = ny_local;
    ni    = nx_local;
    nj    = nz_local;
    n_skewers_local = Analysis.n_skewers_local_y;
    momentum_los = C.momentum_y;
    HI_density_los    = Analysis.skewers_HI_density_local_y;
    HeII_density_los  = Analysis.skewers_HeII_density_local_y;
    velocity_los      = Analysis.skewers_velocity_local_y;
    temperature_los   = Analysis.skewers_temperature_local_y;
  }
  
  // Z axis
  if ( axis == 2 ){
    n_los = nz_local;
    ni    = nx_local;
    nj    = ny_local;
    n_skewers_local = Analysis.n_skewers_local_z;
    momentum_los = C.momentum_z;
    HI_density_los    = Analysis.skewers_HI_density_local_z;
    HeII_density_los  = Analysis.skewers_HeII_density_local_z;
    velocity_los      = Analysis.skewers_velocity_local_z;
    temperature_los   = Analysis.skewers_temperature_local_z;
  } 
    
  int n_iter_i, n_iter_j, id_grid;   
  n_iter_i = ni / stride;
  n_iter_j = nj / stride;
  int id_i, id_j, skewer_id;
  Real density, HI_density, HeII_density, velocity, temperature ;
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
        HI_density    = Cool.fields.HI_density[id_grid]   * Cosmo.rho_0_gas;
        HeII_density  = Cool.fields.HeII_density[id_grid] * Cosmo.rho_0_gas;
        temperature   = Cool.temperature[id_grid];
        #else
        chprintf( "ERROR: Lya Statistics only supported for Grackle Cooling \n");
        exit(-1);
        #endif
        HI_density_los[skewer_id*n_los + id_los]   = HI_density;
        HeII_density_los[skewer_id*n_los + id_los] = HeII_density;
        velocity_los[skewer_id*n_los + id_los]     = velocity;
        temperature_los[skewer_id*n_los + id_los]  = temperature;
      }
      skewer_id += 1;
    }
  }  
  
  if ( skewer_id != n_skewers_local ){
    printf( "ERROR: Skewers numbers don't match.  ID: %d   N_skewers: %d \n ", skewer_id, n_skewers_local );
    exit(-1);
  }
  

}


void Analysis_Module::Initialize_Lya_Statistics( struct parameters *P ){
  
  
  chprintf(" Initializing Lya Statistics...\n");
  
  
  n_ghost_skewer = max( nx_total, ny_total );
  n_ghost_skewer = max( nz_total, n_ghost_skewer );
  n_ghost_skewer = 0.1 * n_ghost_skewer;
  n_los_full_x = nx_total + 2 * n_ghost_skewer;
  n_los_full_y = ny_total + 2 * n_ghost_skewer;
  n_los_full_z = nz_total + 2 * n_ghost_skewer;
  
  n_fft_x = nx_total/2 + 1;
  n_fft_y = ny_total/2 + 1;
  n_fft_z = nz_total/2 + 1;
   
  
  
  n_stride = P->lya_skewers_stride;
  chprintf("  Lya Skewers Stride: %d\n", n_stride );
  
  d_log_k = P->lya_Pk_d_log_k;
  chprintf("  Power Spectrum d_log_k: %f\n", d_log_k );
  
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
  
  skewers_HeII_density_local_x = (Real *) malloc(n_skewers_local_x*nx_local*sizeof(Real));
  skewers_HeII_density_local_y = (Real *) malloc(n_skewers_local_y*ny_local*sizeof(Real));
  skewers_HeII_density_local_z = (Real *) malloc(n_skewers_local_z*nz_local*sizeof(Real));
  
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
  
  #ifdef PRINT_ANALYSIS_LOG
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
  #endif
  
  if ( am_I_root_x ){
    skewers_HI_density_root_x    = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
    skewers_HeII_density_root_x  = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
    skewers_velocity_root_x      = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
    skewers_temperature_root_x   = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
    full_HI_density_x            = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_HeII_density_x          = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_velocity_x              = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_temperature_x           = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_optical_depth_HI_x      = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_optical_depth_HeII_x    = (Real *) malloc(n_los_full_x*sizeof(Real));
    full_vel_Hubble_x            = (Real *) malloc(n_los_full_x*sizeof(Real));
    skewers_transmitted_flux_HI_x   = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
    skewers_transmitted_flux_HeII_x = (Real *) malloc(n_skewers_local_x*nx_total*sizeof(Real));
    
    // Alocate Memory For Power Spectrum Calculation
    delta_F_x             = (Real *) malloc(nx_total*sizeof(Real));
    vel_Hubble_x          = (Real *) malloc(nx_total*sizeof(Real));
    fft_delta_F_x         = (fftw_complex*) fftw_malloc(n_fft_x*sizeof(fftw_complex));
    fft2_delta_F_x        = (Real *) malloc(n_fft_x*sizeof(Real));
    fftw_plan_x           = fftw_plan_dft_r2c_1d( nx_total, delta_F_x, fft_delta_F_x, FFTW_ESTIMATE);
  }
  k_vals_x              = (Real *) malloc(n_fft_x*sizeof(Real));
  
  if ( am_I_root_y ){
    skewers_HI_density_root_y    = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    skewers_HeII_density_root_y  = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    skewers_velocity_root_y      = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    skewers_temperature_root_y   = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    full_HI_density_y            = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_HeII_density_y          = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_velocity_y              = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_temperature_y           = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_optical_depth_HI_y      = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_optical_depth_HeII_y    = (Real *) malloc(n_los_full_y*sizeof(Real));
    full_vel_Hubble_y            = (Real *) malloc(n_los_full_y*sizeof(Real));
    skewers_transmitted_flux_HI_y   = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    skewers_transmitted_flux_HeII_y = (Real *) malloc(n_skewers_local_y*ny_total*sizeof(Real));
    
    // Alocate Memory For Power Spectrum Calculation
    delta_F_y             = (Real *) malloc(ny_total*sizeof(Real));
    vel_Hubble_y          = (Real *) malloc(ny_total*sizeof(Real));
    fft_delta_F_y         = (fftw_complex*) fftw_malloc(n_fft_y*sizeof(fftw_complex));
    fft2_delta_F_y        = (Real *) malloc(n_fft_y*sizeof(Real));
    fftw_plan_y           = fftw_plan_dft_r2c_1d( ny_total, delta_F_y, fft_delta_F_y, FFTW_ESTIMATE);
  }
  k_vals_y              = (Real *) malloc(n_fft_y*sizeof(Real));
  
  if ( am_I_root_z ){
    skewers_HI_density_root_z    = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    skewers_HeII_density_root_z  = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    skewers_velocity_root_z      = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    skewers_temperature_root_z   = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    full_HI_density_z            = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_HeII_density_z          = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_velocity_z              = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_temperature_z           = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_optical_depth_HI_z      = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_optical_depth_HeII_z    = (Real *) malloc(n_los_full_z*sizeof(Real));
    full_vel_Hubble_z            = (Real *) malloc(n_los_full_z*sizeof(Real));
    skewers_transmitted_flux_HI_z   = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    skewers_transmitted_flux_HeII_z = (Real *) malloc(n_skewers_local_z*nz_total*sizeof(Real));
    
    // Alocate Memory For Power Spectrum Calculation
    delta_F_z             = (Real *) malloc(nz_total*sizeof(Real));
    vel_Hubble_z          = (Real *) malloc(nz_total*sizeof(Real));
    fft_delta_F_z         = (fftw_complex*) fftw_malloc(n_fft_z*sizeof(fftw_complex));
    fft2_delta_F_z        = (Real *) malloc(n_fft_z*sizeof(Real));
    fftw_plan_z           = fftw_plan_dft_r2c_1d( nz_total, delta_F_z, fft_delta_F_z, FFTW_ESTIMATE);
  }
  k_vals_z              = (Real *) malloc(n_fft_z*sizeof(Real));
  
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


#endif //LYA_STATISTICS
#endif //ANALYSIS
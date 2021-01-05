#ifdef ANALYSIS

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include"../global.h"
#include <vector>
#include <fftw3.h>

using namespace std;

class Analysis_Module{
public:
  
  Real Lbox_x;
  Real Lbox_y;
  Real Lbox_z;
  
  Real xMin;
  Real yMin;
  Real zMin;
  
  Real dx;
  Real dy;
  Real dz;
  
  Real xMin_global;
  Real yMin_global;
  Real zMin_global;
  
  int nx_total;
  int ny_total;
  int nz_total;
  
  int nx_local;
  int ny_local;
  int nz_local;
  int n_ghost;
  
  int n_outputs;
  int next_output_indx;
  real_vector_t scale_outputs;
  Real next_output;
  bool Output_Now;
  int n_file;
  
  #ifdef COSMOLOGY
  Real current_z;
  #endif
  
  
  
  #ifdef PHASE_DIAGRAM
  int n_dens;
  int n_temp;
  Real temp_min;
  Real temp_max;
  Real dens_min;
  Real dens_max;
  float *phase_diagram;
  #ifdef MPI_CHOLLA
  float *phase_diagram_global;
  #endif
  #endif
  
  
  #ifdef LYA_STATISTICS
  int n_stride;
  int n_skewers_local_x;
  int n_skewers_local_y;
  int n_skewers_local_z;
  int n_skewers_total_x;
  int n_skewers_total_y;
  int n_skewers_total_z;
  
  int n_skewers_processed_root_x;
  int n_skewers_processed_root_y;
  int n_skewers_processed_root_z;
  
  int n_skewers_processed_x;
  int n_skewers_processed_y;
  int n_skewers_processed_z;
  
  int skewers_direction;
  int root_id_x;
  int root_id_y;
  int root_id_z;
  bool am_I_root_x;
  bool am_I_root_y;
  bool am_I_root_z;
  Real *skewers_HI_density_local_x;
  Real *skewers_HI_density_local_y;
  Real *skewers_HI_density_local_z;
  
  Real *skewers_velocity_local_x;
  Real *skewers_velocity_local_y;
  Real *skewers_velocity_local_z;
  
  Real *skewers_temperature_local_x;
  Real *skewers_temperature_local_y;
  Real *skewers_temperature_local_z;
  
  Real *skewers_HI_density_root_x;
  Real *skewers_HI_density_root_y;
  Real *skewers_HI_density_root_z;
  
  Real *skewers_velocity_root_x;
  Real *skewers_velocity_root_y;
  Real *skewers_velocity_root_z;
  
  Real *skewers_temperature_root_x;
  Real *skewers_temperature_root_y;
  Real *skewers_temperature_root_z;
  
  Real *full_HI_density_x;
  Real *full_HI_density_y;
  Real *full_HI_density_z;
  
  Real *full_velocity_x;
  Real *full_velocity_y;
  Real *full_velocity_z;
  
  Real *full_temperature_x;
  Real *full_temperature_y;
  Real *full_temperature_z;
  
  Real *full_optical_depth_x;
  Real *full_optical_depth_y;
  Real *full_optical_depth_z;
  
  Real *full_vel_Hubble_x;
  Real *full_vel_Hubble_y;
  Real *full_vel_Hubble_z;
  
  Real *transmitted_flux_x;
  Real *transmitted_flux_y;
  Real *transmitted_flux_z;
  
  Real Flux_mean_root_x;
  Real Flux_mean_root_y;
  Real Flux_mean_root_z;
  
  Real Flux_mean_x;
  Real Flux_mean_y;
  Real Flux_mean_z;
    
  Real Flux_mean;
  int n_skewers_processed;
    
  int n_ghost_skewer;
  int n_los_full_x;
  int n_los_full_y;
  int n_los_full_z;
  
  Real *vel_Hubble_x;
  Real *vel_Hubble_y;
  Real *vel_Hubble_z;
  Real *delta_F_x;
  Real *delta_F_y;
  Real *delta_F_z;
  
  Real d_log_k;
  int n_PS_processed_x;
  int n_PS_processed_y;
  int n_PS_processed_z;
  int n_PS_axis_x;
  int n_PS_axis_y;
  int n_PS_axis_z;
  int n_hist_edges_x;
  int n_hist_edges_y;
  int n_hist_edges_z;
  int n_fft_x;
  int n_fft_y;
  int n_fft_z;
  fftw_complex *fft_delta_F_x;
  fftw_complex *fft_delta_F_y;
  fftw_complex *fft_delta_F_z;
  Real *fft2_delta_F_x;
  Real *fft2_delta_F_y;
  Real *fft2_delta_F_z;
  Real *k_vals_x;
  Real *k_vals_y;
  Real *k_vals_z;
  fftw_plan fftw_plan_x;
  fftw_plan fftw_plan_y;
  fftw_plan fftw_plan_z;
  Real *hist_k_edges_x;
  Real *hist_k_edges_y;
  Real *hist_k_edges_z;
  Real *hist_PS_x;
  Real *hist_PS_y;
  Real *hist_PS_z;
  Real *hist_n_x;
  Real *hist_n_y;
  Real *hist_n_z;
  Real *ps_root_x;
  Real *ps_root_y;
  Real *ps_root_z;
  Real *ps_global_x;
  Real *ps_global_y;
  Real *ps_global_z;
  Real *ps_mean;
  Real *k_ceters;
  
  #ifdef MPI_CHOLLA
  Real *mpi_domain_boundary_x;
  Real *mpi_domain_boundary_y;
  Real *mpi_domain_boundary_z;
  vector<int> mpi_indices_x;
  vector<int> mpi_indices_y;
  vector<int> mpi_indices_z;  
  #endif
  
  #endif
  
  
  Analysis_Module( void );
  void Initialize( Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx_real, Real dy_real, Real dz_real, int n_ghost_hydro, Real z_now, struct parameters *P );
  void Reset(void);
  
  void Load_Scale_Outputs( struct parameters *P );
  void Set_Next_Scale_Output(  );

  
  
  #ifdef PHASE_DIAGRAM
  void Initialize_Phase_Diagram( struct parameters *P );
  #endif
  
  #ifdef LYA_STATISTICS
  void Initialize_Lya_Statistics( struct parameters *P );
  void Initialize_Lya_Statistics_Measurements( int axis );
  void Transfer_Skewers_Data( int axis );
  void Compute_Lya_Statistics_Skewer( int skewer_id, int axis );
  void Reduce_Lya_Statists_Axis( int axis );
  void Reduce_Lya_Statists_Global( void );
  void Clear_Power_Spectrum_Measurements( void );
  void Reduce_Power_Spectrum_Axis( int axis );
  void Reduce_Power_Spectrum_Global( );
  #endif
};



#endif
#endif
#ifdef COSMOLOGY

#ifndef COSMOLOGY_H
#define COSMOLOGY_H

#include <stdio.h>
#include <cmath>
#include "../global/global.h"
#include "../particles/particles_3D.h"
#include "../gravity/grav3D.h"
#include "power_spectrum.h"
#include "../fft/fft_3D.h"

class Cosmology
{
public:

  Real H0;
  Real Omega_M;
  Real Omega_L;
  Real Omega_K;
  Real Omega_b;

  Real cosmo_G;
  Real cosmo_h;
  Real current_z;
  Real current_a;
  Real max_delta_a;
  Real delta_a;

  Real r_0_dm;
  Real t_0_dm;
  Real v_0_dm;
  Real rho_0_dm;
  Real phi_0_dm;
  Real rho_mean_baryon;

  Real time_conversion;
  Real dt_secs;
  Real t_secs;

  // Real dens_avrg;

  Real r_0_gas;
  Real v_0_gas;
  Real t_0_gas;
  Real phi_0_gas;
  Real rho_0_gas;
  Real p_0_gas;
  Real e_0_gas;

  int n_outputs;
  int next_output_indx;
  real_vector_t scale_outputs;
  Real next_output;
  bool exit_now;
  
  bool generate_initial_conditions;
  struct Initial_Conditions{
  
    // For generating cosmological initial conditions
    
    int nx_local;
    int ny_local;
    int nz_local;
    
    int nx_total;
    int ny_total;
    int nz_total;
    
    Real *random_fluctiations;
    Real *rescaled_random_fluctiations_dm;
    Real *rescaled_random_fluctiations_gas;
    
    Cosmo_Power_Spectrum Power_Spectrum;
    
    FFT_3D FFT;
    
  } ICs;
   
  
  
  Cosmology( void );
  void Initialize( struct parameters *P, Grav3D &Grav, Particles_3D &Particles );

  void Load_Scale_Outputs( struct parameters *P );
  void Set_Scale_Outputs( struct parameters *P );

  void Set_Next_Scale_Output( );

  Real Get_Hubble_Parameter( Real a );

  Real Get_da_from_dt( Real dt );
  Real Get_dt_from_da( Real da );

};

#endif
#endif

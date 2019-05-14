#ifdef COSMOLOGY

#include"cosmology.h"
#include "../io.h"



Cosmology::Cosmology( void ){}

void Cosmology::Initialize( struct parameters *P, Grav3D &Grav, Particles_3D &Particles){
  
  chprintf( "Cosmological Simulation\n");
  
  H0 = P-> H0;
  cosmo_h = H0/100;
  H0 /= 1000;               //[km/s / kpc]
  Omega_M = P-> Omega_M;
  Omega_L = P-> Omega_L;
  Omega_K = 1 - ( Omega_M + Omega_L );
  
  if(strcmp(P->init, "Read_Grid")==0){
    // Read scale factor vaue from Particles
    current_z = Particles.current_z;
    current_a = Particles.current_a;
  }
  else{
    current_z = P->Init_redshift;
    current_a = 1. / ( current_z + 1 );
  }  
    

  // Set Scale factor in Gravity
  Grav.current_a = current_a;

  // Gravitational Constant in Cosmological Units
  cosmo_G = G_COSMO;

  // Set gravitational constant to use for potential calculation
  Grav.Gconst = cosmo_G;

  max_delta_a = 0.001;
  delta_a = max_delta_a;

  // Initialize Time and set the time conversion
  t_secs = 0;
  time_conversion = KPC;


  // Set Normalization factors
  r_0_dm   = P->xlen/P->nx;
  t_0_dm   = 1. / H0;
  v_0_dm   = r_0_dm / t_0_dm / cosmo_h;
  rho_0_dm = 3*H0*H0 / ( 8*M_PI*cosmo_G ) * Omega_M /cosmo_h/cosmo_h;
  // dens_avrg = 0;

  r_0_gas = 1.0;
  rho_0_gas = 3*H0*H0 / ( 8*M_PI*cosmo_G ) * Omega_M /cosmo_h/cosmo_h;
  t_0_gas = 1/H0*cosmo_h;
  v_0_gas = r_0_gas / t_0_gas;
  phi_0_gas = v_0_gas * v_0_gas;
  p_0_gas = rho_0_gas * v_0_gas * v_0_gas;
  e_0_gas = v_0_gas * v_0_gas;

  chprintf( " H0: %f\n", H0 * 1000 );
  chprintf( " Omega_M: %f\n", Omega_M );
  chprintf( " Omega_L: %f\n", Omega_L );
  chprintf( " Omega_K: %f\n", Omega_K );
  chprintf( " Current_a: %f\n", current_a );
  chprintf( " Current_z: %f\n", current_z );
  chprintf( " rho_0: %f\n", rho_0_gas );
  chprintf( " v_0: %f \n", v_0_gas );

  Load_Scale_Outputs( P );
  


}













#endif
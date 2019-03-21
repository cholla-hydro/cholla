#ifdef COOLING_GRACKLE


#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../io.h"
#include "cool_grackle.h"



Cool_GK::Cool_GK( void ){}

void Grid3D::Initialize_Grackle( struct parameters *P ){
  
  chprintf( "Initializing Grackle... \n");
  
  Cool.Initialize( P, Cosmo );
  
  chprintf( "Grackle Initialized Successfully. \n\n");
  
  
}


void Cool_GK::Initialize( struct parameters *P, Cosmology &Cosmo ){
  
  chprintf( " Using Grackle for chemistry and cooling \n" );
  chprintf( " N scalar fields: %d \n", NSCALARS );
  
  grackle_verbose = 1;
  #ifdef MPI_CHOLLA
  // Enable output
  if (procID != 0 ) grackle_verbose = 0;
  #endif


  tiny_number = 1.e-20;
  gamma = P->gamma;

  dens_conv = Cosmo.rho_0_gas;
  energy_conv =   Cosmo.v_0_gas * Cosmo.v_0_gas ;

  Real Msun = MSUN_CGS;
  Real kpc = KPC_CGS;
  Real km = KM_CGS


  dens_to_CGS = dens_conv * Msun / kpc / kpc / kpc * Cosmo.cosmo_h * Cosmo.cosmo_h;
  vel_to_CGS = km;
  energy_to_CGS =  km * km;
  
  // First, set up the units system.
  // These are conversions from code units to cgs.
  units.comoving_coordinates = 1; // 1 if cosmological sim, 0 if not
  units.a_units = 1.0 ; // units for the expansion factor
  units.a_value = Cosmo.current_a / units.a_units;
  units.density_units = dens_to_CGS  / Cosmo.current_a / Cosmo.current_a / Cosmo.current_a ;
  units.length_units = kpc / Cosmo.cosmo_h * Cosmo.current_a;
  units.time_units = KPC / Cosmo.cosmo_h ;
  units.velocity_units = units.length_units / Cosmo.current_a / units.time_units; // since u = a * dx/dt
  
  // Second, create a chemistry object for parameters.  This needs to be a pointer.
  data = new chemistry_data;
  if (set_default_chemistry_parameters(data) == 0) {
    chprintf( "GRACKLE: Error in set_default_chemistry_parameters.\n");
    exit(-1) ;
  }
  // Set parameter values for chemistry.
  // Access the parameter storage with the struct you've created
  // or with the grackle_data pointer declared in grackle.h (see further below).
  data->use_grackle = 1;            // chemistry on
  data->with_radiative_cooling = 1; // G.Cooling on
  data->primordial_chemistry = 1;   // molecular network with H, He
  data->metal_cooling = 1;          // metal cooling on
  data->UVbackground = 1;           // UV background on
  data->grackle_data_file = "src/cooling/CloudyData_UVB=HM2012.h5"; // data file
  // data->grackle_data_file = "src/cooling/CloudyData_UVB=FG2011.h5"; // data file
  data->use_specific_heating_rate = 0;
  data->use_volumetric_heating_rate = 0;
  data->cmb_temperature_floor = 1;
  data->omp_nthreads = N_OMP_THREADS_GRACKLE;
  
  if ( data->UVbackground == 1) chprintf( "GRACKLE: Loading UV Background File: %s\n", data->grackle_data_file );
  
  // Finally, initialize the chemistry object.
  if (initialize_chemistry_data(&units) == 0) {
    chprintf( "GRACKLE: Error in initialize_chemistry_data.\n");
    exit(-1) ;
  }
  
  if ( data->UVbackground == 1){
    scale_factor_UVB_on = 1 / (data->UVbackground_redshift_on + 1 );
    chprintf( "GRACKLE: UVB on: %f \n", scale_factor_UVB_on  );
  }

  
}












#endif


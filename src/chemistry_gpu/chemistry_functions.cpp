#ifdef CHEMISTRY_GPU


#include "chemistry_gpu.h"
#include "../grid3D.h"
#include "../io.h"

#ifdef DE
#include"../hydro_cuda.h"
#endif

void Grid3D::Initialize_Chemistry( struct parameters *P ){
  
  chprintf( "Initializing the GPU Chemistry Solver... \n");
  
  Chem.nx = H.nx;
  Chem.ny = H.ny;
  Chem.nz = H.nz;

  Chem.gamma = gama;
  
  Chem.Initialize( P );
  
  // Initialize the Chemistry Header
  #ifdef COSMOLOGY
  Real kpc_cgs = KPC_CGS;
  Chem.H.density_conversion = Cosmo.rho_0_gas * Cosmo.cosmo_h * Cosmo.cosmo_h / pow( kpc_cgs, 3) * MSUN_CGS ; 
  Chem.H.energy_conversion  =  Cosmo.v_0_gas * Cosmo.v_0_gas * 1e10;  //km^2 -> cm^2 ;
  Chem.H.cosmological_parameters_d = Chem.cosmo_params_d;
  #else // Not COSMOLOGY
  Chem.H.density_conversion = 1.0;
  Chem.H.energy_conversion  = 1.0;
  Chem.H.cosmological_parameters_d = NULL;
  #endif
  Chem.H.n_uvb_rates_samples  = Chem.n_uvb_rates_samples;
  Chem.H.uvb_rates_redshift_d = Chem.rates_z_d;
  Chem.H.photo_ion_HI_rate_d   = Chem.Ion_rates_HI_d;
  Chem.H.photo_ion_HeI_rate_d  = Chem.Ion_rates_HeI_d;
  Chem.H.photo_ion_HeII_rate_d = Chem.Ion_rates_HeII_d;
  Chem.H.photo_heat_HI_rate_d   = Chem.Heat_rates_HI_d;
  Chem.H.photo_heat_HeI_rate_d  = Chem.Heat_rates_HeI_d;
  Chem.H.photo_heat_HeII_rate_d = Chem.Heat_rates_HeII_d;
  
  chprintf( "Allocating Memory. \n\n");
  int n_cells = H.nx * H.ny * H.nz;
  Chem.Fields.temperature_h = (Real *) malloc(n_cells * sizeof(Real));
  
  chprintf( "Chemistry Solver Successfully Initialized. \n\n");

}


void Chem_GPU::Initialize( struct parameters *P ){
  
  chprintf( " Initializing Cosmological Parameters... \n");
  cosmo_params_h = (float *)malloc(3*sizeof(Real));
  cosmo_params_h[0] = (float)P->H0;
  cosmo_params_h[1] = (float)P->Omega_M;
  cosmo_params_h[2] = (float)P->Omega_L;
  
  Allocate_Array_GPU_float( &cosmo_params_d, 3 );
  Copy_Float_Array_to_Device( 3, cosmo_params_h, cosmo_params_d );

  Initialize_UVB_Ionization_and_Heating_Rates( P );
  
  
  
}

void Chem_GPU::Initialize_UVB_Ionization_and_Heating_Rates( struct parameters *P ){
  
  Load_UVB_Ionization_and_Heating_Rates( P );
  
  Copy_UVB_Rates_to_GPU( );
  
  #ifdef TEXTURES_UVB_INTERPOLATION
  Bind_GPU_Textures( n_uvb_rates_samples, Heat_rates_HI_h, Heat_rates_HeI_h, Heat_rates_HeII_h, Ion_rates_HI_h, Ion_rates_HeI_h, Ion_rates_HeII_h);
  #endif
}

void Chem_GPU::Copy_UVB_Rates_to_GPU( ){
  
  Allocate_Array_GPU_float( &rates_z_d,         n_uvb_rates_samples );
  Allocate_Array_GPU_float( &Heat_rates_HI_d,   n_uvb_rates_samples );
  Allocate_Array_GPU_float( &Heat_rates_HeI_d,  n_uvb_rates_samples );
  Allocate_Array_GPU_float( &Heat_rates_HeII_d, n_uvb_rates_samples );
  Allocate_Array_GPU_float( &Ion_rates_HI_d,    n_uvb_rates_samples );
  Allocate_Array_GPU_float( &Ion_rates_HeI_d,   n_uvb_rates_samples );
  Allocate_Array_GPU_float( &Ion_rates_HeII_d,  n_uvb_rates_samples );
  
  Copy_Float_Array_to_Device( n_uvb_rates_samples, rates_z_h, rates_z_d );
  Copy_Float_Array_to_Device( n_uvb_rates_samples, Heat_rates_HI_h, Heat_rates_HI_d );
  Copy_Float_Array_to_Device( n_uvb_rates_samples, Heat_rates_HeI_h, Heat_rates_HeI_d );
  Copy_Float_Array_to_Device( n_uvb_rates_samples, Heat_rates_HeII_h, Heat_rates_HeII_d );
  Copy_Float_Array_to_Device( n_uvb_rates_samples, Ion_rates_HI_h, Ion_rates_HI_d );
  Copy_Float_Array_to_Device( n_uvb_rates_samples, Ion_rates_HeI_h, Ion_rates_HeI_d );
  Copy_Float_Array_to_Device( n_uvb_rates_samples, Ion_rates_HeII_h, Ion_rates_HeII_d );
  
}

void Grid3D::Update_Chemistry_Header(){
  
  #ifdef COSMOLOGY
  Chem.H.current_z = Cosmo.current_z;
  #else
  Chem.H.current_z = 0;
  #endif
  
  Chem.H.runtime_chemistry_step = 0;
    
}

void Grid3D::Compute_Gas_Temperature(  Real *temperature, bool convert_cosmo_units ){
  
  int k, j, i, id;
  Real dens_HI, dens_HII, dens_HeI, dens_HeII, dens_HeIII, dens_e, gamma;
  Real d, vx, vy, vz, E, GE, mu, temp, cell_dens, cell_n; 
  Real current_a, a2;
  gamma = gama;
  
    
  for (k=0; k<H.nz; k++) {
    for (j=0; j<H.ny; j++) {
      for (i=0; i<H.nx; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;
        
        d  =  C.density[id];
        vx =  C.momentum_x[id] / d;
        vy =  C.momentum_y[id] / d;
        vz =  C.momentum_z[id] / d;
        E = C.Energy[id];

        #ifdef DE
        GE = C.GasEnergy[id];
        #else 
        GE = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz));
        #endif
        
        dens_HI    = C.HI_density[id];
        dens_HII   = C.HII_density[id];
        dens_HeI   = C.HeI_density[id];
        dens_HeII  = C.HeII_density[id];
        dens_HeIII = C.HeIII_density[id]; 
        dens_e     = C.e_density[id];
        
        cell_dens = dens_HI + dens_HII + dens_HeI + dens_HeII + dens_HeIII;
        cell_n =  dens_HI + dens_HII + ( dens_HeI + dens_HeII + dens_HeIII )/4 + dens_e;
        mu = cell_dens / cell_n;
        
        #ifdef COSMOLOGY
        if ( convert_cosmo_units ){
          current_a = Cosmo.current_a;
          a2 = current_a * current_a;
          GE *= Chem.H.energy_conversion / a2; 
        } else {
          GE *= 1e10; // convert from (km/s)^2 to (cm/s)^2
        }
        #endif
        
        temp = GE * MP  * mu / d / KB * (gamma - 1.0);  ;
        temperature[id] = temp;
        // chprintf( "mu: %e \n", mu );
        // if ( temp > 1e7 ) chprintf( "Temperature: %e   mu: %e \n", temp, mu );  
        
      }
    }  
  } 
  
}

void Chem_GPU::Reset(){
  
  free( rates_z_h );
  free( Heat_rates_HI_h );
  free( Heat_rates_HeI_h );
  free( Heat_rates_HeII_h );
  free( Ion_rates_HI_h );
  free( Ion_rates_HeI_h );
  free( Ion_rates_HeII_h );
  
  Free_Array_GPU_float( rates_z_d );
  Free_Array_GPU_float( Heat_rates_HI_d );
  Free_Array_GPU_float( Heat_rates_HeI_d );
  Free_Array_GPU_float( Heat_rates_HeII_d );
  Free_Array_GPU_float( Ion_rates_HI_d );
  Free_Array_GPU_float( Ion_rates_HeI_d );
  Free_Array_GPU_float( Ion_rates_HeII_d );
  
  free( Fields.temperature_h );
  
}











#endif
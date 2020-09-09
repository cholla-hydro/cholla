#ifdef CHEMISTRY_GPU


#include "chemistry_gpu.h"
#include "../grid3D.h"
#include "../io.h"


void Grid3D::Initialize_Chemistry( struct parameters *P ){
  
  chprintf( "Initializing the GPU Chemistry Solver... \n");

  
  Chem.Initialize( P );
  
  
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
  
  Bind_GPU_Textures( n_uvb_rates_samples, Heat_rates_HI_h, Heat_rates_HeI_h, Heat_rates_HeII_h, Ion_rates_HI_h, Ion_rates_HeI_h, Ion_rates_HeII_h);




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
  
}











#endif
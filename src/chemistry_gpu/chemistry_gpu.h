#ifndef CHEMISTRY_GPU_H
#define CHEMISTRY_GPU_H

#include"../global.h"

// #define TEXTURES_UVB_INTERPOLATION

struct Chemistry_Header
{
  Real density_conversion;
  Real energy_conversion;
  Real current_z;
  int n_uvb_rates_samples;
  Real runtime_chemistry_step;
  
  float *cosmological_parameters_d;
  float *uvb_rates_redshift_d;
  float *photo_ion_HI_rate_d;
  float *photo_ion_HeI_rate_d;
  float *photo_ion_HeII_rate_d;
  float *photo_heat_HI_rate_d;
  float *photo_heat_HeI_rate_d;
  float *photo_heat_HeII_rate_d;
  
};




#ifdef CHEMISTRY_GPU

class Chem_GPU
{
public:
  
  int nx;
  int ny;
  int nz;
  
  Real gamma;
  
  Real scale_factor_UVB_on;

  float *cosmo_params_h;
  float *cosmo_params_d;
  
  int n_uvb_rates_samples;
  float *rates_z_h;  
  float *Heat_rates_HI_h;
  float *Heat_rates_HeI_h;
  float *Heat_rates_HeII_h;
  float *Ion_rates_HI_h;
  float *Ion_rates_HeI_h;
  float *Ion_rates_HeII_h;

  float *rates_z_d;
  float *Heat_rates_HI_d;
  float *Heat_rates_HeI_d;
  float *Heat_rates_HeII_d;
  float *Ion_rates_HI_d;
  float *Ion_rates_HeI_d;
  float *Ion_rates_HeII_d;
  
  struct Chemistry_Header H;
  
  
  struct Fields
  {
    Real *temperature_h;
  } Fields;
  
  
  void Allocate_Array_GPU_Real( Real **array_dev, int size );
  void Allocate_Array_GPU_float( float **array_dev, int size );
  void Copy_Float_Array_to_Device( int size, float *array_h, float *array_d );
  void Free_Array_GPU_float( float *array_dev );
      
  void Initialize( struct parameters *P );
  
  void Initialize_UVB_Ionization_and_Heating_Rates( struct parameters *P );
  
  void Load_UVB_Ionization_and_Heating_Rates(  struct parameters *P );
  
  void Copy_UVB_Rates_to_GPU();
    
  void Reset( );
  
  #ifdef TEXTURES_UVB_INTERPOLATION
  void Bind_GPU_Textures( int size,  float *H_HI_h, float *H_HeI_h, float *H_HeII_h , float *I_HI_h, float *I_HeI_h, float *I_HeII_h );
  #endif



};

#endif
#endif
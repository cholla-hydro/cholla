#ifndef CHEMISTRY_GPU_H
#define CHEMISTRY_GPU_H

#include "../global/global.h"

#define CHEM_TINY 1e-20

// Define the type of a generic rate function.
typedef Real (*Rate_Function_T)(Real, Real);

// #define TEXTURES_UVB_INTERPOLATION

struct Chemistry_Header {
  Real gamma;
  Real density_conversion;
  Real energy_conversion;
  Real current_z;
  Real runtime_chemistry_step;
  Real H_fraction;

  // Units system
  Real a_value;
  Real density_units;
  Real length_units;
  Real time_units;
  Real velocity_units;
  Real cooling_units;
  Real reaction_units;
  Real dens_number_conv;

  // Cosmological parameters
  Real H0;
  Real Omega_M;
  Real Omega_L;

  // Interpolation tables for the rates
  int N_Temp_bins;
  Real Temp_start;
  Real Temp_end;

  Real *cool_ceHI_d;
  Real *cool_ceHeI_d;
  Real *cool_ceHeII_d;

  Real *cool_ciHI_d;
  Real *cool_ciHeI_d;
  Real *cool_ciHeII_d;
  Real *cool_ciHeIS_d;

  Real *cool_reHII_d;
  Real *cool_reHeII_1_d;
  Real *cool_reHeII_2_d;
  Real *cool_reHeIII_d;

  Real *cool_brem_d;

  Real cool_compton;

  Real *k_coll_i_HI_d;
  Real *k_coll_i_HeI_d;
  Real *k_coll_i_HeII_d;
  Real *k_coll_i_HI_HI_d;
  Real *k_coll_i_HI_HeI_d;

  Real *k_recomb_HII_d;
  Real *k_recomb_HeII_d;
  Real *k_recomb_HeIII_d;

  int max_iter;

  int n_uvb_rates_samples;
  float *uvb_rates_redshift_d;
  float *photo_ion_HI_rate_d;
  float *photo_ion_HeI_rate_d;
  float *photo_ion_HeII_rate_d;
  float *photo_heat_HI_rate_d;
  float *photo_heat_HeI_rate_d;
  float *photo_heat_HeII_rate_d;

  //inherit the temperature floor
  //from Parameters
  float temperature_floor;
};

#ifdef CHEMISTRY_GPU

class Chem_GPU
{
 public:
  int nx;
  int ny;
  int nz;

  bool use_case_B_recombination;

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

  struct Fields {
    Real *temperature_h;
  } Fields;

  void Allocate_Array_GPU_Real(Real **array_dev, int size);
  void Copy_Real_Array_to_Device(int size, Real *array_h, Real *array_d);
  void Free_Array_GPU_Real(Real *array_dev);
  void Allocate_Array_GPU_float(float **array_dev, int size);
  void Copy_Float_Array_to_Device(int size, float *array_h, float *array_d);
  void Free_Array_GPU_float(float *array_dev);

  void Initialize(struct Parameters *P);

  void Generate_Reaction_Rate_Table(Real **rate_table_array_d, Rate_Function_T rate_function, Real units);

  void Initialize_Cooling_Rates();

  void Initialize_Reaction_Rates();

  void Initialize_UVB_Ionization_and_Heating_Rates(struct Parameters *P);

  void Load_UVB_Ionization_and_Heating_Rates(struct Parameters *P);

  void Copy_UVB_Rates_to_GPU();

  void Reset();

  #ifdef TEXTURES_UVB_INTERPOLATION
  void Bind_GPU_Textures(int size, float *H_HI_h, float *H_HeI_h, float *H_HeII_h, float *I_HI_h, float *I_HeI_h,
                         float *I_HeII_h);
  #endif
};

/*! \fn void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int
n_ghost, int n_fields, Real dt, Real gamma)
*  \brief When passed an array of conserved variables and a timestep, update the
ionization fractions of H and He and update the internal energy to account for
radiative cooling and photoheating from the UV background. */
void Do_Chemistry_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                         Chemistry_Header &Chem_H);

#endif
#endif

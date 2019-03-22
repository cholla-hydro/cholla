#ifdef COOLING_GRACKLE

#ifndef INIT_GRACKLE_H
#define INIT_GRACKLE_H

#include"../global.h"

extern "C" {
#include <grackle.h>
}

class Cool_GK
{
  public:

  code_units units;
  chemistry_data *data;

  Real dens_conv;
  Real energy_conv;

  Real dens_to_CGS;
  Real vel_to_CGS;
  Real energy_to_CGS;

  Real gamma;

  Real temperature_units;

  #ifdef OUTPUT_TEMPERATURE
  Real *temperature;
  #endif

  bool *flags_DE;

  Real tiny_number;

  Real scale_factor_UVB_on;

  // Struct for storing grackle field data
  grackle_field_data fields;
  int field_size;


Cool_GK( void );

void Initialize( struct parameters *P, Cosmology &Cosmo );

void Free_Memory();
// void Do_Cooling_Step( Real dt );

};

#endif
#endif
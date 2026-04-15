/*! \file
* This file defines \ref TabulatedDynamicalDarkEnergyEoS
*/

#pragma once

#include "../global/global.h"
#include <vector>

class TabulatedDynamicalDarkEnergyEoS
{
 public:
  std::vector<float> dynamicalDE_table_z;
  std::vector<float> dynamicalDE_table_w;
  // dynamical dark energy energy density normalized to the present-date rho_DE(z) / rho_DE(z=0)
  std::vector<float> dynamicalDE_table_density;

  /*! Load redshift and dark energy equation of state table z, wDE(z), only called once to setup dynamical DE case */
  void Setup_DynamicalDE_EquationOfState_(struct Parameters *P);

  /*! Calculate dark energy density normalized to z=0, populate dynamicalDE_table_density */
  void Set_DynamicalDE_Density();

  /*! Interpolate dynamicalDE_table_density to find rhoDE(z) / rhoDE(z=0) at z=1/a - 1 */
  Real Get_DynamicalDE_Density_from_a(Real a);
};


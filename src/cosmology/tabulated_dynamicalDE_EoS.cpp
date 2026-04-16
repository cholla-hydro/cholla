/*! \file
 * This file implements \ref TabulatedDynamicalDarkEnergyEoS
 */

#include "tabulated_dynamicalDE_EoS.h"

#include <fstream>
#include <iostream>
#include <vector>

#include "../io/io.h"

Real TabulatedDynamicalDarkEnergyEoS::Get_DynamicalDE_Density_from_a(Real a)
{
  Real z_low, z_high;
  Real dynDE_density_low, dynDE_density_high, dynDE_density_interp;
  Real lin_slope;
  Real z = (1. / a) - 1.;

  if (a == 1.) {
    return dynamicalDE_table_density.front();
  }
  if (z > dynamicalDE_table_z.back()) {
    return dynamicalDE_table_density.back();
  }

  int i  = 1;
  z_high = dynamicalDE_table_z[i];
  while (z_high < z) {
    i++;
    z_high = dynamicalDE_table_z[i];
  }
  dynDE_density_high = dynamicalDE_table_density[i];

  z_low             = dynamicalDE_table_z[i - 1];
  dynDE_density_low = dynamicalDE_table_density[i - 1];

  lin_slope            = (dynDE_density_low - dynDE_density_high) / (z_low - z_high);
  dynDE_density_interp = lin_slope * (z - z_low) + dynDE_density_low;

  return dynDE_density_interp;
}

void TabulatedDynamicalDarkEnergyEoS::Set_DynamicalDE_Density()
{
  Real integrand_prev, integrand;
  Real prev_integral_sum, my_integral, integral;
  Real z_prev, onez_prev;
  Real z, onez, onez_2, onez_3;
  Real wDE_prev, wDE;

  int i;
  Real cumulative_integral = 0.;
  dynamicalDE_table_density.push_back(1.);

  for (int i = 1; i < dynamicalDE_table_z.size(); i++) {
    z_prev = dynamicalDE_table_z[i - 1];
    z      = dynamicalDE_table_z[i];

    onez_prev = 1. + z_prev;
    onez      = 1. + z;
    onez_2    = onez * onez;
    onez_3    = onez_2 * onez;

    wDE_prev = dynamicalDE_table_w[i - 1];
    wDE      = dynamicalDE_table_w[i];

    integrand_prev = wDE_prev / onez_prev;
    integrand      = wDE / onez;

    // mid-point rectangle integral
    my_integral = ((integrand_prev + integrand) / 2.) * (z - z_prev);

    // add i-th contribution to cumulative sum
    cumulative_integral += my_integral;

    dynamicalDE_table_density.push_back(onez_3 * exp(3. * cumulative_integral));
  }
}

void TabulatedDynamicalDarkEnergyEoS::Setup_DynamicalDE_EquationOfState_(struct Parameters *P)
{
  if (P->wDE_file.empty()) {
    chprintf("wDE_file not found in parameter file \n");
    exit(1);
  }

  chprintf("Loading wDE info... \n");

  std::fstream in(P->wDE_file);
  std::string line;
  std::vector<std::vector<float>> v;
  int i = 0;
  if (in.is_open()) {
    while (std::getline(in, line)) {
      if (line.find("#") == 0) continue;

      float value;
      std::stringstream ss(line);
      v.push_back(std::vector<float>());

      while (ss >> value) {
        v[i].push_back(value);
      }
      i += 1;
    }
    in.close();
  } else {
    chprintf(" Error: Unable to open DE equation of state file: %s\n", P->wDE_file);
    exit(1);
  }
  int n_lines = i;

  for (i = 0; i < n_lines; i++) {
    dynamicalDE_table_z.push_back(v[i][0]);
    dynamicalDE_table_w.push_back(v[i][1]);
  }

  for (i = 0; i < n_lines - 1; i++) {
    if (dynamicalDE_table_z[i] > dynamicalDE_table_z[i + 1]) {
      chprintf(
          " ERROR: equation of state must be ordered such that redshift is increasing "
          "as the rows increase in the file\n",
          P->wDE_file);
      exit(2);
    }
  }

  chprintf(" Loaded DE equation of state file : \n");
  chprintf("  N redshift values: %d \n", dynamicalDE_table_z.size());
  chprintf("  z_min = %f    z_max = %f \n", dynamicalDE_table_z.front(), dynamicalDE_table_z.back());
  chprintf("  w(z_min) = %f    w(z_max) = %f \n", dynamicalDE_table_w.front(), dynamicalDE_table_w.back());

  if (dynamicalDE_table_z[0] != 0.) {
    chprintf("We require z_min = 0 so that w(z=0) is well defined \n");
    exit(1);
  }
}

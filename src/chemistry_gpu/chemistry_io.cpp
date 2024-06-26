#ifdef CHEMISTRY_GPU

  #include <cstring>  // provides std::strcpy (strcpy in this file)
  #include <fstream>
  #include <iostream>
  #include <sstream>
  #include <string>
  #include <vector>

  #include "../io/io.h"
  #include "chemistry_gpu.h"

void Chem_GPU::Load_UVB_Ionization_and_Heating_Rates(struct Parameters *P)
{
  char uvb_filename[100];
  // create the filename to read from
  strcpy(uvb_filename, P->UVB_rates_file);
  chprintf(" Loading UVB rates: %s\n", uvb_filename);

  std::fstream in(uvb_filename);
  std::string line;
  std::vector<std::vector<float>> v;
  int i = 0;
  if (in.is_open()) {
    while (std::getline(in, line)) {
      if (line.find("#") == 0) continue;

      float value;
      std::stringstream ss(line);
      // chprintf( "%s \n", line.c_str() );
      v.push_back(std::vector<float>());

      while (ss >> value) {
        v[i].push_back(value);
      }
      i += 1;
    }
    in.close();
  } else {
    chprintf(" Error: Unable to open UVB rates file: %s\n", uvb_filename);
    exit(1);
  }

  int n_lines = i;

  chprintf(" Loaded %d lines in file\n", n_lines);

  rates_z_h         = (float *)malloc(sizeof(float) * n_lines);
  Heat_rates_HI_h   = (float *)malloc(sizeof(float) * n_lines);
  Heat_rates_HeI_h  = (float *)malloc(sizeof(float) * n_lines);
  Heat_rates_HeII_h = (float *)malloc(sizeof(float) * n_lines);
  Ion_rates_HI_h    = (float *)malloc(sizeof(float) * n_lines);
  Ion_rates_HeI_h   = (float *)malloc(sizeof(float) * n_lines);
  Ion_rates_HeII_h  = (float *)malloc(sizeof(float) * n_lines);

  Real eV_to_ergs, heat_units, ion_units;
  eV_to_ergs = 1.60218e-12;
  heat_units = eV_to_ergs / H.cooling_units;
  ion_units  = H.time_units;

  for (i = 0; i < n_lines; i++) {
    rates_z_h[i]         = v[i][0];
    Ion_rates_HI_h[i]    = v[i][1] * ion_units;
    Heat_rates_HI_h[i]   = v[i][2] * heat_units;
    Ion_rates_HeI_h[i]   = v[i][3] * ion_units;
    Heat_rates_HeI_h[i]  = v[i][4] * heat_units;
    Ion_rates_HeII_h[i]  = v[i][5] * ion_units;
    Heat_rates_HeII_h[i] = v[i][6] * heat_units;
    // chprintf( " %f  %e  %e  %e   \n", rates_z_h[i], Heat_rates_HI_h[i],
    // Heat_rates_HeI_h[i],  Heat_rates_HeII_h[i]); chprintf( " %f  %f  \n",
    // rates_z_h[i], Heat_rates_HI_h[i] );
  }

  for (i = 0; i < n_lines - 1; i++) {
    if (rates_z_h[i] > rates_z_h[i + 1]) {
      chprintf(
          " ERROR: UVB rates must be ordered such that redshift is increasing "
          "as the rows increase in the file\n",
          uvb_filename);
      exit(2);
    }
  }

  n_uvb_rates_samples = n_lines;
  scale_factor_UVB_on = 1 / (rates_z_h[n_uvb_rates_samples - 1] + 1);
  chprintf(" Loaded UVB rates: \n");
  chprintf("  N redshift values: %d \n", n_uvb_rates_samples);
  chprintf("  z_min = %f    z_max = %f \n", rates_z_h[0], rates_z_h[n_uvb_rates_samples - 1]);
  chprintf("  UVB on:  a=%f \n", scale_factor_UVB_on);
}

/*! \fn void Show_Chemistry_Units(void)
*  \brief Show the chemsitry unit system. */
int Chem_GPU::Show_Chemistry_Units( void )
{
  chprintf("********\n\n");
  chprintf("Chemistry Header time_units          %10.9e [same as TIME_UNIT].\n",H.time_units);
  chprintf("Chemistry Header length_units        %10.9e [same as LENGTH_UNIT].\n",H.length_units);
  chprintf("Chemistry Header density_units       %10.9e [same as DENSITY_UNIT].\n",H.density_units);
  chprintf("Chemistry Header energy_units        %10.9e [same as ENERGY_UNIT].\n",H.energy_units);
  chprintf("Chemistry Header energy_conversion   %10.9e [v_0_gas**2 * 1e10].\n",H.energy_conversion);
  chprintf("Chemistry Header density_conversion  %10.9e [rho_0_gas h^2 / kpc^3 * Msun_cgs].\n",H.density_conversion);
  chprintf("Chemistry Header dens_number_conv    %10.9e [density_units/MH].\n",H.dens_number_conv);
  chprintf("Chemistry Header reaction_units      %10.9e [MH / (DENSITY_UNIT * TIME_UNIT)].\n",H.reaction_units);
  chprintf("Chemistry Header cooling_units       %10.9e [1e10 * MH * reaction_units].\n",H.cooling_units);
  chprintf("Chemistry Header heat_units          %10.9e [eV_to_ergs / cooling_units].\n",H.heat_units);
  chprintf("Chemistry Header ion_units           %10.9e [same as TIME_UNIT].\n",H.ion_units);
  chprintf("Chemistry Header eV_to_ergs          %10.9e [electron volts in cgs].\n",H.eV_to_ergs);
#ifdef COSMOLOGY
  chprintf("Chemistry Header a_value             %10.9e.\n",H.a_value);
  chprintf("Chemistry Header H0                  %10.9e.\n",H.H0);
  chprintf("Chemistry Header Omega_M             %10.9e.\n",H.Omega_M);
  chprintf("Chemistry Header Omega_L             %10.9e.\n",H.Omega_L);
#endif //COSMOLOGY
  chprintf("\n********\n");
}

#endif

#ifdef CHEMISTRY_GPU

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstring> // provides std::strcpy (strcpy in this file)
#include "chemistry_gpu.h"
#include "../io/io.h"


using namespace std;


void Chem_GPU::Load_UVB_Ionization_and_Heating_Rates(  struct parameters *P ){


  char uvb_filename[100];
  // create the filename to read from
  strcpy(uvb_filename, P->UVB_rates_file);
  chprintf( " Loading UVB rates: %s\n", uvb_filename);
  

  
  std::fstream in(uvb_filename);
  std::string line;
  std::vector<std::vector<float>> v;
  int i = 0;
  if (in.is_open()){
    while (std::getline(in, line))
    {
       if ( line.find("#") == 0 ) continue;
      
       float value;
       std::stringstream ss(line);
       // chprintf( "%s \n", line.c_str() );
       v.push_back(std::vector<float>());
       
       while (ss >> value){
         v[i].push_back(value);
       }
       i += 1;    
    }
    in.close();
  } else{
    chprintf(" Error: Unable to open UVB rates file: %s\n", uvb_filename);
    exit(1);
  }
  
  int n_lines = i;
  
  chprintf( " Loaded %d lines in file\n", n_lines);
  
  rates_z_h         = (float *)malloc(sizeof(float)*n_lines);
  Heat_rates_HI_h   = (float *)malloc(sizeof(float)*n_lines);
  Heat_rates_HeI_h  = (float *)malloc(sizeof(float)*n_lines);
  Heat_rates_HeII_h = (float *)malloc(sizeof(float)*n_lines);
  Ion_rates_HI_h    = (float *)malloc(sizeof(float)*n_lines);
  Ion_rates_HeI_h   = (float *)malloc(sizeof(float)*n_lines);
  Ion_rates_HeII_h  = (float *)malloc(sizeof(float)*n_lines);
  
  Real eV_to_ergs, heat_units, ion_units; 
  eV_to_ergs = 1.60218e-12;
  heat_units = eV_to_ergs / H.cooling_units;
  ion_units  = H.time_units;
  
  for (i=0; i<n_lines; i++ ){
    rates_z_h[i] = v[i][0] ;
    Ion_rates_HI_h[i]    = v[i][1] * ion_units;
    Heat_rates_HI_h[i]   = v[i][2] * heat_units;
    Ion_rates_HeI_h[i]   = v[i][3] * ion_units;
    Heat_rates_HeI_h[i]  = v[i][4] * heat_units;
    Ion_rates_HeII_h[i]  = v[i][5] * ion_units;
    Heat_rates_HeII_h[i] = v[i][6] * heat_units;
    // chprintf( " %f  %e  %e  %e   \n", rates_z_h[i], Heat_rates_HI_h[i],  Heat_rates_HeI_h[i],  Heat_rates_HeII_h[i]);
    // chprintf( " %f  %f  \n", rates_z_h[i], Heat_rates_HI_h[i] );
  }
  
  for ( i=0; i<n_lines-1; i++ ){
    if ( rates_z_h[i] > rates_z_h[i+1] ){
      chprintf( " ERROR: UVB rates must be ordered such that redshift is increasing as the rows increase in the file\n", uvb_filename);
      exit(2);
    }
  }
  
  n_uvb_rates_samples = n_lines;
  scale_factor_UVB_on = 1 / (rates_z_h[n_uvb_rates_samples-1] + 1 );
  chprintf(" Loaded UVB rates: \n");
  chprintf("  N redshift values: %d \n", n_uvb_rates_samples );
  chprintf("  z_min = %f    z_max = %f \n", rates_z_h[0], rates_z_h[n_uvb_rates_samples-1] );
  chprintf("  UVB on:  a=%f \n", scale_factor_UVB_on  );
  

}















#endif

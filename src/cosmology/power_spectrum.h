#ifdef COSMOLOGY

#ifndef POWER_SPECTRUM_H
#define POWER_SPECTRUM_H

#include <stdio.h>
#include <cmath>
#include "../global/global.h"

class Cosmo_Power_Spectrum
{ 
  public:  
    
    int  host_size;
    Real *host_k;
    Real *host_pk_dm;
    Real *host_pk_gas;
    
    int  *dev_size;
    Real *dev_k;
    Real *dev_pk_dm;
    Real *dev_pk_gas;

    
    // Constructor
    Cosmo_Power_Spectrum();
    
    void Load_Power_Spectum_From_File( struct parameters *P );
  
  
};












#endif //POWER_SPECTRUM_H
#endif //COSMOLOGY
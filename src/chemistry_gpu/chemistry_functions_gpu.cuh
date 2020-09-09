#ifdef CHEMISTRY_GPU

#ifndef CHEMISTRY_FUNCTIONS_GPU_H
#define CHEMISTRY_FUNCTIONS_GPU_H

#include"../global.h"



__global__ void Update_Chemistry( Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt_hydro, Real gamma,  Real density_conv, Real energy_conv, 
                                  Real current_z,  float* cosmo_params, int n_uvb_rates_samples, float *rates_z   );















#endif
#endif
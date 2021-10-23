/*! \file cooling_cuda.h
 *  \brief Declarations of cooling functions. */

#ifdef CUDA
#ifdef COOLING_GPU

#ifndef COOLING_CUDA_H
#define COOLING_CUDA_H

#include "../utils/gpu.hpp"
#include <math.h>
#include "../global/global.h"


/*! \fn void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust the value
           of the total energy for each cell according to the specified cooling function. */
void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, Real *dt_array, Real *return_total_energy, Real *return_mask_energy);

/*! \fn void Cooling_Calc_dt(Real *dev_dt_array)
 *  \brief Calculate cooling-defined minimum timestep */
Real Cooling_Calc_dt(Real *d_dt_array, Real *h_dt_array, int nx, int ny, int nz);

#endif //COOLING_CUDA_H
#endif //COOLING_GPU
#endif //CUDA

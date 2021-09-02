/*! \file cooling_wrapper.h
 *  \brief Wrapper file to load CUDA cooling tables. */

#ifdef CUDA
#ifdef CLOUDY_COOL

#ifndef COOLING_WRAPPER_H
#define COOLING_WRAPPER_H

#include "../global/global.h"

/* \fn void Load_Cuda_Textures()
 * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
void Load_Cuda_Textures();

/* \fn void Load_Cooling_Tables(float* cooling_table, float* heating_table)
 * \brief Load the Cloudy cooling tables into host (CPU) memory. */
void Load_Cooling_Tables(float* cooling_table, float* heating_table);


/* \fn void Free_Cuda_Textures()
 * \brief Unbind the texture memory on the GPU, and free the associated Cuda arrays. */
void Free_Cuda_Textures();


#endif
#endif
#endif


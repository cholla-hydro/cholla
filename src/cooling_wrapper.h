/*! \file cooling_wrapper.h
 *  \brief Wrapper file for to load CUDA cooling tables. */

#ifdef CUDA
#ifdef COOLING_GPU

#ifndef COOLING_WRAPPER_H
#define COOLING_WRAPPER_H

#include"global.h"


/* \fn void Load_Cuda_Textures()
 * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
void Load_Cuda_Textures();

/* \fn void Free_Cuda_Textures()
 * \brief Free the memory associated with the Cloudy cooling tables. */
void Free_Cuda_Textures();

#endif
#endif
#endif


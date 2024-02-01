/*! \file load_cloudy_texture.h
 *  \brief Wrapper file to load cloudy cooling table as CUDA texture. */

#ifdef CLOUDY_COOL

  #pragma once

  #include "../global/global.h"

/* \fn void Load_Cuda_Textures()
 * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
void Load_Cuda_Textures();

/* \fn void Free_Cuda_Textures()
 * \brief Unbind the texture memory on the GPU, and free the associated Cuda
 * arrays. */
void Free_Cuda_Textures();

#endif  // CLOUDY_COOL

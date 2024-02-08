/*! \file texture_utilities.h
 *  \brief Declarations of functions needed for textures. */

// WARNING: do not include this header file in any .cpp file or any .h file that
// would be included into a .cpp file because tex2D is undefined when compiling
// with gcc.

#pragma once

#include <math.h>

#include "../global/global.h"
#include "../utils/gpu.hpp"

inline __device__ float lerp(float v0, float v1, float f) { return fma(f, v1, fma(-f, v0, v0)); }

/* \fn float Bilinear_Texture(cudaTextureObject_t tex, float x, float y)
   \brief Access texture values from tex at coordinates (x,y) using bilinear
   interpolation
*/
inline __device__ float Bilinear_Texture(cudaTextureObject_t tex, float x, float y)
{
  // Split coordinates into integer px/py and fractional fx/fy parts
  float px = floorf(x);
  float py = floorf(y);
  float fx = x - px;
  float fy = y - py;

  // 0.5 offset is necessary to represent half-pixel offset built into texture
  // coordinates
  px += 0.5;
  py += 0.5;

  float t00 = tex2D<float>(tex, px, py);
  float t01 = tex2D<float>(tex, px, py + 1);
  float t10 = tex2D<float>(tex, px + 1, py);
  float t11 = tex2D<float>(tex, px + 1, py + 1);
  // The inner lerps interpolate along x
  // The outer lerp interpolates along y
  return lerp(lerp(t00, t10, fx), lerp(t01, t11, fx), fy);
}

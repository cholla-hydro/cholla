/*! \file cooling_cuda.h
 *  \brief Declarations of cooling functions. */

#pragma once

#include <math.h>

#include <functional>
#include <string>

#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../io/ParameterMap.h"
#include "../utils/gpu.hpp"

/* \fn __device__ Real primordial_cool(Real n, Real T)
 * \brief Primordial hydrogen/helium cooling curve
          derived according to Katz et al. 1996. */
__device__ Real primordial_cool(Real n, Real T);

/*! construct a callback that performs heating/cooling (to approximate the impact of chemistry).
 *
 *  The returned callback can be used as the chemistry_callback */
std::function<void(Grid3D&)> configure_cooling_callback(std::string kind, ParameterMap& pmap);
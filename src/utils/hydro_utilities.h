/*!
 * \file hydro_utilities.h
 * \author Helena Richie (helenarichie@pitt.edu)
 * \brief Contains the declaration of various utility functions for hydro
 *
 */

#pragma once

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"

namespace hydroUtils {
    namespace {
        inline __host__ __device__ Real _Calc_Pressure(Real const &E, Real const &d_gas, Real const &vx, Real const &vy, Real const &vz, Real const &gamma) {
        Real p;
        p = (E - 0.5 * d_gas * (vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
        p = fmax(p, (Real) TINY_NUMBER);
        return p;
        }
                                            
    }
}
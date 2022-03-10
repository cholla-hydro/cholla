/*!
 * \file hydro_utilities.h
 * \author Helena Richie (helenarichie@pitt.edu)
 * \brief Contains the declaration of various utility functions for hydro
 *
 */

#pragma once

#include <iostream>
#include <string>

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"


/*!
* INDEX OF VARIABLES
* p : pressure
* vx, vy, vz : x, y, and z velocity
* d : density
* E : energy
* T : temperature
* px, py, pz : x, y, and z momentum
* n : number density
*/

namespace hydro_utilities {
    namespace {
        inline __host__ __device__ Real Calc_Pressure_Primitive(Real const &E, Real const &d, Real const &vx, Real const &vy, Real const &vz, Real const &gamma) {
            Real p;
            std::cout << "\n" << 0.5 * d * (vx*vx + vy*vy + vz*vz) << "\n";
            p = (E - 0.5 * d * (vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
            //std::cout << "\n" << p << "\n";
            p = fmax(p, TINY_NUMBER);
            return p;
        }

        inline __host__ __device__ Real Calc_Pressure_Conserved(Real const &E, Real const &d, Real const &px, Real const &py, Real const &pz, Real const &gamma) {
            Real p = (E - 0.5 * (px*px + py*py + pz*pz) / d) * (gamma - 1.);
            return fmax(p, TINY_NUMBER);
        }

        inline __host__ __device__ Real Calc_Temp(Real const &p, Real const &n) {
            Real T = p * PRESSURE_UNIT / (n * KB);
            return T;
        }

        #ifdef DE
        inline __host__ __device__ Real _Calc_Temp_DE(Real const &d, Real const &ge, Real const &gamma, Real const&n) {
            Real T =  d * ge * (gamma - 1.0) * PRESSURE_UNIT / (n * KB);
            return T;
        }
        #endif // DE

        inline __host__ __device__ Real Calc_Energy_Primitive(Real const &p, Real const &d, Real const &vx, Real const &vy, Real const &vz, Real const &gamma) {
        // Compute and return energy
        return (fmax(p, TINY_NUMBER)/(gamma - 1.)) + 0.5 * d * (vx*vx + vy*vy + vz*vz);
        }
                                            
    }
}
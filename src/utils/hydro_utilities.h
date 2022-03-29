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
* P : pressure
* vx, vy, vz : x, y, and z velocity
* d : density
* E : energy
* T : temperature
* mx, my, mz : x, y, and z momentum
* n : number density

* "k" prefix in a variable indicates that it is a const.
*/

namespace hydro_utilities {
    inline __host__ __device__ Real Calc_Pressure_Primitive(Real const &k_E, Real const &k_d, Real const &k_vx, Real const &k_vy, Real const &k_vz, Real const &k_gamma) {
        Real P;
        P = (k_E - 0.5 * k_d * (k_vx*k_vx + k_vy*k_vy + k_vz*k_vz)) * (k_gamma - 1.0);
        P = fmax(P, TINY_NUMBER);
        return P;
    }

    inline __host__ __device__ Real Calc_Pressure_Conserved(Real const &k_E, Real const &k_d, Real const &k_mx, Real const &k_my, Real const &k_mz, Real const &k_gamma) {
        Real P = (k_E - 0.5 * (k_mx*k_mx + k_my*k_my + k_mz*k_mz) / k_d) * (k_gamma - 1.);
        return fmax(P, TINY_NUMBER);
    }

    inline __host__ __device__ Real Calc_Temp(Real const &k_P, Real const &k_n) {
        Real T = k_P * PRESSURE_UNIT / (k_n * KB);
        return T;
    }

    #ifdef DE
    inline __host__ __device__ Real Calc_Temp_DE(Real const &k_d, Real const &kge, Real const &k_gamma, Real const&k_n) {
        Real T =  k_d * kge * (k_gamma - 1.0) * PRESSURE_UNIT / (k_n * KB);
        return T;
    }
    #endif // DE

    inline __host__ __device__ Real Calc_Energy_Primitive(Real const &k_P, Real const &k_d, Real const &k_vx, Real const &k_vy, Real const &k_vz, Real const &k_gamma) {
        // Compute and return energy
        return (fmax(P, TINY_NUMBER)/(k_gamma - 1.)) + 0.5 * k_d * (k_vx*k_vx + k_vy*k_vy + k_vz*k_vz);
    }

    inline __host__ __device__ Real Get_Pressure_From_DE(Real const &k_E, Real const &k_U_total, Real const &k_U_advected, Real const &k_gamma) {
        Real U, P;
        Real const k_eta = DE_ETA_1;
        // Apply same condition as Byan+2013 to select the internal energy from which compute pressure.
        if (k_U_total/k_E > k_eta) {
            U = k_U_total;
        } else {
            U = k_U_advected;
        }
        P = U * (k_gamma - 1.0);
        return P;
    }

}
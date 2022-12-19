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
*/

namespace hydro_utilities {

    inline __host__ __device__ Real Calc_Pressure_Primitive(Real const &E, Real const &d, Real const &vx, Real const &vy, Real const &vz, Real const &gamma) {
        Real P;
        P = (E - 0.5 * d * (vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
        P = fmax(P, TINY_NUMBER);
        return P;
    }

    inline __host__ __device__ Real Calc_Pressure_Conserved(Real const &E, Real const &d, Real const &mx, Real const &my, Real const &mz, Real const &gamma) {
        Real P= (E - 0.5 * (mx*mx + my*my + mz*mz) / d) * (gamma - 1.);
        return fmax(P, TINY_NUMBER);
    }

    inline __host__ __device__ Real Calc_Temp(Real const &P, Real const &n) {
        Real T = P * PRESSURE_UNIT / (n * KB);
        return T;
    }

    #ifdef DE
    inline __host__ __device__ Real Calc_Temp_DE(Real const &d, Real const &ge, Real const &gamma, Real const&n) {
        Real T =  d * ge * (gamma - 1.0) * PRESSURE_UNIT / (n * KB);
        return T;
    }
    #endif // DE

    inline __host__ __device__ Real Calc_Energy_Primitive(Real const &P, Real const &d, Real const &vx, Real const &vy, Real const &vz, Real const &gamma) {
        // Compute and return energy
        return (fmax(P, TINY_NUMBER)/(gamma - 1.)) + 0.5 * d * (vx*vx + vy*vy + vz*vz);
    }

    inline __host__ __device__ Real Get_Pressure_From_DE(Real const &E, Real const &U_total, Real const &U_advected, Real const &gamma) {
        Real U, P;
        Real eta = DE_ETA_1;
        // Apply same condition as Byan+2013 to select the internal energy from which compute pressure.
        if (U_total/E > eta) {
            U = U_total;
        } else {
            U = U_advected;
        }
        P = U * (gamma - 1.0);
        return P;
    }

    /*!
     * \brief Compute the kinetic energy from the density and velocities
     *
     * \param[in] d The density
     * \param[in] vx The x velocity
     * \param[in] vy The y velocity
     * \param[in] vz The z velocity
     * \return Real The kinetic energy
     */
    inline __host__ __device__ Real Calc_Kinetic_Energy_From_Velocity(Real const &d,
                                                                      Real const &vx,
                                                                      Real const &vy,
                                                                      Real const &vz)
    {
        return 0.5 * d * (vx*vx + vy*vy * vz*vz);
    }

    /*!
     * \brief Compute the kinetic energy from the density and momenta
     *
     * \param[in] d The density
     * \param[in] mx The x momentum
     * \param[in] my The y momentum
     * \param[in] mz The z momentum
     * \return Real The kinetic energy
     */
    inline __host__ __device__ Real Calc_Kinetic_Energy_From_Momentum(Real const &d,
                                                                      Real const &mx,
                                                                      Real const &my,
                                                                      Real const &mz)
    {
        return (0.5 / d) * (mx*mx + my*my * mz*mz);
    }

    inline __host__ __device__ Real Calc_Sound_Speed(Real const &E, Real const &d, Real const &mx, Real const &my, Real const &mz, Real const &gamma) {
        Real P = Calc_Pressure_Conserved(E, d, mx, my, mz, gamma);
        return sqrt(gamma * P / d);
    }


}

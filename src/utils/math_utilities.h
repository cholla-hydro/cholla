/*!
 * \file math_utilities.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains various functions for common mathematical operations
 *
 */

#pragma once

// STL Includes
#include <cmath>
#include <tuple>

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"

namespace math_utils
{
    // =========================================================================
    /*!
     * \brief Rotate cartesian coordinates. All arguments are cast to double
     * then rotated. If the type is 'int' then the value is rounded to the
     * nearest int
     *
     * \details Rotation such that when pitch=90 and yaw=0 x1_rot = -x3 and when
     * pitch=0 and yaw=90 x1_rot = -x2
     *
     * \tparam T The return type
     * \param[in] x_1 x1 coordinate
     * \param[in] x_2 x2 coordinate
     * \param[in] x_3 x3 coordinate
     * \param[in] pitch Pitch angle in radians
     * \param[in] yaw Yaw angle in radians
     * \return std::tuple<T, T, T> The new, rotated, coordinates in the
     * order <x1, x2, x2>. Intended to be captured with structured binding
     */
    template<typename T>
    inline std::tuple<T, T, T> rotateCoords(Real const &x_1, Real const &x_2,
        Real const &x_3, Real const &pitch, Real const &yaw)
    {
        // Compute the sines and cosines. Correct for floating point errors if
        // the angle is 0.5*M_PI
        Real const sin_yaw   = std::sin(yaw);
        Real const cos_yaw   = (yaw==0.5*M_PI)? 0: std::cos(yaw);
        Real const sin_pitch = std::sin(pitch);
        Real const cos_pitch = (pitch==0.5*M_PI)? 0: std::cos(pitch);

        // Perform the rotation
        Real const x_1_rot = (x_1 *  cos_pitch * cos_yaw) + (x_2 * sin_yaw) + (x_3 * sin_pitch * cos_yaw);
        Real const x_2_rot = (x_1 *  cos_pitch * sin_yaw) + (x_2 * cos_yaw) + (x_3 * sin_pitch * sin_yaw);
        Real const x_3_rot = (x_1 *  sin_pitch) + (x_3 * cos_pitch);

        if (std::is_same<T, int>::value)
        {
            return {round(x_1_rot),
                    round(x_2_rot),
                    round(x_3_rot)};
        }
        else if (std::is_same<T, Real>::value)
        {
            return {x_1_rot, x_2_rot, x_3_rot};
        }
    }
    // =========================================================================

    // =========================================================================
    /*!
     * \brief Compute the dot product of a and b.
     *
     * \param[in] a1 The first element of a
     * \param[in] a2 The second element of a
     * \param[in] a3 The third element of a
     * \param[in] b1 The first element of b
     * \param[in] b2 The second element of b
     * \param[in] b3 The third element of b
     *
     * \return Real The dot product of a and b
     */
    inline __device__ __host__ Real dotProduct(Real const &a1,
                                               Real const &a2,
                                                Real const &a3,
                                                Real const &b1,
                                                Real const &b2,
                                                Real const &b3)
    {return a1*b1 + ((a2*b2) + (a3*b3));};
    // =========================================================================

}//math_utils

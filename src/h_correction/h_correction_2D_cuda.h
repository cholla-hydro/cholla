/*! \file h_correction_2D_cuda.h
 *  \brief Functions declarations for the H correction kernels.
           Written following Sanders et al. 1998. */

#ifdef H_CORRECTION
  #ifndef H_CORRECTION_2D_H
    #define H_CORRECTION_2D_H

    #include <math.h>

    #include "../global/global.h"
    #include "../global/global_cuda.h"
    #include "../utils/gpu.hpp"

/*! \fn void calc_eta_x(Real *dev_bounds_L, Real *dev_bounds_R, Real *eta_x, int
 nx, int ny, int nz, int n_ghost, Real gamma)
 *  \brief When passed the left and right boundary values at an interface,
 calculates the eta value for the interface according to the forumulation in
 Sanders et al, 1998. */
__global__ void calc_eta_x_2D(Real *dev_bounds_L, Real *dev_bounds_R, Real *eta_x, int nx, int ny, int n_ghost,
                              Real gamma);

/*! \fn void calc_eta_y(Real *dev_bounds_L, Real *dev_bounds_R, Real *eta_y, int
 nx, int ny, int nz, int n_ghost, Real gamma)
 *  \brief When passed the left and right boundary values at an interface,
 calculates the eta value for the interface according to the forumulation in
 Sanders et al, 1998. */
__global__ void calc_eta_y_2D(Real *dev_bounds_L, Real *dev_bounds_R, Real *eta_y, int nx, int ny, int n_ghost,
                              Real gamma);

/*! \fn void calc_etah_x_2D(Real *eta_x, Real *eta_y, Real *etah_x, int nx, int
 ny, int n_ghost)
 *  \brief When passed the eta values at every interface, calculates
           the eta_h value for the interface according to the forumulation in
 Sanders et al, 1998. */
__global__ void calc_etah_x_2D(Real *eta_x, Real *eta_y, Real *etah_x, int nx, int ny, int n_ghost);

/*! \fn void calc_etah_y_2D(Real *eta_x, Real *eta_y, Real *etah_y, int nx, int
 ny, int n_ghost)
 *  \brief When passed the eta values at every interface, calculates
           the eta_h value for the interface according to the forumulation in
 Sanders et al, 1998. */
__global__ void calc_etah_y_2D(Real *eta_x, Real *eta_y, Real *etah_y, int nx, int ny, int n_ghost);

  #endif  // H_CORRECTION_2D_H
#endif    // H_CORRECTION

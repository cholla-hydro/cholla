/*! \file rt_source_state.h
 *  \brief Declaration of RT_State, the snapshot of grid and time information
 *         that the RT source model needs each step. Filled by Grid3D::Update_RT_State(). */

#ifndef RT_SOURCE_STATE_H
#define RT_SOURCE_STATE_H

#ifdef RT

  #include "../global/global.h"

struct RT_State {
  /*! \var t_mid_myr
   *  \brief Simulation time at the middle of the current step [Myr].
   *  Source lifetimes and luminosities are evaluated here. */
  Real t_mid_myr = 0;

  /*! \var dt_myr
   *  \brief Length of the current step in myr. */
  Real dt_myr = 0;

  /*! \var drift_factor
   *  \brief Factor that converts a source velocity into a position change over the step:
   *  pos += vel * drift_factor. Non-cosmological: H.dt. Cosmological: the same drift
   *  the particle KDK integrator applies (see Grid3D::Update_RT_State). */
  Real drift_factor = 0;

  /*! \var current_a
   *  \brief Cosmological scale factor at the start of the step (1 without COSMOLOGY). */
  Real current_a = 1;

  /*! \var dx
   *  \brief Cell width in x-direction [code length units, comoving under COSMOLOGY]. */
  Real dx;

  /*! \var dy
   *  \brief Cell width in y-direction [code length units, comoving under COSMOLOGY]. */
  Real dy;

  /*! \var dz
   *  \brief Cell width in z-direction [code length units, comoving under COSMOLOGY]. */
  Real dz;

  /*! \var xbound
   *  \brief Global domain minimum in x-direction */
  Real xbound = 0;

  /*! \var xdglobal
   *  \brief Global domain length in x-direction */
  Real xdglobal = 0;

  /*! \var ybound
   *  \brief Global domain minimum in y-direction */
  Real ybound = 0;

  /*! \var ydglobal
   *  \brief Global domain length in y-direction */
  Real ydglobal = 0;

  /*! \var zbound
   *  \brief Global domain minimum in z-direction */
  Real zbound = 0;

  /*! \var zdglobal
   *  \brief Global domain length in z-direction */
  Real zdglobal = 0;

  /*! \var xblocal
   *  \brief Local domain minimum in x-direction */
  Real xblocal = 0;

  /*! \var xblocal_max
   *  \brief Local domain maximum in x-direction */
  Real xblocal_max = 0;

  /*! \var yblocal
   *  \brief Local domain minimum in y-direction */
  Real yblocal = 0;

  /*! \var yblocal_max
   *  \brief Local domain maximum in y-direction */
  Real yblocal_max = 0;

  /*! \var zblocal
   *  \brief Local domain minimum in z-direction */
  Real zblocal = 0;

  /*! \var zblocal_max
   *  \brief Local domain maximum in z-direction */
  Real zblocal_max = 0;

  /*! \var nx
   *! \brief Local number of grid cells in x-direction, including ghost cells */
  int nx = 0;

  /*! \var ny
   *! \brief Local number of grid cells in y-direction, including ghost cells */
  int ny = 0;

  /*! \var nz
   *! \brief Local number of grid cells in z-direction, including ghost cells */
  int nz = 0;

  /*! \var nghost
   *! \brief Number of ghost cells */
  int nghost = 0;

  /*! \var ncells
   *! \brief Total number of cells */
  int ncells = 0;

  /*! \brief Boundary flags (xl, xu, yl, yu, zl, zu): 1 = periodic on this rank, 5 = MPI neighbor. */
  int flags[6] = {0, 0, 0, 0, 0, 0};

  /*! \brief Convenience accessors by direction (0 = x, 1 = y, 2 = z). */
  Real Local_Min(int dir) const { return dir == 0 ? xblocal : (dir == 1 ? yblocal : zblocal); }
  Real Local_Max(int dir) const { return dir == 0 ? xblocal_max : (dir == 1 ? yblocal_max : zblocal_max); }
  Real Global_Min(int dir) const { return dir == 0 ? xbound : (dir == 1 ? ybound : zbound); }
  Real Global_Len(int dir) const { return dir == 0 ? xdglobal : (dir == 1 ? ydglobal : zdglobal); }
};

#endif  // RT
#endif  // RT_SOURCE_STATE_H
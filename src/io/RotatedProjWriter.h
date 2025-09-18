/*!
 * \file
 * Declares the RotatedProjWriter type
 */

#pragma once

#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../io/FnameTemplate.h"  // define FnameTemplate
#include "../io/ParameterMap.h"   // define ParameterMap

namespace io
{

struct Rotation {
  // TODO: refactor so we can remove the default constructor
  Rotation() = default;

  /*! primary constructor */
  Rotation(const Parameters &P);

  /*! \var nx
   *   \brief Number of pixels in x-dir of rotated, projected image*/
  int nx;

  /*! \var nz
   *   \brief Number of pixels in z-dir of rotated, projected image*/
  int nz;

  /*! \var nx_min
   *   \brief Left most point in the projected image for this subvolume*/
  int nx_min;

  /*! \var nx_max
   *   \brief Right most point in the projected image for this subvolume*/
  int nx_max;

  /*! \var nz_min
   *   \brief Bottom most point in the projected image for this subvolume*/
  int nz_min;

  /*! \var nz_max
   *   \brief Top most point in the projected image for this subvolume*/
  int nz_max;

  /*! \var delta
   *   \brief Rotation angle about z axis in simulation frame*/
  Real delta;

  /*! \var theta
   *   \brief Rotation angle about x axis in simulation frame*/
  Real theta;

  /*! \var phi
   *   \brief Rotation angle about y axis in simulation frame*/
  Real phi;

  /*! \var Lx
   *   \brief Physical x-dir size of projected image*/
  Real Lx;

  /*! \var Lz
   *   \brief Physical z-dir size of projected image*/
  Real Lz;

  /*! \var i_delta
   *   \brief number of output projection for delta rotation*/
  int i_delta;

  /*! \var n_delta
   *   \brief total number of output projection for delta rotation*/
  Real n_delta;

  /*! \var ddelta_dt
   *   \brief rate of delta rotation*/
  Real ddelta_dt;

  /*! \var flag_delta
   *  \brief output mode for box rotation*/
  int flag_delta;
};

/*! \brief A callable that writes rotated projections
 *
 *  \note
 *  The initial skeleton is basically a placeholder
 */
class RotatedProjWriter
{
  /*! Tracks the rotation information */
  Rotation rot_info_;

 public:
  RotatedProjWriter() = delete;
  RotatedProjWriter(const Parameters &P, ParameterMap &pmap) : rot_info_(P) {}

  /*! A callable method that writes a rotated projection of the grid data to file.
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template);
};

}  // namespace io
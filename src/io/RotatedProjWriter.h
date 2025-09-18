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

/*! Tracks rotation information for creating Rotated Projections
 */
struct Rotation {
  // the default constructor would put the instance in an invalid state
  Rotation() = delete;

  /*! Construct a new instance */
  Rotation(ParameterMap &pmap);

  /*! Number of pixels in x-dir of rotated, projected image*/
  int nx;

  /*! Number of pixels in z-dir of rotated, projected image*/
  int nz;

  /*! Left most point in the projected image for this subvolume*/
  int nx_min;

  /*! Right most point in the projected image for this subvolume*/
  int nx_max;

  /*! Bottom most point in the projected image for this subvolume*/
  int nz_min;

  /*! Top most point in the projected image for this subvolume*/
  int nz_max;

  /*! Rotation angle about z axis in simulation frame*/
  Real delta;

  /*! Rotation angle about x axis in simulation frame*/
  Real theta;

  /*! Rotation angle about y axis in simulation frame*/
  Real phi;

  /*! Physical x-dir size of projected image*/
  Real Lx;

  /*! Physical z-dir size of projected image*/
  Real Lz;

  /*! number of output projection for delta rotation*/
  int i_delta;

  /*! total number of output projection for delta rotation*/
  Real n_delta;

  /*! rate of delta rotation*/
  Real ddelta_dt;

  /*! output mode for box rotation*/
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
  RotatedProjWriter(ParameterMap &pmap) : rot_info_(pmap) {}

  /*! A callable method that writes a rotated projection of the grid data to file.
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template);
};

}  // namespace io
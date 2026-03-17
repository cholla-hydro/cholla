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
 *
 *  \note
 *  The reason that the members of this struct aren't direct members of
 *  @ref RotatedProjWriter is purely historical. At this point, there is nothing
 *  stopping us from doing this
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
 *  For more context, a "callable" object is sometimes called a "functor." Essentially
 *  a "callable" object carries around state and can be called like a function.
 */
class RotatedProjWriter
{
  /*! Tracks the rotation information */
  Rotation rot_info_;

 public:
  RotatedProjWriter() = delete;
  RotatedProjWriter(ParameterMap &pmap) : rot_info_(pmap) {}

  /*! Writes the rotated project to disk.
   *
   *  \note
   *  In case you are unaware, this overloads the "function call operator". If we have
   *  an instance, `obj`, then you call this method by invoking
   *  `obj(G, P, nfile, fname_template)`. In python, this method would be named
   *  `__call__`.
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template);
};

}  // namespace io
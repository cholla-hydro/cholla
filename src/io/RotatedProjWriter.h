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

/*! \brief A callable that writes rotated projections
 *
 *  \note
 *  The initial skeleton is basically a placeholder
 */
class RotatedProjWriter
{
 public:
  RotatedProjWriter() = delete;
  RotatedProjWriter(const Parameters &P, ParameterMap &pmap) {}

  /*! A callable method that writes a rotated projection of the grid data to file.
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template);
};

}  // namespace io
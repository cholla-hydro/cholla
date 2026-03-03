/*!
 * \file
 * Declares the ProjectionWriter type
 */

#pragma once

#include <string>
#include <utility>  // std::pair
#include <vector>

#include "../global/global.h"
#include "../grid/grid3D.h"

struct ParameterMap;
struct FieldInfo;
struct FnameTemplate;

namespace io
{

/*! \brief A callable object that writes slice data
 *
 *  Specifically, the object writes xy and xz cell-cenetered projections of the grid
 *  data.
 *
 *  For more context, a "callable" object is sometimes called a "functor." Essentially
 *  a "callable" object carries around state and can be called like a function.
 *
 *  \note
 *  At the time of writing, this doesn't get configured at startup and doesn't need to
 *  a full-blown class. Right now, this mostly exists for consistency with other output
 *  approaches.
 */
class ProjectionWriter
{
 public:
  ProjectionWriter() = default;

  /*! Implements the a callable method that writes slice data
   *
   *  \note
   *  For less experienced C++ developers: this overloads the "function call operator".
   *  If we have an instance, `obj`, then you call this method by invoking
   *  `obj(G, P, nfile, fname_template)`.
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template) const;
};

}  // namespace io
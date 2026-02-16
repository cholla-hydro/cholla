/*!
 * \file
 * Declares the SliceWriter type
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
 *  Specifically, the object writes xy, xz, and yz cell-cenetered slices of the grid
 *  data.
 *
 *  For more context, a "callable" object is sometimes called a "functor." Essentially
 *  a "callable" object carries around state and can be called like a function.
 */
class SliceWriter
{
  /// for each cell-centered field that will be written, this holds a
  /// (field_id, dset_name) pair
  std::vector<std::pair<int, std::string>> cc_field_id_dset_name_pairs_;

 public:
  SliceWriter() = delete;
  SliceWriter(ParameterMap &pmap, const FieldInfo &field_info);

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
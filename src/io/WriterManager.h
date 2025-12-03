/*!
 * \file WriterManager.h
 * \brief Declares the WriterManager type
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../io/FnameTemplate.h"  // define FnameTemplate
#include "../io/ParameterMap.h"   // define ParameterMap

namespace io
{

namespace detail
{

/*! \brief bundles information about a writer */
struct WriterPack {
  /*! specifies the name of the writer (for debugging) */
  std::string name;
  /*! specifies the cadence for invoking the writer */
  int cadence;
  /*! specifes the writer-function or function-like object */
  const std::function<void(Grid3D &, Parameters, int, const FnameTemplate &)> fn;
};

}  // namespace detail

/*! \brief Manages each configured file-writer
 *
 *  The premise is that this instance is created on startup and tracks
 *  and tracks all relevant properties
 */
class WriterManager
{
  FnameTemplate fname_template_;
  std::vector<io::detail::WriterPack> packs_;

 public:
  WriterManager() = delete;
  WriterManager(const Parameters &P, ParameterMap &pmap);

  /*! get the fname-template */
  const FnameTemplate &fname_template() const noexcept { return fname_template_; }

  /*! apply the writers */
  void Apply_Writers(Grid3D &G, const Parameters &P, int nfile) const
  {
    for (const io::detail::WriterPack &pack : packs_) {
      if (nfile % pack.cadence == 0) {
        pack.fn(G, P, nfile, fname_template_);
      }
    }
  }
};

}  // namespace io

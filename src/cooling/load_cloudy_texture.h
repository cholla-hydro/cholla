/*! \file load_cloudy_texture.h
 *  \brief Wrapper file to load cloudy cooling table as CUDA texture. */

#pragma once

#include <cmath>  // pow, log10
#include <string>

#include "../cooling/texture_utilities.h"  // Bilinear_Texture
#include "../global/global.h"
#include "../io/ParameterMap.h"
#include "../utils/shared.h"

namespace cool_component
{

/*! \brief A callable type that uses texture mapping to interpolate Cloudy cooling/heating
 *
 *  For more context, a "callable" object is sometimes called a "functor." Essentially
 *  a "callable" object carries around state and can be called like a function.
 *
 *  This is intended to be used as a cooling component that is used to construct a
 *  cooling recipe.
 *
 *  \note
 *  The built-in tables shipped with Cholla were constructed at z = 0 with solar
 *  metallicity and an HM05 UV background.
 *
 *  Future Ideas
 *  ============
 *  As of now, this class internally tracks both a cooling table AND a heating table. If
 *  we want to support the use of just a single table:
 *  - the easiest thing to do is *probably* to make a new class.
 *  - Since we are fully embracing the approach of building cooling recipes out of 1 or
 *    more cooling components, we could then have the option of replacing this type
 *    with 2 instances of the single-table type (but we then need to figure out how to
 *    elegantly handle constructors)
 */
class CloudyHeatAndCool
{
  // our usage of SharedHandle allows for the wrapped texture objects to be shared
  // among multiple owners (i.e. multiple copies of CloudyHeatAndCool) and ensures that
  // texture objects are properly cleaned up when the number of owners go to 0
  SharedHandle<cudaTextureObject_t> coolTexObj_;
  SharedHandle<cudaTextureObject_t> heatTexObj_;

 public:
  /*! \brief Construct an instance from the appropriate file name
   *
   *  When passed an empty string, it tries to guess the location of the standard data
   *  file.
   *
   *  \note This constructor is useful for testing
   */
  __host__ explicit CloudyHeatAndCool(std::string filename);

  /*! \brief Construct an instance from the ParameterMap */
  __host__ explicit CloudyHeatAndCool(ParameterMap& pmap)
      : CloudyHeatAndCool(pmap.value_or("chemistry.data_file", ""))  // delegate to other constructor
  {
  }

  /*! \brief compute the net cooling contribution
   *
   *  \todo Stop hardcoding properties of the interpolation grid
   *
   *  This primarily exists for testing purposes.
   *
   *  \note
   *  Although I haven't explicitly checked that __forceinline__ is necessary there
   *  isn't any harm since normal operation of Cholla always calls this method through
   *  the operator()(args) method (the only other time this is invoked is in our tests)
   *  is only directly invoked in the tests)
   */
  template <bool TABLE_ONLY>
  __device__ __forceinline__ Real calc_contrib_(Real n, Real T) const
  {
    Real lambda  = 0.0;  // log cooling rate, erg s^-1 cm^3
    Real cooling = 0.0;  // cooling per unit volume, erg /s / cm^3
    Real heating = 0.0;  // heating per unit volume, erg /s / cm^3

    // To keep texture code simple, we use floats (which have built-in support) as opposed to doubles (which would
    // require casting)
    float log_n, log_T;
    log_n = log10(n);
    log_T = log10(T);
    // this temp.
    if ((not TABLE_ONLY) and (log10(T) > 9.0)) {
      lambda = 0.45 * log10(T) - 26.065;
    } else if (TABLE_ONLY or (log10(T) >= 1.0)) {
      // remap coordinates for texture
      // remapped = (input - TABLE_MIN_VALUE)*(1/TABLE_SPACING)
      // remapped = (input - TABLE_MIN_VALUE)*(NUM_CELLS_PER_DECADE)
      const Real remap_log_T = (log_T - 1.0) * 10;
      const Real remap_log_n = (log_n + 6.0) * 10;

      lambda       = Bilinear_Texture(this->coolTexObj_.get(), remap_log_T, remap_log_n);
      const Real H = Bilinear_Texture(this->heatTexObj_.get(), remap_log_T, remap_log_n);
      // heating      = pow(10, H); //TODO: uncomment
    } else {
      // Do nothing below 10 K
      return 0.0;
    }

    cooling = pow(10, lambda);
    return n * n * (cooling - heating);
  }

  /*! \brief compute the net cooling/heating rate
   *
   *  The docstring of the \ref cool_component namespace provides further context about
   *  how this method is used and describes some relevant optimization considerations
   *
   *  \todo Stop hardcoding properties of the interpolation grid
   *
   *  \note
   *  In case you are unaware, this overloads the "function call operator". If we have an
   *  instance, `obj`, then you call this method by invoking `obj(n, T)`. In python,
   *  this method would be called `__call__`
   */
  __device__ Real operator()(Real n, Real T) const { return calc_contrib_<false>(n, T); }
};

}  // namespace cool_component
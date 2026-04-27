/*! \file load_cloudy_texture.h
 *  \brief Wrapper file to load cloudy cooling table as CUDA texture. */

#pragma once

#include <cmath>  // pow, log10
#include <string>

#include "../cooling/texture_utilities.h"  // Bilinear_Texture
#include "../global/global.h"

// todo: stop tracking these as globals
extern cudaTextureObject_t coolTexObj;
extern cudaTextureObject_t heatTexObj;

/* \fn void Load_Cuda_Textures()
 * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
void Load_Cuda_Textures(std::string filename);

/* \fn void Free_Cuda_Textures()
 * \brief Unbind the texture memory on the GPU, and free the associated Cuda
 * arrays. */
void Free_Cuda_Textures();

/*! \brief Describes cooling components
 *
 *  \todo if we create a file that holds all non-tabulated cool-components, we may want
 *        to move this docstring to that file
 *
 *  \note Maybe we should call this namespace edot_component? (Since there's heating & cooling)
 *
 *  Each cooling recipe is composed of one or more cooling components. A cooling
 *  component is a "callable" (i.e. it's a function or it's a class that implements
 *  the appropriate method that let's be called just like a function).
 *
 *  Optimization Notes
 *  ==================
 *  It's important that that "core-logic" of each cooling component is implemented in a
 *  header file or in the same source file where a recipe actually invokes the
 *  "core logic". The "core-logic" includes code paths accessible through function-call
 *  syntax. To be more explicit,
 *  - when a cool-component is a regular function, the "core-logic" includes the whole
 *    function itself and any (user-defined) functions called by that function
 *  - for a callable class, the "core-logic" includes the definition of the
 *    `Real operator()(args...)` member function and any (user-defined) functions
 *    called by that member-function
 *
 *  We may also want to consider using ``__forceinline__``. With that said, we should
 *  only use ``__forceinline__`` if it explicitly provides a performance improvement
 *  (blindly using ``__forceinline__`` can actually hurt performance)
 *
 *  In the future, if most heating and cooling contributiond need ``log10(n)`` and
 *  ``log10(T)``, we may want to consider pre-computing those values. In this scenario,
 *  we probably want to continue providing ``n`` and ``T``. This could potentially
 *  improve performance in recipies using multiple independent contributions since
 *  ``log10`` and ``pow`` are generally a lot more expensive than most other operations
 *  relevant for computing heating and cooling
 */
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
 *  - Since we are fully embracing the approach of building cooling recipies out of 1
 *    more cooling components, we could then have the option of replacing this type
 *    with 2 instances of the single-table type (but we then need to figure out how to
 *    elegantly handle constructors)
 */
class CloudyHeatAndCool
{
  cudaTextureObject_t coolTexObj_;
  cudaTextureObject_t heatTexObj_;

 public:
  /*! \brief Primary Constructor */
  __host__ explicit CloudyHeatAndCool(std::string filename);

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
  __device__ Real operator()(Real n, Real T) const
  {
    Real lambda  = 0.0;  // log cooling rate, erg s^-1 cm^3
    Real cooling = 0.0;  // cooling per unit volume, erg /s / cm^3
    Real heating = 0.0;  // heating per unit volume, erg /s / cm^3

    // To keep texture code simple, we use floats (which have built-in support) as opposed to doubles (which would
    // require casting)
    float log_n, log_T;
    log_n = log10(n);
    log_T = log10(T);

    // remap coordinates for texture
    // remapped = (input - TABLE_MIN_VALUE)*(1/TABLE_SPACING)
    // remapped = (input - TABLE_MIN_VALUE)*(NUM_CELLS_PER_DECADE)
    const Real remap_log_T = (log_T - 1.0) * 10;
    const Real remap_log_n = (log_n + 6.0) * 10;

    // Note: although the cloudy table columns are n,T,L,H , T is the fastest
    // variable so it is treated as "x" This is why the Texture calls are T first,
    // then n: Bilinear_Texture(tex, remap_log_T, remap_log_n)

    // cloudy cooling tables cut off at 10^9 K, use the CIE analytic fit above
    // this temp.
    if (log10(T) > 9.0) {
      lambda = 0.45 * log10(T) - 26.065;
    } else if (log10(T) >= 1.0) {
      lambda       = Bilinear_Texture(this->coolTexObj_, remap_log_T, remap_log_n);
      const Real H = Bilinear_Texture(this->heatTexObj_, remap_log_T, remap_log_n);
      heating      = pow(10, H);
    } else {
      // Do nothing below 10 K
      return 0.0;
    }

    cooling = pow(10, lambda);
    return n * n * (cooling - heating);
  }
};

}  // namespace cool_component
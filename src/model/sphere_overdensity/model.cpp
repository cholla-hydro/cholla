/*! \file
 *  \brief Define logic pertaining to the SphericalOverdensity model
 */

#include "model.h"

#include "../../io/ParameterMap.h"

SphericalOverdensity::SphericalOverdensity(ParameterMap& pmap)
    : bkg_density{0.0005}, overdensity{1.0}, radius{0.2}, center_xyz{0.5, 0.5, 0.5}
{
  // the following assignments were commented out in the location where we originally
  // took the initial values from:

  // bkg_density = mu * MP / DENSITY_UNIT; // 1 particles per cm^3)
  // over_density = 1000 * mu * MP / DENSITY_UNIT; // 100 particles per cm^3
}
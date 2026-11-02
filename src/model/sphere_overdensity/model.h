/*! \file
 *  \brief Define/declare machinery pertaining to spherical overdensity test problem
 */

#pragma once

#include "../../global/global.h"  // Real

// forward declarations to avoid directly (or indirectly) including the ParameterMap
// header file
struct ParameterMap;

/*! \brief A centralized for aggregating properties of the model used with the spherical
 *  overdensity model.
 *
 *  This acts as a model because certain properties need to be known at initialization
 *  and when updating boundary conditions.
 *
 *  \note
 *  Unless you can get the legacy SOR gravity solver to work properly, it seems highly
 *  unlikely that this logic will work at all
 */
struct SphericalOverdensity {
  Real bkg_density;
  Real overdensity;
  Real radius;
  Real center_xyz[3];

  /*! \brief primary constructor */
  explicit SphericalOverdensity(ParameterMap& pmap);
};
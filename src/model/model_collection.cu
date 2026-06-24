/*! \file
 *  Implements methods for the \ref ModelCollection type
 */

#include "../io/ParameterMap.h"
#include "disk_galaxy.h"
#include "model_collection.h"

ModelCollection::ModelCollection(ParameterMap& pmap)
{
  // in the future, we will dynamically build this thing from pmap
  ClusteredDiskGalaxy galaxy_model = galaxies::get_MW_model();
  vec_.emplace_back(galaxy_model);
}
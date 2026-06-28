/*! \file
 *  Implements methods for the \ref ModelCollection type
 */

#include "../io/ParameterMap.h"
#include "galaxy/disk_galaxy.h"
#include "model_collection.h"

ModelCollection::ModelCollection(ParameterMap& pmap)
{
  // as we add new models, we will add more snipets like the following
  //
  //   if (pmap.Contains_Table("model.<model-subtable>")) {
  //     vec_.emplace_back(<ModelClass>(pmap));
  //   }
  //
  // where <ModelClass> will be replaced with the name of the model class and
  // <model-subtable> will be replaced with the name of the corresponding parameter
  // file subtable

  // the galaxy_model is still a special case, we will start treating it like a normal
  // case soon
  DiskGalaxy galaxy_model = galaxies::make_MW_model();
  vec_.emplace_back(galaxy_model);

  // once we parse at least one model in the "normal way," we should insert
  //    pmap.Enforce_Table_Content_Uniform_Access_Status("model", false);
  // to abort if we haven't read all parameters from within the "model" table
}
/*LICENSE*/

#ifndef RT_PHYSICS_ATOMIC_DATA_H
#define RT_PHYSICS_ATOMIC_DATA_H

//
//  Various atomic, chemical, and thermal rates.
//
#include "atomic_data_decl.h"

namespace rt_physics
{
namespace rt_atomic_data
{
//
//  Cross-sections are in barns (to limit the numeric range and fit into float).
//  The x-axis in the table (frequency) is represented as xi = log(hnu/1Ry).
//
const CrossSection* CrossSections();
const CrossSection* CrossSectionsGPU();

void Create();
void Delete();
};  // namespace rt_atomic_data
};  // namespace rt_physics

#endif  // RT_PHYSICS_ATOMIC_DATA_H

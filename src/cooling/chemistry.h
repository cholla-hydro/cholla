// this is the public header for unified chemistry heating and cooling

#pragma once

#include <functional>

#include "../global/global.h"
#include "../grid/grid3D.h"

/* construct the chemistry callback (or not based on the specified parameters & compilation mode)
 *
 * \note
 * we always define the following function regardless of the defined compiler flags */
std::function<void(Grid3D&)> configure_chemistry_callback(ParameterMap& pmap);
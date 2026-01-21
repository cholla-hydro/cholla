/*! \file
 *  Define machinery for accessing field information
 */

#pragma once

#include "../utils/FrozenKeyIdxBiMap.h"

/*! Construct a bidirectional mapping between field name and field value
 */
utils::FrozenKeyIdxBiMap get_field_id_mapping();
#pragma once

// An enum which holds offsets for grid quantities
// In the final form of this approach, this file will also set nfields (not yet) and NSCALARS (done)
// so that adding a field only requires registering it here: 
// grid knows to allocate memory based on nfields and NSCALARS 
// and values can be accessed with density[ncells*grid_enum::enum_name + id]
// example: C.device[H.n_cells*grid_enum::basic_scalar + id]


// enum notes:
// For advanced devs: must be "unscoped" to be implicitly treated as int: this means cannot use "enum class" or "enum struct"
// Wrapped in namespace to give it an effective scope to prevent collisions
// enum values (i.e. density) belong to their enclosing scope, which necessitates the namespace wrapping
// --otherwise "density" would be available in global scope
// ": int" forces underlying type to be int

namespace grid_enum {
enum : int {

  // Don't change order of hydro quantities until all of hydro is made consistent with grid_enum (if ever) because enum values depend on order
  density,
  momentum_x,
  momentum_y,
  momentum_z,
  Energy,

  // Code assumes scalars are a contiguous block
  // Always define scalar, scalar_minus_1, finalscalar_plus_1, finalscalar to compute NSCALARS
  scalar,
  scalar_minus_1 = scalar - 1,// so that next enum item starts at same index as scalar

  #ifdef SCALAR
  // Add scalars here, wrapped appropriately with ifdefs:
  #ifdef BASIC_SCALAR
  basic_scalar,
  #endif

  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  HI_density,
  HII_density,
  HeI_density,
  HeII_density,
  HeIII_density,
  e_density,
  #ifdef GRACKLE_METALS
  metal_density,
  #endif
  #endif

  #endif // SCALAR

  finalscalar_plus_1, // needed to calculate NSCALARS
  finalscalar = finalscalar_plus_1 - 1, // resets enum to finalscalar so fields afterwards are correct  
  // so that anything after starts with scalar + NSCALARS

  #ifdef MHD
  magnetic_x,
  magnetic_y,
  magnetic_z,
  #endif
  #ifdef DE
  GasEnergy,
  #endif
  num_fields,

//Aliases
  nscalars = finalscalar_plus_1 - scalar,

};
}

#define NSCALARS grid_enum::nscalars

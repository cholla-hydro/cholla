#pragma once

// An experimental enum which holds offsets for grid quantities
// In the final form of this approach, this file will also set nfields and NSCALARS, 
// so that adding a field only requires registering it here.


// Must be unscoped to be treated as int
// ": int" forces underlying type to be int
enum grid_enum : int {

  // Don't touch hydro quantities until all of hydro is refactored (if ever)
  density,
  momentum_x,
  momentum_y,
  momentum_z,
  Energy,

  // Code assumes scalars are a contiguous block
  #ifdef SCALAR
  scalar,
  scalar_minus_1 = scalar - 1,// so that next enum item starts at same index as scalar

  // TODO: Add scalars here:


  finalscalar_plus_1,
  // TODO: set finalscalar = finalscalar_plus_1 - 1, and then define NSCALARS equivalent from here. 
  finalscalar = scalar + NSCALARS - 1, 
  // so that anything after starts with scalar + NSCALARS
  #endif // SCALAR
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
  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  HI_density = scalar,
  HII_density,
  HeI_density,
  HeII_density,
  HeIII_density,
  e_density,
  #ifdef GRACKLE_METALS
  metal_density,
  #endif
  #endif
};

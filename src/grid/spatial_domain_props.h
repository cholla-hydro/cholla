#pragma once

#include "../global/global.h"

class Grav3D;
class Grid3D;
struct Parameters;

/* This is a collection of 15 quantities that appear in 3 other locations throughout the codebase.
 *
 * The struct primarily exists to simplify the process of copying these values from one place to
 * another. (But it may make sense to refactor other parts of the code in terms of this object)
 */
struct SpatialDomainProps {
  // number of cells in the local domain
  int nx_local, ny_local, nz_local;

  // total number of cells in the entire (global) domain
  int nx_total, ny_total, nz_total;

  // Left boundaries of the local domain
  Real xMin, yMin, zMin;

  // Right boundaries of the local domain
  Real xMax, yMax, zMax;

  // cell widths
  Real dx, dy, dz;

  static SpatialDomainProps From_Grav3D(Grav3D& grav);
  static SpatialDomainProps From_Grid3D(Grid3D& grid, Parameters* P);
};
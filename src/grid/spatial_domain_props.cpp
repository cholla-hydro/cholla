#include "../grid/spatial_domain_props.h"

#include "../gravity/grav3D.h"
#include "../grid/grid3D.h"

SpatialDomainProps SpatialDomainProps::From_Grav3D(Grav3D& grav) { return grav.spatial_props; }

SpatialDomainProps SpatialDomainProps::From_Grid3D(Grid3D& grid, struct Parameters* P)
{
  SpatialDomainProps out;

  // Set Box Left boundary positions
  out.xMin = grid.H.xblocal;  // x_min
  out.yMin = grid.H.yblocal;  // y_min
  out.zMin = grid.H.zblocal;  // z_min

  // Set Box Right boundary positions
  out.xMax = grid.H.xblocal_max;  // x_max;
  out.yMax = grid.H.yblocal_max;  // y_max;
  out.zMax = grid.H.zblocal_max;  // z_max;

  // Set uniform ( dx, dy, dz )
  out.dx = grid.H.dx;  // dx_real;
  out.dy = grid.H.dy;  // dy_real;
  out.dz = grid.H.dz;  // dz_real;

  // Set Box Total number of cells
  out.nx_total = P->nx;  // nx;
  out.ny_total = P->ny;  // ny;
  out.nz_total = P->nz;  // nz;

  // Set Box local domain number of cells
  out.nx_local = grid.H.nx_real;  // nx_real;
  out.ny_local = grid.H.ny_real;  // ny_real;
  out.nz_local = grid.H.nz_real;  // nz_real;
  return out;
};

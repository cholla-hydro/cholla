/*! \file rt_source_io.cpp
 *  \brief Per-rank source files: the initial catalog and the RT restart file share one
 *         layout, so a single reader serves both.
 *
 *  Layout of group /sources (one file per rank):
 *    attribute n_local   (int)        number of sources in this file
 *    attribute n_fields  (int)        SRC_NFIELDS
 *    attribute local_min (double[3])  lower corner of the local domain the file was written for
 *    attribute local_max (double[3])  upper corner
 *    dataset   data      (double, n_local x n_fields), fields in SourceField order:
 *              x y z vx vy vz lum t_birth t_life id
 *  Positions in code length units (comoving kpc/h under COSMOLOGY); velocities in the
 *  particle convention; lum = L / L_ref; t_birth and t_life in Myr (t_life <= 0: never dies). */

#ifdef RT

  #include <cmath>
  #include <string>
  #include <vector>

  #include "../../io/io.h"
  #include "../../utils/error_handling.h"
  #include "rt_sources.h"

using namespace rt_sources;

  #ifdef HDF5

namespace
{

void Write_Int_Attr(hid_t loc, const char *name, int value)
{
  hsize_t one   = 1;
  hid_t space   = H5Screate_simple(1, &one, NULL);
  hid_t attr_id = H5Acreate(loc, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, H5T_NATIVE_INT, &value);
  H5Aclose(attr_id);
  H5Sclose(space);
}

void Write_Vec3_Attr(hid_t loc, const char *name, const double v[3])
{
  hsize_t three = 3;
  hid_t space   = H5Screate_simple(1, &three, NULL);
  hid_t attr_id = H5Acreate(loc, name, H5T_IEEE_F64BE, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, H5T_NATIVE_DOUBLE, v);
  H5Aclose(attr_id);
  H5Sclose(space);
}

void Read_Attr(hid_t loc, const char *name, hid_t mem_type, void *buf, const char *fname)
{
  if (H5Aexists(loc, name) <= 0) CHOLLA_ERROR("RT sources: %s: /sources has no attribute %s.", fname, name);
  hid_t attr_id = H5Aopen(loc, name, H5P_DEFAULT);
  H5Aread(attr_id, mem_type, buf);
  H5Aclose(attr_id);
}

}  // namespace

void RTSourceModel::Write_Restart_HDF5(hid_t file_id) const
{
  std::vector<Real> packed;
  store.Copy_To_Host(packed);

  hid_t group_id = H5Gcreate(file_id, "/sources", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  Write_Int_Attr(group_id, "n_local", store.n_local);
  Write_Int_Attr(group_id, "n_fields", SRC_NFIELDS);
  const double lmin[3] = {state.xblocal, state.yblocal, state.zblocal};
  const double lmax[3] = {state.xblocal_max, state.yblocal_max, state.zblocal_max};
  Write_Vec3_Attr(group_id, "local_min", lmin);
  Write_Vec3_Attr(group_id, "local_max", lmax);

  // Ranks without sources write no dataset; the reader only opens it when n_local > 0.
  if (store.n_local > 0) {
    hsize_t dims[2] = {(hsize_t)store.n_local, (hsize_t)SRC_NFIELDS};
    hid_t space_id  = H5Screate_simple(2, dims, NULL);
    Write_HDF5_Dataset(file_id, space_id, packed.data(), "/sources/data");
    H5Sclose(space_id);
  }
  H5Gclose(group_id);
}

void RTSourceModel::Read_Sources_HDF5(const char *fname)
{
  hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) CHOLLA_ERROR("RT sources: cannot open per-rank source file %s.", fname);
  if (H5Lexists(file_id, "/sources", H5P_DEFAULT) <= 0) {
    CHOLLA_ERROR("RT sources: %s has no /sources group (restart written with rt_n_pops = 0?).", fname);
  }

  hid_t group_id = H5Gopen(file_id, "/sources", H5P_DEFAULT);
  int n = 0, nf = 0;
  double fmin[3], fmax[3];
  Read_Attr(group_id, "n_local", H5T_NATIVE_INT, &n, fname);
  Read_Attr(group_id, "n_fields", H5T_NATIVE_INT, &nf, fname);
  Read_Attr(group_id, "local_min", H5T_NATIVE_DOUBLE, fmin, fname);
  Read_Attr(group_id, "local_max", H5T_NATIVE_DOUBLE, fmax, fname);
  H5Gclose(group_id);

  if (nf != SRC_NFIELDS) {
    CHOLLA_ERROR("RT sources: %s has %d fields per source, the code expects %d.", fname, nf, SRC_NFIELDS);
  }

  // The file must have been split for this rank's subvolume. Compare corners to a small
  // fraction of a cell: a different decomposition differs by whole cells.
  const Real cell[3] = {state.dx, state.dy, state.dz};
  for (int d = 0; d < 3; d++) {
    const Real tol = 1.0e-6 * cell[d];
    if (std::fabs(fmin[d] - state.Local_Min(d)) > tol || std::fabs(fmax[d] - state.Local_Max(d)) > tol) {
      CHOLLA_ERROR(
          "RT sources: %s was written for local domain [%g, %g) in direction %d, but rank %d owns [%g, %g). "
          "Re-split the catalog for this decomposition (python/scripts/rt_split_sources.py).",
          fname, fmin[d], fmax[d], d, procID, state.Local_Min(d), state.Local_Max(d));
    }
  }

  std::vector<Real> packed((size_t)n * SRC_NFIELDS);
  if (n > 0) {
    // check the dataset really holds n x n_fields values before reading into the buffer
    hid_t dset_id   = H5Dopen(file_id, "/sources/data", H5P_DEFAULT);
    hid_t space_id  = H5Dget_space(dset_id);
    hssize_t npoint = H5Sget_simple_extent_npoints(space_id);
    H5Sclose(space_id);
    H5Dclose(dset_id);
    if (npoint != (hssize_t)packed.size()) {
      CHOLLA_ERROR("RT sources: %s: /sources/data has %lld values, expected %d x %d.", fname, (long long)npoint, n,
                   SRC_NFIELDS);
    }
    Read_HDF5_Dataset(file_id, packed.data(), "/sources/data");
  }
  H5Fclose(file_id);

  // Every source must lie in this subvolume. Allow one cell of slack for rounding at the
  // faces; a source just outside is handed to its owner by the first Migrate.
  for (int i = 0; i < n; i++) {
    for (int d = 0; d < 3; d++) {
      const Real x = packed[(size_t)i * SRC_NFIELDS + SRC_X + d];
      if (x < state.Local_Min(d) - cell[d] || x >= state.Local_Max(d) + cell[d]) {
        CHOLLA_ERROR("RT sources: %s: source %d (id %.0f) is at %g in direction %d, outside [%g, %g).", fname, i,
                     packed[(size_t)i * SRC_NFIELDS + SRC_ID], x, d, state.Local_Min(d), state.Local_Max(d));
      }
    }
  }

  store.Append_Host(packed.data(), n);
}

  #endif  // HDF5

#endif  // RT
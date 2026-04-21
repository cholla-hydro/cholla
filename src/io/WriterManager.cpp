/*!
 * \file WriterManager.h
 * \brief Contains the WriterManager type
 */

#include "../io/WriterManager.h"

#include <functional>
#include <limits>
#include <numeric>  // std::lcm
#include <string>
#include <vector>

#include "../gravity/grav3D.h"
#include "../io/FieldWriter.h"        // FieldWriter
#include "../io/ParameterMap.h"       // define ParameterMap
#include "../io/ProjectionWriter.h"   // ProjectionWriter
#include "../io/RotatedProjWriter.h"  // RotatedProjWriter
#include "../io/SliceWriter.h"        // SliceWriter
#include "../io/io.h"
#include "../utils/error_handling.h"

io::WriterManager::WriterManager(const Parameters& P, ParameterMap& pmap, const FieldInfo& field_info)
    : fname_template_(FnameTemplate::from_pmap(pmap))
{
  int ndim;
  if ((P.nx > 1) and (P.ny > 1) and (P.nz > 1)) {
    ndim = 3;
  } else if ((P.nx > 1) and (P.ny > 1) and (P.nz == 1)) {
    ndim = 2;
  } else if ((P.nx > 1) and (P.ny == 1) and (P.nz == 1)) {
    ndim = 1;
  } else {
    CHOLLA_ERROR("Parameter file had unexpected dimensions");
  }

  // in the future, the goal is to read directly from ParameterMap (so we can stop storing
  // some of the relevant variables in Parameters)
  const int n_hydro = pmap.value_or("n_hydro", 1);
  CHOLLA_ASSERT(n_hydro >= 1, "n_hydro must be positive");

#ifndef ONLY_PARTICLES
  {
    // setup the data output routine for Hydro data
    std::pair<io::WriterFn, std::string> rslt = io::FieldWriter::try_create(ndim, pmap, field_info, false);
    if (rslt.first) {
      packs_.push_back(io::detail::WriterPack{"hydro", n_hydro, rslt.first});
    } else {
      chprintf("WARNING: %s\n", rslt.second.c_str());
    }
  }
#endif

  int n_out_float32 = pmap.value_or("n_out_float32", 0);
  if (n_out_float32) {
    CHOLLA_ASSERT(n_out_float32 > 0, "n_out_float32 can't be negative");

    // Historically, we would invoke float32 output function at a cadence set by n_hydro and
    // immediately exit if nfile isn't also a multiple of `n_out_float32`
    // -> for consistency, we now just set the cadence to the lcm of n_hydro & n_out_float32
    int64_t lcm = std::lcm(int64_t{n_hydro}, int64_t{n_out_float32});
    CHOLLA_ASSERT(lcm <= int64_t{std::numeric_limits<int>::max()},
                  "the lcm of n_hydro and n_out_float32 can't be represented by an int");
    int cadence = static_cast<int>(lcm);

    std::pair<io::WriterFn, std::string> rslt = io::FieldWriter::try_create(ndim, pmap, field_info, true);
    if (not rslt.first) {
      CHOLLA_ERROR("Error while preparing dumps for f32 hdf5 outputs: %s", rslt.second.c_str());
    }
    packs_.push_back(io::detail::WriterPack{"hydro-f32", cadence, rslt.first});
  }

#ifdef PROJECTION
  packs_.push_back(io::detail::WriterPack{"projection", pmap.value_or("n_projection", 1), ProjectionWriter()});
#endif /*PROJECTION*/

#ifdef ROTATED_PROJECTION
  packs_.push_back(io::detail::WriterPack{
      "rotated_projection", pmap.value_or("n_rotated_projection", 1), {io::RotatedProjWriter(pmap)}});
#endif /*ROTATED_PROJECTION*/

#ifdef SLICES
  packs_.push_back(io::detail::WriterPack{"slice", pmap.value_or("n_slice", 1), {io::SliceWriter(pmap, field_info)}});
#endif /*SLICES*/

#ifdef PARTICLES
  // define a lambda function
  auto write_particle = [](Grid3D& G, Parameters P, int nfile, const FnameTemplate& fname_template) {
    G.WriteData_Particles(P, nfile, fname_template);
  };
  packs_.push_back(io::detail::WriterPack{"particle", pmap.value_or("n_particle", 1), write_particle});

#endif

#if defined(GRAVITY) && defined(HDF5)
  auto write_gravity = [](Grid3D& G, Parameters P, int nfile, const FnameTemplate& fname_template) {
    G.Grav.Write_Restart_HDF5(&P, nfile, fname_template);
  };
  int n_gravity = 1;  // <- this is the historical choice
  packs_.push_back(io::detail::WriterPack{"gravity", n_gravity, write_gravity});

#endif
}

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
#include "../io/FieldWriter.h"   // FieldWriter
#include "../io/ParameterMap.h"  // define ParameterMap
#include "../io/io.h"
#include "../utils/error_handling.h"

io::WriterManager::WriterManager(const Parameters& P, ParameterMap& pmap, const FieldInfo& field_info)
    : fname_template_(P)
{
  bool is_3D = (P.ny > 1) && (P.nz > 1);
  // in the future, the goal is to read directly from ParameterMap (so we can stop storing
  // some of the relevant variables in Parameters)
  const int n_hydro = pmap.value_or("n_hydro", 1);
  CHOLLA_ASSERT(n_hydro >= 0, "n_hydro must be positive");

#ifndef ONLY_PARTICLES
  // setup the data output routine for Hydro data
  packs_.push_back(io::detail::WriterPack{"hydro", n_hydro, {io::FieldWriter(pmap, field_info)}});
#endif

#ifdef HDF5
  // TODO: move these checks to a factory function of F32FieldWriter that may fail
  int n_out_float32 = pmap.value_or("n_out_float32", 0);
  if (n_out_float32) {
    CHOLLA_ASSERT(is_3D, "float32 outputs only supported in 3D simulations");
    CHOLLA_ASSERT(n_out_float32 > 0, "n_out_float32 can't be negative");

    // Historically, we would invoke float32 output function at a cadence set by n_hydro and
    // immediately exit if nfile isn't also a multiple of `n_out_float32`
    // -> for consistency, we now just set the cadence to the lcm of n_hydro & n_out_float32
    int64_t lcm = std::lcm(int64_t{n_hydro}, int64_t{n_out_float32});
    CHOLLA_ASSERT(lcm <= int64_t{std::numeric_limits<int>::max()},
                  "the lcm of n_hydro and n_out_float32 can't be represented by an int");
    int cadence = static_cast<int>(lcm);

    packs_.push_back(io::detail::WriterPack{"hydro-f32", cadence, {io::F32FieldWriter(pmap, field_info)}});
  }
#endif

#ifdef PROJECTION
  packs_.push_back(io::detail::WriterPack{"projection", pmap.value_or("n_projection", 1), &Output_Projected_Data});
#endif /*PROJECTION*/

#ifdef ROTATED_PROJECTION
  packs_.push_back(io::detail::WriterPack{"rotated_projection", pmap.value_or("n_rotated_projection", 1),
                                          &Output_Rotated_Projected_Data});
#endif /*ROTATED_PROJECTION*/

#ifdef SLICES
  packs_.push_back(io::detail::WriterPack{"slice", pmap.value_or("n_slice", 1), &Output_Slices});
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

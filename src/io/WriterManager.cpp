/*!
 * \file WriterManager.h
 * \brief Contains the WriterManager type
 */

#include "../io/WriterManager.h"

#include <functional>
#include <string>
#include <vector>

#include "../gravity/grav3D.h"
#include "../io/ParameterMap.h"  // define ParameterMap
#include "../io/io.h"

io::WriterManager::WriterManager(const Parameters& P, ParameterMap& pmap) : fname_template_(P)
{
  // in the future, the goal is to read directly from ParameterMap (so we can stop storing
  // some of the relevant variables in Parameters)
  const int n_hydro = pmap.value_or("n_hydro", 1);

#ifndef ONLY_PARTICLES
  // setup the data output routine for Hydro data
  packs_.push_back(io::detail::WriterPack{"hydro", n_hydro, &Output_Data});
#endif

  // This function does other checks to make sure it is valid (3D only)
#ifdef HDF5
  if (pmap.value_or("n_out_float32", 0)) {
    packs_.push_back(io::detail::WriterPack{"hydro-f32", n_hydro, &Output_Float32});
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

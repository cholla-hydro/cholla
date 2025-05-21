/*!
 * \file WriterManager.h
 * \brief Contains the WriterManager type
 */

#include "../io/WriterManager.h"

#include <functional>
#include <string>
#include <vector>

#include "../io/io.h"

io::WriterManager::WriterManager(const Parameters &P)
{
  // in the future, the goal is to read directly from ParameterMap (so we can stop storing
  // some of the relevant variables in Parameters)

#ifndef ONLY_PARTICLES
  /*call the data output routine for Hydro data*/
  packs_.push_back(io::detail::WriterPack{"hydro", P.n_hydro, &Output_Data});
#endif

// This function does other checks to make sure it is valid (3D only)
#ifdef HDF5
  if (P.n_out_float32) {
    packs_.push_back(io::detail::WriterPack{"hydro-f32", P.n_hydro, &Output_Float32});
  }
#endif

#ifdef PROJECTION
  packs_.push_back(io::detail::WriterPack{"projection", P.n_projection, &Output_Projected_Data});
#endif /*PROJECTION*/

#ifdef ROTATED_PROJECTION
  packs_.push_back(
      io::detail::WriterPack{"rotated_projection", P.n_rotated_projection, &Output_Rotated_Projected_Data});
#endif /*ROTATED_PROJECTION*/

#ifdef SLICES
  packs_.push_back(io::detail::WriterPack{"slice", P.n_slice, &Output_Slices});
#endif /*SLICES*/

#ifdef PARTICLES
  // define a lambda function
  auto write_particle = [](Grid3D &G, Parameters P, int nfile) { G.WriteData_Particles(P, nfile); };
  packs_.push_back(io::detail::WriterPack{"particle", P.n_particle, write_particle});

#endif
}

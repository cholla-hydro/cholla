/*!
 * \file
 * Implements the RotatedProjWriter type
 */

#include "RotatedProjWriter.h"

#include <cmath>  // M_PI (note: not guaranteed by the C++ standard)

#include "../global/global.h"  // Parameters
#include "../grid/grid3D.h"
#include "../io/io.h"

io::Rotation::Rotation(ParameterMap &pmap)
{
  // thoughts for the future:
  // do we want to enforce valid domains for arguments? We might require that
  // - nxr & nzr are positive
  // - delta, phi, & theta satisfy 0 <= x < 180 (maybe we want to be more flexible)
  // - Lx & Lz are positive (or at least non-zero)
  // - I think flag_delta should only be 0, 1, or 2

  // x-dir pixels in projection
  this->nx = pmap.value<int>("nxr");
  // z-dir pixels in projection
  this->nz = pmap.value<int>("nzr");
  // minimum x location to project
  this->nx_min = 0;
  // minimum z location to project
  this->nz_min = 0;
  // maximum x location to project
  this->nx_max = this->nx;
  // maximum z location to project
  this->nz_max = this->nz;
  // rotation angle about z direction
  this->delta = Real(M_PI * (pmap.value_or("delta", 0.0) / 180.));  // convert to radians
  // rotation angle about x direction
  this->theta = Real(M_PI * (pmap.value_or("theta", 0.0) / 180.));  // convert to radians
  // rotation angle about y direction
  this->phi = Real(M_PI * (pmap.value_or("phi", 0.0) / 180.));  // convert to radians
  // x-dir physical size of projection
  this->Lx = Real(pmap.value<double>("Lx"));
  // z-dir physical size of projection
  this->Lz = Real(pmap.value<double>("Lz"));
  // initialize a counter for rotated outputs
  this->i_delta = 0;
  // number of rotated outputs in a complete revolution
  this->n_delta = pmap.value_or("n_delta", 0);
  // rate of rotation between outputs, for an actual simulation
  this->ddelta_dt = Real(pmap.value_or("ddelta_dt", 0.0));
  // are we not rotating about z(0)?
  // are we outputting multiple rotations(1)? or rotating during a
  // simulation(2)?
  this->flag_delta = pmap.value_or("flag_delta", 0);
}

void io::RotatedProjWriter::operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template)
{
  Output_Rotated_Projected_Data(G, P, nfile, fname_template, this->rot_info_);
}
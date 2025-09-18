/*!
 * \file
 * Implements the RotatedProjWriter type
 */

#include "RotatedProjWriter.h"

#include <cmath>  // M_PI (note: not guaranteed by the C++ standard)

#include "../global/global.h"  // Parameters
#include "../grid/grid3D.h"
#include "../io/io.h"

io::Rotation::Rotation(const Parameters &P, ParameterMap &pmap)
{
#ifdef ROTATED_PROJECTION
  // x-dir pixels in projection
  this->nx = P.nxr;
  // z-dir pixels in projection
  this->nz = P.nzr;
  // minimum x location to project
  this->nx_min = 0;
  // minimum z location to project
  this->nz_min = 0;
  // maximum x location to project
  this->nx_max = this->nx;
  // maximum z location to project
  this->nz_max = this->nz;
  // rotation angle about z direction
  this->delta = M_PI * (P.delta / 180.);  // convert to radians
  // rotation angle about x direction
  this->theta = M_PI * (P.theta / 180.);  // convert to radians
  // rotation angle about y direction
  this->phi = M_PI * (P.phi / 180.);  // convert to radians
  // x-dir physical size of projection
  this->Lx = P.Lx;
  // z-dir physical size of projection
  this->Lz = P.Lz;
  // initialize a counter for rotated outputs
  this->i_delta = 0;
  // number of rotated outputs in a complete revolution
  this->n_delta = P.n_delta;
  // rate of rotation between outputs, for an actual simulation
  this->ddelta_dt = P.ddelta_dt;
  // are we not rotating about z(0)?
  // are we outputting multiple rotations(1)? or rotating during a
  // simulation(2)?
  this->flag_delta = P.flag_delta;
#endif /*ROTATED_PROJECTION*/
}

void io::RotatedProjWriter::operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template)
{
  Output_Rotated_Projected_Data(G, P, nfile, fname_template, this->rot_info_);
}
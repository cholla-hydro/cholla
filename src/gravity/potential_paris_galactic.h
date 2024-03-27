#pragma once

#ifdef PARIS_GALACTIC

  #include "../global/global.h"
  #include "../model/disk_galaxy.h"
  #include "paris/PoissonZero3DBlockedGPU.hpp"

class PotentialParisGalactic
{
 public:
  PotentialParisGalactic();
  ~PotentialParisGalactic();

  /*!
   * \brief computes the dynamical gravitational potential for a galaxy. (Any analytic,
   * static component is handled separately
   *
   * \param[in] density Specifies the dynamic density field sourcing the gravity (i.e.
   * it includes density from gas and particles
   */
  void Get_Potential(const Real *density, Real *potential, Real grav_const, const DiskGalaxy &galaxy);
  void Initialize(Real lx, Real ly, Real lz, Real xMin, Real yMin, Real zMin, int nx, int ny, int nz, int nxReal,
                  int nyReal, int nzReal, Real dx, Real dy, Real dz);
  void Reset();

 protected:
  int dn_[3];
  Real dr_[3], lo_[3], lr_[3], myLo_[3];

  /* Pointer to the object that does the hard part of solving Poisson's equation */
  PoissonZero3DBlockedGPU *pp_;

  /* minimum sizes of the buffers, passed to pp_, in order to be large enough to
   * hold values at all spatial locations */
  long densityBytes_;
  /* minimum sizes of the buffers, passed to pp_, in order to be large enough to
   * hold intermediate calculated values. */
  long minBytes_;

  /* device buffer used to hold the values corresponding to the RHS of poisson's equation,
   * `4 * pi * G / scale_factor * ( dens - dens_bkg )`. */
  Real *da_;  // a better name might be poisson_rhs_

  /* device buffer used to store the the gravitational potential computed from da_ (it does
   * NOT include phi contributions from dens_bkg) */
  Real *db_;  // a better name might be phi_from_poisson_

  #ifndef GRAVITY_GPU
  /* length of the temporary buffer represented by dc_ (in bytes) */
  long potentialBytes_;
  /* when the GRAVITY_GPU macro is undefined, this is a gpu-allocated temporary buffer used
   * to temporarily hold the combined gravitational potential from self-gravity and the
   * static potential */
  Real *dc_;  // a better name might be total_phi_
  #endif
};

#endif

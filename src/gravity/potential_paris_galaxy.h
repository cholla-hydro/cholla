#pragma once

#ifdef PARIS_GALAXY

#include "paris/PoissonZero3DBlockedGPU.hpp"
#include "../global/global.h"
#include "../model/disk_galaxy.h"

class Potential_Paris_Galaxy {
  public:
    Potential_Paris_Galaxy();
    ~Potential_Paris_Galaxy();
    void Get_Potential(const Real *density, Real *potential, Real g, const DiskGalaxy &galaxy);
    void Initialize(Real lx, Real ly, Real lz, Real xMin, Real yMin, Real zMin, int nx, int ny, int nz, int nxReal, int nyReal, int nzReal, Real dx, Real dy, Real dz);
    void Reset();
  protected:
    int dn_[3];
    Real dr_[3],lo_[3],lr_[3],myLo_[3];
    PoissonZero3DBlockedGPU *pp_;
    long densityBytes_;
    long minBytes_;
    Real *da_;
    Real *db_;
#ifndef GRAVITY_GPU
    long potentialBytes_;
    Real *dc_;
#endif
};

#endif

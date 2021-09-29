#pragma once

#if defined(GRAVITY) && defined(PARIS_BC)

#include "paris/PoissonZero3DBlockedGPU.hpp"
#include "../global/global.h"

class Potential_Paris_BC_3D {
  public:
    Potential_Paris_BC_3D();
    ~Potential_Paris_BC_3D();
    void Get_Potential(const Real *density, Real *potential, Real g, Real massInfo, Real a);
    void Initialize(Real lx, Real ly, Real lz, Real xMin, Real yMin, Real zMin, int nx, int ny, int nz, int nxReal, int nyReal, int nzReal, Real dx, Real dy, Real dz);
    void Reset();
  protected:
    int dn_[3];
    Real dr_[3],lo_[3],lr_[3],myLo_[3];
    PoissonZero3DBlockedGPU *pp_;
    long minBytes_;
    long densityBytes_;
    long potentialBytes_;
    Real *da_;
    Real *db_;
};

#endif

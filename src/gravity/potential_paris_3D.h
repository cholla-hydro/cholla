#pragma once

#if defined(GRAVITY) && defined(PARIS)

#include "paris/PoissonPeriodic3x1DBlockedGPU.hpp"
#include "paris/PoissonZero3DBlockedGPU.hpp"
#include "../global/global.h"

class Potential_Paris_3D {
  public:
    Potential_Paris_3D();
    ~Potential_Paris_3D();
    void Get_Analytic_Potential(const Real *density, Real *potential);
    void Get_Potential(const Real *density, Real *potential, Real g, Real massInfo, Real a);
    void Initialize(Real lx, Real ly, Real lz, Real xMin, Real yMin, Real zMin, int nx, int ny, int nz, int nxReal, int nyReal, int nzReal, Real dx, Real dy, Real dz, bool periodic);
    void Reset();
  protected:
    int dn_[3];
    Real dr_[3],lo_[3],lr_[3],myLo_[3];
    PoissonPeriodic3x1DBlockedGPU *pp_;
    PoissonZero3DBlockedGPU *pz_;
    long minBytes_;
    long densityBytes_;
    long potentialBytes_;
    Real *da_;
    Real *db_;
};

#endif

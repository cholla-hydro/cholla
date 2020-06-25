#pragma once

#if defined(GRAVITY) && defined(PARIS)

#include "paris/PoissonPeriodic3DBlockedGPU.hpp"
#include "../global.h"

class Potential_Paris_3D {
  public:
    Potential_Paris_3D();
    ~Potential_Paris_3D();
    void Get_Potential(const Real *density, Real *potential, Real g, Real avgDensity, Real a);
    void Initialize(Real lx, Real ly, Real lz, Real xMin, Real yMin, Real zMin, int nx, int ny, int nz, int nxReal, int nyReal, int nzReal, Real dx, Real dy, Real dz);
    void Reset();
  protected:
    int n_[3];
    PoissonPeriodic3DBlockedGPU *p_;
    long minBytes_;
    long densityBytes_;
    long potentialBytes_;
    double *da_;
    double *db_;
};

#endif

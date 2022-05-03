#if defined(PARTICLES) && defined(DE) && defined(PARTICLE_AGE)
#pragma once

#include "../global/global.h"

const int N_INFO = 5;

Real getClusterEnergyFeedback(Real t, Real dt, Real age);
Real getClusterMassFeedback(Real t, Real dt, Real age);
std::tuple<int, Real, Real, Real, Real> getClusterFeedback(Real t, Real dt, Real age, Real density);

#endif // PARTICLES et. al

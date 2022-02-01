#ifdef PARTICLES
#ifdef DE
#ifdef PARTICLE_AGE


#ifndef FEEDBACK_CIC_H
#define FEEDBACK_CIC_H
#include "../global/global.h"

const int N_INFO = 5;

Real getClusterEnergyFeedback(Real t, Real dt, Real age);
Real getClusterMassFeedback(Real t, Real dt, Real age);
std::tuple<int, Real, Real, Real, Real> getClusterFeedback(Real t, Real dt, Real age, Real density);

#endif
#endif
#endif
#endif

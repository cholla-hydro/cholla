#if defined(PARTICLES) && defined(DE) && defined(PARTICLE_AGE) && defined(PARTICLE_CPU) && defined(FEEDBACK)

#ifndef FEEDBACK_CIC_H
#define FEEDBACK_CIC_H

#define ENERGY_FEEDBACK_RATE 5.25958e-07  //  Rate is 1e51 erg/100M_solar spread out over 10Myr

Real getClusterEnergyFeedback(Real t, Real dt, Real age);
Real getClusterMassFeedback(Real t, Real dt, Real age);

#endif
#endif
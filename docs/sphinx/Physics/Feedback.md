# Particle-based Feedback

Models supernova (SN) feedback from star cluster particles as a Poisson processes following the prescription in [Kim & Ostriker (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...815...67K/abstract). Energy or momentum is injected into the interstellar medium depending on whether the SN is sufficiently numerically resolved.

## Required Compilation Flags
* `SUPERNOVA` - Supernova rate (SNR) information from a Starburst99 generated *.snr file can be read in by specifying the path as the value of the `snr_filename` parameter.  If this parameter is not set, then a default constant SNR is used.  The default SNR corresponds to 1 supernova per {math}`100~\mathrm{M}_\odot` of cluster mass, spread out over 36 Myr, starting when the cluster is 4 Myr old.  A sample Starburst99 file is included in the source code at `src/particles/starburst99_snr.txt`.  The sample represents a {math}`10^6~\mathrm{M}_\odot` fixed mass cluster, created using a Kroupa initial mass function, and with an {math}`8~\mathrm{M}_\odot` supernova cutoff.
* `PARTICLES_GPU` - Particle-based feedback requires that the particle data be on the GPUs.
* `PARTICLE_AGE` - Feedback varies with cluster age.
* `PARTICLE_IDS` - Cluster IDs are used to prevent possible correlations or biases when generating random numbers used by the feedback algorithm.
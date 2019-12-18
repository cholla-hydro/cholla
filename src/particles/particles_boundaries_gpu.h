#if defined(PARTICLES) && defined(PARTICLES_GPU)

#ifndef PARTICLES_BOUNDARIES_H
#define PARTICLES_BOUNDARIES_H

int Select_Particles_to_Transfer_GPU_function( part_int_t n_local, int side, Real domainMin, Real domainMax, Real *pos,  int *n_transfer_d, int *n_transfer_h, bool *transfer_flags, int *transfer_indxs, int *transfer_partial_sum, int *transfer_sum  );


#endif //PARTICLES_H
#endif //PARTICLES
/*! \file subgrid_routines_2D.h
 *  \brief Declarations of the subgrid staging functions for 2D CTU. */

#ifdef CUDA
#ifndef SUBGRID_ROUTINES_2D_H
#define SUBGRID_ROUTINES_2D_H

void sub_dimensions_2D(int nx, int ny, int n_ghost, int *nx_s, int *ny_s, int *block1_tot, int *block2_tot, int *remainder1, int *remainder2, int n_fields);


void allocate_buffers_2D(int block1_tot, int block2_tot, int BLOCK_VOL, Real **&buffer, int n_fields);


void host_copy_init_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int remainder1, int BLOCK_VOL, Real *host_conserved, Real **buffer, Real **tmp1, Real **tmp2, int n_fields);


void host_copy_next_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real **buffer, Real **tmp1, int n_fields);


void host_return_values_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real **buffer, int n_fields);


void free_buffers_2D(int nx, int ny, int nx_s, int ny_s, int block1_tot, int block2_tot, Real **buffer);

#endif //SUBGRID_ROUTINES_2D_H
#endif //CUDA

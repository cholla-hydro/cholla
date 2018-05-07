/*! \file subgrid_routines_2D.h
 *  \brief Declarations of the subgrid staging functions for 2D CTU. */

#ifdef CUDA
#ifndef SUBGRID_ROUTINES_2D_H
#define SUBGRID_ROUTINES_2D_H

void sub_dimensions_2D(int nx, int ny, int n_ghost, int *nx_s, int *ny_s, int *block1_tot, int *block2_tot, int *remainder1, int *remainder2, int n_fields);


void get_offsets_2D(int nx_s, int ny_s, int n_ghost, int x_off, int y_off, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int *x_off_s, int *y_off_s);


// copy the conserved variable block into the buffer
void host_copy_block_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real *buffer, int n_fields);


// return the values from buffer to the host_conserved array
void host_return_block_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real *buffer, int n_fields);


#endif //SUBGRID_ROUTINES_2D_H
#endif //CUDA

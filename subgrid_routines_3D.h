/*! \file subgrid_routines_3D.h
 *  \brief Declarations of the subgrid staging functions for 3D CTU. */

#ifdef CUDA
#ifndef SUBGRID_ROUTINES_3D_H
#define SUBGRID_ROUTINES_3D_H


void sub_dimensions_3D(int nx, int ny, int nz, int n_ghost, int *nx_s, int *ny_s, int *nz_s, int *block1_tot, int *block2_tot, int *block3_tot, int *remainder1, int *remainder2, int *remainder3);


void allocate_buffers_3D(int block1_tot, int block2_tot, int block3_tot, int BLOCK_VOL, Real **&buffer);


void host_copy_init_3D(int nx, int ny, int nz, int nx_s, int ny_s, int nz_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real **buffer, Real **tmp1, Real **tmp2);


void host_copy_next_3D(int nx, int ny, int nz, int nx_s, int ny_s, int nz_s, int n_ghost, int block, int block1_tot, int block2_tot, int block3_tot, int remainder1, int remainder2, int remainder3, int BLOCK_VOL, Real *host_conserved, Real **buffer, Real **tmp1);


void host_return_values_3D(int nx, int ny, int nz, int nx_s, int ny_s, int nz_s, int n_ghost, int block, int block1_tot, int block2_tot, int block3_tot, int remainder1, int remainder2, int remainder3, int BLOCK_VOL, Real *host_conserved, Real **buffer);


void free_buffers_3D(int nx, int ny, int nz, int nx_s, int ny_s, int nz_s, int block1_tot, int block2_tot, int block3_tot, Real **buffer);



#endif //SUBGRID_ROUTINES_3D_H
#endif //CUDA

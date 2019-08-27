/*! \file global_cuda.cu
 *  \brief Declarations of the cuda global variables. */

#ifdef CUDA

#include"global.h"

// Declare global variables
bool memory_allocated, block_size;
Real *dev_conserved, *dev_conserved_half;
Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;
Real *eta_x, *eta_y, *eta_z, *etah_x, *etah_y, *etah_z;
Real *dev_dti_array;
#ifdef COOLING_GPU
Real *dev_dt_array;
#endif
Real *host_dti_array;
#ifdef COOLING_GPU
Real *host_dt_array;
#endif
#if defined( GRAVITY ) && defined( GRAVITY_COUPLE_GPU )
Real *dev_grav_potential;
#endif
Real *buffer, *tmp1, *tmp2;
int nx_s, ny_s, nz_s;
int x_off_s, y_off_s, z_off_s;
int block1_tot, block2_tot, block3_tot, block_tot;
int remainder1, remainder2, remainder3;
int BLOCK_VOL;
int ngrid;


#endif //CUDA

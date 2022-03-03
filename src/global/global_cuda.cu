/*! \file global_cuda.cu
 *  \brief Declarations of the cuda global variables. */

#ifdef CUDA

#include "../global/global.h"

// Declare global variables
bool dt_memory_allocated, memory_allocated;
Real *dev_conserved, *dev_conserved_half;
Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;
Real *eta_x, *eta_y, *eta_z, *etah_x, *etah_y, *etah_z;
#ifdef COOLING_GPU
Real *dev_dt_array;
#endif
#ifdef COOLING_GPU
Real *host_dt_array;
#endif
Real *buffer, *tmp1, *tmp2;
int nx_s, ny_s, nz_s;
int x_off_s, y_off_s, z_off_s;
int block1_tot, block2_tot, block3_tot, block_tot;
int remainder1, remainder2, remainder3;
int BLOCK_VOL;
int ngrid;

//Arrays for potential in GPU: Will be set to NULL if not using GRAVITY
Real *dev_grav_potential;
Real *temp_potential;
Real *buffer_potential;

#endif //CUDA

/*! \file global_cuda.cu
 *  \brief Declarations of the cuda global variables. */

#ifdef CUDA

#include "../global/global.h"

// Declare global variables
Real *dev_conserved, *dev_conserved_half;
Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;
Real *eta_x, *eta_y, *eta_z, *etah_x, *etah_y, *etah_z;
Real *host_dti_array;
Real *dev_dti_array;
int ngrid;

//Arrays for potential in GPU: Will be set to NULL if not using GRAVITY
Real *dev_grav_potential;
Real *temp_potential;
Real *buffer_potential;

#endif //CUDA

/*! \file CTU_3D_cuda.h
 *  \brief Declarations for the cuda version of the 3D CTU algorithm. */

#ifdef CUDA

#ifndef ERROR_CHECK_CUDA_H
#define ERROR_CHECK_CUDA_H

#include"global.h"


#define N_Z 24
#define N_Y 24


int Check_Field_Along_Axis( Real *dev_array, int n_field, int nx, int ny, int nz, int n_ghost, dim3 Grid_Error, dim3 Block_Error );

__global__ void Check_Value_Along_Axis( Real *dev_array, int n_field, int nx, int ny, int nz, int n_ghost, int *return_value);



#endif //ERROR_CHECK_CUDA_H
#endif //CUDA

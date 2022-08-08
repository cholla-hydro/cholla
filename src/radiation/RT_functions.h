/*! \file RT_functions.h
 *  \brief Declarations for the gpu RT functions. */

#ifdef CUDA

#ifndef RT_FUNCTIONS_H
#define RT_FUNCTIONS_H

#include "../global/global.h"
#include "radiation.h"

void rtSolve(Real *dev_scalar);

void rtBoundaries(Real *dev_scalar, struct Rad3D::RT_Fields &rtFields);

#endif //VL_3D_CUDA_H
#endif //CUDA

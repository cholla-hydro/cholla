/*! \file gravity.h
 *  \brief Declaration of functions for the gravity source corrections.*/

 #ifdef CUDA

#ifndef GRAVITY_H 
#define GRAVITY_H 

#include"global.h"



/*! \fn Correct_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_Q_Ly, Real *dev_Q_Ry, int nx, int ny, Real dt)
 *  \brief Correct the input states to the Riemann solver based on gravitational source terms. */
__global__ void Correct_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_Q_Ly, Real *dev_Q_Ry, int nx, int ny, Real dt);


#endif // GRAVITY_H
#endif // CUDA

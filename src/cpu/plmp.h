/*! \file plmp.h
 *  \brief Declarations of the piecewise linear method. */
#ifndef CUDA
#ifdef PLMP

#ifndef PLMP_H
#define PLMP_H

#include "../global/global.h"

/*! \fn plmp(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
*  \brief Use a piece-wise linear method to calculate boundary values for each cell. */
void plmp(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma);



#endif //PLMP_H
#endif //PLMP
#endif //CUDA

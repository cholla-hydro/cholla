/*! \file ppmc.h
 *  \brief Declarations of the piecewise parabolic method as described in Stone et al. 2008. */
#ifndef CUDA
#ifdef PPMC

#ifndef PPMC_H
#define PPMC_H

#include "../global/global.h"


/*! \fn void ppmc(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
 *  \brief When passed a stencil of conserved variables, returns right and left 
           boundary values for the interfaces calculated using ppm. */
void ppmc(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma);



#endif //PPMC_H
#endif //PPMC
#endif //CUDA

/*! \file plmc.h
 *  \brief Declarations of the piecewise linear method, as described in Stone et al. 2008 */
#ifdef PLMC

#ifndef PLMC_H 
#define PLMC_H 

#include "../global/global.h"

/*! \fn plmc(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma)
*  \brief Use a piece-wise linear method to calculate boundary values for each cell. */
void plmc(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma);



#endif //PLMC_H
#endif //PLMC

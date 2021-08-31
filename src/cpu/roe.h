/*! \file roe.h
 *  \brief Header file for the Roe Riemann solver. */

#ifndef ROE_H
#define ROE_H

#include"global.h"


/* \fn Calculate_Roe_Fluxes(Real cW[], Real fluxes[], Real gamma, Real etah)
 * \brief Returns the density, momentum, and Energy fluxes at an interface.
   Inputs are an array containing left and right density, momentum, and Energy. */
void Calculate_Roe_Fluxes(Real cW[], Real fluxes[], Real gamma, Real etah);



#endif // ROE_H


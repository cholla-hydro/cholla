/*! \file hllc.h
 *  \brief Function declaration for the HLLC Riemann solver.*/

#ifndef HLLC_H
#define HLLC_H

#include "../global/global.h"


/*! \fn Calculate_HLLC_Fluxes(Real cW[], Real fluxes[], Real gamma, Real etah)
 *  \brief HLLC Riemann solver based on the version described in Toro (2006), Sec. 10.4. */
void Calculate_HLLC_Fluxes(Real cW[], Real fluxes[], Real gamma, Real etah);

#endif //HLLC_H

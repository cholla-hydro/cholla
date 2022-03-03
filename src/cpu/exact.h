/*! \file exact.h
 *  \brief Header file for Toro exact Riemann solver. */

#ifndef EXACT_H
#define EXACT_H

#include <stdio.h>
#include <math.h>
#include "../global/global.h"


/* Function declarations */

/*! \fn void Calculate_Exact_Fluxes(Real input[], Real fluxes[], Real gamma)
 *  \brief Given an array of the density, velocity, and pressure
	  on both sides of an interface, returns a vector with the
	  density flux, momentum flux, and total Energy flux. */
void Calculate_Exact_Fluxes(Real input[], Real fluxes[], Real gamma);

/*! \fn void guessp(Real *pm)
 *  \brief Provides a guessed value for pressure pm in the star region. */
Real guessp(Real dl, Real ul, Real pl, Real cl, Real dr, Real ur, Real pr, Real cr, Real gamma);

/*! \fn void prefun(Real *f, Real *fd, Real *p, Real *dk, Real *pk, Real *ck)
 *  \brief Evaluates the pressure functions fl and fr and their first derivatives. */
void prefun(Real *f, Real *fd, Real *p, Real *dk, Real *pk, Real *ck, Real gamma);

/*! \fn void starpu(Real *p, Real *u, const Real pscale)
 *  \brief Computes the solution for pressure and velocity in the star region. */
void starpu(Real *p, Real *u, Real dl, Real ul, Real pl, Real cl,
      Real dr, Real ur, Real pr, Real cr, Real gamma);

/*! \fn void sample(const Real pm, const Real um,
	    Real *d, Real *u, Real *p)
 *  \brief Samples the solution at the cell interface. */
void sample(const Real pm, const Real vm,
	    Real *d, Real *u, Real *p,
      Real dl, Real ul, Real pl, Real cl,
      Real dr, Real ur, Real pr, Real cr, Real gamma);



#endif //EXACT_H

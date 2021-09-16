/*! \file ppmp.h
 *  \brief Declarations of the piecewise parabolic method. */
#ifndef CUDA
#ifdef PPMP

#ifndef PPMP_H
#define PPMP_H

#include "../global/global.h"

/*! \fn void ppmp(Real stencil[], Real bounds[], Real dx, Real dt)
 *  \brief When passed a stencil of primative variables, returns the left and right
           boundary values for the interface calculated using ppm. */
void ppmp(Real stencil[], Real bounds[], Real dx, Real dt, Real gamma);



/*! \fn interface_value
 *  \brief Returns the right hand interpolated value at imo | i cell interface.*/
Real interface_value(Real q_imo, Real q_i, Real q_ipo, Real q_ipt,
	               Real dximo, Real dx_i, Real dx_ipo, Real dx_ipt);

/*! \fn calc_delta_q
 *  \brief Returns the average slope in zone i of the parabola with zone averages
	   of imo, i, and ipo. See Fryxell Eqn 24. */
Real calc_delta_q(Real q_imo, Real q_i, Real q_ipo,
	              Real dx_imo, Real dx_i, Real dx_ipo);


/*! \fn limit_delta_q
 *  \brief Limits the value of delta_rho according to Fryxell Eqn 26
	   to ensure monotonic interface values. */
Real limit_delta_q(Real del_in, Real q_imo, Real q_i, Real q_ipo);


/*! \fn test_interface_value
 *  \brief Returns the right hand interpolated value at imo | i cell interface,
	   assuming equal cell widths. */
Real test_interface_value(Real q_imo, Real q_i, Real q_ipo, Real q_ipt);


/*! \fn calc_d2_rho
 *  \brief Returns the second derivative of rho across zone i. (Fryxell Eqn 35) */
Real calc_d2_rho(Real rho_imo, Real rho_i, Real rho_ipo,
		Real dx_imo, Real dx_i, Real dx_ipo);


/*! \fn calc_eta
 *  \brief Returns a dimensionless quantity relating the 1st and 3rd derivatives
		See Fryxell Eqn 36. */
Real calc_eta(Real d2rho_imo, Real d2rho_ipo, Real dx_imo, Real dx_i, Real dx_ipo,
		Real rho_imo, Real rho_ipo);



#endif //PPMP_H
#endif //PPMP
#endif //CUDA

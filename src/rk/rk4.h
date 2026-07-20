#pragma once

/*! \file
 *  \brief Defines \ref RKIntegrator
 */

#include <vector>

#include "../global/global.h"

/*! \brief Class for evolving coupled
 *   differential equations using the
 *   Runga-Kutta method. */
class RKIntegrator
{
 public:
  /*! \brief Order of RK integrator */
  static constexpr int nrk  = 7;
  static constexpr int cols = 6;
  int ny;

  std::vector<Real> ai  = {0, 0, 0.2, 0.3, 0.6, 1.0, 0.875};
  std::vector<Real> ci  = {0, 37. / 378., 0, 250. / 621., 125. / 594., 0, 512. / 1771.};
  std::vector<Real> csi = {0, 2825. / 27648, 0, 18575. / 48384, 13525. / 55296., 277. / 14336., 0.25};
  std::vector<std::vector<Real>> bij{7, std::vector<Real>(6, 0)};
  std::vector<Real> yi;
  std::vector<Real> yprime;
  std::vector<Real> error;

  /*! \fn void InitializeRK(int ny_in)
   *  \brief Initialize Runga-Kutta integrator */
  void InitializeRK(int ny_in);

  /*! \fn void FreeMemory(void)
   *  \brief Free RK integrator memory */
  void FreeMemory(void);

  /*! \brief Evolve the ODE system using the RK method.
  *    dydx() returns the derivatives of y w.r.t. x.
  *    h is a pointer to the current stepsize that actually
  *    was executed and hpass is a pointer to the next stepsize
  *    to take in the integration.  The updated values of the
  *    y parameters are returned in yp. The errors associated
  *    with each integration variable are returned in error. 
  *    The params vector allows for other parameters required
  *    for the integrator to be provided. None of the arguments
  *    should be a null pointer. The sizes of the arrays are set
  *    by the intialization routine InitializeRK().*/
  void rk4_ode(std::vector<Real>(dydx)(Real x, std::vector<Real> y, std::vector<Real> params), Real x,
               std::vector<Real> y, Real *h, Real *hpass, std::vector<Real> params, std::vector<Real> &yp, Real *error);
};

/*! \file
 *  \brief Defines \ref RKIntegrator
 */

#pragma once

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

  Real ai[nrk]  = {0, 0, 0.2, 0.3, 0.6, 1.0, 0.875};
  Real ci[nrk]  = {0, 37. / 378., 0, 250. / 621., 125. / 594., 0, 512. / 1771.};
  Real csi[nrk] = {0, 2825. / 27648, 0, 18575. / 48384, 13525. / 55296., 277. / 14336., 0.25};
  Real bij[nrk][cols];
  std::vector<Real> yi;
  std::vector<Real> yprime;
  std::vector<Real> error;

  RKIntegrator() = delete;  // ensure that instances are always fully initialized

  /*! \brief Constructor */
  explicit RKIntegrator(int ny_in);

  /*! \brief Evolve the ODE system using the RK method.
   *    dydx() returns the derivatives of y w.r.t. x.
   *    h is a pointer to the current stepsize that actually
   *    was executed and hpass is a pointer to the next stepsize
   *    to take in the integration.  The updated values of the
   *    y parameters are returned in yp. The errors associated
   *    with each integration variable are returned in error.
   *    The params vector allows for other parameters required
   *    for the integrator to be provided. None of the arguments
   *    should be a null pointer. .*/
  void rk4_ode(std::vector<Real>(dydx)(Real x, const std::vector<Real>& y, const std::vector<Real>& params), Real x,
               const std::vector<Real>& y, Real *h, Real *hpass, const std::vector<Real>& params, std::vector<Real> &yp, Real *error);
};

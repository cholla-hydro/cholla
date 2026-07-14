#pragma once

#ifndef RK_H
  #define RK_H
  #include <vector>

  #include "../global/global.h"

/*! \class RKIntegrator
 *  \brief Class for evolving coupled
 *   differential equations using the
 *   Runga-Kutta method. */
class RKIntegrator
{
 public:
  /*! \var int nrk
   *  \brief Order of RK integrator */
  const int nrk  = 7;
  const int cols = 6;
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

  /*! \fn rk4_ode(std::vector<Real> (dydx)(Real x, std::vector<Real> y, std::vector<Real> params), Real x,
   * std::vector<Real> y, Real *h, Real *hpass, std::vector<Real> params, std::vector<Real> &yp, Real *error, int *recdepth) \brief
   * Evolve the ODE system using the RK method */
  void rk4_ode(std::vector<Real>(dydx)(Real x, std::vector<Real> y, std::vector<Real> params), Real x,
               std::vector<Real> y, Real *h, Real *hpass, std::vector<Real> params, std::vector<Real> &yp, Real *error, int *recdepth);
};
#endif  // RK_H

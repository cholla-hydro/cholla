#include "rk4.h"

#include <stdio.h>

#include <cmath>
#include <vector>

#include "../utils/error_handling.h"

// Initialize the RK integrator
RKIntegrator::RKIntegrator(int ny_in)
{
  // integer
  ny = ny_in;

  // resize the variables
  yi.resize(ny);
  yprime.resize(ny);
  error.resize(ny);

  int i, j;
  for (i = 0; i < 7; i++)
    for (j = 0; j < 6; j++) bij[i][j] = 0;

  // initialize bij
  bij[2][1] = 0.2;
  bij[3][1] = 3. / 40.;
  bij[3][2] = 9. / 40.;
  bij[4][1] = 0.3;
  bij[4][2] = -0.9;
  bij[4][3] = 1.2;
  bij[5][1] = -11. / 54.;
  bij[5][2] = 5. / 2.;
  bij[5][3] = -70. / 27.;
  bij[5][4] = 35. / 27.;
  bij[6][1] = 1631. / 55296.;
  bij[6][2] = 175. / 512.;
  bij[6][3] = 575. / 13824.;
  bij[6][4] = 44275. / 110592.;
  bij[6][5] = 253. / 4096.;
}

void RKIntegrator::rk4_ode(std::vector<Real> (*dydx)(Real x, const std::vector<Real>& y, const std::vector<Real>& params), Real x,
                           const std::vector<Real>& y, Real *h_this, Real *h_pass, const std::vector<Real>& params,
                           std::vector<Real> &yp, Real *error_pass)
{
  Real Safety    = 0.9;
  Real error_tol = 1.0e-5;  // absolute error
  Real error;
  Real max_error = 0;
  Real error_factor;
  Real h;
  int max_iters = 20;
  int iters     = 0;
  bool flag     = true;

  int ny = y.size();

  std::vector<std::vector<Real>> kij{static_cast<unsigned long>(nrk), std::vector<Real>(ny, 0)};

  std::vector<Real> yy;

  while (flag) {
    // set the current step
    h = *h_this;

    for (int i = 1; i < nrk; i++) {
      for (int k = 0; k < ny; k++) {
        yi[k] = y[k];
      }
      for (int j = 1; j < i; j++) {
        for (int k = 0; k < ny; k++) {
          yi[k] += h * bij[i][j] * kij[j][k];
        }
      }
      kij[i] = dydx(x + ai[i] * h, yi, params);
    }

    for (int k = 0; k < ny; k++) {
      yp[k]     = y[k];
      yprime[k] = y[k];
      for (int i = 1; i < nrk; i++) {
        yp[k] += h * ci[i] * kij[i][k];
        yprime[k] += h * csi[i] * kij[i][k];
      }
    }

    for (int k = 0; k < ny; k++) {
      error = (yp[k] - yprime[k]);

      if ((fabs(y[k]) > 0) & (fabs(yp[k] - y[k]) / fabs(y[k]) > 0.01)) error = 0.1;

      if (fabs(error) > fabs(max_error)) max_error = error;

      *error_pass = max_error;
    }

    if (fabs(max_error) > error_tol) {
      // decrease h
      error_factor = Safety * pow(fabs(max_error / error_tol), -0.25);
      if (error_factor < 0.1) error_factor = 0.1;

      *h_pass = h * error_factor;
      *h_this = h * error_factor;

      // limit the number of iterations
      iters++;  // increment the number of iterations
      if (iters >= max_iters) {
        printf("RKIntegrator: procID %d: Max Number of Iterations Exceeded (%d)!", procID, max_iters);
        chexit(0);
      }

    } else {
      flag = false;

      // increase h
      if (fabs(max_error) > 0) {
        error_factor = Safety * pow(fabs(max_error / error_tol), -0.20);
      } else {
        error_factor = 5.0;
      }

      // limit to a factor of 5
      if (error_factor > 5.0) error_factor = 5.0;

      // step size cannot go down
      if (error_factor < 1.0) error_factor = 1.0;
      *h_pass = h * error_factor;
    }
  }
}

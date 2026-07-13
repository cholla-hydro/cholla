#include "rk4.h"

#include <stdio.h>

#include <cmath>
#include <vector>

using namespace std;

// Initialize the RK integrator
void RKIntegrator::InitializeRK(int ny_in)
{
  // integer
  ny = ny_in;

  // resize the variables
  yi.resize(ny);
  yprime.resize(ny);
  error.resize(ny);

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

// Free RK integrator memory
void RKIntegrator::FreeMemory(void)
{
  ai.clear();
  ci.clear();
  csi.clear();
  bij.clear();
  yi.clear();
  yprime.clear();
  error.clear();

  ai.shrink_to_fit();
  ci.shrink_to_fit();
  csi.shrink_to_fit();
  bij.shrink_to_fit();
  yi.shrink_to_fit();
  yprime.shrink_to_fit();
  error.shrink_to_fit();
}

/*! \fn void rk4_ode(Real* (*dydx)(Real x, Real *y, int iy, void *params, int np), Real x, Real *y, Real *h, Real
 * *hpass, void *params, int np, Real *yp, int iy, Real *error) \brief Evolve the ODE system one time step using the RK
 * method */
// void RKIntegrator::rk4_ode( Real* (*dydx) (Real x, Real *y, int ny, void *params, int np), Real x, Real *y, Real
// *h_this, Real *h_pass, void *params, int np, Real *yp, int ny, Real *error_pass)
void RKIntegrator::rk4_ode(std::vector<Real> (*dydx)(Real x, std::vector<Real> y, std::vector<Real> params), Real x,
                            std::vector<Real> y, Real *h_this, Real *h_pass, std::vector<Real> params,
                            std::vector<Real> &yp, Real *error_pass)
{
  Real Safety    = 0.9;
  Real error_tol = 1.0e-5;  // absolute error
  Real error;
  Real max_error = 0;
  Real error_factor;
  Real h = *h_this;

  int ny = y.size();

  std::vector<std::vector<Real>> kij{static_cast<unsigned long>(nrk), std::vector<Real>(ny, 0)};

  std::vector<Real> yy;

  for (int i = 1; i < nrk; i++) {
    for (int k = 0; k < ny; k++) yi[k] = y[k];
    for (int j = 1; j < i; j++)
      for (int k = 0; k < ny; k++) yi[k] += h * bij[i][j] * kij[j][k];
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

    // redo step
    rk4_ode(dydx, x, y, h_this, h_pass, params, yp, error_pass);

  } else {
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

/*
Real OmegaDEz(Real z, Real Omega_DE, Real w0, Real wa)
{
        Real A = pow(1+z,3*(1+w0+wa));
        Real B = Omega_DE * exp(-3*wa*z/(1+z));
        return A*B;
}

Real Hubble(Real a, Real H0, Real Omega_r, Real Omega_m, Real Omega_DE, Real w0, Real wa)
{
        //set redshifts, limit scale factor to 1.0e-6 minimum
        Real aa = a;
        if(aa<1.0e-6)
                aa = 1.0e-6;
        Real z = 1./aa -1.;
  return H0*sqrt( Omega_r*pow(1+z,4) + Omega_m*pow(1+z,3) + OmegaDEz(z,Omega_DE,w0,wa) );
}


std::vector<Real> growth_factor_system(Real z, std::vector<Real> y, std::vector<Real> params)
{

        int ny = y.size();
        std::vector<Real> dydz(ny);

        Real aa;
        Real a = y[0];
        Real delta = y[1];
        Real delta_dot = y[2];
        Real da_dt, d2delta_dt2;

        Real H0 = params[0];
        Real Omega_r = params[1];
        Real Omega_m = params[2];
        Real Omega_DE = params[3];
        Real w0 = params[4];
        Real wa = params[5];

        aa=a;
        if(aa<1.0e-7)
                aa = 1.0e-7;

  // get current hubble parameter at this
  // scale factor and time
  Real H = Hubble(aa, H0, Omega_r, Omega_m, Omega_DE, w0, wa);

  // get the current redshift
  z = 1./aa -1.;

  // Get the current fraction of the critical
  // density contributed by matter and DE
  Real Omega_r_z = Omega_r * pow(1+z,4);
  Real Omega_m_z = Omega_m * pow(1+z,3);
  Real Omega_DE_z = OmegaDEz(z, Omega_DE, w0, wa);
  Real Omega_tot = Omega_m_z + Omega_DE_z  + Omega_r_z;

  // get the current da/dt = H*a
  da_dt = H * a;

  // get the current d^2 delta/dt^2 = -2 H ddelta/dt + 4\piG\rho_0 \delta
  // \rho_0 = 3 \Omega_m(z)/\Omega_tot H^2 / 8 \pi G
  // so the second term is 1.5*(Omega_m_z/Omega_tot)*(H**2)*delta
  d2delta_dt2 = -2*H*delta_dot + 1.5*(Omega_m_z/Omega_tot)*(H*H)*delta;

  dydz[0] = da_dt;
  dydz[1] = delta_dot;
  dydz[2] = d2delta_dt2;
  return dydz;
}

int main(int argc, char **argv)
{
        Real dz_new;
        int np = 6;
        //Real *params = (Real *) malloc(np*sizeof(Real));
        int ny = 3;
        //Real *yp = (Real *) malloc(ny*sizeof(Real));

        std::vector<Real> params(np);

        Real error;
        //Real *y_igm = (Real *) malloc(ny*sizeof(Real));
        std::vector<Real> y_igm(ny,0);
        std::vector<Real> yp (ny,0);

        Real z = 10.0;
        Real dz = -1.0e-3;

        RKIntegrator RK;

        RK.InitializeRK(3);

        std::vector<Real> y;

        std::vector<Real> z_array;
        std::vector<Real> ya_array;
        std::vector<Real> yb_array;
        std::vector<Real> yc_array;

        //evolve the ode by one step
        params[0] = dz;
        params[1] = 1.;

        //planck 2018
        Real H0 = 67.32117;
        params[0] = H0;
        params[1] = 9.231186e-5;
        params[2] = 0.3144;
        params[3] = 0.685508;
        //params[4] = -1.0;
        //params[5] = 0;
        params[4] = -0.8;
        params[5] = -0.5;

//	RK.rk4_ode( dydz_igm, z , &y_igm[0], &dz, &dz_new, params, np, yp, ny, &error);
//	RK.rk4_ode( dydz_test, z , &y_igm[0], &dz, &dz_new, params, np, yp, ny, &error);
        //RK.rk4_ode( dydz_test, z , y_igm, &dz, &dz_new, params, np, yp, ny, &error);

        Real dz_max = -1.0e-2;

        // initial scale factor
        y_igm[0] = 1.0e-7;
        y_igm[1] = 1.0e-8;
        y_igm[2] = 1.0e-8;

        Real t = 0;

        z_array.push_back(t);
        ya_array.push_back(y_igm[0]);
        yb_array.push_back(y_igm[1]);
        yc_array.push_back(y_igm[2]);
        Real tmax = 1./H0;

        Real dt = 1.0e-4 * tmax;
        Real dt_new;
        Real dt_max = 1.0e-2 * tmax;

        Real a_max = 1.0;
        while( (t<tmax)&(y_igm[0]<a_max) )
        {
                if(t+dt>tmax)
                {
                        dt = tmax-t;
                }

                //RK.rk4_ode( dydz_test, z , y_igm, &dz, &dz_new, params, yp, &error);
                //RK.rk4_ode( dydz_test, z , y_igm, &dz, &dz_new, params, yp, &error);
                RK.rk4_ode( growth_factor_system, t , y_igm, &dt, &dt_new, params, yp, &error);

                //printf("z %e dz %e y_igm %e %e %e yp %e %e %e\n",z,dz,y_igm[0],y_igm[1],y_igm[2],yp[0],yp[1],yp[2]);

                t += dt;

                for(int i=0;i<yp.size();i++)
                        y_igm[i] = yp[i];

                // limit to the largest dz allowable
                if(dt_new<dt_max)
                        dt_new = dt_max;
                // update the redshift step
                dt = dt_new;

                z_array.push_back(t);
                ya_array.push_back(y_igm[0]);
                yb_array.push_back(y_igm[1]);
                yc_array.push_back(y_igm[2]);
        }


        int i;

        for(i=0;i<RK.ai.size();i++)
        {
                printf("ai[%d] %e\n",i,RK.ai[i]);
        }
        for(i=0;i<z_array.size();i++)
                printf("%e\t%e\t%e\t%e\n",z_array[i],ya_array[i],yb_array[i],yc_array[i]);

        RK.FreeMemory();

        return 0;
}
*/

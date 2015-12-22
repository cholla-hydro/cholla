/*! \file cooling_cpu.cpp
 *  \brief Functions to calculate cooling rate for a given rho, P, dt. */

#ifdef COOLING_CPU

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "global.h"
#include "grid3D.h"
#include "error_handling.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

Real test_cool(const Real n, const Real T);
Real Cloudy_cool(const Real n, const Real T);


void Grid3D::Cool_CPU(void)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real d, vx, vy, vz, E, P, n;
  Real T, T_init, T_init_P, del_T, dt_left, dt_sub, cool, E_old;
  #ifdef DE
  Real ge, T_init_ge;
  #endif

  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny-H.n_ghost;
  }
  else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;

        // load values of density and pressure
        d  =  C.density[id];
        vx =  C.momentum_x[id] / d;
        vy =  C.momentum_y[id] / d;
        vz =  C.momentum_z[id] / d;
        E  =  C.Energy[id];
        P  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gama - 1.0);
        #ifdef DE
        ge = C.GasEnergy[id] / d;
        P  = d * ge * (gama - 1.0);
        #endif
        P  = fmax(P, (Real) TINY_NUMBER);
    
        // calculate the number density of the gas (in cgs)
        n = d*DENSITY_UNIT / MP;

        // calculate the temperature of the gas
        T_init_P = P*PRESSURE_UNIT/ (n*KB);
        T_init = T_init_P;
        #ifdef DE
        T_init_ge = ge*(gamma-1.0)*SP_ENERGY_UNIT*MP/KB;
        T_init = T_init_ge;
        #endif

        dt_left = H.dt;

        // only allow cooling between 10^1 K and 10^9 K
        //if (T_init > 1e1 && T_init < 1e9) {

          // calculate cooling rate per volume
          T = T_init;
          cool = Cloudy_cool(n, T); 

          // calculate change in temperature given dt
          del_T = cool*dt_left*TIME_UNIT*(gama-1.0)/(n*KB);

          // limit change in temperature to 5%
          while (del_T/T > 0.05) {

            // what dt gives del_T = 0.1*T?
            dt_sub = 0.05*T*n*KB/(cool*TIME_UNIT*(gama-1.0));
            // apply that dt
            T -= cool*dt_sub*TIME_UNIT*(gama-1.0)/(n*KB);
            // how much time is left from the original timestep?
            dt_left -= dt_sub;
            // calculate cooling again
            cool = Cloudy_cool(n, T);
            // calculate new change in temperature
            del_T = cool*dt_left*TIME_UNIT*(gama-1.0)/(n*KB);

          }

          // calculate final temperature
          T -= del_T;

          // adjust value of energy based on total change in temperature
          del_T = T_init - T; // total change in T
          E_old = E;
          E -= n*KB*del_T / ((gama-1.0)*ENERGY_UNIT);
          if (E < 0.0) printf("%3d %3d %3d Negative E after cooling. %f %f %f %f %f\n", i, j, k, del_T, T_init, E_old, n, E);
          #ifdef DE
          ge -= KB*del_T / (MP*(gamma-1.0)*SP_ENERGY_UNIT);
          #endif

          // update grid
          C.Energy[id] = E;
          #ifdef DE
          C.GasEnergy[id] = d*ge;
          #endif
        //}

      }
    }
  }

}

/* \fn Real test_cool(Real n, Real T)
 * \brief Cooling function from Creasey 2011. */
Real test_cool(const Real n, const Real T)
{
  Real T0, T1, lambda, cool;
  T0 = 1000.0;
  T1 = 20*T0;
  cool = 0.0;
  lambda = 5.0e-24; //cooling coefficient, erg cm^3 s^-1 

  // constant cooling rate 
  //cool = n*n*lambda;

  // Creasey cooling function
  if (T >= T0 && T <= 0.5*(T1+T0)) {
    cool = n*n*lambda*(T - T0) / T0;
  }
  if (T >= 0.5*(T1+T0) && T <= T1) {
    cool = n*n*lambda*(T1 - T) / T0;
  }
 
  return cool;


}


Real Cloudy_cool(const Real n, const Real T)
{
  Real log_T, lambda, cool;
  log_T = log10(T);
  cool = 0.0;
  printf("%f\n", log_T);

  if (T > 1e1 && T < 1e9) {


  }

  cool = n*n*lambda;

  return cool;
}



void Grid3D::Load_Cooling_Tables()
{
  Real L_arr[81];
  Real T_arr[81];

  int i;
  double xi, yi;

  FILE *infile;
  char buffer[0x1000];
  char * pch;
  char * T;
  char * L;

  i=0;
  infile = fopen("hazy_coolingcurve.txt", "r");
  if (infile == NULL) {
    printf("Unable to open Cloudy file.\n");
    chexit(1);
  }
  while (fgets(buffer, sizeof(buffer), infile) != NULL)
  {
    if (buffer[0] == '#') continue;
    pch = strtok(buffer, "\t");
    T_arr[i] = atof(pch);
    while (pch != NULL)
    {
      pch = strtok(NULL, "\t");
      if (pch != NULL)
        L_arr[i] = atof(pch);
    }
    i++;
  }
  fclose(infile);

  {
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, 81);

    gsl_spline_init(spline, T_arr, L_arr, 81);

    lambda = gsl_spline_eval(spline, log_T, acc);

    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
  }

}

#endif //COOLING_CPU

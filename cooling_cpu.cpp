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
#include <gsl/gsl_spline2d.h>

gsl_interp_accel *xacc;
gsl_interp_accel *yacc;
gsl_spline *highT_C_spline;
gsl_spline2d *lowT_C_spline;
gsl_spline2d *lowT_H_spline;


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
        T_init_ge = ge*(gama-1.0)*SP_ENERGY_UNIT*MP/KB;
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
          ge -= KB*del_T / (MP*(gama-1.0)*SP_ENERGY_UNIT);
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
  Real log_n, log_T, lambda, H, cool;
  log_n = log10(n);
  log_T = log10(T);
  cool = 0.0;

  // Use 2d interpolation to calculate the cooling & heating rates for
  // for low temperature gas
  if (T > 1e1 && T < 1e5) {
    lambda = gsl_spline2d_eval(lowT_C_spline, log_n, log_T, xacc, yacc);
    cool = n*n*pow(10, lambda);
    H = gsl_spline2d_eval(lowT_H_spline, log_n, log_T, xacc, yacc);
    H = n*n*pow(10, H);
    cool -= H;
  }
  // At high temps, 1D interpolation is fine
  else if (T >= 1e5 && T < 1e9) {
    lambda = gsl_spline_eval(highT_C_spline, log_T, xacc);
    cool = n*n*pow(10, lambda);
  }

  return cool;
}



void Grid3D::Load_Cooling_Tables()
{
  Real *n_arr;
  Real *T_arr;
  Real *L_arr;
  Real *H_arr;
  Real *za; 

  int i, xi, yi;
  int nx = 33;
  int ny = 17;
  double x[nx], y[ny];
  za = (Real *) malloc(nx*ny*sizeof(Real));

  FILE *infile;
  char buffer[0x1000];
  char * pch;

  // Allocate arrays for high temperature data
  n_arr = (Real *) malloc(41*sizeof(Real));
  T_arr = (Real *) malloc(41*sizeof(Real));
  L_arr = (Real *) malloc(41*sizeof(Real));
  H_arr = (Real *) malloc(41*sizeof(Real));

  // Read in high T cooling curve (single density)
  i=0;
  infile = fopen("./cloudy_coolingcurve_highT.txt", "r");
  if (infile == NULL) {
    printf("Unable to open Cloudy file.\n");
    chexit(1);
  }
  while (fgets(buffer, sizeof(buffer), infile) != NULL)
  {
    if (buffer[0] == '#') {
      continue;
    }
    else {
      pch = strtok(buffer, "\t");
      n_arr[i] = atof(pch);
      while (pch != NULL)
      {
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          T_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          L_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          H_arr[i] = atof(pch);
      }
      i++;
    }
  }
  fclose(infile);

  xacc = gsl_interp_accel_alloc();
  yacc = gsl_interp_accel_alloc();
  // Initialize the spline for the high T cooling curve
  highT_C_spline = gsl_spline_alloc(gsl_interp_cspline, 41);
  gsl_spline_init(highT_C_spline, T_arr, L_arr, 41);

  free(n_arr);
  free(T_arr);
  free(L_arr);
  free(H_arr);

  // Reallocate arrays for low temperature data
  n_arr = (Real *) malloc(nx*ny*sizeof(Real));
  T_arr = (Real *) malloc(nx*ny*sizeof(Real));
  L_arr = (Real *) malloc(nx*ny*sizeof(Real));
  H_arr = (Real *) malloc(nx*ny*sizeof(Real));

  // Read in low T cooling curve (function of density and temperature)
  i=0;
  infile = fopen("./cloudy_coolingcurve_lowT.txt", "r");
  if (infile == NULL) {
    printf("Unable to open Cloudy file.\n");
    chexit(1);
  }
  while (fgets(buffer, sizeof(buffer), infile) != NULL)
  {
    if (buffer[0] == '#') {
      continue;
    }
    else {
      pch = strtok(buffer, "\t");
      n_arr[i] = atof(pch);
      while (pch != NULL)
      {
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          T_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          L_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          H_arr[i] = atof(pch);
      }
      i++;
    }
  }
  fclose(infile);

  // Allocate memory for 2d interpolation 
  lowT_C_spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, nx, ny);
  lowT_H_spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, nx, ny);

  // Set x and y values
  for (xi=0; xi<nx; xi++) {
    x[xi] = -4.0 + 0.25*xi;
  }
  for (yi=0; yi<ny; yi++) {
    y[yi] = 1.0 + 0.25*yi;
  }
  
  // Set z grid values for cooling interpolation
  for (xi=0; xi<nx; xi++) {
    for (yi=0; yi<ny; yi++) {
      gsl_spline2d_set(lowT_C_spline, za, xi, yi, L_arr[yi + ny*xi]);
    }
  }
  
  // Initialize the interpolation for the low T cooling curve
  gsl_spline2d_init(lowT_C_spline, x, y, za, nx, ny);


  // Set z grid values for heating interpolation
  for (xi=0; xi<nx; xi++) {
    for (yi=0; yi<ny; yi++) {
      gsl_spline2d_set(lowT_H_spline, za, xi, yi, H_arr[yi + ny*xi]);
    }
  }
  
  // Initialize the interpolation for the low T heating curve
  gsl_spline2d_init(lowT_H_spline, x, y, za, nx, ny);

  free(n_arr);
  free(T_arr);
  free(L_arr);
  free(H_arr);
  free(za);

}


void Grid3D::Free_Cooling_Tables()
{
  gsl_interp_accel_free(xacc);
  gsl_interp_accel_free(yacc);
  gsl_spline_free(highT_C_spline);
  gsl_spline2d_free(lowT_C_spline);
  gsl_spline2d_free(lowT_H_spline);
}

#endif //COOLING_CPU

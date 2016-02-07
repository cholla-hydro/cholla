/*! \file cooling_cuda.cu
 *  \brief Functions to calculate cooling rate for a given rho, P, dt. */

#ifdef CUDA
#ifdef COOLING_GPU

#include<cuda.h>
#include<math.h>
#include<io.h>
#include"global.h"
#include"cooling_cuda.h"



/*! \fn void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust the value
           of the total energy for each cell according to the specified cooling function. */
__global__ void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dt, Real gamma)
{
  int n_cells = nx*ny*nz;
  
  Real d, vx, vy, vz, p, E, E_old;
  Real n, T, T_init, T_init_p;
  Real del_T, dt_sub;
  Real cool; //cooling rate per volume, erg/s/cm^3
  #ifdef DE
  Real ge, T_init_ge;
  #endif

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId * blockDim.x;
  int id;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost) {
  //if (xid < nx && yid < ny && zid < nz) {

    // load values of density and pressure
    id = xid + yid*nx + zid*nx*ny;
    d  =  dev_conserved[            id];
    vx =  dev_conserved[1*n_cells + id] / d;
    vy =  dev_conserved[2*n_cells + id] / d;
    vz =  dev_conserved[3*n_cells + id] / d;
    E  =  dev_conserved[4*n_cells + id];
    p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    p  = fmax(p, (Real) TINY_NUMBER);
    #ifdef DE
    ge = dev_conserved[5*n_cells + id] / d;
    p  = d * ge * (gamma - 1.0);
    #endif
    
    // calculate the number density of the gas (in cgs)
    n = d*DENSITY_UNIT / MP;

    // calculate the temperature of the gas
    T_init_p = p*PRESSURE_UNIT/ (n*KB);
    T_init = T_init_p;
    #ifdef DE
    T_init_ge = ge*(gamma-1.0)*SP_ENERGY_UNIT*MP/KB;
    T_init = T_init_ge;
    #endif
    //if (xid == 130 && yid == 6 && zid ==81) printf("%f %f\n", T_init_p, T_init_ge);

    // only allow cooling above 10^4 K
    if (T_init > 1e4 && T_init < 1e9) {
    // calculate cooling rate per volume
    T = T_init;

    if (T > 1e4) {
      cool = Schure_cool(n, T); 
    } else {
      cool = KI_cool(n, T); 
    }
    

    // calculate change in temperature given dt
    del_T = cool*dt*TIME_UNIT*(gamma-1.0)/(n*KB);
    //printf("%d %f %f %f\n", tid, cool, del_T, del_T/T);

    // limit change in temperature to 5%
    while (del_T/T > 0.05) {
      // what dt gives del_T = 0.1*T?
      dt_sub = 0.05*T*n*KB/(cool*TIME_UNIT*(gamma-1.0));
      // apply that dt
      T -= cool*dt_sub*TIME_UNIT*(gamma-1.0)/(n*KB);
      // how much time is left from the original timestep?
      dt -= dt_sub;
      // calculate cooling again
      if (T > 1e4) {
        cool = Schure_cool(n, T);
      } else {
        cool = KI_cool(n, T);
      }
      // calculate new change in temperature
      del_T = cool*dt*TIME_UNIT*(gamma-1.0)/(n*KB);
    }

    // calculate final temperature
    T -= del_T;

    // adjust value of energy based on total change in temperature
    del_T = T_init - T; // total change in T
    E_old = E;
    E -= n*KB*del_T / ((gamma-1.0)*ENERGY_UNIT);
    if (E < 0.0) printf("%3d %3d %3d Negative E after cooling. %f %f %f %f %f\n", xid, yid, zid, del_T, T_init, E_old, n, E);
    #ifdef DE
    ge -= KB*del_T / (MP*(gamma-1.0)*SP_ENERGY_UNIT);
    if (ge < 0.0) printf("%3d %3d %3d Negative ge after cooling. %f %f %f %f %f %f\n", xid, yid, zid, dev_conserved[4*n_cells + id], d*dev_conserved[5*n_cells + id], n, T_init_p, T_init_ge, del_T);
    #endif

    // and send back from kernel
    dev_conserved[4*n_cells + id] = E;
    #ifdef DE
    dev_conserved[5*n_cells + id] = d*ge;
    #endif
    }

  }

}


/* \fn __device__ Real test_cool(Real n, Real T)
 * \brief Cooling function from Creasey 2011. */
__device__ Real test_cool(int tid, Real n, Real T)
{
  Real T0, T1, lambda, cool;
  T0 = 10000.0;
  T1 = 20*T0;
  cool = 0.0;
  //lambda = 5.0e-24; //cooling coefficient, 5e-24 erg cm^3 s^-1
  lambda = 5.0e-20; //cooling coefficient, 5e-24 erg cm^3 s^-1

  // constant cooling rate 
  //cool = n*n*lambda;

  // Creasey cooling function
  if (T >= T0 && T <= 0.5*(T1+T0)) {
    cool = n*n*lambda*(T - T0) / T0;
  }
  if (T >= 0.5*(T1+T0) && T <= T1) {
    cool = n*n*lambda*(T1 - T) / T0;
  }
 

  //printf("%d %f %f\n", tid, T, cool);
  return cool;

}


/* \fn __device__ Real primordial_cool(Real n, Real T)
 * \brief Primordial hydrogen/helium cooling curve 
          derived according to Katz et al. 1996. */
__device__ Real primordial_cool(Real n, Real T)
{
  Real n_h, Y, y, g_ff, cool;
  Real n_h0, n_hp, n_he0, n_hep, n_hepp, n_e, n_e_old; 
  Real alpha_hp, alpha_hep, alpha_d, alpha_hepp, gamma_eh0, gamma_ehe0, gamma_ehep;
  Real le_h0, le_hep, li_h0, li_he0, li_hep, lr_hp, lr_hep, lr_hepp, ld_hep, l_ff;
  Real gamma_lh0, gamma_lhe0, gamma_lhep, e_h0, e_he0, e_hep, H;
  int heat_flag, n_iter;
  Real diff, tol;

  // set flag to 1 for photoionization & heating
  heat_flag = 0;

  //Real X = 0.76; //hydrogen abundance by mass
  Y = 0.24; //helium abundance by mass
  y = Y/(4 - 4*Y);  

  // set the hydrogen number density 
  n_h = n; 

  // calculate the recombination and collisional ionziation rates
  // (Table 2 from Katz 1996)
  alpha_hp   = (8.4e-11) * (1.0/sqrt(T)) * pow((T/1e3),(-0.2)) * (1.0 / (1.0 + pow((T/1e6),(0.7))));
  alpha_hep  = (1.5e-10) * (pow(T,(-0.6353)));
  alpha_d    = (1.9e-3)  * (pow(T,(-1.5))) * exp(-470000.0/T) * (1.0 + 0.3*exp(-94000.0/T));
  alpha_hepp = (3.36e-10)* (1.0/sqrt(T)) * pow((T/1e3),(-0.2)) * (1.0 / (1.0 + pow((T/1e6),(0.7))));
  gamma_eh0  = (5.85e-11)* sqrt(T) * exp(-157809.1/T) * (1.0 / (1.0 + sqrt(T/1e5)));
  gamma_ehe0 = (2.38e-11)* sqrt(T) * exp(-285335.4/T) * (1.0 / (1.0 + sqrt(T/1e5)));
  gamma_ehep = (5.68e-12)* sqrt(T) * exp(-631515.0/T) * (1.0 / (1.0 + sqrt(T/1e5)));
  // externally evaluated integrals for photoionziation rates
  // assumed J(nu) = 10^-22 (nu_L/nu)
  gamma_lh0 = 3.19851e-13;
  gamma_lhe0 = 3.13029e-13;
  gamma_lhep = 2.00541e-14; 
  // externally evaluated integrals for heating rates
  e_h0 = 2.4796e-24;
  e_he0 = 6.86167e-24;
  e_hep = 6.21868e-25; 
  

  // assuming no photoionization, solve equations for number density of
  // each species
  n_e = n_h; //as a first guess, use the hydrogen number density
  n_iter = 20;
  diff = 1.0;
  tol = 1.0e-6;
  if (heat_flag) { 
    for (int i=0; i<n_iter; i++) {
      n_e_old = n_e;
      n_h0   = n_h*alpha_hp / (alpha_hp + gamma_eh0 + gamma_lh0/n_e);
      n_hp   = n_h - n_h0;
      n_hep  = y*n_h / (1.0 + (alpha_hep + alpha_d)/(gamma_ehe0 + gamma_lhe0/n_e) + (gamma_ehep + gamma_lhep/n_e)/alpha_hepp );
      n_he0  = n_hep*(alpha_hep + alpha_d) / (gamma_ehe0 + gamma_lhe0/n_e);
      n_hepp = n_hep*(gamma_ehep + gamma_lhep/n_e)/alpha_hepp;
      n_e    = n_hp + n_hep + 2*n_hepp;
      diff = fabs(n_e_old - n_e);
      if (diff < tol) break;
    }
  }  
  else {
    n_h0   = n_h*alpha_hp / (alpha_hp + gamma_eh0);
    n_hp   = n_h - n_h0;
    n_hep  = y*n_h / (1.0 + (alpha_hep + alpha_d)/(gamma_ehe0) + (gamma_ehep)/alpha_hepp );
    n_he0  = n_hep*(alpha_hep + alpha_d) / (gamma_ehe0);
    n_hepp = n_hep*(gamma_ehep)/alpha_hepp;
    n_e    = n_hp + n_hep + 2*n_hepp;
  }

  // using number densities, calculate cooling rates for
  // various processes (Table 1 from Katz 1996)
  le_h0 = (7.50e-19) * exp(-118348.0/T) * (1.0 / (1.0 + sqrt(T/1e5))) * n_e * n_h0;
  le_hep = (5.54e-17) * pow(T,(-0.397)) * exp(-473638.0/T) * (1.0 / (1.0 + sqrt(T/1e5))) * n_e * n_hep;
  li_h0 = (1.27e-21) * sqrt(T) * exp(-157809.1/T) * (1.0 / (1.0 + sqrt(T/1e5))) * n_e * n_h0;
  li_he0 = (9.38e-22) * sqrt(T) * exp(-285335.4/T) * (1.0 / (1.0 + sqrt(T/1e5))) * n_e * n_he0;
  li_hep = (4.95e-22) * sqrt(T) * exp(-631515.0/T) * (1.0 / (1.0 + sqrt(T/1e5))) * n_e * n_hep;
  lr_hp = (8.70e-27) * sqrt(T) * pow((T/1e3),(-0.2)) * (1.0 / (1.0 + pow((T/1e6),(0.7)))) * n_e * n_hp;
  lr_hep = (1.55e-26) * pow(T,(0.3647)) * n_e * n_hep;
  lr_hepp = (3.48e-26) * sqrt(T) * pow((T/1e3),(-0.2)) * (1.0 / (1.0 + pow((T/1e6),(0.7)))) * n_e * n_hepp;
  ld_hep = (1.24e-13) * pow(T,(-1.5)) * exp(-470000.0/T) * (1.0 + 0.3*exp(-94000.0/T)) * n_e * n_hep;
  g_ff = 1.1 + 0.34*exp(-(5.5-log(T))*(5.5-log(T))/3.0); // Gaunt factor
  l_ff = (1.42e-27) * g_ff * sqrt(T) * (n_hp + n_hep + 4*n_hepp) * n_e;

  // calculate total cooling rate (erg s^-1 cm^-3)
  cool = le_h0 + le_hep + li_h0 + li_he0 + li_hep + lr_hp + lr_hep + lr_hepp + ld_hep + l_ff;

  // calculate total photoionization heating rate
  H = 0.0;
  if (heat_flag) {
    H = n_h0*e_h0 + n_he0*e_he0 + n_hep*e_hep; 
  }
  
  cool -= H;

  return cool;

}


/* \fn __device__ Real KI_cool(Real n, Real T)
 * \brief Analytic fit to solar metallicity ISM cooling curve 
          defined in Koyama & Inutsuka, 2002. */
__device__ Real KI_cool(Real n, Real T)
{
  Real heat = 2.0e-26; //heating rate, erg s^-1 
  Real lambda = 0.0; //cooling rate, erg s^-1 cm^3
  Real cool = 0.0; //cooling per unit volume, erg /s / cm^3

  // KI cooling function 
  lambda = heat * (1.0e7 * exp(-1.14800e5/(T + 1000.)) + 14. * sqrt(T) * exp(-92./T));

  if (T > 10) {
    cool = n*n*lambda;
  }

  return cool;

}


/* \fn __device__ Real Schure_cool(Real n, Real T)
 * \brief Analytic fit to the solar metallicity CIE cooling curve 
          defined in Schure et al., 2009. */
__device__ Real Schure_cool(Real n, Real T)
{
  Real lambda = 0.0; //cooling rate, erg s^-1 cm^3
  Real cool = 0.0; //cooling per unit volume, erg /s / cm^3
  
  // fit to Schure cooling function 
  if (log10(T) > 5.36) {
    lambda = pow(10.0, (0.38 * (log10(T) -7.5) * (log10(T) - 7.5) - 22.6));
  }
  else if (log10(T) < 4.0) {
    lambda = 0.0;
  }
  else {
    lambda = pow(10.0, (-2.5 * (log10(T) - 5.1) * (log10(T) - 5.1) - 20.7));
  }

  // cooling rate per unit volume
  cool = n*n*lambda;

  return cool;

}


#endif //COOLING_GPU
#endif //CUDA

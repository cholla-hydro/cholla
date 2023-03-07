/*! \file cooling_cuda.cu
 *  \brief Functions to calculate cooling rate for a given rho, P, dt. */

#ifdef CUDA
  #ifdef COOLING_GPU

    #include <math.h>

    #include "../cooling/cooling_cuda.h"
    #include "../global/global.h"
    #include "../global/global_cuda.h"
    #include "../utils/gpu.hpp"

    #ifdef CLOUDY_COOL
      #include "../cooling/texture_utilities.h"
    #endif

cudaTextureObject_t coolTexObj = 0;
cudaTextureObject_t heatTexObj = 0;

void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma)
{
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(cooling_kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dt,
                     gama, coolTexObj, heatTexObj);
  CudaCheckError();
}

/*! \fn void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int
 n_ghost, int n_fields, Real dt, Real gamma, cudaTextureObject_t coolTexObj,
 cudaTextureObject_t heatTexObj)
 *  \brief When passed an array of conserved variables and a timestep, adjust
 the value of the total energy for each cell according to the specified cooling
 function. */
__global__ void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                               Real gamma, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj)
{
  int n_cells = nx * ny * nz;
  int is, ie, js, je, ks, ke;
  is = n_ghost;
  ie = nx - n_ghost;
  if (ny == 1) {
    js = 0;
    je = 1;
  } else {
    js = n_ghost;
    je = ny - n_ghost;
  }
  if (nz == 1) {
    ks = 0;
    ke = 1;
  } else {
    ks = n_ghost;
    ke = nz - n_ghost;
  }

  Real d, E;
  Real n, T, T_init;
  Real del_T, dt_sub;
  Real mu;    // mean molecular weight
  Real cool;  // cooling rate per volume, erg/s/cm^3
  // #ifndef DE
  Real vx, vy, vz, p;
    // #endif
    #ifdef DE
  Real ge;
    #endif

  mu = 0.6;
  // mu = 1.27;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int id      = threadIdx.x + blockId * blockDim.x;
  int zid     = id / (nx * ny);
  int yid     = (id - zid * nx * ny) / nx;
  int xid     = id - zid * nx * ny - yid * nx;

  // only threads corresponding to real cells do the calculation
  if (xid >= is && xid < ie && yid >= js && yid < je && zid >= ks && zid < ke) {
    // load values of density and pressure
    d = dev_conserved[id];
    E = dev_conserved[4 * n_cells + id];
    // don't apply cooling if this thread crashed
    if (E < 0.0 || E != E) return;
    // #ifndef DE
    vx = dev_conserved[1 * n_cells + id] / d;
    vy = dev_conserved[2 * n_cells + id] / d;
    vz = dev_conserved[3 * n_cells + id] / d;
    p  = (E - 0.5 * d * (vx * vx + vy * vy + vz * vz)) * (gamma - 1.0);
    p  = fmax(p, (Real)TINY_NUMBER);
    // #endif
    #ifdef DE
    ge = dev_conserved[(n_fields - 1) * n_cells + id] / d;
    ge = fmax(ge, (Real)TINY_NUMBER);
    #endif

    // calculate the number density of the gas (in cgs)
    n = d * DENSITY_UNIT / (mu * MP);

    // calculate the temperature of the gas
    T_init = p * PRESSURE_UNIT / (n * KB);
    #ifdef DE
    T_init = d * ge * (gamma - 1.0) * PRESSURE_UNIT / (n * KB);
    #endif

    // calculate cooling rate per volume
    T = T_init;
    // call the cooling function
    #ifdef CLOUDY_COOL
    cool = Cloudy_cool(n, T, coolTexObj, heatTexObj);
    #else
    cool = CIE_cool(n, T);
    #endif

    // calculate change in temperature given dt
    del_T = cool * dt * TIME_UNIT * (gamma - 1.0) / (n * KB);

    // limit change in temperature to 1%
    while (del_T / T > 0.01) {
      // what dt gives del_T = 0.01*T?
      dt_sub = 0.01 * T * n * KB / (cool * TIME_UNIT * (gamma - 1.0));
      // apply that dt
      T -= cool * dt_sub * TIME_UNIT * (gamma - 1.0) / (n * KB);
      // how much time is left from the original timestep?
      dt -= dt_sub;
    // calculate cooling again
    #ifdef CLOUDY_COOL
      cool = Cloudy_cool(n, T, coolTexObj, heatTexObj);
    #else
      cool = CIE_cool(n, T);
    #endif
      // calculate new change in temperature
      del_T = cool * dt * TIME_UNIT * (gamma - 1.0) / (n * KB);
    }

    // calculate final temperature
    T -= del_T;

    // adjust value of energy based on total change in temperature
    del_T = T_init - T;  // total change in T
    E -= n * KB * del_T / ((gamma - 1.0) * ENERGY_UNIT);
    #ifdef DE
    ge -= KB * del_T / (mu * MP * (gamma - 1.0) * SP_ENERGY_UNIT);
    #endif

    // calculate cooling rate for new T
    #ifdef CLOUDY_COOL
    cool = Cloudy_cool(n, T, coolTexObj, heatTexObj);
    #else
    cool = CIE_cool(n, T);
    // printf("%d %d %d %e %e %e\n", xid, yid, zid, n, T, cool);
    #endif

    // and send back from kernel
    dev_conserved[4 * n_cells + id] = E;
    #ifdef DE
    dev_conserved[(n_fields - 1) * n_cells + id] = d * ge;
    #endif
  }
}

/* \fn __device__ Real test_cool(Real n, Real T)
 * \brief Cooling function from Creasey 2011. */
__device__ Real test_cool(int tid, Real n, Real T)
{
  Real T0, T1, lambda, cool;
  T0   = 10000.0;
  T1   = 20 * T0;
  cool = 0.0;
  // lambda = 5.0e-24; //cooling coefficient, 5e-24 erg cm^3 s^-1
  lambda = 5.0e-20;  // cooling coefficient, 5e-24 erg cm^3 s^-1

  // constant cooling rate
  // cool = n*n*lambda;

  // Creasey cooling function
  if (T >= T0 && T <= 0.5 * (T1 + T0)) {
    cool = n * n * lambda * (T - T0) / T0;
  }
  if (T >= 0.5 * (T1 + T0) && T <= T1) {
    cool = n * n * lambda * (T1 - T) / T0;
  }

  // printf("%d %f %f\n", tid, T, cool);
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

  // Real X = 0.76; //hydrogen abundance by mass
  Y = 0.24;  // helium abundance by mass
  y = Y / (4 - 4 * Y);

  // set the hydrogen number density
  n_h = n;

  // calculate the recombination and collisional ionization rates
  // (Table 2 from Katz 1996)
  alpha_hp   = (8.4e-11) * (1.0 / sqrt(T)) * pow((T / 1e3), (-0.2)) * (1.0 / (1.0 + pow((T / 1e6), (0.7))));
  alpha_hep  = (1.5e-10) * (pow(T, (-0.6353)));
  alpha_d    = (1.9e-3) * (pow(T, (-1.5))) * exp(-470000.0 / T) * (1.0 + 0.3 * exp(-94000.0 / T));
  alpha_hepp = (3.36e-10) * (1.0 / sqrt(T)) * pow((T / 1e3), (-0.2)) * (1.0 / (1.0 + pow((T / 1e6), (0.7))));
  gamma_eh0  = (5.85e-11) * sqrt(T) * exp(-157809.1 / T) * (1.0 / (1.0 + sqrt(T / 1e5)));
  gamma_ehe0 = (2.38e-11) * sqrt(T) * exp(-285335.4 / T) * (1.0 / (1.0 + sqrt(T / 1e5)));
  gamma_ehep = (5.68e-12) * sqrt(T) * exp(-631515.0 / T) * (1.0 / (1.0 + sqrt(T / 1e5)));
  // externally evaluated integrals for photoionization rates
  // assumed J(nu) = 10^-22 (nu_L/nu)
  gamma_lh0  = 3.19851e-13;
  gamma_lhe0 = 3.13029e-13;
  gamma_lhep = 2.00541e-14;
  // externally evaluated integrals for heating rates
  e_h0  = 2.4796e-24;
  e_he0 = 6.86167e-24;
  e_hep = 6.21868e-25;

  // assuming no photoionization, solve equations for number density of
  // each species
  n_e    = n_h;  // as a first guess, use the hydrogen number density
  n_iter = 20;
  diff   = 1.0;
  tol    = 1.0e-6;
  if (heat_flag) {
    for (int i = 0; i < n_iter; i++) {
      n_e_old = n_e;
      n_h0    = n_h * alpha_hp / (alpha_hp + gamma_eh0 + gamma_lh0 / n_e);
      n_hp    = n_h - n_h0;
      n_hep   = y * n_h /
              (1.0 + (alpha_hep + alpha_d) / (gamma_ehe0 + gamma_lhe0 / n_e) +
               (gamma_ehep + gamma_lhep / n_e) / alpha_hepp);
      n_he0  = n_hep * (alpha_hep + alpha_d) / (gamma_ehe0 + gamma_lhe0 / n_e);
      n_hepp = n_hep * (gamma_ehep + gamma_lhep / n_e) / alpha_hepp;
      n_e    = n_hp + n_hep + 2 * n_hepp;
      diff   = fabs(n_e_old - n_e);
      if (diff < tol) break;
    }
  } else {
    n_h0   = n_h * alpha_hp / (alpha_hp + gamma_eh0);
    n_hp   = n_h - n_h0;
    n_hep  = y * n_h / (1.0 + (alpha_hep + alpha_d) / (gamma_ehe0) + (gamma_ehep) / alpha_hepp);
    n_he0  = n_hep * (alpha_hep + alpha_d) / (gamma_ehe0);
    n_hepp = n_hep * (gamma_ehep) / alpha_hepp;
    n_e    = n_hp + n_hep + 2 * n_hepp;
  }

  // using number densities, calculate cooling rates for
  // various processes (Table 1 from Katz 1996)
  le_h0   = (7.50e-19) * exp(-118348.0 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_h0;
  le_hep  = (5.54e-17) * pow(T, (-0.397)) * exp(-473638.0 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_hep;
  li_h0   = (1.27e-21) * sqrt(T) * exp(-157809.1 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_h0;
  li_he0  = (9.38e-22) * sqrt(T) * exp(-285335.4 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_he0;
  li_hep  = (4.95e-22) * sqrt(T) * exp(-631515.0 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_hep;
  lr_hp   = (8.70e-27) * sqrt(T) * pow((T / 1e3), (-0.2)) * (1.0 / (1.0 + pow((T / 1e6), (0.7)))) * n_e * n_hp;
  lr_hep  = (1.55e-26) * pow(T, (0.3647)) * n_e * n_hep;
  lr_hepp = (3.48e-26) * sqrt(T) * pow((T / 1e3), (-0.2)) * (1.0 / (1.0 + pow((T / 1e6), (0.7)))) * n_e * n_hepp;
  ld_hep  = (1.24e-13) * pow(T, (-1.5)) * exp(-470000.0 / T) * (1.0 + 0.3 * exp(-94000.0 / T)) * n_e * n_hep;
  g_ff    = 1.1 + 0.34 * exp(-(5.5 - log(T)) * (5.5 - log(T)) / 3.0);  // Gaunt factor
  l_ff    = (1.42e-27) * g_ff * sqrt(T) * (n_hp + n_hep + 4 * n_hepp) * n_e;

  // calculate total cooling rate (erg s^-1 cm^-3)
  cool = le_h0 + le_hep + li_h0 + li_he0 + li_hep + lr_hp + lr_hep + lr_hepp + ld_hep + l_ff;

  // calculate total photoionization heating rate
  H = 0.0;
  if (heat_flag) {
    H = n_h0 * e_h0 + n_he0 * e_he0 + n_hep * e_hep;
  }

  cool -= H;

  return cool;
}

/* \fn __device__ Real CIE_cool(Real n, Real T)
 * \brief Analytic fit to a solar metallicity CIE cooling curve
          calculated using Cloudy. */
__device__ Real CIE_cool(Real n, Real T)
{
  Real lambda = 0.0;  // cooling rate, erg s^-1 cm^3
  Real cool   = 0.0;  // cooling per unit volume, erg /s / cm^3

  // fit to CIE cooling function
  if (log10(T) < 4.0) {
    lambda = 0.0;
  } else if (log10(T) >= 4.0 && log10(T) < 5.9) {
    lambda = pow(10.0, (-1.3 * (log10(T) - 5.25) * (log10(T) - 5.25) - 21.25));
  } else if (log10(T) >= 5.9 && log10(T) < 7.4) {
    lambda = pow(10.0, (0.7 * (log10(T) - 7.1) * (log10(T) - 7.1) - 22.8));
  } else {
    lambda = pow(10.0, (0.45 * log10(T) - 26.065));
  }

  // cooling rate per unit volume
  cool = n * n * lambda;

  return cool;
}

    #ifdef CLOUDY_COOL
/* \fn __device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t
 coolTexObj, cudaTextureObject_t heatTexObj)
 * \brief Uses texture mapping to interpolate Cloudy cooling/heating
          tables at z = 0 with solar metallicity and an HM05 UV background. */
__device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj)
{
  Real lambda = 0.0;  // cooling rate, erg s^-1 cm^3
  Real H      = 0.0;  // heating rate, erg s^-1 cm^3
  Real cool   = 0.0;  // cooling per unit volume, erg /s / cm^3
  float log_n, log_T;
  log_n = log10(n);
  log_T = log10(T);

  // remap coordinates for texture
  // remapped = (input - TABLE_MIN_VALUE)*(1/TABLE_SPACING)
  // remapped = (input - TABLE_MIN_VALUE)*(NUM_CELLS_PER_DECADE)
  log_T = (log_T - 1.0) * 10;
  log_n = (log_n + 6.0) * 10;

  // Note: although the cloudy table columns are n,T,L,H , T is the fastest
  // variable so it is treated as "x" This is why the Texture calls are T first,
  // then n: Bilinear_Texture(tex, log_T, log_n)

  // don't cool below 10 K
  if (log10(T) > 1.0) {
    lambda = Bilinear_Texture(coolTexObj, log_T, log_n);
  } else
    lambda = 0.0;
  H = Bilinear_Texture(heatTexObj, log_T, log_n);

  // cooling rate per unit volume
  cool = n * n * (powf(10, lambda) - powf(10, H));
  // printf("DEBUG Cloudy L350: %.17e\n",cool);
  return cool;
}
    #endif  // CLOUDY_COOL

  #endif  // COOLING_GPU
#endif    // CUDA

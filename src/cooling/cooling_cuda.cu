/*! \file cooling_cuda.cu
 *  \brief Functions to calculate cooling rate for a given rho, P, dt.
 *
 *  Nearly all of the functionality implemented in this file follow a common
 *  strategy. At this time of writing, there are essentially 2 functions that
 *  deviate from the strategy (`test_cool` and `primordial_cool`), which are
 *  left over from earlier implementations.
 *
 *  Interface
 *  ---------
 *  In detail, the `configure_cooling_callback` function produces a std::function, which serves as a callback that
 *  performs cooling
 *  - a `std::function` instance acts a more generalized function-pointer that is able to wrap ordinary
 *    functions **OR** a struct that can act like a function (sometimes known as a "functor" or "callable")
 *  - at this time, the callback will perform cooling, with specialized code based on the cooling recipe, that acts
 *    modifies the fields tracked by a `grid` object. The actual implementation of the callback is opaque to the
 *    rest of cholla.
 *
 *  Implementation Strategy
 *  -----------------------
 *  At this time of writing, we implement cooling functionality with some basic template-machinery
 *  - our use of templates allows us to create optimal code for each "cooling recipe", while minimizing duplicated
 *    code and avoiding conditional compilation with ifdef statements
 *
 *  Our idea revolves around the concept of a `CoolingRecipe`.
 *  - we loosely define a `CoolingRecipe` as any type that implements a `__device__` member-function with the
 *    `Real cool_rate(Real n, Real T)` function signature (i.e. it computes the cooling rate per unit volume at a
 *    given number density and temperature)
 *  - in principle, this may or may not include the effects of photoelectric heating (it may eventually make more
 *    sense to model photoelectric heating separately)
 *
 *  To perform cooling with a given recipe, we an instance of the cooling_recipe to the  __global__ function,
 * `cooling_kernel`. The concrete type of the `CoolingRecipe` is a template parameter, `cooling_kernel`, so that
 * invocations of kernels are effectively specialized for each type of recipe.
 *
 *  The `CoolingUpdateExecutor` class template simply serves as a nice way to package the particular kind of
 * CoolingRecipe (and and cooling-recipe-specific parameters) with the logic for launching `cooling_kernel`.
 */

#include <math.h>

#include "../cooling/cooling_cuda.h"
#include "../cooling/load_cloudy_texture.h"  // provides Load_Cuda_Textures and Free_Cuda_Textures
#include "../cooling/texture_utilities.h"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/error_handling.h"
#include "../utils/gpu.hpp"

static bool allocated_heating_cooling_textures = false;
cudaTextureObject_t coolTexObj                 = 0;
cudaTextureObject_t heatTexObj                 = 0;

template <typename CoolingRecipe>
__global__ void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                               Real gamma, CoolingRecipe recipe);

/*! \brief Instances of this class template are callables that serve as callback functions for applying
 *   cooling to the grid.
 *
 *  In more detail:
 *  - This class template is specialized with a "cooling recipe," which encapsulates the
 *    type of cooling (e.g. cloudy cooling, analytic cie cooling, analytic ti cooling, etc.).
 *  - After constructing an instance of this class, the instance is typically wrapped within
 *    ``std::function`` and then returned to the rest of Cholla
 *  - For the uninitiated, ``std::function`` performs type-erasure on its contents. Essentially,
 *    the rest of Cholla is totally agnostic about which function is contained by ``std::function``
 *    (essentially, a ``std::function`` instance is a more general-purpose kind of function pointer
 *    that can be used on any callable like a callable struct with some associated state or an
 *    ordinary function)
 */
template <typename CoolingRecipe>
class CoolingUpdateExecutor
{
  CoolingRecipe recipe_;

 public:
  CoolingUpdateExecutor(CoolingRecipe recipe) : recipe_(recipe) {}

  void operator()(Grid3D &grid) const
  {
    Header &H           = grid.H;
    Real *dev_conserved = grid.C.device;
    int n_cells         = H.nx * H.ny * H.nz;
    int ngrid           = (n_cells + TPB - 1) / TPB;
    dim3 dim1dGrid(ngrid, 1, 1);
    dim3 dim1dBlock(TPB, 1, 1);

    hipLaunchKernelGGL(cooling_kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, H.nx, H.ny, H.nz, H.n_ghost,
                       H.n_fields, H.dt, gama, this->recipe_);
    GPU_Error_Check();
  }
};

/*! \fn void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int
 n_ghost, int n_fields, Real dt, Real gamma, cudaTextureObject_t coolTexObj,
 cudaTextureObject_t heatTexObj)
 *  \brief When passed an array of conserved variables and a timestep, adjust
 the value of the total energy for each cell according to the specified cooling
 function. */
template <typename CoolingRecipe>
__global__ void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                               Real gamma, CoolingRecipe recipe)
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
    if (E < 0.0 || E != E) {
      return;
    }
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
    cool = recipe.cool_rate(n, T);

    // calculate change in temperature given dt
    del_T = cool * dt * TIME_UNIT * (gamma - 1.0) / (n * KB);

    // limit change in temperature to 1% (we use fabs for when heating dominates)
    while (fabs(del_T / T) > 0.01) {
      // what dt gives del_T with a magnitude of 0.01*T? (we use fabs for cases when heating dominates)
      dt_sub = fabs(0.01 * T * n * KB / (cool * TIME_UNIT * (gamma - 1.0)));
      // apply that dt
      T -= cool * dt_sub * TIME_UNIT * (gamma - 1.0) / (n * KB);
      // how much time is left from the original timestep?
      dt -= dt_sub;

      // calculate cooling again
      cool = recipe.cool_rate(n, T);
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
      if (diff < tol) {
        break;
      }
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

namespace detail
{

/*! \brief computes the cooling rate, based on an analytic fit to a solar metallicity
 *     CIE cooling curve calculated using Cloudy. For log10T, this returns 0
 *
 *   \return The cooling rate, lambda, in units of erg s^-1 cm^3 (it is NEVER negative)
 *
 *   \note
 *   It may not be necessary to use __forceinline__, I just used it to ensure I didn't harm existing
 *   performance
 *
 *   \note
 *   The actual formula for the fit is first described in the appendix of
 *   (Schneider & Robertson 2018)[https://ui.adsabs.harvard.edu/abs/2018ApJ...860..135S/abstract
 */
__forceinline__ __device__ Real analytic_cie_lambda(Real log10T)
{
  // fit to CIE cooling function
  if (log10T < 4.0) {
    return 0.0;
  } else if (log10T >= 4.0 && log10T < 5.9) {
    return pow(10.0, (-1.3 * (log10T - 5.25) * (log10T - 5.25) - 21.25));
  } else if (log10T >= 5.9 && log10T < 7.4) {
    return pow(10.0, (0.7 * (log10T - 7.1) * (log10T - 7.1) - 22.8));
  } else {
    return pow(10.0, (0.45 * log10T - 26.065));
  }
}

}  // namespace detail

/*! \brief Analytic fit to a solar metallicity CIE cooling curve calculated using Cloudy.
 */
struct CoolRecipeCIE {
  __device__ static Real cool_rate(Real n, Real T)
  {
    Real lambda = detail::analytic_cie_lambda(log10(T));  // cooling rate, erg s^-1 cm^3
    Real cool   = n * n * lambda;                         // cooling per unit volume, erg /s / cm^3
    return cool;
  }
};

/*! \brief Uses texture mapping to interpolate Cloudy cooling/heating
 *         tables at z = 0 with solar metallicity and an HM05 UV background. */
class CoolRecipeCloudy
{
  cudaTextureObject_t coolTexObj_;
  cudaTextureObject_t heatTexObj_;

 public:
  __host__ CoolRecipeCloudy(std::string filename)
  {
    // for now, we simply don't deallocate the textures
    // -> this is poor form and something that should be fixed...
    // -> in reality, this won't cause any immediate issues since the textures
    //    are global and will live for the lifetime of the simulation
    if (!allocated_heating_cooling_textures) {
      allocated_heating_cooling_textures = true;
      Load_Cuda_Textures(filename);
    }
    this->coolTexObj_ = coolTexObj;
    this->heatTexObj_ = heatTexObj;
  }

  __device__ Real cool_rate(Real n, Real T) const;
};

__device__ Real CoolRecipeCloudy::cool_rate(Real n, Real T) const
{
  Real lambda  = 0.0;  // log cooling rate, erg s^-1 cm^3
  Real cooling = 0.0;  // cooling per unit volume, erg /s / cm^3
  Real heating = 0.0;  // heating per unit volume, erg /s / cm^3

  // To keep texture code simple, we use floats (which have built-in support) as opposed to doubles (which would require
  // casting)
  float log_n, log_T;
  log_n = log10(n);
  log_T = log10(T);

  // remap coordinates for texture
  // remapped = (input - TABLE_MIN_VALUE)*(1/TABLE_SPACING)
  // remapped = (input - TABLE_MIN_VALUE)*(NUM_CELLS_PER_DECADE)
  const Real remap_log_T = (log_T - 1.0) * 10;
  const Real remap_log_n = (log_n + 6.0) * 10;

  // Note: although the cloudy table columns are n,T,L,H , T is the fastest
  // variable so it is treated as "x" This is why the Texture calls are T first,
  // then n: Bilinear_Texture(tex, remap_log_T, remap_log_n)

  // cloudy cooling tables cut off at 10^9 K, use the CIE analytic fit above
  // this temp.
  if (log10(T) > 9.0) {
    lambda = 0.45 * log10(T) - 26.065;
  } else if (log10(T) >= 1.0) {
    lambda       = Bilinear_Texture(this->coolTexObj_, remap_log_T, remap_log_n);
    const Real H = Bilinear_Texture(this->heatTexObj_, remap_log_T, remap_log_n);
    heating      = pow(10, H);
  } else {
    // Do nothing below 10 K
    return 0.0;
  }

  cooling = pow(10, lambda);
  return n * n * (cooling - heating);
}

/*! Encapsulates our model and configuration for photoelectric heating
 *
 *  This implements a very simple model
 *  - we apply uniform photoelectric heating (over all space and time) to all gas at temperatures
 *    below 1e4 K
 *  - this model is described within
 *    [Kim & Ostriker 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract)
 *
 *  @note
 *  In the future, one could imagine implementing a more sophisticated recipe like TIGRESS
 *  - For example the amount of heating could be coupled with the properties of clusters
 *    within the simulation volume
 *  - If we started to model varying mmw, we could also adopt the TIGRESS strategy to more
 *    smoothly turn off heating at higher temperatures
 */
struct PhotoelectricHeatingModel {
  /*! This theoretically represents the mean density in the simulation volume. A value of 0.0
   *  indicates that there is no heating.
   *
   *  @note
   *  I can't remember the precise interpretation, but I think the idea may be that it may be
   *  used because it loosely relates to the rate of star formation...
   */
  double n_av_cgs = 0.0;

  bool is_active() const { return n_av_cgs != 0.0; }

  /*! \brief computes the heating rate per unit volume, erg /s / cm^3.
   *
   *  This **NEVER** returns a negative value.
   */
  __device__ Real operator()(Real n, Real T) const { return (T < 1e4) ? n * n_av_cgs * 1.0e-26 : 0.0; }
};

class CoolRecipeCloudyAndPhotoHeating
{
  CoolRecipeCloudy pure_cloudy_recipe;
  PhotoelectricHeatingModel photoelectric_fn;

 public:
  __host__ CoolRecipeCloudyAndPhotoHeating(std::string filename, PhotoelectricHeatingModel photoelectric_fn)
      : pure_cloudy_recipe(filename), photoelectric_fn{photoelectric_fn}
  {
  }

  __device__ Real cool_rate(Real n, Real T) const
  {
    return pure_cloudy_recipe.cool_rate(n, T) - photoelectric_fn(n, T);
  }
};

/*! \brief Analytic cooling/heating recipe that roughly matches the "TI" cooling runs shown in
 *     in [Kim & Ostriker 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract)
 *
 *  For temperatures below 1e4 K:
 *  - We adopt the same analytic fitting formula as Kim & Ostriker 2015 for T < 1e4 K, which is an
 *    analytic fit to the results of Koyama & Inutsuka (2002).
 *  - a description of this fit is provided within
 *    [Kim+2008](https://ui.adsabs.harvard.edu/abs/2008ApJ...681.1148K/abstract)
 *  For temperatures above 1e4 K
 *  - we directly use the exact same analytic CIE fit as CoolRecipeCIE
 *
 * \warning
 * Be aware, that all of our cooling infrastructure probably does not properly account for changes in
 * mean molecular weights. Historically, we just assumed a fixed mean molecular weight of 0.6 when we
 * used a CIE analytic fit. In practice, the fit below 1e4 K is intended to be used with a mean
 * molecular weight fixed to ~1.25
 */
class CoolRecipeTI
{
  PhotoelectricHeatingModel photoelectric_fn;

  // doesn't include any photoelectric heating!
  __device__ static Real cool_rate_only_(Real n, Real T)
  {
    Real lambda;  // cooling rate, erg s^-1 cm^3
    if (T < 10.0) {
      lambda = 0.0;  // no cooling below 10 K
    } else if (T >= 10.0 && T < 1e4) {
      // Koyama & Inutsaka 2002 analytic fit
      lambda = 2e-26 * (1e7 * exp(-1.148e5 / (T + 1000.0)) + 1.4e-2 * sqrt(T) * exp(-92.0 / T));
    } else {
      lambda = detail::analytic_cie_lambda(log10(T));
    }

    return n * (n * lambda);  // cooling rate per unit volume, erg /s / cm^3
  }

 public:
  __host__ CoolRecipeTI(PhotoelectricHeatingModel photoelectric_fn) : photoelectric_fn{photoelectric_fn} {}

  __device__ Real cool_rate(Real n, Real T) { return cool_rate_only_(n, T) - photoelectric_fn(n, T); }
};

std::function<void(Grid3D &)> configure_cooling_callback(std::string kind, ParameterMap &pmap)
{
  // the caller of this function will is responsible for raising an error when:
  // - "chemistry.data_file" is set, but we aren't using a recipe that doesn't need a datafile

  // First, we configure an instance of PhotoelectricHeatingModel, based off the parameters
  // -> to help provide informative error messages, we store the names of the parameters in variables
  // -> maybe we should only use a single parameter, to just specify the value of n_av_cgs?
  const char *use_photoelectric_parname  = "chemistry.photoelectric_heating";
  const char *photoelectric_n_av_parname = "chemistry.photoelectric_n_av_cgs";

  PhotoelectricHeatingModel photoelectric_fn;
  if (pmap.value_or(use_photoelectric_parname, false)) {
    // In this case, we want to actually use photoelectric heating
    double n_av_cgs = pmap.value_or(photoelectric_n_av_parname, 100.0);
    CHOLLA_ASSERT(n_av_cgs > 0.0, "The \"%s\" parameter cannot specify a non-positive value",
                  photoelectric_n_av_parname);
    photoelectric_fn = PhotoelectricHeatingModel{n_av_cgs};
  } else {
    CHOLLA_ASSERT(!pmap.has_param(photoelectric_n_av_parname),
                  "It is an error to specify the \"%s\" parameter when the \"%s\" hasn't "
                  "explicitly been set to true.",
                  photoelectric_n_av_parname, use_photoelectric_parname);
    photoelectric_fn = PhotoelectricHeatingModel{0.0};  // this means that there isn't heating
  }

  // Next, we branch based on the cooling-recipe
  if (kind == "tabulated-cloudy") {
    // since photoelectric_fn can be configured to be inactive, we could probably just
    // consolidate the definitions of CoolRecipeCloudyAndPhotoHeating and CoolRecipeCloudy

    std::string filename = pmap.value_or("chemistry.data_file", std::string());
    if (photoelectric_fn.is_active()) {
      CoolRecipeCloudyAndPhotoHeating recipe(filename, photoelectric_fn);
      CoolingUpdateExecutor<CoolRecipeCloudyAndPhotoHeating> updater(recipe);
      return {updater};
    } else {
      CoolRecipeCloudy recipe(filename);
      CoolingUpdateExecutor<CoolRecipeCloudy> updater(recipe);
      return {updater};
    }
  } else if (kind == "piecewise-cie") {
    CHOLLA_ASSERT(not photoelectric_fn.is_active(),
                  "The \"%s\" cooling recipe is **NOT** compatible with photoelectric heating", kind.c_str());
    CoolRecipeCIE recipe{};
    CoolingUpdateExecutor<CoolRecipeCIE> updater(recipe);
    return {updater};
  } else if (kind == "piecewise-ti") {
    CoolRecipeTI recipe{photoelectric_fn};
    CoolingUpdateExecutor<CoolRecipeTI> updater(recipe);
    return {updater};
  }
  return {};
}

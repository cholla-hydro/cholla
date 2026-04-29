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

#include "../cooling/cool_components.h"
#include "../cooling/cooling_cuda.h"
#include "../cooling/load_cloudy_texture.h"  // provides cool_component::CloudyHeatAndCool
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/error_handling.h"
#include "../utils/gpu.hpp"

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

/*! \brief Analytic fit to a solar metallicity CIE cooling curve calculated using Cloudy.
 */
struct CoolRecipeCIE {
  __device__ static Real cool_rate(Real n, Real T)
  {
    Real lambda = cool_component::analytic_cie_lambda(log10(T));  // cooling rate, erg s^-1 cm^3
    Real cool   = n * n * lambda;                                 // cooling per unit volume, erg /s / cm^3
    return cool;
  }
};

/*! \brief Uses texture mapping to interpolate Cloudy cooling/heating
 *         tables at z = 0 with solar metallicity and an HM05 UV background. */
class CoolRecipeCloudy
{
  cool_component::CloudyHeatAndCool net_cloudy_;

 public:
  __host__ CoolRecipeCloudy(ParameterMap &pmap) : net_cloudy_(pmap) {}
  __device__ Real cool_rate(Real n, Real T) const { return net_cloudy_(n, T); }
};

class CoolRecipeCloudyAndPhotoHeating
{
  cool_component::CloudyHeatAndCool net_cloudy_;
  cool_component::PhotoelectricHeatingModel photoelectric_fn_;

 public:
  __host__ CoolRecipeCloudyAndPhotoHeating(ParameterMap &pmap) : net_cloudy_(pmap), photoelectric_fn_(pmap) {}

  __device__ Real cool_rate(Real n, Real T) const { return net_cloudy_(n, T) - photoelectric_fn_(n, T); }
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
class CoolRecipeTIAndCIE
{
  cool_component::PhotoelectricHeatingModel photoelectric_fn;

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
      lambda = cool_component::analytic_cie_lambda(log10(T));
    }

    return n * (n * lambda);  // cooling rate per unit volume, erg /s / cm^3
  }

 public:
  explicit __host__ CoolRecipeTIAndCIE(ParameterMap &pmap) : photoelectric_fn(pmap) {}

  __device__ Real cool_rate(Real n, Real T) { return cool_rate_only_(n, T) - photoelectric_fn(n, T); }
};

std::function<void(Grid3D &)> configure_cooling_callback(std::string kind, ParameterMap &pmap)
{
  // the caller of this function will is responsible for raising an error when:
  // - "chemistry.data_file" is set, but we aren't using a recipe that doesn't need a datafile

  bool use_photoelectric_heating = cool_component::PhotoelectricHeatingModel::is_specified_by_params(pmap);

  // Next, we branch based on the cooling-recipe
  if (kind == "tabulated-cloudy") {
    // since photoelectric_fn can be configured to be inactive, we could probably just
    // consolidate the definitions of CoolRecipeCloudyAndPhotoHeating and CoolRecipeCloudy

    if (use_photoelectric_heating) {
      CoolRecipeCloudyAndPhotoHeating recipe(pmap);
      CoolingUpdateExecutor<CoolRecipeCloudyAndPhotoHeating> updater(recipe);
      return {updater};
    } else {
      CoolRecipeCloudy recipe(pmap);
      CoolingUpdateExecutor<CoolRecipeCloudy> updater(recipe);
      return {updater};
    }
  } else if (kind == "piecewise-cie") {
    CHOLLA_ASSERT(not use_photoelectric_heating,
                  "The \"%s\" cooling recipe is **NOT** compatible with photoelectric heating", kind.c_str());
    CoolRecipeCIE recipe{};
    CoolingUpdateExecutor<CoolRecipeCIE> updater(recipe);
    return {updater};
  } else if (kind == "piecewise-ti+cie") {
    CoolRecipeTIAndCIE recipe(pmap);
    CoolingUpdateExecutor<CoolRecipeTIAndCIE> updater(recipe);
    return {updater};
  }
  return {};
}

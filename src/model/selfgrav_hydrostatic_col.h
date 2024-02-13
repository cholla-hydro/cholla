#ifndef SELFGRAV_HYDROSTATIC_COL
#define SELFGRAV_HYDROSTATIC_COL

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <type_traits>

#include "../global/global.h"
#include "../utils/error_handling.h"

namespace ode_detail{

struct NullFn{
  /// acts as a dummy placeholder for representing a "logger function"
  template<int N>
  void operator()(Real x, Real cur_step, std::array<Real, N> yvec,
                  std::array<Real, N> cur_yvec_step) const noexcept
  {}
};

} // namespace ode_detail

template <int N, class DerivFn, class LogFn = ode_detail::NullFn>
class ODEIntegrator{
  /// Represents an integrator of a system of (coupled) 1D ODEs.
  ///
  /// @tparam N the number of ODEs that are integrated
  /// @tparam DerivFn Specifies the derivative of the ODEs
  /// @tparam LogFn An optional logger function that can be used to log
  ///     information as the integrator progresses.

public:
  ODEIntegrator() = delete;

  ODEIntegrator(DerivFn deriv_fn, LogFn log_fn = ode_detail::NullFn{})
    : deriv_fn_(deriv_fn), log_fn_(log_fn)
  {}

  std::array<Real, N> integrate(Real xstart, Real xend, Real nominal_step,
                                std::array<Real, N> yvec_start) const noexcept
  {
    Real cur_x = xstart;
    std::array<Real, N> cur_yvec = yvec_start;

    while (true) {
      Real cur_step = std::min(xend - cur_x, nominal_step);

      // calculate the change in yvec over cur_step
#if 1
      std::array<Real, N> d_yvec = calc_midpoint_step(cur_x, cur_yvec,
                                                      cur_step);
#else
      std::array<Real, N> d_yvec = calc_rk4_step(cur_x, cur_yvec, cur_step);
#endif
        
      // potentially call the logger function
      if constexpr (not std::is_same_v<LogFn, ode_detail::NullFn>) {
        log_fn_(cur_x, cur_step, cur_yvec, d_yvec);
      }

      // update cur_x and cur_yvec
      cur_x += cur_step;
      for (int i = 0; i < N; i++) { cur_yvec[i] += d_yvec[i]; }

      if (cur_step < nominal_step) {break;}
    }

    return cur_yvec;
  }

private: // helper functions

  /// Calculate the change in some vector `y`, over some step in the independent
  /// variable `x` using the explicit midpoint method.
  ///
  /// @param cur_x The current value of x
  /// @param yvec  The value of y at cur_x
  /// @param step The amount to increase cur_x by
  ///
  /// @returns ystep Equivalent to `y(cur_x + step) - y(cur_x)`
  std::array<Real, N> calc_midpoint_step(Real cur_x, std::array<Real, N> yvec, 
                                         Real step) const noexcept
  {
    // first, estimate the yvec at cur_x + 0.5*step
    std::array<Real, N> deriv_guess = deriv_fn_(cur_x, yvec);
    std::array<Real, N> y_midpoint;
    for (int i = 0; i < 3; i++) {
      y_midpoint[i] = yvec[i] + 0.5 * step * deriv_guess[i];
    }

    std::array<Real, N> full_step = deriv_fn_(cur_x + 0.5 * step, y_midpoint);
    for (int i = 0; i < 3; i++) { full_step[i] *= step; }
    return full_step;
  }

  std::array<Real, N> calc_rk4_step(Real cur_x, std::array<Real, N> yvec, 
                                    Real step) const noexcept
  {

    auto step_multiply = [=](std::array<Real, N> arg) -> std::array<Real, N> {
      std::array<Real, N> out;
      for (int i = 0; i < N; i++) { out[i] = step * arg[i]; }
      return out;
    };

    std::array<Real, N> k1 = step_multiply(deriv_fn_(cur_x, yvec));

    std::array<Real, N> tmp;
    for (int i = 0; i < N; i++) { tmp[i] = yvec[i] + 0.5 * k1[i]; }
    std::array<Real, N> k2 = step_multiply(deriv_fn_(cur_x + 0.5 * step, tmp));

    for (int i = 0; i < N; i++) { tmp[i] = yvec[i] + 0.5 * k2[i]; }
    std::array<Real, N> k3 = step_multiply(deriv_fn_(cur_x + 0.5 * step, tmp));

    for (int i = 0; i < N; i++) { tmp[i] = yvec[i] + k3[i]; }
    std::array<Real, N> k4 = step_multiply(deriv_fn_(cur_x + step, tmp));

    std::array<Real, N> d_yvec{};  // uses value-initialization to initialize
                                   // all elements to Real{} (aka 0.0)
    for (int i = 0; i < N; i++) {
      d_yvec[i] = (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0;
    }
    return d_yvec;
  }

private:
  DerivFn deriv_fn_;
  LogFn log_fn_;
};

// tracks some basic components about the grid's z-component
struct ZGridProps{
  const Real global_min;
  const Real global_max;
  const int global_ncells;
  const Real cell_width;

public:
  ZGridProps() = delete;

  ZGridProps(Real global_min, Real global_width, int global_ncells)
    : global_min(global_min), global_max(global_min + global_width),
      global_ncells(global_ncells), cell_width(global_width/global_ncells)
  {
    CHOLLA_ASSERT((global_min < 0) and (global_width > 0),
                  "global-min must be negative & global-width must be positive");
    Real diff = std::fabs((global_min * -2) - global_width);
    CHOLLA_ASSERT(diff <= (4 * std::numeric_limits<Real>::epsilon() * global_width),
                  "The global-domain must be centered on z = 0");
    CHOLLA_ASSERT((global_ncells > 0) and ((global_ncells % 2) == 0),
                  "Currently, there is only support for simulations with an integer number of cells");
  }

  bool origin_is_cell_edge_aligned() const { return true; }
  Real left_cell_edge(int i) const { return global_min + i * cell_width; }
  Real right_cell_edge(int i) const { return global_min + (i+1) * cell_width; }
};

namespace selfgrav_hydrostatic_col{

// a crude lookup table mapping the quantity names to indices
struct LUT {
  enum { dPhiGasZ_dz = 0, PhiGasZ = 1, posZ_unnormalized_Sigma = 2 };
};


/// This is all functionality related to initialization strategy described in
/// Wang+ (2010), https://ui.adsabs.harvard.edu/abs/2010MNRAS.407..705W
///
/// In this paper, they show how to initialize a self-gravitating gas disk in
/// the limit where the disk is relatively thin. This lets them come up with a
/// reduced form of the Poisson Equation for the gas potential:
///    d2_PhiGas_dz2 = 4 * pi * G    (Eqn 14)
/// where `d2_PhiGas_dz2` is the second derivative of the gas gravitational
/// potential with respect to z.
///
/// In particular, we use their "Potential method", to solve for a vertical
/// density profiles at fixed cyclindrical radius. This only works for a disk
/// where the gas is isothermal. By assuming that the gas is isothermal, they
/// derive that the gas density is given by: 
///   rho_gas(z) = rho_midplane * exp(-1 * PhiZ(z) / isoth_term),    (Eqn 18)
///
/// The primary equation we solve comes from plugging Eqn 18 into Eqn 14. This
/// produces the following equation:
///   d2_PhiGasZ_dz2 = alpha*rho_midplane * exp(-1*PhiZ(z)/isoth_term)  (Eqn 19)
///
/// where:
///    - `alpha` is `4 * pi * G`
///    - `isoth_term` has multiple equivalent definitions. In short it is
///      the constant ratio between p and rho in the disk. Other equivalent
///      definitions include:
///         - `c_s,isothermal^2` where `c_s,isothermal` is the isothermal
///           sound-speed in the disk. This sound speed is given by
///           `sqrt(kboltz*T/(mu*mH))`c_s,ad)` 
///         - `(gamma - 1) * specific_internal_energy` or `(c_s,ad)^2/gamma`.
///           In these formulas, `gamma` is the adiabatic index held fixed
///           throughout the sim. `c_s,ad` is 
///    - `PhiZ(z)` is the TOTAL gravitational potential measured with respect
///      to `z = 0` (usually gravitational potentials are written such that
///      they approach zero at infinity). It's equal to `PhiOtherZ + PhiGasZ`
///    - `d2_PhiGasZ_dz2` is the second derivative of the gravitational
///      potential of just the gas, where the potential has been measured with
///      respect to `z=0`.
///
/// At the same time that we solve Eqn 19, we also want to solve for
/// `posZ_unnormalized_Sigma0`, (the integral or rho_gas(z) from z = 0
/// to some large z, assuming rho_midplane is temporarily 1). This last
/// quantity is important if we know `Sigma(R)`, the surface density as a
/// function of cylindrical radius.
///
/// We can recast these 2 equations into a system of 3 coupled first-order
/// ordinary differential equations:
///
///     (d/dz) dPhiGasZ_dz             = alpha * rho_midplane *
///                                      exp(-1 * (PhiGasZ(z) + PhiOtherZ(z)) /
///                                          isoth_term)
///     (d/dz) PhiGasZ                 = d_PhiGasZ_dz`
///     (d/dz) posZ_unnormalized_Sigma = exp(-(PhiGasZ(z) + PhiOtherZ(z)) /
///                                          isoth_term)
///
/// With this in mind here is our procedure (again this is just for a single
/// cylindrical radius):
///   1. Start with an initial guess for rho_midplane
///   2. Use our guess for `rho_midplane` to integrate the system of 3
///      differential equations from z = 0 out to "large z." At z = 0:
///      `d_PhiGasZ_dz`, `PhiGasZ`, & `posZ_unnormalized_Sigma` are all 0.
///   3. Compute a new guess for `rho_midplane` from
///      `Sigma(R) / (2 * posZ_unnormalized_Sigma)`
///   4. If our new guess was "close enough" to our previous guess, we're done.
///      Otherwise, go back to step 2 using our newest guess.
template<typename OtherPhiFn>
class DerivFn{

private: // attributes
  Real alpha_;               ///!< 4 * pi * G
  Real isoth_term_;          ///!< see the explanation up above
  Real cur_R_;               ///!< The current cylindrical radius
  Real rho_midplane_;        ///!< Mass density at the midplane
  OtherPhiFn other_phi_fn_;  ///!< Specifies potential provided by the non-gas
                             ///!< galaxy component(s)
  Real other_phi_midplane_;  ///!< Caches the value of `other_phi_fn(0.0)`

public:
  DerivFn() = delete;

  DerivFn(Real isoth_term, Real cur_R,
          Real rho_midplane_guess, OtherPhiFn other_phi_fn) noexcept
    : alpha_(4.0 * M_PI * GN),
      isoth_term_(isoth_term),
      cur_R_(cur_R),
      rho_midplane_(rho_midplane_guess),
      other_phi_fn_(other_phi_fn),
      other_phi_midplane_(other_phi_fn(cur_R, 0.0))
  {}

  /// helper function that computes the gravitational potential (not
  /// contributed by the gas) measured with respect to the midplane
  Real calc_PhiOtherZ(Real z) const noexcept
  { return other_phi_fn_(cur_R_, z) - other_phi_midplane_; }

  /// Compute the actual derivative of the 3 quantities at the given `z` and
  /// current values.
  std::array<Real,3> operator()(Real z,
                                std::array<Real,3> cur_val) const noexcept
  {
    // compute the gravitational potential (with respect to the midplane) of
    // the non-gas component
    Real PhiOtherZ = calc_PhiOtherZ(z);
    Real exp_term =
      std::exp(-1.0 * (cur_val[LUT::PhiGasZ] + PhiOtherZ)/isoth_term_);

    std::array<Real,3> deriv;
    deriv[LUT::dPhiGasZ_dz]             = alpha_ * rho_midplane_ * exp_term;
    deriv[LUT::PhiGasZ]                 = cur_val[LUT::dPhiGasZ_dz];
    deriv[LUT::posZ_unnormalized_Sigma] = exp_term;
    return deriv;
  }
};

}// namespace selfgrav_hydrostatic_col


/// Identify the maximum distance above the gas disk (at the given cylindrical
/// radius) where we will consider initializing gas cells. This serves as the
/// upper-bound for all of the associated integrals.
///
/// We set this to the first cell-edge aligned value that satisfies:
///
///     calc_other_phi(z) >= (eta * isoth_term) + calc_other_phi(0)
///
/// We do this because:
///
///     rho(z)/rho_midplane = exp(-PhiZ(z)/isoth_term)
///
/// -> `rho_midplane` is the mass-density at z=0 (We don't need to know it yet,
///    to do this calculation)
/// -> `PhiZ(z)` is the TOTAL gravitational potential measured with respect
///    to the midplane.
///
/// Since `(calc_other_phi(z) - calc_other_phi(0))` is an underestimate for
/// `PhiZ(z)`, (it omits the gas disk's contribution to the potential), our
/// prescription for `zend` ensures that
///      rho(zend)/rho_midplane <= exp(-eta)
///
/// By default, We adopt eta = 7, since exp(-7) is approximately 1e-3
template <typename NonGasPhiFn>
Real find_zend_(Real cur_R, ZGridProps z_grid_props, Real isoth_term,
                NonGasPhiFn& calc_other_phi, Real eta = 7.0)
{
  assert(z_grid_props.origin_is_cell_edge_aligned());

  Real dz = z_grid_props.cell_width;
  if (calc_other_phi(cur_R,dz) >= ((2.303 * isoth_term) +
                                   calc_other_phi(cur_R,0)) ) {
    std::printf("WARNING: Simulation resolution is too coarse!\n"
                "-> At Rcyl = %e, the gas mass density drops by >= 90%% when "
                " increasing z by a cell-width of %e\n", cur_R, dz);
  }
  Real thresh = (eta * isoth_term) + calc_other_phi(cur_R, 0.0);

  Real z = 0.0;
  int current_offset = 0;
  while ((calc_other_phi(cur_R,z) < thresh) and (z < z_grid_props.global_max)) {
    current_offset++;
    z = current_offset * dz;
  }
  return std::fmin(z,z_grid_props.global_max);
}


template<typename NonGasPhiFn>
class SelfGravHydroStaticColMaker{
public:
  SelfGravHydroStaticColMaker() = delete;
  
  SelfGravHydroStaticColMaker(int ghost_depth, ZGridProps z_grid_props,
                              Real isoth_term, NonGasPhiFn calc_other_phi,
                              Real initial_scale_height_guess)
    : ghost_depth(ghost_depth), z_grid_props(z_grid_props), isoth_term(isoth_term),
      calc_other_phi(calc_other_phi),
      initial_scale_height_guess(initial_scale_height_guess)
  {}

  /* global total number of ghost zones along z-axis plus twice the ghost depth */
  int buffer_len() const noexcept {return this->z_grid_props.global_ncells + 2*ghost_depth; }
  
  /// Compute the vertical density column in hydrostatic equilibrium
  ///
  /// @param[in]  cur_R The current cylindrical radius
  /// @param[in]  cur_Sigma The current surface density
  /// @param[out] buffer is filled by this function. It is assumed to be an
  ///     array of length `this->buffer_len()`
  /// @returns best estimate for the midplane mass-density
  Real construct_col(Real cur_R, Real cur_Sigma, Real* buffer) const noexcept
  {
    for (int i = 0; i < this->ghost_depth; i++){
      buffer[i] = 0.0;
    }
    for (int i = this->z_grid_props.global_ncells + ghost_depth; i < this->buffer_len(); i++){
      buffer[i] = 0.0;
    }
    return this->construct_col_helper(cur_R, cur_Sigma, buffer + this->ghost_depth);
  }

private: // helper methods

  /// Does the heavy lifting of computing the hydrostatic vertical density profile
  ///
  /// @param[in]  cur_R The current cylindrical radius
  /// @param[in]  cur_Sigma The current surface density
  /// @param[out] buffer is filled by this function. It is assumed to be an
  ///     array of length `this->z_grid_props.global_ncells`. This is DIFFERENT from
  ///     `this->buffer_len()`.
  /// @returns best estimate for the midplane mass-density
  Real construct_col_helper(Real cur_R, Real cur_Sigma, Real* buffer) const noexcept;

private:

  int ghost_depth;
  ZGridProps z_grid_props;
  Real isoth_term;
  NonGasPhiFn calc_other_phi;
  Real initial_scale_height_guess;

};

template <typename NonGasPhiFn>
Real SelfGravHydroStaticColMaker<NonGasPhiFn>::construct_col_helper(Real cur_R, Real cur_Sigma, Real* buffer)
  const noexcept
{
  using MyDerivFn = selfgrav_hydrostatic_col::DerivFn<NonGasPhiFn>;

  //CHOLLA_ASSERT(cur_Sigma > 0, "Surface density must be positive");
  // STEP 0: come up with an initial guess for the midplane mass density.
  //  -> to do that, we assume mass is distributed in the vertical direction
  //     with an exponential distribution
  //  -> if we came up with something more sensible, our solution would
  //     converge faster
  Real initial_rho_midplane_guess = cur_Sigma / (2*initial_scale_height_guess);

  // STEP 1: identify max z, zend, that we will integrate up to
  const Real zstart = 0.0;
  const Real zend = find_zend_(cur_R, z_grid_props, isoth_term, calc_other_phi);

  // STEP 2: come up with the nominal step-size that we'll use throughout the
  //         rest of this function during integration
  const Real nominal_step = z_grid_props.cell_width / 10.0;

  // STEP 3: iteratively solve for density profile
  // -> each time we enter the loop, we integrate over the vertical density
  //    profile given the latest guess for the midplane density profile
  // -> During that integration, we effectively compute the unnormalized
  //    surface density for z>=0. We can use this to compute a new estimate
  //    for the midplane mass density.
  // -> If we are satisfied by the agreement between the estimates, then we're
  //    done! Otherwise, we re-enter the loop
  Real prev_rho_midplane_est = NAN;
  Real latest_rho_midplane_est = initial_rho_midplane_guess;
  Real est_abs_diff = NAN;
  const Real TOL = 1.0e-12; // may be smaller than necessary
  for (int i = 0; i < 100; i++) {

    // construct a new integrator instance configured with the latest estimate
    // for the midplane density
    MyDerivFn deriv_fn(isoth_term, cur_R, latest_rho_midplane_est,
                       calc_other_phi);
    const ODEIntegrator<3, MyDerivFn> integrator(deriv_fn);

    // the integrator integrates a vector yvec over independent variable x
    // -> the independent variable, is actually just the height above the disk
    Real xstart = zstart;
    Real xend = zend;

    // -> the entries of the vector are specified in
    //    selfgrav_hydrostatic_col::LUT
    // -> at z = 0, they all have values of 0.0
    std::array<Real,3> yvec_start = {0.0,0.0,0.0};

    // perform the integration:
    std::array<Real,3> yvec_end = integrator.integrate(xstart, xend,
                                                       nominal_step,
                                                       yvec_start);

    // we can now come up with a new estimate for the midplane density:
    prev_rho_midplane_est = latest_rho_midplane_est;
    latest_rho_midplane_est =
      (cur_Sigma /
       (2 * yvec_end[selfgrav_hydrostatic_col::LUT::posZ_unnormalized_Sigma]));

    est_abs_diff = std::fabs(prev_rho_midplane_est- latest_rho_midplane_est);
    //printf("R = %e, midplane density est, old: %.15e, new: %.15e, rdiff: %e\n",
    //       cur_R, prev_rho_midplane_est, latest_rho_midplane_est,
    //       est_abs_diff/prev_rho_midplane_est);

    if (est_abs_diff < fabs(TOL * prev_rho_midplane_est)) { break; }
  }

  // STEP 4: some error checking
  if ((not std::isfinite(est_abs_diff)) or (est_abs_diff >= fabs(TOL * prev_rho_midplane_est))) {
    CHOLLA_ERROR("vertical density profile unconverged at Rcyl = %g.\n"
                 "   rho_midplane estimate used to compute profile: %e\n"
                 "   rho_midplane estimate computed from profile: %e\n"
                 "   The relative difference is %e",
                cur_R, prev_rho_midplane_est, prev_rho_midplane_est,
                est_abs_diff/prev_rho_midplane_est);
  }

  // so I think it probably makes more sense to use prev_rho_midplane_est for
  // the rest of this since we know the integral using it produces a consistent
  // result (latest_rho_midplane_est is probably more accurate, but we don't
  // know for sure)
  const Real rho_midplane = prev_rho_midplane_est;

  // STEP 5: clear contents of buffer
  for (int i = 0; i < z_grid_props.global_ncells; i ++){
    buffer[i] = 0.0;
  }

  // STEP 6: selectively fill in the contents of buffer
  if (not z_grid_props.origin_is_cell_edge_aligned()) {
    exit(1);
  } else {
    // we are going to integrate over the density/gravitational profiles 1 cell
    // at a time.
    // -> As we do that, we'll use the following logger function to accumulate
    //    changes in selfgrav_hydrostatic_col::LUT::posZ_unnormalized_Sigma.
    //    We can use these changes to determine the average mass-density in the
    //    cell.

    Real inv_dz = 1.0 / z_grid_props.cell_width;
    Real accum = 0.0;

    auto log_fn = [&accum](Real z, Real cur_step, std::array<Real, 3> vec,
                           std::array<Real, 3> v_step)
    {accum += v_step[selfgrav_hydrostatic_col::LUT::posZ_unnormalized_Sigma];};


    MyDerivFn deriv_fn(isoth_term, cur_R, rho_midplane, calc_other_phi);
    const ODEIntegrator<3, MyDerivFn, decltype(log_fn)> integrator
      (deriv_fn, log_fn);

    // we start the integral at the midplane. The entries of the integrated
    // vector all start out with a value of 0.
    std::array<Real,3> cur_vec = {0.0,0.0,0.0};

    // Let's call each cell-width a "segment". We are going to iterate over
    // segments
    const int num_segments = z_grid_props.global_ncells / 2;
    for (int seg_ind = 0; seg_ind < num_segments; seg_ind++){
      // clear the accumulator variable:
      accum = 0.0;

      // determine z_start and z_end for current segment (it's important that
      // z_start of the current segment matches z_end of previous segment):
      Real seg_z_start = seg_ind * z_grid_props.cell_width;
      Real seg_z_end   = (seg_ind+1) * z_grid_props.cell_width;

      // actually perform the integral, updating cur_vec
      cur_vec = integrator.integrate
        (seg_z_start, std::fmin(seg_z_end, zend), nominal_step, cur_vec);

      // compute the average mass-density in the current segment
      Real avg_rho = rho_midplane * accum * inv_dz;

      //printf("%e, ", avg_rho);

      // set values above and below the disk:
      buffer[num_segments + seg_ind] = avg_rho;
      buffer[num_segments - (seg_ind+1)] = avg_rho;

      if (seg_z_end >= zend) { break; }
    }
    //printf("\n");

  }

  return rho_midplane;
}

#endif /* SELFGRAV_HYDROSTATIC_COL */
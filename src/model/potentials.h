#ifndef POTENTIALS
#define POTENTIALS

// this file contains objects representing (semi-)analytic gravitational potentials

#include <cmath>

#include "../global/global.h"
#include "../utils/error_handling.h"

struct NFWHaloPotential {
  Real M_h;   /*!< total halo mass in Msolar */
  Real R_h;   /*!< halo scale length (NOT the virial radius) */
  Real c_vir; /*!< halo concentration parameter (to account for adiabatic contraction) */

  /* function with logarithms used in NFW definitions */
  __host__ __device__ static Real log_func(Real y) { return log(1 + y) - y / (1 + y); };

  /* Cylindrical radial acceleration */
  Real gr_halo_D3D(Real R, Real z) const noexcept
  {
    Real r      = sqrt(R * R + z * z);  // spherical radius
    Real x      = r / R_h;
    Real r_comp = R / r;

    Real A = log_func(x);
    Real B = 1.0 / (r * r);
    Real C = GN * M_h / log_func(c_vir);

    return -C * A * B * r_comp;
  };

  /* vertical acceleration in NFW halo */
  Real gz_halo_D3D(Real R, Real z)
  {
    Real r      = sqrt(R * R + z * z);  // spherical radius
    Real x      = r / R_h;
    Real z_comp = z / r;

    Real A = log_func(x);
    Real B = 1.0 / (r * r);
    Real C = GN * M_h / log_func(c_vir);

    return -C * A * B * z_comp;  // checked with wolfram alpha
  }

  /* Mass density profile */
  __host__ __device__ Real rho_halo_D3D(Real R, Real z) const noexcept
  {
    // by equating eqn 2.67 from Binney and Tremaine with eqn 2 from
    // Schneider & Robertson (2018) -- these are alternative forms of Phi --
    // I find that the rho0 normalization is:
    Real rho0 = M_h / (4 * M_PI * (R_h * R_h * R_h) * log_func(c_vir));

    Real rdivRh = sqrt(R * R + z * z) / R_h;  // spherical radius divided by R_h

    Real rdivRh_p_1 = rdivRh + 1;

    // eqn 2.64 from Binney and Tremmaine:
    return rho0 / (rdivRh * (rdivRh_p_1 * rdivRh_p_1));
  }

  /* Potential of NFW halo */
  __host__ __device__ Real phi_halo_D3D(Real R, Real z) const noexcept
  {
    Real r = sqrt(R * R + z * z);  // spherical radius
    Real x = r / R_h;
    Real C = GN * M_h / (R_h * log_func(c_vir));

    // limit x to non-zero value
    if (x < 1.0e-9) {
      x = 1.0e-9;
    }

    return -C * log(1 + x) / x;
  };
};

/*!
 * Encapsulates properties of an analytic Miyamoto Nagai gravitational potential.
 *
 * This is commonly used to model the static potential of a stellar disk. Take
 * note that this is NOT the same as an exponential disk!
 */
struct MiyamotoNagaiPotential {
  Real M_d; /*!< total mass (in Msolar) */
  Real R_d; /*!< scale-length (in kpc) */
  Real Z_d; /*!< scale-height (in kpc). */

  /* Radial acceleration in miyamoto nagai */
  Real gr_disk_D3D(Real R, Real z) const noexcept
  {
    Real A = R_d + sqrt(Z_d * Z_d + z * z);
    Real B = pow(A * A + R * R, 1.5);

    return -GN * M_d * R / B;
  };

  // vertical acceleration in miyamoto nagai
  Real gz_disk_D3D(Real R, Real z) const noexcept
  {
    Real a = R_d;
    Real b = Z_d;
    Real A = sqrt(b * b + z * z);
    Real B = a + A;
    Real C = pow(B * B + R * R, 1.5);

    // checked with wolfram alpha
    return -GN * M_d * z * B / (A * C);
  }

  /* Miyamoto-Nagai potential */
  __host__ __device__ Real phi_disk_D3D(Real R, Real z) const noexcept
  {
    Real A = sqrt(z * z + Z_d * Z_d);
    Real B = R_d + A;
    Real C = sqrt(R * R + B * B);

    // patel et al. 2017, eqn 2
    return -GN * M_d / C;
  };

  /* Mass profile of disk */
  __host__ __device__ Real rho_disk_D3D(const Real r, const Real z) const noexcept
  {
    const Real a = R_d;
    const Real c = Z_d;
    const Real b = sqrt(z * z + c * c);
    const Real d = a + b;
    const Real s = r * r + d * d;
    return M_d * c * c * (a * (d * d + r * r) + 3.0 * b * d * d) / (4.0 * M_PI * b * b * b * pow(s, 2.5));

    /* version that was ripped out of Potential_Paris_Galactic::Get_Potential
     *
     * const Real rho0 = md * zd * zd / (4.0 * M_PI);
     * const Real a    = sqrt(z * z + Z_d * Z_d);
     * const Real b    = R_d + a;
     * const Real c    = r * r + b * b;
     * return rho0 * (rd * c + 3.0 * a * b * b) / (a * a * a * pow(c, 2.5));
     */
  }
};

/* Approximates the potential of a Exponential Disk as the sum of 3 MiyamotoNagaiPotential
 * disks.
 *
 * This uses the table from
 *   https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract
 * to determine the properties of each component
 */
struct AprroxExponentialDisk3MN {
  MiyamotoNagaiPotential comps[3];

  /* Returns a properly configured disk with
   *
   * The arguments determine what kind of disk we model:
   * - when `scale_height` is 0.0, we always model an infinitely thin disk
   * - when `scale_height>0` and `exponential_scaleheight` is `true`, the vertical
   *   density distribution exponentially decays as `exp(-fabs(z)/scale_height)`
   * - when `scale_height>0` and `exponential_scaleheight` is `false`, the vertical
   *   density distribution exponentially decays as `sech^2(-fabs(z)/scale_height)`
   *
   * \param[in] mass Total mass of the disk
   * \param[in] scale_length The desired radial exponential scale length (in code units)
   * \param[in] scale_height The desired scale_height of the disk (in code units)
   * \param[in] exponential_scaleheight Controls interpretation of scale-height
   */
  static AprroxExponentialDisk3MN create(Real mass, Real scale_length, Real scale_height, bool exponential_scaleheight)
  {
    if ((mass <= 0) or (scale_length <= 0)) CHOLLA_ERROR("invalid args");

    // step 1: determine the disk thickness parameter b (it's shared by all components)
    Real b_div_scale_length;
    if (scale_height < 0.0) {
      CHOLLA_ERROR("scale_height must be positive");
    } else if (scale_height == 0.0) {
      b_div_scale_length = 0.0;
    } else if (exponential_scaleheight) {
      Real x             = scale_height / scale_length;
      b_div_scale_length = (-0.269 * x + 1.080) * x + 1.092 * x;
    } else {
      Real x             = scale_height / scale_length;
      b_div_scale_length = (-0.033 * x + 0.262) * x + 0.659 * x;
    }
    Real b = b_div_scale_length * scale_length;

    if (b_div_scale_length > 3.0) CHOLLA_ERROR("The disk is too thick for this approx");

    // step 2 determine other parameters.
    // -> we use values from table 1 (although this potential technically
    //    corresponds to negative densities in the outer disk, that's probably
    //    fine for our purposes)
    const Real k[6][5] = {{-0.0090, 0.0640, -0.1653, 0.1164, 1.9487},    // M_MN0 / mass
                          {0.0173, -0.0903, 0.0877, 0.2029, -1.3077},    // M_MN1 / mass
                          {-0.0051, 0.0287, -0.0361, -0.0544, 0.2242},   // M_MN2 / mass
                          {-0.0358, 0.2610, -0.6987, -0.1193, 2.0074},   // a0 / scale_length
                          {-0.0830, 0.4992, -0.7967, -1.2966, 4.4441},   // a1 / scale_length
                          {-0.0247, 0.1718, -0.4124, -0.5944, 0.7333}};  // a2 / scale_length
    auto param         = [&k, b_div_scale_length](int i) {
      Real x = b_div_scale_length;
      return ((((k[i][0] * x + k[i][1]) * x + k[i][2]) * x) + k[i][3]) * x + k[i][4];
    };

    AprroxExponentialDisk3MN out;
    for (int j = 0; j < 3; j++) {
      out.comps[j] = MiyamotoNagaiPotential{param(j) * mass, param(j + 3) * scale_length, b};
    }
    return out;
  }

  /* Radial acceleration */
  Real gr_disk_D3D(Real R, Real z) const noexcept
  {
    return (comps[0].gr_disk_D3D(R, z) + comps[1].gr_disk_D3D(R, z) + comps[2].gr_disk_D3D(R, z));
  }

  /* vertical acceleration */
  Real gz_disk_D3D(Real R, Real z) const noexcept
  {
    return (comps[0].gz_disk_D3D(R, z) + comps[1].gz_disk_D3D(R, z) + comps[2].gz_disk_D3D(R, z));
  }

  /* computes the potential */
  __host__ __device__ Real phi_disk_D3D(Real R, Real z) const noexcept
  {
    return (comps[0].phi_disk_D3D(R, z) + comps[1].phi_disk_D3D(R, z) + comps[2].phi_disk_D3D(R, z));
  }

  /* computes the mass profile that corresponds to the potential
   *
   * \note
   * Technically, this may contain negative values. But, that's long as we are not using the results
   * to directly initialize a gas density profile. This is mostly useful in certain kinds of gravity
   * solvers.
   */
  __host__ __device__ Real rho_disk_D3D(Real R, Real z) const noexcept
  {
    return (comps[0].rho_disk_D3D(R, z) + comps[1].rho_disk_D3D(R, z) + comps[2].rho_disk_D3D(R, z));
  }
};

// It probably makes more sense for the following class to live in disk_ICs.h.
// - we currently put the definition here (rather than the in disk_ICs.h) since it holds
//   AprroxExponentialDisk3MN as an attribute
// - thus if we put this class in the disk_galaxy.h, we would also need to add an include
//   this header file to disk_galaxy.h. That produces issues for any regular .cpp file
//   that (directly or transitively includes) disk_galaxy.h

/* Aggregates properties related to a gas disk
 *
 * The radial surface-density distribution satisfies
 *   `Sigma(r) = Sigma_0 * exp(-r_cyl/R_d)
 */
struct GasDiskProps {
  Real M_d;        /*!< total mass (in Msolar) */
  Real R_d;        /*!< scale-length (in kpc) */
  Real H_d;        /*!< initial guess at the scale-height (in kpc) */
  Real T_d;        /*!< gas temperature */
  bool isothermal; /*!< Indicates whether to initialize an isothermal or adiabatic disk
                    *!< (it's unclear whether the adiabatic configuration still works)
                    */

  /* A rough approximation for the gravitational potential produced by self-gravity.
   * - It is generally used to help initialize the circular-velocity in the ICs.
   * - It is also employed while using the Paris-Galactic gravity solver. In this latter case, it's
   *   critical that this approximation is accurate at the domain boundaries (elsewhere, accuracy
   *   is entirely unimportant).
   */
  AprroxExponentialDisk3MN selfgrav_approx_potential;

  GasDiskProps(Real M_d, Real R_d, Real H_d, Real T_d, bool isothermal, Real selfgrav_scale_height_estimate)
      : M_d(M_d),
        R_d(R_d),
        H_d(H_d),
        T_d(T_d),
        isothermal(isothermal),
        selfgrav_approx_potential(AprroxExponentialDisk3MN::create(M_d, R_d, selfgrav_scale_height_estimate, true))
  {
  }

  /* Returns Sigma_0. This is just
   * \f$\Sigma_0 = \frac{M_d}{\int Sigma(r)\ dA} =  \frac{M_d}{2\pi \int_0^\infty r\ \Sigma\ dr} \f$
   */
  Real CentralSurfaceDensity() const noexcept { return M_d / (2 * M_PI * R_d * R_d); }

  /* Compute the surface density at cylindrical radius*/
  Real surface_density(Real R) const noexcept { return CentralSurfaceDensity() * exp(-R / R_d); };
};

#endif /* POTENTIALS */
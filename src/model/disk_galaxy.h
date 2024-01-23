#ifndef DISK_GALAXY
#define DISK_GALAXY

#define SIMULATED_FRACTION 0.1

#include <cmath>
#include <iostream>
#include <random>

#include "../global/global.h"
#include "../utils/error_handling.h"

/* Aggregates properties related to a disk
 *
 * The radial surface-density distribution satisfies
 *   `Sigma(r) = Sigma_0 * exp(-r_cyl/R_d)
 */
struct MiyamotoNagaiDiskProps{
  Real M_d;  /*!< total mass (in Msolar) */
  Real R_d;  /*!< scale-length (in kpc) */
  Real Z_d;  /*!< scale-height (in kpc). In the case of a gas disk, this is more
              *!< of an initial guess */

  /* Returns Sigma_0. This is just
   * \f$\Sigma_0 = \frac{M_d}{\int Sigma(r)\ dA} =  \frac{M_d}{2\pi \int_0^\infty r\ \Sigma\ dr} \f$
   */
  Real CentralSurfaceDensity() const noexcept {return M_d / (2 * M_PI * R_d * R_d);}

  /* Compute the surface density at cylindrical radius*/
  Real surface_density(Real R) const noexcept {return CentralSurfaceDensity() * exp(-R / R_d);};

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
    Real a   = R_d;
    Real b   = Z_d;
    Real A   = sqrt(b * b + z * z);
    Real B   = a + A;
    Real C   = pow(B * B + R * R, 1.5);

    // checked with wolfram alpha
    return -GN * M_d * z * B / (A * C);
  }

   /* Miyamoto-Nagai potential */
  Real phi_disk_D3D(Real R, Real z) const noexcept
  {
    Real A = sqrt(z * z + Z_d * Z_d);
    Real B = R_d + A;
    Real C = sqrt(R * R + B * B);

    // patel et al. 2017, eqn 2
    return -GN * M_d / C;
  };

  Real rho_disk_D3D(const Real r, const Real z) const noexcept
  {
    const Real a = R_d;
    const Real c = Z_d;
    const Real b = sqrt(z * z + c * c);
    const Real d = a + b;
    const Real s = r * r + d * d;
    return M_d * c * c * (a * (d * d + r * r) + 3.0 * b * d * d) / (4.0 * M_PI * b * b * b * pow(s, 2.5));
  }
};

/* Aggregates properties related to a gas disk
 *
 * The radial surface-density distribution satisfies
 *   `Sigma(r) = Sigma_0 * exp(-r_cyl/R_d)
 */
struct GasDiskProps{
  Real M_d;  /*!< total mass (in Msolar) */
  Real R_d;  /*!< scale-length (in kpc) */
  Real H_d;  /*!< initial guess at the scale-height (in kpc) */
  Real T_d;  /*!< gas temperature */
  bool isothermal; /*!< Indicates whether to initialize an isothermal or adiabatic disk
                    *!< (it's unclear whether the adiabatic configuration still works)
                    */

  /* Returns Sigma_0. This is just
   * \f$\Sigma_0 = \frac{M_d}{\int Sigma(r)\ dA} =  \frac{M_d}{2\pi \int_0^\infty r\ \Sigma\ dr} \f$
   */
  Real CentralSurfaceDensity() const noexcept {return M_d / (2 * M_PI * R_d * R_d);}

  /* Compute the surface density at cylindrical radius*/
  Real surface_density(Real R) const noexcept {return CentralSurfaceDensity() * exp(-R / R_d);};
};

/* Intended to serve as a centralized location where all properties of the underlying galaxy-model
 * are agregated.
 *
 * This object also defines some methods for computing an analytic gravitational potential. At this
 * time, that gravitational potential is only for the stellar disk and the background halo.
 */
class DiskGalaxy
{
 private:
  MiyamotoNagaiDiskProps stellar_disk;
  GasDiskProps gas_disk;
  Real M_vir, R_vir, c_vir, r_cool, M_h, R_h;
  Real log_func(Real y) { return log(1 + y) - y / (1 + y); };

 public:
  DiskGalaxy(MiyamotoNagaiDiskProps stellar_disk, GasDiskProps gas_disk,
             Real mvir, Real rvir, Real cvir, Real rcool)
    : stellar_disk(stellar_disk), gas_disk(gas_disk)
  {
    M_vir  = mvir;
    R_vir  = rvir;
    c_vir  = cvir;
    r_cool = rcool;
    M_h    = M_vir - stellar_disk.M_d;
    R_h    = R_vir / c_vir;
  };

  /* Radial acceleration in miyamoto nagai */
  Real gr_disk_D3D(Real R, Real z)
  {
    return stellar_disk.gr_disk_D3D(R, z);
  };

  /**
   *     Radial acceleration in NFW halo
   */
  Real gr_halo_D3D(Real R, Real z)
  {
    Real r      = sqrt(R * R + z * z);  // spherical radius
    Real x      = r / R_h;
    Real r_comp = R / r;

    Real A = log_func(x);
    Real B = 1.0 / (r * r);
    Real C = GN * M_h / log_func(c_vir);

    return -C * A * B * r_comp;
  };

  /**
   * Convenience method that returns the combined radial acceleration
   * of a disk galaxy at a specified point.
   * @param R the cylindrical radius at the desired point
   * @param z the distance perpendicular to the plane of the disk of the desired
   * point
   * @return
   */
  Real gr_total_D3D(Real R, Real z) { return gr_disk_D3D(R, z) + gr_halo_D3D(R, z); };

  /**
   *    Potential of NFW halo
   */
  Real phi_halo_D3D(Real R, Real z)
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

  /* Miyamoto-Nagai potential */
  Real phi_disk_D3D(Real R, Real z)
  {
    return stellar_disk.phi_disk_D3D(R, z);
  };

  Real rho_disk_D3D(const Real r, const Real z)
  {
    return stellar_disk.rho_disk_D3D(r, z);
  }

  /**
   *  Convenience method that returns the combined gravitational potential
   *  of the disk and halo.
   */
  Real phi_total_D3D(Real R, Real z) { return phi_halo_D3D(R, z) + phi_disk_D3D(R, z); };

  /**
   * epicyclic frequency
   */
  Real kappa2(Real R, Real z)
  {
    const Real R_d = stellar_disk.R_d;
    const Real M_d = stellar_disk.M_d;
    const Real Z_d = stellar_disk.Z_d;

    Real r = sqrt(R * R + z * z);
    Real x = r / R_h;
    Real C = GN * M_h / (R_h * log_func(c_vir));
    Real A = R_d + sqrt(z * z + Z_d * Z_d);
    Real B = sqrt(R * R + A * A);

    Real phiH_prime = -C * R / (r * r) / (1 + x) + C * log(1 + x) * R_h * R / (r * r * r) + GN * M_d * R / (B * B * B);
    Real phiH_prime_prime = -C / (r * r) / (1 + x) + 2 * C * R * R / (r * r * r * r) / (1 + x) +
                            C / ((1 + x) * (1 + x)) * R * R / R_h / (r * r * r) +
                            C * R * R / (1 + x) / (r * r * r * r) +
                            C * log(1 + x) * R_h / (r * r * r) * (1 - 3 * R * R / (r * r)) +
                            GN * M_d / (B * B * B) * (1 - 3 * R * R / (B * B));

    return 3 / R * phiH_prime + phiH_prime_prime;
  };

  Real sigma_crit(Real R)
  {
    return 3.36 * GN * stellar_disk.surface_density(R) / sqrt(kappa2(R, 0.0));
  };

  Real getM_d() const { return stellar_disk.M_d; };
  Real getR_d() const { return stellar_disk.R_d; };
  Real getZ_d() const { return stellar_disk.Z_d; };
  MiyamotoNagaiDiskProps getStellarDisk() const { return stellar_disk; };
  GasDiskProps getGasDisk() const { return gas_disk; };
  Real getM_vir() const { return M_vir; };
  Real getR_vir() const { return R_vir; };
  Real getC_vir() const { return c_vir; };
  Real getR_cool() const { return r_cool; };
};

/* Encapsulates the cluster-mass distribution function
 *
 * There is 0 probability of drawing a cluster with mass M < lower_mass or M >= higher_mass.
 * The probability of drawing a cluster mass M, satisfying lower_mass <= M < higher_mass is given
 * by a pdf that is proportional to std::pow(M, -1 * alpha);
 * -> alpha is not allowed to be 1 (that corresponds to the log-uniform distribution)
 * -> when you have N particles and alpha = 2, then the total mass particles in equal sized
 *    logarithmic mass bins is constant
 */
class ClusterMassDistribution{

public: // interface
  ClusterMassDistribution() = delete;

  ClusterMassDistribution(Real lower_mass, Real higher_mass, Real alpha)
  : lo_mass_(lower_mass), hi_mass_(higher_mass), alpha_(alpha)
  {
    CHOLLA_ASSERT(lower_mass > 0.0, "The minimum cluster-mass must exceed 0");
    CHOLLA_ASSERT(higher_mass > lower_mass, "The max mass must exceed the min mass");
    CHOLLA_ASSERT(alpha_ > 1.0, "alpha must exceed 1.0");
  }

  Real getLowerClusterMass() const { return lo_mass_; }
  Real getHigherClusterMass() const { return hi_mass_; }

  Real meanClusterMass() const {
    Real normalization = (1-alpha_) / (std::pow(hi_mass_, 1-alpha_) - std::pow(lo_mass_, 1-alpha_));
    if (alpha_ == 2.0) {
      return normalization * std::log(hi_mass_/lo_mass_);
    } else {
      CHOLLA_ERROR("UNTESTED LOGIC");
      return normalization * (std::pow(hi_mass_, 2-alpha_) - 
                              std::pow(lo_mass_, 2-alpha_)) / (2-alpha_);
    }
  }

  Real singleClusterMass(std::mt19937_64 generator) const
  {
    std::uniform_real_distribution<Real> uniform_distro(0, 1);
    Real X = uniform_distro(generator);
    Real mclmin = lo_mass_;
    Real mclmax = hi_mass_;

    Real tmp = std::pow(mclmin, -alpha_+1) - (std::pow(mclmin, -alpha_+1) - 
                                              std::pow(mclmax, -alpha_+1))*X;
    return std::pow(tmp, 1.0/(-alpha_+1));
  }

private: // attributes
  Real lo_mass_;
  Real hi_mass_;
  Real alpha_;
};

// TODO: consider doing away with the ClusteredDiskGalaxy class and instead storing
//       cluster_mass_distribution_ as an optional attribute of DiskGalaxy
class ClusteredDiskGalaxy : public DiskGalaxy
{
 private:
  ClusterMassDistribution cluster_mass_distribution_;

 public:
  ClusteredDiskGalaxy(ClusterMassDistribution cluster_mass_distribution,
                      MiyamotoNagaiDiskProps stellar_disk, GasDiskProps gas_disk,
                      Real mvir, Real rvir, Real cvir, Real rcool)
      : DiskGalaxy{stellar_disk, gas_disk, mvir, rvir, cvir, rcool},
        cluster_mass_distribution_(cluster_mass_distribution)
  { }

  ClusterMassDistribution getClusterMassDistribution() const {
    // we should return a CONST reference or a copy (so that the internal object isn't mutated)
    return cluster_mass_distribution_;
  }
};

// in the future, it may be better to make the following 2 choices more configurable
inline Real Get_StarCluster_Truncation_Radius(const parameters& p)
{
  if ((20.4 < p.xlen) and (p.xlen < 20.5)) return 9.5;
  return p.xlen / 2.0 - 0.2;
}

inline Real Get_Gas_Truncation_Radius(const parameters& p)
{
  if ((20.4 < p.xlen) and (p.xlen < 20.5)) return 9.9;
  return p.xlen / 2.0 - 0.1;
}


namespace Galaxies
{
// all masses in M_sun and all distances in kpc
static ClusteredDiskGalaxy MW(ClusterMassDistribution{1e2, 5e5, 2.0},
                              MiyamotoNagaiDiskProps{6.5e10, 2.7, 0.7}, // stellar_disk
                              GasDiskProps{0.15 * 6.5e10, 2*2.7, 0.7, 1e4, true}, // gas_disk
                              1.077e12, 261, 18, 157.0);
static DiskGalaxy M82(MiyamotoNagaiDiskProps{1.0e10, 0.8, 0.15}, // stellar_disk
                      GasDiskProps{0.25 * 1.0e10, 2*0.8, 0.15, 1e4, true}, // gas_disk
                      5.0e10, 0.8 / 0.015, 10, 100.0);
};  // namespace Galaxies

#endif  // DISK_GALAXY

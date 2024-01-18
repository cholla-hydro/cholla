#ifndef DISK_GALAXY
#define DISK_GALAXY

#define SIMULATED_FRACTION 0.1

#include <cmath>
#include <iostream>
#include <random>

#include "../global/global.h"
#include "../utils/error_handling.h"

class DiskGalaxy
{
 private:
  Real M_vir, M_d, R_d, Z_d, R_vir, c_vir, r_cool, M_h, R_h;
  Real log_func(Real y) { return log(1 + y) - y / (1 + y); };

 public:
  DiskGalaxy(Real md, Real rd, Real zd, Real mvir, Real rvir, Real cvir, Real rcool)
  {
    M_d    = md;
    R_d    = rd;
    Z_d    = zd;
    M_vir  = mvir;
    R_vir  = rvir;
    c_vir  = cvir;
    r_cool = rcool;
    M_h    = M_vir - M_d;
    R_h    = R_vir / c_vir;
  };

  /**
   *     Radial acceleration in miyamoto nagai
   */
  Real gr_disk_D3D(Real R, Real z)
  {
    Real A = R_d + sqrt(Z_d * Z_d + z * z);
    Real B = pow(A * A + R * R, 1.5);

    return -GN * M_d * R / B;
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

  /**
   *  Miyamoto-Nagai potential
   */
  Real phi_disk_D3D(Real R, Real z)
  {
    Real A = sqrt(z * z + Z_d * Z_d);
    Real B = R_d + A;
    Real C = sqrt(R * R + B * B);

    // patel et al. 2017, eqn 2
    return -GN * M_d / C;
  };

  Real rho_disk_D3D(const Real r, const Real z)
  {
    const Real a = R_d;
    const Real c = Z_d;
    const Real b = sqrt(z * z + c * c);
    const Real d = a + b;
    const Real s = r * r + d * d;
    return M_d * c * c * (a * (d * d + r * r) + 3.0 * b * d * d) / (4.0 * M_PI * b * b * b * pow(s, 2.5));
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

  Real surface_density(Real R) { return M_d / (2 * M_PI) / (R_d * R_d) * exp(-R / R_d); };

  Real sigma_crit(Real R) { return 3.36 * GN * surface_density(R) / sqrt(kappa2(R, 0.0)); };

  Real getM_d() const { return M_d; };
  Real getR_d() const { return R_d; };
  Real getZ_d() const { return Z_d; };
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
                      Real md, Real rd, Real zd, Real mvir, Real rvir, Real cvir, Real rcool)
      : DiskGalaxy{md, rd, zd, mvir, rvir, cvir, rcool}, cluster_mass_distribution_(cluster_mass_distribution)
  { }

  ClusterMassDistribution getClusterMassDistribution() const {
    // we should return a CONST reference or a copy (so that the internal object isn't mutated)
    return cluster_mass_distribution_;
  }
};

namespace Galaxies
{
// all masses in M_sun and all distances in kpc
static ClusteredDiskGalaxy MW(ClusterMassDistribution{1e2, 5e5, 2.0},
                              6.5e10, 2.7, 0.7, 1.077e12, 261, 18, 157.0);
static DiskGalaxy M82(1.0e10, 0.8, 0.15, 5.0e10, 0.8 / 0.015, 10, 100.0);
};  // namespace Galaxies

#endif  // DISK_GALAXY

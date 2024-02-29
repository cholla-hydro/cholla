#ifndef DISK_GALAXY
#define DISK_GALAXY

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "../global/global.h"
#include "../utils/error_handling.h"

// we are bending over backwards to ensure that the functionality defined in
// "potentials.h" can be used on CPUs and on GPUs
// -> previously, this functionality was simply duplicated in a number of places
//    and this created some headaches
// -> we can't simply include the "potentials.h" header here since that would
//    produce problems for any source file that includes this header and makes
//    use of a regular c++ compiler
// -> we instead forward declare the relevant classes/structs from "potentials.h"
//    and store pointers to these classes within DiskGalaxy (without including
//    the full definitions of these classes in this header, we can't directly
//    include them in this file)
// -> this means that DiskGalaxy's methods that forward onto the methods of these
//    other classes will be a little slower. In files compiled with a CUDA/HIP
//    compiler, this slowdown can be avoided by including the "potentials.h" 
//    header and using DiskGalaxy's accessor methods to directly access the
//    underlying objects.
struct NFWHaloPotential;
struct MiyamotoNagaiDiskProps;
struct GasDiskProps;

/* Intended to serve as a centralized location where all properties of the underlying galaxy-model
 * are agregated.
 *
 * This object also defines some methods for computing an analytic gravitational potential. At this
 * time, that gravitational potential is only for the stellar disk and the background halo.
 */
class DiskGalaxy
{
private:
  // we store pointers to stellar_disk, gas_disk, and halo_potential purely to
  // sidestep some compilation issues with non-CUDA/HIP source files including 
  // this file (this is described in greater depth up above)
  std::shared_ptr<MiyamotoNagaiDiskProps> stellar_disk;
  std::shared_ptr<GasDiskProps> gas_disk;
  std::shared_ptr<NFWHaloPotential> halo_potential;
  Real M_vir, R_vir, r_cool;

public:

  /* To properly deallocate the internally tracked shared pointers we need to define a
   * destructor in a source file where the full definitions of the referenced classes
   * are visible.
   *
   * \note
   * we need to declare this as virtual so code like `delete ptr;`, where `ptr` has
   * type `DiskGalaxy*`, but references a subclass is executed properly (if the virtual
   * specifier were missing the code snippet would invoke undefined behavior.
   */
  virtual ~DiskGalaxy();

  DiskGalaxy(const MiyamotoNagaiDiskProps& stellar_disk,
             const GasDiskProps& gas_disk,
             Real mvir, Real rvir, Real cvir, Real rcool);

  /* Radial acceleration in miyamoto nagai */
  Real gr_disk_D3D(Real R, Real z) const noexcept;

  /* Radial acceleration in NFW halo */
  Real gr_halo_D3D(Real R, Real z) const noexcept;

  /**
   * Convenience method that returns the combined radial acceleration
   * of a disk galaxy at a specified point.
   * @param R the cylindrical radius at the desired point
   * @param z the distance perpendicular to the plane of the disk of the desired
   * point
   * @return
   */
  Real gr_total_D3D(Real R, Real z) const noexcept { return gr_disk_D3D(R, z) + gr_halo_D3D(R, z); };

  Real gr_total_with_GasSelfGravEstimate(Real R, Real z) const noexcept;

  /* returns the circular velocity of a massless test particle in the static gravitational
   * potential at the specified (cylindrical radius, z) pair. */
  Real circular_vel2(Real R, Real z) const noexcept
  {
    return R * std::fabs(gr_total_D3D(R, z));
  }

  /* returns the circular velocity of a massless test particle in the gravitational
   * potential (including an estimate for self-gravity) at the specified 
   * (cylindrical radius, z) pair. */
  Real circular_vel2_with_selfgrav_estimates(Real R, Real z) const noexcept
  {
    return R * std::fabs(gr_total_with_GasSelfGravEstimate(R, z));
  }

  /* Potential of NFW halo */
  Real phi_halo_D3D(Real R, Real z) const noexcept;

  /* Miyamoto-Nagai potential */
  Real phi_disk_D3D(Real R, Real z) const noexcept;

  /**
   *  Convenience method that returns the combined gravitational potential
   *  of the disk and halo.
   */
  Real phi_total_D3D(Real R, Real z) const noexcept { return phi_halo_D3D(R, z) + phi_disk_D3D(R, z); };

  /**
   * epicyclic frequency
   */
  Real kappa2(Real R, Real z) const;

  //Real sigma_crit(Real R)
  //{
  //  return 3.36 * GN * stellar_disk.surface_density(R) / sqrt(kappa2(R, 0.0));
  //};

  Real getM_d() const;
  Real getR_d() const;
  Real getZ_d() const;
  Real getGasDiskR_d() const;
  const MiyamotoNagaiDiskProps& getStellarDisk() const;
  const GasDiskProps& getGasDisk() const;
  const NFWHaloPotential& getHaloPotential() const;
  Real getM_vir() const { return M_vir; };
  Real getR_vir() const { return R_vir; };
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
                      const MiyamotoNagaiDiskProps& stellar_disk,
                      const GasDiskProps& gas_disk,
                      Real mvir, Real rvir, Real cvir, Real rcool)
      : DiskGalaxy{stellar_disk, gas_disk, mvir, rvir, cvir, rcool},
        cluster_mass_distribution_(cluster_mass_distribution)
  { }

  ClusterMassDistribution getClusterMassDistribution() const {
    // we should return a CONST reference or a copy (so that the internal object isn't mutated)
    return cluster_mass_distribution_;
  }
};

// in the future, it may be better to make the following 2 choices more configurable (and maybe
// store them inside the Galaxy object)
inline Real Get_StarCluster_Truncation_Radius(const Parameters& p)
{
  if ((20.4 < p.xlen) and (p.xlen < 20.5)) return 9.5;
  return p.xlen / 2.0 - 0.2;
}

inline Real Get_Gas_Truncation_Radius(const Parameters& p)
{
  if ((20.4 < p.xlen) and (p.xlen < 20.5)) return 9.9;
  return p.xlen / 2.0 - 0.1;
}


// Forward declare galaxy instances. These are defined in disk_galaxy.cu
namespace galaxies
{
 extern const ClusteredDiskGalaxy MW;
 extern const DiskGalaxy M82;
};   // namespace Galaxies

#endif  // DISK_GALAXY

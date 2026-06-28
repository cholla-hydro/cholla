#include "../../io/ParameterMap.h"
#include "disk_galaxy.h"
#include "potentials.h"

ClusterMassDistribution::ClusterMassDistribution(ParameterMap& pmap)
    : ClusterMassDistribution(pmap.value<double>("model.galaxy.cluster_mass_dist.lo_Msun"),
                              pmap.value<double>("model.galaxy.cluster_mass_dist.hi_Msun"),
                              pmap.value<double>("model.galaxy.cluster_mass_dist.alpha"))
{
}

// this is empty since the std::shared_ptr automatically handles things
// (but we need to explicitly handle the case since the definitions of the wrapped
//  classes aren't available when define DiskGalaxy)
DiskGalaxy::~DiskGalaxy() {}

DiskGalaxy::DiskGalaxy(ParameterMap& pmap)
    : stellar_disk(new MiyamotoNagaiPotential(pmap)),
      gas_disk(new GasDiskProps(pmap)),
      halo_potential(new NFWHaloPotential(pmap)),
      cluster_mass_distribution_{nullptr},
      r_cool(pmap.value<double>("model.galaxy.initial_cool_radius_kpc"))
{
  CHOLLA_ASSERT(r_cool > 0, "initial_cool_radius must be positive: %g", r_cool);
  if (pmap.Contains_Table("model.galaxy.cluster_mass_dist")) {
    cluster_mass_distribution_ = std::make_shared<ClusterMassDistribution>(pmap);
  }
}

Real DiskGalaxy::gr_disk_D3D(Real R, Real z) const noexcept { return stellar_disk->gr_disk_D3D(R, z); }

/* Radial acceleration in NFW halo */
Real DiskGalaxy::gr_halo_D3D(Real R, Real z) const noexcept { return halo_potential->gr_halo_D3D(R, z); }

Real DiskGalaxy::gr_total_with_GasSelfGravEstimate(Real R, Real z) const noexcept
{
  return gas_disk->selfgrav_approx_potential.gr_disk_D3D(R, z) + gr_total_D3D(R, z);
}

Real DiskGalaxy::phi_halo_D3D(Real R, Real z) const noexcept { return halo_potential->phi_halo_D3D(R, z); }

Real DiskGalaxy::phi_disk_D3D(Real R, Real z) const noexcept { return stellar_disk->phi_disk_D3D(R, z); }

Real DiskGalaxy::kappa2(Real R, Real z) const
{
  const Real R_d = stellar_disk->R_d;
  const Real M_d = stellar_disk->M_d;
  const Real Z_d = stellar_disk->Z_d;
  const Real M_h = halo_potential->M_h;
  const Real R_h = halo_potential->R_h;

  Real r = sqrt(R * R + z * z);
  Real x = r / R_h;
  Real C = GN * M_h / (R_h * NFWHaloPotential::log_func(halo_potential->c_vir));
  Real A = R_d + sqrt(z * z + Z_d * Z_d);
  Real B = sqrt(R * R + A * A);

  Real phiH_prime = -C * R / (r * r) / (1 + x) + C * log(1 + x) * R_h * R / (r * r * r) + GN * M_d * R / (B * B * B);
  Real phiH_prime_prime = -C / (r * r) / (1 + x) + 2 * C * R * R / (r * r * r * r) / (1 + x) +
                          C / ((1 + x) * (1 + x)) * R * R / R_h / (r * r * r) + C * R * R / (1 + x) / (r * r * r * r) +
                          C * log(1 + x) * R_h / (r * r * r) * (1 - 3 * R * R / (r * r)) +
                          GN * M_d / (B * B * B) * (1 - 3 * R * R / (B * B));

  return 3 / R * phiH_prime + phiH_prime_prime;
}

Real DiskGalaxy::getM_d() const { return stellar_disk->M_d; };
Real DiskGalaxy::getR_d() const { return stellar_disk->R_d; };
Real DiskGalaxy::getZ_d() const { return stellar_disk->Z_d; };
Real DiskGalaxy::getGasDiskR_d() const { return gas_disk->R_d; };
const MiyamotoNagaiPotential& DiskGalaxy::getStaticStellarDiskPotential() const { return *stellar_disk; };
const GasDiskProps& DiskGalaxy::getGasDisk() const { return *gas_disk; };
const NFWHaloPotential& DiskGalaxy::getHaloPotential() const { return *halo_potential; }

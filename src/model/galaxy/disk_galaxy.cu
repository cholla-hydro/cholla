#include "../../io/ParameterMap.h"
#include "disk_galaxy.h"
#include "gas_props.h"
#include "potentials.h"

ClusterMassDistribution::ClusterMassDistribution(ParameterMap& pmap)
    : ClusterMassDistribution(pmap.value<double>("model.galaxy.star_forming_disk.cluster_mass_dist.lo_Msun"),
                              pmap.value<double>("model.galaxy.star_forming_disk.cluster_mass_dist.hi_Msun"),
                              pmap.value<double>("model.galaxy.star_forming_disk.cluster_mass_dist.alpha"))
{
}

StarFormingDiskProps::StarFormingDiskProps(ParameterMap& pmap)
    : cluster_mass_distribution(pmap),
      global_sfr_Msun_per_kyr{pmap.value<double>("model.galaxy.star_forming_disk.global_sfr_Msun_per_kyr")},
      poisson_point_process{pmap.value<bool>("model.galaxy.star_forming_disk.poisson_point_process")},
      kennicut_schmidt_power{pmap.value_or("model.galaxy.star_forming_disk.kennicut_schmidt_power", 1.4)},
      earliest_t_formation{pmap.value<double>("model.galaxy.star_forming_disk.earliest_t_formation")}
{
  CHOLLA_ASSERT(global_sfr_Msun_per_kyr >= 0.0, "global_sfr_Msun_per_kyr must be non-negative: %g",
                global_sfr_Msun_per_kyr);

  std::string latest_t_formation_param_name = "model.galaxy.star_forming_disk.latest_t_formation";
  if (pmap.has_param(latest_t_formation_param_name)) {
    latest_t_formation = std::make_optional<double>(pmap.value<double>(latest_t_formation_param_name));
  }
}

// this is empty since the std::shared_ptr automatically handles things
// (but we need to explicitly handle the case since the definitions of the wrapped
//  classes aren't available when define DiskGalaxy)
DiskGalaxy::~DiskGalaxy() {}

DiskGalaxy::DiskGalaxy(ParameterMap& pmap)
    : stellar_disk(new MiyamotoNagaiPotential(pmap)),
      gas_disk(new GasDiskProps(pmap)),
      halo_potential(new NFWHaloPotential(pmap)),
      initial_cgm_props(new galaxy_detail::InitialCGMProps(pmap)),
      star_forming_disk_props_{nullptr}
{
  if (pmap.Contains_Table("model.galaxy.star_forming_disk")) {
    star_forming_disk_props_ = std::make_shared<StarFormingDiskProps>(pmap);
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

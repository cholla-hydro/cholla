#include "disk_galaxy.h"
#include "../model/potentials.h"

// Here we actually define the galaxy models that can be accessed from elsewhere

// all masses in M_sun and all distances in kpc

// For the MilkyWay model, we adopt radial scale lengths of 2.5 kpc and 3.5 kpc for
// the stellar and gas disks, respectively. If the newly formed stars follow the
// Kennicut-Schmidt law with a power of 1.4, the newly formed stars will organize
// into a disk with scale-length of 2.5 kpc
const ClusteredDiskGalaxy galaxies::MW(ClusterMassDistribution{1e2, 5e5, 2.0},
                                       MiyamotoNagaiDiskProps{6.5e10, 2.5, 0.7}, // stellar_disk
                                       GasDiskProps{0.15 * 6.5e10, 3.5, 0.7, 1e4, true, 0.02}, // gas_disk
                                       1.077e12, 261, 18, 157.0);
const DiskGalaxy galaxies::M82(MiyamotoNagaiDiskProps{1.0e10, 0.8, 0.15}, // stellar_disk
                               GasDiskProps{0.25 * 1.0e10, 2*0.8, 0.15, 1e4, true, 2*0.8}, // gas_disk
                               5.0e10, 0.8 / 0.015, 10, 100.0);



// here we define the methods

// this is empty since the std::shared_ptr automatically handles things
// (but we need to explicitly handle the case since the definitions of the wrapped 
//  classes aren't available when define DiskGalaxy)
DiskGalaxy::~DiskGalaxy() {}

DiskGalaxy::DiskGalaxy(const MiyamotoNagaiDiskProps& stellar_disk,
                       const GasDiskProps& gas_disk,
                       Real mvir, Real rvir, Real cvir, Real rcool)
  : stellar_disk(new MiyamotoNagaiDiskProps(stellar_disk)),
    gas_disk(new GasDiskProps(gas_disk)),
    halo_potential(new NFWHaloPotential{/* halo mass: */ mvir - stellar_disk.M_d,
                                        /* scale length:*/ (rvir / cvir), cvir})
{
  M_vir  = mvir;
  R_vir  = rvir;
  r_cool = rcool;
}

Real DiskGalaxy::gr_disk_D3D(Real R, Real z) const noexcept
{
  return stellar_disk->gr_disk_D3D(R, z);
}

/* Radial acceleration in NFW halo */
Real DiskGalaxy::gr_halo_D3D(Real R, Real z) const noexcept
{
  return halo_potential->gr_halo_D3D(R, z);
}

Real DiskGalaxy::gr_total_with_GasSelfGravEstimate(Real R, Real z) const noexcept
{
  return gas_disk->approx_selfgrav_for_vcirc.gr_disk_D3D(R,z) + gr_total_D3D(R,z);
}

Real DiskGalaxy::phi_halo_D3D(Real R, Real z) const noexcept
{
  return halo_potential->phi_halo_D3D(R, z);
}

Real DiskGalaxy::phi_disk_D3D(Real R, Real z) const noexcept
{
  return stellar_disk->phi_disk_D3D(R, z);
}

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
                          C / ((1 + x) * (1 + x)) * R * R / R_h / (r * r * r) +
                          C * R * R / (1 + x) / (r * r * r * r) +
                          C * log(1 + x) * R_h / (r * r * r) * (1 - 3 * R * R / (r * r)) +
                          GN * M_d / (B * B * B) * (1 - 3 * R * R / (B * B));

  return 3 / R * phiH_prime + phiH_prime_prime;
}

Real DiskGalaxy::getM_d() const { return stellar_disk->M_d; };
Real DiskGalaxy::getR_d() const { return stellar_disk->R_d; };
Real DiskGalaxy::getZ_d() const { return stellar_disk->Z_d; };
const MiyamotoNagaiDiskProps& DiskGalaxy::getStellarDisk() const { return *stellar_disk; };
const GasDiskProps& DiskGalaxy::getGasDisk() const { return *gas_disk; };
const NFWHaloPotential& DiskGalaxy::getHaloPotential() const { return *halo_potential; }

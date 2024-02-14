/*! \file disk_ICs.cpp
 *  \brief Definitions of initial conditions for hydrostatic disks.
           Note that the grid is mapped to 1D as i + (x_dim)*j +
 (x_dim*y_dim)*k. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cmath>
#include <vector>

#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../mpi/mpi_routines.h"
#include "../utils/error_handling.h"
#include "disk_galaxy.h"
#include "selfgrav_hydrostatic_col.h"

/* Originally, we passed around a 22-element array of parameters. Mapping the 
 * index to the parameter-name made the code harder to follow. The following
 * is introduced to replace that array
 */
struct DataPack{
  MiyamotoNagaiDiskProps stellar_disk;
  NFWHaloPotential halo_potential;
  Real T_d;
  Real Sigma_0;
  Real R_g;
  Real H_g;
  Real K_eos;
  Real gamma;      // adiabatic index
  Real rho_floor;
  Real rho_eos;
  Real cs;

  // these were defined somewhat separately
  Real K_eos_h;
  Real rho_eos_h;
  Real cs_h;
  Real r_cool;    // cooling radius
  Real Rgas_truncation_radius;

};

// radial acceleration in NFW halo
Real gr_halo_D3D(Real R, Real z, const DataPack& hdp)
{
  return hdp.halo_potential.gr_halo_D3D(R, z);
}

/* Compute the standard logistic function f(x) = 1.0 / (1.0 + exp(-x))
 *
 * This is sometimes called "the sigmoid" or "expit". 
 */
static Real standard_logistic_function(Real x) noexcept {
  // we make use of tanh since it is probably better behaved than directly
  // computing 1.0 /(1.0 + std::exp(-x))
  return 0.5 + 0.5 * std::tanh(0.5*x);
}

// disk radial surface density profile
Real Sigma_disk_D3D(Real r, const DataPack& hdp)
{
  // return the exponential surface density
  Real Sigma_0 = hdp.Sigma_0; // surface density at the center of the disk
  Real R_g     = hdp.R_g;
  Real Sigma = Sigma_0 * exp(-r / R_g);

  // taper the edge of the disk to 0
  Real R_c     = hdp.Rgas_truncation_radius;
  Real taper_factor = 1.0 - standard_logistic_function((r - R_c) / 0.005);

  // force surface density to 0 when taper_factor drops below 1e-6
  // -> this is a crude hack to limit how far we are setting the circular velocity
  //    outside of R_c (at the time of writing this, we set the circular velocity
  //    everywhere that the density from the disk exceeds 0)
  // -> we explain down below (where we initialize azimuthal velocity) why this is
  //    necessary
  if (taper_factor < 1e-6) taper_factor = 0.0;

  return Sigma*taper_factor;
}

// NFW halo potential
Real phi_halo_D3D(Real R, Real z, const DataPack& hdp)
{
  return hdp.halo_potential.phi_halo_D3D(R, z);
}

// total potential
Real phi_total_D3D(Real R, Real z, const DataPack& hdp)
{
  Real Phi_A = phi_halo_D3D(R, z, hdp);
  Real Phi_B = hdp.stellar_disk.phi_disk_D3D(R, z);
  return Phi_A + Phi_B;
}

Real phi_hot_halo_D3D(Real r, const DataPack& hdp)
{
  Real Phi_A = phi_halo_D3D(0, r, hdp);
  Real Phi_B = hdp.stellar_disk.phi_disk_D3D(0, r);
  // return Phi_A;
  return Phi_A + Phi_B;
}

// returns the cell-centered vertical
// location of the cell with index k
// k is indexed at 0 at the lowest ghost cell
Real z_hc_D3D(int k, Real dz, int nz, int ng)
{
  // checked that this works, such that the
  // if dz = L_z/nz for the real domain, then the z positions
  // are set correctly for cell centers with nz spanning
  // the real domain, and nz + 2*ng spanning the real + ghost domains
  if (!(nz % 2)) {
    // even # of cells
    return 0.5 * dz + ((Real)(k - ng - (int)(nz / 2))) * dz;
  } else {
    // odd # of cells
    return ((Real)(k - ng - (int)((nz - 1) / 2))) * dz;
  }
}

// returns the cell-centered radial
// location of the cell with index i
Real r_hc_D3D(int i, Real dr)
{
  // the zeroth cell is centered at 0.5*dr
  return 0.5 * dr + ((Real)i) * dr;
}

/*! \fn void hydrostatic_ray_analytical_D3D(Real *rho, Real *r, const DataPack& hdp, Real
 dr, int nr)
 *  \brief Calculate the density at spherical radius r due to a hydrostatic
 halo. Uses an analytic expression normalized by the value of the potential at
 the cooling radius. */
void hydrostatic_ray_analytical_D3D(Real *rho, Real *r, const DataPack& hdp, Real dr, int nr)
{
  // Routine to determine the hydrostatic density profile
  // along a ray from the galaxy center
  int i;  // index along r direction

  Real gamma   = hdp.gamma;  // adiabatic index
  Real rho_eos = hdp.rho_eos_h;  // density where K_EOS is set
  Real cs      = hdp.cs_h;  // sound speed at rho_eos
  Real r_cool  = hdp.r_cool;  // cooling radius

  Real Phi_0;  // potential at cooling radius

  Real D_rho;  // ratio of density at mid plane and rho_eos

  Real gmo = gamma - 1.0;  // gamma-1

  // compute the potential at the cooling radius
  Phi_0 = phi_hot_halo_D3D(r_cool, hdp);

  // We are normalizing to the central density
  // so D_rho == 1
  D_rho = 1.0;

  // store densities
  for (i = 0; i < nr; i++) {
    r[i]   = r_hc_D3D(i, dr);
    rho[i] = rho_eos * pow(D_rho - gmo * (phi_hot_halo_D3D(r[i], hdp) - Phi_0) / (cs * cs), 1. / gmo);
  }
}

namespace hydrostatic_isothermal_detail{

/* find the cell above the disk where the density falls by exp(-7) < 1.0e-3. */
Real find_z1(int ks, int nzt, Real R, const DataPack& hdp, Real dz, int nz, int ng, Real Phi_0,
             Real cs)
{
  // TODO: make sure problems won't arise if/when ks > nzt

  Real D_rho; // ratio of density at mid plane and rho_eos

  // perform a simple check about the fraction of density within
  // a single cell
  Real z_1   = z_hc_D3D(ks, dz, nz, ng) + 0.5 * dz;  // cell ceiling
  D_rho = (phi_total_D3D(R, z_1, hdp) - Phi_0) / (cs * cs);

  if (exp(-1 * D_rho) < 0.1) {
    printf(
        "WARNING: >0.9 density in single cell R %e D_rho %e z_1 %e Phi(z) %e "
        "Phi_0 %E cs %e\n",
        R, D_rho, z_1, phi_total_D3D(R, z_1, hdp), Phi_0, cs);
  }

  // let's find the cell above the disk where the
  // density falls by exp(-7) < 1.0e-3.
  for (int k = ks; k < nzt; k++) {
    z_1   = z_hc_D3D(k, dz, nz, ng) + 0.5 * dz;  // cell ceiling
    D_rho = (phi_total_D3D(R, z_1, hdp) - Phi_0) / (cs * cs);
    if (D_rho >= 7.0) {
      break;
    }
  }

  // if(R<1.0)
  //   printf("Cells above disk (k-ks) = %d, z_1 = %e, exp(-D) = %e, R =
  //   %e\n",k-ks,z_1,exp(-1*D_rho),R);

  return z_1;
}

/* integrate the denstiy along the z-axis
 *
 * This approximates the integral with a Riemann sum (this method for numerical
 * integration is sometimes called the rectangle rule or the midpoint rule)
 *
 * \param z_int_min, z_int_max lower and upper bounds of the integral
 * \param n_int number of subintervals to use while approximating the integral.
 *     This should be 1 or larger.
 * \param R The radial position in a disk where this will be evaluated.
 * \param hdp, rho_0, Phi_0, cs parameters for the governing the solution to
 *     the vertical hydrostatic density profile.
 */
Real integrate_density_zax(Real z_int_min, Real z_int_max, int n_int, Real R,
                           const DataPack& hdp, Real rho_0, Real Phi_0,
                           Real cs) {
  // compute the size of every integration step
  const Real dz_int  = (z_int_max - z_int_min) / (Real(n_int));
  Real phi_int = 0.0;
  for (int i = 0; i < n_int; i++) {
    const Real z_0  = 0.5 * dz_int + dz_int * ((Real)i) + z_int_min;
    const Real Delta_phi = (phi_total_D3D(R, z_0, hdp) - Phi_0) / (cs * cs);
    phi_int += rho_0 * exp(-1 * Delta_phi) * dz_int;
  }
  return phi_int;
}

} // hydrostatic_isothermal_detail

/* Calculate the 1D density distribution in a hydrostatic column, assuming an isothermal gas.
 *
 * Uses an iterative to scheme to determine the density at (R, z=0) relative to (R=0,z=0), then sets
 * the densities according to an analytic expression.
 *
 * \param[out] rho A buffer that will be filled with the computed densities. This has a size of
 *     nz * 2*ng
 * \param[in]  R the radial position in the disk where the calculation is to be performed.
 * \param[in]  cur_Sigma the surface density at the given R
 * \param[in]  dz is cell width in z direction
 * \param[in]  nz is number of real cells (across all grids)
 * \param[in]  ng is number of ghost cells
 *
 * \returns the midplane mass-density
 */
Real hydrostatic_column_isothermal_D3D(Real *rho, Real R, Real cur_Sigma, const DataPack& hdp, Real dz, int nz, int ng)
{

  Real cs = hdp.cs;

  // start of integrals above disk plane
  const int ks = ((nz % 2) == 1) ? (ng + (nz - 1) / 2) : (ng + nz / 2);

  // prologue:

  // set the z-column size, including ghost cells
  const int nzt = nz + 2 * ng; // total number of cells in z-direction

  // compute the mid plane potential (aka the potential at z = 0)
  const Real Phi_0 = phi_total_D3D(R, 0, hdp);

  /* For an isothermal gas, we have

    grad P = - g rho
    cs^2 drho/dz = - g rho
    drho/rho = -g / cs^2 * dz
    ln rho - ln rho_0 = -cs^-2 \int g dz = -cs^-2 \Phi(z) + C
    rho/rho_0 = exp[ -cs^-2 ( \Phi(z) + C)]
    at z=0, rho = rho_0 exp[ - cs^-2 ( \Phi_0 +C) ] = rho_0
    so, C = -\Phi_0.
    We then have rho(z) = rho_0 exp[ - cs^-2 (\Phi(z) - \Phi_0)]

  */

  // Step 1: compute z_1. This is height used for "iteration"
  // -> Note while refactoring: I believe this is just the maximum height of the disk
  //    and that "iteration" was a typo for integration
  Real z_1 = hydrostatic_isothermal_detail::find_z1(ks, nzt, R, hdp, dz, nz, ng, Phi_0, cs);
  const Real z_disk_max = z_1;

  // Step 2: compute the density at the midplane, rho_0
  // -> we want this because it's the normalization of the vertical hydrostatic
  //    density profile
  // -> This normalization is based on the surface density

  // Step2a: compute the unnormalized integral
  // -> equivalent to evaluating the normalized integral with (rho_0 = 1)
  // -> for computational efficiency, we actually just compute half of the integral
  //    (this is okay since the profile is symmetric above & below the midplane)
  Real half_unnormalized_integral = hydrostatic_isothermal_detail::integrate_density_zax(
    /* integration lims: */ 0.0, z_1, /* n_int: */ 1000, R, hdp, 1.0, Phi_0, cs);

  // Step2b: actually compute rho_0
  // -> we leverage the fact that unnormalized_integral is equal to the
  //    disk surface density divided by rho_0
  const Real rho_0 = 0.5 * cur_Sigma / half_unnormalized_integral;

  // Step2c: exit early if the density is 0 here
  // -> this may not be strictly necessary (the rest of the function may work properly),
  //    but include this just to be safe
  if (rho_0 == 0.0) {
    for (int k = 0; k < nzt; k++) {
      rho[k] = 0;
    }
    return rho_0;
  }

  // Step 3: Let's now compute the cell-averaged density in each cell
  bool flag  = false;
  const int n_int = 10;  // integrate over a 1/10 cell
  for (int k = ks; k < nzt; k++) {
    // find cell center, bottom, and top
    Real z_int_min = z_hc_D3D(k, dz, nz, ng) - 0.5 * dz;
    Real z_int_max = z_hc_D3D(k, dz, nz, ng) + 0.5 * dz;
    if (z_int_max > z_disk_max) {
      z_int_max = z_disk_max;
    }

    if (not flag) {
      Real phi_int = hydrostatic_isothermal_detail::integrate_density_zax(
        z_int_min, z_int_max, n_int, R, hdp, rho_0, Phi_0, cs);

      // set density based on integral
      // of density in this cell
      rho[k] = phi_int / dz;

      flag = (z_int_max == z_disk_max);
    } else {
      // no mass up here!
      rho[k] = 0;
    }

    // mirror densities above and below disk plane
    int km;
    if (nz % 2) {
      km = (ng + (nz - 1) / 2) - (k - ks);
    } else {
      km = ng + nz / 2 - (k - ks) - 1;
    }
    rho[km] = rho[k];
  }

  return rho_0;
}

class IsothermalStaticGravHydroStaticColMaker{

public: // interface

  IsothermalStaticGravHydroStaticColMaker() = delete;
  IsothermalStaticGravHydroStaticColMaker(Real dz, int nz, int ghost_depth, DataPack hdp)
    : dz(dz), nz(nz), ghost_depth(ghost_depth), hdp(hdp)
  {}

  /* global total number of ghost zones along z-axis plus twice the ghost depth */
  int buffer_len() const noexcept {return nz + 2*ghost_depth; }

  /* Fill the buffer with the density values of a hydrostatic column
   *
   * \param[in]  cur_R The current cylindrical radius
   * \param[in]  cur_Sigma The surface density at `cur_R`.
   * \param[out] buffer is filled by this function. It is assumed to be an
   *     array of length `this->buffer_len()`
   * \returns The midplane mass density
   */
  Real construct_col(Real cur_R, Real cur_Sigma, Real* buffer) const noexcept
  {
    return hydrostatic_column_isothermal_D3D(buffer, cur_R, cur_Sigma, hdp, dz, nz, ghost_depth);
  }

private:
  /* cell width */
  const Real dz;
  /* global number of active-zone cells along the z-direction */
  const int nz;
  /* depth of the ghost-zone */
  const int ghost_depth;
  /* assorted data parameters */
  const DataPack hdp;

};

/*! \fn void hydrostatic_column_analytical_D3D(Real *rho, Real R, const DataPack& hdp,
 Real dz, int nz, int ng)
 *  \brief Calculate the 1D density distribution in a hydrostatic column.
     Uses an iterative to scheme to determine the density at (R, z=0) relative
 to (R=0,z=0), then sets the densities according to an analytic expression. */
void hydrostatic_column_analytical_D3D(Real *rho, Real R, Real cur_Sigma, const DataPack& hdp, Real dz, int nz, int ng)
{
  // x is cell center in x direction
  // y is cell center in y direction
  // dz is cell width in z direction
  // nz is number of real cells
  // ng is number of ghost cells
  // total number of cells in column is nz * 2*ng

  int i, k;               // index along z axis
  int nzt;                // total number of cells in z-direction
  Real Sigma_r;           // surface density expected at r
  Real Sigma_0 = hdp.Sigma_0;  // central surface density
  Real gamma   = hdp.gamma;
  // Real gamma = 1.001; // CHANGED FOR ISOTHERMAL

  Real rho_eos = hdp.rho_eos;
  Real cs      = hdp.cs;

  Real Phi_0;  // potential at z=0

  Real D_rho;  // ratio of density at mid plane and rho_eos
  Real D_new;  // new ratio of density at mid plane and rho_eos

  Real z_0, z_1, z_2;  // heights for iteration
  Real z_disk_max;
  Real A_0, A_1;  // density function to find roots

  // density integration
  Real phi_int, A;
  Real z_int_min, z_int_max, dz_int;
  Real Delta_phi;
  int n_int = 1000;

  // tolerance for secant method
  Real tol = 1.0e-6;

  // tolerance for surface density
  // fractional
  Real D_tol = 1.0e-5;

  int ks;  // start of integrals above disk plane
  int km;  // mirror of k
  if (nz % 2) {
    ks = ng + (nz - 1) / 2;
  } else {
    ks = ng + nz / 2;
  }

  // get the disk surface density
  Sigma_r = cur_Sigma;

  // set the z-column size, including ghost cells
  nzt = nz + 2 * ng;

  // compute the mid plane potential
  Phi_0 = phi_total_D3D(R, 0, hdp);

  // pick a fiducial guess for density ratio
  D_rho = pow(Sigma_r / Sigma_0, gamma - 1.);

  // begin iterative determination of density field
  int flag = 0;
  int iter = 0;  // number if iterations
  int flag_phi;  // flag for density extent
  int iter_phi;

  D_new = D_rho;
  while (!flag) {
    // iterate the density ratio
    D_rho = D_new;

    // first determine the range of z where
    // the density above the central disk plane is
    // non-zero

    // get started, find the maximum extent of the disk
    iter_phi = 0;
    flag_phi = 0;
    z_0      = 1.0e-3;
    z_1      = 1.0e-2;
    while (!flag_phi) {
      A_0 = D_rho - (phi_total_D3D(R, z_0, hdp) - Phi_0) / (cs * cs);
      A_1 = D_rho - (phi_total_D3D(R, z_1, hdp) - Phi_0) / (cs * cs);
      z_2 = z_1 - A_1 * (z_1 - z_0) / (A_1 - A_0);
      if (fabs(z_2 - z_1) / fabs(z_1) > 10.) {
        z_2 = 10. * z_1;
      }
      // advance limit
      z_0 = z_1;
      z_1 = z_2;

      if (fabs(z_1 - z_0) < tol) {
        flag_phi = 1;
        A_0      = D_rho - (phi_total_D3D(R, z_0, hdp) - Phi_0) / (cs * cs);
        A_1      = D_rho - (phi_total_D3D(R, z_1, hdp) - Phi_0) / (cs * cs);
        // make sure we haven't crossed 0
        if (A_1 < 0) {
          z_1 = z_0;
        }
      }
      iter_phi++;
      if (iter_phi > 1000) {
        printf("Something wrong in determining central density...\n");
        printf("iter_phi = %d\n", iter_phi);
        printf("z_0 %e z_1 %e z_2 %e A_0 %e A_1 %e phi_0 %e phi_1 %e\n", z_0, z_1, z_2, A_0, A_1,
               phi_total_D3D(R, z_0, hdp), phi_total_D3D(R, z_1, hdp));
#ifdef MPI_CHOLLA
        MPI_Finalize();
#endif
        exit(0);
      }
    }
    A_1        = D_rho - (phi_total_D3D(R, z_1, hdp) - Phi_0) / (cs * cs);
    z_disk_max = z_1;

    // Compute surface density
    z_int_min = 0.0;  // kpc
    z_int_max = z_1;  // kpc
    dz_int    = (z_int_max - z_int_min) / ((Real)(n_int));
    phi_int   = 0.0;
    for (k = 0; k < n_int; k++) {
      z_0       = 0.5 * dz_int + dz_int * ((Real)k);
      Delta_phi = (phi_total_D3D(R, z_0, hdp) - Phi_0) / (cs * cs);
      A         = D_rho - Delta_phi;
      phi_int += rho_eos * pow((gamma - 1) * A, 1. / (gamma - 1.)) * dz_int;
    }

    // update density constant
    D_new = D_rho * pow(phi_int / (0.5 * Sigma_r), 1. - gamma);

    // if we have converged, exit!
    if (fabs(phi_int - 0.5 * Sigma_r) / (0.5 * Sigma_r) < D_tol) {
      // done!
      flag = 1;
    }

    iter++;

    if (iter > 100) {
      printf("About to exit...\n");
#ifdef MPI_CHOLLA
      MPI_Finalize();
#endif
      exit(0);
    }
  }

  // OK, at this stage we know how to set the densities
  // so let's take cell averages
  flag  = 0;
  n_int = 10;  // integrate over a 1/10 cell
  for (k = ks; k < nzt; k++) {
    // find cell center, bottom, and top
    z_int_min = z_hc_D3D(k, dz, nz, ng) - 0.5 * dz;
    z_int_max = z_hc_D3D(k, dz, nz, ng) + 0.5 * dz;
    if (z_int_max > z_disk_max) {
      z_int_max = z_disk_max;
    }
    if (!flag) {
      dz_int  = (z_int_max - z_int_min) / ((Real)(n_int));
      phi_int = 0.0;
      for (i = 0; i < n_int; i++) {
        z_0       = 0.5 * dz_int + dz_int * ((Real)i) + z_int_min;
        Delta_phi = (phi_total_D3D(R, z_0, hdp) - Phi_0) / (cs * cs);
        A         = D_rho - Delta_phi;
        phi_int += rho_eos * pow((gamma - 1) * A, 1. / (gamma - 1.)) * dz_int;
      }

      // set density based on integral
      // of density in this cell
      rho[k] = phi_int / dz;

      if (z_int_max == z_disk_max) {
        flag = 1;
      }
    } else {
      // no mass up here!
      rho[k] = 0;
    }

    // mirror densities
    // above and below disk plane
    if (nz % 2) {
      km = (ng + (nz - 1) / 2) - (k - ks);
    } else {
      km = ng + nz / 2 - (k - ks) - 1;
    }
    rho[km]   = rho[k];
    Delta_phi = (phi_total_D3D(R, z_hc_D3D(k, dz, nz, ng), hdp) - Phi_0) / (cs * cs);
  }

  // check the surface density
  phi_int = 0.0;
  for (k = 0; k < nzt; k++) {
    phi_int += rho[k] * dz;
  }
}

Real determine_rho_eos_D3D(Real cs, Real Sigma_0, const DataPack& hdp)
{
  // OK, we need to set rho_eos based on the central surface density.
  // and the central potential
  int k;
  Real z_pos, rho_eos;
  Real Phi_0 = phi_total_D3D(0, 0, hdp);
  Real gamma = hdp.gamma;
  // Real gamma = 1.001; // CHANGED FOR ISOTHERMAL
  Real Delta_phi;
  Real A = 0.0;

  // determine the maximum scale height by finding the
  // zero crossing of 1-(Phi-Phi_0)/cs^2
  // using the secant method
  Real z_0, z_1, z_2, A_0, A_1;
  Real tol     = 1.0e-6;
  int flag_phi = 0;
  int iter_phi = 0;

  // get started
  z_0 = 1.0e-3;
  z_1 = 1.0e-2;
  while (!flag_phi) {
    A_0 = 1.0 - (phi_total_D3D(0, z_0, hdp) - Phi_0) / (cs * cs);
    A_1 = 1.0 - (phi_total_D3D(0, z_1, hdp) - Phi_0) / (cs * cs);
    z_2 = z_1 - A_1 * (z_1 - z_0) / (A_1 - A_0);

    if (fabs(z_2 - z_1) / fabs(z_1) > 10.) {
      z_2 = 10. * z_1;
    }

    // advance limit
    z_0 = z_1;
    z_1 = z_2;

    // printf("z_0 %e z_1 %e\n",z_0,z_1);
    if (fabs(z_1 - z_0) < tol) {
      flag_phi = 1;
      A_0      = 1.0 - (phi_total_D3D(0, z_0, hdp) - Phi_0) / (cs * cs);
      A_1      = 1.0 - (phi_total_D3D(0, z_1, hdp) - Phi_0) / (cs * cs);
      // make sure we haven't crossed 0
      if (A_1 < 0) {
        z_1 = z_0;
      }
    }
    iter_phi++;
    if (iter_phi > 1000) {
      printf("Something wrong in determining central density...\n");
      printf("iter_phi = %d\n", iter_phi);
      printf("z_0 %e z_1 %e z_2 %e A_0 %e A_1 %e phi_0 %e phi_1 %e\n", z_0, z_1, z_2, A_0, A_1,
             phi_total_D3D(0, z_0, hdp), phi_total_D3D(0, z_1, hdp));
#ifdef MPI_CHOLLA
      MPI_Finalize();
#endif
      exit(0);
    }
  }

  // generate a high resolution density and z profile
  int n_int      = 1000;
  Real z_int_min = 0.0;  // kpc
  Real z_int_max = z_1;  // kpc
  Real dz_int    = (z_int_max - z_int_min) / ((Real)(n_int));
  Real phi_int   = 0.0;

  // now integrate the density profile
  for (k = 0; k < n_int; k++) {
    z_pos     = 0.5 * dz_int + dz_int * ((Real)k);
    Delta_phi = phi_total_D3D(0, z_pos, hdp) - Phi_0;
    A         = 1.0 - Delta_phi / (cs * cs);
    phi_int += pow((gamma - 1) * A, 1. / (gamma - 1.)) * dz_int;
  }
  // use the potential integral to set central density at r=0
  rho_eos = 0.5 * Sigma_0 / phi_int;

  // check
  /*
  phi_int = 0.0;
  for(k=0;k<n_int;k++)
  {
    z_pos = 0.5*dz_int + dz_int*((Real) k);
    A = 1.0 - (phi_total_D3D(0,z_pos,hdp)-Phi_0)/(cs*cs);
    phi_int += rho_eos*pow((gamma-1)*A,1./(gamma-1.))*dz_int;
  }
  printf("phi_int %e Sigma_0/2 %e\n",phi_int,0.5*Sigma_0);
  */

  // return the central density
  return rho_eos;
}

Real halo_density_D3D(Real r, Real *r_halo, Real *rho_halo, Real dr, int nr)
{
  // interpolate the halo density profile
  int i;

  // find the index of the current
  // position in r_halo (based on r_hc_D3D)
  i = (int)((r - 0.5 * dr) / dr);
  if (i < 0 || i >= nr - 1) {
    if (i < 0) {
      i = 0;
    } else {
      i = nr - 2;
    }
  }
  // return the interpolated density profile
  return (rho_halo[i + 1] - rho_halo[i]) * (r - r_halo[i]) / (r_halo[i + 1] - r_halo[i]) + rho_halo[i];
}

// we need to forward declare the following functions
// -> we opt to forward declare them rather than move them because the functions are fairly large
// -> in both cases, the functions only initialize thermal-energy in the total-energy field
template<typename HydroStaticColMaker, typename Vrot2FromPotential>
void partial_initialize_isothermal_disk(const parameters& p, const Header& H,
                                        const Grid3D& grid, const Grid3D::Conserved& C,
                                        const DataPack hdp, const HydroStaticColMaker& col_maker,
                                        const Vrot2FromPotential& vrot2_from_phi_fn);
void partial_initialize_halo(const parameters& p, const Header& H,
                             const Grid3D& grid, const Grid3D::Conserved& C,
                             DataPack hdp);



/*! \fn void Disk_3D(parameters P)
 *  \brief Initialize the grid with a 3D disk. */
void Grid3D::Disk_3D(parameters p)
{
  Real T_d, T_h, mu;
  Real K_eos, rho_eos, cs, rho_eos_h, cs_h, r_cool;

  // MW model
  DiskGalaxy galaxy = Galaxies::MW;
  // M82 model Galaxies::M82;

  const MiyamotoNagaiDiskProps stellar_disk = galaxy.getStellarDisk();
  const GasDiskProps gas_disk               = galaxy.getGasDisk();

  r_cool = galaxy.getR_cool();  // cooling radius in kpc (MW)

  T_h       = 1.0e6;  // halo temperature, at density floor
  rho_eos   = 1.0e7;  // gas eos normalized at 1e7 Msun/kpc^3
  rho_eos_h = 3.0e3;  // gas eos normalized at 3e3 Msun/kpc^3 (about n_h = 10^-3.5)
  mu        = 0.6;

  Real Sigma_0 = gas_disk.CentralSurfaceDensity();  // (in Msun/kpc^2)
  // changing the following 3 lines directly assign T_d the value stored in gas_disk.T_d slightly
  // changes the result of the simulation (its worrying that I can't explain why!)
  T_d       = 1.0e4;
  if (T_d != gas_disk.T_d) {
    CHOLLA_ERROR("unexpected disk temperature");
  }

  if (true){
    chprintf("\nNominal Disk properties:\n");
    chprintf("                                            Stellar            Gas\n");
    chprintf("                                            -------          -------\n");
    chprintf("scale length (kpc):                      %.7e    %.7e\n", stellar_disk.R_d, gas_disk.R_d);
    chprintf("scale height (kpc):                      %.7e          - \n", stellar_disk.Z_d);
    chprintf("total mass (Msolar):                     %.7e    %.7e\n", stellar_disk.M_d, gas_disk.M_d);
    chprintf("central surface density (Msolar/kpc^2):        -          %.7e\n", Sigma_0);

    chprintf("\n");
  }


  // EOS info
  cs   = sqrt(KB * T_d / (mu * MP)) * TIME_UNIT / LENGTH_UNIT;  // sound speed in kpc/kyr
  cs_h = sqrt(KB * T_h / (mu * MP)) * TIME_UNIT / LENGTH_UNIT;  // sound speed in kpc/kyr

  // set some initial parameters
  // these parameters are mostly passed to hydrostatic column
  DataPack hdp;  // parameters
  hdp.stellar_disk   = stellar_disk;
  hdp.halo_potential = galaxy.getHaloPotential();
  hdp.T_d            = T_d;
  hdp.Sigma_0        = Sigma_0;
  hdp.R_g            = gas_disk.R_d;
  hdp.H_g            = gas_disk.H_d;  // initial guess for gas scale height (kpc)
  hdp.gamma          = p.gamma;

  if (gas_disk.isothermal){
    // determine rho_eos by setting central density of disk based on central temperature
    rho_eos = determine_rho_eos_D3D(cs, Sigma_0, hdp);
    K_eos   = cs * cs * rho_eos;  // CHANGED FOR ISOTHERMAL
  } else {
    CHOLLA_ERROR("Initializing a non-isothermal gas disk hasn't been tested");
    // this branch represents older logic that was partially commented out throughout
    // this function
    // - a lot of this logic was scattered throughout this function so I consolidated it
    //   into a single spot

    // previously, when T_d was set to 1e4 K in this function, there was a note that the value
    // was changed for an Isothermal disk. These lines preceded the definition of T_d
    // T_d = 5.9406e5; // SET TO MATCH K_EOS SET BY HAND for K_eos = 1.859984e-14
    // T_d = 2.0e5;

    // previously, we set the value of rho_eos near the top of the function with the following line:
    //   rho_eos   = 1.0e7;  // gas eos normalized at 1e7 Msun/kpc^3
    // Then further down in the function, overwrote the value of rho_eos by setting central
    // density of disk based on central temperature with the following line:
    //   rho_eos = determine_rho_eos_D3D(cs, Sigma_0, hdp);
    // It's unclear whether we always did this, or if the 2nd line got added when we started
    // using an isothermal disk

    // finally, here is the logic for initializing K_eos:
    //   K_eos = cs*cs*pow(rho_eos,1.0-p.gamma)/p.gamma; //P = K\rho^gamma
  }

  // Store remaining parameters
  hdp.K_eos     = K_eos;
  hdp.rho_floor = 0.0;  // rho_floor, set to 0
  hdp.rho_eos   = rho_eos;
  hdp.cs        = cs;
  hdp.K_eos_h   = cs_h * cs_h * pow(rho_eos_h, 1.0 - p.gamma) / p.gamma;
  hdp.rho_eos_h = rho_eos_h;
  hdp.cs_h      = cs_h;
  hdp.r_cool    = r_cool;

  hdp.Rgas_truncation_radius = Get_Gas_Truncation_Radius(p);

  // most of the heavy-lifting happens in the following function-calls below:
  // - they all assume that the fields start out with values that are uniformly zero
  // - they then update the density and momenta fields. They also store the 
  //   thermal-energy-density field in the total-energy-density field (we need to
  //   add the kinetic energy contribution afterwards)

  bool self_gravity = false;
#ifdef GRAVITY
  self_gravity = true;
#endif

  if (gas_disk.isothermal){
    if (self_gravity){
      // nongas_phi calculates the gravitational potential contributed by material other than the
      // gas disk at a given (R,z), where R is cylindrical radius
      // -> currently this just includes the static-gravitational potentials (from the stellar-disk
      //    and the dark-matter halo)
      // -> in principle this is also allowed to include other contributions (e.g. an estimate for 
      //    potential contributed by a star-particle disk, gas in the halo)
      auto nongas_phi_fn = [hdp](Real R, Real z) -> Real {return phi_total_D3D(R, z, hdp); };

      // isoth_term is constant value of pressure/rho at the desired isothermal temperature.
      // equivalent to:  `(isothermal_sound_speed)^2` OR `(adiabatic_sound_speed)^2/gamma` OR
      //                 `(gamma - 1) * specific_internal_energy`
      Real isoth_term = hdp.cs * hdp.cs; // <- square of the isothermal sound speed
      Real initial_gas_scale_height_guess = gas_disk.H_d;
      SelfGravHydroStaticColMaker col_maker(H.n_ghost, ZGridProps(p.zmin, p.zlen, p.nz),
                                            isoth_term, nongas_phi_fn, initial_gas_scale_height_guess);
      // the following function is used to compute the rotational velocity for a collisionless particle
      auto vrot2_from_phi_fn = [hdp](Real R, Real z) -> Real {
        return R * (std::fabs(hdp.stellar_disk.gr_disk_D3D(R, z)) +
                    std::fabs(gr_halo_D3D(R, z, hdp)));
      };
      partial_initialize_isothermal_disk(p, this->H, *this, this->C, hdp, col_maker,
                                         vrot2_from_phi_fn);
    } else {
      IsothermalStaticGravHydroStaticColMaker col_maker(p.zlen / ((Real)p.nz), p.nz, H.n_ghost, hdp);
      auto vrot2_from_phi_fn = [hdp](Real R, Real z) -> Real {
        return R * (std::fabs(hdp.stellar_disk.gr_disk_D3D(R, z)) +
                    std::fabs(gr_halo_D3D(R, z, hdp)));
      };
      partial_initialize_isothermal_disk(p, this->H, *this, this->C, hdp, col_maker,
                                         vrot2_from_phi_fn);
    }
  } else {
    CHOLLA_ERROR("Currently, there isn't support for a non-isothermal gas disk");
  }
  partial_initialize_halo(p, this->H, *this, this->C, hdp);

  // Final Step: add kinetic energy to total energy
  for (int k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        int id = i + j * H.nx + k * H.nx * H.ny;

        // set internal energy
#ifdef DE
        C.GasEnergy[id] = C.Energy[id];
#endif

        // add kinetic contribution to total energy
        C.Energy[id] += 0.5 *
                        (C.momentum_x[id] * C.momentum_x[id] + C.momentum_y[id] * C.momentum_y[id] +
                         C.momentum_z[id] * C.momentum_z[id]) /
                        C.density[id];
      }
    }
  }

}

template<typename Vrot2FromPotential>
void assign_vcirc_from_midplane_isothermal_disk(const parameters& p, const Header& H,
                                                const Grid3D& grid, const Grid3D::Conserved& C,
                                                const DataPack hdp,
                                                const Vrot2FromPotential& vrot2_from_phi_fn,
                                                const std::vector<Real>& rho_midplane_2Dbuffer)
{
  // we are adopting the strategy for initializing the circular velocity in an isothermal disk 
  // outlined in Wang+ (2010), https://ui.adsabs.harvard.edu/abs/2010MNRAS.407..705W
  // -> in that paper, they show for an isothermal disk that circular velocity has no z-dependence

  // At each (x,y) pair (with non-zero midplane density), we adopt the following stratgey:
  // -> we use the midplane density values to compute the local gradient in the pressure field
  // -> we compute the contributions from the gravitational potential (with the specified function)
  // -> from both of these quantities, we can determine the circular velocity. 
  //    -> there are some places where the radial acceleration may have an unexpected sign. In these
  //       cases, we consider circular velocity to be 0 (this is important after we start tapering the velocity)
  // -> we use the circular velocity to initialize the momentum of all cells in the column with non-zero density
  //    (any cell with zero density isn't considered to be a part of the disk)

  const Real dx = p.xlen / ((Real)p.nx);  // cell-width x
  const Real dy = p.xlen / ((Real)p.nx);  // cell-width y
  bool any_error = false;
  for (int j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
    for (int i = H.n_ghost; i < H.nx - H.n_ghost; i++) {

      // fetch the associated midplane density
      Real rho_midplane = rho_midplane_2Dbuffer[i + j * H.nx];

      // if the midplane density is zero, skip this location (this happens after the disk completely tapers off)
      if (rho_midplane == 0.0)  continue;

      // get the centered x & y positions (the way the function is written, we also get a z position)
      const int dummy_k = H.n_ghost + H.ny;
      Real x_pos, y_pos, dummy_z_pos;
      grid.Get_Position(i, j, dummy_k, &x_pos, &y_pos, &dummy_z_pos);

      // calculate radial position and phi, we're assuming the disk is centered at (x,y) = (0,0)
      Real r   = sqrt(x_pos * x_pos + y_pos * y_pos);
      Real phi = atan2(y_pos, x_pos);  // azimuthal angle (in x-y plane)

      // calculate the squared circular velocity in the midplane of a collisionless particle due the specified
      // gravitational potential
      // -> this is equal to the `-1 * accel_radial * r` (it should be a positive number)
      // -> this may or may not include an estimate for the gas disk's own gravitational potential
      //    (that decision is made by the caller of this function)
      Real vrot2_contrib_from_phi = vrot2_from_phi_fn(r, 0.0);

      // calculate the radial pressure gradient in the midplane (gradient calc is first order at boundaries)

      //  pressure gradient along x direction
      int i_left   = std::max<int>((i-1), H.n_ghost);
      int i_right  = std::min<int>((i+1), H.nx - H.n_ghost - 1);
      Real P_xL = rho_midplane_2Dbuffer[i_left + j * H.nx] * (hdp.cs * hdp.cs);
      Real P_xR = rho_midplane_2Dbuffer[i_right + j * H.nx] * (hdp.cs * hdp.cs);
      Real dPdx = (P_xR - P_xL) / (dx*(i_right - i_left));

      // pressure gradient along y direction
      int j_left   = std::max<int>((j-1), H.n_ghost);
      int j_right  = std::min<int>((j+1), H.ny - H.n_ghost - 1);
      // Currently, C.Energy just stores internal energy density
      Real P_yL = rho_midplane_2Dbuffer[i + j_left * H.nx] * (hdp.cs * hdp.cs);
      Real P_yR = rho_midplane_2Dbuffer[i + j_right * H.nx] * (hdp.cs * hdp.cs);
      Real dPdy = (P_yR - P_yL) / (dy * (j_right - j_left));

      // the cylindrical radius multiplied by the radial pressure gradient is:
      Real r_times_dPdr = x_pos * dPdx + y_pos * dPdy;

      // compute the circular velocity (in the midplane)
      //  vcirc^2 / r = -a_radial = (rhat * grad Phi) - dPdr / rho_midplane
      //  vcirc^2   = r * (radial_grav_accel - dPdr / rho_midplane)
      //            = (r * radial_grav_accel ) - (r * dPdr) / rho_midplaned
      Real vcirc_sq = (vrot2_contrib_from_phi) - r_times_dPdr / rho_midplane;
      if (vcirc_sq <= 0){
        continue;
      } else if (not std::isfinite(vcirc_sq)) {
        any_error = true;
        continue;
      }

      Real v  = std::sqrt(vcirc_sq); // circular velocity
      Real vx = -std::sin(phi) * v;
      Real vy = std::cos(phi) * v;
      Real vz = 0;

      // initialize the momentum for all cells in the column
      for (int k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
        int id = i + j * H.nx + k * H.nx * H.ny;

        Real d = C.density[id];
        if (d > 0.0) {
          C.momentum_x[id] = d * vx;
          C.momentum_y[id] = d * vy;
          C.momentum_z[id] = d * vz;
        }
      }

    }
  }
  CHOLLA_ASSERT(not any_error, "There was a problem when initializing the circular velocity");
}

template<typename Vrot2FromPotential>
void older_assign_vels(const parameters& p, const Header& H,
                       const Grid3D& grid, const Grid3D::Conserved& C,
                       const Vrot2FromPotential& vrot2_from_phi_fn)
{
  // Assign the circular velocities
  // -> everywhere that density >= 0, we compute the radial acceleration and use
  //    it to initialize the circular velocity
  // -> radial acceleration = -(rhat * grad Phi) + (dP/dr) / density
  //     -> of course "radial" is along the cylindrical radius
  //     -> Phi is the total gravitational potential of the stellar-disk and the halo
  //        (At this time, the gravitational potential does NOT consider gas-disk contributions)
  //     -> dPdr is the radial component of the pressure gradient
  //     -> within the loop we actually flip the sign
  // -> when radial acceleration has an unexpected sign, we set circular velocity to 0
  //     -> This is important after we start tapering the velocity
  //
  // We currently rely upon the gas-density getting truncated to 0 outside the gas-disk.
  // When we didn't do that, some issues cropped up
  // -> we previously were seeing an issue in a MW galaxy were the velocity would go to
  //    zero relatively close to a taper-radius of 1.9 and then a brief spike of velocity
  //    at r_cyl ~2.25
  // -> this effect was most pronounced at cylindrical-phi = 0.25*pi, 0.75*pi, 1.25*pi, 1.75*pi
  //    (the effect is cut off at cylindrical-phi = 0, 0.5*pi, pi, 1.5*pi)
  // -> at these locations, Pressure gradient seems to change signs
  const Real dx = p.xlen / ((Real)p.nx);  // cell-width x
  const Real dy = p.xlen / ((Real)p.nx);  // cell-width y
  bool any_accel_error = false;
  bool any_vel_error = false;
  for (int k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        int id = i + j * H.nx + k * H.nx * H.ny;

        // get density
        Real d = C.density[id];

        // restrict to regions where the density
        // has been set
        if (d > 0.0) {
          // get the centered x, y, and z positions
          Real x_pos, y_pos, z_pos;
          grid.Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

          // calculate radial position and phi (assumes disk is centered at 0,
          // 0)
          Real r   = sqrt(x_pos * x_pos + y_pos * y_pos);
          Real phi = atan2(y_pos, x_pos);  // azimuthal angle (in x-y plane)

          // consider radial acceleration from gravitational potential
          Real agrav_times_r = vrot2_from_phi_fn(r,z_pos);

          //  pressure gradient along x direction
          // gradient calc is first order at boundaries
          int i_left   = std::max<int>((i-1), H.n_ghost);
          int i_right  = std::min<int>((i+1), H.nx - H.n_ghost - 1);
          // Currently, C.Energy just stores internal energy density
          Real P_xL = C.Energy[i_left + j * H.nx + k * H.nx * H.ny] * (gama - 1.0);
          Real P_xR = C.Energy[i_right + j * H.nx + k * H.nx * H.ny] * (gama - 1.0);
          Real dPdx = (P_xR - P_xL) / (dx*(i_right - i_left));

          // pressure gradient along y direction
          int j_left   = std::max<int>((j-1), H.n_ghost);
          int j_right  = std::min<int>((j+1), H.ny - H.n_ghost - 1);
          // Currently, C.Energy just stores internal energy density
          Real P_yL = C.Energy[i + j_left * H.nx + k * H.nx * H.ny] * (gama - 1.0);
          Real P_yR = C.Energy[i + j_right * H.nx + k * H.nx * H.ny] * (gama - 1.0);
          Real dPdy = (P_yR - P_yL) / (dy * (j_right - j_left));

          // radial pressure gradient
          Real dPdr = x_pos * dPdx / r + y_pos * dPdy / r;

          // radial acceleration (multiplied by cylindrical radius)
          Real a_times_r = agrav_times_r + r * (dPdr / d); // = r * (a_grav + dPdr / d)

          if (a_times_r < 0){
            continue;
          } else if (not std::isfinite(a_times_r)) {
            any_accel_error = true;
            continue;
          }

          Real v  = sqrt(a_times_r); // circular velocity
          Real vx = -sin(phi) * v;
          Real vy = cos(phi) * v;
          Real vz = 0;
          any_vel_error = ( any_vel_error or (not std::isfinite(vx)) or (not std::isfinite(vy)) or
                            (not std::isfinite(vz)) );

          // set the momenta
          C.momentum_x[id] = d * vx;
          C.momentum_y[id] = d * vy;
          C.momentum_z[id] = d * vz;
        }

      }
    }
  }


  // todo: consider writing another function to write a error messages with problematic values
  CHOLLA_ASSERT(not any_accel_error, "There was a problem with a computed acceleration");
  CHOLLA_ASSERT(not any_vel_error, "There was a problem with a computed velocity");
}


template<typename HydroStaticColMaker, typename Vrot2FromPotential>
void partial_initialize_isothermal_disk(const parameters& p, const Header& H,
                                        const Grid3D& grid, const Grid3D::Conserved& C,
                                        const DataPack hdp, const HydroStaticColMaker& col_maker,
                                        const Vrot2FromPotential& vrot2_from_phi_fn)
{
  // Step 0: allocate a buffer to track the midplane mass density
  std::vector<Real> rho_midplane_2Dbuffer((H.ny * H.nx), 0.0);


  // Step 1: add the gas-disk density and thermal energy to the density and energy arrays
  // -> At each (x,y) pair, we use col_maker to loop over all z-values and compute "hydrostatic column"
  //    (i.e. the vertical density profile for the gas in the disk that is in hydrostatic equlibrium).
  // -> in slightly more detail, col_maker stores the density-profile in a buffer
  // -> then we compute the disk density & thermal energy based on values in that buffer
  std::vector<Real> rho_buffer(col_maker.buffer_len(), 0.0);
  bool any_density_error = false;
  for (int j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
    for (int i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
      // get the centered x & y positions (the way the function is written, we also get a z position)
      const int dummy_k = H.n_ghost + H.ny;
      Real x_pos, y_pos, dummy_z_pos;
      grid.Get_Position(i, j, dummy_k, &x_pos, &y_pos, &dummy_z_pos);

      // cylindrical radius
      Real r = sqrt(x_pos * x_pos + y_pos * y_pos);

      // Compute the hydrostatic density profile in this z column
      Real cur_Sigma = Sigma_disk_D3D(r, hdp);
      if (cur_Sigma == 0.0) continue;
      Real rho_midplane = col_maker.construct_col(r, cur_Sigma, rho_buffer.data());

      // record the midplane mass-density
      rho_midplane_2Dbuffer[i + j * H.nx] = rho_midplane;

      // store densities (from the column)
      for (int k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
        int id = i + j * H.nx + k * H.nx * H.ny;

        // get density from hydrostatic column computation
#ifdef MPI_CHOLLA
        Real d = rho_buffer[nz_local_start + H.n_ghost + (k - H.n_ghost)];
#else
        Real d = rho_buffer[H.n_ghost + (k - H.n_ghost)];
#endif
        any_density_error = any_density_error or (d < 0) or (not std::isfinite(d));

        // set pressure adiabatically
        // P = K_eos*pow(d,p.gamma);
        // set pressure isothermally
        Real P = d * (hdp.cs * hdp.cs);  // CHANGED FOR ISOTHERMAL

        // store density in density
        C.density[id] = d;

        // store internal energy in Energy array
        C.Energy[id] = P / (gama - 1.0);
      }
    }
  }
  // todo: consider writing another function to write a error message with problematic denisty
  CHOLLA_ASSERT(not any_density_error, "There was a problem initializing disk density");

  if (false){
    assign_vcirc_from_midplane_isothermal_disk(p, H, grid, C, hdp, vrot2_from_phi_fn,
                                               rho_midplane_2Dbuffer);
  } else {
    older_assign_vels(p, H, grid, C, vrot2_from_phi_fn);
  }

}


// This is called after initializing the disk
void partial_initialize_halo(const parameters& p, const Header& H,
                             const Grid3D& grid, const Grid3D::Conserved& C,
                             DataPack hdp)
{
    // create a look up table for the halo gas profile
  const int nr  = 1000;
  const Real dr = sqrt(3) * 0.5 * fmax(p.xlen, p.zlen) / ((Real)nr);
  std::vector<Real> rho_halo(nr, 0.0);
  std::vector<Real> r_halo(nr, 0.0);

  //////////////////////////////////////////////
  //////////////////////////////////////////////
  // Produce a look up table for a hydrostatic hot halo
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  hydrostatic_ray_analytical_D3D(rho_halo.data(), r_halo.data(), hdp, dr, nr);
  chprintf("Hot halo lookup table generated...\n");

  //////////////////////////////////////////////
  //////////////////////////////////////////////
  // Add a hot, hydrostatic halo
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  for (int k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        // get the cell index
        int id = i + j * H.nx + k * H.nx * H.ny;

        // get the centered x, y, and z positions
        Real x_pos, y_pos, z_pos;
        grid.Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        // calculate 3D radial position and phi (assumes halo is centered at 0,
        // 0)
        Real r = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);

        // interpolate the density at this position
        Real d = halo_density_D3D(r, r_halo.data(), rho_halo.data(), dr, nr);

        // set pressure adiabatically
        Real P = hdp.K_eos_h * pow(d, p.gamma);

        // store density in density
        C.density[id] += d;

        // store internal energy in Energy array
        C.Energy[id] += P / (gama - 1.0);
      }
    }
  }

}

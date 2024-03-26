/* This provides unit-tests for gravitational potential calculations.
 *
 * This mainly includes things like the deposition stencil
 */

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

#include <cmath>
#include <vector>

#include "../global/global.h"
#include "../io/io.h"
#include "../utils/error_handling.h"
#include "potentials.h"

/* This provides some analytic solutions for a razor-thin exponential disk*/
struct RazorThinExponentialDisk {
  Real Sigma0;  // surface density at R = 0
  Real scale_length;

  /* Radial acceleration */
  Real gr_disk_D3D(Real R, Real z) const noexcept
  {
    CHOLLA_ASSERT(z == 0.0, "z must be zero");

    // equation 2.164b from Binney & Tremaine
    Real y = R / (2 * scale_length);
    // equation 2.165 from Binney & Tremaine
    Real vrot_sq =
        4 * M_PI * GN * this->Sigma0 * this->scale_length * (y * y) *
        ((std::cyl_bessel_i(0, y) * std::cyl_bessel_k(0, y)) - (std::cyl_bessel_i(1, y) * std::cyl_bessel_k(1, y)));

    return -vrot_sq / R;
  }

  /* computes the potential */
  Real phi_disk_D3D(Real R, Real z) const noexcept
  {
    CHOLLA_ASSERT(z == 0.0, "z must be zero");

    // equation 2.164b from Binney & Tremaine
    Real y = R / (2 * scale_length);
    // equation 2.164a from Binney & Tremaine
    return -M_PI * GN * this->Sigma0 * R *
           ((std::cyl_bessel_i(0, y) * std::cyl_bessel_k(1, y)) - (std::cyl_bessel_i(1, y) * std::cyl_bessel_k(0, y)));
  }
};

TEST(tALLPotentialAprroxExponentialDisk3MN, comparison)
{
  Real disk_mass                  = 0.15 * 6.5e10;  // in solar masses
  Real scale_length               = 3.5;            // in kpc
  Real scale_height               = 0.0;            // in kpc, A value of 0 means it's infinitely thin
  AprroxExponentialDisk3MN approx = AprroxExponentialDisk3MN::create(disk_mass, scale_length, scale_height, true);

  Real Sigma0 = disk_mass / (2 * M_PI * scale_length * scale_length);
  RazorThinExponentialDisk ref{Sigma0, scale_length};

  Real R_div_scale_length_step = 0.01;

  const Real z       = 0.0;
  const Real Phi_tol = 0.02;

  for (int i = 1; i <= 300; i++) {  // skip i = 0

    Real R = i * R_div_scale_length_step * scale_length;  // in kpc

    Real approx_phi = approx.phi_disk_D3D(R, z);
    Real ref_phi    = ref.phi_disk_D3D(R, z);
    // chprintf("R = %e, approx Phi = %e, Ref Phi = %e, rel_diff = %e\n",
    //          R, approx_phi, ref_phi, std::fabs((approx_phi - ref_phi)/ref_phi));
    ASSERT_LE(std::fabs(approx_phi - ref_phi), std::fabs(Phi_tol * ref_phi))
        << "For R/scale_length = " << i * R_div_scale_length_step
        << ", the relative error exceeds relative error of: " << Phi_tol;

    // Real approx_gr = approx.gr_disk_D3D(R,z);
    // Real ref_gr = ref.gr_disk_D3D(R,z);
    //
    // chprintf("R = %e, approx gr = %e, Ref gr = %e, rel_diff = %e\n",
    //          R, approx_gr, ref_gr, std::fabs((approx_gr - ref_gr)/ref_gr));
    //
    // ASSERT_LE(std::fabs(approx_gr - ref_gr),
    //           std::fabs(gr_percentage * ref_gr));
  }
}
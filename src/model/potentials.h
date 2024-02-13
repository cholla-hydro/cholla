#ifndef POTENTIALS
#define POTENTIALS


// this file contains objects representing (semi-)analytic gravitational potentials

#include <cmath>

#include "../global/global.h"
#include "../utils/error_handling.h"

struct NFWHaloPotential{
  Real M_h;   /*!< total halo mass in Msolar */
  Real R_h;   /*!< halo scale length (NOT the virial radius) */
  Real c_vir;  /*!< halo concentration parameter (to account for adiabatic contraction) */

  /* function with logarithms used in NFW definitions */
  static Real log_func(Real y) { return log(1 + y) - y / (1 + y); };

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

  /* Potential of NFW halo */
  Real phi_halo_D3D(Real R, Real z) const noexcept
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

/* Aggregates properties related to a disk
 *
 * Take note that this is NOT an exponential disk!
 */
struct MiyamotoNagaiDiskProps{
  Real M_d;  /*!< total mass (in Msolar) */
  Real R_d;  /*!< scale-length (in kpc) */
  Real Z_d;  /*!< scale-height (in kpc). In the case of a gas disk, this is more
              *!< of an initial guess */

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


/* Approximates the potential of a Exponential Disk as the sum of 3 MiyamotoNagaiDiskProps
 * disks.
 *
 * This uses the table from
 *   https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract
 * to determine the properties of each component
 */
struct AprroxExponentialDisk3MN{
  MiyamotoNagaiDiskProps comps[3];

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
  static AprroxExponentialDisk3MN create(Real mass, Real scale_length, Real scale_height, 
                                         bool exponential_scaleheight)
  {
    if ((mass <= 0) or (scale_length <= 0))  CHOLLA_ERROR("invalid args");

    // step 1: determine the disk thickness parameter b (it's shared by all components)
    Real b_div_scale_length;
    if (scale_height < 0.0) {
      CHOLLA_ERROR("scale_height must be positive");
    } else if (scale_height == 0.0) {
      b_div_scale_length = 0.0;
    } else if (exponential_scaleheight){
      Real x = scale_height / scale_length;
      b_div_scale_length = (-0.269*x + 1.080) * x + 1.092 * x;
    } else {
      Real x = scale_height / scale_length;
      b_div_scale_length = (-0.033*x + 0.262) * x + 0.659 * x;
    }
    Real b = b_div_scale_length * scale_length;


    if (b_div_scale_length > 3.0) CHOLLA_ERROR("The disk is too thick for this approx");

    // step 2 determine other parameters.
    // -> we use values from table 1 (although this potential technically
    //    corresponds to negative densities in the outer disk, that's probably 
    //    fine for our purposes)
    const Real k[6][5] = {{-0.0090,  0.0640, -0.1653,  0.1164,  1.9487},   // M_MN0 / mass
                          { 0.0173, -0.0903,  0.0877,  0.2029, -1.3077},   // M_MN1 / mass
                          {-0.0051,  0.0287, -0.0361, -0.0544,  0.2242},   // M_MN2 / mass
                          {-0.0358,  0.2610, -0.6987, -0.1193,  2.0074},   // a0 / scale_length
                          {-0.0830,  0.4992, -0.7967, -1.2966,  4.4441},   // a1 / scale_length
                          {-0.0247,  0.1718, -0.4124, -0.5944,  0.7333}};  // a2 / scale_length
    auto param = [&k, b_div_scale_length](int i)
    {
        Real x = b_div_scale_length;
        return ((((k[i][0] * x + k[i][1]) * x + k[i][2]) * x) + k[i][3]) * x + k[i][4];
    };

    AprroxExponentialDisk3MN out;
    for (int j = 0; j < 3; j++){
      out.comps[j] = MiyamotoNagaiDiskProps{param(j) * mass, param(j + 3) * scale_length, b};
    }
    return out;
  }

  /* Radial acceleration */
  Real gr_disk_D3D(Real R, Real z) const noexcept
  {
    return (comps[0].gr_disk_D3D(R,z) + comps[1].gr_disk_D3D(R,z) +
            comps[2].gr_disk_D3D(R,z));
  }

  /* vertical acceleration */
  Real gz_disk_D3D(Real R, Real z) const noexcept
  {
    return (comps[0].gz_disk_D3D(R,z) + comps[1].gz_disk_D3D(R,z) +
            comps[2].gz_disk_D3D(R,z));
  }

  /* computes the potential */
  Real phi_disk_D3D(Real R, Real z) const noexcept
  {
    return (comps[0].phi_disk_D3D(R,z) + comps[1].phi_disk_D3D(R,z) +
            comps[2].phi_disk_D3D(R,z));
  }

};


#endif /* POTENTIALS */
/*! \file gravity_cuda.cu
 *  \brief Definitions of functions to calculate gravitational
           acceleration in 1, 2, and 3D. Called in Update_Conserved_Variables
           functions in hydro_cuda.cu. */
#ifdef CUDA

  #pragma once

  #include <math.h>  // provides sqrt log cos sin atan etc.
  #include <stdio.h>

  #include "../global/global.h"  // provides GN etc.

// Work around lack of pow(Real,int) in Hip Clang for Rocm 3.5
static inline __device__ Real pow2(const Real x) { return x * x; }

inline __device__ void calc_g_1D(int xid, int x_off, int n_ghost, Real dx, Real xbound, Real *gx)
{
  Real x_pos, r_disk, r_halo;
  x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;

  // for disk components, calculate polar r
  // r_disk = 0.220970869121;
  // r_disk = 6.85009694274;
  r_disk = 13.9211647546;
  // r_disk = 20.9922325665;
  //  for halo, calculate spherical r
  r_halo = sqrt(x_pos * x_pos + r_disk * r_disk);

  // set properties of halo and disk (these must match initial conditions)
  Real a_disk_z, a_halo, M_vir, M_d, R_vir, R_d, z_d, R_h, M_h, c_vir, phi_0_h, x;
  M_vir   = 1.0e12;         // viral mass of MW in M_sun
  M_d     = 6.5e10;         // mass of disk in M_sun
  M_h     = M_vir - M_d;    // halo mass in M_sun
  R_vir   = 261;            // viral radius in kpc
  c_vir   = 20.0;           // halo concentration
  R_h     = R_vir / c_vir;  // halo scale length in kpc
  R_d     = 3.5;            // disk scale length in kpc
  z_d     = 3.5 / 5.0;      // disk scale height in kpc
  phi_0_h = GN * M_h / (log(1.0 + c_vir) - c_vir / (1.0 + c_vir));
  x       = r_halo / R_h;

  // calculate acceleration due to NFW halo & Miyamoto-Nagai disk
  a_halo = -phi_0_h * (log(1 + x) - x / (1 + x)) / (r_halo * r_halo);
  a_disk_z =
      -GN * M_d * x_pos * (R_d + sqrt(x_pos * x_pos + z_d * z_d)) /
      (pow(r_disk * r_disk + pow2(R_d + sqrt(x_pos * x_pos + z_d * z_d)), 1.5) * sqrt(x_pos * x_pos + z_d * z_d));

  // total acceleration is the sum of the halo + disk components
  *gx = (x_pos / r_halo) * a_halo + a_disk_z;

  return;
}

inline __device__ void calc_g_gresho_2D(int xid, int yid, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound,
                                 Real ybound, Real *gx, Real *gy)
{
  Real x_pos, y_pos, r, phi;
  // use the subgrid offset and global boundaries to calculate absolute
  // positions on the grid
  x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;
  y_pos = (y_off + yid - n_ghost + 0.5) * dy + ybound;
    // for Gresho, also need r & phi
    r   = sqrt(x_pos * x_pos + y_pos * y_pos);
    phi = atan2(y_pos, x_pos);

    // set acceleration to balance v_phi in Gresho problem
    if (r < 0.2) {
    *gx = -cos(phi)*25.0*r;
    *gy = -sin(phi)*25.0*r;
    }
    else if (r >= 0.2 && r < 0.4) {
    *gx = -cos(phi)*(4.0 - 20.0*r + 25.0*r*r)/r;
    *gy = -sin(phi)*(4.0 - 20.0*r + 25.0*r*r)/r;
    }
    else {
    *gx = 0.0;
    *gy = 0.0;
    }
  return;
}

inline __device__ void calc_g_keplerian_2D(int xid, int yid, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound,
                                 Real ybound, Real *gx, Real *gy)
{
  Real x_pos, y_pos, r, phi;
  // use the subgrid offset and global boundaries to calculate absolute
  // positions on the grid
  x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;
  y_pos = (y_off + yid - n_ghost + 0.5) * dy + ybound;
    r   = sqrt(x_pos * x_pos + y_pos * y_pos);
    phi = atan2(y_pos, x_pos);
     // set gravitational acceleration for Keplerian potential
    Real M;
    M = 1*MSUN_CGS;
    *gx = -cos(phi)*GN*M/(r*r);
    *gy = -sin(phi)*GN*M/(r*r);
  return;
}

inline __device__ void calc_g_kuzmin_2D(int xid, int yid, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound,
                                 Real ybound, Real *gx, Real *gy)
{
  Real x_pos, y_pos, r, phi;
  // use the subgrid offset and global boundaries to calculate absolute
  // positions on the grid
  x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;
  y_pos = (y_off + yid - n_ghost + 0.5) * dy + ybound;
    r   = sqrt(x_pos * x_pos + y_pos * y_pos);
    phi = atan2(y_pos, x_pos);
      // set gravitational acceleration for Kuzmin disk + NFW halo
      Real a_d, a_h, a, M_vir, M_d, R_vir, R_d, R_s, M_h, c_vir, x;
      M_vir = 1.0e12;         // viral mass of MW in M_sun
      M_d   = 6.5e10;         // mass of disk in M_sun (assume all gas)
      M_h   = M_vir - M_d;    // halo mass in M_sun
      R_vir = 261;            // viral radius in kpc
      c_vir = 20;             // halo concentration
      R_s   = R_vir / c_vir;  // halo scale length in kpc
      R_d   = 3.5;            // disk scale length in kpc

      // calculate acceleration
      x   = r / R_s;
      a_d = GN * M_d * r * pow(r * r + R_d * R_d, -1.5);
      a_h = GN * M_h * (log(1 + x) - x / (1 + x)) / ((log(1 + c_vir) - c_vir / (1 + c_vir)) * r * r);
      a   = a_d + a_h;

      *gx = -cos(phi) * a;
      *gy = -sin(phi) * a;
  return;
}

inline __device__ void calc_g_rayleigh_taylor_2D(int xid, int yid, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound,
                                 Real ybound, Real *gx, Real *gy)
{
  Real x_pos, y_pos;
  // use the subgrid offset and global boundaries to calculate absolute
  // positions on the grid
  x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;
  y_pos = (y_off + yid - n_ghost + 0.5) * dy + ybound;
      *gx = 0;
      *gy = -1;
  return;
}

inline __device__ void calc_g_3D(int xid, int yid, int zid, int x_off, int y_off, int z_off, int n_ghost, Real dx,
                                 Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real *gx, Real *gy, Real *gz)
{
  Real x_pos, y_pos, z_pos, r_disk, r_halo;
  // use the subgrid offset and global boundaries to calculate absolute
  // positions on the grid
  x_pos = (x_off + xid - n_ghost + 0.5) * dx + xbound;
  y_pos = (y_off + yid - n_ghost + 0.5) * dy + ybound;
  z_pos = (z_off + zid - n_ghost + 0.5) * dz + zbound;

  // for disk components, calculate polar r
  r_disk = sqrt(x_pos * x_pos + y_pos * y_pos);
  // for halo, calculate spherical r
  r_halo = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);

  // set properties of halo and disk (these must match initial conditions)
  Real a_disk_r, a_disk_z, a_halo, a_halo_r, a_halo_z;
  Real M_vir, M_d, R_vir, R_d, z_d, R_h, M_h, c_vir, phi_0_h, x;
  // MW model
  M_vir = 1.0e12;     // viral mass of in M_sun
  M_d   = 6.5e10;     // viral mass of in M_sun
  R_d   = 3.5;        // disk scale length in kpc
  z_d   = 3.5 / 5.0;  // disk scale height in kpc
  R_vir = 261.;       // virial radius in kpc
  c_vir = 20.0;       // halo concentration
  // M82 model
  // M_vir = 5.0e10; // viral mass of in M_sun
  // M_d = 1.0e10; // mass of disk in M_sun
  // R_d = 0.8; // disk scale length in kpc
  // z_d = 0.15; // disk scale height in kpc
  // R_vir = R_d/0.015; // viral radius in kpc
  // c_vir = 10.0; // halo concentration

  M_h     = M_vir - M_d;    // halo mass in M_sun
  R_h     = R_vir / c_vir;  // halo scale length in kpc
  phi_0_h = GN * M_h / (log(1.0 + c_vir) - c_vir / (1.0 + c_vir));
  x       = r_halo / R_h;

  // calculate acceleration due to NFW halo & Miyamoto-Nagai disk
  a_halo   = -phi_0_h * (log(1 + x) - x / (1 + x)) / (r_halo * r_halo);
  a_halo_r = a_halo * (r_disk / r_halo);
  a_halo_z = a_halo * (z_pos / r_halo);
  a_disk_r = -GN * M_d * r_disk * pow(r_disk * r_disk + pow2(R_d + sqrt(z_pos * z_pos + z_d * z_d)), -1.5);
  a_disk_z =
      -GN * M_d * z_pos * (R_d + sqrt(z_pos * z_pos + z_d * z_d)) /
      (pow(r_disk * r_disk + pow2(R_d + sqrt(z_pos * z_pos + z_d * z_d)), 1.5) * sqrt(z_pos * z_pos + z_d * z_d));

  // total acceleration is the sum of the halo + disk components
  *gx = (x_pos / r_disk) * (a_disk_r + a_halo_r);
  *gy = (y_pos / r_disk) * (a_disk_r + a_halo_r);
  *gz = a_disk_z + a_halo_z;

  return;
}

#endif  // CUDA

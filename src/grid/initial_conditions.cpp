/*! \file initial_conditions.cpp
 *  \brief Definitions of initial conditions for different tests.
           Note that the grid is mapped to 1D as i + (x_dim)*j +
 (x_dim*y_dim)*k. Functions are members of the Grid3D class. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../mpi/mpi_routines.h"
#include "../utils/error_handling.h"
#include "../utils/hydro_utilities.h"
#include "../utils/math_utilities.h"
#include "../utils/mhd_utilities.h"
#ifdef COSMOLOGY
  #include "../cosmology/cosmology.h"
#endif

/*! Set the initial conditions based on info in the parameters structure. */
void Grid3D::Set_Initial_Conditions(Parameters P, const ParameterMap &pmap)
{
  Set_Gammas(P.gamma);

  if (strcmp(P.init, "Constant") == 0 or strcmp(P.init, "Isolated_Stellar_Cluster") == 0) {
    Constant(P);
  } else if (strcmp(P.init, "Sound_Wave") == 0) {
    Sound_Wave(P);
  } else if (strcmp(P.init, "Linear_Wave") == 0) {
    Linear_Wave(P);
  } else if (strcmp(P.init, "Square_Wave") == 0) {
    Square_Wave(P);
  } else if (strcmp(P.init, "Riemann") == 0) {
    Riemann(P);
  } else if (strcmp(P.init, "Shu_Osher") == 0) {
    Shu_Osher();
  } else if (strcmp(P.init, "Blast_1D") == 0) {
    Blast_1D();
  } else if (strcmp(P.init, "KH") == 0) {
    KH();
  } else if (strcmp(P.init, "KH_res_ind") == 0) {
    KH_res_ind();
  } else if (strcmp(P.init, "Rayleigh_Taylor") == 0) {
    Rayleigh_Taylor();
  } else if (strcmp(P.init, "Implosion_2D") == 0) {
    Implosion_2D();
  } else if (strcmp(P.init, "Gresho") == 0) {
    Gresho();
  } else if (strcmp(P.init, "Noh_2D") == 0) {
    Noh_2D();
  } else if (strcmp(P.init, "Noh_3D") == 0) {
    Noh_3D();
  } else if (strcmp(P.init, "Disk_2D") == 0) {
    Disk_2D();
  } else if (strcmp(P.init, "Disk_3D") == 0 || strcmp(P.init, "Disk_3D_particles") == 0) {
    Disk_3D(P);
  } else if (strcmp(P.init, "Spherical_Overpressure_3D") == 0) {
    Spherical_Overpressure_3D();
  } else if (strcmp(P.init, "Spherical_Overdensity_3D") == 0) {
    Spherical_Overdensity_3D();
  } else if (strcmp(P.init, "Clouds") == 0) {
    Clouds();
  } else if (strcmp(P.init, "Read_Grid") == 0) {
#ifndef ONLY_PARTICLES
    Read_Grid(P);
#else   // ONLY_PARTICLES
    // Initialize a uniform hydro grid when only integrating particles
    Uniform_Grid();
#endif  // ONLY_PARTICLES
  } else if (strcmp(P.init, "Read_Grid_Cat") == 0) {
    Read_Grid_Cat(P);
  } else if (strcmp(P.init, "Uniform") == 0) {
    Uniform_Grid();
  } else if (strcmp(P.init, "Zeldovich_Pancake") == 0) {
    Zeldovich_Pancake(P);
  } else if (strcmp(P.init, "Adiabatic_Expansion") == 0) {
    Adiabatic_Expansion(P);
  } else if (strcmp(P.init, "Cosmological_ICs") == 0) {
    Cosmological_ICs(P);
  } else if (strcmp(P.init, "Chemistry_Test") == 0) {
    Chemistry_Test(P);
  } else if (strcmp(P.init, "Iliev0") == 0) {
    Iliev0(P);
  } else if (strcmp(P.init, "Iliev1") == 0) {
    Iliev15(P, 1);
  } else if (strcmp(P.init, "Iliev5") == 0) {
    Iliev15(P, 5);
  } else if (strcmp(P.init, "Iliev6") == 0) {
    Iliev6(P);
#ifdef MHD
  } else if (strcmp(P.init, "Circularly_Polarized_Alfven_Wave") == 0) {
    Circularly_Polarized_Alfven_Wave(P);
  } else if (strcmp(P.init, "Advecting_Field_Loop") == 0) {
    Advecting_Field_Loop(P);
  } else if (strcmp(P.init, "MHD_Spherical_Blast") == 0) {
    MHD_Spherical_Blast(P);
  } else if (strcmp(P.init, "Orszag_Tang_Vortex") == 0) {
    Orszag_Tang_Vortex();
#endif  // MHD
  } else {
    chprintf("ABORT: %s: Unknown initial conditions!\n", P.init);
    chexit(-1);
  }

  if (C.device != NULL) {
    GPU_Error_Check(cudaMemcpy(C.device, C.density, H.n_fields * H.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
  }
}

/*! \fn void Set_Domain_Properties(struct Parameters P)
 *  \brief Set local domain properties */
void Grid3D::Set_Domain_Properties(struct Parameters P)
{
  // Global Boundary Coordinates
  H.xbound = P.xmin;
  H.ybound = P.ymin;
  H.zbound = P.zmin;

  // Global Domain Lengths
  H.xdglobal = P.xlen;
  H.ydglobal = P.ylen;
  H.zdglobal = P.zlen;

#ifndef MPI_CHOLLA
  Real nx_param = (Real)(H.nx - 2 * H.n_ghost);
  Real ny_param = (Real)(H.ny - 2 * H.n_ghost);
  Real nz_param = (Real)(H.nz - 2 * H.n_ghost);

  // Local Boundary Coordinates
  H.xblocal = H.xbound;
  H.yblocal = H.ybound;
  H.zblocal = H.zbound;

  H.xblocal_max = H.xblocal + P.xlen;
  H.yblocal_max = H.yblocal + P.ylen;
  H.zblocal_max = H.zblocal + P.zlen;

#else
  Real nx_param = (Real)nx_global;
  Real ny_param = (Real)ny_global;
  Real nz_param = (Real)nz_global;

  // Local Boundary Coordinates
  /*
  H.xblocal = H.xbound + P.xlen * ((Real) nx_local_start) / nx_param;
  H.yblocal = H.ybound + P.ylen * ((Real) ny_local_start) / ny_param;
  H.zblocal = H.zbound + P.zlen * ((Real) nz_local_start) / nz_param;
  */
  H.xblocal = H.xbound + ((Real)nx_local_start) * (P.xlen / nx_param);
  H.yblocal = H.ybound + ((Real)ny_local_start) * (P.ylen / ny_param);
  H.zblocal = H.zbound + ((Real)nz_local_start) * (P.zlen / nz_param);

  H.xblocal_max = H.xbound + ((Real)(nx_local_start + H.nx - 2 * H.n_ghost)) * (P.xlen / nx_param);
  H.yblocal_max = H.ybound + ((Real)(ny_local_start + H.ny - 2 * H.n_ghost)) * (P.ylen / ny_param);
  H.zblocal_max = H.zbound + ((Real)(nz_local_start + H.nz - 2 * H.n_ghost)) * (P.zlen / nz_param);

#endif

  /*perform 1-D first*/
  if (H.nx > 1 && H.ny == 1 && H.nz == 1) {
    H.dx = P.xlen / nx_param;
    H.dy = P.ylen;
    H.dz = P.zlen;
  }

  /*perform 2-D next*/
  if (H.nx > 1 && H.ny > 1 && H.nz == 1) {
    H.dx = P.xlen / nx_param;
    H.dy = P.ylen / ny_param;
    H.dz = P.zlen;
  }

  /*perform 3-D last*/
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    H.dx = P.xlen / nx_param;
    H.dy = P.ylen / ny_param;
    H.dz = P.zlen / nz_param;
  }
}

/*! \fn void Constant(Real rho, Real vx, Real vy, Real vz, Real P, Real Bx, Real
 * By, Real Bz) \brief Constant gas properties. */
void Grid3D::Constant(Parameters const &P)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;
  Real mu = 0.6;
  Real n, T;

  istart = H.n_ghost;
  iend   = H.nx - H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny - H.n_ghost;
  } else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz - H.n_ghost;
  } else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for (k = kstart - 1; k < kend; k++) {
    for (j = jstart - 1; j < jend; j++) {
      for (i = istart - 1; i < iend; i++) {
        // get cell index
        id = i + j * H.nx + k * H.nx * H.ny;

// Set the magnetic field including the rightmost ghost cell on the
// left side which is really the left face of the first grid cell
#ifdef MHD
        C.magnetic_x[id] = P.Bx;
        C.magnetic_y[id] = P.By;
        C.magnetic_z[id] = P.Bz;
#endif  // MHD

        // Exclude the rightmost ghost cell on the "left" side
        if ((k >= kstart) and (j >= jstart) and (i >= istart)) {
          // set constant initial states
          C.density[id]    = P.rho;
          C.momentum_x[id] = P.rho * P.vx;
          C.momentum_y[id] = P.rho * P.vy;
          C.momentum_z[id] = P.rho * P.vz;
          C.Energy[id]     = P.P / (gama - 1.0) + 0.5 * P.rho * (P.vx * P.vx + P.vy * P.vy + P.vz * P.vz);
#ifdef DE
          C.GasEnergy[id] = P.P / (gama - 1.0);
#endif  // DE
        }
        if (i == istart && j == jstart && k == kstart) {
          n = P.rho * DENSITY_UNIT / (mu * MP);
          T = P.P * PRESSURE_UNIT / (n * KB);
          printf("Initial n = %e, T = %e\n", n, T);
        }
      }
    }
  }
}

/*! \fn void Sound_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
 *  \brief Sine wave perturbation. */
void Grid3D::Sound_Wave(Parameters const &P)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;

  istart = H.n_ghost;
  iend   = H.nx - H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny - H.n_ghost;
  } else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz - H.n_ghost;
  } else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for (k = kstart; k < kend; k++) {
    for (j = jstart; j < jend; j++) {
      for (i = istart; i < iend; i++) {
        // get cell index
        id = i + j * H.nx + k * H.nx * H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        // set constant initial states
        C.density[id]    = P.rho;
        C.momentum_x[id] = P.rho * P.vx;
        C.momentum_y[id] = P.rho * P.vy;
        C.momentum_z[id] = P.rho * P.vz;
        C.Energy[id]     = P.P / (gama - 1.0) + 0.5 * P.rho * (P.vx * P.vx + P.vy * P.vy + P.vz * P.vz);
        // add small-amplitude perturbations
        C.density[id]    = C.density[id] + P.A * sin(2.0 * M_PI * x_pos);
        C.momentum_x[id] = C.momentum_x[id] + P.A * sin(2.0 * M_PI * x_pos);
        C.momentum_y[id] = C.momentum_y[id] + P.A * sin(2.0 * M_PI * x_pos);
        C.momentum_z[id] = C.momentum_z[id] + P.A * sin(2.0 * M_PI * x_pos);
        C.Energy[id]     = C.Energy[id] + P.A * (1.5) * sin(2 * M_PI * x_pos);
#ifdef DE
        C.GasEnergy[id] = P.P / (gama - 1.0);
#endif  // DE
#ifdef DE
        C.GasEnergy[id] = P.P / (gama - 1.0);
#endif  // DE
      }
    }
  }
}

/*! \fn void Linear_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
 *  \brief Sine wave perturbation. */
void Grid3D::Linear_Wave(Parameters const &P)
{
  // Compute any test parameters needed
  // ==================================
  // Angles
  Real const sin_yaw   = std::sin(P.yaw);
  Real const cos_yaw   = std::cos(P.yaw);
  Real const sin_pitch = std::sin(P.pitch);
  Real const cos_pitch = std::cos(P.pitch);

  Real const wavenumber = 2.0 * M_PI / P.wave_length;  // the angular wave number k

#ifdef MHD
  // TODO: This method of setting the magnetic fields via the vector potential should work but instead leads to small
  // TODO: errors in the magnetic field that tend to amplify over time until the solution diverges. I don't know why
  // TODO: that is the case and can't figure out the reason. Without this we can't run linear waves at an angle to the
  // TODO: grid.
  // // Compute the vector potential
  // // ============================
  // std::vector<Real> vectorPotential(3 * H.n_cells, 0);

  // // lambda function for computing the vector potential
  // auto Compute_Vector_Potential = [&](Real const &x_loc, Real const &y_loc, Real const &z_loc) {
  //   // The "_rot" variables are the rotated version
  //   Real const x_rot = x_loc * cos_pitch * cos_yaw + y_loc * cos_pitch * sin_yaw + z_loc * sin_pitch;
  //   Real const y_rot = -x_loc * sin_yaw + y_loc * cos_yaw;

  //   Real const a_y = P.Bz * x_rot - (P.A * P.rEigenVec_Bz / wavenumber) * std::cos(wavenumber * x_rot);
  //   Real const a_z = -P.By * x_rot + (P.A * P.rEigenVec_By / wavenumber) * std::cos(wavenumber * x_rot) + P.Bx *
  //   y_rot;

  //   return std::make_pair(a_y, a_z);
  // };

  // for (size_t k = 0; k < H.nz; k++) {
  //   for (size_t j = 0; j < H.ny; j++) {
  //     for (size_t i = 0; i < H.nx; i++) {
  //       // Get cell index
  //       size_t const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

  //       Real x, y, z;
  //       Get_Position(i, j, k, &x, &y, &z);

  //       auto vectorPot                         = Compute_Vector_Potential(x, y + H.dy / 2., z + H.dz / 2.);
  //       vectorPotential.at(id + 0 * H.n_cells) = -vectorPot.first * sin_yaw - vectorPot.second * sin_pitch * cos_yaw;

  //       vectorPot                              = Compute_Vector_Potential(x + H.dx / 2., y, z + H.dz / 2.);
  //       vectorPotential.at(id + 1 * H.n_cells) = vectorPot.first * cos_yaw - vectorPot.second * sin_pitch * sin_yaw;

  //       vectorPot                              = Compute_Vector_Potential(x + H.dx / 2., y + H.dy / 2., z);
  //       vectorPotential.at(id + 2 * H.n_cells) = vectorPot.second * cos_pitch;
  //     }
  //   }
  // }

  // // Compute the magnetic field from the vector potential
  // // ====================================================
  // mhd::utils::Init_Magnetic_Field_With_Vector_Potential(H, C, vectorPotential);

  Real shift = H.dx;
  size_t dir = 0;
  if (sin_yaw == 1.0) {
    shift = H.dy;
    dir   = 1;
  } else if (sin_pitch == 1.0) {
    shift = H.dz;
    dir   = 2;
  }

  // set initial values of conserved variables
  for (int k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        // get cell index
        size_t const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        // get cell-centered position
        Real x_pos, y_pos, z_pos;
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        Real const x_pos_rot = cos_pitch * (x_pos * cos_yaw + y_pos * sin_yaw) + z_pos * sin_pitch;

        Real const sine_x = std::sin(x_pos_rot * wavenumber);

        Real bx = P.Bx + P.A * P.rEigenVec_Bx * sine_x;
        Real by = P.By + P.A * P.rEigenVec_By * sine_x;
        Real bz = P.Bz + P.A * P.rEigenVec_Bz * sine_x;

        C.magnetic_x[id] = bx * cos_pitch * cos_yaw - by * sin_yaw - bz * sin_pitch * cos_yaw;
        C.magnetic_y[id] = bx * cos_pitch * sin_yaw + by * cos_yaw - bz * sin_pitch * sin_yaw;
        C.magnetic_z[id] = bx * sin_pitch + bz * cos_pitch;
      }
    }
  }
#endif  // MHD

  // Compute the hydro variables
  // ===========================
  for (size_t k = H.n_ghost - 1; k < H.nz - H.n_ghost; k++) {
    for (size_t j = H.n_ghost - 1; j < H.ny - H.n_ghost; j++) {
      for (size_t i = H.n_ghost - 1; i < H.nx - H.n_ghost; i++) {
        // get cell index
        size_t const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        // get cell-centered position
        Real x_pos, y_pos, z_pos;
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        Real const x_pos_rot = cos_pitch * (x_pos * cos_yaw + y_pos * sin_yaw) + z_pos * sin_pitch;

        Real const sine_x = std::sin(x_pos_rot * wavenumber);

        // Density
        C.density[id] = P.rho + P.A * P.rEigenVec_rho * sine_x;

        // Momenta
        Real mx = P.rho * P.vx + P.A * P.rEigenVec_MomentumX * sine_x;
        Real my = P.A * P.rEigenVec_MomentumY * sine_x;
        Real mz = P.A * P.rEigenVec_MomentumZ * sine_x;

        C.momentum_x[id] = mx * cos_pitch * cos_yaw - my * sin_yaw - mz * sin_pitch * cos_yaw;
        C.momentum_y[id] = mx * cos_pitch * sin_yaw + my * cos_yaw - mz * sin_pitch * sin_yaw;
        C.momentum_z[id] = mx * sin_pitch + mz * cos_pitch;

        // Energy
        C.Energy[id] = P.P / (P.gamma - 1.0) + 0.5 * P.rho * P.vx * P.vx + P.A * sine_x * P.rEigenVec_E;
#ifdef MHD
        C.Energy[id] += 0.5 * (P.Bx * P.Bx + P.By * P.By + P.Bz * P.Bz);
#endif  // MHD
      }
    }
  }
}

/*! \fn void Square_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
 *  \brief Square wave density perturbation with amplitude A*rho in pressure
 * equilibrium. */
void Grid3D::Square_Wave(Parameters const &P)
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;

  istart = H.n_ghost;
  iend   = H.nx - H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny - H.n_ghost;
  } else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz - H.n_ghost;
  } else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for (k = kstart; k < kend; k++) {
    for (j = jstart; j < jend; j++) {
      for (i = istart; i < iend; i++) {
        // get cell index
        id = i + j * H.nx + k * H.nx * H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        C.density[id] = P.rho;
        // C.momentum_x[id] = 0.0;
        C.momentum_x[id] = P.rho * P.vx;
        C.momentum_y[id] = P.rho * P.vy;
        C.momentum_z[id] = P.rho * P.vz;
        // C.momentum_z[id] = rho_l * v_l;
        C.Energy[id] = P.P / (gama - 1.0) + 0.5 * P.rho * (P.vx * P.vx + P.vy * P.vy + P.vz * P.vz);
#ifdef DE
        C.GasEnergy[id] = P.P / (gama - 1.0);
#endif
#ifdef SCALAR
  #ifdef BASIC_SCALAR
        C.basic_scalar[id] = C.density[id] * 0.0;
  #endif
#endif
        if (x_pos > 0.25 * H.xdglobal && x_pos < 0.75 * H.xdglobal) {
          C.density[id]    = P.rho * P.A;
          C.momentum_x[id] = P.rho * P.A * P.vx;
          C.momentum_y[id] = P.rho * P.A * P.vy;
          C.momentum_z[id] = P.rho * P.A * P.vz;
          C.Energy[id]     = P.P / (gama - 1.0) + 0.5 * P.rho * P.A * (P.vx * P.vx + P.vy * P.vy + P.vz * P.vz);
#ifdef DE
          C.GasEnergy[id] = P.P / (gama - 1.0);
#endif
#ifdef SCALAR
  #ifdef BASIC_SCALAR
          C.basic_scalar[id] = C.density[id] * 1.0;
  #endif
#endif
        }
      }
    }
  }
}

/*! \fn void Riemann(Real rho_l, Real vx_l, Real vy_l, Real vz_l, Real P_l, Real
 Bx_l, Real By_l, Real Bz_l, Real rho_r, Real vx_r, Real vy_r, Real vz_r, Real
 P_r, Real Bx_r, Real By_r, Real Bz_r, Real diaph)
 *  \brief Initialize the grid with a Riemann problem. */
void Grid3D::Riemann(Parameters const &P)
{
  size_t const istart = H.n_ghost - 1;
  size_t const iend   = H.nx - H.n_ghost;
  size_t jstart, kstart, jend, kend;
  if (H.ny > 1) {
    jstart = H.n_ghost - 1;
    jend   = H.ny - H.n_ghost;
  } else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost - 1;
    kend   = H.nz - H.n_ghost;
  } else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for (size_t k = kstart; k < kend; k++) {
    for (size_t j = jstart; j < jend; j++) {
      for (size_t i = istart; i < iend; i++) {
        // get cell index
        size_t const id = i + j * H.nx + k * H.nx * H.ny;

        // get cell-centered position
        Real x_pos, y_pos, z_pos;
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

#ifdef MHD
        // Set the magnetic field including the rightmost ghost cell on the
        // left side which is really the left face of the first grid cell
        // WARNING: Only correct in 3-D
        if (x_pos < P.diaph) {
          C.magnetic_x[id] = P.Bx_l;
          C.magnetic_y[id] = P.By_l;
          C.magnetic_z[id] = P.Bz_l;
        } else {
          C.magnetic_x[id] = P.Bx_r;
          C.magnetic_y[id] = P.By_r;
          C.magnetic_z[id] = P.Bz_r;
        }
#endif  // MHD

        // Exclude the rightmost ghost cell on the "left" side
        if ((k >= kstart) and (j >= jstart) and (i >= istart)) {
          if (x_pos < P.diaph) {
            C.density[id]    = P.rho_l;
            C.momentum_x[id] = P.rho_l * P.vx_l;
            C.momentum_y[id] = P.rho_l * P.vy_l;
            C.momentum_z[id] = P.rho_l * P.vz_l;
            C.Energy[id] = hydro_utilities::Calc_Energy_Primitive(P.P_l, P.rho_l, P.vx_l, P.vy_l, P.vz_l, gama, P.Bx_l,
                                                                  P.By_l, P.Bz_l);
#ifdef SCALAR
  #ifdef BASIC_SCALAR
            C.basic_scalar[id] = 1.0 * P.rho_l;
  #endif
#endif  // SCALAR
#ifdef DE
            C.GasEnergy[id] = P.P_l / (gama - 1.0);
#endif  // DE
          } else {
            C.density[id]    = P.rho_r;
            C.momentum_x[id] = P.rho_r * P.vx_r;
            C.momentum_y[id] = P.rho_r * P.vy_r;
            C.momentum_z[id] = P.rho_r * P.vz_r;
            C.Energy[id] = hydro_utilities::Calc_Energy_Primitive(P.P_r, P.rho_r, P.vx_r, P.vy_r, P.vz_r, gama, P.Bx_r,
                                                                  P.By_r, P.Bz_r);
#ifdef SCALAR
  #ifdef BASIC_SCALAR
            C.basic_scalar[id] = 0.0 * P.rho_r;
  #endif
#endif  // SCALAR
#ifdef DE
            C.GasEnergy[id] = P.P_r / (gama - 1.0);
#endif  // DE
          }
        }
      }
    }
  }
}

/*! \fn void Shu_Osher()
 *  \brief Initialize the grid with the Shu-Osher shock tube problem. See Stone
 * 2008, Section 8.1 */
void Grid3D::Shu_Osher()
{
  int i, id;
  Real x_pos, y_pos, z_pos;
  Real vx, P;

  // set initial values of conserved variables
  for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
    id = i;
    // get centered x position
    Get_Position(i, H.n_ghost, H.n_ghost, &x_pos, &y_pos, &z_pos);

    if (x_pos < -0.8) {
      C.density[id]    = 3.857143;
      vx               = 2.629369;
      C.momentum_x[id] = C.density[id] * vx;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      P                = 10.33333;
      C.Energy[id]     = P / (gama - 1.0) + 0.5 * C.density[id] * vx * vx;
    } else {
      C.density[id]    = 1.0 + 0.2 * sin(5.0 * M_PI * x_pos);
      Real vx          = 0.0;
      C.momentum_x[id] = C.density[id] * vx;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      Real P           = 1.0;
      C.Energy[id]     = P / (gama - 1.0) + 0.5 * C.density[id] * vx * vx;
    }
#ifdef DE
    C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE
  }
}

/*! \fn void Blast_1D()
 *  \brief Initialize the grid with two interacting blast waves. See Stone 2008,
 * Section 8.1.*/
void Grid3D::Blast_1D()
{
  int i, id;
  Real x_pos, y_pos, z_pos;
  Real vx, P;

  // set initial values of conserved variables
  for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
    id = i;
    // get the centered x position
    Get_Position(i, H.n_ghost, H.n_ghost, &x_pos, &y_pos, &z_pos);

    if (x_pos < 0.1) {
      C.density[id]    = 1.0;
      C.momentum_x[id] = 0.0;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      P                = 1000.0;
    } else if (x_pos > 0.9) {
      C.density[id]    = 1.0;
      C.momentum_x[id] = 0.0;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      P                = 100;
    } else {
      C.density[id]    = 1.0;
      C.momentum_x[id] = 0.0;
      C.momentum_y[id] = 0.0;
      C.momentum_z[id] = 0.0;
      P                = 0.01;
    }
    C.Energy[id] = P / (gama - 1.0);
#ifdef DE
    C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE
  }
}

/*! \fn void KH()
 *  \brief Initialize the grid with a Kelvin-Helmholtz instability.
           This version of KH test has a discontinuous boundary.
           Use KH_res_ind for a version that is resolution independent. */
void Grid3D::KH()
{
  int i, j, k, id;
  int istart, iend, jstart, jend, kstart, kend;
  Real x_pos, y_pos, z_pos;
  Real vx, vy, vz;
  Real d1, d2, v1, v2, P, A;

  d1 = 2.0;
  d2 = 1.0;
  v1 = 0.5;
  v2 = -0.5;
  P  = 2.5;
  A  = 0.1;

  istart = H.n_ghost;
  iend   = H.nx - H.n_ghost;
  jstart = H.n_ghost;
  jend   = H.ny - H.n_ghost;
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz - H.n_ghost;
  } else {
    kstart = 0;
    kend   = H.nz;
  }

  // set the initial values of the conserved variables
  for (k = kstart; k < kend; k++) {
    for (j = jstart; j < jend; j++) {
      for (i = istart; i < iend; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;
        // get the centered x and y positions
        Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

        // outer quarters of slab
        if ((y_pos <= 1.0 * H.ydglobal / 4.0) or (y_pos >= 3.0 * H.ydglobal / 4.0)) {
          C.density[id]    = d2;
          C.momentum_x[id] = v2 * C.density[id];
          C.momentum_y[id] = C.density[id] * A * sin(4 * M_PI * x_pos);
          C.momentum_z[id] = 0.0;
#ifdef SCALAR
  #ifdef BASIC_SCALAR
          C.basic_scalar[id] = 0.0;
  #endif
#endif
          // inner half of slab
        } else {
          C.density[id]    = d1;
          C.momentum_x[id] = v1 * C.density[id];
          C.momentum_y[id] = C.density[id] * A * sin(4 * M_PI * x_pos);
          C.momentum_z[id] = 0.0;
#ifdef SCALAR
  #ifdef BASIC_SCALAR
          C.basic_scalar[id] = 1.0 * d1;
  #endif
#endif
        }
        C.Energy[id] =
            P / (gama - 1.0) +
            0.5 * (C.momentum_x[id] * C.momentum_x[id] + C.momentum_y[id] * C.momentum_y[id]) / C.density[id];
#ifdef DE
        C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE
      }
    }
  }
}

/*! \fn void KH_res_ind()
 *  \brief Initialize the grid with a Kelvin-Helmholtz instability whose modes
 * are resolution independent. */
void Grid3D::KH_res_ind()
{
  int i, j, k, id;
  int istart, iend, jstart, jend, kstart, kend;
  Real x_pos, y_pos, z_pos;
  Real mx, my, mz;
  Real r, yc, zc, phi;
  Real d1, d2, v1, v2, P, dy, A;
  istart = H.n_ghost;
  iend   = H.nx - H.n_ghost;
  jstart = H.n_ghost;
  jend   = H.ny - H.n_ghost;
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz - H.n_ghost;
  } else {
    kstart = 0;
    kend   = H.nz;
  }

  // y, z center of cylinder (assuming x is long direction)
  yc = 0.0;
  zc = 0.0;

  d1 = 100.0;  // inner density
  d2 = 1.0;    // outer density
  v1 = 0.5;    // inner velocity
  v2 = -0.5;   // outer velocity
  P  = 2.5;    // pressure
  dy = 0.05;   // width of ramp function (see Robertson 2009)
  A  = 0.1;    // amplitude of the perturbation

  // Note: ramp function from Robertson 2009 is 1/Ramp(y) = (1 +
  // exp(2*(y-0.25)/dy))*(1 + exp(2*(0.75 - y)/dy));

  // set the initial values of the conserved variables
  for (k = kstart; k < kend; k++) {
    for (j = jstart; j < jend; j++) {
      for (i = istart; i < iend; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;
        // get the centered x and y positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        // 2D initial conditions:
        if (H.nz == 1) {
          // inner fluid
          if (fabs(y_pos - 0.5) < 0.25) {
            if (y_pos > 0.5) {
              C.density[id] =
                  d1 - (d1 - d2) * exp(-0.5 * pow(y_pos - 0.75 - sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
              C.momentum_x[id] = v1 * C.density[id] -
                                 C.density[id] * (v1 - v2) *
                                     exp(-0.5 * pow(y_pos - 0.75 - sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
              C.momentum_y[id] = C.density[id] * A * sin(4 * M_PI * x_pos) *
                                 exp(-0.5 * pow(y_pos - 0.75 - sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            } else {
              C.density[id] =
                  d1 - (d1 - d2) * exp(-0.5 * pow(y_pos - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
              C.momentum_x[id] = v1 * C.density[id] -
                                 C.density[id] * (v1 - v2) *
                                     exp(-0.5 * pow(y_pos - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
              C.momentum_y[id] = C.density[id] * A * sin(4 * M_PI * x_pos) *
                                 exp(-0.5 * pow(y_pos - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            }
          }
          // outer fluid
          else {
            if (y_pos > 0.5) {
              C.density[id] =
                  d2 + (d1 - d2) * exp(-0.5 * pow(y_pos - 0.75 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
              C.momentum_x[id] = v2 * C.density[id] +
                                 C.density[id] * (v1 - v2) *
                                     exp(-0.5 * pow(y_pos - 0.75 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
              C.momentum_y[id] = C.density[id] * A * sin(4 * M_PI * x_pos) *
                                 exp(-0.5 * pow(y_pos - 0.75 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            } else {
              C.density[id] =
                  d2 + (d1 - d2) * exp(-0.5 * pow(y_pos - 0.25 - sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
              C.momentum_x[id] = v2 * C.density[id] +
                                 C.density[id] * (v1 - v2) *
                                     exp(-0.5 * pow(y_pos - 0.25 - sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
              C.momentum_y[id] = C.density[id] * A * sin(4 * M_PI * x_pos) *
                                 exp(-0.5 * pow(y_pos - 0.25 - sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            }
          }
          // C.momentum_y[id] = C.density[id] * A*sin(4*PI*x_pos);
          C.momentum_z[id] = 0.0;

          // 3D initial conditions:
        } else {
          // cylindrical version (3D only)
          r   = sqrt((z_pos - zc) * (z_pos - zc) + (y_pos - yc) * (y_pos - yc));  // center the cylinder at yc, zc
          phi = atan2((z_pos - zc), (y_pos - yc));

          if (r < 0.25)  // inside the cylinder
          {
            C.density[id] = d1 - (d1 - d2) * exp(-0.5 * pow(r - 0.25 - sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            C.momentum_x[id] =
                v1 * C.density[id] -
                C.density[id] * exp(-0.5 * pow(r - 0.25 - sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            C.momentum_y[id] = cos(phi) * C.density[id] * A * sin(4 * M_PI * x_pos) *
                               exp(-0.5 * pow(r - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            C.momentum_z[id] = sin(phi) * C.density[id] * A * sin(4 * M_PI * x_pos) *
                               exp(-0.5 * pow(r - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
          } else  // outside the cylinder
          {
            C.density[id] = d2 + (d1 - d2) * exp(-0.5 * pow(r - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            C.momentum_x[id] =
                v2 * C.density[id] +
                C.density[id] * exp(-0.5 * pow(r - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy));
            C.momentum_y[id] = cos(phi) * C.density[id] * A * sin(4 * M_PI * x_pos) *
                               (1.0 - exp(-0.5 * pow(r - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy)));
            C.momentum_z[id] = sin(phi) * C.density[id] * A * sin(4 * M_PI * x_pos) *
                               (1.0 - exp(-0.5 * pow(r - 0.25 + sqrt(-2.0 * dy * dy * log(0.5)), 2) / (dy * dy)));
          }
        }

        // No matter what we do with the density and momentum, set the Energy
        // and GasEnergy appropriately
        mx           = C.momentum_x[id];
        my           = C.momentum_y[id];
        mz           = C.momentum_z[id];
        C.Energy[id] = P / (gama - 1.0) + 0.5 * (mx * mx + my * my + mz * mz) / C.density[id];

#ifdef DE
        C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE

      }  // i loop
    }    // j loop
  }      // k loop
}

/*! \fn void Rayleigh_Taylor()
 *  \brief Initialize the grid with a 2D Rayleigh-Taylor instability. */
void Grid3D::Rayleigh_Taylor()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real dl, du, vy, g, P, P_0;
  dl = 1.0;
  du = 2.0;
  g  = -0.1;

  // set the initial values of the conserved variables
  for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
    for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
      id = i + j * H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      // set the y velocities (small perturbation tapering off from center)
      vy = 0.01 * cos(6 * M_PI * x_pos + M_PI) * exp(-(y_pos - 0.5 * H.ydglobal) * (y_pos - 0.5 * H.ydglobal) / 0.1);
      // vy = 0.0;

      // lower half of slab
      if (y_pos <= 0.5 * H.ydglobal) {
        P_0              = 1.0 / gama - dl * g * 0.5;
        P                = P_0 + dl * g * y_pos;
        C.density[id]    = dl;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = dl * vy;
        C.momentum_z[id] = 0.0;
      }
      // upper half of slab
      else {
        P_0              = 1.0 / gama - du * g * 0.5;
        P                = P_0 + du * g * y_pos;
        C.density[id]    = du;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = du * vy;
        C.momentum_z[id] = 0.0;
      }

      C.Energy[id] = P / (gama - 1.0) + 0.5 * (C.momentum_y[id] * C.momentum_y[id]) / C.density[id];
#ifdef DE
      C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE
    }
  }
}

/*! \fn void Gresho()
 *  \brief Initialize the grid with the 2D Gresho problem described in LW03. */
void Grid3D::Gresho()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos, xc, yc, r, phi;
  Real d, vx, vy, P, v_boost;
  Real x, y, dx, dy;
  int ran, N;
  N       = 100000;
  d       = 1.0;
  v_boost = 0.0;

  // center the vortex at (0.0,0.0)
  xc = 0.0;
  yc = 0.0;

  // seed the random number generator
  srand(0);

  // set the initial values of the conserved variables
  for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
    for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
      id = i + j * H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      // calculate centered radial position and phi
      r   = sqrt((x_pos - xc) * (x_pos - xc) + (y_pos - yc) * (y_pos - yc));
      phi = atan2((y_pos - yc), (x_pos - xc));

      /*
            // set vx, vy, P to zero before integrating
            vx = 0.0;
            vy = 0.0;
            P = 0.0;

            // monte carlo sample to get an integrated value for vx, vy, P
            for (int ii = 0; ii<N; ii++) {
              // get a random dx and dy to sample within the cell
              ran = rand() % 1000;
              dx = H.dx*(ran/1000.0 - 0.5);
              ran = rand() % 1000;
              dy = H.dy*(ran/1000.0 - 0.5);
              x = x_pos + dx;
              y = y_pos + dy;
              // calculate r and phi using the new x & y positions
              r = sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc));
              phi = atan2((y-yc), (x-xc));
              if (r < 0.2) {
                vx += -sin(phi)*5.0*r + v_boost;
                vy += cos(phi)*5.0*r;
                P += 5.0 + 0.5*25.0*r*r;
              }
              else if (r >= 0.2 && r < 0.4) {
                vx += -sin(phi)*(2.0-5.0*r) + v_boost;
                vy += cos(phi)*(2.0-5.0*r);
                P += 9.0 - 4.0*log(0.2) + 0.5*25.0*r*r - 20.0*r + 4.0*log(r);
              }
              else {
                vx += 0.0;
                vy += 0.0;
                P += 3.0 + 4.0*log(2.0);
              }
            }
            vx = vx/N;
            vy = vy/N;
            P = P/N;
      */
      if (r < 0.2) {
        vx = -sin(phi) * 5.0 * r + v_boost;
        vy = cos(phi) * 5.0 * r;
        P  = 5.0 + 0.5 * 25.0 * r * r;
      } else if (r >= 0.2 && r < 0.4) {
        vx = -sin(phi) * (2.0 - 5.0 * r) + v_boost;
        vy = cos(phi) * (2.0 - 5.0 * r);
        P  = 9.0 - 4.0 * log(0.2) + 0.5 * 25.0 * r * r - 20.0 * r + 4.0 * log(r);
      } else {
        vx = 0.0;
        vy = 0.0;
        P  = 3.0 + 4.0 * log(2.0);
      }
      // set P constant for modified Gresho problem
      // P = 5.5;

      // set values of conserved variables
      C.density[id]    = d;
      C.momentum_x[id] = d * vx;
      C.momentum_y[id] = d * vy;
      C.momentum_z[id] = 0.0;
      C.Energy[id]     = P / (gama - 1.0) + 0.5 * d * (vx * vx + vy * vy);
#ifdef DE
      C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE

      // r = sqrt((x_pos-xc)*(x_pos-xc) + (y_pos-yc)*(y_pos-yc));
      // printf("%f %f %f %f %f\n", x_pos, y_pos, r, vx, vy);
    }
  }
}

/*! \fn void Implosion_2D()
 *  \brief Implosion test described in Liska, 2003. */
void Grid3D::Implosion_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real P;

  // set the initial values of the conserved variables
  for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
    for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
      id = i + j * H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      // inner corner of box
      if (y_pos < (0.1500001 - x_pos)) {
        C.density[id]    = 0.125;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        P                = 0.14;
        C.Energy[id]     = P / (gama - 1.0);
#ifdef DE
        C.GasEnergy[id] = P / (gama - 1.0);
#endif
      }
      // everywhere else
      else {
        C.density[id]    = 1.0;
        C.momentum_x[id] = 0.0;
        C.momentum_y[id] = 0.0;
        C.momentum_z[id] = 0.0;
        P                = 1.0;
        C.Energy[id]     = P / (gama - 1.0);
#ifdef DE
        C.GasEnergy[id] = P / (gama - 1.0);
#endif
      }
    }
  }
}

/*! \fn void Noh_2D()
 *  \brief Noh test described in Liska, 2003. */
void Grid3D::Noh_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos;
  Real vx, vy, P, r;

  P = 1.0e-6;
  // set the initial values of the conserved variables
  for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
    for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
      id = i + j * H.nx;
      // get the centered x and y positions at (x,y,z)
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      C.density[id]    = 1.0;
      r                = sqrt(x_pos * x_pos + y_pos * y_pos);
      vx               = x_pos / r;
      vy               = y_pos / r;
      C.momentum_x[id] = -x_pos / r;
      C.momentum_y[id] = -y_pos / r;
      C.momentum_z[id] = 0.0;
      C.Energy[id]     = P / (gama - 1.0) + 0.5;
#ifdef DE
      C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE
    }
  }
}

/*! \fn void Noh_3D()
 *  \brief Noh test described in Stone, 2008. */
void Grid3D::Noh_3D()
{
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r;

  Real P = 1.0e-6;

  // set the initial values of the conserved variables
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        C.density[id]    = 1.0;
        r                = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);
        C.momentum_x[id] = -x_pos / r;
        C.momentum_y[id] = -y_pos / r;
        C.momentum_z[id] = -z_pos / r;
        C.Energy[id]     = P / (gama - 1.0) + 0.5;
#ifdef DE
        C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE
      }
    }
  }
}

/*! \fn void Disk_2D()
 *  \brief Initialize the grid with a 2D disk following a Kuzmin profile. */
void Grid3D::Disk_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos, r, phi;
  Real d, n, a, a_d, a_h, v, vx, vy, P, T_d, x;
  Real M_vir, M_h, M_d, c_vir, R_vir, R_h, R_d, Sigma;

  M_vir = 1.0e12;         // viral mass of MW in M_sun
  M_d   = 6.5e10;         // mass of disk in M_sun
  M_h   = M_vir - M_d;    // halo mass in M_sun
  R_vir = 261;            // viral radius in kpc
  c_vir = 20;             // halo concentration
  R_h   = R_vir / c_vir;  // halo scale length in kpc
  R_d   = 3.5;            // disk scale length in kpc
  T_d   = 10000;          // disk temperature, 10^4K

  // set the initial values of the conserved variables
  for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
    for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
      id = i + j * H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);

      // calculate centered radial position and phi
      r   = sqrt(x_pos * x_pos + y_pos * y_pos);
      phi = atan2(y_pos, x_pos);

      // Disk surface density [M_sun / kpc^2]
      // Assume gas surface density is exponential with scale length 2*R_d and
      // mass 0.25*M_d
      Sigma = 0.25 * M_d * exp(-r / (2 * R_d)) / (8 * M_PI * R_d * R_d);
      d     = Sigma;                         // just use sigma for mass density since height is arbitrary
      n     = d * DENSITY_UNIT / MP;         // number density, cgs
      P     = n * KB * T_d / PRESSURE_UNIT;  // disk pressure, code units

      // radial acceleration due to Kuzmin disk + NFW halo
      x   = r / R_h;
      a_d = GN * M_d * r * pow(r * r + R_d * R_d, -1.5);
      a_h = GN * M_h * (log(1 + x) - x / (1 + x)) / ((log(1 + c_vir) - c_vir / (1 + c_vir)) * r * r);
      a   = a_d + a_h;

      // circular velocity
      v  = sqrt(r * a);
      vx = -sin(phi) * v;
      vy = cos(phi) * v;

      // set values of conserved variables
      C.density[id]    = d;
      C.momentum_x[id] = d * vx;
      C.momentum_y[id] = d * vy;
      C.momentum_z[id] = 0.0;
      C.Energy[id]     = P / (gama - 1.0) + 0.5 * d * (vx * vx + vy * vy);

#ifdef DE
      C.GasEnergy[id] = P / (gama - 1.0);
#endif  // DE
        // printf("%e %e %f %f %f %f %f\n", x_pos, y_pos, d, Sigma, vx, vy, P);
    }
  }
}

void Grid3D::Spherical_Overpressure_3D()
{
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r, center_x, center_y, center_z;
  Real density, pressure, overDensity, overPressure, energy;
  Real vx, vy, vz, v2;
  center_x     = 0.5;
  center_y     = 0.5;
  center_z     = 0.5;
  overDensity  = 1;
  overPressure = 10;
  vx           = 0;
  vy           = 0;
  vz           = 0;

  // set the initial values of the conserved variables
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        // // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        density  = 0.1;
        pressure = 1;

        r = sqrt((x_pos - center_x) * (x_pos - center_x) + (y_pos - center_y) * (y_pos - center_y) +
                 (z_pos - center_z) * (z_pos - center_z));
        if (r < 0.2) {
          density = overDensity;
          pressure += overPressure;
        }
        v2               = vx * vx + vy * vy + vz * vz;
        energy           = pressure / (gama - 1) + 0.5 * density * v2;
        C.density[id]    = density;
        C.momentum_x[id] = density * vx;
        C.momentum_y[id] = density * vy;
        C.momentum_z[id] = density * vz;
        C.Energy[id]     = energy;

#ifdef DE
        C.GasEnergy[id] = pressure / (gama - 1);
#endif
      }
    }
  }
}

void Grid3D::Spherical_Overdensity_3D()
{
  // in the future, we should either:
  // - factor out the common logic shared with Grid3D::Spherical_Overpressure_3D, OR
  // - move this logic to the function where we define the SphericalOverdensity model
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r;
  Real density, pressure, overPressure, energy;
  Real vx, vy, vz, v2;

  const SphericalOverdensity *overdensity_model = models().try_get<SphericalOverdensity>();
  CHOLLA_ASSERT(overdensity_model != nullptr, "spherical overdensity model wasn't initialized");
  Real center_x           = overdensity_model->center_xyz[0];
  Real center_y           = overdensity_model->center_xyz[1];
  Real center_z           = overdensity_model->center_xyz[2];
  Real overDensity        = overdensity_model->overdensity;
  Real background_density = overdensity_model->bkg_density;
  Real radius             = overdensity_model->radius;

  overPressure = 0;
  vx           = 0;
  vy           = 0;
  vz           = 0;

  // set the initial values of the conserved variables
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        // // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        density  = background_density;
        pressure = 0.0005;

        r = sqrt((x_pos - center_x) * (x_pos - center_x) + (y_pos - center_y) * (y_pos - center_y) +
                 (z_pos - center_z) * (z_pos - center_z));
        if (r < radius) {
          density = overDensity;
          pressure += overPressure;
        }
        v2               = vx * vx + vy * vy + vz * vz;
        energy           = pressure / (gama - 1) + 0.5 * density * v2;
        C.density[id]    = density;
        C.momentum_x[id] = density * vx;
        C.momentum_y[id] = density * vy;
        C.momentum_z[id] = density * vz;
        C.Energy[id]     = energy;

#ifdef DE
        C.GasEnergy[id] = pressure / (gama - 1);
#endif
      }
    }
  }
}

/*! \fn void Clouds()
 *  \brief Bunch of clouds. */
void Grid3D::Clouds()
{
  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;
  Real n_bg, n_cl;      // background and cloud number density
  Real rho_bg, rho_cl;  // background and cloud density
  Real vx_bg, vx_cl;    // background and cloud velocity
  Real vy_bg, vy_cl;
  Real vz_bg, vz_cl;
  Real T_bg, T_cl;       // background and cloud temperature
  Real p_bg, p_cl;       // background and cloud pressure
  Real mu   = 0.6;       // mean atomic weight
  int N_cl  = 1;         // number of clouds
  Real R_cl = 2.5;       // cloud radius in code units (kpc)
  Real cl_pos[N_cl][3];  // array of cloud positions
  Real r;

  // Multiple Cloud Setup
  // for (int nn=0; nn<N_cl; nn++) {
  //  cl_pos[nn][0] = (nn+1)*0.1*H.xdglobal+0.5*H.xdglobal;
  //  cl_pos[nn][1] = (nn%2*0.1+0.45)*H.ydglobal;
  //  cl_pos[nn][2] = 0.5*H.zdglobal;
  //  printf("Cloud positions: %f %f %f\n", cl_pos[nn][0], cl_pos[nn][1],
  //  cl_pos[nn][2]);
  //}

  // single centered cloud setup
  for (int nn = 0; nn < N_cl; nn++) {
    cl_pos[nn][0] = 0.5 * H.xdglobal;
    cl_pos[nn][1] = 0.5 * H.ydglobal;
    cl_pos[nn][2] = 0.5 * H.zdglobal;
    printf("Cloud positions: %f %f %f\n", cl_pos[nn][0], cl_pos[nn][1], cl_pos[nn][2]);
  }

  n_bg   = 1.68e-4;
  n_cl   = 5.4e-2;
  rho_bg = n_bg * mu * MP / DENSITY_UNIT;
  rho_cl = n_cl * mu * MP / DENSITY_UNIT;
  vx_bg  = 0.0;
  // vx_c  = -200*TIME_UNIT/KPC; // convert from km/s to kpc/kyr
  vx_cl = 0.0;
  vy_bg = vy_cl = 0.0;
  vz_bg = vz_cl = 0.0;
  T_bg          = 3e6;
  T_cl          = 1e4;
  p_bg          = n_bg * KB * T_bg / PRESSURE_UNIT;
  p_cl          = p_bg;

  istart = H.n_ghost;
  iend   = H.nx - H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny - H.n_ghost;
  } else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz - H.n_ghost;
  } else {
    kstart = 0;
    kend   = H.nz;
  }

  // set initial values of conserved variables
  for (k = kstart; k < kend; k++) {
    for (j = jstart; j < jend; j++) {
      for (i = istart; i < iend; i++) {
        // get cell index
        id = i + j * H.nx + k * H.nx * H.ny;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        // set background state
        C.density[id]    = rho_bg;
        C.momentum_x[id] = rho_bg * vx_bg;
        C.momentum_y[id] = rho_bg * vy_bg;
        C.momentum_z[id] = rho_bg * vz_bg;
        C.Energy[id]     = p_bg / (gama - 1.0) + 0.5 * rho_bg * (vx_bg * vx_bg + vy_bg * vy_bg + vz_bg * vz_bg);
#ifdef DE
        C.GasEnergy[id] = p_bg / (gama - 1.0);
#endif
#ifdef SCALAR
  #ifdef DUST
        C.host[id + H.n_cells * grid_enum::dust_density] = 0.0;
  #endif
#endif
        // add clouds
        for (int nn = 0; nn < N_cl; nn++) {
          r = sqrt((x_pos - cl_pos[nn][0]) * (x_pos - cl_pos[nn][0]) +
                   (y_pos - cl_pos[nn][1]) * (y_pos - cl_pos[nn][1]) +
                   (z_pos - cl_pos[nn][2]) * (z_pos - cl_pos[nn][2]));
          if (r < R_cl) {
            C.density[id]    = rho_cl;
            C.momentum_x[id] = rho_cl * vx_cl;
            C.momentum_y[id] = rho_cl * vy_cl;
            C.momentum_z[id] = rho_cl * vz_cl;
            C.Energy[id]     = p_cl / (gama - 1.0) + 0.5 * rho_cl * (vx_cl * vx_cl + vy_cl * vy_cl + vz_cl * vz_cl);
#ifdef DE
            C.GasEnergy[id] = p_cl / (gama - 1.0);
#endif  // DE
#ifdef SCALAR
  #ifdef DUST
            C.host[id + H.n_cells * grid_enum::dust_density] = rho_cl * 1e-2;
  #endif  // DUST
#endif    // SCALAR
          }
        }
      }
    }
  }
}

void Grid3D::Uniform_Grid()
{
  chprintf(" Initializing Uniform Grid\n");
  int i, j, k, id;

  // Set limits
  size_t const istart = H.n_ghost;
  size_t const iend   = H.nx - H.n_ghost;
  size_t const jstart = H.n_ghost;
  size_t const jend   = H.ny - H.n_ghost;
  size_t const kstart = H.n_ghost;
  size_t const kend   = H.nz - H.n_ghost;

  // set the initial values of the conserved variables
  for (k = kstart - 1; k < kend; k++) {
    for (j = jstart - 1; j < jend; j++) {
      for (i = istart - 1; i < iend; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

#ifdef MHD
        // Set the magnetic field including the rightmost ghost cell on the
        // left side which is really the left face of the first grid cell
        C.magnetic_x[id] = 0;
        C.magnetic_y[id] = 0;
        C.magnetic_z[id] = 0;
#endif  // MHD

        // Exclude the rightmost ghost cell on the "left" side
        if ((k >= kstart) and (j >= jstart) and (i >= istart)) {
          C.density[id]    = 0;
          C.momentum_x[id] = 0;
          C.momentum_y[id] = 0;
          C.momentum_z[id] = 0;
          C.Energy[id]     = 0;

#ifdef DE
          C.GasEnergy[id] = 0;
#endif
        }
      }
    }
  }
}

void Grid3D::Zeldovich_Pancake(struct Parameters P)
{
#ifndef COSMOLOGY
  chprintf("To run a Zeldovich Pancake COSMOLOGY has to be turned ON \n");
  exit(-1);
#else

  int i, j, k, id;
  Real x_pos, y_pos, z_pos;
  Real H0, h, Omega_M, rho_0, G, z_zeldovich, z_init, x_center, T_init, k_x;

  chprintf("Setting Zeldovich Pancake initial conditions...\n");
  H0      = P.H0;
  h       = H0 / 100;
  Omega_M = P.Omega_M;

  chprintf(" h = %f \n", h);
  chprintf(" Omega_M = %f \n", Omega_M);

  H0 /= 1000;  //[km/s / kpc]

  G     = G_COSMO;
  rho_0 = 3 * H0 * H0 / (8 * M_PI * G) * Omega_M / h / h;

  z_zeldovich = 1;
  z_init      = P.Init_redshift;
  chprintf(" rho_0 = %f \n", rho_0);
  chprintf(" z_init = %f \n", z_init);
  chprintf(" z_zeldovich = %f \n", z_zeldovich);

  x_center = H.xdglobal / 2;
  chprintf(" Peak Center = %f \n", x_center);

  T_init = 100;
  chprintf(" T initial = %f \n", T_init);

  k_x = 2 * M_PI / H.xdglobal;

  char filename[100];
  // create the filename to read from
  strcpy(filename, P.indir);
  strcat(filename, "ics_zeldovich.dat");
  chprintf(" Loading ICs File: %s\n", filename);

  real_vector_t ics_values;

  std::ifstream file_in(filename);
  std::string line;
  Real ic_val;
  if (file_in.is_open()) {
    while (getline(file_in, line)) {
      ic_val = atof(line.c_str());
      ics_values.push_back(ic_val);
      // chprintf("%f\n", ic_val);
    }
    file_in.close();
  } else {
    chprintf("  Error: Unable to open ics zeldovich file\n");
    exit(1);
  }
  int nPoints = 256;

  Real dens, vel, temp, U, E, gamma;
  gamma = P.gamma;

  int index;
  // set the initial values of the conserved variables
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        // // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        // Analytical Initial Conditions
        //  dens = rho_0 / ( 1 - ( 1 + z_zeldovich ) / ( 1 + z_init ) * cos(
        //  k_x*( x_pos - x_center )) ); vel = - H0 * ( 1 + z_zeldovich ) /
        //  sqrt( 1 + z_init ) * sin( k_x*( x_pos - x_center )) / k_x; temp =
        //  T_init * pow( dens / rho_0, 2./3 ); U = temp / (gamma - 1) / MP * KB
        //  * 1e-10 * dens; E = 0.5 * dens * vel * vel + U;

        index = (int(x_pos / H.dx) + 0) % 256;
        // index = ( index + 16 ) % 256;
        dens = ics_values[0 * nPoints + index];
        vel  = ics_values[1 * nPoints + index];
        E    = ics_values[2 * nPoints + index];
        U    = ics_values[3 * nPoints + index];
        // //
        // temp = T_init * pow( dens/rho_0, 2./3.);
        // U = temp / (gamma-1) / MP * KB * 1e-10 * dens;
        // E = 0.5*dens*vel*vel + U;

        // chprintf( "%f \n", vel );
        C.density[id]    = dens;
        C.momentum_x[id] = dens * vel;
        C.momentum_y[id] = 0;
        C.momentum_z[id] = 0;
        C.Energy[id]     = E;

  #ifdef DE
        C.GasEnergy[id] = U;
  #endif
      }
    }
  }

#endif  // COSMOLOGY
}

void Grid3D::Adiabatic_Expansion(struct Parameters P)
{
#ifndef COSMOLOGY
  chprintf("To run an Adiabatic Expansion test, COSMOLOGY has to be turned ON \n");
  exit(-1);
#else

  int i, j, k, id;
  Real x_pos, y_pos, z_pos;
  Real H0, h, Omega_M, rho_0, G, z_zeldovich, z_init, x_center, T_init, k_x;

  chprintf("Setting Adiabatic Expansion initial conditions...\n");
  H0      = P.H0;
  h       = H0 / 100;
  Omega_M = P.Omega_M;

  chprintf(" h = %f \n", h);
  chprintf(" Omega_M = %f \n", Omega_M);

  H0 /= 1000;  //[km/s / kpc]
  G      = G_COSMO;
  rho_0  = 3 * H0 * H0 / (8 * M_PI * G) * Omega_M / h / h;
  z_init = P.Init_redshift;
  chprintf(" rho_0 = %f \n", rho_0);
  chprintf(" z_init = %f \n", z_init);

  T_init = 100;
  chprintf(" T initial = %f \n", T_init);

  k_x = 2 * M_PI / H.xdglobal;

  Real dens, vel, temp, U, E, gamma;
  gamma = P.gamma;

  int index;
  // set the initial values of the conserved variables
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        // Analytical Initial Conditions
        //  dens = rho_0 / ( 1 - ( 1 + z_zeldovich ) / ( 1 + z_init ) * cos(
        //  k_x*( x_pos - x_center )) ); vel = - H0 * ( 1 + z_zeldovich ) /
        //  sqrt( 1 + z_init ) * sin( k_x*( x_pos - x_center )) / k_x; temp =
        //  T_init * pow( dens / rho_0, 2./3 ); U = temp / (gamma - 1) / MP * KB
        //  * 1e-10 * dens; E = 0.5 * dens * vel * vel + U;

        dens = rho_0;
        vel  = 0;
        U    = T_init / (gamma - 1) / MP * KB * 1e-10 * rho_0;
        E    = U;

        // chprintf( "%f \n", vel );
        C.density[id]    = dens;
        C.momentum_x[id] = dens * vel;
        C.momentum_y[id] = 0;
        C.momentum_z[id] = 0;
        C.Energy[id]     = E;

  #ifdef DE
        C.GasEnergy[id] = U;
  #endif
      }
    }
  }

#endif  // COSMOLOGY
}

void Grid3D::Cosmological_ICs(struct Parameters const P)
{
#if !defined(COSMOLOGY) || !defined(FFT)
  chprintf("To run a Cosmological ICs simulation, COSMOLOGY and FFT have to be turned ON \n");
  exit(-1);
#else

  int n_cells = nx_local * ny_local * nz_local;

  chprintf("Cosmological ICs: Setting gas grid initial conditions ...\n");

  int i, j, k, id;

  #ifdef GRAVITY_5_POINTS_GRADIENT
  Real phi_ll, phi_rr;
  int id_ll, id_rr;
  #endif
  Real phi_l, phi, phi_r;
  Real phi_ld, phi_lu, phi_rd, phi_ru;
  int id_l, id_r;
  int id_ld, id_lu;
  int id_rd, id_ru;

  Real x_pos, y_pos, z_pos;
  Real H0, h, Ha, rho_0, G, z_zeldovich, z_init, x_center, T_init, k_x;
  Real gamma;

  Real a_init, D, dDdt, dDda, dlogDdloga;

  Real vx, vy, vz;        // velocities
  Real dx, dy, dz;        // cell sizes
  Real xi_x, xi_y, xi_z;  // particle offsets

  Real xHp  = P.xHp_ion_init;   // hydrogen ionization fraction
  Real xHep = P.xHep_ion_init;  // helium ionization fraction
  Real YHe  = P.YHe;            // helium mass fraction

  H0 = P.H0;
  h  = H0 / 100;
  H0 /= 1000;  // to km/s/kpc
  G             = G_COSMO;
  Real Omega_b  = P.Omega_b;
  Real Omega_m  = P.Omega_M;
  Real Omega_r  = P.Omega_R;
  Real Omega_DE = P.Omega_L;
  Real w0       = P.w0;
  Real wa       = P.wa;

  Real f_b    = Omega_b / Omega_m;
  Real f_c    = 1.0 - f_b;
  Real rho_b  = 3 * H0 * H0 / (8 * M_PI * G) * Omega_b / h / h;  // correctly set
  Real d_mean = 0, n_d_mean = 0;
  // src/cosmology/cosmology.cpp:  rho_0_gas = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_M / cosmo_h / cosmo_h;

  // src/global/global.h:#define G_COSMO  4.300927161e-06;  // gravitational constant, kpc km^2 s^-2 Msun^-1
  //  (km/s/kpc)^2 / (kpc km^2 s^-2 Msun^-1) = Msun/kpc^3
  //  G in cgs is 6.674e-8 cm^3 g^-1 s^-2
  //  1.327104878e+26 cm^3 Msun^-1 s^-2
  //  1.327104878e+16 cm km^2 Msun^-1 s^-2
  //  4.300854005621676e-06 kpc km^2 Msun^-1 s^-2

  chprintf("Cosmological ICs: H0: %e [km/s/kpc]\n", H0);
  chprintf("Cosmological ICs: Omega_m:  %e\n", Omega_m);
  chprintf("Cosmological ICs: Omega_b:  %e\n", Omega_b);
  chprintf("Cosmological ICs: Omega_r:  %e\n", Omega_r);
  chprintf("Cosmological ICs: Omega_DE: %e\n", Omega_DE);
  chprintf("Cosmological ICs: w0:       %e\n", w0);
  chprintf("Cosmological ICs: wa:       %e\n", wa);
  chprintf("Cosmological ICs: Baryon density rho_b %e [h^2 Msun/kpc^3]\n", rho_b);
  chprintf("Cosmological ICs: MP %e [g]\n", MP);
  chprintf("Cosmological ICs: KB %e [ergs/K]\n", KB);
  chprintf("Cosmological ICs: DENSITY_UNIT %e\n", DENSITY_UNIT);
  chprintf("Cosmological ICs: ENERGY_UNIT %e\n", ENERGY_UNIT);
  chprintf("Cosmological ICs: LENGTH_UNIT %e\n", LENGTH_UNIT);
  chprintf("Cosmological ICs: PRESSURE_UNIT %e\n", PRESSURE_UNIT);
  chprintf("Cosmological ICs: VELOCITY_UNIT %e\n", VELOCITY_UNIT);

  Real r_0_dm          = P.xlen / P.nx;
  Real t_0_dm          = 1. / H0;
  Real v_0_dm          = r_0_dm / t_0_dm / h;
  Real rho_0_dm        = 3 * H0 * H0 / (8 * M_PI * G) * Omega_m / h / h;
  Real rho_mean_baryon = 3 * H0 * H0 / (8 * M_PI * G) * Omega_b / h / h;
  // dens_avrg = 0;

  Real r_0_gas   = 1.0;
  Real rho_0_gas = 3 * H0 * H0 / (8 * M_PI * G) * Omega_m / h / h;
  Real t_0_gas   = 1 / H0 * h;
  Real v_0_gas   = r_0_gas / t_0_gas;
  Real phi_0_gas = v_0_gas * v_0_gas;
  Real p_0_gas   = rho_0_gas * v_0_gas * v_0_gas;
  Real e_0_gas   = v_0_gas * v_0_gas;

  Real dens_factor;
  Real momentum_factor;
  Real energy_factor;

  chprintf("Cosmological ICs: H0     %e [(km/s)/kpc]\n", H0);
  chprintf("Cosmological ICs: r_0_dm %e [kpc/h]\n", r_0_dm);
  chprintf("Cosmological ICs: t_0_dm %e [kpc/(km/s)]\n", t_0_dm);
  chprintf("Cosmological ICs: v_0_dm %e [km/s]\n", v_0_dm);
  chprintf("Cosmological ICs: rho_0_dm %e [Msun/kpc^3]\n", rho_0_dm);
  chprintf("Cosmological ICs: rho_mean_baryon %e [Msun/kpc^3]\n", rho_mean_baryon);
  chprintf("Cosmological ICs: r_0_gas   %e [kpc]\n", r_0_gas);
  chprintf("Cosmological ICs: rho_0_gas %e [Msun/kpc^3]\n", rho_0_gas);
  chprintf("Cosmological ICs: t_0_gas   %e [kpc/(km/s)]\n", t_0_gas);
  chprintf("Cosmological ICs: v_0_gas   %e [km/s]\n", v_0_gas);
  chprintf("Cosmological ICs: phi_0_gas %e [(km/s)^2]\n", phi_0_gas);
  chprintf("Cosmological ICs: p_0_gas   %e [(Msun/kpc^3) (km/s)^2]\n", p_0_gas);
  chprintf("Cosmological ICs: e_0_gas   %e [(km/s)^2]\n", e_0_gas);

  /*
  H0: 6.732117e-02 [km/s/kpc]
  Omega_b: 4.938700e-02
  Baryon density rho_b 1.370667e+01 [Msun/kpc^3]
  MP 1.672622e-24 [g]
  KB 1.380658e-16 [cgs]
  DENSITY_UNIT 6.768110e-32
  ENERGY_UNIT 6.471126e-10
  LENGTH_UNIT 3.085678e+21
  PRESSURE_UNIT 6.471126e-10
  VELOCITY_UNIT 9.778139e+10
  H0     6.732117e-02 [(km/s)/kpc]
  r_0_dm 1.953125e+02 [kpc/h]
  t_0_dm 1.485417e+01 [kpc/(km/s)]
  v_0_dm 1.953125e+01 [km/s]
  rho_0_dm 8.725731e+01 [Msun/kpc^3]
  rho_mean_baryon 1.370667e+01 [Msun/kpc^3]
  r_0_gas   1.000000e+00 [kpc]
  rho_0_gas 8.725731e+01 [Msun/kpc^3]
  t_0_gas   1.000000e+01 [kpc/(km/s)]
  v_0_gas   1.000000e-01 [km/s]
  phi_0_gas 1.000000e-02 [(km/s)^2]
  p_0_gas   8.725731e-01 [(Msun/kpc^3) (km/s)^2]
  e_0_gas   1.000000e-02 [(km/s)^2]
  */

  gamma = P.gamma;

  z_init = P.Init_redshift;
  a_init = 1. / (1 + z_init);

  chprintf("Cosmological ICs: z_init %e a_init %e\n", z_init, a_init);

  // get growth function and time derivative
  Ha = Hubble_Growth_Function(a_init, H0, Omega_r, Omega_m, Omega_DE, w0, wa);  // H(a) is about 39 km/s/kpc at z~100
  Real dHda =
      dHda_Growth_Function(a_init, H0, Omega_r, Omega_m, Omega_DE, w0, wa);  // dHda is about -5.9e3 km/s/kpc at z~100
  // For a CDM universe the manual and direct derivatives agree, maybe implement DE
  chprintf("Cosmological ICs: H(%e) = %e, dHda = %e\n", a_init, Ha, dHda);

  D          = Cosmo.D_Growth(a_init);
  dDdt       = Cosmo.dDdt_Growth(a_init);
  dDda       = Cosmo.dDda_Growth(a_init);
  dlogDdloga = dDda * (a_init / D);
  chprintf("Cosmological ICs: Growth Function Info: D %e dDdt %e dDda %e dlnDdlna %e\n", D, dDdt, dDda,
           dlogDdloga);  // note dDdt is in km/s/kpc

  // scaling is A = a*H*dlogD/dloga = a^2 (H/D) dD/da
  // these units will be in km/s/kpc, and multiplying by
  // displacement will convert into a velocity
  // the d logD / d log a is the growth rate function
  // this function is close to 1 at high redshift
  // so the velocity scaling should be ~ a * H
  // at redshift 100, this is ~ 0.0099 * 38.9 km/s/kpc ~ 0.38 km/s/kpc
  Real A_vel = a_init * Ha * dlogDdloga;  // km/s/kpc
  chprintf("Cosmological ICs: Velocity scaling = %e\n", A_vel);

  // Determine the gas temperature -- maybe move below
  if (P.T_init == -1) {
    T_init = 2.72548 * (1.0 + z_init);  // set to CMB
  } else {
    T_init = P.T_init;  // set to precomputed gas temperature
  }
  chprintf("Cosmological ICs: T initial = %f \n", T_init);

  // cell sizes
  dx = H.dx;
  dy = H.dy;
  dz = H.dz;

  chprintf("Cosmological ICs: Cell sizes dx %e dy %e dz %e\n", dx, dy, dz);

  // create gas
  // f_b = Omega_b / Omega_m
  // f_c = 1.0 - f_b

  // delta_m = D \nabla^2 \phi_ini
  // delta_b_ini =  f_c * delta_bc
  // delta_c_ini = -f_b * delta_bc
  // delta_c = \delta_m + delta_c_ini = delta_m - f_b * delta_bc
  // delta_b = \delta_m + delta_b_ini = delta_m + f_c * delta_bc
  // delta_b = delta_c + f_b delta_bc + f_c delta_bc = delta_c + delta_bc
  // exit
  // chprintf("Saving potential...\n");
  // Save_Cosmo_Potential(P);

  // We have verified that the delta_m and delta_bc
  // reflect the expected Pm(k) and Pbc(k)

  // dm particle masses could inherit mass perturbations
  // m_c(q) = \bar{m}_c * (1 + delta_c_ini(q))

  // or to first-order the lagrangian displacement is
  // xi_c_pert = D xi_m(1) - \nabla^-2 \grad \delta_c_init
  // xi_m(1) = - \grad \phi_ini
  // x_c = q - D(z_start) \grad \phi_ini - \nabla^-2 \grad \delta_c_init
  // x_c = q - D(z_start) \grad \phi_ini + f_b * \nabla^-2 \grad \delta_bc
  // v_c = dD/dt * \grad \phi_ini
  // v_b = dD/dt * \grad \phi_ini
  // \phi_ini = \nabla^-2 \delta_m (a_Ref)/D+(a_Ref) * D+(a)/a

  // the first method still requires the PT total mass displacements
  // for the dark matter

  // so we should compute
  // 1) phi_ini from delta_m (check)
  // 4) delta_b from delta_m and delta_bc
  // 3) v_c and v_b from \grad phi_ini
  // 5) replace delta_bc with \nabla^-2 delta_bc
  // 6) x_c from \grad phi_ini and gradient of delta_bc

  // compute dD/da * da/dt / a = a H dD/da

  ////////////////////////////////////////////////
  // 4) Compute delta b from delta_m and delta_bc
  ////////////////////////////////////////////////

  // At this point, we need the potentials, and delta_bc
  // and we can delta_m to help transfer the potentials

  // We can copy the potential fields to the density field
  // for the real cells only
  int index;
  int ii, jj, kk;

  // copy memory -- only real, local cells
  cudaMemcpy(CP.delta_m, CP.phi_1, n_cells * sizeof(Real), cudaMemcpyHostToHost);

  // check before the remap
  Real delta_rm_check   = 0;
  Real delta_rm_check_n = 0;
  for (k = 0; k < H.nz - 2 * H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2 * H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2 * H.n_ghost; i++) {
        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        delta_rm_check += CP.phi_1[id];
        delta_rm_check_n += 1;
      }
    }
  }
  // get the total of the grid to compute the mean
  MPI_Allreduce(MPI_IN_PLACE, &delta_rm_check, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // find the average
  delta_rm_check = delta_rm_check / (nx_global * ny_global * nz_global);

  Real delta_m_check   = 0;
  Real delta_m_check_n = 0;

  Real grad_phi_x = 0, grad_phi_y = 0, grad_phi_z = 0;
  Real xix_mean = 0;
  Real xiy_mean = 0;
  Real xiz_mean = 0;
  Real xix_rms  = 0;
  Real xiy_rms  = 0;
  Real xiz_rms  = 0;
  Real xix_n    = 0;
  Real phi_mean = 0;
  Real vx_rms   = 0;
  Real vy_rms   = 0;
  Real vz_rms   = 0;
  Real vx_mean  = 0;
  Real vy_mean  = 0;
  Real vz_mean  = 0;

  // now, we remap the potential fields from the density fields
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        // this is the full index with ghost cells
        id = i + j * H.nx + k * H.nx * H.ny;

        kk = k - H.n_ghost;
        jj = j - H.n_ghost;
        ii = i - H.n_ghost;

        // this is the real index with only local real cells
        index = ii + jj * nx_local + kk * nx_local * ny_local;

        // copy from delta back to phi
        CP.phi_1[id] = CP.delta_m[index];
        delta_m_check += CP.delta_m[index];
        delta_m_check_n += 1;
      }
    }
  }

  // get the total of the grid to compute the mean
  MPI_Allreduce(MPI_IN_PLACE, &delta_m_check, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  delta_m_check = delta_m_check / (nx_global * ny_global * nz_global);
  chprintf("Cosmological ICs: Delta m check %e\n", delta_m_check);

  // repeat for baryons if present
  #ifndef ONLY_PARTICLES
  cudaMemcpy(CP.delta_m, CP.phi_2, n_cells * sizeof(Real), cudaMemcpyHostToHost);
  // now, we remap the potential fields from the density fields
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        // this is the full index with ghost cells
        id = i + j * H.nx + k * H.nx * H.ny;

        kk = k - H.n_ghost;
        jj = j - H.n_ghost;
        ii = i - H.n_ghost;

        // this is the real index with only local real cells
        index = ii + jj * nx_local + kk * nx_local * ny_local;

        // copy from delta back to phi
        CP.phi_2[id] = CP.delta_m[index];
      }
    }
  }
  #endif  // ONLY_PARTICLES

  // The potentials are now in their correct
  // places, indexed against the entire local
  // grid including ghost cells. Deal with the
  // boundary conditions appropriately.

  // perform the ghost cell transfers between
  // subdomains of the computational volume

  chprintf("Cosmological ICs: Exchanging ghost cells for cosmological potential phi_1.\n");
  Allocate_Boundary_Conditions_Field_MPI();
  Set_Boundary_Conditions_Field(P, CP.phi_1);
  #ifndef ONLY_PARTICLES
  chprintf("Cosmological ICs: Exchanging ghost cells for cosmological potential phi_2.\n");
  Set_Boundary_Conditions_Field(P, CP.phi_2);
  #endif  // ONLY_PARTICLES
  Free_Boundary_Conditions_Field_MPI();

  // now we can proceed with computing the gradients
  #ifndef ONLY_PARTICLES

  Real grad_x_T[3][3], det_grad_x;
  Real det_grad_x_min = 1e9;
  Real det_grad_x_max = -1e9;
  Real dens_min       = 1e9;
  Real dens_max       = -1e9;
  Real xix_min        = 1e9;
  Real xix_max        = -1e9;
  Real d_rms          = 0;

  Real dens, vel, U, E;
  Real Daf = Cosmo.D_Growth(a_init) / a_init;

  // With radiation the perturbation ratio D/a keeps
  // increasing to large redshift
  chprintf("Cosmological ICs: lim a->0 D(a)/a = %e\n", Daf);
  chprintf("Cosmological ICs: D(a)/a = %e\n", Daf);

  // gradient operators
  // d/dx
  // 2: 0.5*[-1 0 1]
  // 4: (1/12)*[1 -8 0 8 -1]
  // d2/dx2
  // 2: [1 -2 1]
  // 4: (1/12)*[-1 16 -30 16 -1]

  // set the initial values of the conserved variables
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.ny * H.nx;

        kk    = k - H.n_ghost;
        jj    = j - H.n_ghost;
        ii    = i - H.n_ghost;
        index = ii + jj * nx_local + kk * nx_local * ny_local;

        //////////////////////////////////////////
        // take the potential gradient
        // along the x direction
        //////////////////////////////////////////

        //////////////////////////////////////////
        // We have that
        // x = q - D\nabla^-2 \grad \delta
        // v = dDdt \nabla^-2 \grad \delta
        // Then for the laplacian
        // \grad x = \grad q - D \nabla^2 \nabla^-2 \delta
        // -> we subtract the second derivative
        //////////////////////////////////////////

        id   = i + j * H.nx + k * H.ny * H.nx;
        id_l = (i - 1) + j * H.nx + k * H.ny * H.nx;
        id_r = (i + 1) + j * H.nx + k * H.ny * H.nx;

        phi   = CP.phi_1[id];
        phi_l = CP.phi_1[id_l];
        phi_r = CP.phi_1[id_r];

    #ifdef GRAVITY_5_POINTS_GRADIENT
        id_ll          = (i - 2) + j * H.nx + k * H.ny * H.nx;
        id_rr          = (i + 2) + j * H.nx + k * H.ny * H.nx;
        phi_ll         = CP.phi_1[id_ll];
        phi_rr         = CP.phi_1[id_rr];
        grad_phi_x     = (-phi_rr + 8 * phi_r - 8 * phi_l + phi_ll) / (12 * dx);
        grad_x_T[0][0] = D * (-phi_rr + 16 * phi_r - 30 * phi + 16 * phi_l - phi_ll) / (12 * dx * dx);
    #else
        grad_phi_x     = 0.5 * (phi_r - phi_l) / dx;
        grad_x_T[0][0] = D * (phi_r - 2 * phi + phi_l) / (dx * dx);
    #endif
        /*
        2nd
        Cosmological ICs: x-displacement field average = 1.035328e-17, rms = 5.977859e+01
        Cosmological ICs: y-displacement field average = -3.631385e-15, rms = 5.373778e+01
        Cosmological ICs: z-displacement field average = 4.274555e-13, rms = 4.206997e+01
        Cosmological ICs: x-velocity     field average = 1.086150e-17, rms = 3.334087e+01
        Cosmological ICs: y-velocity     field average = -3.553962e-15, rms = 2.997167e+01
        Cosmological ICs: z-velocity     field average = 3.764261e-13, rms = 2.346408e+01
        Cosmological ICs: overdensity    field average = 3.909169e-03, rms = 6.561782e-02
        Cosmological ICs: corr overdens. field average = 5.729687e-16, rms = 6.550127e-02
        4th
        Cosmological ICs: x-displacement field average = -8.011153e-17, rms = 5.984219e+01
        Cosmological ICs: y-displacement field average = 7.371302e-15, rms = 5.380806e+01
        Cosmological ICs: z-displacement field average = 2.038536e-13, rms = 4.215598e+01
        Cosmological ICs: x-velocity     field average = -3.418498e-17, rms = 3.337634e+01
        Cosmological ICs: y-velocity     field average = 4.785711e-15, rms = 3.001087e+01
        Cosmological ICs: z-velocity     field average = -6.382516e-14, rms = 2.351205e+01
        Cosmological ICs: overdensity    field average = 3.909169e-03, rms = 6.561782e-02
        Cosmological ICs: corr overdens. field average = 5.729687e-16, rms = 6.550127e-02
        */

        // TODO(brant): for baryon-dm fluctuations
        /*
                // repeat for nabla^-2 \delta_bc
                // which affects LLA only
                phi   = f_c*CP.phi_2[id];
                phi_l = f_c*CP.phi_2[id_l];
                phi_r = f_c*CP.phi_2[id_r];
            #ifdef GRAVITY_5_POINTS_GRADIENT
                phi_ll  = f_c*CP.phi_2[id_ll];
                phi_rr  = f_c*CP.phi_2[id_rr];
                grad_x_T[0][0]   += ((-phi_rr + 16 * phi_r - 30 * phi + 16 * phi_l - phi_ll)/(12 * dx*dx));
            #else
                grad_x_T[0][0]   += ((phi_rr - 2 * phi + - phi_ll)/(dx*dx));
            #endif
        */

        //////////////////////////////////////////
        // take the potential gradient
        // along the y direction
        //////////////////////////////////////////

        id_l  = i + (j - 1) * H.nx + k * H.ny * H.nx;
        id_r  = i + (j + 1) * H.nx + k * H.ny * H.nx;
        phi   = CP.phi_1[id];
        phi_l = CP.phi_1[id_l];
        phi_r = CP.phi_1[id_r];
    #ifdef GRAVITY_5_POINTS_GRADIENT
        id_ll          = i + (j - 2) * H.nx + k * H.ny * H.nx;
        id_rr          = i + (j + 2) * H.nx + k * H.ny * H.nx;
        phi_ll         = CP.phi_1[id_ll];
        phi_rr         = CP.phi_1[id_rr];
        grad_phi_y     = (-phi_rr + 8 * phi_r - 8 * phi_l + phi_ll) / (12 * dy);
        grad_x_T[1][1] = D * (-phi_rr + 16 * phi_r - 30 * phi + 16 * phi_l - phi_ll) / (12 * dy * dy);
    #else
        grad_phi_y     = 0.5 * (phi_r - phi_l) / dy;
        grad_x_T[1][1] = D * (phi_r - 2 * phi + phi_l) / (dy * dy);
    #endif

        // TODO(brant): for baryon-dm fluctuations
        /*
                // repeat for nabla^-2 \delta_bc
                // which affects which affects LLA only
                phi   = f_c*CP.phi_2[id];
                phi_l = f_c*CP.phi_2[id_l];
                phi_r = f_c*CP.phi_2[id_r];

            #ifdef GRAVITY_5_POINTS_GRADIENT
                phi_ll  = f_c*CP.phi_2[id_ll];
                phi_rr  = f_c*CP.phi_2[id_rr];
                grad_x_T[1][1]   += ((-phi_rr + 16 * phi_r - 30 * phi + 16 * phi_l - phi_ll)/(12 * dy*dy));
            #else
                grad_x_T[1][1]   += ((phi_rr - 2 * phi + - phi_ll)/(dy*dy));
            #endif
        */

        //////////////////////////////////////////
        // take the potential gradient
        // along the z direction
        //////////////////////////////////////////

        id_l  = i + j * H.nx + (k - 1) * H.ny * H.nx;
        id_r  = i + j * H.nx + (k + 1) * H.ny * H.nx;
        phi   = CP.phi_1[id];
        phi_l = CP.phi_1[id_l];
        phi_r = CP.phi_1[id_r];
    #ifdef GRAVITY_5_POINTS_GRADIENT
        id_ll          = i + j * H.nx + (k - 2) * H.ny * H.nx;
        id_rr          = i + j * H.nx + (k + 2) * H.ny * H.nx;
        phi_ll         = CP.phi_1[id_ll];
        phi_rr         = CP.phi_1[id_rr];
        grad_phi_z     = (-phi_rr + 8 * phi_r - 8 * phi_l + phi_ll) / (12 * dz);
        grad_x_T[2][2] = D * (-phi_rr + 16 * phi_r - 30 * phi + 16 * phi_l - phi_ll) / (12 * dz * dz);
    #else
        grad_phi_z     = 0.5 * (phi_r - phi_l) / dz;
        grad_x_T[2][2] = D * (phi_r - 2 * phi + phi_l) / (dz * dz);
    #endif
        grad_phi_z = 0.5 * (phi_r - phi_l) / dz;

        // TODO(brant): for baryon-dm fluctuations
        /*
                // repeat for nabla^-2 \delta_bc
                // which affects which affects LLA only
                phi   = f_c*CP.phi_2[id];
                phi_l = f_c*CP.phi_2[id_l];
                phi_r = f_c*CP.phi_2[id_r];

            #ifdef GRAVITY_5_POINTS_GRADIENT
                phi_ll  = f_c*CP.phi_2[id_ll];
                phi_rr  = f_c*CP.phi_2[id_rr];
                grad_x_T[2][2] += ((-phi_rr + 16 * phi_r - 30 * phi + 16 * phi_l - phi_ll)/(12 * dz*dz));
            #else
                grad_x_T[2][2] += ((phi_rr - 2 * phi + - phi_ll)/(dz*dz));
            #endif
        */

        // now we need the cross terms for LLA
        // first xy
        id_ld = (i - 1) + (j - 1) * H.nx + k * H.ny * H.nx;
        id_lu = (i - 1) + (j + 1) * H.nx + k * H.ny * H.nx;
        id_rd = (i + 1) + (j - 1) * H.nx + k * H.ny * H.nx;
        id_ru = (i + 1) + (j + 1) * H.nx + k * H.ny * H.nx;

        // TODO(brant): for baryon-dm fluctuations
        // phi_ld = D*CP.phi_1[id_ld] + f_c*CP.phi_2[id_ld];
        // phi_lu = D*CP.phi_1[id_lu] + f_c*CP.phi_2[id_lu];
        // phi_rd = D*CP.phi_1[id_rd] + f_c*CP.phi_2[id_rd];
        // phi_ru = D*CP.phi_1[id_ru] + f_c*CP.phi_2[id_ru];

        phi_ld         = D * CP.phi_1[id_ld];
        phi_lu         = D * CP.phi_1[id_lu];
        phi_rd         = D * CP.phi_1[id_rd];
        phi_ru         = D * CP.phi_1[id_ru];
        grad_x_T[0][1] = 0.25 * (phi_ld - phi_lu - phi_rd + phi_ru) / (dx * dy);
        // grad_x_T[0][1] = -1*0.25*(phi_ld - phi_lu - phi_rd + phi_ru)/(dx*dy);
        grad_x_T[1][0] = grad_x_T[0][1];

        // second xz
        id_ld = (i - 1) + j * H.nx + (k - 1) * H.ny * H.nx;
        id_lu = (i - 1) + j * H.nx + (k + 1) * H.ny * H.nx;
        id_rd = (i + 1) + j * H.nx + (k - 1) * H.ny * H.nx;
        id_ru = (i + 1) + j * H.nx + (k + 1) * H.ny * H.nx;

        // TODO(brant): for baryon-dm fluctuations
        // phi_ld = D*CP.phi_1[id_ld] + f_c*CP.phi_2[id_ld];
        // phi_lu = D*CP.phi_1[id_lu] + f_c*CP.phi_2[id_lu];
        // phi_rd = D*CP.phi_1[id_rd] + f_c*CP.phi_2[id_rd];
        // phi_ru = D*CP.phi_1[id_ru] + f_c*CP.phi_2[id_ru];

        phi_ld         = D * CP.phi_1[id_ld];
        phi_lu         = D * CP.phi_1[id_lu];
        phi_rd         = D * CP.phi_1[id_rd];
        phi_ru         = D * CP.phi_1[id_ru];
        grad_x_T[0][2] = 0.25 * (phi_ld - phi_lu - phi_rd + phi_ru) / (dx * dz);
        // grad_x_T[0][2] = -1*0.25*(phi_ld - phi_lu - phi_rd + phi_ru)/(dx*dz);
        grad_x_T[2][0] = grad_x_T[0][2];

        // third yz
        id_ld = i + (j - 1) * H.nx + (k - 1) * H.ny * H.nx;
        id_lu = i + (j - 1) * H.nx + (k + 1) * H.ny * H.nx;
        id_rd = i + (j + 1) * H.nx + (k - 1) * H.ny * H.nx;
        id_ru = i + (j + 1) * H.nx + (k + 1) * H.ny * H.nx;

        // TODO(brant): for baryon-dm fluctuations
        // phi_ld = D*CP.phi_1[id_ld] + f_c*CP.phi_2[id_ld];
        // phi_lu = D*CP.phi_1[id_lu] + f_c*CP.phi_2[id_lu];
        // phi_rd = D*CP.phi_1[id_rd] + f_c*CP.phi_2[id_rd];
        // phi_ru = D*CP.phi_1[id_ru] + f_c*CP.phi_2[id_ru];

        phi_ld         = D * CP.phi_1[id_ld];
        phi_lu         = D * CP.phi_1[id_lu];
        phi_rd         = D * CP.phi_1[id_rd];
        phi_ru         = D * CP.phi_1[id_ru];
        grad_x_T[1][2] = 0.25 * (phi_ld - phi_lu - phi_rd + phi_ru) / (dy * dz);
        // grad_x_T[1][2] = -1*0.25*(phi_ld - phi_lu - phi_rd + phi_ru)/(dy*dz);
        grad_x_T[2][1] = grad_x_T[1][2];

        //////////////////////////////////////////
        // compute displacements
        // xi = D * \grad \phi
        //////////////////////////////////////////
        xi_x = D * grad_phi_x;  // in kpc/h
        xi_y = D * grad_phi_y;  // in kpc/h
        xi_z = D * grad_phi_z;  // in kpc/h

        if (xi_x < xix_min) xix_min = xi_x;
        if (xi_x > xix_max) xix_max = xi_x;
        xix_mean += xi_x;
        xiy_mean += xi_y;
        xiz_mean += xi_z;
        xix_rms += xi_x * xi_x;
        xiy_rms += xi_y * xi_y;
        xiz_rms += xi_z * xi_z;
        xix_n += 1;

        //////////////////////////////////////////
        // compute velocities field
        //////////////////////////////////////////

        vx = A_vel * xi_x / h;  // km/s
        vy = A_vel * xi_y / h;  // km/s
        vz = A_vel * xi_z / h;  // km/s

        vx_mean += vx;
        vy_mean += vy;
        vz_mean += vz;
        vx_rms += vx * vx;
        vy_rms += vy * vy;
        vz_rms += vz * vz;

        C.momentum_x[id] = vx;  // store velocities
        C.momentum_y[id] = vy;  // store velocities
        C.momentum_z[id] = vz;  // store velocities

        ////////////////////////////////////////////////////////
        // Displacements
        // x = q - D \grad phi_init - grad \nabla^-2 \delta_b
        // x = q - D \grad phi_init - f_c grad \nabla^-2 \delta_bc
        ////////////////////////////////////////////////////////

        // add dq/dq to grad x
        grad_x_T[0][0] += 1;
        grad_x_T[1][1] += 1;
        grad_x_T[2][2] += 1;

        // A = | a b c |
        //     | d e f |
        //     | g h i |
        // det(A) = a |e f| - b |d f| + c |d e|
        //            |h i|     |g i|     |g h|
        det_grad_x = grad_x_T[0][0] * (grad_x_T[1][1] * grad_x_T[2][2] - grad_x_T[2][1] * grad_x_T[1][2]);
        det_grad_x -= grad_x_T[0][1] * (grad_x_T[1][0] * grad_x_T[2][2] - grad_x_T[2][0] * grad_x_T[1][2]);
        det_grad_x += grad_x_T[0][2] * (grad_x_T[1][0] * grad_x_T[2][1] - grad_x_T[2][0] * grad_x_T[1][1]);
        // det_grad_x = grad_x_T[0][0] * grad_x_T[1][1] * grad_x_T[2][2];  // first order

        if (det_grad_x < det_grad_x_min) det_grad_x_min = det_grad_x;
        if (det_grad_x > det_grad_x_max) det_grad_x_max = det_grad_x;

        // retrieve the gas overdensity
        /// dens = ((1 + CP.delta_bc[index])/det_grad_x - 1);
        dens = (1 / det_grad_x - 1);
        // TODO(brant): baryon-dm relative fluctuations
        // delta_b = (1+f_c \delta_bc )/(det grad x) - 1

        if (dens < dens_min) dens_min = dens;
        if (dens > dens_max) dens_max = dens;

        C.density[id] = dens;
        d_mean += dens;
        d_rms += dens * dens;
        n_d_mean += 1.;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &xix_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &xiy_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &xiz_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &xix_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &xiy_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &xiz_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vx_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vy_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vz_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vx_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vy_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vz_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &xix_n, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &d_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &d_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n_d_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dens_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dens_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  xix_mean = xix_mean / xix_n;
  xiy_mean = xiy_mean / xix_n;
  xiz_mean = xiz_mean / xix_n;
  xix_rms  = xix_rms / xix_n;
  xiy_rms  = xiy_rms / xix_n;
  xiz_rms  = xiz_rms / xix_n;
  xix_rms  = sqrt(xix_rms);
  xiy_rms  = sqrt(xiy_rms);
  xiz_rms  = sqrt(xiz_rms);
  vx_mean  = vx_mean / xix_n;
  vy_mean  = vy_mean / xix_n;
  vz_mean  = vz_mean / xix_n;
  vx_rms   = vx_rms / xix_n;
  vy_rms   = vy_rms / xix_n;
  vz_rms   = vz_rms / xix_n;
  vx_rms   = sqrt(vx_rms);
  vy_rms   = sqrt(vy_rms);
  vz_rms   = sqrt(vz_rms);
  d_mean /= n_d_mean;
  d_rms /= n_d_mean;
  d_rms = sqrt(d_rms);

  chprintf("Cosmological ICs: x-displacement field average = %e, rms = %e\n", xix_mean, xix_rms);
  chprintf("Cosmological ICs: y-displacement field average = %e, rms = %e\n", xiy_mean, xiy_rms);
  chprintf("Cosmological ICs: z-displacement field average = %e, rms = %e\n", xiz_mean, xiz_rms);
  chprintf("Cosmological ICs: x-velocity     field average = %e, rms = %e\n", vx_mean, vx_rms);
  chprintf("Cosmological ICs: y-velocity     field average = %e, rms = %e\n", vy_mean, vy_rms);
  chprintf("Cosmological ICs: z-velocity     field average = %e, rms = %e\n", vz_mean, vz_rms);
  chprintf("Cosmological ICs: overdensity    field average = %e, rms = %e\n", d_mean, d_rms);
  chprintf("Cosmological ICs: overdensity    field minimum = %e, max = %e\n", dens_min, dens_max);

  // correct the overdensity for non-zero mean
  d_rms         = 0;
  n_d_mean      = 0;
  Real dm_check = 0;
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;
        C.density[id] -= d_mean;  // correct for imprecision in delta
        dm_check += C.density[id];
        d_rms += C.density[id] * C.density[id];
        n_d_mean += 1;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &dm_check, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &d_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n_d_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  dm_check /= n_d_mean;
  d_rms /= n_d_mean;
  d_rms = sqrt(d_rms);
  chprintf("Cosmological ICs: corr overdens. field average = %e, rms = %e\n", dm_check, d_rms);

  // set the initial values of the conserved variables
  dens_min = 1.0e9;
  dens_max = -1.0e9;
  dm_check = 0;
  d_rms    = 0;
  n_d_mean = 0;
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        kk    = k - H.n_ghost;
        jj    = j - H.n_ghost;
        ii    = i - H.n_ghost;
        index = ii + jj * nx_local + kk * nx_local * ny_local;

        C.density[id] += 1;      // convert to 1 + delta
        C.density[id] *= rho_b;  // convert to density
        dens = C.density[id];    // store baryon density
        if (dens < dens_min) {
          dens_min = dens;
        }
        if (C.density[id] > dens_max) {
          dens_max = dens;
        }
        dm_check += dens;
        d_rms += dens * dens;
        n_d_mean += 1;

    #ifdef CHEMISTRY_GPU
        C.HI_density[id]    = (1 - xHp) * (1 - YHe) * dens;  // HI    density
        C.HII_density[id]   = xHp * (1 - YHe) * dens;        // HII   density
        C.HeI_density[id]   = (1 - xHep) * YHe * dens;       // HeI   density
        C.HeII_density[id]  = xHep * YHe * dens;             // HeII  density
        C.HeIII_density[id] = 0;                             // HeIII density
                                                             // C.e_density[id]     = dens_factor;
    #endif

        // compute the momenta
        C.momentum_x[id] *= dens;
        C.momentum_y[id] *= dens;
        C.momentum_z[id] *= dens;

        // note that if gamma = 1.6667
        // then 1./(gamma-1) * KB *1e-10 / MP = 0.012381617873714293 in (km/s)^2/K
        // so a 100K gas has U ~ 1.24 * density
        // with the comoving mean density, these units are correct for U
        // U is in Msun/kpc^3 (km/s)^2
        U = T_init / (gamma - 1) / MP * KB * 1e-10 * dens;

        // kinetic energy
        vx = C.momentum_x[id];
        vy = C.momentum_y[id];
        vz = C.momentum_z[id];
        E  = 0.5 * (vx * vx + vy * vy + vz * vz) / dens;

        // add the kinetic energy to the
        // total energy
        C.Energy[id] = U + E;

    #ifdef DE
        // initialize the internal energy
        C.GasEnergy[id] = U;
    #endif
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &dm_check, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &d_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n_d_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dens_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dens_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  dm_check /= n_d_mean;
  d_rms /= n_d_mean;
  d_rms = sqrt(d_rms);
  chprintf("Cosmological ICs: final baryon field average = %e (rho_b = %e), sqrt ave sq = %e\n", dm_check, rho_b,
           d_rms);
  chprintf("Cosmological ICs: final baryon field minimum = %e, max = %e\n", dens_min, dens_max);

  #endif

  // Now, for all the other quantities we need
  // to wait until the particles are initialized
  chprintf("Cosmological ICs: Gas grid initialized...\n");
  chprintf("Cosmological ICs: Ready for particle initialization...\n");

  // write potential to file
  // chprintf("Writing cosmological potential to file...\n");
  // Save_Cosmo_Potential(&P);

#endif  // COSMOLOGY
}

void Grid3D::Chemistry_Test(struct Parameters P)
{
  chprintf("Initializing Chemistry Test...\n");

#ifdef COSMOLOGY
  Real H0, Omega_M, Omega_L, Omega_b, current_z, rho_gas_mean, kpc_cgs, G, z, h, mu, T0, U, rho_gas;
  Real HI_frac, HII_frac, HeI_frac, HeII_frac, HeIII_frac, e_frac, metal_frac, _min;

  H0      = P.H0;
  Omega_M = P.Omega_M;
  Omega_L = P.Omega_L;
  Omega_b = P.Omega_b;
  z       = P.Init_redshift;
  kpc_cgs = KPC_CGS;
  G       = G_COSMO;
  h       = H0 / 100;
  T0      = 230.0;

  // M_sun = MSUN_CGS;
  rho_gas_mean = 3 * pow(H0 * 1e-3, 2) / (8 * M_PI * G) * Omega_b / pow(h, 2);
  chprintf(" z = %f \n", z);
  chprintf(" HO = %f \n", H0);
  chprintf(" Omega_L = %f \n", Omega_L);
  chprintf(" Omega_M = %f \n", Omega_M);
  chprintf(" Omega_b = %f \n", Omega_b);
  chprintf(" rho_gas_mean = %f h^2 Msun kpc^-3\n", rho_gas_mean);
  chprintf(" T0 = %f k\n", T0);
  rho_gas = rho_gas_mean * pow(h, 2) / pow(kpc_cgs, 3) * MSUN_CGS;
  chprintf(" rho_gas = %e g/cm^3\n", rho_gas);

  // frac_min = 1e-10;
  // HI_frac = INITIAL_FRACTION_HI;
  // HII_frac = frac_min;
  // HeI_frac = INITIAL_FRACTION_HEI;
  // HeII_frac = frac_min;
  // HeIII_frac = frac_min;
  // e_frac = HII_frac + HeII_frac + 2*HeIII_frac;
  //
  HI_frac    = INITIAL_FRACTION_HI;
  HII_frac   = INITIAL_FRACTION_HII;
  HeI_frac   = INITIAL_FRACTION_HEI;
  HeII_frac  = INITIAL_FRACTION_HEII;
  HeIII_frac = INITIAL_FRACTION_HEIII;
  e_frac     = INITIAL_FRACTION_ELECTRON;
  metal_frac = INITIAL_FRACTION_METAL;

  mu = (HI_frac + HII_frac + HeI_frac + HeII_frac + HeIII_frac) /
       (HI_frac + HII_frac + (HeI_frac + HeII_frac + HeIII_frac) / 4 + e_frac);
  U = rho_gas_mean * T0 / (gama - 1) / MP / mu * KB * 1e-10;
  chprintf(" mu = %f \n", mu);
  chprintf(" U0 = %f \n", U);

  chprintf(" HI_0 = %f \n", rho_gas_mean * HI_frac);

  int i, j, k, id;
  // set the initial values of the conserved variables
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        C.density[id]    = rho_gas_mean;
        C.momentum_x[id] = 0;
        C.momentum_y[id] = 0;
        C.momentum_z[id] = 0;
        C.Energy[id]     = U;

  #ifdef DE
        C.GasEnergy[id] = U;
  #endif

  #ifdef CHEMISTRY_GPU
        C.HI_density[id]    = rho_gas_mean * HI_frac;
        C.HII_density[id]   = rho_gas_mean * HII_frac;
        C.HeI_density[id]   = rho_gas_mean * HeI_frac;
        C.HeII_density[id]  = rho_gas_mean * HeII_frac;
        C.HeIII_density[id] = rho_gas_mean * HeIII_frac;
        C.e_density[id]     = rho_gas_mean * e_frac;
  #endif

  #ifdef COOLING_GRACKLE
        C.HI_density[id]    = rho_gas_mean * HI_frac;
        C.HII_density[id]   = rho_gas_mean * HII_frac;
        C.HeI_density[id]   = rho_gas_mean * HeI_frac;
        C.HeII_density[id]  = rho_gas_mean * HeII_frac;
        C.HeIII_density[id] = rho_gas_mean * HeIII_frac;
        C.e_density[id]     = rho_gas_mean * e_frac;
    #ifdef GRACKLE_METALS
        C.metal_density[id] = rho_gas_mean * metal_frac;
    #endif
  #endif
      }
    }
  }

#else   // COSMOLOGY
  chprintf("This requires COSMOLOGY turned on! \n");
  chexit(-1);
#endif  // COSMOLOGY
}

#include "../chemistry_gpu/chemistry_gpu.h"

void Grid3D::Iliev0(struct Parameters P)
{
#if defined(CHEMISTRY_GPU)

  Chem.H.H_fraction       = 1;
  Chem.recombination_case = 1;

  Real rho = MP * 1 / DENSITY_UNIT;         // 1 per cc
  Real U   = 1.5 * KB * 100 / ENERGY_UNIT;  // 100 K

  chprintf("rho=%g U=%g\n", rho, U);

  int i, j, k, id;
  for (k = 0; k < H.nz; k++) {
    for (j = 0; j < H.ny; j++) {
      for (i = 0; i < H.nx; i++) {
        // get cell index
        id = i + j * H.nx + k * H.nx * H.ny;

        C.density[id]    = rho;
        C.momentum_x[id] = 0;
        C.momentum_y[id] = 0;
        C.momentum_z[id] = 0;
        C.Energy[id]     = U;

  #ifdef DE
        C.GasEnergy[id] = U;
  #endif

        C.HI_density[id]    = rho * 1;
        C.HII_density[id]   = rho * 1.0e-10;
        C.HeI_density[id]   = rho * 1.0e-10;
        C.HeII_density[id]  = rho * 1.0e-10;
        C.HeIII_density[id] = rho * 1.0e-10;
        C.e_density[id]     = 0;
      }
    }
  }
#else   // defined(CHEMISTRY_GPU)
  chprintf("This requires CHEMISTRY_GPU turned on! \n");
  chexit(-1);
#endif  // defined(CHEMISTRY_GPU)
  chprintf("Iliev0 ICs set....\n");
}

#ifdef RT
  #include "../radiation/alt/atomic_data.h"
  #include "../radiation/alt/photo_rates_csi_gpu.h"
  #include "../radiation/alt/rt_constants.h"
  #include "../radiation/alt/spectral_shape.h"
#endif

void Grid3D::Iliev15(struct Parameters P, int test)
{
#if defined(RT) && defined(CHEMISTRY_GPU)

  Chem.H.H_fraction       = 1;
  Chem.recombination_case = (test == 1 ? 2 : 1);

  Real U, rho = 1.670673249e-24 * 1.0e-3 / DENSITY_UNIT;  // 1.0e-3 per cc
  Real xe = 1.2e-3;
  switch (test) {
    case 1: {
      U = 1.5 * KB * 1.0e4 * 1.0e-3 / ENERGY_UNIT;
      break;
    }
    case 5: {
      U = 1.5 * KB * 1.0e2 * 1.0e-3 / ENERGY_UNIT;
      break;
    }
    default: {
      fprintf(stderr, "Invalid test parameter %d.\n", test);
    }
  }

  chprintf("rho=%g U=%g recombination_case %d\n", rho, U, Chem.recombination_case);

  double xcen[3] = {H.xbound + 0.5 * H.xdglobal, H.ybound + 0.5 * H.ydglobal, H.zbound + 0.5 * H.zdglobal};
  double dx2     = H.dx * H.dx;

  #ifdef RT_OTVET
  Rad.rtFields.et = (Real *)malloc(H.n_cells * sizeof(Real) * 6);
  #endif  // RT_OTVET
  Rad.rtFields.rs = (Real *)malloc(H.n_cells * sizeof(Real));

  auto xs = rt_physics::rt_atomic_data::CrossSections();
  std::vector<float> spectralShape(xs->nxi, 0);
  if (test == 1) {
    //
    //  6.34/5.92 is because the frequency bin at HI threshold has the left edge at Ry, and the bin center is at
    //  Ry*exp(0.5*xiStep) = 1.025*Ry, where the cross section is 5.92e-18, not 6.34e-18.
    //
    // normal solution
    spectralShape[xs->thresholds[rt_physics::rt_atomic_data::CrossSection::IonizationHI].idx] =
        5e48 / RTConstant::c / pow(LENGTH_UNIT, 2) / xs->dxi * 6.34 / 5.92;
  } else {
    SpectralShape::BlackBody(1.0e5, spectralShape);
    for (auto &s : spectralShape) {
      s *= 5e48 / RTConstant::c / pow(LENGTH_UNIT, 2);
    }
  }

  Rad.photoRates->Update(0, spectralShape.data(), xs->dxi * RTConstant::c * 1.0e-24);

  int i, j, k, id;
  for (k = 0; k < H.nz; k++) {
    for (j = 0; j < H.ny; j++) {
      for (i = 0; i < H.nx; i++) {
        // get cell index
        id = i + H.nx * (j + H.ny * k);

        C.density[id]    = rho;
        C.momentum_x[id] = 0;
        C.momentum_y[id] = 0;
        C.momentum_z[id] = 0;
        C.Energy[id]     = U;

  #ifdef DE
        C.GasEnergy[id] = U;
  #endif

        C.HI_density[id]    = rho * (1 - xe);
        C.HII_density[id]   = rho * xe;
        C.HeI_density[id]   = rho * 1.0e-20;
        C.HeII_density[id]  = rho * 1.0e-20;
        C.HeIII_density[id] = rho * 1.0e-20;

        // TODO(brant): e_density may be duplicative without metals
        C.e_density[id] = C.HII_density[id] + C.HeI_density[id] + 2 * C.HeIII_density[id];

        double x[3] = {H.xblocal + H.dx * (i + 0.5 - H.n_ghost), H.yblocal + H.dy * (j + 0.5 - H.n_ghost),
                       H.zblocal + H.dz * (k + 0.5 - H.n_ghost)};

        double r2 = 0;
        for (int axis = 0; axis < 3; axis++) {
          x[axis] -= xcen[axis];
          r2 += x[axis] * x[axis];
        }

        auto eps2ot = dx2;
        //
        //  NG 230117: ET seems to require larger softening than OT, why this is so I do not understand, need to explore
        //  further.
        //
        auto eps2et = 4 * eps2ot;

        Rad.rtFields.rs[id] = (r2 < dx2 ? 0.125 / pow(H.dx, 3) : 0);
        Rad.rtFields.rf[id] = 1 / (12.5664 * (eps2ot + r2));

  #ifdef RT_OTVET
        for (int ii = 1; ii < 1 + Rad.n_fpfreq * Rad.n_freq; ii++) Rad.rtFields.rf[id + ii * H.n_cells] = 0;
        Rad.rtFields.et[id + 0 * H.n_cells] = (eps2et / 3 + x[0] * x[0]) / (eps2et + r2);
        Rad.rtFields.et[id + 1 * H.n_cells] = (x[1] * x[0]) / (eps2et + r2);
        Rad.rtFields.et[id + 2 * H.n_cells] = (eps2et / 3 + x[1] * x[1]) / (eps2et + r2);
        Rad.rtFields.et[id + 3 * H.n_cells] = (x[2] * x[0]) / (eps2et + r2);
        Rad.rtFields.et[id + 4 * H.n_cells] = (x[2] * x[1]) / (eps2et + r2);
        Rad.rtFields.et[id + 5 * H.n_cells] = (eps2et / 3 + x[2] * x[2]) / (eps2et + r2);
  #elif defined(RT_M1)
        for (int ii = 0; ii < Rad.n_fpfreq * Rad.n_freq; ii++) Rad.rtFields.rf[id + ii * H.n_cells] = 0;
  #endif  // RT_M1
      }
    }
  }

  chprintf("Iliev15 ICs finalized...\n");

#else   // !defined(RT) && defined(CHEMISTRY_GPU)
  chprintf("This requires RT && CHEMISTRY_GPU turned on! \n");
  chexit(-1);
#endif  // defined(RT) && defined(CHEMISTRY_GPU)
}

void Grid3D::Iliev6(struct Parameters P)
{
#if defined(RT) && defined(CHEMISTRY_GPU)

  Chem.H.H_fraction       = 1;
  Chem.recombination_case = 1;

  Real rho0 = 1.670673249e-24 * 3.2 / DENSITY_UNIT;  // 3.2 per cc
  Real xe   = 0;
  Real U0   = 1.5 * KB * 1.0e2 * 3.2 / ENERGY_UNIT;

  double xcen[3] = {H.xbound + 0.5 * H.xdglobal, H.ybound + 0.5 * H.ydglobal, H.zbound + 0.5 * H.zdglobal};
  double dx2     = H.dx * H.dx;

  #ifdef RT_OTVET
  Rad.rtFields.et = (Real *)malloc(H.n_cells * sizeof(Real) * 6);
  #endif  // RT_OTVET
  Rad.rtFields.rs = (Real *)malloc(H.n_cells * sizeof(Real));

  auto xs = rt_physics::rt_atomic_data::CrossSections();
  std::vector<float> spectralShape(xs->nxi, 0);
  SpectralShape::BlackBody(1.0e5, spectralShape);
  for (auto &s : spectralShape) {
    s *= 1.0e50 / RTConstant::c / pow(LENGTH_UNIT, 2);
  }

  Rad.photoRates->Update(0, spectralShape.data(), xs->dxi * RTConstant::c * 1.0e-24);

  Real r2core = pow(0.0915 / (2 * 0.8) * H.xdglobal, 2);

  int i, j, k, id;
  for (k = 0; k < H.nz; k++) {
    for (j = 0; j < H.ny; j++) {
      for (i = 0; i < H.nx; i++) {
        // get cell index
        id = i + H.nx * (j + H.ny * k);

        double x[3] = {H.xblocal + H.dx * (i + 0.5 - H.n_ghost), H.yblocal + H.dy * (j + 0.5 - H.n_ghost),
                       H.zblocal + H.dz * (k + 0.5 - H.n_ghost)};

        double r2 = 0;
        for (int axis = 0; axis < 3; axis++) {
          x[axis] -= xcen[axis];
          r2 += x[axis] * x[axis];
        }

        Real rho = rho0 * (r2 < r2core ? 1 : (r2core / r2));
        Real U   = U0 * (r2 < r2core ? 1 : (r2core / r2));

        C.density[id]    = rho;
        C.momentum_x[id] = 0;
        C.momentum_y[id] = 0;
        C.momentum_z[id] = 0;
        C.Energy[id]     = U;

  #ifdef DE
        C.GasEnergy[id] = U;
  #endif

        C.HI_density[id]    = rho * (1 - xe);
        C.HII_density[id]   = rho * xe;
        C.HeI_density[id]   = rho * 1.0e-20;
        C.HeII_density[id]  = rho * 1.0e-20;
        C.HeIII_density[id] = rho * 1.0e-20;

        auto eps2ot = 4 * dx2;
        //
        //  NG 230117: ET seems to require larger softening than OT, why this is so I do not understand, need to explore
        //  further.
        //
        auto eps2et = 4 * eps2ot;

        Rad.rtFields.rs[id] = (r2 < dx2 ? 0.125 / pow(H.dx, 3) : 0);
        Rad.rtFields.rf[id] = 1 / (12.5664 * (eps2ot + r2));

  #ifdef RT_OTVET
        for (int ii = 1; ii < 1 + Rad.n_fpfreq * Rad.n_freq; ii++) Rad.rtFields.rf[id + ii * H.n_cells] = 0;
        Rad.rtFields.et[id + 0 * H.n_cells] = (eps2et / 3 + x[0] * x[0]) / (eps2et + r2);
        Rad.rtFields.et[id + 1 * H.n_cells] = (x[1] * x[0]) / (eps2et + r2);
        Rad.rtFields.et[id + 2 * H.n_cells] = (eps2et / 3 + x[1] * x[1]) / (eps2et + r2);
        Rad.rtFields.et[id + 3 * H.n_cells] = (x[2] * x[0]) / (eps2et + r2);
        Rad.rtFields.et[id + 4 * H.n_cells] = (x[2] * x[1]) / (eps2et + r2);
        Rad.rtFields.et[id + 5 * H.n_cells] = (eps2et / 3 + x[2] * x[2]) / (eps2et + r2);
  #endif  // RT_OTVET
  #ifdef RT_M1
        for (int ii = 1; ii < Rad.n_fpfreq * Rad.n_freq; ii++) Rad.rtFields.rf[id + ii * H.n_cells] = 0;
  #endif  // RT_M1
      }
    }
  }

#else   // !defined(RT) && defined(CHEMISTRY_GPU)
  chprintf("This requires RT && CHEMISTRY_GPU turned on! \n");
  chexit(-1);
#endif  // defined(RT) && defined(CHEMISTRY_GPU)
}

#ifdef MHD
void Grid3D::Circularly_Polarized_Alfven_Wave(struct Parameters const P)
{
  // This test is only meaningful for a limited number of parameter values so I will check them here
  assert(P.polarization == 1.0 or
         P.polarization == -1.0 and
             "The polarization for this test must be 1 (right polarized) or -1 (left polarized).");
  assert(std::abs(P.vx) == 1.0 or
         P.vx == 0.0 and "The x velocity for this test must be 0 (traveling wave) or 1 (standing wave).");

  // Check the domain and angles
  auto checkDomain = [](int const &nx, int const &ny, int const &nz, Real const &xlen, Real const &ylen,
                        Real const &zlen) {
    assert(nx == 2 * ny and nx == 2 * nz and "This test requires that the number of cells be of shape 2L x L x L");
    assert(xlen == 2 * ylen and xlen == 2 * zlen and "This test requires that the domain be of shape 2L x L x L");
  };
  if ((P.pitch == 0.0 and P.yaw == 0.0) or (P.pitch == std::asin(2. / 3.) and P.yaw == std::asin(2. / std::sqrt(5.)))) {
    checkDomain(P.nx, P.ny, P.nz, P.xlen, P.ylen, P.zlen);
  } else if (P.pitch == 0.5 * M_PI and P.yaw == 0.0) {
    checkDomain(P.ny, P.nz, P.nx, P.ylen, P.zlen, P.xlen);
  } else if (P.pitch == 0.0 and P.yaw == 0.5 * M_PI) {
    checkDomain(P.nz, P.nx, P.ny, P.zlen, P.xlen, P.ylen);
  } else {
    assert(false and "This test does not support these angles");
  }

  // Parameters for tests.
  Real const density    = 1.0;
  Real const pressure   = 0.1;
  Real const velocity_x = P.vx;
  Real const amplitude  = 0.1;  // the amplitude of the wave
  Real const magnetic_x = 1.0;

  // Angles
  Real const sin_yaw   = std::sin(P.yaw);
  Real const cos_yaw   = std::cos(P.yaw);
  Real const sin_pitch = std::sin(P.pitch);
  Real const cos_pitch = std::cos(P.pitch);

  // Compute the wave quantities
  Real const wavelength = 1.;
  Real const wavenumber = 2.0 * M_PI / wavelength;  // the angular wave number k

  // Compute the vector potentials
  std::vector<Real> vectorPotential(3 * H.n_cells, 0);
  auto Compute_Vector_Potential = [&](Real const &x_loc, Real const &y_loc, Real const &z_loc) {
    // The "_rot" variables are the rotated version
    Real const x_rot = x_loc * cos_pitch * cos_yaw + y_loc * cos_pitch * sin_yaw + z_loc * sin_pitch;
    Real const y_rot = -x_loc * sin_yaw + y_loc * cos_yaw;

    Real const a_y = P.polarization * (amplitude / wavenumber) * std::sin(wavenumber * x_rot);
    Real const a_z = (amplitude / wavenumber) * std::cos(wavenumber * x_rot) + magnetic_x * y_rot;

    return std::make_pair(a_y, a_z);
  };

  for (int k = 0; k < H.nz; k++) {
    for (int j = 0; j < H.ny; j++) {
      for (int i = 0; i < H.nx; i++) {
        // Get cell index
        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        Real x, y, z;
        Get_Position(i, j, k, &x, &y, &z);

        auto vectorPot                         = Compute_Vector_Potential(x, y + H.dy / 2., z + H.dz / 2.);
        vectorPotential.at(id + 0 * H.n_cells) = -vectorPot.first * sin_yaw - vectorPot.second * sin_pitch * cos_yaw;

        vectorPot                              = Compute_Vector_Potential(x + H.dx / 2., y, z + H.dz / 2.);
        vectorPotential.at(id + 1 * H.n_cells) = vectorPot.first * cos_yaw - vectorPot.second * sin_pitch * sin_yaw;

        vectorPot                              = Compute_Vector_Potential(x + H.dx / 2., y + H.dy / 2., z);
        vectorPotential.at(id + 2 * H.n_cells) = vectorPot.second * cos_pitch;
      }
    }
  }

  // Compute the magnetic field
  mhd::utils::Init_Magnetic_Field_With_Vector_Potential(H, C, vectorPotential);

  // set initial values of non-magnetic conserved variables
  for (int k = H.n_ghost - 1; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost - 1; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost - 1; i < H.nx - H.n_ghost; i++) {
        // get cell index
        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        // get cell-centered position
        Real x_pos, y_pos, z_pos;
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        Real const x_pos_rot = x_pos * cos_pitch * cos_yaw + y_pos * cos_pitch * sin_yaw + z_pos * sin_pitch;

        // Compute the momentum
        Real const momentum_x = density * velocity_x;
        Real const momentum_y = -P.polarization * density * amplitude * std::sin(wavenumber * x_pos_rot);
        Real const momentum_z = -density * amplitude * std::cos(wavenumber * x_pos_rot);
        Real const momentum_x_rot =
            momentum_x * cos_pitch * cos_yaw - momentum_y * sin_yaw - momentum_z * sin_pitch * cos_yaw;
        Real const momentum_y_rot =
            momentum_x * cos_pitch * sin_yaw + momentum_y * cos_yaw - momentum_z * sin_pitch * sin_yaw;
        Real const momentum_z_rot = momentum_x * sin_pitch + momentum_z * cos_pitch;

        // Compute the Energy
        auto const magnetic_centered =
            mhd::utils::cellCenteredMagneticFields(C.host, id, i, j, k, H.n_cells, H.nx, H.ny);
        Real const energy = hydro_utilities::Calc_Energy_Conserved(pressure, density, momentum_x_rot, momentum_y_rot,
                                                                   momentum_z_rot, ::gama, magnetic_centered.x(),
                                                                   magnetic_centered.y(), magnetic_centered.z());

        // Final assignment
        C.density[id]    = density;
        C.momentum_x[id] = momentum_x_rot;
        C.momentum_y[id] = momentum_y_rot;
        C.momentum_z[id] = momentum_z_rot;
        C.Energy[id]     = energy;
      }
    }
  }
}

void Grid3D::Advecting_Field_Loop(struct Parameters const P)
{
  // This test is only meaningful for a limited number of parameter values so I will check them here
  // Check that the domain is centered on zero
  assert((P.xmin + P.xlen / 2) == 0 and (P.ymin + P.ylen / 2) == 0 and (P.zmin + P.zlen / 2 == 0) and
         "Domain must be centered at zero");

  // Check that P.radius is smaller than the size of the domain
  Real const domain_size = std::hypot(P.xlen / 2, P.ylen / 2, P.zlen / 2);
  assert(domain_size > P.radius and "The size of the domain must be greater than P.radius");

  // Compute the vector potential. Since the vector potential std::vector is initialized to zero I will only assign new
  // values when required and ignore the cases where I would be assigning zero
  std::vector<Real> vectorPotential(3 * H.n_cells, 0);
  for (int k = 0; k < H.nz; k++) {
    for (int j = 0; j < H.ny; j++) {
      for (int i = 0; i < H.nx; i++) {
        // Get cell index
        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        // Get the cell centered positions
        Real x, y, z;
        Get_Position(i, j, k, &x, &y, &z);

        // Y vector potential
        Real radius = std::hypot(x + H.dx / 2., y, z + H.dz / 2.);
        if (radius < P.radius) {
          vectorPotential.at(id + 1 * H.n_cells) = P.A * (P.radius - radius);
        }

        // Z vector potential
        radius = std::hypot(x + H.dx / 2., y + H.dy / 2., z);
        if (radius < P.radius) {
          vectorPotential.at(id + 2 * H.n_cells) = P.A * (P.radius - radius);
        }
      }
    }
  }

  // Initialize the magnetic fields
  mhd::utils::Init_Magnetic_Field_With_Vector_Potential(H, C, vectorPotential);

  // Initialize the hydro variables
  for (int k = H.n_ghost - 1; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost - 1; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost - 1; i < H.nx - H.n_ghost; i++) {
        // get cell index
        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        // Compute the cell centered magnetic fields
        auto const magnetic_centered =
            mhd::utils::cellCenteredMagneticFields(C.host, id, i, j, k, H.n_cells, H.nx, H.ny);

        // Assignment
        C.density[id]    = P.rho;
        C.momentum_x[id] = P.rho * P.vx;
        C.momentum_y[id] = P.rho * P.vy;
        C.momentum_z[id] = P.rho * P.vz;
        C.Energy[id]     = hydro_utilities::Calc_Energy_Conserved(P.P, P.rho, C.momentum_x[id], C.momentum_y[id],
                                                                  C.momentum_z[id], ::gama, magnetic_centered.x(),
                                                                  magnetic_centered.y(), magnetic_centered.z());
      }
    }
  }
}

void Grid3D::MHD_Spherical_Blast(struct Parameters const P)
{
  // This test is only meaningful for a limited number of parameter values so I will check them here
  // Check that the domain is centered on zero
  assert((P.xmin + P.xlen / 2) == 0 and (P.ymin + P.ylen / 2) == 0 and (P.zmin + P.zlen / 2 == 0) and
         "Domain must be centered at zero");

  // Check that P.radius is smaller than the size of the domain
  Real const domain_size = std::hypot(P.xlen / 2, P.ylen / 2, P.zlen / 2);
  assert(domain_size > P.radius and "The size of the domain must be greater than P.radius");

  // Initialize the magnetic field
  for (int k = H.n_ghost - 1; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost - 1; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost - 1; i < H.nx - H.n_ghost; i++) {
        // get cell index
        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        C.magnetic_x[id] = P.Bx;
        C.magnetic_y[id] = P.By;
        C.magnetic_z[id] = P.Bz;
      }
    }
  }

  for (int k = H.n_ghost - 1; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost - 1; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost - 1; i < H.nx - H.n_ghost; i++) {
        // get cell index
        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        // Set the fields that don't depend on pressure
        C.density[id]    = P.rho;
        C.momentum_x[id] = P.rho * P.vx;
        C.momentum_y[id] = P.rho * P.vy;
        C.momentum_z[id] = P.rho * P.vz;

        // Get the cell centered positions
        Real x, y, z;
        Get_Position(i, j, k, &x, &y, &z);

        // Compute the magnetic field in this cell
        auto const magnetic_centered =
            mhd::utils::cellCenteredMagneticFields(C.host, id, i, j, k, H.n_cells, H.nx, H.ny);

        // Set the field(s) that do depend on pressure. That's just energy
        Real const radius = std::hypot(x, y, z);
        Real pressure;
        if (radius < P.radius) {
          pressure = P.P_blast;
        } else {
          pressure = P.P;
        }
        C.Energy[id] = hydro_utilities::Calc_Energy_Conserved(
            pressure, C.density[id], C.momentum_x[id], C.momentum_y[id], C.momentum_z[id], ::gama,
            magnetic_centered.x(), magnetic_centered.y(), magnetic_centered.z());
      }
    }
  }
}

void Grid3D::Orszag_Tang_Vortex()
{
  // This problem requires specific parameters so I will define them here
  Real const magnetic_background = 1.0 / std::sqrt(4.0 * M_PI);
  Real const density_background  = 25.0 / (36.0 * M_PI);
  Real const velocity_background = 1.0;
  Real const pressure_background = 5.0 / (12.0 * M_PI);

  // Compute the vector potential. Since the vector potential std::vector is initialized to zero I will only assign new
  // values when required and ignore the cases where I would be assigning zero
  std::vector<Real> vectorPotential(3 * H.n_cells, 0);
  for (int k = 0; k < H.nz; k++) {
    for (int j = 0; j < H.ny; j++) {
      for (int i = 0; i < H.nx; i++) {
        // Get cell index
        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        // Get the cell centered positions
        Real x, y, z;
        Get_Position(i, j, k, &x, &y, &z);

        // Z vector potential
        vectorPotential.at(id + 2 * H.n_cells) =
            magnetic_background / (4.0 * M_PI) * (std::cos(4.0 * M_PI * x) + 2.0 * std::cos(2.0 * M_PI * y));
      }
    }
  }

  // Initialize the magnetic fields
  mhd::utils::Init_Magnetic_Field_With_Vector_Potential(H, C, vectorPotential);

  // Initialize the hydro variables
  for (int k = H.n_ghost - 1; k < H.nz - H.n_ghost; k++) {
    for (int j = H.n_ghost - 1; j < H.ny - H.n_ghost; j++) {
      for (int i = H.n_ghost - 1; i < H.nx - H.n_ghost; i++) {
        // get cell index
        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);

        // Get the cell centered positions
        Real x, y, z;
        Get_Position(i, j, k, &x, &y, &z);

        // Compute the cell centered magnetic fields
        auto const magnetic_centered =
            mhd::utils::cellCenteredMagneticFields(C.host, id, i, j, k, H.n_cells, H.nx, H.ny);

        // Assignment
        C.density[id]    = density_background;
        C.momentum_x[id] = density_background * velocity_background * std::sin(2.0 * M_PI * y);
        C.momentum_y[id] = -density_background * velocity_background * std::sin(2.0 * M_PI * x);
        C.momentum_z[id] = 0.0;
        C.Energy[id]     = hydro_utilities::Calc_Energy_Conserved(
            pressure_background, C.density[id], C.momentum_x[id], C.momentum_y[id], C.momentum_z[id], ::gama,
            magnetic_centered.x(), magnetic_centered.y(), magnetic_centered.z());
      }
    }
  }
}
#endif  // MHD

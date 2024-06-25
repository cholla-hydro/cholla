/*! \file ppm_cuda.cu
 *  \brief Functions definitions for the ppm kernels, using characteristic
 tracing. Written following Stone et al. 2008. */

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/ppm_cuda.h"
#include "../reconstruction/reconstruction_internals.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif

void __device__ __host__ PPM_Characteristic_Evolution(hydro_utilities::Primitive const cell_i, Real const dt,
                                                      Real const dx, Real const gamma,
                                                      hydro_utilities::Primitive &interface_R_imh,
                                                      hydro_utilities::Primitive &interface_L_iph)
{
  // This is the beginning of the characteristic tracing
  // Step 8 - Compute the coefficients for the monotonized parabolic
  // interpolation function
  //          Stone Eqn 54
  hydro_utilities::Primitive interface_slope;
  interface_slope.density      = interface_L_iph.density - interface_R_imh.density;
  interface_slope.velocity.x() = interface_L_iph.velocity.x() - interface_R_imh.velocity.x();
  interface_slope.velocity.y() = interface_L_iph.velocity.y() - interface_R_imh.velocity.y();
  interface_slope.velocity.z() = interface_L_iph.velocity.z() - interface_R_imh.velocity.z();
  interface_slope.pressure     = interface_L_iph.pressure - interface_R_imh.pressure;

  Real const d_6  = 6.0 * (cell_i.density - 0.5 * (interface_R_imh.density + interface_L_iph.density));
  Real const vx_6 = 6.0 * (cell_i.velocity.x() - 0.5 * (interface_R_imh.velocity.x() + interface_L_iph.velocity.x()));
  Real const vy_6 = 6.0 * (cell_i.velocity.y() - 0.5 * (interface_R_imh.velocity.y() + interface_L_iph.velocity.y()));
  Real const vz_6 = 6.0 * (cell_i.velocity.z() - 0.5 * (interface_R_imh.velocity.z() + interface_L_iph.velocity.z()));
  Real const p_6  = 6.0 * (cell_i.pressure - 0.5 * (interface_R_imh.pressure + interface_L_iph.pressure));

#ifdef DE
  interface_slope.gas_energy_specific = interface_L_iph.gas_energy_specific - interface_R_imh.gas_energy_specific;
  Real const ge_6                     = 6.0 * (cell_i.gas_energy_specific -
                           0.5 * (interface_R_imh.gas_energy_specific + interface_L_iph.gas_energy_specific));
#endif  // DE

#ifdef SCALAR
  Real scalar_6[NSCALARS];
  for (int i = 0; i < NSCALARS; i++) {
    interface_slope.scalar_specific[i] = interface_L_iph.scalar_specific[i] - interface_R_imh.scalar_specific[i];
    scalar_6[i]                        = 6.0 * (cell_i.scalar_specific[i] -
                         0.5 * (interface_R_imh.scalar_specific[i] + interface_L_iph.scalar_specific[i]));
  }
#endif  // SCALAR

  // Compute the eigenvalues of the linearized equations in the
  // primitive variables using the cell-centered primitive variables

  // recalculate the adiabatic sound speed in cell i
  Real const sound_speed = hydro_utilities::Calc_Sound_Speed(cell_i.pressure, cell_i.density, gamma);

  Real const lambda_m = cell_i.velocity.x() - sound_speed;
  Real const lambda_0 = cell_i.velocity.x();
  Real const lambda_p = cell_i.velocity.x() + sound_speed;

  // Step 9 - Compute the left and right interface values using monotonized
  // parabolic interpolation
  //          Stone Eqns 55 & 56

  // largest eigenvalue
  Real const lambda_max = fmax(lambda_p, (Real)0);
  // smallest eigenvalue
  Real const lambda_min = fmin(lambda_m, (Real)0);

  // left interface value, i+1/2
  Real const dtodx = dt / dx;
  interface_L_iph.density =
      interface_L_iph.density -
      lambda_max * (0.5 * dtodx) * (interface_slope.density - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * d_6);
  interface_L_iph.velocity.x() =
      interface_L_iph.velocity.x() -
      lambda_max * (0.5 * dtodx) * (interface_slope.velocity.x() - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vx_6);
  interface_L_iph.velocity.y() =
      interface_L_iph.velocity.y() -
      lambda_max * (0.5 * dtodx) * (interface_slope.velocity.y() - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vy_6);
  interface_L_iph.velocity.z() =
      interface_L_iph.velocity.z() -
      lambda_max * (0.5 * dtodx) * (interface_slope.velocity.z() - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vz_6);
  interface_L_iph.pressure =
      interface_L_iph.pressure -
      lambda_max * (0.5 * dtodx) * (interface_slope.pressure - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * p_6);

  // right interface value, i-1/2
  interface_R_imh.density =
      interface_R_imh.density -
      lambda_min * (0.5 * dtodx) * (interface_slope.density + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * d_6);
  interface_R_imh.velocity.x() =
      interface_R_imh.velocity.x() -
      lambda_min * (0.5 * dtodx) * (interface_slope.velocity.x() + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vx_6);
  interface_R_imh.velocity.y() =
      interface_R_imh.velocity.y() -
      lambda_min * (0.5 * dtodx) * (interface_slope.velocity.y() + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vy_6);
  interface_R_imh.velocity.z() =
      interface_R_imh.velocity.z() -
      lambda_min * (0.5 * dtodx) * (interface_slope.velocity.z() + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vz_6);
  interface_R_imh.pressure =
      interface_R_imh.pressure -
      lambda_min * (0.5 * dtodx) * (interface_slope.pressure + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * p_6);

#ifdef DE
  interface_L_iph.gas_energy_specific =
      interface_L_iph.gas_energy_specific -
      lambda_max * (0.5 * dtodx) *
          (interface_slope.gas_energy_specific - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * ge_6);
  interface_R_imh.gas_energy_specific =
      interface_R_imh.gas_energy_specific -
      lambda_min * (0.5 * dtodx) *
          (interface_slope.gas_energy_specific + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * ge_6);
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar_specific[i] =
        interface_L_iph.scalar_specific[i] -
        lambda_max * (0.5 * dtodx) *
            (interface_slope.scalar_specific[i] - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * scalar_6[i]);
    interface_R_imh.scalar_specific[i] =
        interface_R_imh.scalar_specific[i] -
        lambda_min * (0.5 * dtodx) *
            (interface_slope.scalar_specific[i] + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * scalar_6[i]);
  }
#endif  // SCALAR

  // Step 10 - Perform the characteristic tracing
  //           Stone Eqns 57 - 60

  // left-hand interface value, i+1/2
  Real sum_1 = 0, sum_2 = 0, sum_3 = 0, sum_4 = 0, sum_5 = 0;
#ifdef DE
  Real sum_ge = 0;
  Real chi_ge = 0;
#endif  // DE
#ifdef SCALAR
  Real chi_scalar[NSCALARS];
  Real sum_scalar[NSCALARS];
  for (Real &val : sum_scalar) {
    val = 0;
  }
#endif  // SCALAR

  if (lambda_m >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_m);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_m * lambda_m);

    Real const chi_1 = A * (interface_slope.density - d_6) + B * d_6;
    Real const chi_2 = A * (interface_slope.velocity.x() - vx_6) + B * vx_6;
    Real const chi_3 = A * (interface_slope.velocity.y() - vy_6) + B * vy_6;
    Real const chi_4 = A * (interface_slope.velocity.z() - vz_6) + B * vz_6;
    Real const chi_5 = A * (interface_slope.pressure - p_6) + B * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / sound_speed - chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 - chi_5 / (sound_speed * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * sound_speed - chi_5);
  }
  if (lambda_0 >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_0);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_0 * lambda_0);

    Real const chi_1 = A * (interface_slope.density - d_6) + B * d_6;
    Real const chi_2 = A * (interface_slope.velocity.x() - vx_6) + B * vx_6;
    Real const chi_3 = A * (interface_slope.velocity.y() - vy_6) + B * vy_6;
    Real const chi_4 = A * (interface_slope.velocity.z() - vz_6) + B * vz_6;
    Real const chi_5 = A * (interface_slope.pressure - p_6) + B * p_6;
#ifdef DE
    chi_ge = A * (interface_slope.gas_energy_specific - ge_6) + B * ge_6;
#endif  // DE
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = A * (interface_slope.scalar_specific[i] - scalar_6[i]) + B * scalar_6[i];
    }
#endif  // SCALAR

    sum_1 += chi_1 - chi_5 / (sound_speed * sound_speed);
    sum_3 += chi_3;
    sum_4 += chi_4;
#ifdef DE
    sum_ge += chi_ge;
#endif  // DE
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += chi_scalar[i];
    }
#endif  // SCALAR
  }
  if (lambda_p >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_p);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_p * lambda_p);

    Real const chi_1 = A * (interface_slope.density - d_6) + B * d_6;
    Real const chi_2 = A * (interface_slope.velocity.x() - vx_6) + B * vx_6;
    Real const chi_3 = A * (interface_slope.velocity.y() - vy_6) + B * vy_6;
    Real const chi_4 = A * (interface_slope.velocity.z() - vz_6) + B * vz_6;
    Real const chi_5 = A * (interface_slope.pressure - p_6) + B * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / sound_speed + chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 + chi_5 / (sound_speed * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * sound_speed + chi_5);
  }

  // add the corrections to the initial guesses for the interface values
  interface_L_iph.density += sum_1;
  interface_L_iph.velocity.x() += sum_2;
  interface_L_iph.velocity.y() += sum_3;
  interface_L_iph.velocity.z() += sum_4;
  interface_L_iph.pressure += sum_5;
#ifdef DE
  interface_L_iph.gas_energy_specific += sum_ge;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar_specific[i] += sum_scalar[i];
  }
#endif  // SCALAR

  // right-hand interface value, i-1/2
  sum_1 = 0;
  sum_2 = 0;
  sum_3 = 0;
  sum_4 = 0;
  sum_5 = 0;
#ifdef DE
  sum_ge = 0;
#endif  // DE
#ifdef SCALAR
  for (Real &val : sum_scalar) {
    val = 0;
  }
#endif  // SCALAR
  if (lambda_m <= 0) {
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_m);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_m * lambda_m);

    Real const chi_1 = C * (interface_slope.density + d_6) + D * d_6;
    Real const chi_2 = C * (interface_slope.velocity.x() + vx_6) + D * vx_6;
    Real const chi_3 = C * (interface_slope.velocity.y() + vy_6) + D * vy_6;
    Real const chi_4 = C * (interface_slope.velocity.z() + vz_6) + D * vz_6;
    Real const chi_5 = C * (interface_slope.pressure + p_6) + D * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / sound_speed - chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 - chi_5 / (sound_speed * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * sound_speed - chi_5);
  }
  if (lambda_0 <= 0) {
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_0);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_0 * lambda_0);

    Real const chi_1 = C * (interface_slope.density + d_6) + D * d_6;
    Real const chi_2 = C * (interface_slope.velocity.x() + vx_6) + D * vx_6;
    Real const chi_3 = C * (interface_slope.velocity.y() + vy_6) + D * vy_6;
    Real const chi_4 = C * (interface_slope.velocity.z() + vz_6) + D * vz_6;
    Real const chi_5 = C * (interface_slope.pressure + p_6) + D * p_6;
#ifdef DE
    chi_ge = C * (interface_slope.gas_energy_specific + ge_6) + D * ge_6;
#endif  // DE
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = C * (interface_slope.scalar_specific[i] + scalar_6[i]) + D * scalar_6[i];
    }
#endif  // SCALAR

    sum_1 += chi_1 - chi_5 / (sound_speed * sound_speed);
    sum_3 += chi_3;
    sum_4 += chi_4;
#ifdef DE
    sum_ge += chi_ge;
#endif  // DE
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      sum_scalar[i] += chi_scalar[i];
    }
#endif  // SCALAR
  }
  if (lambda_p <= 0) {
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_p);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_p * lambda_p);

    Real const chi_1 = C * (interface_slope.density + d_6) + D * d_6;
    Real const chi_2 = C * (interface_slope.velocity.x() + vx_6) + D * vx_6;
    Real const chi_3 = C * (interface_slope.velocity.y() + vy_6) + D * vy_6;
    Real const chi_4 = C * (interface_slope.velocity.z() + vz_6) + D * vz_6;
    Real const chi_5 = C * (interface_slope.pressure + p_6) + D * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / sound_speed + chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 + chi_5 / (sound_speed * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * sound_speed + chi_5);
  }

  // add the corrections
  interface_R_imh.density += sum_1;
  interface_R_imh.velocity.x() += sum_2;
  interface_R_imh.velocity.y() += sum_3;
  interface_R_imh.velocity.z() += sum_4;
  interface_R_imh.pressure += sum_5;
#ifdef DE
  interface_R_imh.gas_energy_specific += sum_ge;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar_specific[i] += sum_scalar[i];
  }
#endif  // SCALAR

  // This is the end of the characteristic tracing
}
// =====================================================================================================================
template <int dir>
__global__ __launch_bounds__(TPB) void PPM_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                                int ny, int nz, Real dx, Real dt, Real gamma)
{
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Ensure that we are only operating on cells that will be used
  if (reconstruction::Thread_Guard<3>(nx, ny, nz, xid, yid, zid)) {
    return;
  }

  // Compute the total number of cells
  int const n_cells = nx * ny * nz;

  // Set the field indices for the various directions
  int o1, o2, o3;
  if constexpr (dir == 0) {
    o1 = grid_enum::momentum_x;
    o2 = grid_enum::momentum_y;
    o3 = grid_enum::momentum_z;
  } else if constexpr (dir == 1) {
    o1 = grid_enum::momentum_y;
    o2 = grid_enum::momentum_z;
    o3 = grid_enum::momentum_x;
  } else if constexpr (dir == 2) {
    o1 = grid_enum::momentum_z;
    o2 = grid_enum::momentum_x;
    o3 = grid_enum::momentum_y;
  }

  // load the 5-cell stencil into registers
  // cell i
  hydro_utilities::Primitive const cell_i =
      hydro_utilities::Load_Cell_Primitive<dir>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);

  // cell i-1. The equality checks the direction and will subtract one from the correct direction
  // im1 stands for "i minus 1"
  hydro_utilities::Primitive const cell_im1 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny, n_cells, gamma);

  // cell i+1.  The equality checks the direction and add one to the correct direction
  // ip1 stands for "i plus 1"
  hydro_utilities::Primitive const cell_ip1 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid + int(dir == 0), yid + int(dir == 1), zid + int(dir == 2), nx, ny, n_cells, gamma);

  // cell i-2. The equality checks the direction and will subtract two from the correct direction
  // im2 stands for "i minus 2"
  hydro_utilities::Primitive const cell_im2 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid - 2 * int(dir == 0), yid - 2 * int(dir == 1), zid - 2 * int(dir == 2), nx, ny, n_cells, gamma);

  // cell i+2.  The equality checks the direction and add two to the correct direction
  // ip2 stands for "i plus 2"
  hydro_utilities::Primitive const cell_ip2 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid + 2 * int(dir == 0), yid + 2 * int(dir == 1), zid + 2 * int(dir == 2), nx, ny, n_cells, gamma);

#ifdef PPMC
  // Compute the eigenvectors
  reconstruction::EigenVecs const eigenvectors = reconstruction::Compute_Eigenvectors(cell_i, gamma);

  // Cell i
  reconstruction::Characteristic const cell_i_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_i, eigenvectors, gamma);

  // Cell i-1
  reconstruction::Characteristic const cell_im1_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_im1, eigenvectors, gamma);

  // Cell i-2
  reconstruction::Characteristic const cell_im2_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_im2, eigenvectors, gamma);

  // Cell i+1
  reconstruction::Characteristic const cell_ip1_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_ip1, eigenvectors, gamma);

  // Cell i+2
  reconstruction::Characteristic const cell_ip2_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_ip2, eigenvectors, gamma);

  // Compute the interface states for each field
  auto const [interface_L_iph_characteristic, interface_R_imh_characteristic] =
      reconstruction::PPM_Interfaces(cell_im2_characteristic, cell_im1_characteristic, cell_i_characteristic,
                                     cell_ip1_characteristic, cell_ip2_characteristic);

  // Convert back to primitive variables
  hydro_utilities::Primitive interface_L_iph =
      reconstruction::Characteristic_To_Primitive(cell_i, interface_L_iph_characteristic, eigenvectors, gamma);
  hydro_utilities::Primitive interface_R_imh =
      reconstruction::Characteristic_To_Primitive(cell_i, interface_R_imh_characteristic, eigenvectors, gamma);

  // Compute the interfaces for the variables that don't have characteristics
  #ifdef DE
  reconstruction::PPM_Single_Variable(cell_im2.gas_energy_specific, cell_im1.gas_energy_specific,
                                      cell_i.gas_energy_specific, cell_ip1.gas_energy_specific,
                                      cell_ip2.gas_energy_specific, interface_L_iph.gas_energy_specific,
                                      interface_R_imh.gas_energy_specific);
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    reconstruction::PPM_Single_Variable(cell_im2.scalar_specific[i], cell_im1.scalar_specific[i],
                                        cell_i.scalar_specific[i], cell_ip1.scalar_specific[i],
                                        cell_ip2.scalar_specific[i], interface_L_iph.scalar_specific[i],
                                        interface_R_imh.scalar_specific[i]);
  }
  #endif  // SCALAR
#else     // PPMC
  auto [interface_L_iph, interface_R_imh] =
      reconstruction::PPM_Interfaces(cell_im2, cell_im1, cell_i, cell_ip1, cell_ip2);
#endif    // PPMC

  // Do the characteristic tracing
#ifndef VL
  PPM_Characteristic_Evolution(cell_i, dt, dx, gamma, interface_R_imh, interface_L_iph);
#endif  // VL

  // enforce minimum values
  interface_R_imh.density  = fmax(interface_R_imh.density, (Real)TINY_NUMBER);
  interface_L_iph.density  = fmax(interface_L_iph.density, (Real)TINY_NUMBER);
  interface_R_imh.pressure = fmax(interface_R_imh.pressure, (Real)TINY_NUMBER);
  interface_L_iph.pressure = fmax(interface_L_iph.pressure, (Real)TINY_NUMBER);

  // Step 11 - Send final values back from kernel

  // Convert the left and right states in the primitive to the conserved variables send final values back from kernel
  // bounds_R refers to the right side of the i-1/2 interface
  size_t id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
  reconstruction::Write_Data(interface_L_iph, dev_bounds_L, dev_conserved, id, n_cells, o1, o2, o3, gamma);

  id = cuda_utilities::compute1DIndex(xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny);
  reconstruction::Write_Data(interface_R_imh, dev_bounds_R, dev_conserved, id, n_cells, o1, o2, o3, gamma);
}
// Instantiate the relevant template specifications
template __global__ __launch_bounds__(TPB) void PPM_cuda<0>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                            int nx, int ny, int nz, Real dx, Real dt, Real gamma);
template __global__ __launch_bounds__(TPB) void PPM_cuda<1>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                            int nx, int ny, int nz, Real dx, Real dt, Real gamma);
template __global__ __launch_bounds__(TPB) void PPM_cuda<2>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                            int nx, int ny, int nz, Real dx, Real dt, Real gamma);
// =====================================================================================================================

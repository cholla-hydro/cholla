/*! \file ppmc_cuda.cu
 *  \brief Functions definitions for the ppm kernels, using characteristic
 tracing. Written following Stone et al. 2008. */

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../reconstruction/reconstruction.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif

/*!
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using ppm. */
__global__ void PPMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                          Real dt, Real gamma, int dir)
{
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Thread guard to prevent overrun
  if (xid < 2 or xid >= nx - 3 or yid < 2 or yid >= ny - 3 or zid < 2 or zid >= nz - 3) {
    return;
  }

  // Compute the total number of cells
  int const n_cells = nx * ny * nz;

  // Set the field indices for the various directions
  int o1, o2, o3;
  switch (dir) {
    case 0:
      o1 = grid_enum::momentum_x;
      o2 = grid_enum::momentum_y;
      o3 = grid_enum::momentum_z;
      break;
    case 1:
      o1 = grid_enum::momentum_y;
      o2 = grid_enum::momentum_z;
      o3 = grid_enum::momentum_x;
      break;
    case 2:
      o1 = grid_enum::momentum_z;
      o2 = grid_enum::momentum_x;
      o3 = grid_enum::momentum_y;
      break;
  }

  // declare primitive variables for each stencil these will be placed into registers for each thread
  reconstruction::Primitive cell_i, cell_im1, cell_im2, cell_ip1, cell_ip2;

  // declare other variables to be used
  reconstruction::Primitive del_L, del_R, del_C, del_G;                        // primitive slopes
  reconstruction::Characteristic del_a_L, del_a_R, del_a_C, del_a_G, del_a_m;  // characteristic slopes
  Real del_d_m_imo, del_vx_m_imo, del_vy_m_imo, del_vz_m_imo, del_p_m_imo;
  Real del_d_m_i, del_vx_m_i, del_vy_m_i, del_vz_m_i, del_p_m_i;
  Real del_d_m_ipo, del_vx_m_ipo, del_vy_m_ipo, del_vz_m_ipo, del_p_m_ipo;
  reconstruction::Primitive interface_R_imh, interface_L_iph;  // Interface states

#ifdef DE
  Real del_ge_m_imo, del_ge_m_i, del_ge_m_ipo;
#endif  // DE

#ifdef SCALAR
  Real del_scalar_m_imo[NSCALARS], del_scalar_m_i[NSCALARS], del_scalar_m_ipo[NSCALARS];
#endif  // SCALAR

  // load the 5-cell stencil into registers
  // cell i
  int id            = xid + yid * nx + zid * nx * ny;
  cell_i.density    = dev_conserved[id];
  cell_i.velocity_x = dev_conserved[o1 * n_cells + id] / cell_i.density;
  cell_i.velocity_y = dev_conserved[o2 * n_cells + id] / cell_i.density;
  cell_i.velocity_z = dev_conserved[o3 * n_cells + id] / cell_i.density;
#ifdef DE  // PRESSURE_DE
  Real E     = dev_conserved[4 * n_cells + id];
  Real E_kin = 0.5 * cell_i.density *
               (cell_i.velocity_x * cell_i.velocity_x + cell_i.velocity_y * cell_i.velocity_y +
                cell_i.velocity_z * cell_i.velocity_z);
  Real gas_energy = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_i.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, gas_energy, gamma);
#else   // not DE
  cell_i.pressure = (dev_conserved[4 * n_cells + id] -
                     0.5 * cell_i.density *
                         (cell_i.velocity_x * cell_i.velocity_x + cell_i.velocity_y * cell_i.velocity_y +
                          cell_i.velocity_z * cell_i.velocity_z)) *
                    (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_i.pressure = fmax(cell_i.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_i.gas_energy = gas_energy / cell_i.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_i.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_i.density;
  }
#endif  // SCALAR
  // cell i-1
  switch (dir) {
    case 0:
      id = xid - 1 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid - 1) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid - 1) * nx * ny;
      break;
  }

  cell_im1.density    = dev_conserved[id];
  cell_im1.velocity_x = dev_conserved[o1 * n_cells + id] / cell_im1.density;
  cell_im1.velocity_y = dev_conserved[o2 * n_cells + id] / cell_im1.density;
  cell_im1.velocity_z = dev_conserved[o3 * n_cells + id] / cell_im1.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_im1.density *
          (cell_im1.velocity_x * cell_im1.velocity_x + cell_im1.velocity_y * cell_im1.velocity_y +
           cell_im1.velocity_z * cell_im1.velocity_z);
  gas_energy        = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_im1.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, gas_energy, gamma);
#else   // not DE
  cell_im1.pressure = (dev_conserved[4 * n_cells + id] -
                       0.5 * cell_im1.density *
                           (cell_im1.velocity_x * cell_im1.velocity_x + cell_im1.velocity_y * cell_im1.velocity_y +
                            cell_im1.velocity_z * cell_im1.velocity_z)) *
                      (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_im1.pressure = fmax(cell_im1.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_im1.gas_energy = gas_energy / cell_im1.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_im1.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_im1.density;
  }
#endif  // SCALAR
  // cell i+1
  switch (dir) {
    case 0:
      id = xid + 1 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid + 1) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid + 1) * nx * ny;
      break;
  }
  cell_ip1.density    = dev_conserved[id];
  cell_ip1.velocity_x = dev_conserved[o1 * n_cells + id] / cell_ip1.density;
  cell_ip1.velocity_y = dev_conserved[o2 * n_cells + id] / cell_ip1.density;
  cell_ip1.velocity_z = dev_conserved[o3 * n_cells + id] / cell_ip1.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_ip1.density *
          (cell_ip1.velocity_x * cell_ip1.velocity_x + cell_ip1.velocity_y * cell_ip1.velocity_y +
           cell_ip1.velocity_z * cell_ip1.velocity_z);
  gas_energy        = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_ip1.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, gas_energy, gamma);
#else   // not DE
  cell_ip1.pressure = (dev_conserved[4 * n_cells + id] -
                       0.5 * cell_ip1.density *
                           (cell_ip1.velocity_x * cell_ip1.velocity_x + cell_ip1.velocity_y * cell_ip1.velocity_y +
                            cell_ip1.velocity_z * cell_ip1.velocity_z)) *
                      (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_ip1.pressure = fmax(cell_ip1.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_ip1.gas_energy = gas_energy / cell_ip1.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_ip1.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_ip1.density;
  }
#endif  // SCALAR
  // cell i-2
  switch (dir) {
    case 0:
      id = xid - 2 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid - 2) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid - 2) * nx * ny;
      break;
  }
  cell_im2.density    = dev_conserved[id];
  cell_im2.velocity_x = dev_conserved[o1 * n_cells + id] / cell_im2.density;
  cell_im2.velocity_y = dev_conserved[o2 * n_cells + id] / cell_im2.density;
  cell_im2.velocity_z = dev_conserved[o3 * n_cells + id] / cell_im2.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_im2.density *
          (cell_im2.velocity_x * cell_im2.velocity_x + cell_im2.velocity_y * cell_im2.velocity_y +
           cell_im2.velocity_z * cell_im2.velocity_z);
  gas_energy        = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_im2.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, gas_energy, gamma);
#else   // not DE
  cell_im2.pressure = (dev_conserved[4 * n_cells + id] -
                       0.5 * cell_im2.density *
                           (cell_im2.velocity_x * cell_im2.velocity_x + cell_im2.velocity_y * cell_im2.velocity_y +
                            cell_im2.velocity_z * cell_im2.velocity_z)) *
                      (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_im2.pressure = fmax(cell_im2.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_im2.gas_energy = gas_energy / cell_im2.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_im2.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_im2.density;
  }
#endif  // SCALAR
  // cell i+2
  switch (dir) {
    case 0:
      id = xid + 2 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid + 2) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid + 2) * nx * ny;
      break;
  }
  cell_ip2.density    = dev_conserved[id];
  cell_ip2.velocity_x = dev_conserved[o1 * n_cells + id] / cell_ip2.density;
  cell_ip2.velocity_y = dev_conserved[o2 * n_cells + id] / cell_ip2.density;
  cell_ip2.velocity_z = dev_conserved[o3 * n_cells + id] / cell_ip2.density;
#ifdef DE  // PRESSURE_DE
  E     = dev_conserved[4 * n_cells + id];
  E_kin = 0.5 * cell_ip2.density *
          (cell_ip2.velocity_x * cell_ip2.velocity_x + cell_ip2.velocity_y * cell_ip2.velocity_y +
           cell_ip2.velocity_z * cell_ip2.velocity_z);
  gas_energy        = dev_conserved[grid_enum::GasEnergy * n_cells + id];
  cell_ip2.pressure = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, gas_energy, gamma);
#else   // not DE
  cell_ip2.pressure = (dev_conserved[4 * n_cells + id] -
                       0.5 * cell_ip2.density *
                           (cell_ip2.velocity_x * cell_ip2.velocity_x + cell_ip2.velocity_y * cell_ip2.velocity_y +
                            cell_ip2.velocity_z * cell_ip2.velocity_z)) *
                      (gamma - 1.0);
#endif  // PRESSURE_DE
  cell_ip2.pressure = fmax(cell_ip2.pressure, (Real)TINY_NUMBER);
#ifdef DE
  cell_ip2.gas_energy = gas_energy / cell_ip2.density;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    cell_ip2.scalar[i] = dev_conserved[(5 + i) * n_cells + id] / cell_ip2.density;
  }
#endif  // SCALAR

  // printf("%d %d %d %f %f %f %f %f\n", xid, yid, zid, cell_i.density, cell_i.velocity_x, cell_i.velocity_y,
  // cell_i.velocity_z, cell_i.pressure);

  // Steps 2 - 5 are repeated for cell i-1, i, and i+1
  // Step 2 - Compute the left, right, centered, and van Leer differences of
  // the primitive variables
  //          Note that here L and R refer to locations relative to the cell
  //          center Stone Eqn 36

  // calculate the adiabatic sound speed in cell imo
  Real sound_speed = hydro_utilities::Calc_Sound_Speed(cell_im1.pressure, cell_im1.density, gamma);

  // left
  del_L.density    = cell_im1.density - cell_im2.density;
  del_L.velocity_x = cell_im1.velocity_x - cell_im2.velocity_x;
  del_L.velocity_y = cell_im1.velocity_y - cell_im2.velocity_y;
  del_L.velocity_z = cell_im1.velocity_z - cell_im2.velocity_z;
  del_L.pressure   = cell_im1.pressure - cell_im2.pressure;

  // right
  del_R.density    = cell_i.density - cell_im1.density;
  del_R.velocity_x = cell_i.velocity_x - cell_im1.velocity_x;
  del_R.velocity_y = cell_i.velocity_y - cell_im1.velocity_y;
  del_R.velocity_z = cell_i.velocity_z - cell_im1.velocity_z;
  del_R.pressure   = cell_i.pressure - cell_im1.pressure;

  // centered
  del_C.density    = 0.5 * (cell_i.density - cell_im2.density);
  del_C.velocity_x = 0.5 * (cell_i.velocity_x - cell_im2.velocity_x);
  del_C.velocity_y = 0.5 * (cell_i.velocity_y - cell_im2.velocity_y);
  del_C.velocity_z = 0.5 * (cell_i.velocity_z - cell_im2.velocity_z);
  del_C.pressure   = 0.5 * (cell_i.pressure - cell_im2.pressure);

  // Van Leer
  if (del_L.density * del_R.density > 0.0) {
    del_G.density = 2.0 * del_L.density * del_R.density / (del_L.density + del_R.density);
  } else {
    del_G.density = 0.0;
  }
  if (del_L.velocity_x * del_R.velocity_x > 0.0) {
    del_G.velocity_x = 2.0 * del_L.velocity_x * del_R.velocity_x / (del_L.velocity_x + del_R.velocity_x);
  } else {
    del_G.velocity_x = 0.0;
  }
  if (del_L.velocity_y * del_R.velocity_y > 0.0) {
    del_G.velocity_y = 2.0 * del_L.velocity_y * del_R.velocity_y / (del_L.velocity_y + del_R.velocity_y);
  } else {
    del_G.velocity_y = 0.0;
  }
  if (del_L.velocity_z * del_R.velocity_z > 0.0) {
    del_G.velocity_z = 2.0 * del_L.velocity_z * del_R.velocity_z / (del_L.velocity_z + del_R.velocity_z);
  } else {
    del_G.velocity_z = 0.0;
  }
  if (del_L.pressure * del_R.pressure > 0.0) {
    del_G.pressure = 2.0 * del_L.pressure * del_R.pressure / (del_L.pressure + del_R.pressure);
  } else {
    del_G.pressure = 0.0;
  }

#ifdef DE
  del_L.gas_energy = cell_im1.gas_energy - cell_im2.gas_energy;
  del_R.gas_energy = cell_i.gas_energy - cell_im1.gas_energy;
  del_C.gas_energy = 0.5 * (cell_i.gas_energy - cell_im2.gas_energy);
  if (del_L.gas_energy * del_R.gas_energy > 0.0) {
    del_G.gas_energy = 2.0 * del_L.gas_energy * del_R.gas_energy / (del_L.gas_energy + del_R.gas_energy);
  } else {
    del_G.gas_energy = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_L.scalar[i] = cell_im1.scalar[i] - cell_im2.scalar[i];
    del_R.scalar[i] = cell_i.scalar[i] - cell_im1.scalar[i];
    del_C.scalar[i] = 0.5 * (cell_i.scalar[i] - cell_im2.scalar[i]);
    if (del_L.scalar[i] * del_R.scalar[i] > 0.0) {
      del_G.scalar[i] = 2.0 * del_L.scalar[i] * del_R.scalar[i] / (del_L.scalar[i] + del_R.scalar[i]);
    } else {
      del_G.scalar[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 3 - Project the left, right, centered and van Leer differences onto
  // the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables,
  //          see Stone for notation) Use the eigenvectors given in Stone
  //          2008, Appendix A

  del_a_L.a0 =
      -0.5 * cell_im1.density * del_L.velocity_x / sound_speed + 0.5 * del_L.pressure / (sound_speed * sound_speed);
  del_a_L.a1 = del_L.density - del_L.pressure / (sound_speed * sound_speed);
  del_a_L.a2 = del_L.velocity_y;
  del_a_L.a3 = del_L.velocity_z;
  del_a_L.a4 =
      0.5 * cell_im1.density * del_L.velocity_x / sound_speed + 0.5 * del_L.pressure / (sound_speed * sound_speed);

  del_a_R.a0 =
      -0.5 * cell_im1.density * del_R.velocity_x / sound_speed + 0.5 * del_R.pressure / (sound_speed * sound_speed);
  del_a_R.a1 = del_R.density - del_R.pressure / (sound_speed * sound_speed);
  del_a_R.a2 = del_R.velocity_y;
  del_a_R.a3 = del_R.velocity_z;
  del_a_R.a4 =
      0.5 * cell_im1.density * del_R.velocity_x / sound_speed + 0.5 * del_R.pressure / (sound_speed * sound_speed);

  del_a_C.a0 =
      -0.5 * cell_im1.density * del_C.velocity_x / sound_speed + 0.5 * del_C.pressure / (sound_speed * sound_speed);
  del_a_C.a1 = del_C.density - del_C.pressure / (sound_speed * sound_speed);
  del_a_C.a2 = del_C.velocity_y;
  del_a_C.a3 = del_C.velocity_z;
  del_a_C.a4 =
      0.5 * cell_im1.density * del_C.velocity_x / sound_speed + 0.5 * del_C.pressure / (sound_speed * sound_speed);

  del_a_G.a0 =
      -0.5 * cell_im1.density * del_G.velocity_x / sound_speed + 0.5 * del_G.pressure / (sound_speed * sound_speed);
  del_a_G.a1 = del_G.density - del_G.pressure / (sound_speed * sound_speed);
  del_a_G.a2 = del_G.velocity_y;
  del_a_G.a3 = del_G.velocity_z;
  del_a_G.a4 =
      0.5 * cell_im1.density * del_G.velocity_x / sound_speed + 0.5 * del_G.pressure / (sound_speed * sound_speed);

  // Step 4 - Apply monotonicity constraints to the differences in the
  // characteristic variables
  //          Stone Eqn 38

  del_a_m.a0 = del_a_m.a1 = del_a_m.a2 = del_a_m.a3 = del_a_m.a4 = 0.0;
  Real lim_slope_a, lim_slope_b;
  if (del_a_L.a0 * del_a_R.a0 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a0), fabs(del_a_R.a0));
    lim_slope_b = fmin(fabs(del_a_C.a0), fabs(del_a_G.a0));
    del_a_m.a0  = sgn_CUDA(del_a_C.a0) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a1 * del_a_R.a1 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a1), fabs(del_a_R.a1));
    lim_slope_b = fmin(fabs(del_a_C.a1), fabs(del_a_G.a1));
    del_a_m.a1  = sgn_CUDA(del_a_C.a1) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a2 * del_a_R.a2 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a2), fabs(del_a_R.a2));
    lim_slope_b = fmin(fabs(del_a_C.a2), fabs(del_a_G.a2));
    del_a_m.a2  = sgn_CUDA(del_a_C.a2) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a3 * del_a_R.a3 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a3), fabs(del_a_R.a3));
    lim_slope_b = fmin(fabs(del_a_C.a3), fabs(del_a_G.a3));
    del_a_m.a3  = sgn_CUDA(del_a_C.a3) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a4 * del_a_R.a4 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a4), fabs(del_a_R.a4));
    lim_slope_b = fmin(fabs(del_a_C.a4), fabs(del_a_G.a4));
    del_a_m.a4  = sgn_CUDA(del_a_C.a4) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
#ifdef DE
  if (del_L.gas_energy * del_R.gas_energy > 0.0) {
    lim_slope_a  = fmin(fabs(del_L.gas_energy), fabs(del_R.gas_energy));
    lim_slope_b  = fmin(fabs(del_C.gas_energy), fabs(del_G.gas_energy));
    del_ge_m_imo = sgn_CUDA(del_C.gas_energy) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  } else {
    del_ge_m_imo = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    if (del_L.scalar[i] * del_R.scalar[i] > 0.0) {
      lim_slope_a         = fmin(fabs(del_L.scalar[i]), fabs(del_R.scalar[i]));
      lim_slope_b         = fmin(fabs(del_C.scalar[i]), fabs(del_G.scalar[i]));
      del_scalar_m_imo[i] = sgn_CUDA(del_C.scalar[i]) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
    } else {
      del_scalar_m_imo[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 5 - Project the monotonized difference in the characteristic
  // variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_imo  = del_a_m.a0 + del_a_m.a1 + del_a_m.a4;
  del_vx_m_imo = -sound_speed * del_a_m.a0 / cell_im1.density + sound_speed * del_a_m.a4 / cell_im1.density;
  del_vy_m_imo = del_a_m.a2;
  del_vz_m_imo = del_a_m.a3;
  del_p_m_imo  = sound_speed * sound_speed * del_a_m.a0 + sound_speed * sound_speed * del_a_m.a4;

  // Step 2 - Compute the left, right, centered, and van Leer differences of
  // the primitive variables
  //          Note that here L and R refer to locations relative to the cell
  //          center Stone Eqn 36

  // calculate the adiabatic sound speed in cell i
  sound_speed = hydro_utilities::Calc_Sound_Speed(cell_i.pressure, cell_i.density, gamma);

  // left
  del_L.density    = cell_i.density - cell_im1.density;
  del_L.velocity_x = cell_i.velocity_x - cell_im1.velocity_x;
  del_L.velocity_y = cell_i.velocity_y - cell_im1.velocity_y;
  del_L.velocity_z = cell_i.velocity_z - cell_im1.velocity_z;
  del_L.pressure   = cell_i.pressure - cell_im1.pressure;

  // right
  del_R.density    = cell_ip1.density - cell_i.density;
  del_R.velocity_x = cell_ip1.velocity_x - cell_i.velocity_x;
  del_R.velocity_y = cell_ip1.velocity_y - cell_i.velocity_y;
  del_R.velocity_z = cell_ip1.velocity_z - cell_i.velocity_z;
  del_R.pressure   = cell_ip1.pressure - cell_i.pressure;

  // centered
  del_C.density    = 0.5 * (cell_ip1.density - cell_im1.density);
  del_C.velocity_x = 0.5 * (cell_ip1.velocity_x - cell_im1.velocity_x);
  del_C.velocity_y = 0.5 * (cell_ip1.velocity_y - cell_im1.velocity_y);
  del_C.velocity_z = 0.5 * (cell_ip1.velocity_z - cell_im1.velocity_z);
  del_C.pressure   = 0.5 * (cell_ip1.pressure - cell_im1.pressure);

  // van Leer
  if (del_L.density * del_R.density > 0.0) {
    del_G.density = 2.0 * del_L.density * del_R.density / (del_L.density + del_R.density);
  } else {
    del_G.density = 0.0;
  }
  if (del_L.velocity_x * del_R.velocity_x > 0.0) {
    del_G.velocity_x = 2.0 * del_L.velocity_x * del_R.velocity_x / (del_L.velocity_x + del_R.velocity_x);
  } else {
    del_G.velocity_x = 0.0;
  }
  if (del_L.velocity_y * del_R.velocity_y > 0.0) {
    del_G.velocity_y = 2.0 * del_L.velocity_y * del_R.velocity_y / (del_L.velocity_y + del_R.velocity_y);
  } else {
    del_G.velocity_y = 0.0;
  }
  if (del_L.velocity_z * del_R.velocity_z > 0.0) {
    del_G.velocity_z = 2.0 * del_L.velocity_z * del_R.velocity_z / (del_L.velocity_z + del_R.velocity_z);
  } else {
    del_G.velocity_z = 0.0;
  }
  if (del_L.pressure * del_R.pressure > 0.0) {
    del_G.pressure = 2.0 * del_L.pressure * del_R.pressure / (del_L.pressure + del_R.pressure);
  } else {
    del_G.pressure = 0.0;
  }

#ifdef DE
  del_L.gas_energy = cell_i.gas_energy - cell_im1.gas_energy;
  del_R.gas_energy = cell_ip1.gas_energy - cell_i.gas_energy;
  del_C.gas_energy = 0.5 * (cell_ip1.gas_energy - cell_im1.gas_energy);
  if (del_L.gas_energy * del_R.gas_energy > 0.0) {
    del_G.gas_energy = 2.0 * del_L.gas_energy * del_R.gas_energy / (del_L.gas_energy + del_R.gas_energy);
  } else {
    del_G.gas_energy = 0.0;
  }
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_L.scalar[i] = cell_i.scalar[i] - cell_im1.scalar[i];
    del_R.scalar[i] = cell_ip1.scalar[i] - cell_i.scalar[i];
    del_C.scalar[i] = 0.5 * (cell_ip1.scalar[i] - cell_im1.scalar[i]);
    if (del_L.scalar[i] * del_R.scalar[i] > 0.0) {
      del_G.scalar[i] = 2.0 * del_L.scalar[i] * del_R.scalar[i] / (del_L.scalar[i] + del_R.scalar[i]);
    } else {
      del_G.scalar[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 3 - Project the left, right, centered, and van Leer differences onto
  // the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables,
  //          see Stone for notation) Use the eigenvectors given in Stone
  //          2008, Appendix A

  del_a_L.a0 =
      -0.5 * cell_i.density * del_L.velocity_x / sound_speed + 0.5 * del_L.pressure / (sound_speed * sound_speed);
  del_a_L.a1 = del_L.density - del_L.pressure / (sound_speed * sound_speed);
  del_a_L.a2 = del_L.velocity_y;
  del_a_L.a3 = del_L.velocity_z;
  del_a_L.a4 =
      0.5 * cell_i.density * del_L.velocity_x / sound_speed + 0.5 * del_L.pressure / (sound_speed * sound_speed);

  del_a_R.a0 =
      -0.5 * cell_i.density * del_R.velocity_x / sound_speed + 0.5 * del_R.pressure / (sound_speed * sound_speed);
  del_a_R.a1 = del_R.density - del_R.pressure / (sound_speed * sound_speed);
  del_a_R.a2 = del_R.velocity_y;
  del_a_R.a3 = del_R.velocity_z;
  del_a_R.a4 =
      0.5 * cell_i.density * del_R.velocity_x / sound_speed + 0.5 * del_R.pressure / (sound_speed * sound_speed);

  del_a_C.a0 =
      -0.5 * cell_i.density * del_C.velocity_x / sound_speed + 0.5 * del_C.pressure / (sound_speed * sound_speed);
  del_a_C.a1 = del_C.density - del_C.pressure / (sound_speed * sound_speed);
  del_a_C.a2 = del_C.velocity_y;
  del_a_C.a3 = del_C.velocity_z;
  del_a_C.a4 =
      0.5 * cell_i.density * del_C.velocity_x / sound_speed + 0.5 * del_C.pressure / (sound_speed * sound_speed);

  del_a_G.a0 =
      -0.5 * cell_i.density * del_G.velocity_x / sound_speed + 0.5 * del_G.pressure / (sound_speed * sound_speed);
  del_a_G.a1 = del_G.density - del_G.pressure / (sound_speed * sound_speed);
  del_a_G.a2 = del_G.velocity_y;
  del_a_G.a3 = del_G.velocity_z;
  del_a_G.a4 =
      0.5 * cell_i.density * del_G.velocity_x / sound_speed + 0.5 * del_G.pressure / (sound_speed * sound_speed);

  // Step 4 - Apply monotonicity constraints to the differences in the
  // characteristic variables
  //          Stone Eqn 38

  del_a_m.a0 = del_a_m.a1 = del_a_m.a2 = del_a_m.a3 = del_a_m.a4 = 0.0;

  if (del_a_L.a0 * del_a_R.a0 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a0), fabs(del_a_R.a0));
    lim_slope_b = fmin(fabs(del_a_C.a0), fabs(del_a_G.a0));
    del_a_m.a0  = sgn_CUDA(del_a_C.a0) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a1 * del_a_R.a1 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a1), fabs(del_a_R.a1));
    lim_slope_b = fmin(fabs(del_a_C.a1), fabs(del_a_G.a1));
    del_a_m.a1  = sgn_CUDA(del_a_C.a1) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a2 * del_a_R.a2 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a2), fabs(del_a_R.a2));
    lim_slope_b = fmin(fabs(del_a_C.a2), fabs(del_a_G.a2));
    del_a_m.a2  = sgn_CUDA(del_a_C.a2) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a3 * del_a_R.a3 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a3), fabs(del_a_R.a3));
    lim_slope_b = fmin(fabs(del_a_C.a3), fabs(del_a_G.a3));
    del_a_m.a3  = sgn_CUDA(del_a_C.a3) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a4 * del_a_R.a4 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a4), fabs(del_a_R.a4));
    lim_slope_b = fmin(fabs(del_a_C.a4), fabs(del_a_G.a4));
    del_a_m.a4  = sgn_CUDA(del_a_C.a4) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
#ifdef DE
  if (del_L.gas_energy * del_R.gas_energy > 0.0) {
    lim_slope_a = fmin(fabs(del_L.gas_energy), fabs(del_R.gas_energy));
    lim_slope_b = fmin(fabs(del_C.gas_energy), fabs(del_G.gas_energy));
    del_ge_m_i  = sgn_CUDA(del_C.gas_energy) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  } else {
    del_ge_m_i = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    if (del_L.scalar[i] * del_R.scalar[i] > 0.0) {
      lim_slope_a       = fmin(fabs(del_L.scalar[i]), fabs(del_R.scalar[i]));
      lim_slope_b       = fmin(fabs(del_C.scalar[i]), fabs(del_G.scalar[i]));
      del_scalar_m_i[i] = sgn_CUDA(del_C.scalar[i]) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
    } else {
      del_scalar_m_i[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 5 - Project the monotonized difference in the characteristic
  // variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_i  = del_a_m.a0 + del_a_m.a1 + del_a_m.a4;
  del_vx_m_i = -sound_speed * del_a_m.a0 / cell_i.density + sound_speed * del_a_m.a4 / cell_i.density;
  del_vy_m_i = del_a_m.a2;
  del_vz_m_i = del_a_m.a3;
  del_p_m_i  = sound_speed * sound_speed * del_a_m.a0 + sound_speed * sound_speed * del_a_m.a4;

  // Step 2 - Compute the left, right, centered, and van Leer differences of
  // the primitive variables
  //          Note that here L and R refer to locations relative to the cell
  //          center Stone Eqn 36

  // calculate the adiabatic sound speed in cell ipo
  sound_speed = hydro_utilities::Calc_Sound_Speed(cell_ip1.pressure, cell_ip1.density, gamma);

  // left
  del_L.density    = cell_ip1.density - cell_i.density;
  del_L.velocity_x = cell_ip1.velocity_x - cell_i.velocity_x;
  del_L.velocity_y = cell_ip1.velocity_y - cell_i.velocity_y;
  del_L.velocity_z = cell_ip1.velocity_z - cell_i.velocity_z;
  del_L.pressure   = cell_ip1.pressure - cell_i.pressure;

  // right
  del_R.density    = cell_ip2.density - cell_ip1.density;
  del_R.velocity_x = cell_ip2.velocity_x - cell_ip1.velocity_x;
  del_R.velocity_y = cell_ip2.velocity_y - cell_ip1.velocity_y;
  del_R.velocity_z = cell_ip2.velocity_z - cell_ip1.velocity_z;
  del_R.pressure   = cell_ip2.pressure - cell_ip1.pressure;

  // centered
  del_C.density    = 0.5 * (cell_ip2.density - cell_i.density);
  del_C.velocity_x = 0.5 * (cell_ip2.velocity_x - cell_i.velocity_x);
  del_C.velocity_y = 0.5 * (cell_ip2.velocity_y - cell_i.velocity_y);
  del_C.velocity_z = 0.5 * (cell_ip2.velocity_z - cell_i.velocity_z);
  del_C.pressure   = 0.5 * (cell_ip2.pressure - cell_i.pressure);

  // van Leer
  if (del_L.density * del_R.density > 0.0) {
    del_G.density = 2.0 * del_L.density * del_R.density / (del_L.density + del_R.density);
  } else {
    del_G.density = 0.0;
  }
  if (del_L.velocity_x * del_R.velocity_x > 0.0) {
    del_G.velocity_x = 2.0 * del_L.velocity_x * del_R.velocity_x / (del_L.velocity_x + del_R.velocity_x);
  } else {
    del_G.velocity_x = 0.0;
  }
  if (del_L.velocity_y * del_R.velocity_y > 0.0) {
    del_G.velocity_y = 2.0 * del_L.velocity_y * del_R.velocity_y / (del_L.velocity_y + del_R.velocity_y);
  } else {
    del_G.velocity_y = 0.0;
  }
  if (del_L.velocity_z * del_R.velocity_z > 0.0) {
    del_G.velocity_z = 2.0 * del_L.velocity_z * del_R.velocity_z / (del_L.velocity_z + del_R.velocity_z);
  } else {
    del_G.velocity_z = 0.0;
  }
  if (del_L.pressure * del_R.pressure > 0.0) {
    del_G.pressure = 2.0 * del_L.pressure * del_R.pressure / (del_L.pressure + del_R.pressure);
  } else {
    del_G.pressure = 0.0;
  }

#ifdef DE
  del_L.gas_energy = cell_ip1.gas_energy - cell_i.gas_energy;
  del_R.gas_energy = cell_ip2.gas_energy - cell_ip1.gas_energy;
  del_C.gas_energy = 0.5 * (cell_ip2.gas_energy - cell_i.gas_energy);
  if (del_L.gas_energy * del_R.gas_energy > 0.0) {
    del_G.gas_energy = 2.0 * del_L.gas_energy * del_R.gas_energy / (del_L.gas_energy + del_R.gas_energy);
  } else {
    del_G.gas_energy = 0.0;
  }
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    del_L.scalar[i] = cell_ip1.scalar[i] - cell_i.scalar[i];
    del_R.scalar[i] = cell_ip2.scalar[i] - cell_ip1.scalar[i];
    del_C.scalar[i] = 0.5 * (cell_ip2.scalar[i] - cell_i.scalar[i]);
    if (del_L.scalar[i] * del_R.scalar[i] > 0.0) {
      del_G.scalar[i] = 2.0 * del_L.scalar[i] * del_R.scalar[i] / (del_L.scalar[i] + del_R.scalar[i]);
    } else {
      del_G.scalar[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 3 - Project the left, right, centered, and van Leer differences onto
  // the characteristic variables
  //          Stone Eqn 37 (del_a are differences in characteristic variables,
  //          see Stone for notation) Use the eigenvectors given in Stone
  //          2008, Appendix A

  del_a_L.a0 =
      -0.5 * cell_ip1.density * del_L.velocity_x / sound_speed + 0.5 * del_L.pressure / (sound_speed * sound_speed);
  del_a_L.a1 = del_L.density - del_L.pressure / (sound_speed * sound_speed);
  del_a_L.a2 = del_L.velocity_y;
  del_a_L.a3 = del_L.velocity_z;
  del_a_L.a4 =
      0.5 * cell_ip1.density * del_L.velocity_x / sound_speed + 0.5 * del_L.pressure / (sound_speed * sound_speed);

  del_a_R.a0 =
      -0.5 * cell_ip1.density * del_R.velocity_x / sound_speed + 0.5 * del_R.pressure / (sound_speed * sound_speed);
  del_a_R.a1 = del_R.density - del_R.pressure / (sound_speed * sound_speed);
  del_a_R.a2 = del_R.velocity_y;
  del_a_R.a3 = del_R.velocity_z;
  del_a_R.a4 =
      0.5 * cell_ip1.density * del_R.velocity_x / sound_speed + 0.5 * del_R.pressure / (sound_speed * sound_speed);

  del_a_C.a0 =
      -0.5 * cell_ip1.density * del_C.velocity_x / sound_speed + 0.5 * del_C.pressure / (sound_speed * sound_speed);
  del_a_C.a1 = del_C.density - del_C.pressure / (sound_speed * sound_speed);
  del_a_C.a2 = del_C.velocity_y;
  del_a_C.a3 = del_C.velocity_z;
  del_a_C.a4 =
      0.5 * cell_ip1.density * del_C.velocity_x / sound_speed + 0.5 * del_C.pressure / (sound_speed * sound_speed);

  del_a_G.a0 =
      -0.5 * cell_ip1.density * del_G.velocity_x / sound_speed + 0.5 * del_G.pressure / (sound_speed * sound_speed);
  del_a_G.a1 = del_G.density - del_G.pressure / (sound_speed * sound_speed);
  del_a_G.a2 = del_G.velocity_y;
  del_a_G.a3 = del_G.velocity_z;
  del_a_G.a4 =
      0.5 * cell_ip1.density * del_G.velocity_x / sound_speed + 0.5 * del_G.pressure / (sound_speed * sound_speed);

  // Step 4 - Apply monotonicity constraints to the differences in the
  // characteristic variables
  //          Stone Eqn 38

  del_a_m.a0 = del_a_m.a1 = del_a_m.a2 = del_a_m.a3 = del_a_m.a4 = 0.0;

  if (del_a_L.a0 * del_a_R.a0 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a0), fabs(del_a_R.a0));
    lim_slope_b = fmin(fabs(del_a_C.a0), fabs(del_a_G.a0));
    del_a_m.a0  = sgn_CUDA(del_a_C.a0) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a1 * del_a_R.a1 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a1), fabs(del_a_R.a1));
    lim_slope_b = fmin(fabs(del_a_C.a1), fabs(del_a_G.a1));
    del_a_m.a1  = sgn_CUDA(del_a_C.a1) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a2 * del_a_R.a2 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a2), fabs(del_a_R.a2));
    lim_slope_b = fmin(fabs(del_a_C.a2), fabs(del_a_G.a2));
    del_a_m.a2  = sgn_CUDA(del_a_C.a2) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a3 * del_a_R.a3 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a3), fabs(del_a_R.a3));
    lim_slope_b = fmin(fabs(del_a_C.a3), fabs(del_a_G.a3));
    del_a_m.a3  = sgn_CUDA(del_a_C.a3) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
  if (del_a_L.a4 * del_a_R.a4 > 0.0) {
    lim_slope_a = fmin(fabs(del_a_L.a4), fabs(del_a_R.a4));
    lim_slope_b = fmin(fabs(del_a_C.a4), fabs(del_a_G.a4));
    del_a_m.a4  = sgn_CUDA(del_a_C.a4) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  }
#ifdef DE
  if (del_L.gas_energy * del_R.gas_energy > 0.0) {
    lim_slope_a  = fmin(fabs(del_L.gas_energy), fabs(del_R.gas_energy));
    lim_slope_b  = fmin(fabs(del_C.gas_energy), fabs(del_G.gas_energy));
    del_ge_m_ipo = sgn_CUDA(del_C.gas_energy) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
  } else {
    del_ge_m_ipo = 0.0;
  }
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    if (del_L.scalar[i] * del_R.scalar[i] > 0.0) {
      lim_slope_a         = fmin(fabs(del_L.scalar[i]), fabs(del_R.scalar[i]));
      lim_slope_b         = fmin(fabs(del_C.scalar[i]), fabs(del_G.scalar[i]));
      del_scalar_m_ipo[i] = sgn_CUDA(del_C.scalar[i]) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
    } else {
      del_scalar_m_ipo[i] = 0.0;
    }
  }
#endif  // SCALAR

  // Step 5 - Project the monotonized difference in the characteristic
  // variables back onto the
  //          primitive variables
  //          Stone Eqn 39

  del_d_m_ipo  = del_a_m.a0 + del_a_m.a1 + del_a_m.a4;
  del_vx_m_ipo = -sound_speed * del_a_m.a0 / cell_ip1.density + sound_speed * del_a_m.a4 / cell_ip1.density;
  del_vy_m_ipo = del_a_m.a2;
  del_vz_m_ipo = del_a_m.a3;
  del_p_m_ipo  = sound_speed * sound_speed * del_a_m.a0 + sound_speed * sound_speed * del_a_m.a4;

  // Step 6 - Use parabolic interpolation to compute values at the left and
  // right of each cell center
  //          Here, the subscripts L and R refer to the left and right side of
  //          the ith cell center Stone Eqn 46

  interface_R_imh.density    = 0.5 * (cell_i.density + cell_im1.density) - (del_d_m_i - del_d_m_imo) / 6.0;
  interface_R_imh.velocity_x = 0.5 * (cell_i.velocity_x + cell_im1.velocity_x) - (del_vx_m_i - del_vx_m_imo) / 6.0;
  interface_R_imh.velocity_y = 0.5 * (cell_i.velocity_y + cell_im1.velocity_y) - (del_vy_m_i - del_vy_m_imo) / 6.0;
  interface_R_imh.velocity_z = 0.5 * (cell_i.velocity_z + cell_im1.velocity_z) - (del_vz_m_i - del_vz_m_imo) / 6.0;
  interface_R_imh.pressure   = 0.5 * (cell_i.pressure + cell_im1.pressure) - (del_p_m_i - del_p_m_imo) / 6.0;

  interface_L_iph.density    = 0.5 * (cell_ip1.density + cell_i.density) - (del_d_m_ipo - del_d_m_i) / 6.0;
  interface_L_iph.velocity_x = 0.5 * (cell_ip1.velocity_x + cell_i.velocity_x) - (del_vx_m_ipo - del_vx_m_i) / 6.0;
  interface_L_iph.velocity_y = 0.5 * (cell_ip1.velocity_y + cell_i.velocity_y) - (del_vy_m_ipo - del_vy_m_i) / 6.0;
  interface_L_iph.velocity_z = 0.5 * (cell_ip1.velocity_z + cell_i.velocity_z) - (del_vz_m_ipo - del_vz_m_i) / 6.0;
  interface_L_iph.pressure   = 0.5 * (cell_ip1.pressure + cell_i.pressure) - (del_p_m_ipo - del_p_m_i) / 6.0;

#ifdef DE
  interface_R_imh.gas_energy = 0.5 * (cell_i.gas_energy + cell_im1.gas_energy) - (del_ge_m_i - del_ge_m_imo) / 6.0;
  interface_L_iph.gas_energy = 0.5 * (cell_ip1.gas_energy + cell_i.gas_energy) - (del_ge_m_ipo - del_ge_m_i) / 6.0;
#endif  // DE
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar[i] =
        0.5 * (cell_i.scalar[i] + cell_im1.scalar[i]) - (del_scalar_m_i[i] - del_scalar_m_imo[i]) / 6.0;
    interface_L_iph.scalar[i] =
        0.5 * (cell_ip1.scalar[i] + cell_i.scalar[i]) - (del_scalar_m_ipo[i] - del_scalar_m_i[i]) / 6.0;
  }
#endif  // SCALAR

  // Step 7 - Apply further monotonicity constraints to ensure the values on
  // the left and right side
  //          of cell center lie between neighboring cell-centered values
  //          Stone Eqns 47 - 53

  if ((interface_L_iph.density - cell_i.density) * (cell_i.density - interface_R_imh.density) <= 0) {
    interface_R_imh.density = interface_L_iph.density = cell_i.density;
  }
  if ((interface_L_iph.velocity_x - cell_i.velocity_x) * (cell_i.velocity_x - interface_R_imh.velocity_x) <= 0) {
    interface_R_imh.velocity_x = interface_L_iph.velocity_x = cell_i.velocity_x;
  }
  if ((interface_L_iph.velocity_y - cell_i.velocity_y) * (cell_i.velocity_y - interface_R_imh.velocity_y) <= 0) {
    interface_R_imh.velocity_y = interface_L_iph.velocity_y = cell_i.velocity_y;
  }
  if ((interface_L_iph.velocity_z - cell_i.velocity_z) * (cell_i.velocity_z - interface_R_imh.velocity_z) <= 0) {
    interface_R_imh.velocity_z = interface_L_iph.velocity_z = cell_i.velocity_z;
  }
  if ((interface_L_iph.pressure - cell_i.pressure) * (cell_i.pressure - interface_R_imh.pressure) <= 0) {
    interface_R_imh.pressure = interface_L_iph.pressure = cell_i.pressure;
  }

  if (6.0 * (interface_L_iph.density - interface_R_imh.density) *
          (cell_i.density - 0.5 * (interface_R_imh.density + interface_L_iph.density)) >
      (interface_L_iph.density - interface_R_imh.density) * (interface_L_iph.density - interface_R_imh.density)) {
    interface_R_imh.density = 3.0 * cell_i.density - 2.0 * interface_L_iph.density;
  }
  if (6.0 * (interface_L_iph.velocity_x - interface_R_imh.velocity_x) *
          (cell_i.velocity_x - 0.5 * (interface_R_imh.velocity_x + interface_L_iph.velocity_x)) >
      (interface_L_iph.velocity_x - interface_R_imh.velocity_x) *
          (interface_L_iph.velocity_x - interface_R_imh.velocity_x)) {
    interface_R_imh.velocity_x = 3.0 * cell_i.velocity_x - 2.0 * interface_L_iph.velocity_x;
  }
  if (6.0 * (interface_L_iph.velocity_y - interface_R_imh.velocity_y) *
          (cell_i.velocity_y - 0.5 * (interface_R_imh.velocity_y + interface_L_iph.velocity_y)) >
      (interface_L_iph.velocity_y - interface_R_imh.velocity_y) *
          (interface_L_iph.velocity_y - interface_R_imh.velocity_y)) {
    interface_R_imh.velocity_y = 3.0 * cell_i.velocity_y - 2.0 * interface_L_iph.velocity_y;
  }
  if (6.0 * (interface_L_iph.velocity_z - interface_R_imh.velocity_z) *
          (cell_i.velocity_z - 0.5 * (interface_R_imh.velocity_z + interface_L_iph.velocity_z)) >
      (interface_L_iph.velocity_z - interface_R_imh.velocity_z) *
          (interface_L_iph.velocity_z - interface_R_imh.velocity_z)) {
    interface_R_imh.velocity_z = 3.0 * cell_i.velocity_z - 2.0 * interface_L_iph.velocity_z;
  }
  if (6.0 * (interface_L_iph.pressure - interface_R_imh.pressure) *
          (cell_i.pressure - 0.5 * (interface_R_imh.pressure + interface_L_iph.pressure)) >
      (interface_L_iph.pressure - interface_R_imh.pressure) * (interface_L_iph.pressure - interface_R_imh.pressure)) {
    interface_R_imh.pressure = 3.0 * cell_i.pressure - 2.0 * interface_L_iph.pressure;
  }

  if (6.0 * (interface_L_iph.density - interface_R_imh.density) *
          (cell_i.density - 0.5 * (interface_R_imh.density + interface_L_iph.density)) <
      -(interface_L_iph.density - interface_R_imh.density) * (interface_L_iph.density - interface_R_imh.density)) {
    interface_L_iph.density = 3.0 * cell_i.density - 2.0 * interface_R_imh.density;
  }
  if (6.0 * (interface_L_iph.velocity_x - interface_R_imh.velocity_x) *
          (cell_i.velocity_x - 0.5 * (interface_R_imh.velocity_x + interface_L_iph.velocity_x)) <
      -(interface_L_iph.velocity_x - interface_R_imh.velocity_x) *
          (interface_L_iph.velocity_x - interface_R_imh.velocity_x)) {
    interface_L_iph.velocity_x = 3.0 * cell_i.velocity_x - 2.0 * interface_R_imh.velocity_x;
  }
  if (6.0 * (interface_L_iph.velocity_y - interface_R_imh.velocity_y) *
          (cell_i.velocity_y - 0.5 * (interface_R_imh.velocity_y + interface_L_iph.velocity_y)) <
      -(interface_L_iph.velocity_y - interface_R_imh.velocity_y) *
          (interface_L_iph.velocity_y - interface_R_imh.velocity_y)) {
    interface_L_iph.velocity_y = 3.0 * cell_i.velocity_y - 2.0 * interface_R_imh.velocity_y;
  }
  if (6.0 * (interface_L_iph.velocity_z - interface_R_imh.velocity_z) *
          (cell_i.velocity_z - 0.5 * (interface_R_imh.velocity_z + interface_L_iph.velocity_z)) <
      -(interface_L_iph.velocity_z - interface_R_imh.velocity_z) *
          (interface_L_iph.velocity_z - interface_R_imh.velocity_z)) {
    interface_L_iph.velocity_z = 3.0 * cell_i.velocity_z - 2.0 * interface_R_imh.velocity_z;
  }
  if (6.0 * (interface_L_iph.pressure - interface_R_imh.pressure) *
          (cell_i.pressure - 0.5 * (interface_R_imh.pressure + interface_L_iph.pressure)) <
      -(interface_L_iph.pressure - interface_R_imh.pressure) * (interface_L_iph.pressure - interface_R_imh.pressure)) {
    interface_L_iph.pressure = 3.0 * cell_i.pressure - 2.0 * interface_R_imh.pressure;
  }

  interface_R_imh.density    = fmax(fmin(cell_i.density, cell_im1.density), interface_R_imh.density);
  interface_R_imh.density    = fmin(fmax(cell_i.density, cell_im1.density), interface_R_imh.density);
  interface_L_iph.density    = fmax(fmin(cell_i.density, cell_ip1.density), interface_L_iph.density);
  interface_L_iph.density    = fmin(fmax(cell_i.density, cell_ip1.density), interface_L_iph.density);
  interface_R_imh.velocity_x = fmax(fmin(cell_i.velocity_x, cell_im1.velocity_x), interface_R_imh.velocity_x);
  interface_R_imh.velocity_x = fmin(fmax(cell_i.velocity_x, cell_im1.velocity_x), interface_R_imh.velocity_x);
  interface_L_iph.velocity_x = fmax(fmin(cell_i.velocity_x, cell_ip1.velocity_x), interface_L_iph.velocity_x);
  interface_L_iph.velocity_x = fmin(fmax(cell_i.velocity_x, cell_ip1.velocity_x), interface_L_iph.velocity_x);
  interface_R_imh.velocity_y = fmax(fmin(cell_i.velocity_y, cell_im1.velocity_y), interface_R_imh.velocity_y);
  interface_R_imh.velocity_y = fmin(fmax(cell_i.velocity_y, cell_im1.velocity_y), interface_R_imh.velocity_y);
  interface_L_iph.velocity_y = fmax(fmin(cell_i.velocity_y, cell_ip1.velocity_y), interface_L_iph.velocity_y);
  interface_L_iph.velocity_y = fmin(fmax(cell_i.velocity_y, cell_ip1.velocity_y), interface_L_iph.velocity_y);
  interface_R_imh.velocity_z = fmax(fmin(cell_i.velocity_z, cell_im1.velocity_z), interface_R_imh.velocity_z);
  interface_R_imh.velocity_z = fmin(fmax(cell_i.velocity_z, cell_im1.velocity_z), interface_R_imh.velocity_z);
  interface_L_iph.velocity_z = fmax(fmin(cell_i.velocity_z, cell_ip1.velocity_z), interface_L_iph.velocity_z);
  interface_L_iph.velocity_z = fmin(fmax(cell_i.velocity_z, cell_ip1.velocity_z), interface_L_iph.velocity_z);
  interface_R_imh.pressure   = fmax(fmin(cell_i.pressure, cell_im1.pressure), interface_R_imh.pressure);
  interface_R_imh.pressure   = fmin(fmax(cell_i.pressure, cell_im1.pressure), interface_R_imh.pressure);
  interface_L_iph.pressure   = fmax(fmin(cell_i.pressure, cell_ip1.pressure), interface_L_iph.pressure);
  interface_L_iph.pressure   = fmin(fmax(cell_i.pressure, cell_ip1.pressure), interface_L_iph.pressure);

#ifdef DE
  if ((interface_L_iph.gas_energy - cell_i.gas_energy) * (cell_i.gas_energy - interface_R_imh.gas_energy) <= 0) {
    interface_R_imh.gas_energy = interface_L_iph.gas_energy = cell_i.gas_energy;
  }
  if (6.0 * (interface_L_iph.gas_energy - interface_R_imh.gas_energy) *
          (cell_i.gas_energy - 0.5 * (interface_R_imh.gas_energy + interface_L_iph.gas_energy)) >
      (interface_L_iph.gas_energy - interface_R_imh.gas_energy) *
          (interface_L_iph.gas_energy - interface_R_imh.gas_energy)) {
    interface_R_imh.gas_energy = 3.0 * cell_i.gas_energy - 2.0 * interface_L_iph.gas_energy;
  }
  if (6.0 * (interface_L_iph.gas_energy - interface_R_imh.gas_energy) *
          (cell_i.gas_energy - 0.5 * (interface_R_imh.gas_energy + interface_L_iph.gas_energy)) <
      -(interface_L_iph.gas_energy - interface_R_imh.gas_energy) *
          (interface_L_iph.gas_energy - interface_R_imh.gas_energy)) {
    interface_L_iph.gas_energy = 3.0 * cell_i.gas_energy - 2.0 * interface_R_imh.gas_energy;
  }
  interface_R_imh.gas_energy = fmax(fmin(cell_i.gas_energy, cell_im1.gas_energy), interface_R_imh.gas_energy);
  interface_R_imh.gas_energy = fmin(fmax(cell_i.gas_energy, cell_im1.gas_energy), interface_R_imh.gas_energy);
  interface_L_iph.gas_energy = fmax(fmin(cell_i.gas_energy, cell_ip1.gas_energy), interface_L_iph.gas_energy);
  interface_L_iph.gas_energy = fmin(fmax(cell_i.gas_energy, cell_ip1.gas_energy), interface_L_iph.gas_energy);
#endif  // DE

#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    if ((interface_L_iph.scalar[i] - cell_i.scalar[i]) * (cell_i.scalar[i] - interface_R_imh.scalar[i]) <= 0) {
      interface_R_imh.scalar[i] = interface_L_iph.scalar[i] = cell_i.scalar[i];
    }
    if (6.0 * (interface_L_iph.scalar[i] - interface_R_imh.scalar[i]) *
            (cell_i.scalar[i] - 0.5 * (interface_R_imh.scalar[i] + interface_L_iph.scalar[i])) >
        (interface_L_iph.scalar[i] - interface_R_imh.scalar[i]) *
            (interface_L_iph.scalar[i] - interface_R_imh.scalar[i])) {
      interface_R_imh.scalar[i] = 3.0 * cell_i.scalar[i] - 2.0 * interface_L_iph.scalar[i];
    }
    if (6.0 * (interface_L_iph.scalar[i] - interface_R_imh.scalar[i]) *
            (cell_i.scalar[i] - 0.5 * (interface_R_imh.scalar[i] + interface_L_iph.scalar[i])) <
        -(interface_L_iph.scalar[i] - interface_R_imh.scalar[i]) *
            (interface_L_iph.scalar[i] - interface_R_imh.scalar[i])) {
      interface_L_iph.scalar[i] = 3.0 * cell_i.scalar[i] - 2.0 * interface_R_imh.scalar[i];
    }
    interface_R_imh.scalar[i] = fmax(fmin(cell_i.scalar[i], cell_im1.scalar[i]), interface_R_imh.scalar[i]);
    interface_R_imh.scalar[i] = fmin(fmax(cell_i.scalar[i], cell_im1.scalar[i]), interface_R_imh.scalar[i]);
    interface_L_iph.scalar[i] = fmax(fmin(cell_i.scalar[i], cell_ip1.scalar[i]), interface_L_iph.scalar[i]);
    interface_L_iph.scalar[i] = fmin(fmax(cell_i.scalar[i], cell_ip1.scalar[i]), interface_L_iph.scalar[i]);
  }
#endif  // SCALAR

#ifndef VL
  // Step 8 - Compute the coefficients for the monotonized parabolic
  // interpolation function
  //          Stone Eqn 54

  del_d_m_i  = interface_L_iph.density - interface_R_imh.density;
  del_vx_m_i = interface_L_iph.velocity_x - interface_R_imh.velocity_x;
  del_vy_m_i = interface_L_iph.velocity_y - interface_R_imh.velocity_y;
  del_vz_m_i = interface_L_iph.velocity_z - interface_R_imh.velocity_z;
  del_p_m_i  = interface_L_iph.pressure - interface_R_imh.pressure;

  Real const d_6  = 6.0 * (cell_i.density - 0.5 * (interface_R_imh.density + interface_L_iph.density));
  Real const vx_6 = 6.0 * (cell_i.velocity_x - 0.5 * (interface_R_imh.velocity_x + interface_L_iph.velocity_x));
  Real const vy_6 = 6.0 * (cell_i.velocity_y - 0.5 * (interface_R_imh.velocity_y + interface_L_iph.velocity_y));
  Real const vz_6 = 6.0 * (cell_i.velocity_z - 0.5 * (interface_R_imh.velocity_z + interface_L_iph.velocity_z));
  Real const p_6  = 6.0 * (cell_i.pressure - 0.5 * (interface_R_imh.pressure + interface_L_iph.pressure));

  #ifdef DE
  del_ge_m_i      = interface_L_iph.gas_energy - interface_R_imh.gas_energy;
  Real const ge_6 = 6.0 * (cell_i.gas_energy - 0.5 * (interface_R_imh.gas_energy + interface_L_iph.gas_energy));
  #endif  // DE

  #ifdef SCALAR
  Real scalar_6[NSCALARS] : for (int i = 0; i < NSCALARS; i++)
  {
    del_scalar_m_i[i] = interface_L_iph.scalar[i] - interface_R_imh.scalar[i];
    scalar_6[i]       = 6.0 * (cell_i.scalar[i] - 0.5 * (interface_R_imh.scalar[i] + interface_L_iph.scalar[i]));
  }
  #endif  // SCALAR

  // Compute the eigenvalues of the linearized equations in the
  // primitive variables using the cell-centered primitive variables

  // recalculate the adiabatic sound speed in cell i
  sound_speed = hydro_utilities::Calc_Sound_Speed(cell_i.pressure, cell_i.density, gamma);

  Real const lambda_m = cell_i.velocity_x - sound_speed;
  Real const lambda_0 = cell_i.velocity_x;
  Real const lambda_p = cell_i.velocity_x + sound_speed;

  // Step 9 - Compute the left and right interface values using monotonized
  // parabolic interpolation
  //          Stone Eqns 55 & 56

  // largest eigenvalue
  Real const lambda_max = fmax(lambda_p, (Real)0);
  // smallest eigenvalue
  Real const lambda_min = fmin(lambda_m, (Real)0);

  // left interface value, i+1/2
  Real const dtodx        = dt / dx;
  interface_L_iph.density = interface_L_iph.density -
                            lambda_max * (0.5 * dtodx) * (del_d_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * d_6);
  interface_L_iph.velocity_x =
      interface_L_iph.velocity_x -
      lambda_max * (0.5 * dtodx) * (del_vx_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vx_6);
  interface_L_iph.velocity_y =
      interface_L_iph.velocity_y -
      lambda_max * (0.5 * dtodx) * (del_vy_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vy_6);
  interface_L_iph.velocity_z =
      interface_L_iph.velocity_z -
      lambda_max * (0.5 * dtodx) * (del_vz_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * vz_6);
  interface_L_iph.pressure = interface_L_iph.pressure -
                             lambda_max * (0.5 * dtodx) * (del_p_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * p_6);

  // right interface value, i-1/2
  interface_R_imh.density = interface_R_imh.density -
                            lambda_min * (0.5 * dtodx) * (del_d_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * d_6);
  interface_R_imh.velocity_x =
      interface_R_imh.velocity_x -
      lambda_min * (0.5 * dtodx) * (del_vx_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vx_6);
  interface_R_imh.velocity_y =
      interface_R_imh.velocity_y -
      lambda_min * (0.5 * dtodx) * (del_vy_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vy_6);
  interface_R_imh.velocity_z =
      interface_R_imh.velocity_z -
      lambda_min * (0.5 * dtodx) * (del_vz_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * vz_6);
  interface_R_imh.pressure = interface_R_imh.pressure -
                             lambda_min * (0.5 * dtodx) * (del_p_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * p_6);

  #ifdef DE
  interface_L_iph.gas_energy =
      interface_L_iph.gas_energy -
      lambda_max * (0.5 * dtodx) * (del_ge_m_i - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * ge_6);
  interface_R_imh.gas_energy =
      interface_R_imh.gas_energy -
      lambda_min * (0.5 * dtodx) * (del_ge_m_i + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * ge_6);
  #endif  // DE

  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar[i] =
        interface_L_iph.scalar[i] -
        lambda_max * (0.5 * dtodx) * (del_scalar_m_i[i] - (1.0 - (2.0 / 3.0) * lambda_max * dtodx) * scalar_6[i]);
    interface_R_imh.scalar[i] =
        interface_R_imh.scalar[i] -
        lambda_min * (0.5 * dtodx) * (del_scalar_m_i[i] + (1.0 + (2.0 / 3.0) * lambda_min * dtodx) * scalar_6[i]);
  }
  #endif  // SCALAR

  // Step 10 - Perform the characteristic tracing
  //           Stone Eqns 57 - 60

  // left-hand interface value, i+1/2
  Real sum_1 = 0, sum_2 = 0, sum_3 = 0, sum_4 = 0, sum_5 = 0;
  #ifdef DE
  Real sum_ge = 0;
  #endif  // DE
  #ifdef SCALAR
  Real sum_scalar[NSCALARS];
  for (int i = 0; i < NSCALARS; i++) {
    sum_scalar[i] = 0;
  }
  #endif  // SCALAR

  if (lambda_m >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_m);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_m * lambda_m);

    Real const chi_1 = A * (del_d_m_i - d_6) + B * d_6;
    Real const chi_2 = A * (del_vx_m_i - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_vy_m_i - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_vz_m_i - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_p_m_i - p_6) + B * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / sound_speed - chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 - chi_5 / (sound_speed * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * sound_speed - chi_5);
  }
  if (lambda_0 >= 0) {
    Real const A = (0.5 * dtodx) * (lambda_p - lambda_0);
    Real const B = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_p * lambda_p - lambda_0 * lambda_0);

    Real const chi_1 = A * (del_d_m_i - d_6) + B * d_6;
    Real const chi_2 = A * (del_vx_m_i - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_vy_m_i - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_vz_m_i - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_p_m_i - p_6) + B * p_6;
  #ifdef DE
    Real chi_ge = A * (del_ge_m_i - ge_6) + B * ge_6;
  #endif  // DE
  #ifdef SCALAR
    Real chi_scalar[NSCALARS];
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = A * (del_scalar_m_i[i] - scalar_6[i]) + B * scalar_6[i];
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

    Real const chi_1 = A * (del_d_m_i - d_6) + B * d_6;
    Real const chi_2 = A * (del_vx_m_i - vx_6) + B * vx_6;
    Real const chi_3 = A * (del_vy_m_i - vy_6) + B * vy_6;
    Real const chi_4 = A * (del_vz_m_i - vz_6) + B * vz_6;
    Real const chi_5 = A * (del_p_m_i - p_6) + B * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / sound_speed + chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 + chi_5 / (sound_speed * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * sound_speed + chi_5);
  }

  // add the corrections to the initial guesses for the interface values
  interface_L_iph.density += sum_1;
  interface_L_iph.velocity_x += sum_2;
  interface_L_iph.velocity_y += sum_3;
  interface_L_iph.velocity_z += sum_4;
  interface_L_iph.pressure += sum_5;
  #ifdef DE
  interface_L_iph.gas_energy += sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_L_iph.scalar[i] += sum_scalar[i];
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
  for (int i = 0; i < NSCALARS; i++) {
    sum_scalar[i] = 0;
  }
  #endif  // SCALAR
  if (lambda_m <= 0) {
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_m);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_m * lambda_m);

    Real const chi_1 = C * (del_d_m_i + d_6) + D * d_6;
    Real const chi_2 = C * (del_vx_m_i + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_vy_m_i + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_vz_m_i + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_p_m_i + p_6) + D * p_6;

    sum_1 += -0.5 * (cell_i.density * chi_2 / sound_speed - chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 - chi_5 / (sound_speed * cell_i.density));
    sum_5 += -0.5 * (cell_i.density * chi_2 * sound_speed - chi_5);
  }
  if (lambda_0 <= 0) {
    Real const C = (0.5 * dtodx) * (lambda_m - lambda_0);
    Real const D = (1.0 / 3.0) * (dtodx) * (dtodx) * (lambda_m * lambda_m - lambda_0 * lambda_0);

    Real const chi_1 = C * (del_d_m_i + d_6) + D * d_6;
    Real const chi_2 = C * (del_vx_m_i + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_vy_m_i + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_vz_m_i + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_p_m_i + p_6) + D * p_6;
  #ifdef DE
    chi_ge = C * (del_ge_m_i + ge_6) + D * ge_6;
  #endif  // DE
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      chi_scalar[i] = C * (del_scalar_m_i[i] + scalar_6[i]) + D * scalar_6[i];
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

    Real const chi_1 = C * (del_d_m_i + d_6) + D * d_6;
    Real const chi_2 = C * (del_vx_m_i + vx_6) + D * vx_6;
    Real const chi_3 = C * (del_vy_m_i + vy_6) + D * vy_6;
    Real const chi_4 = C * (del_vz_m_i + vz_6) + D * vz_6;
    Real const chi_5 = C * (del_p_m_i + p_6) + D * p_6;

    sum_1 += 0.5 * (cell_i.density * chi_2 / sound_speed + chi_5 / (sound_speed * sound_speed));
    sum_2 += 0.5 * (chi_2 + chi_5 / (sound_speed * cell_i.density));
    sum_5 += 0.5 * (cell_i.density * chi_2 * sound_speed + chi_5);
  }

  // add the corrections
  interface_R_imh.density += sum_1;
  interface_R_imh.velocity_x += sum_2;
  interface_R_imh.velocity_y += sum_3;
  interface_R_imh.velocity_z += sum_4;
  interface_R_imh.pressure += sum_5;
  #ifdef DE
  interface_R_imh.gas_energy += sum_ge;
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    interface_R_imh.scalar[i] += sum_scalar[i];
  }
  #endif  // SCALAR

#endif  // not VL, i.e. CTU or SIMPLE was used for this section

  // enforce minimum values
  interface_R_imh.density  = fmax(interface_R_imh.density, (Real)TINY_NUMBER);
  interface_L_iph.density  = fmax(interface_L_iph.density, (Real)TINY_NUMBER);
  interface_R_imh.pressure = fmax(interface_R_imh.pressure, (Real)TINY_NUMBER);
  interface_L_iph.pressure = fmax(interface_L_iph.pressure, (Real)TINY_NUMBER);

  // Step 11 - Send final values back from kernel

  // bounds_R refers to the right side of the i-1/2 interface
  switch (dir) {
    case 0:
      id = xid - 1 + yid * nx + zid * nx * ny;
      break;
    case 1:
      id = xid + (yid - 1) * nx + zid * nx * ny;
      break;
    case 2:
      id = xid + yid * nx + (zid - 1) * nx * ny;
      break;
  }
  dev_bounds_R[id]                = interface_R_imh.density;
  dev_bounds_R[o1 * n_cells + id] = interface_R_imh.density * interface_R_imh.velocity_x;
  dev_bounds_R[o2 * n_cells + id] = interface_R_imh.density * interface_R_imh.velocity_y;
  dev_bounds_R[o3 * n_cells + id] = interface_R_imh.density * interface_R_imh.velocity_z;
  dev_bounds_R[4 * n_cells + id] =
      interface_R_imh.pressure / (gamma - 1.0) + 0.5 * interface_R_imh.density *
                                                     (interface_R_imh.velocity_x * interface_R_imh.velocity_x +
                                                      interface_R_imh.velocity_y * interface_R_imh.velocity_y +
                                                      interface_R_imh.velocity_z * interface_R_imh.velocity_z);
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_bounds_R[(5 + i) * n_cells + id] = interface_R_imh.density * interface_R_imh.scalar[i];
  }
#endif  // SCALAR
#ifdef DE
  dev_bounds_R[grid_enum::GasEnergy * n_cells + id] = interface_R_imh.density * interface_R_imh.gas_energy;
#endif  // DE
  // bounds_L refers to the left side of the i+1/2 interface
  id                              = xid + yid * nx + zid * nx * ny;
  dev_bounds_L[id]                = interface_L_iph.density;
  dev_bounds_L[o1 * n_cells + id] = interface_L_iph.density * interface_L_iph.velocity_x;
  dev_bounds_L[o2 * n_cells + id] = interface_L_iph.density * interface_L_iph.velocity_y;
  dev_bounds_L[o3 * n_cells + id] = interface_L_iph.density * interface_L_iph.velocity_z;
  dev_bounds_L[4 * n_cells + id] =
      interface_L_iph.pressure / (gamma - 1.0) + 0.5 * interface_L_iph.density *
                                                     (interface_L_iph.velocity_x * interface_L_iph.velocity_x +
                                                      interface_L_iph.velocity_y * interface_L_iph.velocity_y +
                                                      interface_L_iph.velocity_z * interface_L_iph.velocity_z);
#ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    dev_bounds_L[(5 + i) * n_cells + id] = interface_L_iph.density * interface_L_iph.scalar[i];
  }
#endif  // SCALAR
#ifdef DE
  dev_bounds_L[grid_enum::GasEnergy * n_cells + id] = interface_L_iph.density * interface_L_iph.gas_energy;
#endif  // DE
}

/*! \file hydro_cuda.h
 *  \brief Declaration/implementation of functions used in dual-energy formalism.
 *
 *  \note
 *  A major goal of this file is to provide 1 implementation of each function that can be used for
 *  1, 2, or 3 dimensions. Previously, we maintained a different versions of each function for each
 *  number of dimensions. There are 3 particular factors that strongly motivate this goal:
 *  1. **MOST IMPORTANTLY**, it is very hard to come up with robust test-problems for the dual-energy
 *     formalism. Additionally, maintaining 3 variants of a function always invites mistakes. While we
 *     maintain 3 implementations of lots of other functions, those implementations are either simpler or
 *     are a lot easier to test.
 *  2. The dual energy formalism has a lot of subtlties and we make an extension to it. I'd argue that this
 *     makes mistakes more likely here than in other sections of the code.
 *  3. We sometimes need to make adjustments to the formalism based on scenarios only encountered in 3D. When
 *     leaving comments explaining why we implement a particular action (or an action in a particular way),
 *     that means that we need to leave slightly different versions of the same comment in 1D/2D vs 3D
 *     implementations.
 */

#ifndef DUAL_ENERGY_H
#define DUAL_ENERGY_H

#include "../global/global.h"
#include "../grid/grid_enum.h"
#include "../hydro/hydro_cuda.h"
#include "../utils/basic_structs.h"

namespace dual_energy
{

/*! Overwrites the total energy with the sum of the internal energy and the kinetic energy
*
* \tparam NDim The number of dimensions
*
# \param[in,out] dev_conserved Pointer to the conserved quantities on the device
* \param[in] grid_shape Specifies the shape of the grid (including ghost zones). Values along unused dimensions
* aren't used.
* \param[in] n_ghost The number of ghost zones (along each used dimension)
* \param[in] gamma The adiabatic index
* \param[in] n_fields The number of fields used in the current simulation.
*
* \note
* The scenario where E > 0 and E doesn't exceed the kinetic energy (i.e. U_total <= 0), comes up with some frequency
* when we include particle feedback.
*   * in that scenario, we explicitly use the value held by the U_advected field
*   * in the separate `Sync_Energies_3D` kernel, we then override the E field with KE + U_advected
*   * one might argue we should actually override the E field with KE + U_advected before this kernel
*     (we leave that for future consideration)
*   * regardless, it is **REALLY IMPORTANT** that the E field is **NOT** modified in this kernel
*     (modifying it will produce race conditions!!!)
*/
template <int NDim>
__global__ void Select_Internal_Energy(Real *dev_conserved, hydro_utilities::VectorXYZ<int> grid_shape, int n_ghost,
                                       int n_fields)
{
  static_assert((NDim == 1) || (NDim == 2) || (NDim == 3), "NDim must be 1, 2, or 3");

#ifndef DE
  printf("WARNING: this function isn't usable since Cholla wasn't compiled with Dual Energy Formalism!\n");

#else
  int id, n_cells;
  int neighbor_ids[NDim * 2];
  bool is_real_cell;

  // Here we branch and determine some basic details based on the number of dimensions
  // -> there are much more clever ways to do all of this more concisely
  // -> for the uninitiated, the conditions of `constexpr if` statements (within templates) are evaluated
  //    at compile time. In other words, there is no branching
  if constexpr (NDim == 1) {
    n_cells = grid_shape.x();
    id      = threadIdx.x + blockIdx.x * blockDim.x;

    int nx = grid_shape.x();

    int xid = id;

    int imo         = max(xid - 1, n_ghost);
    neighbor_ids[0] = imo;
    int ipo         = min(xid + 1, nx - n_ghost - 1);
    neighbor_ids[1] = ipo;

    is_real_cell = (xid > n_ghost - 1 && xid < nx - n_ghost);

  } else if constexpr (NDim == 2) {
    n_cells = grid_shape.x() * grid_shape.y();

    // we maintain the use of blockId for backwards compatability, but I'm not sure it does anything
    // (i.e. can we just replace blockId with blockIdx.x as in the other 2 branches)?
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    id          = threadIdx.x + blockId * blockDim.x;

    int nx = grid_shape.x();
    int ny = grid_shape.y();

    int yid = id / nx;
    int xid = id - yid * nx;

    int imo         = max(xid - 1, n_ghost);
    neighbor_ids[0] = imo + yid * nx;
    int ipo         = min(xid + 1, nx - n_ghost - 1);
    neighbor_ids[1] = ipo + yid * nx;
    int jmo         = max(yid - 1, n_ghost);
    neighbor_ids[2] = xid + jmo * nx;
    int jpo         = min(yid + 1, ny - n_ghost - 1);
    neighbor_ids[3] = xid + jpo * nx;

    is_real_cell = (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost);

  } else {  // NDim == 3
    n_cells = grid_shape.x() * grid_shape.y() * grid_shape.z();
    id      = threadIdx.x + blockIdx.x * blockDim.x;

    int nx = grid_shape.x();
    int ny = grid_shape.y();
    int nz = grid_shape.z();

    int zid = id / (nx * ny);
    int yid = (id - zid * nx * ny) / nx;
    int xid = id - zid * nx * ny - yid * nx;

    int imo         = max(xid - 1, n_ghost);
    neighbor_ids[0] = imo + yid * nx + zid * nx * ny;
    int ipo         = min(xid + 1, nx - n_ghost - 1);
    neighbor_ids[1] = ipo + yid * nx + zid * nx * ny;
    int jmo         = max(yid - 1, n_ghost);
    neighbor_ids[2] = xid + jmo * nx + zid * nx * ny;
    int jpo         = min(yid + 1, ny - n_ghost - 1);
    neighbor_ids[3] = xid + jpo * nx + zid * nx * ny;
    int kmo         = max(zid - 1, n_ghost);
    neighbor_ids[4] = xid + yid * nx + kmo * nx * ny;
    int kpo         = min(zid + 1, nz - n_ghost - 1);
    neighbor_ids[5] = xid + yid * nx + kpo * nx * ny;

    is_real_cell = (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost &&
                    zid > n_ghost - 1 && zid < nz - n_ghost);
  }

  const Real eta_1 = DE_ETA_1;
  const Real eta_2 = DE_ETA_2;

  // threads corresponding to real cells do the calculation
  if (is_real_cell) {
    // every thread collects the conserved variables it needs from global memory
    Real d          = dev_conserved[id];
    Real d_inv      = 1.0 / d;
    Real vx         = dev_conserved[1 * n_cells + id] * d_inv;
    Real vy         = dev_conserved[2 * n_cells + id] * d_inv;
    Real vz         = dev_conserved[3 * n_cells + id] * d_inv;
    Real E          = dev_conserved[4 * n_cells + id];
    Real U_advected = dev_conserved[(n_fields - 1) * n_cells + id];
    Real U_total    = E - 0.5 * d * (vx * vx + vy * vy + vz * vz);

    // We will deal with this crashed cell later in a different kernel... (at the time of writing, a 1D
    // simulation always uses floors for this purpose)
    if (Cell_Is_Crashed(d, E)) return;

    // find the max nearby total energy (from the local cell and any uncrashed neighbors)
    // -> we take the stance that "crashed" neighbors are unreliable, even if total energy looks ok
    // -> we're making use of a "range-based for loop"
    Real Emax = E;
    for (int neighbor_id : neighbor_ids) {
      Real neighbor_d = dev_conserved[grid_enum::density * n_cells + neighbor_id];
      Real neighbor_E = dev_conserved[grid_enum::Energy * n_cells + neighbor_id];
      Emax            = fmax(Emax, Cell_Is_Crashed(neighbor_d, neighbor_E) ? Emax : neighbor_E);
    }

    // Ordinarily, we only use the "advected" internal energy if both:
    // - the thermal energy divided by total energy is a small fraction (smaller than eta_1)
    // - AND we aren't masking shock heating (details controlled by Emax & eta_2)
    // We also explicitly use the "advected" internal energy if the total energy is positive but doesn't
    // exceed kinetic energy (i.e. U_total <= 0). This scenario comes up in simulations with particle-based
    // feedback.
    bool prefer_U_total = (U_total > E * eta_1) or (U_total > Emax * eta_2);
    Real U              = (prefer_U_total and (U_total > 0)) ? U_total : U_advected;

    // Optional: Avoid Negative Internal  Energies
    U = fmax(U, (Real)TINY_NUMBER);

    // Write Selected internal energy to the GasEnergy array ONLY
    // to avoid mixing updated and non-updated values of E
    // since the Dual Energy condition depends on the neighbour cells
    dev_conserved[(n_fields - 1) * n_cells + id] = U;
  }
#endif /* DE */
}

/*! Overwrites the total energy with the sum of the internal energy and the kinetic energy
 *
 *  This functionality **MUST** be invoke in a separate kernel from the functionality implemented by
 *  Select_Internal_Energy in order to avoid race-conditions (the race-conditions would interfere with the energy
 *  selection in the other function).
 *
 *  \tparam NDim The number of dimensions
 *
 #  \param[in,out] dev_conserved Pointer to the conserved quantities on the device
 *  \param[in] grid_shape Specifies the shape of the grid (including ghost zones). Values along unused dimensions
 *      aren't used.
 *  \param[in] n_ghost The number of ghost zones (along each used dimension)
 *  \param[in] n_fields The number of fields used in the current simulation.
 *
 *  \note
 *  This is not technically a part of the dual energy formalism. But it is a common extension.
 */
template <int NDim>
__global__ void Sync_Energies(Real *dev_conserved, hydro_utilities::VectorXYZ<int> grid_shape, int n_ghost,
                              int n_fields)
{
  static_assert((NDim == 1) || (NDim == 2) || (NDim == 3), "NDim must be 1, 2, or 3");

#ifndef DE
  printf("WARNING: this function isn't usable since Cholla wasn't compiled with Dual Energy Formalism!\n");

#else
  int id, n_cells;
  bool is_real_cell;

  // Here we branch and determine some basic details based on the number of dimensions
  // -> for the uninitiated, the conditions of `constexpr if` statements (within templates) are evaluated
  //    at compile time. In other words, there is no branching
  if constexpr (NDim == 1) {
    n_cells = grid_shape.x();

    // get a global thread ID
    id           = threadIdx.x + blockIdx.x * blockDim.x;
    int xid      = id;
    is_real_cell = (xid > n_ghost - 1 && xid < grid_shape.x() - n_ghost);

  } else if constexpr (NDim == 2) {
    n_cells = grid_shape.x() * grid_shape.y();

    int nx = grid_shape.x();
    int ny = grid_shape.y();

    // get a global thread ID
    int blockId  = blockIdx.x + blockIdx.y * gridDim.x;
    id           = threadIdx.x + blockId * blockDim.x;
    int yid      = id / nx;
    int xid      = id - yid * nx;
    is_real_cell = (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost);

  } else {  // NDim == 3
    n_cells = grid_shape.x() * grid_shape.y() * grid_shape.z();

    int nx = grid_shape.x();
    int ny = grid_shape.y();
    int nz = grid_shape.z();

    // get a global thread ID
    id           = threadIdx.x + blockIdx.x * blockDim.x;
    int zid      = id / (nx * ny);
    int yid      = (id - zid * nx * ny) / nx;
    int xid      = id - zid * nx * ny - yid * nx;
    is_real_cell = (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost &&
                    zid > n_ghost - 1 && zid < nz - n_ghost);
  }

  // threads corresponding to real cells do the calculation
  if (is_real_cell) {
    // every thread collects the conserved variables it needs from global memory
    Real d     = dev_conserved[grid_enum::density * n_cells + id];
    Real d_inv = 1.0 / d;
    Real vx    = dev_conserved[grid_enum::momentum_x * n_cells + id] * d_inv;
    Real vy    = dev_conserved[grid_enum::momentum_y * n_cells + id] * d_inv;
    Real vz    = dev_conserved[grid_enum::momentum_z * n_cells + id] * d_inv;
    Real U     = dev_conserved[(n_fields - 1) * n_cells + id];

    // Use the previously selected Internal Energy to update the total energy
    dev_conserved[grid_enum::Energy * n_cells + id] = 0.5 * d * (vx * vx + vy * vy + vz * vz) + U;
  }

#endif /* DE */
}

}  // namespace dual_energy

#endif /* DUAL_ENERGY_H */
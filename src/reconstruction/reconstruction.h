/*!
 * \file reconstruction.h
 * \brief Contains the "riemann solved facing" controller function to choose, compute, and return the interface states
 *
 */

#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "../reconstruction/pcm_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../reconstruction/reconstruction_internals.h"
#include "../utils/hydro_utilities.h"

namespace reconstruction
{
/*!
 * \brief
 *
 * \tparam reconstruction_order What kind of reconstruction to use, PCM, PLMC, etc. This argument should always be a
 * member of the reconstruction::Kind enum, behaviour is undefined otherwise.
 * \tparam direction The direction that the solve is taking place in. 0=X, 1=Y, 2=Z
 * \param[in] dev_conserved
 * \param[in] xid
 * \param[in] yid
 * \param[in] zid
 * \param[in] nx
 * \param[in] ny
 * \param[in] n_cells
 * \param[in] gamma
 * \param[out] left_interface
 * \param[out] right_interface
 */
template <int reconstruction_order, uint direction>
void __device__ inline Reconstruct_Interface_States(Real const *dev_conserved, size_t const xid, size_t const yid,
                                                    size_t const zid, size_t const nx, size_t const ny,
                                                    size_t const n_cells, Real const gamma, Real const dx,
                                                    Real const dt, reconstruction::InterfaceState &left_interface,
                                                    reconstruction::InterfaceState &right_interface,
                                                    Real const magnetic_x = 0.0)
{
  // First, check that our reconstruction order is correct
  static_assert(
      (reconstruction_order == reconstruction::Kind::pcm) or (reconstruction_order == reconstruction::Kind::plmc) or
          (reconstruction_order == reconstruction::Kind::plmp) or
          (reconstruction_order == reconstruction::Kind::ppmc) or (reconstruction_order == reconstruction::Kind::ppmp),
      "The reconstruction chosen does not exist. You must use a member of reconstruction::Kind to choose the order.");

  // Choose the correct reconstruction method and compute the interface states
  if constexpr (reconstruction_order == reconstruction::Kind::pcm) {
    // Load the PCM interfaces
    left_interface =
        reconstruction::PCM_Reconstruction<direction>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);
    right_interface = reconstruction::PCM_Reconstruction<direction>(
        dev_conserved, xid + size_t(direction == 0), yid + size_t(direction == 1), zid + size_t(direction == 2), nx, ny,
        n_cells, gamma);
  } else if constexpr (reconstruction_order == reconstruction::Kind::plmc) {
    // load the 3-cell stencil into registers

    // cell i-1. The equality checks the direction and will subtract one from the correct direction
    hydro_utilities::Primitive cell_im1 = hydro_utilities::Load_Cell_Primitive<direction>(
        dev_conserved, xid - int(direction == 0), yid - int(direction == 1), zid - int(direction == 2), nx, ny, n_cells,
        gamma);

    // cell i
    hydro_utilities::Primitive cell_i =
        hydro_utilities::Load_Cell_Primitive<direction>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);

    // cell i+1. The equality checks the direction and add one to the correct direction
    hydro_utilities::Primitive cell_ip1 = hydro_utilities::Load_Cell_Primitive<direction>(
        dev_conserved, xid + int(direction == 0), yid + int(direction == 1), zid + int(direction == 2), nx, ny, n_cells,
        gamma);

    // Calculate the left PLMC interface
    hydro_utilities::Primitive left_interface_primitive =
        reconstruction::PLMC_Reconstruction(cell_im1, cell_i, cell_ip1, dx, dt, gamma, 1.0);

    // Move all the data down one and load the next cell
    cell_im1 = cell_i;
    cell_i   = cell_ip1;
    cell_ip1 = hydro_utilities::Load_Cell_Primitive<direction>(dev_conserved, xid + 2 * int(direction == 0),
                                                               yid + 2 * int(direction == 1),
                                                               zid + 2 * int(direction == 2), nx, ny, n_cells, gamma);

    // Calculate the right PLMC interface
    hydro_utilities::Primitive right_interface_primitive =
        reconstruction::PLMC_Reconstruction(cell_im1, cell_i, cell_ip1, dx, dt, gamma, -1.0);

// Convert to an InterfaceState object
#ifdef MHD
    left_interface_primitive.magnetic.x  = magnetic_x;
    right_interface_primitive.magnetic.x = magnetic_x;
#endif  // MHD
    left_interface  = reconstruction::Primitive_2_InterfaceState(left_interface_primitive, gamma);
    right_interface = reconstruction::Primitive_2_InterfaceState(right_interface_primitive, gamma);
  }
  // else if constexpr(reconstruction_order == reconstruction::Kind::plmp)
  // {

  // }
  // else if constexpr(reconstruction_order == reconstruction::Kind::ppmc)
  // {

  // }
  // else if constexpr(reconstruction_order == reconstruction::Kind::ppmp)
  // {

  // }
  // else
  // {
  //   // todo, once refactor is done, move static assert warning to here since having it seperate effectively
  //   duplicates effort
  // }

  // Renable this once all reconstructions have been converted
  // struct LocalReturnStruct {
  //   reconstruction::InterfaceState left, right;
  // };
  // return LocalReturnStruct{left_interface, right_interface};
}

}  // namespace reconstruction

#endif  //! RECONSTRUCTION_H

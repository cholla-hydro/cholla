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

  // Compute the indices at direction+1 for the direction of the solve
  size_t const xid_p1 = xid + size_t(direction == 0);
  size_t const yid_p1 = yid + size_t(direction == 1);
  size_t const zid_p1 = zid + size_t(direction == 2);

  // Choose the correct reconstruction method and compute the interface states
  if constexpr (reconstruction_order == reconstruction::Kind::pcm) {
    // Load the PCM interfaces
    left_interface =
        reconstruction::PCM_Reconstruction<direction>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);
    right_interface =
        reconstruction::PCM_Reconstruction<direction>(dev_conserved, xid_p1, yid_p1, zid_p1, nx, ny, n_cells, gamma);
  } else if constexpr (reconstruction_order == reconstruction::Kind::plmc) {
    // Calculate the PLMC interfaces, note that we have to do this twice to get the "right" interface
    auto [l_iph_i, r_imh_i] =
        reconstruction::PLMC_Reconstruction<direction>(dev_conserved, xid, yid, zid, nx, ny, n_cells, dx, dt, gamma);
    auto [l_iph_ip1, r_imh_ip1] = reconstruction::PLMC_Reconstruction<direction>(dev_conserved, xid_p1, yid_p1, zid_p1,
                                                                                 nx, ny, n_cells, dx, dt, gamma);

// Convert to an InterfaceState object and assign
#ifdef MHD
    l_iph_i.magnetic.x   = magnetic_x;
    r_imh_ip1.magnetic.x = magnetic_x;
#endif  // MHD
    left_interface  = reconstruction::Primitive_2_InterfaceState(l_iph_i, gamma);
    right_interface = reconstruction::Primitive_2_InterfaceState(r_imh_ip1, gamma);
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

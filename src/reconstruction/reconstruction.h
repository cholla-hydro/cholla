/*!
 * \file reconstruction.h
 * \brief Contains the "riemann solved facing" controller function to choose, compute, and return the interface states
 *
 */

#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "../reconstruction/pcm_cuda.h"
#include "../reconstruction/reconstruction_internals.h"
#include "../utils/hydro_utilities.h"

namespace reconstruction
{
/*!
 * \brief Compute the interface states
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
template <int reconstruction_order, size_t direction>
auto __device__ __host__ inline Reconstruct_Interface_States(Real const *dev_conserved, size_t const xid,
                                                             size_t const yid, size_t const zid, size_t const nx,
                                                             size_t const ny, size_t const n_cells, Real const gamma,
                                                             reconstruction::InterfaceState &left_interface,
                                                             reconstruction::InterfaceState &right_interface)
{
  // First, check that our reconstruction order is correct
  static_assert(
      (reconstruction_order == reconstruction::Kind::pcm) or (reconstruction_order == reconstruction::Kind::plmc) or
          (reconstruction_order == reconstruction::Kind::plmp) or
          (reconstruction_order == reconstruction::Kind::ppmc) or (reconstruction_order == reconstruction::Kind::ppmp),
      "The reconstruction chosen does not exist. You must use a member of reconstruction::Kind to choose the order.");

  // Choose the correct reconstruction method and compute the interface states
  if constexpr (reconstruction_order == reconstruction::Kind::pcm) {
    left_interface =
        reconstruction::PCM_Reconstruction<direction>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);
    right_interface = reconstruction::PCM_Reconstruction<direction>(dev_conserved, xid + int(direction == 0),
                                                                    yid + int(direction == 1),
                                                                    zid + int(direction == 2), nx, ny, n_cells, gamma);
  }
  // else if constexpr(reconstruction_order == reconstruction::Kind::plmc)
  // {

  // }
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
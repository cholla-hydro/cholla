/*! \file rt_source_store.h
 *  \brief SourceStore: the sources owned by this rank, stored on the device, plus
 *         drift and MPI migration. */

#ifndef RT_SOURCE_STORE_H
#define RT_SOURCE_STORE_H

#ifdef RT

  #include <vector>

  #include "../../global/global.h"
  #include "rt_source_state.h"

namespace rt_sources
{

/*! \brief Per-source fields. Every field is stored as Real so a source packs into
 *  SRC_NFIELDS contiguous Reals for MPI and restart I/O. IDs are exact in double
 *  precision up to 2^53. Positions are in code length units, velocities follow the
 *  particle convention, times are in Myr, and lum is L / L_ref (dimensionless). */
enum SourceField : int {
  SRC_X = 0,
  SRC_Y,
  SRC_Z,
  SRC_VX,
  SRC_VY,
  SRC_VZ,
  SRC_LUM,
  SRC_TBIRTH,
  SRC_TLIFE,  // <= 0 means the source never dies
  SRC_ID,
  SRC_NFIELDS
};

class SourceStore
{
 public:
  SourceStore() = default;
  ~SourceStore();
  SourceStore(const SourceStore &)            = delete;
  SourceStore &operator=(const SourceStore &) = delete;

  /*! \brief Number of sources owned by this rank. */
  int n_local = 0;

  /*! \brief Allocated number of sources. */
  int capacity = 0;

  /*! \brief Field-major source data on the device: field f of source i is
   *  dev_data[f * capacity + i]. */
  Real *dev_data = nullptr;

  /*! \brief Pointer to field f on the device. */
  Real *Field(int f) const { return dev_data + (size_t)f * capacity; }

  /*! \brief Grow the allocation to hold at least n sources, keeping the current ones. */
  void Reserve(int n);

  /*! \brief Append n sources packed as [i * SRC_NFIELDS + f], from host memory. */
  void Append_Host(const Real *packed, int n);

  /*! \brief Append n sources packed as [i * SRC_NFIELDS + f], from device memory. */
  void Append_Device(const Real *dev_packed, int n);

  /*! \brief Copy all local sources to host, packed as [i * SRC_NFIELDS + f]. */
  void Copy_To_Host(std::vector<Real> &packed) const;

  /*! \brief Advance positions: pos += vel * drift_factor. */
  void Drift(Real drift_factor);

  /*! \brief Move sources that left the local domain to the rank that owns them,
   *  x then y then z, wrapping positions periodically at the global boundary. */
  void Migrate(const RT_State &state);

 private:
  Real *dev_scratch_ = nullptr;             // same layout as dev_data, target of the partition
  Real *dev_send_[2] = {nullptr, nullptr};  // packed sources leaving through lower/upper face
  Real *dev_unpack_  = nullptr;             // staging for Append_Host
  int *dev_counts_   = nullptr;             // [n_keep, n_send_lower, n_send_upper]

  void Migrate_Direction_(const RT_State &state, int dir);
};

}  // namespace rt_sources

#endif  // RT
#endif  // RT_SOURCE_STORE_H
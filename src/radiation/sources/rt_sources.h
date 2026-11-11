/*! \file rt_sources.h
 *  \brief Declaration of RTSourceModel, which owns the source populations and their
 *         sources and writes dev_rs each step.
 *
 *  This header is included by radiation.h, so it must not include radiation.h or
 *  grid3D.h. Anything it needs from them is forward-declared or passed in. */

#ifndef RT_SOURCES_H
#define RT_SOURCES_H

#ifdef RT

  #include <string>
  #include <vector>

  #include "../../global/global.h"  // Real, Parameters; forward-declares ParameterMap
  #include "../rt_source_state.h"
  #include "rt_source_population.h"
  #include "rt_source_store.h"

  #ifdef HDF5
    #include <hdf5.h>
  #endif

namespace rt_photo_rates_csi
{
struct TableWrapperGPU;
}

namespace rt_sources
{

/*! \brief Device and host buffers for the additive ghost-cell fold of dev_rs.
 *  Index 0/1 = data leaving through the lower/upper face, 2/3 = data arriving from the
 *  lower/upper neighbor. */
struct FoldBuffers {
  int size      = 0;  // Reals per buffer: n_ghost * (largest face area including ghosts)
  Real *dev[4]  = {nullptr, nullptr, nullptr, nullptr};
  Real *host[4] = {nullptr, nullptr, nullptr, nullptr};

  void Allocate(const RT_State &state);
  void Free();
};

/*! \brief Spread the sources of one population into dev_rs (code units, not multiplied by
 *  the scale factor; rtSolve applies rs_unit = a). Writes into ghost cells as well; call
 *  Fold_Source_Ghosts afterwards. dev_rs must already hold this step's other contributions
 *  (normally zero). */
void Deposit_Sources(const SourceStore &store, const RTPopulation &pop, const RT_State &state, Real *dev_rs);

/*! \brief Add the contents of the ghost cells of dev_rs into the active cells of the rank
 *  that owns them (x, then y, then z), then zero the ghost cells. */
void Fold_Source_Ghosts(Real *dev_rs, const RT_State &state, FoldBuffers &buf);

/*! \brief Fill the spectral shape of a population and update its photo-rate table. */
void Set_Population_SED(const RTPopulation &pop, rt_photo_rates_csi::TableWrapperGPU *photoRates, Real length_unit_cgs);

}  // namespace rt_sources

class RTSourceModel
{
 public:
  RTSourceModel() = default;
  ~RTSourceModel();
  RTSourceModel(const RTSourceModel &)            = delete;
  RTSourceModel &operator=(const RTSourceModel &) = delete;

  /*! \brief Grid and time information for the current step. Set by Grid3D::Update_RT_State(). */
  RT_State state;

  /*! \brief Number of source populations (0 or 1 for now). */
  int n_pops = 0;

  /*! \brief Population descriptors, one per population. */
  std::vector<rt_sources::RTPopulation> pops;

  /*! \brief Sources owned by this rank. One store while n_pops <= 1. */
  rt_sources::SourceStore store;

  bool Active() const { return n_pops > 0; }

  /*! \brief Read the population parameters, set the spectra, and load this rank's sources
   *  from its per-rank catalog file, or from its RT restart file if restart_fname is not
   *  null. state must be set. */
  void Initialize(const Parameters &P, ParameterMap &pmap, rt_photo_rates_csi::TableWrapperGPU *photoRates,
                  const char *restart_fname);

  /*! \brief Advance, migrate and deposit the sources: rebuilds dev_rs for this step. */
  void Update(Real *dev_rs);

  /*! \brief Number of sources on all ranks. Collective under MPI. */
  long Global_Count() const;

  #ifdef HDF5
  /*! \brief Write this rank's sources into an open HDF5 file (group /sources). Used for
   *  the RT restart file; python/scripts/rt_split_sources.py writes the same layout. */
  void Write_Restart_HDF5(hid_t file_id) const;

  /*! \brief Append the sources in group /sources of a per-rank file (catalog or restart).
   *  Aborts if the file was written for a different local domain. */
  void Read_Sources_HDF5(const char *fname);
  #endif

 private:
  rt_sources::FoldBuffers fold_;

  void Read_Parameters_(const Parameters &P, ParameterMap &pmap);
};

#endif  // RT
#endif  // RT_SOURCES_H
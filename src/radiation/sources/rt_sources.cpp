/*! \file rt_sources.cpp
 *  \brief Definitions of RTSourceModel and the population helpers. */

#ifdef RT

  #include "rt_sources.h"

  #include <cmath>
  #include <string>

  #include "../../global/global_cuda.h"
  #include "../../io/ParameterMap.h"
  #include "../../io/io.h"
  #include "../../utils/error_handling.h"
  #include "../../utils/gpu.hpp"
  #include "../radiation.h"  // Rad3D::Length_Unit_CGS
  #ifdef MPI_CHOLLA
    #include "../../mpi/mpi_routines.h"
  #endif

namespace rt_sources
{

SedType Parse_Sed_Type(const std::string &s)
{
  if (s == "monochromatic_HI") return SedType::monochromatic_HI;
  if (s == "blackbody") return SedType::blackbody;
  if (s == "popII") return SedType::popII;
  if (s == "popIII") return SedType::popIII;
  if (s == "quasar") return SedType::quasar;
  CHOLLA_ERROR("RT sources: unknown SED \"%s\" (monochromatic_HI, blackbody, popII, popIII, quasar).", s.c_str());
  return SedType::monochromatic_HI;
}

KernelType Parse_Kernel_Type(const std::string &s)
{
  if (s == "ngp") return KernelType::ngp;
  if (s == "cic") return KernelType::cic;
  if (s == "tsc") return KernelType::tsc;
  if (s == "tophat") return KernelType::tophat;
  CHOLLA_ERROR("RT sources: unknown kernel \"%s\" (ngp, cic, tsc, tophat).", s.c_str());
  return KernelType::cic;
}

RuleType Parse_Rule_Type(const std::string &s)
{
  if (s == "per_rank_file") return RuleType::per_rank_file;
  CHOLLA_ERROR("RT sources: unknown rule \"%s\" (per_rank_file).", s.c_str());
  return RuleType::per_rank_file;
}

const char *Sed_Type_Name(SedType t)
{
  switch (t) {
    case SedType::monochromatic_HI:
      return "monochromatic_HI";
    case SedType::blackbody:
      return "blackbody";
    case SedType::popII:
      return "popII";
    case SedType::popIII:
      return "popIII";
    case SedType::quasar:
      return "quasar";
  }
  return "unknown";
}

const char *Kernel_Type_Name(KernelType t)
{
  switch (t) {
    case KernelType::ngp:
      return "ngp";
    case KernelType::cic:
      return "cic";
    case KernelType::tsc:
      return "tsc";
    case KernelType::tophat:
      return "tophat";
  }
  return "unknown";
}

int Kernel_Half_Width(KernelType k, Real radius)
{
  switch (k) {
    case KernelType::ngp:
      return 0;
    case KernelType::cic:
      return 1;
    case KernelType::tsc:
      return 1;
    case KernelType::tophat:
      return (int)std::ceil(radius);
  }
  return 0;
}

}  // namespace rt_sources

RTSourceModel::~RTSourceModel() { fold_.Free(); }

void RTSourceModel::Read_Parameters_(const Parameters &P, ParameterMap &pmap)
{
  using namespace rt_sources;

  // Keys are built from the population index so that ParameterMap sees each one accessed
  // and Warn_Unused_Params stays quiet. Example for population 0:
  //   rt_pop0_sed=blackbody  rt_pop0_sed_T=1e5  rt_pop0_Lref=5e48
  //   rt_pop0_kernel=cic  rt_pop0_kernel_radius=1  rt_pop0_rule=per_rank_file
  //   rt_pop0_catalog=sources    (rank p reads sources.h5.p)
  pops.resize(n_pops);
  for (int p = 0; p < n_pops; p++) {
    const std::string key = "rt_pop" + std::to_string(p) + "_";
    RTPopulation &pop     = pops[p];

    pop.index         = p;
    pop.sed           = Parse_Sed_Type(pmap.value_or(key + "sed", std::string("monochromatic_HI")));
    pop.sed_T         = pmap.value_or(key + "sed_T", 1.0e5);
    pop.L_ref         = pmap.value<double>(key + "Lref");  // required: aborts if missing
    pop.kernel        = Parse_Kernel_Type(pmap.value_or(key + "kernel", std::string("cic")));
    pop.kernel_radius = pmap.value_or(key + "kernel_radius", 1.0);
    pop.rule          = Parse_Rule_Type(pmap.value_or(key + "rule", std::string("per_rank_file")));
    if (pop.rule == RuleType::per_rank_file) pop.catalog = pmap.value<std::string>(key + "catalog");

    if (pop.L_ref <= 0) CHOLLA_ERROR("RT sources: %sLref must be positive.", key.c_str());
    if (pop.kernel == KernelType::tophat && pop.kernel_radius < 0.5)
      CHOLLA_ERROR("RT sources: %skernel_radius must be >= 0.5 cells for a tophat kernel.", key.c_str());

    const int hw = Kernel_Half_Width(pop.kernel, pop.kernel_radius);
    if (hw > state.nghost) {
      CHOLLA_ERROR("RT sources: the %s kernel of population %d reaches %d cells, more than n_ghost = %d.",
                   Kernel_Type_Name(pop.kernel), p, hw, state.nghost);
    }
  }
}

void RTSourceModel::Initialize(const Parameters &P, ParameterMap &pmap, rt_photo_rates_csi::TableWrapperGPU *photoRates,
                               const char *restart_fname)
{
  n_pops = P.rt_n_pops;
  if (n_pops == 0) return;
  if (n_pops != 1) CHOLLA_ERROR("RT sources: only rt_n_pops = 0 or 1 is supported (got %d).", n_pops);
  if (state.ncells == 0) CHOLLA_ERROR("RT sources: Initialize called before Update_RT_State.");

  Read_Parameters_(P, pmap);

  // The spectrum of each population replaces whatever the initial conditions set.
  for (const auto &pop : pops) rt_sources::Set_Population_SED(pop, photoRates, Rad3D::Length_Unit_CGS(P));

  fold_.Allocate(state);

  #ifdef HDF5
  // Each rank reads only its own file: the RT restart file when restarting, otherwise the
  // per-rank catalog <catalog>.h5.<procID>. With n_pops == 1 there is a single store.
  if (restart_fname != nullptr) {
    Read_Sources_HDF5(restart_fname);
  } else {
    for (const auto &pop : pops) {
      const std::string fname = pop.catalog + ".h5." + std::to_string(procID);
      Read_Sources_HDF5(fname.c_str());
    }
  }
  #else
  CHOLLA_ERROR("RT sources: rt_n_pops > 0 requires HDF5 to read the per-rank source files.");
  #endif

  const auto &pop = pops[0];
  chprintf("RT sources: %d population(s); pop 0: sed=%s Lref=%e photons/s kernel=%s, %ld sources.\n", n_pops,
           rt_sources::Sed_Type_Name(pop.sed), pop.L_ref, rt_sources::Kernel_Type_Name(pop.kernel), Global_Count());
}

void RTSourceModel::Update(Real *dev_rs)
{
  if (!Active()) return;

  // 1. move the sources, then hand off any that left the local domain
  store.Drift(state.drift_factor);
  store.Migrate(state);

  // 2. rebuild the source field from scratch
  GPU_Error_Check(cudaMemset(dev_rs, 0, (size_t)n_pops * state.ncells * sizeof(Real)));
  for (const auto &pop : pops) {
    Real *rs_pop = dev_rs + (size_t)pop.index * state.ncells;
    rt_sources::Deposit_Sources(store, pop, state, rs_pop);
    rt_sources::Fold_Source_Ghosts(rs_pop, state, fold_);
  }
}

long RTSourceModel::Global_Count() const
{
  long n = store.n_local;
  #ifdef MPI_CHOLLA
  long n_global = 0;
  MPI_Allreduce(&n, &n_global, 1, MPI_LONG, MPI_SUM, world);
  n = n_global;
  #endif
  return n;
}

#endif  // RT
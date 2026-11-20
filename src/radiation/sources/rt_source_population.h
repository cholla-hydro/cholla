/*! \file rt_source_population.h
 *  \brief Definition of RTPopulation: one spectral shape, one spatial kernel, and one
 *         rule for how its sources appear, move and disappear. */

#ifndef RT_SOURCE_POPULATION_H
#define RT_SOURCE_POPULATION_H

#ifdef RT

  #include <string>

  #include "../../global/global.h"

namespace rt_sources
{

/*! \brief Spectral shapes. Each maps onto a SpectralShape function (alt/spectral_shape.h),
 *  except monochromatic_HI, which is a delta function at the HI threshold bin. */
enum class SedType { monochromatic_HI, blackbody, popII, popIII, quasar };

/*! \brief Spatial kernels used to spread a source over cells. The integer values are
 *  passed to the device kernel, so keep them stable. */
enum class KernelType : int { ngp = 0, cic = 1, tsc = 2, tophat = 3 };

/*! \brief How the sources of a population are created and evolved.
 *  per_rank_file: each rank reads only its own file, <catalog>.h5.<procID>, written by
 *  python/scripts/rt_split_sources.py for the run's domain decomposition. */
enum class RuleType { per_rank_file };

struct RTPopulation {
  /*! \brief Population index: the photo-rate table index and the dev_rs field index. */
  int index = 0;

  SedType sed = SedType::monochromatic_HI;

  /*! \brief Blackbody temperature [K] (blackbody only). */
  Real sed_T = 1.0e5;

  /*! \brief Ionizing photon rate [photons/s] of a source with lum = 1. The photo-rate
   *  table carries this normalization; each source carries lum = L / L_ref. */
  Real L_ref = 0;

  KernelType kernel = KernelType::cic;

  /*! \brief Kernel radius in cells (tophat only). */
  Real kernel_radius = 1;

  RuleType rule = RuleType::per_rank_file;

  /*! \brief File prefix for per_rank_file populations; rank p reads <catalog>.h5.<p>. */
  std::string catalog;
};

/*! \brief String <-> enum conversion for the parameter file. Abort on unknown names. */
SedType Parse_Sed_Type(const std::string &s);
KernelType Parse_Kernel_Type(const std::string &s);
RuleType Parse_Rule_Type(const std::string &s);
const char *Sed_Type_Name(SedType t);
const char *Kernel_Type_Name(KernelType t);

/*! \brief Largest number of cells the kernel reaches beyond the cell containing the source.
 *  Must not exceed n_ghost, so that a single ghost fold per direction is enough. */
int Kernel_Half_Width(KernelType k, Real radius);

}  // namespace rt_sources

#endif  // RT
#endif  // RT_SOURCE_POPULATION_H
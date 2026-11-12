/*! \file rt_source_sed.cpp
 *  \brief Spectral shape of a population, written into its photo-rate table. */

#ifdef RT

  #include <cmath>
  #include <vector>

  #include "../../global/global.h"
  #include "../alt/atomic_data.h"
  #include "../alt/photo_rates_csi_gpu.h"
  #include "../alt/rt_constants.h"
  #include "../alt/spectral_shape.h"
  #include "rt_sources.h"

namespace rt_sources
{

void Set_Population_SED(const RTPopulation &pop, rt_photo_rates_csi::TableWrapperGPU *photoRates, Real length_unit_cgs)
{
  // The CSI scheme carries the spectrum in a photo-rate table rather than in the
  // radiation fields: all four M1 groups receive the same dev_rs, and chemistry looks up
  // rates from the group ratios. So the shape and the reference luminosity L_ref live
  // here, and each source only carries lum = L / L_ref.
  //
  // Normalization follows Grid3D::Iliev15 and Shapiro_Giroux_Cosmo: a source of
  // L_ref photons/s whose dev_rs integrates to 1 over code volume.
  using rt_physics::rt_atomic_data::CrossSection;
  const auto *xs = rt_physics::rt_atomic_data::CrossSections();
  std::vector<float> shape(xs->nxi, 0);

  const double scale = pop.L_ref / RTConstant::c / std::pow(length_unit_cgs, 2);

  switch (pop.sed) {
    case SedType::monochromatic_HI:
      // 6.34/5.92: the HI threshold bin is centered at 1.025 Ry, where the cross section is
      // 5.92e-18 cm^2 rather than 6.34e-18 cm^2 (see the Iliev15 comments).
      shape[xs->thresholds[CrossSection::IonizationHI].idx] = scale / xs->dxi * 6.34 / 5.92;
      break;
    case SedType::blackbody:
      SpectralShape::BlackBody(pop.sed_T, shape);
      break;
    case SedType::popII:
      SpectralShape::PopII(shape);
      break;
    case SedType::popIII:
      SpectralShape::PopIII(shape);
      break;
    case SedType::quasar:
      SpectralShape::Quasar(shape);
      break;
  }
  // The SpectralShape functions return shapes normalized to unit ionizing photon rate.
  if (pop.sed != SedType::monochromatic_HI) {
    for (auto &s : shape) s *= scale;
  }

  photoRates->Update(pop.index, shape.data(), xs->dxi * RTConstant::c * 1.0e-24);
}

}  // namespace rt_sources

#endif  // RT
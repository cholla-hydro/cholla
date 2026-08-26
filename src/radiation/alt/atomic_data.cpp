/*LICENSE*/

#include "atomic_data.h"

#include <cstring>

#include "gpu_pointer.h"
#include "rt_constants.h"
#include "static_table.h"

namespace
{
struct AtomicDataOnGPU {
  rt_gpu::DeviceBuffer<rt_physics::rt_atomic_data::CrossSection> dCrossSection;
  rt_gpu::DeviceBuffer<float> dArrs[rt_physics::rt_atomic_data::CrossSection::Num + 3];
};

void CrossSectionBuilder(unsigned int num, float* hnu, float** cs);

rt_physics::rt_atomic_data::CrossSection gCrossSections;
AtomicDataOnGPU gAtomicDataOnGPU;
};  // namespace

const rt_physics::rt_atomic_data::CrossSection* rt_physics::rt_atomic_data::CrossSections() { return &gCrossSections; }

const rt_physics::rt_atomic_data::CrossSection* rt_physics::rt_atomic_data::CrossSectionsGPU()
{
  return gAtomicDataOnGPU.dCrossSection;
}

void rt_physics::rt_atomic_data::Create()
{
  gCrossSections.nxi   = 300;
  gCrossSections.xiMin = -1;
  gCrossSections.dxi   = 0.05F;  // 20 points per e-folding
  gCrossSections.xiMax = gCrossSections.xiMin + gCrossSections.nxi * gCrossSections.dxi;

  gCrossSections.xi     = new float[gCrossSections.nxi];
  gCrossSections.hnu_K  = new float[gCrossSections.nxi];
  gCrossSections.hnu_eV = new float[gCrossSections.nxi];

  for (unsigned int i = 0; i < gCrossSections.nxi; i++) {
    gCrossSections.xi[i]     = gCrossSections.xiMin + gCrossSections.dxi * (i + 0.5F);
    gCrossSections.hnu_K[i]  = rt_physics::rt_atomic_data::TionHI * std::exp(gCrossSections.xi[i]);
    gCrossSections.hnu_eV[i] = rt_physics::rt_atomic_data::Ry_eV * std::exp(gCrossSections.xi[i]);
  }

  // Silence lint because range-based loops with c-style arrays can be confusing

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < rt_physics::rt_atomic_data::CrossSection::Num; k++) {
    gCrossSections.cs[k] = new float[gCrossSections.nxi];
  }
  // NOLINTEND(modernize-loop-convert)

  CrossSectionBuilder(gCrossSections.nxi, gCrossSections.hnu_eV, gCrossSections.cs);

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < rt_physics::rt_atomic_data::CrossSection::Num; k++) {
    gCrossSections.thresholds[k].idx = -1;
    for (unsigned int i = 0; i < gCrossSections.nxi; i++) {
      if (gCrossSections.cs[k][i] > 0) {
        gCrossSections.thresholds[k].idx    = i;
        gCrossSections.thresholds[k].xi     = gCrossSections.xi[i];
        gCrossSections.thresholds[k].hnu_K  = gCrossSections.hnu_K[i];
        gCrossSections.thresholds[k].hnu_eV = gCrossSections.hnu_eV[i];
        gCrossSections.thresholds[k].cs     = gCrossSections.cs[k][i];
        break;
      }
    }
  }
  // NOLINTEND(modernize-loop-convert)

  gCrossSections.csHIatHI =
      gCrossSections.cs[rt_physics::rt_atomic_data::CrossSection::IonizationHI]
                       [gCrossSections.thresholds[rt_physics::rt_atomic_data::CrossSection::IonizationHI].idx];
  gCrossSections.csHIatHeI =
      gCrossSections.cs[rt_physics::rt_atomic_data::CrossSection::IonizationHI]
                       [gCrossSections.thresholds[rt_physics::rt_atomic_data::CrossSection::IonizationHeI].idx];
  gCrossSections.csHIatHeII =
      gCrossSections.cs[rt_physics::rt_atomic_data::CrossSection::IonizationHI]
                       [gCrossSections.thresholds[rt_physics::rt_atomic_data::CrossSection::IonizationHeII].idx];
  gCrossSections.csHeIatHeI =
      gCrossSections.cs[rt_physics::rt_atomic_data::CrossSection::IonizationHeI]
                       [gCrossSections.thresholds[rt_physics::rt_atomic_data::CrossSection::IonizationHeI].idx];
  gCrossSections.csHeIatHeII =
      gCrossSections.cs[rt_physics::rt_atomic_data::CrossSection::IonizationHeI]
                       [gCrossSections.thresholds[rt_physics::rt_atomic_data::CrossSection::IonizationHeII].idx];
  gCrossSections.csHeIIatHeII =
      gCrossSections.cs[rt_physics::rt_atomic_data::CrossSection::IonizationHeII]
                       [gCrossSections.thresholds[rt_physics::rt_atomic_data::CrossSection::IonizationHeII].idx];

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < 3 + rt_physics::rt_atomic_data::CrossSection::Num; k++) {
    gAtomicDataOnGPU.dArrs[k].Alloc(sizeof(float) * gCrossSections.nxi);
  }
  // NOLINTEND(modernize-loop-convert)

  rt_gpu::HostBuffer<float> h;
  h.Alloc(gCrossSections.nxi);
  memcpy(h.Ptr(), gCrossSections.xi, sizeof(float) * gCrossSections.nxi);
  h.BlockingTransferToDevice(gAtomicDataOnGPU.dArrs[0]);
  memcpy(h.Ptr(), gCrossSections.hnu_K, sizeof(float) * gCrossSections.nxi);
  h.BlockingTransferToDevice(gAtomicDataOnGPU.dArrs[1]);
  memcpy(h.Ptr(), gCrossSections.hnu_eV, sizeof(float) * gCrossSections.nxi);
  h.BlockingTransferToDevice(gAtomicDataOnGPU.dArrs[2]);

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < rt_physics::rt_atomic_data::CrossSection::Num; k++) {
    memcpy(h.Ptr(), gCrossSections.cs[k], sizeof(float) * gCrossSections.nxi);
    h.BlockingTransferToDevice(gAtomicDataOnGPU.dArrs[3 + k]);
  }
  // NOLINTEND(modernize-loop-convert)
  h.Free();

  rt_gpu::HostBuffer<rt_physics::rt_atomic_data::CrossSection> hxs(1);
  *hxs.Ptr() = gCrossSections;

  hxs.Ptr()->xi     = gAtomicDataOnGPU.dArrs[0];
  hxs.Ptr()->hnu_K  = gAtomicDataOnGPU.dArrs[1];
  hxs.Ptr()->hnu_eV = gAtomicDataOnGPU.dArrs[2];

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < rt_physics::rt_atomic_data::CrossSection::Num; k++) {
    hxs.Ptr()->cs[k] = gAtomicDataOnGPU.dArrs[3 + k];
  }
  // NOLINTEND(modernize-loop-convert)

  gAtomicDataOnGPU.dCrossSection.Alloc(1);
  hxs.BlockingTransferToDevice(gAtomicDataOnGPU.dCrossSection);
}

void rt_physics::rt_atomic_data::Delete()
{
  gAtomicDataOnGPU.dCrossSection.Free();

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < 3 + rt_physics::rt_atomic_data::CrossSection::Num; k++) {
    gAtomicDataOnGPU.dArrs[k].Free();
  }
  // NOLINTEND(modernize-loop-convert)

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < rt_physics::rt_atomic_data::CrossSection::Num; k++) {
    delete[] gCrossSections.cs[k];
  }
  // NOLINTEND(modernize-loop-convert)

  delete[] gCrossSections.xi;
  delete[] gCrossSections.hnu_K;
  delete[] gCrossSections.hnu_eV;
}

//
//  Actual data
//
namespace
{
void CrossSectionBuilder(unsigned int num, float* hnu_eV, float** cs)
{
  //
  //  Fits are from Verner, Ferland, Korista, Yakovlev  1996ApJ...465..487V.
  //
  static auto csfit = [](double E, double cs0, double E0, double y0, double y1, double yw, double ya, double p) {
    auto x = E / E0 - y0;
    auto y = std::sqrt(x * x + y1 * y1);
    return cs0 * (std::pow(x - 1, 2) + yw * yw) * std::pow(y, 0.5 * p - 5.5) / std::pow(1 + std::sqrt(y / ya), p) /
           1.0e-24;
  };

  for (unsigned int i = 0; i < num; i++) {
    double E = hnu_eV[i];
    cs[rt_physics::rt_atomic_data::CrossSection::IonizationHI][i] =
        (E < rt_physics::rt_atomic_data::Ry_eV ? 0 : csfit(E, 5.475e-14, 4.298e-01, 0.0, 0.0, 0.0, 3.288e+01, 2.963));
    cs[rt_physics::rt_atomic_data::CrossSection::IonizationHeI][i] =
        (E < rt_physics::rt_atomic_data::Ry_eV * rt_physics::rt_atomic_data::TionHeI /
                     rt_physics::rt_atomic_data::TionHI
             ? 0
             : csfit(E, 9.492e-16, 1.361e+01, 4.434e-01, 2.136, 2.039, 1.469, 3.188));
    cs[rt_physics::rt_atomic_data::CrossSection::IonizationHeII][i] =
        (E < rt_physics::rt_atomic_data::Ry_eV * rt_physics::rt_atomic_data::TionHeII /
                     rt_physics::rt_atomic_data::TionHI
             ? 0
             : csfit(E, 1.369e-14, 1.720, 0.0, 0.0, 0.0, 3.288e+01, 2.963));
    cs[rt_physics::rt_atomic_data::CrossSection::IonizationCVI][i] =
        (E < 490 ? 0 : csfit(E, 1.521e-15, 15.48, 0.0, 0.0, 0.0, 3.288e+01, 2.963));
  }
}
};  // namespace

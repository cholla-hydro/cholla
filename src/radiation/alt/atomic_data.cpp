/*LICENSE*/

#include "atomic_data.h"

#include <cstring>

#include "gpu_pointer.h"
#include "rt_constants.h"
#include "static_table.h"

namespace
{
struct AtomicDataOnGPU {
  GPU::DeviceBuffer<Physics::AtomicData::CrossSection> dCrossSection;
  GPU::DeviceBuffer<float> dArrs[Physics::AtomicData::CrossSection::Num + 3];
};

void CrossSectionBuilder(unsigned int num, float* hnu, float** cs);

Physics::AtomicData::CrossSection gCrossSections;
AtomicDataOnGPU gAtomicDataOnGPU;
};  // namespace

const Physics::AtomicData::CrossSection* Physics::AtomicData::CrossSections() { return &gCrossSections; }

const Physics::AtomicData::CrossSection* Physics::AtomicData::CrossSectionsGPU()
{
  return gAtomicDataOnGPU.dCrossSection;
}

void Physics::AtomicData::Create()
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
    gCrossSections.hnu_K[i]  = Physics::AtomicData::TionHI * std::exp(gCrossSections.xi[i]);
    gCrossSections.hnu_eV[i] = Physics::AtomicData::Ry_eV * std::exp(gCrossSections.xi[i]);
  }

  // Silence lint because range-based loops with c-style arrays can be confusing

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < Physics::AtomicData::CrossSection::Num; k++) {
    gCrossSections.cs[k] = new float[gCrossSections.nxi];
  }
  // NOLINTEND(modernize-loop-convert)

  CrossSectionBuilder(gCrossSections.nxi, gCrossSections.hnu_eV, gCrossSections.cs);

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < Physics::AtomicData::CrossSection::Num; k++) {
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
      gCrossSections.cs[Physics::AtomicData::CrossSection::IonizationHI]
                       [gCrossSections.thresholds[Physics::AtomicData::CrossSection::IonizationHI].idx];
  gCrossSections.csHIatHeI =
      gCrossSections.cs[Physics::AtomicData::CrossSection::IonizationHI]
                       [gCrossSections.thresholds[Physics::AtomicData::CrossSection::IonizationHeI].idx];
  gCrossSections.csHIatHeII =
      gCrossSections.cs[Physics::AtomicData::CrossSection::IonizationHI]
                       [gCrossSections.thresholds[Physics::AtomicData::CrossSection::IonizationHeII].idx];
  gCrossSections.csHeIatHeI =
      gCrossSections.cs[Physics::AtomicData::CrossSection::IonizationHeI]
                       [gCrossSections.thresholds[Physics::AtomicData::CrossSection::IonizationHeI].idx];
  gCrossSections.csHeIatHeII =
      gCrossSections.cs[Physics::AtomicData::CrossSection::IonizationHeI]
                       [gCrossSections.thresholds[Physics::AtomicData::CrossSection::IonizationHeII].idx];
  gCrossSections.csHeIIatHeII =
      gCrossSections.cs[Physics::AtomicData::CrossSection::IonizationHeII]
                       [gCrossSections.thresholds[Physics::AtomicData::CrossSection::IonizationHeII].idx];

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < 3 + Physics::AtomicData::CrossSection::Num; k++) {
    gAtomicDataOnGPU.dArrs[k].Alloc(sizeof(float) * gCrossSections.nxi);
  }
  // NOLINTEND(modernize-loop-convert)

  GPU::HostBuffer<float> h;
  h.Alloc(gCrossSections.nxi);
  memcpy(h.Ptr(), gCrossSections.xi, sizeof(float) * gCrossSections.nxi);
  h.BlockingTransferToDevice(gAtomicDataOnGPU.dArrs[0]);
  memcpy(h.Ptr(), gCrossSections.hnu_K, sizeof(float) * gCrossSections.nxi);
  h.BlockingTransferToDevice(gAtomicDataOnGPU.dArrs[1]);
  memcpy(h.Ptr(), gCrossSections.hnu_eV, sizeof(float) * gCrossSections.nxi);
  h.BlockingTransferToDevice(gAtomicDataOnGPU.dArrs[2]);

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < Physics::AtomicData::CrossSection::Num; k++) {
    memcpy(h.Ptr(), gCrossSections.cs[k], sizeof(float) * gCrossSections.nxi);
    h.BlockingTransferToDevice(gAtomicDataOnGPU.dArrs[3 + k]);
  }
  // NOLINTEND(modernize-loop-convert)
  h.Free();

  GPU::HostBuffer<Physics::AtomicData::CrossSection> hxs(1);
  *hxs.Ptr() = gCrossSections;

  hxs.Ptr()->xi     = gAtomicDataOnGPU.dArrs[0];
  hxs.Ptr()->hnu_K  = gAtomicDataOnGPU.dArrs[1];
  hxs.Ptr()->hnu_eV = gAtomicDataOnGPU.dArrs[2];

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < Physics::AtomicData::CrossSection::Num; k++) {
    hxs.Ptr()->cs[k] = gAtomicDataOnGPU.dArrs[3 + k];
  }
  // NOLINTEND(modernize-loop-convert)

  gAtomicDataOnGPU.dCrossSection.Alloc(1);
  hxs.BlockingTransferToDevice(gAtomicDataOnGPU.dCrossSection);
}

void Physics::AtomicData::Delete()
{
  gAtomicDataOnGPU.dCrossSection.Free();

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < 3 + Physics::AtomicData::CrossSection::Num; k++) {
    gAtomicDataOnGPU.dArrs[k].Free();
  }
  // NOLINTEND(modernize-loop-convert)

  // NOLINTBEGIN(modernize-loop-convert)
  for (unsigned int k = 0; k < Physics::AtomicData::CrossSection::Num; k++) {
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
    cs[Physics::AtomicData::CrossSection::IonizationHI][i] =
        (E < Physics::AtomicData::Ry_eV ? 0 : csfit(E, 5.475e-14, 4.298e-01, 0.0, 0.0, 0.0, 3.288e+01, 2.963));
    cs[Physics::AtomicData::CrossSection::IonizationHeI][i] =
        (E < Physics::AtomicData::Ry_eV * Physics::AtomicData::TionHeI / Physics::AtomicData::TionHI
             ? 0
             : csfit(E, 9.492e-16, 1.361e+01, 4.434e-01, 2.136, 2.039, 1.469, 3.188));
    cs[Physics::AtomicData::CrossSection::IonizationHeII][i] =
        (E < Physics::AtomicData::Ry_eV * Physics::AtomicData::TionHeII / Physics::AtomicData::TionHI
             ? 0
             : csfit(E, 1.369e-14, 1.720, 0.0, 0.0, 0.0, 3.288e+01, 2.963));
    cs[Physics::AtomicData::CrossSection::IonizationCVI][i] =
        (E < 490 ? 0 : csfit(E, 1.521e-15, 15.48, 0.0, 0.0, 0.0, 3.288e+01, 2.963));
  }
}
};  // namespace

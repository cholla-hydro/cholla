/*LICENSE*/

#include <cstdio>

#include "../../global/global.h"
#include "../../global/global_cuda.h"
#include "../../utils/gpu.hpp"
#include "atomic_data_decl.h"
#include "gpu_pointer.h"
#include "photo_rates_csi.ANY.h"
#include "static_table_gpu.cuh"

__global__ void PhotoRatesCSIUpdateTableKernel(unsigned int n, const StaticTableGPU<float, 3, 'x'>* dTable,
                                               const PhotoRateTableStretchCSI* dStretch, const float* dSpectralShape,
                                               const Physics::AtomicData::CrossSection* dXS, float norm, int deb)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  const int k = threadIdx.z + blockIdx.z * blockDim.z;

  const int si = dTable->GetSize(0);
  const int sj = dTable->GetSize(1);
  const int sk = dTable->GetSize(2);

  if (k >= sk || j >= sj || i >= si) return;

  const float* csHI   = dXS->cs[Physics::AtomicData::CrossSection::IonizationHI];
  const float* csHeI  = dXS->cs[Physics::AtomicData::CrossSection::IonizationHeI];
  const float* csHeII = dXS->cs[Physics::AtomicData::CrossSection::IonizationHeII];
  // const float* csCVI = dXS->cs[Physics::AtomicData::CrossSection::IonizationCVI];

  const auto* thr = dXS->thresholds;

  float values[7] = {};  // all entries initialized to zero

  const float tauHI   = dStretch->x2tau(dStretch->xMin + dStretch->xBin * i);
  const float tauHeI  = dStretch->x2tau(dStretch->xMin + dStretch->xBin * j);
  const float tauHeII = dStretch->x2tau(dStretch->xMin + dStretch->xBin * k);

  float NeffHI  = (tauHI < 1000 ? tauHI / dXS->csHIatHI : 0);
  float NeffHeI = (tauHeI < 1000 ? (tauHeI - NeffHI * dXS->csHIatHeI) / dXS->csHeIatHeI : 0);
  if (NeffHeI < 0) NeffHeI = 0;
  float NeffHeII =
      (tauHeII < 1000 ? (tauHeII - NeffHI * dXS->csHIatHeII - NeffHeI * dXS->csHeIatHeII) / dXS->csHeIIatHeII : 0);
  if (NeffHeII < 0) NeffHeII = 0;

  if (tauHI < 1000) {
    for (unsigned int l = thr[Physics::AtomicData::CrossSection::IonizationHI].idx;
         l < thr[Physics::AtomicData::CrossSection::IonizationHeI].idx; l++) {
      auto w  = exp(-NeffHI * csHI[l]);
      auto ss = dSpectralShape[l] * w;

      w = csHI[l] * ss;
      values[0] += w;
      values[1] += w * (dXS->hnu_K[l] - Physics::AtomicData::TionHI);
    }
  };

  if (tauHeI < 1000) {
    for (unsigned int l = thr[Physics::AtomicData::CrossSection::IonizationHeI].idx;
         l < thr[Physics::AtomicData::CrossSection::IonizationHeII].idx; l++) {
      auto w  = exp(-NeffHI * csHI[l] - NeffHeI * csHeI[l]);
      auto ss = dSpectralShape[l] * w;

      w = csHI[l] * ss;
      values[0] += w;
      values[1] += w * (dXS->hnu_K[l] - Physics::AtomicData::TionHI);

      if (n > 2) {
        w = csHeI[l] * ss;
        values[2] += w;
        values[3] += w * (dXS->hnu_K[l] - Physics::AtomicData::TionHeI);
      }
    }
  };

  if (tauHeII < 1000) {
    for (unsigned int l = thr[Physics::AtomicData::CrossSection::IonizationHeII].idx; l < dXS->nxi; l++) {
      auto w  = exp(-NeffHI * csHI[l] - NeffHeI * csHeI[l] - NeffHeII * csHeII[l]);
      auto ss = dSpectralShape[l] * w;

      w = csHI[l] * ss;
      values[0] += w;
      values[1] += w * (dXS->hnu_K[l] - Physics::AtomicData::TionHI);

      if (n > 2) {
        w = csHeI[l] * ss;
        values[2] += w;
        values[3] += w * (dXS->hnu_K[l] - Physics::AtomicData::TionHeI);

        w = csHeII[l] * ss;
        values[4] += w;
        values[5] += w * (dXS->hnu_K[l] - Physics::AtomicData::TionHeII);

        // w = csCVI[l]*ss;
        // values[6] += w;
      }
    }
  };

  // The recasting of dTable->GetFullData() here has been flagged as dangerous.
  // The PhotoRatesCSIUpdateTableKernel() function should be refactored to avoid this.
  // Matthew suggests: "The more correct way to fix this: is to pass a
  // StaticTableGPU<float, 3, 'x'> * to this function rather than a
  // const StaticTableGPU<float, 3, 'x'> * and to add a new member function to the
  // TableBase struct. It would probably have the form
  // DEVICE_LOCAL_DECL inline value_t* GetFullData() { return mYs; }
  // This member function would exist alongside the existing
  // const-qualified member of the same name."
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto out = const_cast<float*>(dTable->GetFullData()) + dTable->Lidx(i + si * (j + sj * k), 0);
  for (int m = 0; m < n; m++) {
    out[m] = values[m] * norm;
    /// if (out[m] > 1.0e-4) printf("**FT** %d %g\n", m, out[m]);
  }
}

// NOLINTBEGIN(readability-identifier-naming)
namespace PhotoRatesCSI
{
// NOLINTEND(readability-identifier-naming)
StaticTableGPU<float, 3, 'x'>* CreateTable(float* data, unsigned int numRates, const PhotoRateTableStretchCSI& stretch)
{
  StaticTableGPU<float, 3, 'x'>* d;
  GPU_Error_Check(cudaMalloc(&d, sizeof(StaticTableGPU<float, 3, 'x'>)));
  if (d != nullptr) {
    StaticTableGPU<float, 3, 'x'> h(data, numRates, stretch.size, stretch.xMin, stretch.xMax);
    GPU_Error_Check(cudaMemcpy(d, &h, sizeof(StaticTableGPU<float, 3, 'x'>), cudaMemcpyHostToDevice));
  }
  return d;
}

void DeleteTable(StaticTableGPU<float, 3, 'x'>* d) { GPU_Error_Check(cudaFree(d)); }

void UpdateTable(unsigned int size, unsigned int numRates, const StaticTableGPU<float, 3, 'x'>* dTable,
                 const PhotoRateTableStretchCSI* dStretch, const float* dSpectralShape,
                 const Physics::AtomicData::CrossSection* dXS, float norm, int deb)
{
  int numBlocks = (size + 7) / 8;
  dim3 threads(8, 8, 8);
  dim3 blocks(numBlocks, numBlocks, numBlocks);

  hipLaunchKernelGGL(PhotoRatesCSIUpdateTableKernel, blocks, threads, 0, 0, numRates, dTable, dStretch, dSpectralShape,
                     dXS, norm, deb);
}
};  // namespace PhotoRatesCSI

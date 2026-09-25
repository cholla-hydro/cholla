/*LICENSE*/

#include "photo_rates_csi_gpu.h"

#include <cstring>

#include "atomic_data.h"
#include "gpu_pointer.h"
#include "photo_rates_csi.ANY.h"

template <typename value_t, unsigned int N, char Mode>
class StaticTableGPU;

namespace rt_photo_rates_csi
{
StaticTableGPU<float, 3, 'x'>* CreateTable(float* data, unsigned int numRates, const PhotoRateTableStretchCSI& stretch);
void DeleteTable(StaticTableGPU<float, 3, 'x'>* d);

void UpdateTable(unsigned int size, unsigned int numRates, const StaticTableGPU<float, 3, 'x'>* dTable,
                 const PhotoRateTableStretchCSI* dStretch, const float* dSpectralShape,
                 const rt_physics::rt_atomic_data::CrossSection* dXS, float norm, int deb = 0);
};  // namespace rt_photo_rates_csi

rt_photo_rates_csi::TableWrapperGPU::TableWrapperGPU(unsigned int numRadsPerFreq_, unsigned int numRates_)
{
  numRadsPerFreq = numRadsPerFreq_;
  numRates       = numRates_;

  dTableData.resize(numRadsPerFreq);
  bSpectralShapes.resize(numRadsPerFreq);

  PhotoRateTableStretchCSI stretch;
  //
  //  Table precision (calibrated with ALTAIR):
  //    1.0% -> 96
  //    1.5% -> 64
  //    2.0% -> 40
  //
  stretch.Set(64);

  bTables.Alloc(numRadsPerFreq);
  for (unsigned int rad = 0; rad < numRadsPerFreq; rad++) {
    bSpectralShapes[rad].Alloc(rt_physics::rt_atomic_data::CrossSections()->nxi);

    dTableData[rad].Alloc(numRates * stretch.size * stretch.size * stretch.size);
    bTables[rad] = CreateTable(dTableData[rad].Ptr(), numRates, stretch);
  }
  bTables.BlockingTransferToDevice();

  bStretch.Alloc(1);
  memcpy(bStretch.HostPtr(), &stretch, sizeof(PhotoRateTableStretchCSI));
  bStretch.BlockingTransferToDevice();
}

rt_photo_rates_csi::TableWrapperGPU::~TableWrapperGPU()
{
  for (unsigned int rad = 0; rad < numRadsPerFreq; rad++) {
    DeleteTable(bTables[rad]);
  }
}

void rt_photo_rates_csi::TableWrapperGPU::Update(unsigned int rad, const float* spectralShape, float norm)
{
  memcpy(bSpectralShapes[rad].HostPtr(), spectralShape, sizeof(float) * bSpectralShapes[rad].Count());

  bSpectralShapes[rad].BlockingTransferToDevice();
  UpdateTable(bStretch.HostPtr()->size, numRates, bTables[rad], bStretch.DevicePtr(), bSpectralShapes[rad].DevicePtr(),
              rt_physics::rt_atomic_data::CrossSectionsGPU(), norm);
  cudaStreamSynchronize(0);
}

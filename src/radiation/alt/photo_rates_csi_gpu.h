/*LICENSE*/

#ifndef PHYSICS_RT_PHOTO_RATES_CSI_GPU_H
#define PHYSICS_RT_PHOTO_RATES_CSI_GPU_H

#include <deque>

#include "gpu_pointer.h"
#include "photo_rates_csi.ANY.h"
#include "static_table_gpu.h"

namespace PhotoRatesCSI
{
//
//  Object for GPUShared: a copy of all tables on a single hardware device
//
struct TableWrapperGPU {
  TableWrapperGPU(unsigned int numRadsPerFreq, unsigned int numRates);
  ~TableWrapperGPU();

  void Update(unsigned int rad, const float* spectralShape, float norm);

  GPU::TransferBuffers<StaticTableGPU<float, 3, 'x'>*> bTables;
  std::deque<GPU::DeviceBuffer<float>> dTableData;
  std::deque<GPU::TransferBuffers<float>> bSpectralShapes;
  GPU::TransferBuffers<PhotoRateTableStretchCSI> bStretch;
  unsigned int numRadsPerFreq = 0;
  unsigned int numRates       = 0;
};
};  // namespace PhotoRatesCSI

#endif  // PHYSICS_RT_PHOTO_RATES_CSI_GPU_H

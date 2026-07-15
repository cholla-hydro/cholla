/*LICENSE*/

#ifndef PHYSICS_RT_PHOTO_RATES_CSI_GPU_H
#define PHYSICS_RT_PHOTO_RATES_CSI_GPU_H

#include <deque>

#include "gpu_pointer.h"
#include "photo_rates_csi.ANY.h"
#include "static_table_gpu.h"

namespace rt_photo_rates_csi
{
//
//  Object for GPUShared: a copy of all tables on a single hardware device
//
struct TableWrapperGPU {
  TableWrapperGPU(unsigned int numRadsPerFreq, unsigned int numRates);
  ~TableWrapperGPU();

  void Update(unsigned int rad, const float* spectralShape, float norm);

  rt_gpu::TransferBuffers<StaticTableGPU<float, 3, 'x'>*> bTables;
  std::deque<rt_gpu::DeviceBuffer<float>> dTableData;
  std::deque<rt_gpu::TransferBuffers<float>> bSpectralShapes;
  rt_gpu::TransferBuffers<PhotoRateTableStretchCSI> bStretch;
  unsigned int numRadsPerFreq = 0;
  unsigned int numRates       = 0;
};
};  // namespace rt_photo_rates_csi

#endif  // PHYSICS_RT_PHOTO_RATES_CSI_GPU_H

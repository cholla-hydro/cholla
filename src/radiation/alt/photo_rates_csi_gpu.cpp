/*LICENSE*/


#include "gpu_pointer.h"

#include "atomic_data.h"

#include "photo_rates_csi.ANY.h"
#include "photo_rates_csi_gpu.h"


template<typename value_t, unsigned int N, char Mode> class StaticTableGPU;


namespace PhotoRatesCSI
{
    StaticTableGPU<float,3,'x'>* CreateTable(float* data, unsigned int numRates, const PhotoRateTableStretchCSI& stretch);
    void DeleteTable(StaticTableGPU<float,3,'x'>* d);

    void UpdateTable(unsigned int size, unsigned int numRates, const StaticTableGPU<float,3,'x'>* dTable, const PhotoRateTableStretchCSI* dStretch, const float* dSpectralShape, const Physics::AtomicData::CrossSection* dXS, float norm, int deb = 0);
};


PhotoRatesCSI::TableWrapperGPU::TableWrapperGPU(unsigned int numRadsPerFreq_, unsigned int numRates_, const PhotoRateTableStretchCSI& stretch)
{
    numRadsPerFreq = numRadsPerFreq_;
    numRates = numRates_;

    dTableData.resize(numRadsPerFreq);
    bSpectralShapes.resize(numRadsPerFreq);

    bTables.Alloc(numRadsPerFreq);
    for(unsigned int rad=0; rad<numRadsPerFreq; rad++)
    {
        bSpectralShapes[rad].Alloc(Physics::AtomicData::CrossSections()->nxi);

        dTableData[rad].Alloc(numRates*stretch.size*stretch.size*stretch.size);
        bTables[rad] = CreateTable(dTableData[rad].Ptr(),numRates,stretch);
    }
    bTables.BlockingTransferToDevice();

    bStretch.Alloc(1);
    memcpy(bStretch.HostPtr(),&stretch,sizeof(PhotoRateTableStretchCSI));
    bStretch.BlockingTransferToDevice();
}


PhotoRatesCSI::TableWrapperGPU::~TableWrapperGPU()
{
    for(unsigned int rad=0; rad<numRadsPerFreq; rad++)
    {
        DeleteTable(bTables[rad]);
    }
}


void PhotoRatesCSI::TableWrapperGPU::Update(unsigned int rad, const float* spectralShape, float norm)
{
    memcpy(bSpectralShapes[rad].HostPtr(),spectralShape,sizeof(float)*bSpectralShapes[rad].Count());

    bSpectralShapes[rad].BlockingTransferToDevice();
    UpdateTable(bStretch.HostPtr()->size,numRates,bTables[rad],bStretch.DevicePtr(),bSpectralShapes[rad].DevicePtr(),Physics::AtomicData::CrossSectionsGPU(),norm);
}
  

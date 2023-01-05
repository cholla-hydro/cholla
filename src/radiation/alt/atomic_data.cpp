/*LICENSE*/


#include "atomic_data.h"
#include "constant.h"
#include "gpu_pointer.h"
#include "static_table.h"


namespace
{
    struct AtomicDataOnGPU
    {
        AtomicDataOnGPU()
        {
            auto xs = Physics::AtomicData::CrossSections();

            for(unsigned int k=0; k<3+Physics::AtomicData::CrossSection::Num; k++)
            {
                dArrs[k].Alloc(sizeof(float)*xs->nxi);
            }

            GPU::HostBuffer<float> h;
            h.Alloc(xs->nxi);
            memcpy(h.Ptr(),xs->xi,sizeof(float)*xs->nxi);
            h.BlockingTransferToDevice(dArrs[0]);
            memcpy(h.Ptr(),xs->hnu_K,sizeof(float)*xs->nxi);
            h.BlockingTransferToDevice(dArrs[1]);
            memcpy(h.Ptr(),xs->hnu_eV,sizeof(float)*xs->nxi);
            h.BlockingTransferToDevice(dArrs[2]);
            for(unsigned int k=0; k<Physics::AtomicData::CrossSection::Num; k++)
            {
                memcpy(h.Ptr(),xs->cs[k],sizeof(float)*xs->nxi);
                h.BlockingTransferToDevice(dArrs[3+k]);
            }
            h.Free();

            GPU::HostBuffer<Physics::AtomicData::CrossSection> hxs(1);
            *hxs.Ptr() = *xs;

            hxs.Ptr()->xi = dArrs[0];
            hxs.Ptr()->hnu_K = dArrs[1];
            hxs.Ptr()->hnu_eV = dArrs[2];
            for(unsigned int k=0; k<Physics::AtomicData::CrossSection::Num; k++)
            {
                hxs.Ptr()->cs[k] = dArrs[3+k];
            }

            dCrossSection.Alloc(1);
            hxs.BlockingTransferToDevice(dCrossSection);
        }

        ~AtomicDataOnGPU()
        {
            dCrossSection.Free();

            for(unsigned int k=0; k<3+Physics::AtomicData::CrossSection::Num; k++)
            {
                dArrs[k].Free();
            }
        }

        GPU::DeviceBuffer<Physics::AtomicData::CrossSection> dCrossSection;
        GPU::DeviceBuffer<float> dArrs[Physics::AtomicData::CrossSection::Num+3];
    };

    void CrossSectionBuilder(unsigned int num, float* hnu, float** cs);

    Physics::AtomicData::CrossSection gCrossSections;
    AtomicDataOnGPU gAtomicDataOnGPU;
};


const Physics::AtomicData::CrossSection* Physics::AtomicData::CrossSections()
{
    return &gCrossSections;
}

const Physics::AtomicData::CrossSection* Physics::AtomicData::CrossSectionsGPU()
{
    return gAtomicDataOnGPU.dCrossSection;
}

using namespace Physics::AtomicData;


void Physics::AtomicData::Create()
{
    gCrossSections.nxi = 300;
    gCrossSections.xiMin = -1;
    gCrossSections.dxi = 0.05f; // 20 points per e-folding
    gCrossSections.xiMax = gCrossSections.xiMin + gCrossSections.nxi*gCrossSections.dxi;

    gCrossSections.xi = new float[gCrossSections.nxi];
    gCrossSections.hnu_K = new float[gCrossSections.nxi];
    gCrossSections.hnu_eV = new float[gCrossSections.nxi];

    for(unsigned int i=0; i<gCrossSections.nxi; i++)
    {
        gCrossSections.xi[i] = gCrossSections.xiMin + gCrossSections.dxi*(i+0.5f);
        gCrossSections.hnu_K[i] = TionHI*exp(gCrossSections.xi[i]);
        gCrossSections.hnu_eV[i] = Ry_eV*exp(gCrossSections.xi[i]);
    }

    for(unsigned int k=0; k<CrossSection::Num; k++)
    {
        gCrossSections.cs[k] = new float[gCrossSections.nxi];
    }

    CrossSectionBuilder(gCrossSections.nxi,gCrossSections.hnu_eV,gCrossSections.cs);

    for(unsigned int k=0; k<CrossSection::Num; k++)
    {  
        gCrossSections.thresholds[k].idx = -1;
        for(unsigned int i=0; i<gCrossSections.nxi; i++)
        {
            if(gCrossSections.cs[k][i] > 0)
            {
                gCrossSections.thresholds[k].idx = i;
                gCrossSections.thresholds[k].xi = gCrossSections.xi[i];
                gCrossSections.thresholds[k].hnu_K = gCrossSections.hnu_K[i];
                gCrossSections.thresholds[k].hnu_eV = gCrossSections.hnu_eV[i];
                gCrossSections.thresholds[k].cs = gCrossSections.cs[k][i];
                break;
            }
        }
    }

    gCrossSections.csHIatHI = gCrossSections.cs[CrossSection::IonizationHI][gCrossSections.thresholds[CrossSection::IonizationHI].idx];
    gCrossSections.csHIatHeI = gCrossSections.cs[CrossSection::IonizationHI][gCrossSections.thresholds[CrossSection::IonizationHeI].idx];
    gCrossSections.csHIatHeII = gCrossSections.cs[CrossSection::IonizationHI][gCrossSections.thresholds[CrossSection::IonizationHeII].idx];
    gCrossSections.csHeIatHeI = gCrossSections.cs[CrossSection::IonizationHeI][gCrossSections.thresholds[CrossSection::IonizationHeI].idx];
    gCrossSections.csHeIatHeII = gCrossSections.cs[CrossSection::IonizationHeI][gCrossSections.thresholds[CrossSection::IonizationHeII].idx];
    gCrossSections.csHeIIatHeII = gCrossSections.cs[CrossSection::IonizationHeII][gCrossSections.thresholds[CrossSection::IonizationHeII].idx];
}


void Physics::AtomicData::Delete()
{
    for(unsigned int k=0; k<CrossSection::Num; k++)
    {
        delete [] gCrossSections.cs[k];
    }

    delete [] gCrossSections.xi;
    delete [] gCrossSections.hnu_K;
    delete [] gCrossSections.hnu_eV;
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
        static auto csfit = [](double E, double cs0, double E0, double y0, double y1, double yw, double ya, double p)
        {
            auto x = E/E0 - y0;
            auto y = sqrt(x*x+y1*y1);
            return cs0*(pow(x-1,2)+yw*yw)*pow(y,0.5*p-5.5)/pow(1+sqrt(y/ya),p)/1.0e-24;
        };

        for(unsigned int i=0; i<num; i++)
        {
            double E = hnu_eV[i];
            cs[CrossSection::IonizationHI][i] = (E<Ry_eV ? 0 : csfit(E,5.475e-14,4.298e-01,0.0,0.0,0.0,3.288e+01,2.963));
            cs[CrossSection::IonizationHeI][i] = (E<Ry_eV*TionHeI/TionHI ? 0 : csfit(E,9.492e-16,1.361e+01,4.434e-01,2.136,2.039,1.469,3.188));
            cs[CrossSection::IonizationHeII][i] = (E<Ry_eV*TionHeII/TionHI ? 0 : csfit(E,1.369e-14,1.720,0.0,0.0,0.0,3.288e+01,2.963));
            cs[CrossSection::IonizationCVI][i] = (E<490 ? 0 : csfit(E,1.521e-15,15.48,0.0,0.0,0.0,3.288e+01,2.963));
        }
    }
};

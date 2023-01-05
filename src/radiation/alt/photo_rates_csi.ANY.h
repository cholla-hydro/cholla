/*LICENSE*/

#ifndef PHYSICS_RT_PHOTO_RATES_CSI_ANY_H
#define PHYSICS_RT_PHOTO_RATES_CSI_ANY_H


#include <cmath>

#include "decl.h"


//
//  Functions for interpolating between the table variable x and the optical depth tau
//
struct DEVICE_ALIGN_DECL PhotoRateTableStretchCSI
{
    void Set(unsigned int size_, float tauLow_ = 3, float tauMax_ = 20)
    {
        size = size_;
        tauLow = tauLow_;
        tauMax = tauMax_;

        xMax = log(1+tauMax/tauLow);
        xBin = xMax/(size-2);
        xInf = -0.5f*xBin;
        xMin = -xBin;
    }

    //
    //  Table stretch:
    //  The first point with negative x is a special case - infinite tau (which is represented by a value of tau = 1001).
    //
    DEVICE_LOCAL_DECL inline float tau2x(float tau) const { return (tau<1000 ? log(1+tau/tauLow) : -xBin); }
    DEVICE_LOCAL_DECL inline float x2tau(float x) const { return (x>xInf ? tauLow*(exp((x>0 ? x : 0))-1) : 1001); }

    unsigned int size;
    float tauLow, tauMax;
    float xBin, xMin, xMax, xInf;
};

#endif // PHYSICS_RT_PHOTO_RATES_CSI_ANY_H

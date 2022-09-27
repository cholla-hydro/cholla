#ifdef PARIS

#ifndef FFT_H
#define FFT_H

#include "../global/global.h"
#include "../gravity/paris/HenryPeriodic.hpp"


class FFT_3D
{
  public:
    FFT_3D();
    ~FFT_3D();
    void Initialize(Real lx, Real ly, Real lz, Real xMin, Real yMin, Real zMin, int nx, int ny, int nz, int nxReal, int nyReal, int nzReal, Real dx, Real dy, Real dz);
    void Reset();
    
    void Filter_inv_k2( double *input, double *output, bool in_device ) const; 

    void Filter_rescale_by_power_spectrum( double *input, double *output, bool in_device, int size, double *dev_k, double *dev_pk, Real dx3 ) const;
     
    void Filter_rescale_by_k_k2( double *input, double *output, bool in_device, int direction, double D ) const;
 
    
  protected:
    int dn_[3];
    Real dr_[3],lo_[3],lr_[3],myLo_[3];    
    long minBytes_;
    long inputBytes_;
    long outputBytes_;
    Real *da_;
    Real *db_;
    
  private:
    int ni_,nj_,nk_; //!< Number of elements in X, Y, and Z dimensions
    double ddi_,ddj_,ddk_; //!< Frequency-independent terms for the filter
    HenryPeriodic *henry_; //!< FFT filter object 
};

#endif  
#endif
#pragma once
#if defined(PARIS) && defined(FFT) 

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
    
    void Filter_inv_k2( Real *input, Real *output, bool in_device ) const; 

    void Filter_rescale_by_power_spectrum( Real *input, Real *output, bool in_device, int size, Real *dev_k, Real *dev_pk ) const;
     
    void Filter_rescale_by_k_k2( Real *input, Real *output, bool in_device, int direction, Real D ) const;

    // An identity filter that does nothing, with Paris-like calling API
    void Filter_identity( Real *const input, Real *output, bool in_device ) const;

    // An filter that simply rescales the grid in Fourier space, with Paris-like calling API
    //void Filter_rescale( Real *const input, Real A, Real *const output, bool in_device ) const;
    void Filter_rescale( Real *const input, Real A, Real *output, bool in_device ) const;
    
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
    Real ddi_,ddj_,ddk_; //!< Frequency-independent terms for the filter
    HenryPeriodic *henry_; //!< FFT filter object 
};

#endif  
#endif

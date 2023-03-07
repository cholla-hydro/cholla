/*LICENSE*/

#ifndef TABLE_BASE_ANY_H
#define TABLE_BASE_ANY_H


//
//  A "universal function" base class with generic any-D table functionality.
//
#include <cmath>

#include "decl.h"


template<typename value_t, unsigned int N, bool VarFirst> struct DEVICE_ALIGN_DECL TableBase
{

    static_assert(N != 0,"Zero-dimensional table is not allowed.\n");
    static_assert(N <= 4,"Table with dimension above 3 is not yet implemented.\n");

    static constexpr bool kIsVarFirst = VarFirst;

    //
    //  Table properties
    //
    DEVICE_LOCAL_DECL inline unsigned int NumVars() const { return mNumVars; }
    DEVICE_LOCAL_DECL inline value_t GetXmin(unsigned int n) const { if(n < N) return mXmin[n]; else return 0; }
    DEVICE_LOCAL_DECL inline value_t GetXmax(unsigned int n) const { if(n < N) return mXmax[n]; else return 0; }
    DEVICE_LOCAL_DECL inline value_t GetXbin(unsigned int n) const { if(n < N) return mXbin[n]; else return 0; }
    DEVICE_LOCAL_DECL inline unsigned int GetSize(unsigned int n) const { if(n < N) return mSize[n]; else return 0; }

    //
    //  The whole data array may be needed for integration or transfer to devices
    //
    DEVICE_LOCAL_DECL inline const value_t* GetFullData() const { return mYs; }
    DEVICE_LOCAL_DECL inline unsigned int GetFullDataCount() const { return mNumVars*mVolume; }

    //
    //  Index transform.
    //
    DEVICE_LOCAL_DECL inline unsigned int Lidx(unsigned int idx, unsigned int var) const
    {
        return (VarFirst ? (var+mNumVars*idx) : (idx+mVolume*var));
    }

protected:

    //
    //  Sample the table
    //
    DEVICE_LOCAL_DECL inline bool GetValuesImpl(const value_t x[N], value_t* ys, unsigned int begin, unsigned int end) const
    {
        if(begin>=end || end>mNumVars) return false;
       
        unsigned int ijk[N];
        value_t w0[N], w1[N];

        bool ret = this->Localize(x,ijk,w0,w1);

        this->Interpolate(ijk,w0,w1,ys,begin,end);
        
        return ret;
    }

    //
    //  Sampling helpers
    //
    DEVICE_LOCAL_DECL inline bool Localize(const value_t x[N], unsigned int ijk[N], value_t w0[N], value_t w1[N]) const
    {
        bool ret = true;
        for(unsigned int n=0; n<N; n++)
        {
            auto x1 = mXfac[n]*(x[n]-mXmin[n]); // just in case to avoid weird round-off errors
            if(x1 <= 0) // 0 is exactly representable
            {
                ijk[n] = 0;
                w0[n] = 1;
                w1[n] = 0;
                ret = false;
            }
            else if(x1 >= 1) // 1 is exactly representable
            {
                ijk[n] = mLast[n] - 1;
                w0[n] = 0;
                w1[n] = 1;
                ret = false;
            }
            else
            {
                ijk[n] = static_cast<unsigned int>(mLast[n]*x1);
                w1[n] = mLast[n]*x1 - ijk[n];
                w0[n] = 1 - w1[n];
            }
        }
        return ret;
    }

    DEVICE_LOCAL_DECL inline void Interpolate(const unsigned int ijk[N], const value_t w0[N], const value_t w1[N], value_t* ys, unsigned int begin, unsigned int end) const
    {
        for(unsigned int var=begin; var<end; var++)
        {
            if(N == 1)
            {
                ys[var-begin] = 
                    mYs[Lidx(ijk[0]+0,var)]*w0[0] +
                    mYs[Lidx(ijk[0]+1,var)]*w1[0];
            }
            else if(N == 2)
            {
                ys[var-begin] =
                    (mYs[Lidx(ijk[0]+0+mSize[0]*(ijk[1]+0),var)]*w0[0] +
                     mYs[Lidx(ijk[0]+1+mSize[0]*(ijk[1]+0),var)]*w1[0])*w0[1] +
                    (mYs[Lidx(ijk[0]+0+mSize[0]*(ijk[1]+1),var)]*w0[0] +
                     mYs[Lidx(ijk[0]+1+mSize[0]*(ijk[1]+1),var)]*w1[0])*w1[1];
            }
            else if(N == 3)
            {
                ys[var-begin] = 
                    ((mYs[Lidx(ijk[0]+0+mSize[0]*(ijk[1]+0+mSize[1]*(ijk[2]+0)),var)]*w0[0] +
                      mYs[Lidx(ijk[0]+1+mSize[0]*(ijk[1]+0+mSize[1]*(ijk[2]+0)),var)]*w1[0])*w0[1] +
                     (mYs[Lidx(ijk[0]+0+mSize[0]*(ijk[1]+1+mSize[1]*(ijk[2]+0)),var)]*w0[0] +
                      mYs[Lidx(ijk[0]+1+mSize[0]*(ijk[1]+1+mSize[1]*(ijk[2]+0)),var)]*w1[0])*w1[1])*w0[2] +
                    ((mYs[Lidx(ijk[0]+0+mSize[0]*(ijk[1]+0+mSize[1]*(ijk[2]+1)),var)]*w0[0] +
                      mYs[Lidx(ijk[0]+1+mSize[0]*(ijk[1]+0+mSize[1]*(ijk[2]+1)),var)]*w1[0])*w0[1] +
                     (mYs[Lidx(ijk[0]+0+mSize[0]*(ijk[1]+1+mSize[1]*(ijk[2]+1)),var)]*w0[0] +
                      mYs[Lidx(ijk[0]+1+mSize[0]*(ijk[1]+1+mSize[1]*(ijk[2]+1)),var)]*w1[0])*w1[1])*w1[2];
            }
        }
    }

    //
    //  It may not be possible to set the table in the constructor, keep max flexibility
    //  Data are arranged as (var,i,j,k)
    //
    void Set(value_t* ys, unsigned int numVars, const unsigned int size[N], const value_t xmin[N], const value_t xmax[N])
    {
        mYs = ys;
        mNumVars = numVars;

        mVolume = 1;
        for(unsigned int n=0; n<N; n++)
        {
            mVolume *= size[n];
            mSize[n] = size[n];
            mLast[n] = (size[n]>1 ? size[n]-1 : 1); 
            mXmin[n] = xmin[n];
            mXmax[n] = xmax[n];
            auto xwid = xmax[n] - xmin[n];
            mXbin[n] = (xwid>0 ? xwid/mLast[n] : 0);
            mXfac[n] = (xwid>0 ? 1/xwid : 0);
        }
    }

    void Reset()
    {
        this->mYs = nullptr;
        for(unsigned int n=0; n<N; n++)
        {
            this->mXmin[n] = this->mXmax[n] = this->mXbin[n] = this->mXfac[n] = 0;
            this->mLast[n] = -1;
            this->mSize[n] = 0;
        }
        this->mNumVars = 0;
    }

    value_t *mYs;
    unsigned int mNumVars, mVolume;
    unsigned int mSize[N], mLast[N];
    value_t mXbin[N], mXfac[N], mXmin[N], mXmax[N];
};

#endif // TABLE_BASE_ANY_H

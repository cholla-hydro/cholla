/*LICENSE*/

#ifndef STATIC_TABLE_H
#define STATIC_TABLE_H

#include <functional>

#include "table_base.h"


//
//  Linearly interpolated table
//
template<typename value_t, unsigned int N> class StaticTable : public TableBase<value_t,N,true>
{

public:

    using Builder = std::function<void(const value_t[N],value_t*)>; // numVars, x[N] in, ys[numVars] out

    inline bool IsBuilt() const { return (this->mYs != nullptr); }

    inline bool GetValues(const value_t x[N], value_t* ys, unsigned int begin, unsigned int end) const { ASSERT(ys != nullptr); return this->GetValuesImpl(x,ys,begin,end); }

    void Build(const Builder& builder, unsigned int numVars, const unsigned int size[N], const value_t xmin[N], const value_t xmax[N]);
    void Build(const Builder& builder, unsigned int numVars, unsigned int size, value_t xmin, value_t xmax);
    void Clear();

    StaticTable();
    StaticTable(const Builder& builder, unsigned int numVars, const unsigned int size[N], const value_t xmin[N], const value_t xmax[N]);
    StaticTable(const Builder& builder, unsigned int numVars, unsigned int size, value_t xmin, value_t xmax);
    
    ~StaticTable() { this->Clear(); }
};


//
//  Implementations
//
template<typename value_t, unsigned int N> StaticTable<value_t,N>::StaticTable()
{
     this->Reset();
}


template<typename value_t, unsigned int N> inline StaticTable<value_t,N>::StaticTable(const Builder& builder, unsigned int numVars, const unsigned int size[N], const value_t xmin[N], const value_t xmax[N])
{
    this->mYs = nullptr;
    this->Build(builder,numVars,size,xmin,xmax);
}


template<typename value_t, unsigned int N> inline StaticTable<value_t,N>::StaticTable(const Builder& builder, unsigned int numVars, unsigned int size, value_t xmin, value_t xmax)
{
    this->mYs = nullptr;
    this->Build(builder,numVars,size,xmin,xmax);
}


template<typename value_t, unsigned int N> inline void StaticTable<value_t,N>::Clear()
{
    if(this->mYs != nullptr)
    {
        delete [] this->mYs;
        this->Reset();
    }
}


template<typename value_t, unsigned int N> inline void StaticTable<value_t,N>::Build(const Builder& builder, unsigned int numVars, const unsigned int size[N], const value_t xmin[N], const value_t xmax[N])
{
    unsigned int vol = 1;
    for(unsigned int n=0; n<N; n++)
    {
        vol *= size[n];
    }

    auto ys = new value_t[numVars*vol];
    this->Set(ys,numVars,size,xmin,xmax);

    value_t x[N];
    value_t *y = new value_t[numVars];
    for(unsigned int idx=0; idx<vol; idx++)
    {
        unsigned int l = idx;
        for(unsigned int n=0; n<N; n++)
        {
            x[n] = this->mXmin[n] + this->mXbin[n]*(l % size[n]);
            l /= size[n];
        }

        builder(x,y);

        for(unsigned int var=0; var<numVars; var++)
        {
            ys[this->Lidx(idx,var)] = y[var];
        }
    }
    delete [] y;
}


template<typename value_t, unsigned int N> inline void StaticTable<value_t,N>::Build(const Builder& builder, unsigned int numVars, unsigned int size, value_t xmin, value_t xmax)
{
    unsigned int size1[N];
    value_t xmin1[N], xmax1[N];
    for(unsigned int n=0; n<N; n++)
    {
        size1[n] = size;
        xmin1[n] = xmin;
        xmax1[n] = xmax;
    }

    this->Build(builder,numVars,size1,xmin1,xmax1);
}

#endif // STATIC_TABLE_H


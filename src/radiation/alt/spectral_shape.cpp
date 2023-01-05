/*LICENSE*/
#include "common.h"


#include <cmath>

#include "physics/atomic_data/atomic_data.h"
#include "spectral_shape.h"


using namespace Physics::RT;
using namespace Physics::AtomicData;


void SpectralShape::Normalize(STL::Vector<float>& s)
{
    auto xs = CrossSections();

    float w = 0;
    for(unsigned int i=xs->thresholds[CrossSection::IonizationHI].idx; i<xs->nxi; i++)
    {
        w += s[i];
    }
    ASSERT(w > 0);
    w *= xs->dxi;
    for(unsigned int i=0; i<xs->nxi; i++)
    {
        s[i] /= w;
    }
}


//
//  Actual shapes
//  ------------------
//
//  PopII and PopIII shapes from Ricotti et al 2002ApJ...575...33R.
//
void SpectralShape::PopII(STL::Vector<float>& s)
{
    auto xs = CrossSections();

    s.assign(xs->nxi,0);
    for(unsigned int i=0; i<xs->nxi; i++)
    {
        float e = xs->hnu_eV[i]/Ry_eV;
    
        if(e < 1)
        {
            s[i] = 5.5;
        }
        else if(e < 2.5)
        {
            s[i] = 1.0/pow(e,1.8);
        }
        else if(e < 4)
        {
            s[i] = 1.0/(2.5*pow(e,1.8));
        }
        else if(e < 100)
        {
            s[i] = 2.0e-3*pow(e,3)/(exp(e/1.4)-1);
        }
    }

    Normalize(s);
}


void SpectralShape::PopIII(STL::Vector<float>& s)
{
    auto xs = CrossSections();

    s.assign(xs->nxi,0);
    for(unsigned int i=0; i<xs->nxi; i++)
    {
        float e = xs->hnu_eV[i]/Ry_eV;

        if(e < 1)
        {
            s[i] = 2.5*pow(e,1.4);
        }
        else if(e < 4)
        {
            s[i] = exp(-4.0/e)/exp(-4.0)/pow(e,2);
        }
        else
        {
            s[i] = 2.0e1*exp(-0.25*e)/pow(e,2);
        }
    }

    Normalize(s);
}


//
//  Average QSO spectral shape from Richards et al 2006ApJS..166..470R and Hopkins et al 2007, ApJ, 654, 731
//  (i.e. NG's fit to it).
//
void SpectralShape::Quasar(STL::Vector<float>& s)
{
    auto xs = CrossSections();

    s.assign(xs->nxi,0);
    for(unsigned int i=0; i<xs->nxi; i++)
    {
        s[i] = 9.5e-5/pow(1+(xs->hnu_eV[i]/300),0.8)*exp(-xs->hnu_eV[i]/5.0e5) + 0.1*pow(10/xs->hnu_eV[i],2.2)/pow(1+pow(9/xs->hnu_eV[i],5),0.4) + 5.8*exp(-xs->hnu_eV[i]/0.2);
    }

    Normalize(s);
}


//
//  Black body spectrum.
//
void SpectralShape::BlackBody(float tem, STL::Vector<float>& s)
{
    ASSERT(tem > 1000);

    auto xs = CrossSections();

    s.assign(xs->nxi,0);
    for(unsigned int i=0; i<xs->nxi; i++)
    {
        float x = xs->hnu_K[i]/tem;
        if(x < 70)
        {
            s[i] = pow(x,3)/(exp(x)-1); // n_xi is per xi=log(nu), hence x^3
        }
    }

    Normalize(s);
}

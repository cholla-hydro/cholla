/*! \file RT_functions.cu
 #  \brief Definitions of functions for the RT solver */


#ifdef CUDA
#ifdef RT

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"../utils/gpu.hpp"
#include"../global/global.h"
#include"../global/global_cuda.h"
#include"RT_functions.h"

void rtBoundaries(Real *dev_scalar, Real *rtFields)
{


}

// CPU function that will call the GPU-based RT functions
void rtSolve(Real *dev_scalar)
{
/*

INTRO:

  Radiation field at each frequency is represented with 2 fields:
  "near" field g (from soures inside the box) and "far" field f
  (from sources outside the box). They are combined as

  J = \bar{J} f + L (g - \bar{g} f),  \bar{f} = 1

  where \bar{J} is cosmic background and L is the frequency dependence of
  sources (c.f. stellar spectrum). 

  This is done so that one can account for cosmological effects in \bar{J},
  while having radiation near sources being shaped by the source spectrum.

  One can also consider an approximation f=0 and only track one field per
  frequency, although the only use case for that limit is modeling SF inside
  GMCs.

  Reference: Gnedin_2014_ApJ_793_29

GIVEN:

  1) abundance fields as mass density \rhoHI, \rhoHeI, \rhoHeII
  2) radiation source field \rs
  3) optically thin near radiation field \ot ("0-frequency field", 0-frequency far field is just idenstically 1)

     \ot = \frac{1}{4\pi} int d^3 x_1 \frac{\rs(x_1)}{(x-x_1)^2}

  4) 6 components of the near Eddington tensor \et^{ij} (far Eddington tensor is a unit matrix over 3).

     \ot \et^{ij} = \frac{1}{4\pi} int d^3 x_1 \frac{rs(x_1)(x^i-x_1^i)(x^j-x_1^j)}{(x-x_1)^4}

  5) near and far radiation fields per frequency, \rfn, \rff

  6) 7 temporary storage fields \temp[1:7] (can be reduced to 4 with extra
  calculations)

  7) boundary data for all fields onan 18-point stencil (cube minus vertices)

ALGORITHM:

  loop over iterations: at each iteration

  loop over frequencies: for each frequency \f:

    1) compute the dimensionless absorption coeffcient\abc:

       \abc = \csFact*(\csHI*\rhoHI+\csHeI*\rhoHeI+\csHeII*\rhoHeII)

    where \cs... are cross sections for 3 species at frequency \f (some
    could be zero) and

       \csFact = <unit-conversion-factor>*<cell-size>/<baryon-mass>

    ** uses \temp[0] extra storage for \abc
    ** runs on a separate CUDA kernel

    2) compute edge absortion coefficients \abch and fluxes \flux:

       \abchX[i+1/2,j,k] = epsNum + 0.5*(\abc[i,j,k]+abc[i+1,j,k])
       ...

       \fluxX[i+1/2,j,k) = (ux+uy+uz)/\abch[i+1/2,j,k];
       ux = \rf[i+1,j,k]*\et^{xx}[i+1,j,k] - \rf[i,j,k]*\et^{xx}[i,j,k]
       uy = 0.25*(\rf[i+1,j+1,k]*\et^{xy}[i+1,j+1,k] +
                  \rf[i,j+1,k]*\et^{xy}[i,j+1,k] -
                  \rf[i+1,j-1,k]*\et^{xy}[i+1,j-1,k] -
                  \rf[i,j-1,k]*\et^{xy}[i,j-1,k])
       ...

    where epsNum=1e-6 is to avoid division for zero when \abc=0.

    ** uses \temp[1:4] for \flux
    ** uses \temp[4:7] for \abch, or \abch needs to be recomputed in the
    next step
    ** runs on a separate CUDA kernel

    3) update the radiation field

       minus_wII = \et^{xx}[i,j,k]*(\abchX[i-1/2,j,k]+\abchX[i+1/2,j,k]) + \et^{yy}[i,j,k]*(\abchY[i,j-1/2,k]+\abchY[i,j+1/2,k]) + ...


       A = gamma/(1+gamma*(\abc[i,j,k]+minus_wII))
       d = dx*\rs[i,j,k] - \abc[i,j,k]*\rf[i,j,k] + \fluxX[i+1/2,j,k] - \fluxX[i-1/2,j,k] + ...

       rfNew = \rf[i,j,k] + alpha*A*d

       if(pars.lastIteration && rfNew>facOverOT*\ot[i,j,k]) rfNew = facOverOT*\ot[i,j,k];
       rf[i,j,k] = (rfNew<0 ? 0 : rfNew);

    where

       dx is the cell size

       alpha = 0.8  (a CFL-like number)
       gamma = 1
       facOverOT = 1.5 (a new parameter I am still exploring)

    ** runs on a separate CUDA kernel

    4) repeart for the far field
       
  end loop over frequencies

  end loop over iterations


  ** number of iterations is a simulation parameter.
  ** to limit the signal propagation spped to c, it should be capped at
  each step to

    unsigned int numIterationsAtC = (unsigned int)(1+mSpeedOfLightInCodeUnits*dt/dx);


*/
}

#endif // RT
#endif // CUDA

/*! \file RT_functions.h
 *  \brief Declarations for the gpu RT functions. */

#ifdef CUDA

  #ifndef RT_FUNCTIONS_H
    #define RT_FUNCTIONS_H

    #include "../global/global.h"
    #include "alt/decl.h"
    #include "radiation.h"

// void rtSolve(Real *dev_scalar, struct Rad3D::RT_Fields &rtFields);

struct DEVICE_ALIGN_DECL CrossSectionInCU {
  //
  //  Some cached values that are used often
  //
  Real HIatHI;
  Real HIatHeI;
  Real HIatHeII;
  Real HeIatHeI;
  Real HeIatHeII;
  Real HeIIatHeII;
};

  #endif  // VL_3D_CUDA_H
#endif    // CUDA

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

int Load_RT_Fields_To_Buffer(int direction, int side, int nx, int ny, int nz, int n_ghost, int n_freq,
                                struct Rad3D::RT_Fields& rtFields, Real *buffer, int buffer_start);

void Unload_RT_Fields_From_Buffer(int direction, int side, int nx, int ny, int nz, int n_ghost, int n_freq,
                                struct Rad3D::RT_Fields& rtFields, Real *buffer, int buffer_start);                                

void Set_RT_Boundaries_Periodic(int direction, int side, int nx, int ny, int nz, int n_ghost, int n_freq,
                                struct Rad3D::RT_Fields& rtFields);                                

  #endif  // VL_3D_CUDA_H
#endif    // CUDA

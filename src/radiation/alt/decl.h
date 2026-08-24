/*LICENSE*/

#ifndef DEVICES_DECL_ANY_H
#define DEVICES_DECL_ANY_H

#include "align.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define DEVICE_LOCAL_DECL __host__ __device__
#else  // __CUDACC__
  #define DEVICE_LOCAL_DECL
#endif  // __CUDACC__

#endif  // DEVICES_DECL_ANY_H

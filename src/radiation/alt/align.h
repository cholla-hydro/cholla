/*LICENSE*/

#ifndef DEVICES_ALIGN_ANY_H
#define DEVICES_ALIGN_ANY_H

//
//  Some compiler directives
//
#if defined(__CUDACC__)
  #define COMPILER_ALIGN_DECL(n) __align__(n)
#elif defined(__GNUC__)  // GCC
  #define COMPILER_ALIGN_DECL(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
  #define COMPILER_ALIGN_DECL(n) __declspec(align(n))
#elif defined(__clang__)
  #define COMPILER_ALIGN_DECL(n) __attribute__((align_value(n)))
#else
  #error "ALTAIR has not been ported to this platform."
#endif

#define DEVICE_ALIGN_DECL COMPILER_ALIGN_DECL(16)
#define GPU_ALIGN_DECL    COMPILER_ALIGN_DECL(16)

#endif  // DEVICES_ALIGN_ANY_H

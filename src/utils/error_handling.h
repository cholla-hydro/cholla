#ifndef ERROR_HANDLING_CHOLLA_H
#define ERROR_HANDLING_CHOLLA_H
#include <stdlib.h>

#include "../global/global.h"
[[noreturn]] void chexit(int code);

/*!
 * \brief Check that the Cholla configuration and parameters don't have any significant errors. Mostly compile time
 * checks.
 *
 */
void Check_Configuration(Parameters const& P);

/*!
 * \brief helper function that prints an error message & aborts the program (in
 * an MPI-safe way). Commonly invoked through a macro.
 *
 */
[[noreturn]] void Abort_With_Err_(const char* func_name, const char* file_name, int line_num, const char* msg, ...);

/* __CHOLLA_PRETTY_FUNC__ is a magic constant like __LINE__ or __FILE__ that
 * provides the name of the current function.
 * - The C++11 standard requires that __func__ is provided on all platforms, but
 *   that only provides limited information (just the name of the function).
 * - Where available, we prefer to use compiler-specific features that provide
 *   more information about the function (like the scope of the function & the
 *   the function signature).
 */
#ifdef __GNUG__
  #define __CHOLLA_PRETTY_FUNC__ __PRETTY_FUNCTION__
#else
  #define __CHOLLA_PRETTY_FUNC__ __func__
#endif

/*!
 * \brief print an error-message (with printf formatting) & abort the program.
 *
 * This macro should be treated as a function with the signature:
 *   [[noreturn]] void CHOLLA_ERROR(const char* msg, ...);
 *
 * - The 1st arg is printf-style format argument specifying the error message
 * - The remaining args arguments are used to format error message
 *
 * \note
 * the ``msg`` string is part of the variadic args so that there is always
 * at least 1 variadic argument (even in cases when ``msg`` doesn't format
 * any arguments). There is no way around this until C++ 20.
 */
#define CHOLLA_ERROR(...) Abort_With_Err_(__CHOLLA_PRETTY_FUNC__, __FILE__, __LINE__, __VA_ARGS__)

/*!
 * \brief if the condition is false, print an error-message (with printf
 * formatting) & abort the program.
 *
 * This macro should be treated as a function with the signature:
 *   [[noreturn]] void CHOLLA_ASSERT(bool cond, const char* msg, ...);
 *
 * - The 1st arg is a boolean condition. When true, this does noth
 * - The 2nd arg is printf-style format argument specifying the error message
 * - The remaining args arguments are used to format error message
 *
 * \note
 * the behavior is independent of the ``NDEBUG`` macro
 */
#define CHOLLA_ASSERT(cond, ...)                                              \
  if (not(cond)) {                                                            \
    Abort_With_Err_(__CHOLLA_PRETTY_FUNC__, __FILE__, __LINE__, __VA_ARGS__); \
  }

#endif /*ERROR_HANDLING_CHOLLA_H*/

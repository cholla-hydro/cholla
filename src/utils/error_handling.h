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
void Check_Configuration(parameters const& P);

/*!
 * \brief helper function that prints an error message & aborts the program (in
 * an MPI-safe way). Commonly invoked through a macro.
 *
 */
[[noreturn]] void Abort_With_Err_(const char* func_name, const char* file_name, int line_num, const char* msg, ...);

/*!
 * \brief print an error-message (with printf formatting) & abort the program.
 *
 * This macro should be treated as a function with the signature:
 *   [[noreturn]] void ERROR(const char* func_name, const char* msg, ...);
 *
 * - The 1st arg is the name of the function where it's called
 * - The 2nd arg is printf-style format argument specifying the error message
 * - The remaining args arguments are used to format error message
 *
 * \note
 * the ``msg`` string is part of the variadic args so that there is always
 * at least 1 variadic argument (even in cases when ``msg`` doesn't format
 * any arguments). There is no way around this until C++ 20.
 */
#define ERROR(func_name, ...) Abort_With_Err_(func_name, __FILE__, __LINE__, __VA_ARGS__)

/*!
 * \brief if the condition is false, print an error-message (with printf
 * formatting) & abort the program.
 *
 * This macro should be treated as a function with the signature:
 *   [[noreturn]] void ASSERT(bool cond, const char* func_name, const char* msg, ...);
 *
 * - The 1st arg is a boolean condition. When true, this does noth
 * - The 2nd arg is the name of the function where it's called
 * - The 3rd arg is printf-style format argument specifying the error message
 * - The remaining args arguments are used to format error message
 *
 * \note
 * the behavior is independent of the ``NDEBUG`` macro
 */
#define ASSERT(cond, func_name, ...)                             \
  if (not(cond)) {                                               \
    Abort_With_Err_(func_name, __FILE__, __LINE__, __VA_ARGS__); \
  }

#endif /*ERROR_HANDLING_CHOLLA_H*/

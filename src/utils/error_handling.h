#ifndef ERROR_HANDLING_CHOLLA_H
#define ERROR_HANDLING_CHOLLA_H
#include <stdlib.h>

#include <optional>
#include <type_traits>  // std::false_type

#include "../global/global.h"
[[noreturn]] void chexit(int code);

/*! a standard construct used to write `static_assert(false)`
 *
 *  This comes up with some frequency in code like the following:
 *  \code{C++}
 *    template<typename T>
 *    void pretty_print(T t) {
 *      if constexpr (std::is_same_v<T, int>) {
 *        std::printf("int with value: %d\n", t);
 *      } else if constexpr (std::is_same_v<T, std::string>) {
 *        std::printf("string with value: %s\n", t.c_str());
 *      } else {
 *        static_assert(always_false<T>, "received unexpected type");
 *      }
 *    }
 *  \endcode
 *
 *  \note
 *  In Feb 2023, the C++ standards committee retroactively revised the standard
 *  of C++11 to make this unnecessary:
 *  - https://cplusplus.github.io/CWG/issues/2518.html
 *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2593r1.html
 *  Once we are comfortable requiring a new enough compiler, we can delete
 *  this construct and replace each `always_false<T>` with `false`
 */
template <class...>
constexpr std::false_type always_false{};

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
  if (not(cond)) { /* NOLINT */                                               \
    Abort_With_Err_(__CHOLLA_PRETTY_FUNC__, __FILE__, __LINE__, __VA_ARGS__); \
  }

/*! \brief Unwrap the provided optional or abort with an error
 *
 *  This is basically a glorified version of std::optional<T>'s value method,
 *  but it's explicit about the fact that the program will abort if the
 *  optional doesn't contain a value.
 *
 *  \note
 *  The creation of this function was promptly motivated by the existence of a
 *  clang-tidy lint
 */
template <typename T>
[[gnu::always_inline]] inline T& get_or_abort(std::optional<T>& optional)
{
  if (optional.has_value()) {
    return optional.value();
  }
  CHOLLA_ERROR("the optional is empty");
}

#endif /*ERROR_HANDLING_CHOLLA_H*/

#include "../utils/error_handling.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <string>

#ifdef MPI_CHOLLA
  #include "../mpi/mpi_routines.h"
[[noreturn]] void chexit(int code)
{
  if (code == 0) {
    /*exit normally*/
    MPI_Finalize();
    exit(code);

  } else {
    /*exit with non-zero error code*/
    MPI_Abort(MPI_COMM_WORLD, code);
    exit(code);
  }
}
#else  /*MPI_CHOLLA*/
[[noreturn]] void chexit(int code)
{
  /*exit using code*/
  exit(code);
}
#endif /*MPI_CHOLLA*/

void Check_Configuration(Parameters const& P)
{
// General Checks
// ==============
#ifndef GIT_HASH
  #error "GIT_HASH is not defined"
#endif  //! GIT_HASH

  // Check that GIT_HASH is the correct length. It needs to be 41 and not 40 since strings are null terminated
  static_assert(sizeof(GIT_HASH) == 41);

#ifndef MACRO_FLAGS
  #error "MACRO_FLAGS is not defined"
#endif  //! MACRO_FLAGS

  // Check that MACRO_FLAGS has contents
  static_assert(sizeof(MACRO_FLAGS) > 1);

// Can only have one integrator enabled
#if ((defined(VL) + defined(CTU) + defined(SIMPLE)) != 1)
  #error "Only one integrator can be enabled at a time."
#endif  // Only one integrator check

  // Check the boundary conditions
  auto Check_Boundary = [](int const& boundary, std::string const& direction) {
    bool is_allowed_bc = boundary >= 0 and boundary <= 4;
    CHOLLA_ASSERT(is_allowed_bc,
                  "WARNING: Possibly invalid boundary conditions for direction: %s flag: %d. Must "
                  "select between 0 (no boundary), 1 (periodic), 2 (reflective), 3 (transmissive), "
                  "4 (custom), 5 (mpi).",
                  direction.c_str(), boundary);
  };
  Check_Boundary(P.xl_bcnd, "xl_bcnd");
  Check_Boundary(P.xu_bcnd, "xu_bcnd");
  Check_Boundary(P.yl_bcnd, "yl_bcnd");
  Check_Boundary(P.yu_bcnd, "yu_bcnd");
  Check_Boundary(P.zl_bcnd, "zl_bcnd");
  Check_Boundary(P.zu_bcnd, "zu_bcnd");

  // warn if error checking is disabled
#ifdef DISABLE_GPU_ERROR_CHECKING
  // NOLINTNEXTLINE(clang-diagnostic-#warnings)
  #warning "CUDA error checking is disabled. Enable it by compiling without the DISABLE_GPU_ERROR_CHECKING macro."
#endif  //! DISABLE_GPU_ERROR_CHECKING

  // Check that PRECISION is 2
#ifndef PRECISION
  #error "The PRECISION macro is required"
#endif  //! PRECISION
  static_assert(PRECISION == 2, "PRECISION must be 2. Single precision is not currently supported");

// MHD Checks
// ==========
#ifdef MHD
  assert(P.nx > 1 and P.ny > 1 and P.nz > 1 and "MHD runs must be 3D");

  // Must use the correct integrator
  #if !defined(VL) || defined(SIMPLE) || defined(CTU)
    #error "MHD only supports the Van Leer integrator"
  #endif  //! VL or SIMPLE

  // must only use HLLD
  #if !defined(HLLD) || defined(EXACT) || defined(ROE) || defined(HLL) || defined(HLLC)
    #error "MHD only supports the HLLD Riemann Solver"
  #endif  //! HLLD or EXACT or ROE or HLL or HLLC

  // May only use certain reconstructions
  #if ((defined(PCM) + defined(PLMC) + defined(PPMC)) != 1) || defined(PLMP) || defined(PPMP)
    #error "MHD only supports PCM, PLMC, and PPMC reconstruction"
  #endif  // Reconstruction check

  // must have HDF5
  #if defined(OUTPUT) and (not defined(HDF5))
    #error "MHD only supports HDF5 output"
  #endif  //! HDF5

  // Warn that diode boundaries are disabled
  if (P.xl_bcnd == 3 or P.xu_bcnd == 3 or P.yl_bcnd == 3 or P.yu_bcnd == 3 or P.zl_bcnd == 3 or P.zu_bcnd == 3) {
    std::cerr << "Warning: The diode on the outflow boundaries is disabled for MHD" << std::endl;
  }

  // Error if unsupported boundary condition is used
  assert(P.xl_bcnd != 2 or P.xu_bcnd != 2 or P.yl_bcnd != 2 or P.yu_bcnd != 2 or P.zl_bcnd != 2 or
         P.zu_bcnd != 2 && "MHD does not support reflective boundary conditions");

  // AVERAGE_SLOW_CELLS not supported on MHD
  #ifdef AVERAGE_SLOW_CELLS
    #error "MHD does not support AVERAGE_SLOW_CELLS"
  #endif  // AVERAGE_SLOW_CELLS

#endif  // MHD
}

// NOLINTNEXTLINE(cert-dcl50-cpp)
[[noreturn]] void Abort_With_Err_(const char* func_name, const char* file_name, int line_num, const char* msg, ...)
{
  // considerations when using MPI:
  //  - all processes must execute this function to catch errors that happen on
  //    just one process
  //  - to handle cases where all processes encounter the same error, we
  //    pre-buffer the error message (so that the output remains legible)

  // since we are aborting, it's OK that this isn't the most optimized

  // prepare some info for the error message header
  const char* sanitized_func_name = (func_name == nullptr) ? "{unspecified}" : func_name;

#ifdef MPI_CHOLLA
  std::string proc_info = std::to_string(procID) + " / " + std::to_string(nproc) + " (using MPI)";
#else
  std::string proc_info = "0 / 1 (NOT using MPI)";
#endif

  // prepare the formatted message
  std::string msg_buf;
  if (msg == nullptr) {
    msg_buf = "{nullptr encountered instead of error message}";
  } else {
    std::va_list args, args_copy;
    va_start(args, msg);
    va_copy(args_copy, args);

    // The clang-analyzer-valist.Uninitialized is bugged and triggers improperly on this line
    // NOLINTNEXTLINE(clang-analyzer-valist.Uninitialized)
    std::size_t bufsize_without_terminator = std::vsnprintf(nullptr, 0, msg, args);
    va_end(args);

    // NOTE: starting in C++17 it's possible to mutate msg_buf by mutating msg_buf.data()

    // we initialize a msg_buf with size == bufsize_without_terminator (filled with ' ' chars)
    // - msg_buf.data() returns a ptr with msg_buf.size() + 1 characters. We are allowed to
    //   mutate any of the first msg_buf.size() characters. The entry at
    //   msg_buf.data()[msg_buf.size()] is initially  '\0' (& it MUST remain equal to '\0')
    // - the 2nd argument of std::vsnprintf is the size of the output buffer. We NEED to
    //   include the terminator character in this argument, otherwise the formatted message
    //   will be truncated
    msg_buf = std::string(bufsize_without_terminator, ' ');
    std::vsnprintf(msg_buf.data(), bufsize_without_terminator + 1, msg, args_copy);
    va_end(args_copy);
  }

  // now write the error and exit
  std::fprintf(stderr,
               "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
               "Error occurred in %s on line %d\n"
               "Function: %s\n"
               "Rank: %s\n"
               "Message: %s\n"
               "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n",
               file_name, line_num, sanitized_func_name, proc_info.data(), msg_buf.data());
  std::fflush(stderr);  // may be unnecessary for stderr
  chexit(1);
}
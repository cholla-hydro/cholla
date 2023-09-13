#include "../utils/error_handling.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

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

void Check_Configuration(parameters const& P)
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

  // Must have CUDA
#ifndef CUDA
  #error "The CUDA macro is required"
#endif  //! CUDA

// Can only have one integrator enabled
#if ((defined(VL) + defined(CTU) + defined(SIMPLE)) != 1)
  #error "Only one integrator can be enabled at a time."
#endif  // Only one integrator check

  // Check the boundary conditions
  auto Check_Boundary = [](int const& boundary, std::string const& direction) {
    bool is_allowed_bc = boundary >= 0 and boundary <= 4;
    std::string const error_message =
        "WARNING: Possibly invalid boundary conditions for direction: " + direction +
        " flag: " + std::to_string(boundary) +
        ". Must select between 0 (no boundary), 1 (periodic), 2 (reflective), 3 (transmissive), 4 (custom), 5 (mpi).";
    assert(is_allowed_bc && error_message.c_str());
  };
  Check_Boundary(P.xl_bcnd, "xl_bcnd");
  Check_Boundary(P.xu_bcnd, "xu_bcnd");
  Check_Boundary(P.yl_bcnd, "yl_bcnd");
  Check_Boundary(P.yu_bcnd, "yu_bcnd");
  Check_Boundary(P.zl_bcnd, "zl_bcnd");
  Check_Boundary(P.zu_bcnd, "zu_bcnd");

  // warn if error checking is disabled
#ifndef CUDA_ERROR_CHECK
  #warning "CUDA error checking is disabled. Enable it with the CUDA_ERROR_CHECK macro"
#endif  //! CUDA_ERROR_CHECK

  // Check that PRECISION is 2
#ifndef PRECISION
  #error "The PRECISION macro is required"
#endif  //! PRECISION
  static_assert(PRECISION == 2, "PRECISION must be 2. Single precision is not currently supported");

  // Check that gamma, the ratio of specific heats, is greater than 1
  assert(::gama <= 1.0 and "Gamma must be greater than one.");

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
  const char* santized_func_name = (func_name == nullptr) ? "{unspecified}" : func_name;

  std::string proc_info =
#ifdef MPI_CHOLLA
      std::to_string(procID) + " / " + std::to_string(nproc) + " (using MPI)";
#else
      "0 / 1 (NOT using MPI)"
#endif

  // prepare the formatted message
  std::vector<char> msg_buf;
  if (msg == nullptr) {
    msg_buf = std::vector<char>(80);
    std::snprintf(msg_buf.data(), msg_buf.size(), "{nullptr encountered instead of error message}");
  } else {
    std::va_list args, args_copy;
    va_start(args, msg);
    va_copy(args_copy, args);

    std::size_t msg_len = std::vsnprintf(nullptr, 0, msg, args) + 1;
    va_end(args);

    msg_buf = std::vector<char>(msg_len);
    std::vsnprintf(msg_buf.data(), msg_len, msg, args);
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
               file_name, line_num, santized_func_name, proc_info.data(), msg_buf.data());
  std::fflush(stderr);  // may be unnecessary for stderr
  chexit(1);
}
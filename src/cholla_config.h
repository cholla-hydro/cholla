/*! \file
 *  Holds configuration parameters set at compile time
 *
 *  There are generally 2 versions of this file
 *  1. The template file (i.e. cholla_config.h.in)
 *  2. The generated file (i.e. cholla_config.h). This is generated from the template
 *     file using configure_file.py
 *
 *  NEVER directly mutate the generated file
 *
 *  Before you add or modify any entries in this file, make sure you familiarize
 *  yourself with the best-practices described within
 *  https://cholla.readthedocs.io/en/latest/Development/build-time-config.html
 */

#pragma once

#define GIT_HASH "e831d8c8ce19c3e047f12d6df8d7c7f7b9f3e13d"
#define MACRO_FLAGS "-DMPI_CHOLLA -DPRECISION=2 -DHLLC -DVL -DPPMP -DTEMPERATURE_FLOOR -DSLICES -DPROJECTION -DOUTPUT -DHDF5 -DMPI_GPU -DGIT_HASH=e831d8c8ce19c3e047f12d6df8d7c7f7b9f3e13d"

// sets the precision of REAL
#define PRECISION 2
#if (PRECISION != 1) && (PRECISION != 2)
  #error "PRECISION MUST BE 1 or 2"
#endif

// ==================
// Parallelism Config
// ==================
// Do we actually need to define the following 2 macros? They seems like
// the sorts of things that would be defined by the compiler for us (or
// that could be inferred from the compiler's builtin macros)
/* #undef O_HIP */

/* #undef DISABLE_GPU_ERROR_CHECKING */

#define MPI_CHOLLA
#define MPI_GPU

/* #undef PARALLEL_OMP */
/* #undef N_OMP_THREADS */
// this pertains to Grackle
/* #undef N_OMP_THREADS_GRACKLE */
// this pertains to particles
/* #undef PRINT_OMP_DOMAIN */

// =======================
// HYDRO/MHD Configuration
// =======================
// when MHD isn't defined, cholla uses pure HYDRO
/* #undef MHD */

// integrator choice:
/* #undef SIMPLE */
#define VL

// reconstruction choices (Only 1 of these is allowed to be defined at a time)
/* #undef PCM */
/* #undef PLMC */
/* #undef PLMP */
/* #undef PPMC */
#define PPMP

// reconstruction parameter:
/* #undef CTU */

// Riemann Solver choices (Only 1 of these should be defined)
#define HLLC
/* #undef HLLD */
/* #undef ROE */
/* #undef EXACT */


// "fudging" schemes to deal with challenging hydro conditions
/* #undef AVERAGE_SLOW_CELLS */
/* #undef DENSITY_FLOOR */
/* #undef SCALAR_FLOOR */
/* #undef TEMPERATURE_CEILING */
#define TEMPERATURE_FLOOR

// When defined, the dual-energy formalism is enabled
/* #undef DE */

// Passive-Scalar Configuration and Modules:
/* #undef SCALAR */
/* #undef BASIC_SCALAR */
/* #undef DUST */

// ======================
// Analysis Configuration
// ======================
/* #undef ANALYSIS */
/* #undef LYA_STATISTICS */
/* #undef PHASE_DIAGRAM */


// ===============================
// Cooling/Chemistry Configuration
// ===============================

/* #undef CHEMISTRY_GPU */
/* #undef COOLING_GRACKLE */

/* #undef GRACKLE_METALS */
// the following is grackle-related
// - originally, you had to configure this parameter to match the precision
//   that Grackle was configured with...
// - this hasn't been necessary for several years... (Grackle now provides
//   these macros)
/* #undef CONFIG_BFLOAT_8 */

// =======================
// Cosmology Configuration
// =======================
/* #undef COSMOLOGY */
/* #undef PRINT_INITIAL_STATS */
/* #undef TILED_INITIAL_CONDITIONS */


// =====================
// Gravity Configuration
// =====================
/* #undef GRAVITY */
/* #undef GRAVITY_5_POINTS_GRADIENT */
/* #undef GRAVITY_ANALYTIC_COMP */
/* #undef GRAVITY_GPU */
/* #undef GRAVITY_LONG_INTS */
/* #undef PARIS */
/* #undef PARIS_3PT */
/* #undef PARIS_5PT */
/* #undef PARIS_GALACTIC */
/* #undef PARIS_NO_GPU_MPI */
/* #undef SOR */
/* #undef STATIC_GRAV */

// ======================
// Feedback Configuration
// ======================
// Once PR # 386 is merged, we should look into removing some (all?) of
// these
/* #undef FEEDBACK */
/* #undef NO_SN_FEEDBACK */
/* #undef NO_WIND_FEEDBACK */
/* #undef ONLY_RESOLVED */

// ======================
// Particle Configuration
// ======================
/* #undef ONLY_PARTICLES */
/* #undef PARTICLES */
/* #undef PARTICLES_CPU */
/* #undef PARTICLES_GPU */
/* #undef PARTICLES_KDK */
/* #undef PARTICLES_LONG_INTS */
/* #undef PARTICLE_AGE */
/* #undef PARTICLE_IDS */
/* #undef SINGLE_PARTICLE_MASS */

// ====================
// OUTPUT CONFIGURATION
// ====================
#define HDF5
#define OUTPUT
/* #undef OUTPUT_ALWAYS */
/* #undef OUTPUT_CHEMISTRY */
/* #undef OUTPUT_POTENTIAL */
/* #undef OUTPUT_SKEWERS */
/* #undef OUTPUT_TEMPERATURE */
#define PROJECTION
/* #undef ROTATED_PROJECTION */
#define SLICES
/* #undef N_OUTPUT_COMPLETE */

// ============
// Misc Options
// ============
/* #undef CPU_TIME */
/* #undef DISK_ICS */

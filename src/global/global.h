/*! /file global.h
 *  /brief Declarations of global variables and functions. */

#ifndef GLOBAL_H
#define GLOBAL_H

#include <string>

#include "../grid/grid_enum.h"  // defines NSCALARS
#include "cholla_config.h"

#ifdef PARTICLES
  #include <cstdint>
#endif  // PARTICLES

#if PRECISION == 1
  #ifndef TYPEDEF_DEFINED_REAL
typedef float Real;
  #endif
#endif
#if PRECISION == 2
  #ifndef TYPEDEF_DEFINED_REAL
typedef double Real;
  #endif
#endif

#define MAXLEN      2048
#define TINY_NUMBER 1.0e-20
#define MP          1.672622e-24  // mass of proton, grams
#define KB          1.380658e-16  // boltzmann constant, cgs
// #define GN 6.67259e-8 // gravitational constant, cgs
#define GN  4.49451e-18  // gravitational constant, kpc^3 / M_sun / kyr^2
#define C_L 0.306594593  // speed of light in kpc/kyr

#define MYR      31.536e12         // Myears in secs
#define KPC      3.086e16          // kpc in km
#define G_COSMO  4.300927161e-06;  // gravitational constant, kpc km^2 s^-2 Msun^-1
#define MSUN_CGS 1.98847e33;       // Msun in gr
#define KPC_CGS  3.086e21;         // kpc in cm
#define KM_CGS   1e5;              // km in cm
#define MH       1.67262171e-24    // Mass of hydrogen [g]

#define TIME_UNIT           3.15569e10     // 1 kyr in s
#define LENGTH_UNIT         3.08567758e21  // 1 kpc in cm
#define MASS_UNIT           1.98847e33     // 1 solar mass in grams
#define DENSITY_UNIT        (MASS_UNIT / (LENGTH_UNIT * LENGTH_UNIT * LENGTH_UNIT))
#define FORCE_UNIT          (MASS_UNIT * LENGTH_UNIT / TIME_UNIT / TIME_UNIT)
#define VELOCITY_UNIT       (LENGTH_UNIT / TIME_UNIT)
#define ENERGY_UNIT         (DENSITY_UNIT * VELOCITY_UNIT * VELOCITY_UNIT)
#define PRESSURE_UNIT       (DENSITY_UNIT * VELOCITY_UNIT * VELOCITY_UNIT)
#define SP_ENERGY_UNIT      (VELOCITY_UNIT * VELOCITY_UNIT)
#define MAGNETIC_FIELD_UNIT (sqrt(MASS_UNIT / LENGTH_UNIT) / TIME_UNIT)

#define LOG_FILE_NAME "run_output.log"

// mean molecular weight
#define MU 0.6
// Parameters for Enzo dual Energy Condition
// - Prior to GH PR #356, DE_ETA_1 nominally had a value of 0.001 in all
//   simulations (in practice, the value of DE_ETA_1 had minimal significance
//   in those simulations). In PR #356, we revised the internal-energy
//   synchronization to account for the value of DE_ETA_1. This was necessary
//   for non-cosmology simulations.
// - In Cosmological simulation, we set DE_ETA_1 to a large number (it doesn't
//   really matter what, as long as its >=1) to maintain the older behavior
// - In the future, we run tests and revisit the choice of DE_ETA_1 in
//   cosmological simulations
#ifdef COSMOLOGY
  #define DE_ETA_1 10.0
#else
  #define DE_ETA_1 \
    0.001  // Ratio of U to E for which  Internal Energy is used to compute the
           // Pressure. This also affects when the Internal Energy is used for
           // the update.
#endif

#define DE_ETA_2 \
  0.035  // Ratio of U to max(E_local) used to select which Internal Energy is
         // used for the update.

// Maximum time step for cosmological simulations
#define MAX_DELTA_A        0.001
#define MAX_EXPANSION_RATE 0.01  // Limit delta(a)/a

#ifdef MHD
  #define N_MHD_FIELDS 3
#else
  #define N_MHD_FIELDS 0
#endif  // MHD

// Inital Chemistry fractions
#define INITIAL_FRACTION_HI       0.75984603480
#define INITIAL_FRACTION_HII      1.53965115054e-4
#define INITIAL_FRACTION_HEI      0.24000000008
#define INITIAL_FRACTION_HEII     9.59999999903e-15
#define INITIAL_FRACTION_HEIII    9.59999999903e-18
#define INITIAL_FRACTION_ELECTRON 1.53965115054e-4
#define INITIAL_FRACTION_METAL    1.00000000000e-10

// Default Particles Compiler Flags
#define PARTICLES_LONG_INTS
#define PARTICLES_KDK

#ifdef GRAVITY
  #ifdef GRAVITY_5_POINTS_GRADIENT
    #ifdef PARTICLES
      #define N_GHOST_POTENTIAL \
        3  // 3 ghost cells are needed for 5 point gradient, ( one is for the
           // CIC interpolation of the potential )
    #else
      #define N_GHOST_POTENTIAL 2  // 2 ghost cells are needed for 5 point gradient
    #endif                         // PARTICLES

  #else
    #ifdef PARTICLES
      #define N_GHOST_POTENTIAL \
        2  // 2 ghost cells are needed for 3 point gradient, ( one is for the
           // CIC interpolation of the potential )
    #else
      #define N_GHOST_POTENTIAL 1  // 1 ghost cells are needed for 3 point gradient
    #endif                         // PARTICLES
  #endif                           // GRAVITY_5_POINTS_GRADIENT

typedef long int grav_int_t;
#endif

#ifdef PARTICLES_LONG_INTS
typedef long int part_int_t;
#else
typedef int part_int_t;
#endif  // PARTICLES_LONG_INTS

#include <vector>
typedef std::vector<Real> real_vector_t;
typedef std::vector<part_int_t> int_vector_t;

#ifdef PARTICLES
  #ifdef MPI_CHOLLA
// Constants for the inital size of the buffers for particles transfer
// and the number of data transferred for each particle
extern int N_PARTICLES_TRANSFER;
extern int N_DATA_PER_PARTICLE_TRANSFER;
  #endif  // MPI_CHOLLA

  #ifdef AVERAGE_SLOW_CELLS
    #define SLOW_FACTOR 10
  #endif  // AVERAGE_SLOW_CELLS

#endif  // PARTICLES

#define SIGN(a) (((a) < 0.) ? -1. : 1.)

/* Global variables */
extern Real gama;   // Ratio of specific heats
extern Real C_cfl;  // CFL number (0 - 0.5)
extern Real t_comm;
extern Real t_other;

extern float *cooling_table;
extern float *heating_table;

/*! \fn void Set_Gammas(Real gamma_in)
 *  \brief Set gamma values for Riemann solver. */
extern void Set_Gammas(Real gamma_in);

/*! \fn double Get_Time(void)
 *  \brief Returns the current clock time. */
extern double Get_Time(void);

/*! \fn int sgn
 *  \brief Mathematical sign function. Returns sign of x. */
extern int Sgn(Real x);

/* Global variables for mpi (but they are also initialized to sensible defaults when not using mpi)
 *
 * It may make sense to move these back into mpi_routines (but reorganizing the ifdef statements
 * would take some work). It may make sense to also put these into their own namespace.
 */
extern int procID; /*process rank*/
extern int nproc;  /*number of processes executing simulation*/
extern int root;   /*rank of root process*/

/* Used when MPI_CHOLLA is not defined to initialize a subset of the global mpi-related variables
 * that still meaningful in non-mpi simulations.
 */
void Init_Global_Parallel_Vars_No_MPI();

// forward-declare ParameterMap (it's primarily used to construct Parameters)
class ParameterMap;

/*! A collection of assorted parameter values.
 *
 *  The existence of this type is largely a historical artifact. The plan is to
 *  gradually remove data members from this type.
 *
 *  Guidelines for Removing Parameters
 *  ----------------------------------
 *  In the vast majority of cases, a parameter value is temporarily stored here and
 *  then they are only referenced one subsequent time (while the simulation is being
 *  initialized).
 *   - for example, a bunch of parameters are only ever read while initializing
 *     \ref Cosmology, \ref Gravity, \ref Header, \ref Particles, etc.
 *   - some parameters, are only read once while setting up initial conditions.
 *   - in all these cases, we can just read the data directly from \ref ParameterMap
 *     rather than temporarily storing the values here.
 *
 *  When relocating other parameters, be mindful that a single execution of Cholla
 *  should **NEVER** read a parameter's value from \ref ParameterMap more than once
 *  - in this scenario, it would be extremely easy for different parts of the code
 *    to accidentally start using different default values, which would introduce
 *    lots of hard to debug issues
 *  - while there are some potential workarounds to this, I think they might
 *    produce some undesired long-term behavior (plus, they are imperfect)
 *  - if this situation arises, it's probably a sign that you should make a new struct
 *    (maybe a "Model" type?) where a value can be persistently stored
 */
struct Parameters {
  /*! Construct a new instance using values from \p pmap
   *
   *  \param[in] pmap The map of all parsed parameters. Reminder: the only reason this
   *      isn't marked ``const`` is to reflect the fact that the type internally tracks
   *      each parameter that is accessed.
   */
  explicit Parameters(ParameterMap &pmap);

  // List the parameters
  int nx;
  int ny;
  int nz;
  double tout;

  // The following output time and
  // maximum timestep items control
  // the output times for certain RT
  // tests. These can be revised out
  // of the code and should be
  // considered temporary.
  double outstep;
  Real outstep_dexinc;
  Real max_timestep_dexinc;
  Real max_timestep;

  int n_steps_output;
  Real gamma;
  char init[MAXLEN];
  int nfile;
  // At the moment, the following flag is only meaningful when GRAVITY and GRAVITY_ANALYTIC_COMP
  // are defined. In other cases, we force this to initialize to a sensible value
  bool gas_only_use_static_grav;
  bool output_always = false;
  int n_steps_limit  = -1;  // Note that negative values indicate that there is no limit
#ifdef STATIC_GRAV
  int custom_grav = 0;  // flag to set specific static gravity field
#endif
  Real xmin;
  Real ymin;
  Real zmin;
  Real xlen;
  Real ylen;
  Real zlen;
  int xl_bcnd;
  int xu_bcnd;
  int yl_bcnd;
  int yu_bcnd;
  int zl_bcnd;
  int zu_bcnd;
#ifdef MPI_CHOLLA
  int xlg_bcnd;
  int xug_bcnd;
  int ylg_bcnd;
  int yug_bcnd;
  int zlg_bcnd;
  int zug_bcnd;
#endif /*MPI_CHOLLA*/
  char custom_bcnd[MAXLEN];
  char indir[MAXLEN];  // Folder to load Initial conditions from
  char outdir[MAXLEN];  // Folder to load Initial conditions from
  Real rho;
  Real vx;
  Real vy;
  Real vz;
  Real P;
  Real A;
  Real Bx;
  Real By;
  Real Bz;
  Real rho_l;
  Real vx_l;
  Real vy_l;
  Real vz_l;
  Real P_l;
  Real Bx_l;
  Real By_l;
  Real Bz_l;
  Real rho_r;
  Real vx_r;
  Real vy_r;
  Real vz_r;
  Real P_r;
  Real Bx_r;
  Real By_r;
  Real Bz_r;
  Real diaph;
  Real rEigenVec_rho;
  Real rEigenVec_MomentumX;
  Real rEigenVec_MomentumY;
  Real rEigenVec_MomentumZ;
  Real rEigenVec_E;
  Real rEigenVec_Bx;
  Real rEigenVec_By;
  Real rEigenVec_Bz;
  Real pitch;
  Real yaw;
  Real polarization;
  Real radius;
  Real P_blast;
  Real wave_length;
#ifdef PARTICLES
  // The random seed for particle simulations. With the default of 0 then a
  // machine dependent seed will be generated.
  std::uint_fast64_t prng_seed = 0;
#endif  // PARTICLES

#if defined(CHEMISTRY_GPU) || defined(COSMOLOGY)
  Real YHe;            // helium mass fraction
#endif
#ifdef COSMOLOGY
  Real H0;
  Real Omega_M;
  Real Omega_L;
  Real Omega_b;
  Real Omega_R;
  Real w0;
  Real wa;
  Real Init_redshift;
  Real End_redshift;
  Real T_init;
  unsigned long long cosmo_ics_seed;  // Cosmological ICs seed
  char cosmo_ics_pk_file[MAXLEN];
  Real xHp_ion_init;   // hydrogen ionization fraction
  Real xHep_ion_init;  // helium ionization fraction

  std::string wDE_file;  // File with equation of state as function of redshift

  // File for the scale_factor output values for cosmological simulations
  char scale_outputs_file[MAXLEN];
  #define EXPANSION_HISTORY_FILE_NAME "expansion_history.txt"
  #define GROWTH_FACTOR_FILE_NAME     "growth_factor.txt"
#endif  // COSMOLOGY
#ifdef TILED_INITIAL_CONDITIONS
  Real tile_length;
#endif  // TILED_INITIAL_CONDITIONS

  // Set the MPI Processes grid [n_proc_x, n_proc_y, n_proc_z]
  int n_proc_x;
  int n_proc_y;
  int n_proc_z;

  int bc_potential_type;
#if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  char UVB_rates_file[MAXLEN];  // File for the UVB photoheating and
                                // photoionization rates of HI, HeI and HeII
#endif
#ifdef RT
  int rt_num_iterations;
#endif
  Real temperature_floor = 0;
  Real density_floor     = 0;
  Real scalar_floor      = 0;
#ifdef ANALYSIS
  char analysis_scale_outputs_file[MAXLEN];  // File for the scale_factor output
                                             // values for cosmological
                                             // simulations {{}}
  char analysisdir[MAXLEN];
  int lya_skewers_stride;
  Real lya_Pk_d_log_k;
  #ifdef OUTPUT_SKEWERS
  char skewersdir[MAXLEN];
  #endif
#endif
#ifdef SCALAR
  #ifdef DUST
  Real grain_radius;
  #endif
#endif
};

/*! \brief prints a warning if pmap contains any unused parameters */
void Warn_Unused_Params(ParameterMap &pmap);

/*! \fn int is_param_valid(char *name);
 * \brief Verifies that a param is valid (even if not needed).  Avoids
 * "warnings" in output. */
extern int Is_Param_Valid(const char *name);

#endif  // GLOBAL_H

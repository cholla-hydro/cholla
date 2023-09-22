/*! /file global.h
 *  /brief Declarations of global variables and functions. */

#ifndef GLOBAL_H
#define GLOBAL_H

#include "../grid/grid_enum.h"  // defines NSCALARS

#ifdef COOLING_CPU
  #include <gsl/gsl_spline.h>
  #include <gsl/gsl_spline2d.h>
#endif

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
#define VELOCITY_UNIT       (LENGTH_UNIT / TIME_UNIT)
#define ENERGY_UNIT         (DENSITY_UNIT * VELOCITY_UNIT * VELOCITY_UNIT)
#define PRESSURE_UNIT       (DENSITY_UNIT * VELOCITY_UNIT * VELOCITY_UNIT)
#define SP_ENERGY_UNIT      (VELOCITY_UNIT * VELOCITY_UNIT)
#define MAGNETIC_FIELD_UNIT (sqrt(MASS_UNIT / LENGTH_UNIT) / TIME_UNIT)

#define LOG_FILE_NAME "run_output.log"

// Conserved Floor Values
#define TEMP_FLOOR 1e-3
#define DENS_FLOOR 1e-5  // in code units

// Parameter for Enzo dual Energy Condition
#define DE_ETA_1 \
  0.001  // Ratio of U to E for which  Internal Energy is used to compute the
         // Pressure
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

#ifdef PARTICLES
  #ifdef PARTICLES_LONG_INTS
typedef long int part_int_t;
  #else
typedef int part_int_t;
  #endif  // PARTICLES_LONG_INTS

  #include <vector>
typedef std::vector<Real> real_vector_t;
typedef std::vector<part_int_t> int_vector_t;
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

#ifdef COOLING_CPU
extern gsl_interp_accel *acc;
extern gsl_interp_accel *xacc;
extern gsl_interp_accel *yacc;
extern gsl_spline *highT_C_spline;
extern gsl_spline2d *lowT_C_spline;
extern gsl_spline2d *lowT_H_spline;
#endif
#ifdef COOLING_GPU
extern float *cooling_table;
extern float *heating_table;
#endif

/*! \fn void Set_Gammas(Real gamma_in)
 *  \brief Set gamma values for Riemann solver. */
extern void Set_Gammas(Real gamma_in);

/*! \fn double get_time(void)
 *  \brief Returns the current clock time. */
extern double Get_Time(void);

/*! \fn int sgn
 *  \brief Mathematical sign function. Returns sign of x. */
extern int Sgn(Real x);

#ifndef CUDA
/*! \fn Real calc_eta(Real cW[], Real gamma)
 *  \brief Calculate the eta value for the H correction. */
extern Real Calc_Eta(Real cW[], Real gamma);
#endif

struct parameters {
  int nx;
  int ny;
  int nz;
  double tout;
  double outstep;
  int n_steps_output;
  Real gamma;
  char init[MAXLEN];
  int nfile;
  int n_hydro                = 1;
  int n_particle             = 1;
  int n_projection           = 1;
  int n_rotated_projection   = 1;
  int n_slice                = 1;
  int n_out_float32          = 0;
  int out_float32_density    = 0;
  int out_float32_momentum_x = 0;
  int out_float32_momentum_y = 0;
  int out_float32_momentum_z = 0;
  int out_float32_Energy     = 0;
#ifdef DE
  int out_float32_GasEnergy = 0;
#endif
#ifdef MHD
  int out_float32_magnetic_x = 0;
  int out_float32_magnetic_y = 0;
  int out_float32_magnetic_z = 0;
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
  char outdir[MAXLEN];
  char indir[MAXLEN];  // Folder to load Initial conditions from
  Real rho                 = 0;
  Real vx                  = 0;
  Real vy                  = 0;
  Real vz                  = 0;
  Real P                   = 0;
  Real A                   = 0;
  Real Bx                  = 0;
  Real By                  = 0;
  Real Bz                  = 0;
  Real rho_l               = 0;
  Real vx_l                = 0;
  Real vy_l                = 0;
  Real vz_l                = 0;
  Real P_l                 = 0;
  Real Bx_l                = 0;
  Real By_l                = 0;
  Real Bz_l                = 0;
  Real rho_r               = 0;
  Real vx_r                = 0;
  Real vy_r                = 0;
  Real vz_r                = 0;
  Real P_r                 = 0;
  Real Bx_r                = 0;
  Real By_r                = 0;
  Real Bz_r                = 0;
  Real diaph               = 0;
  Real rEigenVec_rho       = 0;
  Real rEigenVec_MomentumX = 0;
  Real rEigenVec_MomentumY = 0;
  Real rEigenVec_MomentumZ = 0;
  Real rEigenVec_E         = 0;
  Real rEigenVec_Bx        = 0;
  Real rEigenVec_By        = 0;
  Real rEigenVec_Bz        = 0;
  Real pitch               = 0;
  Real yaw                 = 0;
  Real polarization        = 0;
  Real radius              = 0;
  Real P_blast             = 0;
  Real wave_length         = 1.0;
#ifdef PARTICLES
  // The random seed for particle simulations. With the default of 0 then a
  // machine dependent seed will be generated.
  std::uint_fast64_t prng_seed = 0;
#endif  // PARTICLES
#ifdef SUPERNOVA
  char snr_filename[MAXLEN];
#endif
#ifdef ROTATED_PROJECTION
  // initialize rotation parameters to zero
  int nxr;
  int nzr;
  Real delta = 0;
  Real theta = 0;
  Real phi   = 0;
  Real Lx;
  Real Lz;
  int n_delta    = 0;
  Real ddelta_dt = 0;
  int flag_delta = 0;
#endif /*ROTATED_PROJECTION*/
#ifdef COSMOLOGY
  Real H0;
  Real Omega_M;
  Real Omega_L;
  Real Omega_b;
  Real Init_redshift;
  Real End_redshift;
  char scale_outputs_file[MAXLEN];  // File for the scale_factor output values
                                    // for cosmological simulations
#endif                              // COSMOLOGY
#ifdef TILED_INITIAL_CONDITIONS
  Real tile_length;
#endif  // TILED_INITIAL_CONDITIONS

#ifdef SET_MPI_GRID
  // Set the MPI Processes grid [n_proc_x, n_proc_y, n_proc_z]
  int n_proc_x;
  int n_proc_y;
  int n_proc_z;
#endif
  int bc_potential_type;
#if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  char UVB_rates_file[MAXLEN];  // File for the UVB photoheating and
                                // photoionization rates of HI, HeI and HeII
#endif
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
};

/*! \fn void parse_params(char *param_file, struct parameters * parms);
 *  \brief Reads the parameters in the given file into a structure. */
extern void Parse_Params(char *param_file, struct parameters *parms, int argc, char **argv);

/*! \fn int is_param_valid(char *name);
 * \brief Verifies that a param is valid (even if not needed).  Avoids
 * "warnings" in output. */
extern int Is_Param_Valid(const char *name);

#endif  // GLOBAL_H

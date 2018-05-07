/*! /file global.h
 *  /brief Declarations of global variables and functions. */


#ifndef GLOBAL_H 
#define GLOBAL_H 

#ifdef COOLING_CPU
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#endif

#if PRECISION == 1
#ifndef FLOAT_TYPEDEF_DEFINED
typedef float Real;
#endif //FLOAT_TYPEDEF_DEFINED
#endif //PRECISION == 1
#if PRECISION == 2
#ifndef FLOAT_TYPEDEF_DEFINED
typedef double Real;
#endif //FLOAT_TYPEDEF_DEFINED
#endif //PRECISION == 2

#define MAXLEN 100
#define TINY_NUMBER 1.0e-20
#define PI 3.141592653589793
#define MP 1.672622e-24 // mass of proton, grams
#define KB 1.380658e-16 // boltzmann constant, cgs
//#define GN 6.67259e-8 // gravitational constant, cgs
#define GN 4.49451e-18 // gravitational constant, kpc^3 / M_sun / kyr^2

#define TIME_UNIT 3.15569e10 // 1 kyr in s
#define LENGTH_UNIT 3.08567758e21 // 1 kpc in cm
#define MASS_UNIT 1.98855e33 // 1 solar mass in grams
#define DENSITY_UNIT (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT)) 
#define VELOCITY_UNIT (LENGTH_UNIT/TIME_UNIT)
#define ENERGY_UNIT (DENSITY_UNIT*VELOCITY_UNIT*VELOCITY_UNIT)
#define PRESSURE_UNIT (DENSITY_UNIT*VELOCITY_UNIT*VELOCITY_UNIT)
#define SP_ENERGY_UNIT (VELOCITY_UNIT*VELOCITY_UNIT)

#ifdef SCALAR
#define NSCALARS 1
#endif


#define SIGN(a) ( ((a) < 0.) ? -1. : 1. )



/* Global variables */
extern Real gama; // Ratio of specific heats
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
extern double get_time(void);

/*! \fn int sgn
 *  \brief Mathematical sign function. Returns sign of x. */
extern int sgn(Real x);

#ifndef CUDA
/*! \fn Real calc_eta(Real cW[], Real gamma)
 *  \brief Calculate the eta value for the H correction. */
extern Real calc_eta(Real cW[], Real gamma);
#endif


struct parameters
{
  int nx;
  int ny;
  int nz;
  double tout;
  double outstep;
  Real gamma;
  char init[MAXLEN];
  int nfile;
  int nfull;
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
#ifdef   MPI_CHOLLA
  int xlg_bcnd;
  int xug_bcnd;
  int ylg_bcnd;
  int yug_bcnd;
  int zlg_bcnd;
  int zug_bcnd;
#endif /*MPI_CHOLLA*/
  char custom_bcnd[MAXLEN];
  char outdir[MAXLEN];
  Real rho;
  Real vx;
  Real vy;
  Real vz;
  Real P;
  Real A;
  Real rho_l;
  Real v_l;
  Real P_l;
  Real rho_r;
  Real v_r;
  Real P_r;
  Real diaph;
#ifdef ROTATED_PROJECTION
  int nxr;
  int nzr;
  Real delta;
  Real theta;
  Real phi;
  Real Lx;
  Real Lz;
  int n_delta;
  Real ddelta_dt;
  int flag_delta;
#endif /*ROTATED_PROJECTION*/
};


/*! \fn void parse_params(char *param_file, struct parameters * parms);
 *  \brief Reads the parameters in the given file into a structure. */
extern void parse_params (char *param_file, struct parameters * parms);


#endif //GLOBAL_H

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
#endif //SINGLE_PRECISION
#if PRECISION == 2
#ifndef FLOAT_TYPEDEF_DEFINED
typedef double Real;
#endif //FLOAT_TYPEDEF_DEFINED
#endif //DOUBLE_PRECISION

#define MAXLEN 80
#define TINY_NUMBER 1.0e-20
#define PI 3.141592653589793
#define MP 1.672622e-24 // mass of proton, grams
#define KB 1.380658e-16 // boltzmann constant, cgs

#define TIME_UNIT 3.15569e10
#define LENGTH_UNIT 3.08567758e18
#define DENSITY_UNIT 1.672622e-24
#define VELOCITY_UNIT (LENGTH_UNIT/TIME_UNIT)
#define ENERGY_UNIT (DENSITY_UNIT*VELOCITY_UNIT*VELOCITY_UNIT)
#define PRESSURE_UNIT (DENSITY_UNIT*VELOCITY_UNIT*VELOCITY_UNIT)
#define SP_ENERGY_UNIT (VELOCITY_UNIT*VELOCITY_UNIT)


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

#ifndef CUDA
/*! \fn Real maxof3(Real a, Real b, Real c)
 *  \brief Returns the maximum of three floating point numbers. */
extern Real maxof3(Real a, Real b, Real c);

/*! \fn Real minof3(Real a, Real b, Real c)
 *  \brief Returns the minimum of three floating point numbers. */
extern Real minof3(Real a, Real b, Real c);

/*! \fn int sgn
 *  \brief Mathematical sign function. Returns sign of x. */
extern int sgn(Real x);

/*! \fn Real calc_eta(Real cW[], Real gamma)
 *  \brief Calculate the eta value for the H correction. */
extern Real calc_eta(Real cW[], Real gamma);

#endif //NO CUDA


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
};


/*! \fn void parse_params(char *param_file, struct parameters * parms);
 *  \brief Reads the parameters in the given file into a structure. */
extern void parse_params (char *param_file, struct parameters * parms);


#endif //GLOBAL_H

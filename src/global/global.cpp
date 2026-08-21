/*  \file global.cpp
 *  \brief Global function definitions.*/

#include "../global/global.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>

#include "../io/ParameterMap.h"       // define parameter_map
#include "../io/io.h"                 //defines chprintf
#include "../utils/error_handling.h"  // defines ASSERT

/* Global variables */
Real gama;   // Ratio of specific heats
Real C_cfl;  // CFL number

#ifdef PARTICLES
  #ifdef MPI_CHOLLA
// Constants for the inital size of the buffers for particles transfer
// and the number of data transferred for each particle
int N_PARTICLES_TRANSFER;
int N_DATA_PER_PARTICLE_TRANSFER;
  #endif
#endif

/*! \fn void Set_Gammas(Real gamma_in)
 *  \brief Set gamma values for Riemann solver */
void Set_Gammas(Real gamma_in)
{
  // set gamma
  gama = gamma_in;
  CHOLLA_ASSERT(gama > 1.0, "Gamma must be greater than one.");
}

/*! \fn double Get_Time(void)
 *  \brief Returns the current clock time. */
double Get_Time(void)
{
  struct timeval timer;
  gettimeofday(&timer, NULL);
  return timer.tv_sec + 1.0e-6 * timer.tv_usec;
}

/*! \fn int Sgn
 *  \brief Mathematical sign function. Returns sign of x. */
int Sgn(Real x)
{
  if (x < 0) {
    return -1;
  } else {
    return 1;
  }
}

// global mpi-related variables (they are declared here because they are initialized even when
// the MPI_CHOLLA variable is not defined)

int procID; /*process rank*/
int nproc;  /*number of processes in global comm*/
int root;   /*rank of root process*/

/* Used when MPI_CHOLLA is not defined to initialize a subset of the global mpi-related variables
 * that still meaningful in non-mpi simulations.
 */
void Init_Global_Parallel_Vars_No_MPI()
{
#ifdef MPI_CHOLLA
  CHOLLA_ERROR("This function should not be executed when compiled with MPI");
#endif
  procID = 0;
  nproc  = 1;
  root   = 0;
}

/*! \fn char Trim(char *s)
 *  \brief Gets rid of trailing and leading whitespace. */
char *Trim(char *s)
{
  /* Initialize start, end pointers */
  char *s1 = s, *s2 = &s[strlen(s) - 1];

  /* Trim and delimit right side */
  while ((isspace(*s2)) && (s2 >= s1)) {
    s2--;
  }
  *(s2 + 1) = '\0';

  /* Trim left side */
  while ((isspace(*s1)) && (s1 < s2)) {
    s1++;
  }

  /* Copy finished string */
  strcpy(s, s1);
  return s;
}

// NOLINTNEXTLINE(cert-err58-cpp)
// NOLINTNEXTLINE(*)
const std::set<std::string> optionalParams = {"flag_delta",
                                              "ddelta_dt",
                                              "n_delta",
                                              "Lz",
                                              "Lx",
                                              "phi",
                                              "theta",
                                              "delta",
                                              "nzr",
                                              "nxr",
                                              "H0",
                                              "Omega_M",
                                              "Omega_L",
                                              "Omega_R",
                                              "Omega_K",
                                              "w0",
                                              "wa",
                                              "Init_redshift",
                                              "End_redshift",
                                              "tile_length",
                                              "outstep_dexinc",
                                              "max_timestep_dexinc",
                                              "max_timestep"};  // NOLINT //BRANT

void Warn_Unused_Params(ParameterMap &pmap) { pmap.warn_unused_parameters(optionalParams); }

/*! \brief this would be entirely unnecessary if the Parameters struct directly stored a std::string
 */
static void Load_String_Param_Into_Char_Buffer(ParameterMap &pmap, const std::string &param, char *dest_buffer,
                                               const char *dflt_val)
{
  std::string tmp;
  if (dflt_val == nullptr) {
    tmp = pmap.value<std::string>(param);  // an error is reported when the parameter isn't specified
  } else {
    tmp = pmap.value_or(param, dflt_val);
  }
  // according to strncpy documentation, MAXLEN include the nul-terminator character
  // (aside: tmp.size() does not include a nul-terminator character)
  CHOLLA_ASSERT((tmp.size() + 1) <= MAXLEN,
                "the \"%s\" parameter's value is too long. It must be shorter than %d characters", param.c_str(),
                MAXLEN);
  strncpy(dest_buffer, tmp.c_str(), MAXLEN);
}

Parameters::Parameters(ParameterMap &pmap)
{
  Parameters *parms = this;  // <- this is a minor hack to avoid making an enormous diff

  // load the domain dimensions (abort with an error if one of these is missing)
  parms->nx = pmap.value<int>("nx");
  parms->ny = pmap.value<int>("ny");
  parms->nz = pmap.value<int>("nz");

  CHOLLA_ASSERT((parms->nx >= 0) and (parms->ny >= 0) and (parms->nz >= 0), "domain dimensions must be positive");

  // parse the position of the lower left corner of the simulation domain
  parms->xmin = pmap.value<double>("xmin");
  parms->ymin = pmap.value<double>("ymin");
  parms->zmin = pmap.value<double>("zmin");

  // parse the lengths of each domain dimension
  parms->xlen = pmap.value<double>("xlen");
  parms->ylen = pmap.value<double>("ylen");
  parms->zlen = pmap.value<double>("zlen");
  CHOLLA_ASSERT((parms->xlen > 0) and (parms->ylen > 0) and (parms->zlen > 0), "xlen, ylen, & zlen must be positive");

  // Set the MPI Processes grid [n_proc_x, n_proc_y, n_proc_z]
  if (pmap.has_param("n_proc_x") or pmap.has_param("n_proc_y") or pmap.has_param("n_proc_z")) {
    parms->n_proc_x = pmap.value<int>("n_proc_x");
    parms->n_proc_y = pmap.value<int>("n_proc_y");
    parms->n_proc_z = pmap.value<int>("n_proc_z");
    CHOLLA_ASSERT((parms->n_proc_x > 0) and (parms->n_proc_y > 0) and (parms->n_proc_z > 0),
                  "When specified, n_proc_x, n_proc_y, and n_proc_z must be positive");
    // the following check also implicitly ensures that n_proc_[xyz] are all 1 without MPI
    int product = parms->n_proc_x * parms->n_proc_y * parms->n_proc_z;
    CHOLLA_ASSERT(product == nproc,
                  "The product of n_proc_x, n_proc_y, and n_proc_z is %d. It doesn't match the "
                  "number of processes, %d",
                  product, nproc);
  } else {
    parms->n_proc_x = 0;
    parms->n_proc_y = 0;
    parms->n_proc_z = 0;
  }

  // load boundary conditions
  parms->xl_bcnd = pmap.value<int>("xl_bcnd");
  parms->xu_bcnd = pmap.value<int>("xu_bcnd");
  parms->yl_bcnd = pmap.value<int>("yl_bcnd");
  parms->yu_bcnd = pmap.value<int>("yu_bcnd");
  parms->zl_bcnd = pmap.value<int>("zl_bcnd");
  parms->zu_bcnd = pmap.value<int>("zu_bcnd");

#ifdef STATIC_GRAV
  parms->custom_grav = pmap.value_or("custom_grav", 0);
#endif

  parms->tout = pmap.value<double>("tout");  // aborts if missing
  CHOLLA_ASSERT(parms->tout >= 0.0, "tout parameter must be non-negative");

  parms->outstep             = pmap.value<double>("outstep");                    // aborts if missing
  parms->outstep_dexinc      = Real(pmap.value_or("outstep_dexinc", 0.0));       // BRANT
  parms->max_timestep_dexinc = Real(pmap.value_or("max_timestep_dexinc", 0.0));  // BRANT
  parms->max_timestep        = Real(pmap.value_or("max_timestep", 0.0));         // BRANT
  parms->n_steps_output      = pmap.value_or("n_steps_output", 0);

  // in the future, maybe we should provide a default value of 5/3 for gamma
  parms->gamma = Real(pmap.value<double>("gamma"));
  CHOLLA_ASSERT(parms->gamma > 1.0, "gamma parameter must be greater than one.");

  // load in a handful of string parameters (this would look a lot more like parsing other parameters if we
  // stored the values as std::string values)
  Load_String_Param_Into_Char_Buffer(pmap, "init", parms->init, "");
  Load_String_Param_Into_Char_Buffer(pmap, "custom_bcnd", parms->custom_bcnd, "");
  Load_String_Param_Into_Char_Buffer(pmap, "indir", parms->indir, "");

  // Deal with the gravity.gas_only_use_static_grav parameter
  // - it would be great to move reading of this parameter to the Gravity class (that would probably
  //   require us to unify STATIC_GRAV and GRAVITY)
  // - the following flag is only meaningful when GRAVITY and GRAVITY_ANALYTIC_COMP
  //   are defined.
  // - In other cases, we raise an error if specified without a sensible value.
#if defined(GRAVITY) && defined(GRAVITY_ANALYTIC_COMP)
  parms->gas_only_use_static_grav = pmap.value_or("gravity.gas_only_use_static_grav", false);
#elif defined(GRAVITY)
  parms->gas_only_use_static_grav = pmap.value_or("gravity.gas_only_use_static_grav", false);
  CHOLLA_ASSERT(parms->gas_only_use_static_grav == false,
                "It is an error to set gravity.gas_only_use_static_grav to `true` when Cholla is compiled with "
                "GRAVITY but not GRAVITY_ANALYTIC_COMP");
#elif defined(STATIC_GRAV)
  parms->gas_only_use_static_grav = pmap.value_or("gravity.gas_only_use_static_grav", true);
  CHOLLA_ASSERT(
      parms->gas_only_use_static_grav == true,
      "It is an error to set gravity.gas_only_use_static_grav to `true` when Cholla is compiled with STATIC_GRAV");
#else
  CHOLLA_ASSERT(not pmap.has_param("gravity.gas_only_use_static_grav"),
                "it doesn't make sense to specify gravity.gas_only_use_static_grav when cholla isn't compiled "
                "with gravity");
  parms->gas_only_use_static_grav = false;
#endif

  // ideally, we would only try to parse this for certain values of parms->init
  parms->nfile = pmap.value_or("nfile", 0);

  // load in values related to initial conditions
  //
  // In the future, we **REALLY**, want to only load the values when/where we use them.
  // - This usually (maybe always?) means, within the initial-condition method of Grid3D
  // - The benefit of doing this: we will be able to warn if a parameter is specified
  //   but not used (this is REALLY easy to accidentally do)
  //
  // We need to keep 2 things in mind while doing that:
  // 1. We want to be EXTREMELY sure that a single execution of Cholla never reads a
  //    parameter more than once
  //    - in this scenario, it would be extremely easy for different parts of the code
  //      to accidentally start using different default values, which would introduce
  //      lots of hard to debug issues
  //    - while there are some potential workarounds to this, I think they might
  //      produce some undesired long-term behavior (plus, they are imperfect)
  // 2. We may want to consider renaming the parameters. For example, maybe
  //    Grid3D::Sound_Wave should read in "IC.Sound_Wave.rho" instead of just "rho".
  //    This is useful in 2 regards:
  //    a) this helps us avoid situations where we accidentally read in a parameter
  //       more than once
  //    b) we can reduce the number of parameters with extremely generic names in the
  //       parameter file
  parms->rho                 = pmap.value_or("rho", 0.0);
  parms->vx                  = pmap.value_or("vx", 0.0);
  parms->vy                  = pmap.value_or("vy", 0.0);
  parms->vz                  = pmap.value_or("vz", 0.0);
  parms->P                   = pmap.value_or("P", 0.0);
  parms->Bx                  = pmap.value_or("Bx", 0.0);
  parms->By                  = pmap.value_or("By", 0.0);
  parms->Bz                  = pmap.value_or("Bz", 0.0);
  parms->A                   = pmap.value_or("A", 0.0);
  parms->rho_l               = pmap.value_or("rho_l", 0.0);
  parms->vx_l                = pmap.value_or("vx_l", 0.0);
  parms->vy_l                = pmap.value_or("vy_l", 0.0);
  parms->vz_l                = pmap.value_or("vz_l", 0.0);
  parms->P_l                 = pmap.value_or("P_l", 0.0);
  parms->Bx_l                = pmap.value_or("Bx_l", 0.0);
  parms->By_l                = pmap.value_or("By_l", 0.0);
  parms->Bz_l                = pmap.value_or("Bz_l", 0.0);
  parms->rho_r               = pmap.value_or("rho_r", 0.0);
  parms->vx_r                = pmap.value_or("vx_r", 0.0);
  parms->vy_r                = pmap.value_or("vy_r", 0.0);
  parms->vz_r                = pmap.value_or("vz_r", 0.0);
  parms->P_r                 = pmap.value_or("P_r", 0.0);
  parms->Bx_r                = pmap.value_or("Bx_r", 0.0);
  parms->By_r                = pmap.value_or("By_r", 0.0);
  parms->Bz_r                = pmap.value_or("Bz_r", 0.0);
  parms->diaph               = pmap.value_or("diaph", 0.0);
  parms->rEigenVec_rho       = pmap.value_or("rEigenVec_rho", 0.0);
  parms->rEigenVec_MomentumX = pmap.value_or("rEigenVec_MomentumX", 0.0);
  parms->rEigenVec_MomentumY = pmap.value_or("rEigenVec_MomentumY", 0.0);
  parms->rEigenVec_MomentumZ = pmap.value_or("rEigenVec_MomentumZ", 0.0);
  parms->rEigenVec_E         = pmap.value_or("rEigenVec_E", 0.0);
  parms->rEigenVec_Bx        = pmap.value_or("rEigenVec_Bx", 0.0);
  parms->rEigenVec_By        = pmap.value_or("rEigenVec_By", 0.0);
  parms->rEigenVec_Bz        = pmap.value_or("rEigenVec_Bz", 0.0);
  parms->pitch               = pmap.value_or("pitch", 0.0);
  parms->yaw                 = pmap.value_or("yaw", 0.0);
  parms->polarization        = pmap.value_or("polarization", 0.0);
  parms->radius              = pmap.value_or("radius", 0.0);
  parms->P_blast             = pmap.value_or("P_blast", 0.0);
  parms->wave_length         = pmap.value_or("wave_length", 1.0);

#ifdef TILED_INITIAL_CONDITIONS
  parms->tile_length = pmap.value<double>("tile_length");
#endif  // TILED_INITIAL_CONDITIONS

  // parse some assorted values (we should parse them only where we need them)
  {
    int tmp = pmap.value_or("output_always", 0);
    CHOLLA_ASSERT((tmp == 0) or (tmp == 1), "output_always must be 1 or 0.");
    parms->output_always = tmp;
  }

  parms->n_steps_limit = pmap.value_or("n_steps_limit", -1);

#ifdef PARTICLES
  parms->prng_seed = pmap.value_or("prng_seed", 0);
#endif  // PARTICLES

  // a negative value means that the parameter wasn't set
  parms->bc_potential_type = pmap.value_or("bc_potential_type", -1);

#if defined(SCALAR) && defined(DUST)
  parms->grain_radius = pmap.value<double>("grain_radius");
#endif  // defined(SCALAR) && defined(DUST)

  // in the future, it would probably be good to move this logic into Cosmology::Initialize (or somewhere similar)
  // and remove these parameters from the global struct. This would provide a few benefits:
  //   - we could take steps towards removing the optionalParams global variable (there is alternative machinery
  //     in place to check for unused parameters)
  //   - we could remove ifdef statements from here and the global Parameters struct (we would also remove the
  //     these parameters from the Parameters struct)
  //
  // Prior to relocating this parameter-parsing, Cosmological simulations simply assumed that all of these parameters
  // were specified (& there were no default values). Now cosmological simulations will loudly fail if a user forgets
  // parameters like H0, Omega_M, Omega_L, Omega_b, etc.
#ifdef COSMOLOGY
  parms->scale_outputs_file[0] = '\0';  // <- unclear how necessary this is
  if (not pmap.has_param("End_redshift") and not pmap.has_param("scale_outputs_file")) {
    CHOLLA_ERROR("either the scale_outputs_file or End_redshift parameter must be provided in Cosmology sims");
  } else {
    Load_String_Param_Into_Char_Buffer(pmap, "scale_outputs_file", parms->scale_outputs_file, "");
    parms->End_redshift = pmap.value_or("End_redshift", 0.0);
  }
  // it turns out that Init_redshift is only needed for special test-problems
  // -> it commonly isn't given a value.
  // -> Since it never had a default value before, we have it fall back to an obviously wrong value
  parms->Init_redshift = pmap.value_or("Init_redshift", -1.0);
  parms->H0            = pmap.value<double>("H0");
  parms->Omega_M       = pmap.value<double>("Omega_M");
  parms->Omega_L       = pmap.value<double>("Omega_L");
  parms->Omega_b       = pmap.value<double>("Omega_b");
  parms->Omega_R       = pmap.value_or("Omega_R", 0.0);
  parms->T_init        = pmap.value_or("T_init", -1.0);
  parms->w0            = pmap.value_or("w0", -1.0);
  parms->wa            = pmap.value_or("wa", 0.0);
  parms->cosmo_ics_seed = pmap.value_or("cosmo_ics_seed", 1337);
  // Hydrogen, Helium ionization fractions and helium mass fraction
  parms->YHe           = pmap.value_or("YHe", 0.24);
  parms->xHp_ion_init  = pmap.value_or("xHp_ion_init", 0.0);
  parms->xHep_ion_init = pmap.value_or("xHep_ion_init", 0.0);

  Load_String_Param_Into_Char_Buffer(pmap, "cosmo_ics_pk_file", parms->cosmo_ics_pk_file, "Pk.txt");
  if (pmap.has_param("cosmo_ics_pk_file")) {
    chprintf("Power spectrum file: %s\n", parms->cosmo_ics_pk_file);
  }

  // if the wDE table isn't provided, store an empty string
  parms->wDE_file = pmap.value_or("wDE_file", "");

#endif  // COSMOLOGY

#if defined(CHEMISTRY_GPU) || defined(COOLING_GRACKLE)
  // Not all chemistry_gpu will have a rates file
  if (pmap.has_param("UVB_rates_file")) {
    Load_String_Param_Into_Char_Buffer(pmap, "UVB_rates_file", parms->UVB_rates_file, nullptr);
    chprintf("UVB_rates_file %s\n", parms->UVB_rates_file);
  }
#endif

  // number of RT iterations
#ifdef RT
  parms->rt_num_iterations = pmap.value_or("rt_num_iterations", 10);
#endif

  // we should probably revisit this section and come up with different default behaviors.
  // -> for right now, we just use dummy defaults (for everything other that skewersdir) to make sure
  //    we won't break things
  // -> previously, there weren't any defaults
#ifdef ANALYSIS
  Load_String_Param_Into_Char_Buffer(pmap, "analysis_scale_outputs_file", parms->analysis_scale_outputs_file, "");
  Load_String_Param_Into_Char_Buffer(pmap, "analysisdir", parms->analysis_scale_outputs_file, "");
  parms->lya_skewers_stride = pmap.value_or("lya_skewers_stride", 0);
  parms->lya_Pk_d_log_k     = pmap.value_or("lya_Pk_d_log_k", 0.0);
  #ifdef OUTPUT_SKEWERS
  Load_String_Param_Into_Char_Buffer(pmap, "skewersdir", parms->skewersdir, nullptr);
  #endif
#endif

#ifdef TEMPERATURE_FLOOR
  if (not pmap.has_param("temperature_floor")) {
    chprintf("WARNING: parameter file doesn't include temperature_floor parameter. Defaulting to value of 0!\n");
  }
  parms->temperature_floor = pmap.value_or("temperature_floor", 0.0);
#endif
#ifdef DENSITY_FLOOR
  if (not pmap.has_param("density_floor")) {
    chprintf("WARNING: parameter file doesn't include density_floor parameter. Defaulting to value of 0!\n");
  }
  parms->density_floor = pmap.value_or("density_floor", 0.0);
#endif
#ifdef SCALAR_FLOOR
  if (not pmap.has_param("scalar_floor")) {
    chprintf("WARNING: parameter file doesn't include scalar_floor parameter. Defaulting to value of 0!\n");
  }
  parms->scalar_floor = pmap.value_or("scalar_floor", 0.0);
#endif
}

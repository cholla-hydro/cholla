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
const std::set<std::string> optionalParams = {"flag_delta",   "ddelta_dt",  "n_delta", "Lz",  "Lx", "phi",
                                              "theta",        "delta",      "nzr",     "nxr", "H0", "Omega_M",
                                              "Omega_L",      "Omega_R",    "Omega_K", "w0",  "wa", "Init_redshift",
                                              "End_redshift", "tile_length"};  // NOLINT

bool Old_Style_Parse_Param(const char *name, const char *value, struct Parameters *parms);

void Init_Param_Struct_Members(ParameterMap &param, struct Parameters *parms);

void Parse_Params(ParameterMap &pmap, struct Parameters *parms)
{
#ifdef COSMOLOGY
  // Initialize file name as an empty string
  parms->scale_outputs_file[0] = '\0';
#endif

  // the plan is eventually replace Old_Style_Parse_Param entirely with
  // Init_Param_Struct_Members.
  auto fn = [&](const char *name, const char *value) -> bool { return Old_Style_Parse_Param(name, value, parms); };

  pmap.pass_entries_to_legacy_parse_param(fn);

  // the plan is to eventually, use the new parsing functions from Parse_Param like the following
  Init_Param_Struct_Members(pmap, parms);

}

void Warn_Unused_Params(ParameterMap &pmap) { pmap.warn_unused_parameters(optionalParams); }

/*! \fn void Parse_Param(char *name,char *value, struct Parameters *parms);
 *  \brief Parses and sets a single param based on name and value.
 *
 *  \returns true if the parameter was actually used. false otherwise.
 */
bool Old_Style_Parse_Param(const char *name, const char *value, struct Parameters *parms)
{
  /* Copy into correct entry in parameters struct */
  if (strcmp(name, "nfile") == 0) {
    parms->nfile = atoi(value);
  } else if (strcmp(name, "output_always") == 0) {
    int tmp = atoi(value);
    // In this case the CHOLLA_ASSERT macro runs into issuse with the readability-simplify-boolean-expr clang-tidy check
    // due to some weird macro expansion stuff. That check has been disabled here for now but in clang-tidy 18 the
    // IgnoreMacro option should be used instead.
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    CHOLLA_ASSERT((tmp == 0) or (tmp == 1), "output_always must be 1 or 0.");
    parms->output_always = tmp;
  } else if (strcmp(name, "legacy_flat_outdir") == 0) {
    int tmp = atoi(value);
    CHOLLA_ASSERT((tmp == 0) or (tmp == 1), "legacy_flat_outdir must be 1 or 0.");
    parms->legacy_flat_outdir = tmp;
  } else if (strcmp(name, "n_steps_limit") == 0) {
    parms->n_steps_limit = atof(value);
  } else if (strcmp(name, "xmin") == 0) {
    parms->xmin = atof(value);
  } else if (strcmp(name, "ymin") == 0) {
    parms->ymin = atof(value);
  } else if (strcmp(name, "zmin") == 0) {
    parms->zmin = atof(value);
  } else if (strcmp(name, "xlen") == 0) {
    parms->xlen = atof(value);
  } else if (strcmp(name, "ylen") == 0) {
    parms->ylen = atof(value);
  } else if (strcmp(name, "zlen") == 0) {
    parms->zlen = atof(value);
  } else if (strcmp(name, "xl_bcnd") == 0) {
    parms->xl_bcnd = atoi(value);
  } else if (strcmp(name, "xu_bcnd") == 0) {
    parms->xu_bcnd = atoi(value);
  } else if (strcmp(name, "yl_bcnd") == 0) {
    parms->yl_bcnd = atoi(value);
  } else if (strcmp(name, "yu_bcnd") == 0) {
    parms->yu_bcnd = atoi(value);
  } else if (strcmp(name, "zl_bcnd") == 0) {
    parms->zl_bcnd = atoi(value);
  } else if (strcmp(name, "zu_bcnd") == 0) {
    parms->zu_bcnd = atoi(value);
  } else if (strcmp(name, "rho") == 0) {
    parms->rho = atof(value);
  } else if (strcmp(name, "vx") == 0) {
    parms->vx = atof(value);
  } else if (strcmp(name, "vy") == 0) {
    parms->vy = atof(value);
  } else if (strcmp(name, "vz") == 0) {
    parms->vz = atof(value);
  } else if (strcmp(name, "P") == 0) {
    parms->P = atof(value);
  } else if (strcmp(name, "Bx") == 0) {
    parms->Bx = atof(value);
  } else if (strcmp(name, "By") == 0) {
    parms->By = atof(value);
  } else if (strcmp(name, "Bz") == 0) {
    parms->Bz = atof(value);
  } else if (strcmp(name, "A") == 0) {
    parms->A = atof(value);
  } else if (strcmp(name, "rho_l") == 0) {
    parms->rho_l = atof(value);
  } else if (strcmp(name, "vx_l") == 0) {
    parms->vx_l = atof(value);
  } else if (strcmp(name, "vy_l") == 0) {
    parms->vy_l = atof(value);
  } else if (strcmp(name, "vz_l") == 0) {
    parms->vz_l = atof(value);
  } else if (strcmp(name, "P_l") == 0) {
    parms->P_l = atof(value);
  } else if (strcmp(name, "Bx_l") == 0) {
    parms->Bx_l = atof(value);
  } else if (strcmp(name, "By_l") == 0) {
    parms->By_l = atof(value);
  } else if (strcmp(name, "Bz_l") == 0) {
    parms->Bz_l = atof(value);
  } else if (strcmp(name, "rho_r") == 0) {
    parms->rho_r = atof(value);
  } else if (strcmp(name, "vx_r") == 0) {
    parms->vx_r = atof(value);
  } else if (strcmp(name, "vy_r") == 0) {
    parms->vy_r = atof(value);
  } else if (strcmp(name, "vz_r") == 0) {
    parms->vz_r = atof(value);
  } else if (strcmp(name, "P_r") == 0) {
    parms->P_r = atof(value);
  } else if (strcmp(name, "Bx_r") == 0) {
    parms->Bx_r = atof(value);
  } else if (strcmp(name, "By_r") == 0) {
    parms->By_r = atof(value);
  } else if (strcmp(name, "Bz_r") == 0) {
    parms->Bz_r = atof(value);
  } else if (strcmp(name, "diaph") == 0) {
    parms->diaph = atof(value);
  } else if (strcmp(name, "rEigenVec_rho") == 0) {
    parms->rEigenVec_rho = atof(value);
  } else if (strcmp(name, "rEigenVec_MomentumX") == 0) {
    parms->rEigenVec_MomentumX = atof(value);
  } else if (strcmp(name, "rEigenVec_MomentumY") == 0) {
    parms->rEigenVec_MomentumY = atof(value);
  } else if (strcmp(name, "rEigenVec_MomentumZ") == 0) {
    parms->rEigenVec_MomentumZ = atof(value);
  } else if (strcmp(name, "rEigenVec_E") == 0) {
    parms->rEigenVec_E = atof(value);
  } else if (strcmp(name, "rEigenVec_Bx") == 0) {
    parms->rEigenVec_Bx = atof(value);
  } else if (strcmp(name, "rEigenVec_By") == 0) {
    parms->rEigenVec_By = atof(value);
  } else if (strcmp(name, "rEigenVec_Bz") == 0) {
    parms->rEigenVec_Bz = atof(value);
  } else if (strcmp(name, "pitch") == 0) {
    parms->pitch = atof(value);
  } else if (strcmp(name, "yaw") == 0) {
    parms->yaw = atof(value);
  } else if (strcmp(name, "polarization") == 0) {
    parms->polarization = atof(value);
  } else if (strcmp(name, "radius") == 0) {
    parms->radius = atof(value);
  } else if (strcmp(name, "P_blast") == 0) {
    parms->P_blast = atof(value);
  } else if (strcmp(name, "wave_length") == 0) {
    parms->wave_length = atof(value);
#ifdef PARTICLES
  } else if (strcmp(name, "prng_seed") == 0) {
    parms->prng_seed = atoi(value);
#endif  // PARTICLES
#ifdef TILED_INITIAL_CONDITIONS
  } else if (strcmp(name, "tile_length") == 0) {
    parms->tile_length = atof(value);
#endif  // TILED_INITIAL_CONDITIONS

  } else if (strcmp(name, "bc_potential_type") == 0) {
    parms->bc_potential_type = atoi(value);
#ifdef SCALAR
  #ifdef DUST
  } else if (strcmp(name, "grain_radius") == 0) {
    parms->grain_radius = atoi(value);
  #endif
#endif
  } else {
    return false;
  }
  return true;
}

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

/*! \brief Parses and sets a bunch of members of parms from pmap.
 *
 *  The goal is eventually get rid of the old-style function
 */
void Init_Param_Struct_Members(ParameterMap &pmap, struct Parameters *parms)
{
  // load the domain dimensions (abort with an error if one of these is missing)
  parms->nx = pmap.value<int>("nx");
  parms->ny = pmap.value<int>("ny");
  parms->nz = pmap.value<int>("nz");

  CHOLLA_ASSERT((parms->nx >= 0) and (parms->ny >= 0) and (parms->nz >= 0), "domain dimensions must be positive");
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

#ifdef STATIC_GRAV
  parms->custom_grav = pmap.value_or("custom_grav", 0);
#endif

  parms->tout = pmap.value<double>("tout");  // aborts if missing
  CHOLLA_ASSERT(parms->tout >= 0.0, "tout parameter must be non-negative");

  parms->outstep        = pmap.value<double>("outstep");  // aborts if missing
  parms->n_steps_output = pmap.value_or("n_steps_output", 0);

  // in the future, maybe we should provide a default value of 5/3 for gamma
  parms->gamma = Real(pmap.value<double>("gamma"));
  CHOLLA_ASSERT(parms->gamma > 1.0, "gamma parameter must be greater than one.");

  // load in a handful of string parameters (this would look a lot more like parsing other parameters if we
  // stored the values as std::string values)
  Load_String_Param_Into_Char_Buffer(pmap, "init", parms->init, "");
  Load_String_Param_Into_Char_Buffer(pmap, "custom_bcnd", parms->custom_bcnd, "");
  Load_String_Param_Into_Char_Buffer(pmap, "outdir", parms->outdir, "");
  Load_String_Param_Into_Char_Buffer(pmap, "indir", parms->indir, "");

  // in the future, the feedback module will read in its own parameters (the global Parameter struct won't
  // know anything about it)
#ifdef FEEDBACK
  #ifndef NO_SN_FEEDBACK
  Load_String_Param_Into_Char_Buffer(pmap, "snr_filename", parms->snr_filename, "");
  #endif
  #ifndef NO_WIND_FEEDBACK
  Load_String_Param_Into_Char_Buffer(pmap, "sw_filename", parms->sw_filename, "");
  #endif
#endif

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
  parms->seed          = pmap.value_or("seed", 1337);
  if (not pmap.has_param("cosmo_ics_pk_file")) {
    CHOLLA_ERROR("Cosmology sims must specify a power spectrum");
  }
  Load_String_Param_Into_Char_Buffer(pmap, "cosmo_ics_pk_file", parms->cosmo_ics_pk_file, "Pk.txt");
  chprintf("Power spectrum file: %s\n",parms->cosmo_ics_pk_file);
#endif  // COSMOLOGY

#if defined(CHEMISTRY_GPU) || defined(COOLING_GRACKLE)
  Load_String_Param_Into_Char_Buffer(pmap, "UVB_rates_file", parms->UVB_rates_file, nullptr);
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

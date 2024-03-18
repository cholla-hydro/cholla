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
const std::set<const char *> optionalParams = {
    "flag_delta",   "ddelta_dt",   "n_delta",  "Lz",       "Lx",      "phi",     "theta",
    "delta",        "nzr",         "nxr",      "H0",       "Omega_M", "Omega_L", "Init_redshift",
    "End_redshift", "tile_length", "n_proc_x", "n_proc_y", "n_proc_z"};

/*! \fn int Is_Param_Valid(char *name);
 * \brief Verifies that a param is valid (even if not needed).  Avoids
 * "warnings" in output. */
int Is_Param_Valid(const char *param_name)
{
  // for (auto optionalParam = optionalParams.begin(); optionalParam != optionalParams.end(); ++optionalParam) {
  for (const auto *optionalParam : optionalParams) {
    if (strcmp(param_name, optionalParam) == 0) {
      return 1;
    }
  }
  return 0;
}

void Parse_Param(char *name, char *value, struct Parameters *parms);

/*! \fn void Parse_Params(char *param_file, struct Parameters * parms);
 *  \brief Reads the parameters in the given file into a structure. */
void Parse_Params(char *param_file, struct Parameters *parms, int argc, char **argv)
{
  int buf;
  char *s, buff[256];
  FILE *fp = fopen(param_file, "r");
  if (fp == NULL) {
    chprintf("Exiting at file %s line %d: failed to read param file %s \n", __FILE__, __LINE__, param_file);
    exit(1);
    return;
  }

#ifdef COSMOLOGY
  // Initialize file name as an empty string
  parms->scale_outputs_file[0] = '\0';
#endif

  /* Read next line */
  while ((s = fgets(buff, sizeof buff, fp)) != NULL) {
    /* Skip blank lines and comments */
    if (buff[0] == '\n' || buff[0] == '#' || buff[0] == ';') {
      continue;
    }

    /* Parse name/value pair from line */
    char name[MAXLEN], value[MAXLEN];
    s = strtok(buff, "=");
    if (s == NULL) {
      continue;
    } else {
      strncpy(name, s, MAXLEN);
    }
    s = strtok(NULL, "=");
    if (s == NULL) {
      continue;
    } else {
      strncpy(value, s, MAXLEN);
    }
    Trim(value);
    Parse_Param(name, value, parms);
  }
  /* Close file */
  fclose(fp);

  // Parse overriding args from command line
  for (int i = 0; i < argc; ++i) {
    char name[MAXLEN], value[MAXLEN];
    s = strtok(argv[i], "=");
    if (s == NULL) {
      continue;
    } else {
      strncpy(name, s, MAXLEN);
    }
    s = strtok(NULL, "=");
    if (s == NULL) {
      continue;
    } else {
      strncpy(value, s, MAXLEN);
    }
    Parse_Param(name, value, parms);
    chprintf("Override with %s=%s\n", name, value);
  }
#ifdef TEMPERATURE_FLOOR
  if (parms->temperature_floor == 0) {
    chprintf(
        "WARNING: temperature floor is set to its default value (zero)! It can be set to a different value in the "
        "input parameter file.\n");
  }
#endif
#ifdef DENSITY_FLOOR
  if (parms->density_floor == 0) {
    chprintf(
        "WARNING: density floor is set to its default value (zero)! It can be set to a different value in the input "
        "parameter file.\n");
  }
#endif
#ifdef SCALAR_FLOOR
  if (parms->scalar_floor == 0) {
    chprintf(
        "WARNING: scalar floor is set to its default value (zero)! It can be set to a different value in the input "
        "parameter file.\n");
  }
#endif
}

/*! \fn void Parse_Param(char *name,char *value, struct Parameters *parms);
 *  \brief Parses and sets a single param based on name and value. */
void Parse_Param(char *name, char *value, struct Parameters *parms)
{
  /* Copy into correct entry in parameters struct */
  if (strcmp(name, "nx") == 0) {
    parms->nx = atoi(value);
  } else if (strcmp(name, "ny") == 0) {
    parms->ny = atoi(value);
  } else if (strcmp(name, "nz") == 0) {
    parms->nz = atoi(value);
#ifdef STATIC_GRAV
  } else if (strcmp(name, "custom_grav") == 0) {
    parms->custom_grav = atoi(value);
#endif
  } else if (strcmp(name, "tout") == 0) {
    parms->tout = atof(value);
  } else if (strcmp(name, "outstep") == 0) {
    parms->outstep = atof(value);
  } else if (strcmp(name, "n_steps_output") == 0) {
    parms->n_steps_output = atoi(value);
  } else if (strcmp(name, "gamma") == 0) {
    parms->gamma = atof(value);
  } else if (strcmp(name, "init") == 0) {
    strncpy(parms->init, value, MAXLEN);
  } else if (strcmp(name, "nfile") == 0) {
    parms->nfile = atoi(value);
  } else if (strcmp(name, "n_hydro") == 0) {
    parms->n_hydro = atoi(value);
  } else if (strcmp(name, "n_particle") == 0) {
    parms->n_particle = atoi(value);
  } else if (strcmp(name, "n_projection") == 0) {
    parms->n_projection = atoi(value);
  } else if (strcmp(name, "n_rotated_projection") == 0) {
    parms->n_rotated_projection = atoi(value);
  } else if (strcmp(name, "n_slice") == 0) {
    parms->n_slice = atoi(value);
  } else if (strcmp(name, "n_out_float32") == 0) {
    parms->n_out_float32 = atoi(value);
  } else if (strcmp(name, "out_float32_density") == 0) {
    parms->out_float32_density = atoi(value);
  } else if (strcmp(name, "out_float32_momentum_x") == 0) {
    parms->out_float32_momentum_x = atoi(value);
  } else if (strcmp(name, "out_float32_momentum_y") == 0) {
    parms->out_float32_momentum_y = atoi(value);
  } else if (strcmp(name, "out_float32_momentum_z") == 0) {
    parms->out_float32_momentum_z = atoi(value);
  } else if (strcmp(name, "out_float32_Energy") == 0) {
    parms->out_float32_Energy = atoi(value);
#ifdef DE
  } else if (strcmp(name, "out_float32_GasEnergy") == 0) {
    parms->out_float32_GasEnergy = atoi(value);
#endif  // DE
#ifdef MHD
  } else if (strcmp(name, "out_float32_magnetic_x") == 0) {
    parms->out_float32_magnetic_x = atoi(value);
  } else if (strcmp(name, "out_float32_magnetic_y") == 0) {
    parms->out_float32_magnetic_y = atoi(value);
  } else if (strcmp(name, "out_float32_magnetic_z") == 0) {
    parms->out_float32_magnetic_z = atoi(value);
#endif  // MHD
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
  } else if (strcmp(name, "custom_bcnd") == 0) {
    strncpy(parms->custom_bcnd, value, MAXLEN);
  } else if (strcmp(name, "outdir") == 0) {
    strncpy(parms->outdir, value, MAXLEN);
  } else if (strcmp(name, "indir") == 0) {
    strncpy(parms->indir, value, MAXLEN);
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
#ifdef SUPERNOVA
  } else if (strcmp(name, "snr_filename") == 0) {
    strncpy(parms->snr_filename, value, MAXLEN);
#endif
#ifdef ROTATED_PROJECTION
  } else if (strcmp(name, "nxr") == 0) {
    parms->nxr = atoi(value);
  } else if (strcmp(name, "nzr") == 0) {
    parms->nzr = atoi(value);
  } else if (strcmp(name, "delta") == 0) {
    parms->delta = atof(value);
  } else if (strcmp(name, "theta") == 0) {
    parms->theta = atof(value);
  } else if (strcmp(name, "phi") == 0) {
    parms->phi = atof(value);
  } else if (strcmp(name, "Lx") == 0) {
    parms->Lx = atof(value);
  } else if (strcmp(name, "Lz") == 0) {
    parms->Lz = atof(value);
  } else if (strcmp(name, "n_delta") == 0) {
    parms->n_delta = atoi(value);
  } else if (strcmp(name, "ddelta_dt") == 0) {
    parms->ddelta_dt = atof(value);
  } else if (strcmp(name, "flag_delta") == 0) {
    parms->flag_delta = atoi(value);
#endif /*ROTATED_PROJECTION*/
#ifdef COSMOLOGY
  } else if (strcmp(name, "scale_outputs_file") == 0) {
    strncpy(parms->scale_outputs_file, value, MAXLEN);
  } else if (strcmp(name, "Init_redshift") == 0) {
    parms->Init_redshift = atof(value);
  } else if (strcmp(name, "End_redshift") == 0) {
    parms->End_redshift = atof(value);
  } else if (strcmp(name, "H0") == 0) {
    parms->H0 = atof(value);
  } else if (strcmp(name, "Omega_M") == 0) {
    parms->Omega_M = atof(value);
  } else if (strcmp(name, "Omega_L") == 0) {
    parms->Omega_L = atof(value);
  } else if (strcmp(name, "Omega_b") == 0) {
    parms->Omega_b = atof(value);
#endif  // COSMOLOGY
#ifdef TILED_INITIAL_CONDITIONS
  } else if (strcmp(name, "tile_length") == 0) {
    parms->tile_length = atof(value);
#endif  // TILED_INITIAL_CONDITIONS

#ifdef SET_MPI_GRID
    // Set the MPI Processes grid [n_proc_x, n_proc_y, n_proc_z]
  } else if (strcmp(name, "n_proc_x") == 0) {
    parms->n_proc_x = atoi(value);
  } else if (strcmp(name, "n_proc_y") == 0) {
    parms->n_proc_y = atoi(value);
  } else if (strcmp(name, "n_proc_z") == 0) {
    parms->n_proc_z = atoi(value);
#endif
  } else if (strcmp(name, "bc_potential_type") == 0) {
    parms->bc_potential_type = atoi(value);
#ifdef CHEMISTRY_GPU
  } else if (strcmp(name, "UVB_rates_file") == 0) {
    strncpy(parms->UVB_rates_file, value, MAXLEN);
#endif
#ifdef COOLING_GRACKLE
  } else if (strcmp(name, "UVB_rates_file") == 0) {
    strncpy(parms->UVB_rates_file, value, MAXLEN);
#endif
#ifdef TEMPERATURE_FLOOR
  } else if (strcmp(name, "temperature_floor") == 0) {
    parms->temperature_floor = atof(value);
#endif
#ifdef DENSITY_FLOOR
  } else if (strcmp(name, "density_floor") == 0) {
    parms->density_floor = atof(value);
#endif
#ifdef SCALAR_FLOOR
  } else if (strcmp(name, "scalar_floor") == 0) {
    parms->scalar_floor = atof(value);
#endif
#ifdef ANALYSIS
  } else if (strcmp(name, "analysis_scale_outputs_file") == 0) {
    strncpy(parms->analysis_scale_outputs_file, value, MAXLEN);
  } else if (strcmp(name, "analysisdir") == 0) {
    strncpy(parms->analysisdir, value, MAXLEN);
  } else if (strcmp(name, "lya_skewers_stride") == 0) {
    parms->lya_skewers_stride = atoi(value);
  } else if (strcmp(name, "lya_Pk_d_log_k") == 0) {
    parms->lya_Pk_d_log_k = atof(value);
  #ifdef OUTPUT_SKEWERS
  } else if (strcmp(name, "skewersdir") == 0) {
    strncpy(parms->skewersdir, value, MAXLEN);
  #endif
#endif
#ifdef SUPERNOVA
  } else if (strcmp(name, "supernova_e") == 0) {
    parms->supernova_e = atoi(value);
  } else if (strcmp(name, "supernova_rcl") == 0) {
    parms->supernova_rcl = atof(value);
#endif
#ifdef SCALAR
  #ifdef DUST
  } else if (strcmp(name, "grain_radius") == 0) {
    parms->grain_radius = atoi(value);
  #endif
#endif
  } else if (!Is_Param_Valid(name)) {
    chprintf("WARNING: %s/%s: Unknown parameter/value pair!\n", name, value);
  }
}

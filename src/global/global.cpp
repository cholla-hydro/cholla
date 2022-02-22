/*  \file global.cpp
 *  \brief Global function definitions.*/


#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <ctype.h>
#include "../global/global.h"
#include "../io/io.h" //defines chprintf

/* Global variables */
Real gama; // Ratio of specific heats
Real C_cfl; // CFL number

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
    //set gamma
    gama = gamma_in;

}


/*! \fn double get_time(void)
 *  \brief Returns the current clock time. */
double get_time(void)
{
  struct timeval timer;
  gettimeofday(&timer,NULL);
  return timer.tv_sec + 1.0e-6*timer.tv_usec;
}

/*! \fn int sgn
 *  \brief Mathematical sign function. Returns sign of x. */
int sgn(Real x)
{
    if (x < 0) return -1;
    else return 1;
}

#ifndef CUDA
/*! \fn Real calc_eta(Real cW[], Real gamma)
 *  \brief Calculate the eta value for the H correction. */
Real calc_eta(Real cW[], Real gamma)
{
  Real pl, pr, al, ar;

  pl = (cW[8] - 0.5*(cW[2]*cW[2] + cW[4]*cW[4] + cW[6]*cW[6])/cW[0]) * (gamma-1.0);
  pl = fmax(pl, TINY_NUMBER);
  pr = (cW[9] - 0.5*(cW[3]*cW[3] + cW[5]*cW[5] + cW[7]*cW[7])/cW[1]) * (gamma-1.0);
  pr = fmax(pr, TINY_NUMBER);

  al = sqrt(gamma*pl/cW[0]);
  ar = sqrt(gamma*pr/cW[1]);

  return 0.5*fabs((cW[3]/cW[1] + ar) - (cW[2]/cW[0]-al));

}
#endif //NO CUDA


/*! \fn char trim(char *s)
 *  \brief Gets rid of trailing and leading whitespace. */
char *trim (char * s)
{
  /* Initialize start, end pointers */
  char *s1 = s, *s2 = &s[strlen (s) - 1];

  /* Trim and delimit right side */
  while ( (isspace (*s2)) && (s2 >= s1) )
    s2--;
  *(s2+1) = '\0';

  /* Trim left side */
  while ( (isspace (*s1)) && (s1 < s2) )
    s1++;

  /* Copy finished string */
  strcpy (s, s1);
  return s;
}

const std::set<const char*> optionalParams = {"flag_delta", "ddelta_dt", "n_delta",
       "Lz" , "Lx" , "phi" , "theta", "delta", "nzr", "nxr", "H0", "Omega_M", "Omega_L",
       "Init_redshift", "End_redshift", "tile_length", "n_proc_x", "n_proc_y", "n_proc_z" };

/*! \fn int is_param_valid(char *name);
 * \brief Verifies that a param is valid (even if not needed).  Avoids "warnings" in output. */
int is_param_valid(const char* param_name) {
  for (auto it=optionalParams.begin(); it != optionalParams.end(); ++it) {
      if (strcmp(param_name, *it) == 0) return 1;
  }
  return 0;
}

void parse_param(char *name,char *value, struct parameters *parms);


/*! \fn void parse_params(char *param_file, struct parameters * parms);
 *  \brief Reads the parameters in the given file into a structure. */
void parse_params (char *param_file, struct parameters * parms, int argc, char** argv)
{
  int buf;
  char *s, buff[256];
  FILE *fp = fopen (param_file, "r");
  if (fp == NULL)
  {
    return;
  }
  // set default hydro file output parameter
  parms->outstep_hydro=1;
  parms->outstep_particle=1;
  parms->outstep_slice=1;
  parms->outstep_projection=1;
  parms->outstep_rotated_projection=1;

#ifdef ROTATED_PROJECTION
  //initialize rotation parameters to zero
  parms->delta = 0;
  parms->theta = 0;
  parms->phi   = 0;
  parms->n_delta = 0;
  parms->ddelta_dt = 0;
  parms->flag_delta = 0;
#endif /*ROTATED_PROJECTION*/

#ifdef COSMOLOGY
//Initialize file name as an empty string
parms->scale_outputs_file[0] = '\0';
#endif


  /* Read next line */
  while ((s = fgets (buff, sizeof buff, fp)) != NULL)
  {
    /* Skip blank lines and comments */
    if (buff[0] == '\n' || buff[0] == '#' || buff[0] == ';')
      continue;

    /* Parse name/value pair from line */
    char name[MAXLEN], value[MAXLEN];
    s = strtok (buff, "=");
    if (s==NULL)
      continue;
    else
      strncpy (name, s, MAXLEN);
    s = strtok (NULL, "=");
    if (s==NULL)
      continue;
    else
      strncpy (value, s, MAXLEN);
    trim (value);
    parse_param(name,value,parms);
  }
  /* Close file */
  fclose (fp);

  // Parse overriding args from command line
  for (int i = 0; i < argc; ++i) {
    char name[MAXLEN], value[MAXLEN];
    s = strtok (argv[i], "=");
    if (s==NULL)
      continue;
    else
      strncpy (name, s, MAXLEN);
    s = strtok (NULL, "=");
    if (s==NULL)
      continue;
    else
      strncpy (value, s, MAXLEN);
    parse_param(name,value,parms);
    chprintf("Override with %s=%s\n",name,value);

  }
}

/*! \fn void parse_param(char *name,char *value, struct parameters *parms);
 *  \brief Parses and sets a single param based on name and value. */
void parse_param(char *name,char *value, struct parameters *parms){
  /* Copy into correct entry in parameters struct */
  if (strcmp(name, "nx")==0)
    parms->nx = atoi(value);
  else if (strcmp(name, "ny")==0)
    parms->ny = atoi(value);
  else if (strcmp(name, "nz")==0)
    parms->nz = atoi(value);
  else if (strcmp(name, "tout")==0)
    parms->tout = atof(value);
  else if (strcmp(name, "outstep")==0)
    parms->outstep = atof(value);
  else if (strcmp(name, "n_steps_output")==0)
    parms->n_steps_output = atoi(value);
  else if (strcmp(name, "gamma")==0)
    parms->gamma = atof(value);
  else if (strcmp(name, "init")==0)
    strncpy (parms->init, value, MAXLEN);
  else if (strcmp(name, "nfile")==0)
    parms->nfile = atoi(value);
  else if (strcmp(name, "outstep_hydro")==0)
    parms->outstep_hydro = atoi(value);
  else if (strcmp(name, "xmin")==0)
    parms->xmin = atof(value);
  else if (strcmp(name, "ymin")==0)
    parms->ymin = atof(value);
  else if (strcmp(name, "zmin")==0)
    parms->zmin = atof(value);
  else if (strcmp(name, "xlen")==0)
    parms->xlen = atof(value);
  else if (strcmp(name, "ylen")==0)
    parms->ylen = atof(value);
  else if (strcmp(name, "zlen")==0)
    parms->zlen = atof(value);
  else if (strcmp(name, "xl_bcnd")==0)
    parms->xl_bcnd = atoi(value);
  else if (strcmp(name, "xu_bcnd")==0)
    parms->xu_bcnd = atoi(value);
  else if (strcmp(name, "yl_bcnd")==0)
    parms->yl_bcnd = atoi(value);
  else if (strcmp(name, "yu_bcnd")==0)
    parms->yu_bcnd = atoi(value);
  else if (strcmp(name, "zl_bcnd")==0)
    parms->zl_bcnd = atoi(value);
  else if (strcmp(name, "zu_bcnd")==0)
    parms->zu_bcnd = atoi(value);
  else if (strcmp(name, "custom_bcnd")==0)
    strncpy (parms->custom_bcnd, value, MAXLEN);
  else if (strcmp(name, "outdir")==0)
    strncpy (parms->outdir, value, MAXLEN);
  else if (strcmp(name, "indir")==0)
    strncpy (parms->indir, value, MAXLEN);
  else if (strcmp(name, "rho")==0)
    parms->rho = atof(value);
  else if (strcmp(name, "vx")==0)
    parms->vx = atof(value);
  else if (strcmp(name, "vy")==0)
    parms->vy = atof(value);
  else if (strcmp(name, "vz")==0)
    parms->vz = atof(value);
  else if (strcmp(name, "P")==0)
    parms->P = atof(value);
  else if (strcmp(name, "A")==0)
    parms->A = atof(value);
  else if (strcmp(name, "rho_l")==0)
    parms->rho_l = atof(value);
  else if (strcmp(name, "v_l")==0)
    parms->v_l = atof(value);
  else if (strcmp(name, "P_l")==0)
    parms->P_l = atof(value);
  else if (strcmp(name, "rho_r")==0)
    parms->rho_r = atof(value);
  else if (strcmp(name, "v_r")==0)
    parms->v_r = atof(value);
  else if (strcmp(name, "P_r")==0)
    parms->P_r = atof(value);
  else if (strcmp(name, "diaph")==0)
    parms->diaph = atof(value);
#ifdef PARTICLES
  else if (strcmp(name, "prng_seed")==0)
    parms->prng_seed = atoi(value);
#endif // PARTICLES
#ifdef ROTATED_PROJECTION
  else if (strcmp(name, "nxr")==0)
    parms->nxr = atoi(value);
  else if (strcmp(name, "nzr")==0)
    parms->nzr = atoi(value);
  else if (strcmp(name, "delta")==0)
    parms->delta = atof(value);
  else if (strcmp(name, "theta")==0)
    parms->theta = atof(value);
  else if (strcmp(name, "phi")==0)
    parms->phi = atof(value);
  else if (strcmp(name, "Lx")==0)
    parms->Lx  = atof(value);
  else if (strcmp(name, "Lz")==0)
    parms->Lz = atof(value);
  else if (strcmp(name, "n_delta")==0)
    parms->n_delta = atoi(value);
  else if (strcmp(name, "ddelta_dt")==0)
    parms->ddelta_dt = atof(value);
  else if (strcmp(name, "flag_delta")==0)
    parms->flag_delta  = atoi(value);
#endif /*ROTATED_PROJECTION*/
#ifdef COSMOLOGY
  else if (strcmp(name, "scale_outputs_file")==0)
    strncpy (parms->scale_outputs_file, value, MAXLEN);
  else if (strcmp(name, "Init_redshift")==0)
    parms->Init_redshift  = atof(value);
  else if (strcmp(name, "End_redshift")==0)
    parms->End_redshift  = atof(value);
  else if (strcmp(name, "H0")==0)
    parms->H0  = atof(value);
  else if (strcmp(name, "Omega_M")==0)
    parms->Omega_M  = atof(value);
  else if (strcmp(name, "Omega_L")==0)
    parms->Omega_L  = atof(value);
  else if (strcmp(name, "Omega_b")==0)
    parms->Omega_b  = atof(value);
#endif //COSMOLOGY
#ifdef TILED_INITIAL_CONDITIONS
  else if (strcmp(name, "tile_length")==0)
    parms->tile_length  = atof(value);
#endif //TILED_INITIAL_CONDITIONS

#ifdef SET_MPI_GRID
  // Set the MPI Processes grid [n_proc_x, n_proc_y, n_proc_z]
  else if (strcmp(name, "n_proc_x")==0)
    parms->n_proc_x  = atoi(value);
  else if (strcmp(name, "n_proc_y")==0)
    parms->n_proc_y  = atoi(value);
  else if (strcmp(name, "n_proc_z")==0)
    parms->n_proc_z  = atoi(value);
#endif
  else if (strcmp(name, "bc_potential_type")==0)
    parms->bc_potential_type  = atoi(value);

#ifdef COOLING_GRACKLE
  else if (strcmp(name, "UVB_rates_file")==0)
    strncpy (parms->UVB_rates_file, value, MAXLEN);
#endif
#ifdef ANALYSIS
  else if (strcmp(name, "analysis_scale_outputs_file")==0)
    strncpy (parms->analysis_scale_outputs_file, value, MAXLEN);
  else if (strcmp(name, "analysisdir")==0)
    strncpy (parms->analysisdir, value, MAXLEN);
  else if (strcmp(name, "lya_skewers_stride")==0)
    parms->lya_skewers_stride  = atoi(value);
  else if (strcmp(name, "lya_Pk_d_log_k")==0)
    parms->lya_Pk_d_log_k  = atof(value);
#endif
  else if (!is_param_valid(name))
    chprintf ("WARNING: %s/%s: Unknown parameter/value pair!\n",
	    name, value);
}

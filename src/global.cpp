/*  \file global.cpp
 *  \brief Global function definitions.*/


#include<math.h>
#include<sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include"global.h"


/* Global variables */
Real gama; // Ratio of specific heats
Real t_comm;
Real t_other;




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

#ifndef CUDA
/*! \fn Real maxof3(Real a, Real b, Real c)
 *  \brief Returns the maximum of three floating point numbers. */
Real maxof3(Real a, Real b, Real c)
{
    return fmax(a, fmax(b,c));
}


/*! \fn Real minof3(Real a, Real b, Real c)
 *  \brief Returns the minimum of three floating point numbers. */
Real minof3(Real a, Real b, Real c)
{
    return fmin(a, fmin(b,c));
}


/*! \fn int sgn
 *  \brief Mathematical sign function. Returns sign of x. */
int sgn(Real x)
{
    if (x < 0) return -1;
    else return 1;
}

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


/*! \fn void parse_params(char *param_file, struct parameters * parms);
 *  \brief Reads the parameters in the given file into a structure. */
void parse_params (char *param_file, struct parameters * parms)
{
  int buf;
  char *s, buff[256];
  FILE *fp = fopen (param_file, "r");
  if (fp == NULL)
  {
    return;
  }

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
    else if (strcmp(name, "gamma")==0)
      parms->gamma = atof(value);
    else if (strcmp(name, "init")==0)
      strncpy (parms->init, value, MAXLEN);
    else if (strcmp(name, "nfile")==0)
      parms->nfile = atoi(value);
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
    else
      printf ("WARNING: %s/%s: Unknown parameter/value pair!\n",
        name, value);
  }

  /* Close file */
  fclose (fp);
}

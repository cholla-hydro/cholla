/*! \file cooling_wrapper.cu
 *  \brief Wrapper file for to load CUDA cooling tables. */

#ifdef CUDA
#ifdef COOLING_GPU

#include<stdio.h>
#include<stdlib.h>
#include"global.h"
#include"cooling_wrapper.h"

float *cooling_table = 0;
float *heating_table = 0;


/* \fn void Load_Cuda_Textures()
 * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
void Load_Cuda_Textures()
{

  double *n_arr;
  double *T_arr;
  double *L_arr;
  double *H_arr;

  int i;
  int nx = 121;
  int ny = 81;

  FILE *infile;
  char buffer[0x1000];
  char * pch;

  // allocate arrays to be copied to textures
  // these arrays are declared as external pointers in global.h
  cooling_table = (float *) malloc(nx*ny*sizeof(float));
  heating_table = (float *) malloc(nx*ny*sizeof(float));

  // allocate arrays to read in tables 
  n_arr = (double *) malloc(nx*ny*sizeof(double));
  T_arr = (double *) malloc(nx*ny*sizeof(double));
  L_arr = (double *) malloc(nx*ny*sizeof(double));
  H_arr = (double *) malloc(nx*ny*sizeof(double));

  // Read in cooling/heating curves (function of density and temperature)
  i=0;
  infile = fopen("./cloudy_coolingcurve.txt", "r");
  if (infile == NULL) {
    printf("Unable to open Cloudy file.\n");
    exit(1);
  }
  while (fgets(buffer, sizeof(buffer), infile) != NULL)
  {
    if (buffer[0] == '#') {
      continue;
    }
    else {
      pch = strtok(buffer, "\t");
      n_arr[i] = atof(pch);
      while (pch != NULL)
      {
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          T_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          L_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          H_arr[i] = atof(pch);
      }
      i++;
    }
  }
  fclose(infile);

  // copy data from cooling array into the table
  for (i=0; i<nx*ny; i++)
  {
    cooling_table[i] = float(L_arr[i]);
    heating_table[i] = float(H_arr[i]);
  }

  // Free arrays used to read in table data
  free(n_arr);
  free(T_arr);
  free(L_arr);
  free(H_arr);

}


/* \fn void Free_Cuda_Textures()
 * \brief Free the memory associated with the Cloudy cooling tables. */
void Free_Cuda_Textures()
{
  free(cooling_table);
  free(heating_table);
}


#endif
#endif

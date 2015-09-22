#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include"io.h"
#include"mpi_routines.h"

/* Write the initial conditions */
void WriteData(Grid3D G, struct parameters P, int nfile)
{
#ifndef  MPI_CHOLLA
  /*just use the data output routine*/
  OutputData(G,P,nfile);
#else  /*MPI_CHOLLA*/
  OutputDataMPI(G,P,nfile);
#endif /*MPI_CHOLLA*/
}


/* Output the grid data to file. */
void OutputData(Grid3D G, struct parameters P, int nfile)
{
  FILE *out;
  char filename[80];
  char timestep[20];

  strcpy(filename, P.outdir); 
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);   
  strcat(filename,".bin");
  out = fopen(filename, "w");

  if(out == NULL) {printf("Error opening output file.\n"); exit(0); }

  // write the conserved variables to the output file
  G.Write_Grid(out);

  // close the output file
  fclose(out);
}


/* MPI-safe printf routine */
int chprintf(const char * __restrict sdata, ...)
{
  int code = 0;
#ifdef MPI_CHOLLA
  /*limit printf to root process only*/
  if(procID==root)
  {
#endif /*MPI_CHOLLA*/

  va_list ap;
  va_start(ap, sdata);
  code = vfprintf(stdout, sdata, ap);
  va_end(ap);

#ifdef MPI_CHOLLA
  }
#endif /*MPI_CHOLLA*/

  return code;
}

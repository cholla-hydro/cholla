#ifdef MPI_CHOLLA
#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include"io.h"
#include"mpi_routines.h"

/* Output the grid data to file. */
void OutputDataMPI(Grid3D G, struct parameters P, int nfile)
{
  FILE *out;
  char filename[80];
  char timestep[20];

  strcpy(filename, P.outdir); 
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);   
  strcat(filename,".bin");
  sprintf(filename,"%s.%d",filename,procID);
  out = fopen(filename, "w");

  if(out == NULL) {printf("Error opening output file.\n"); fflush(stdout); exit(0); }

  // write the conserved variables to the output file
  G.Write_Grid(out);

  // close the output file
  fclose(out);
}
#endif /*MPI_CHOLLA*/

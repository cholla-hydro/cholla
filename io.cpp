#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include<hdf5.h>
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

  // create the filename
  strcpy(filename, P.outdir); 
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);   
  #if defined BINARY
  strcat(filename,".bin");
  #elif defined HDF5
  strcat(filename,".h5");
  #endif

  // open the file for binary writes
  #if defined BINARY
  out = fopen(filename, "w");
  if(out == NULL) {printf("Error opening output file.\n"); exit(0); }

  // write the header to the output file
  G.Write_Header_Binary(out);

  // write the conserved variables to the output file
  G.Write_Grid_Binary(out);

  // close the output file
  fclose(out);
  
  // create the file for hdf5 writes
  #elif defined HDF5
  hid_t   file_id; /* file identifier */
  herr_t  status;

  // Create a new file using default properties.
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);  

  // Write the header (file attributes)
  G.Write_Header_HDF5(file_id);

  // write the conserved variables to the output file
  G.Write_Grid_HDF5(file_id);

  // close the file
  status = H5Fclose(file_id);
  #endif
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

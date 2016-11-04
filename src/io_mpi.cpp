#ifdef MPI_CHOLLA
#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#ifdef HDF5
#include<hdf5.h>
#endif
#include"io.h"
#include"mpi_routines.h"

/* Output the grid data to file. */
void OutputDataMPI(Grid3D G, struct parameters P, int nfile)
{
  FILE *out;
  char filename[80];
  char timestep[20];

  // create the filename
  strcpy(filename, P.outdir); 
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);   
  // a binary file is created for each process
  #if defined BINARY
  strcat(filename,".bin");
  sprintf(filename,"%s.%d",filename,procID);
  // only one HDF5 file is created 
  #elif defined HDF5
  strcat(filename,".h5");
  sprintf(filename,"%s.%d",filename,procID);
  #endif

  // open the files for binary writes
  #if defined BINARY
  out = fopen(filename, "w");
  if(out == NULL) {printf("Error opening output file.\n"); fflush(stdout); exit(0); }

  // write the header to the output files
  G.Write_Header_Binary(out);

  // write the conserved variables to the output files
  G.Write_Grid_Binary(out);

  // close the output file
  fclose(out);
  

  #elif defined HDF5
  hid_t   file_id;
  herr_t  status;

  // Create a new file collectively
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Write the conserved variables to the output file
  G.Write_Grid_HDF5(file_id);

  // Close the file
  status = H5Fclose(file_id);
  #endif
}
#endif /*MPI_CHOLLA*/

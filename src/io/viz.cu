// Require HDF5
#ifdef HDF5

#include <hdf5.h>

#include "../grid/grid3D.h"

#include "../io/io.h" // To provide io.h with OutputViz3D


void CopyReal3D_CPU(Real* source, Real* destination, Header H)
{
  int i,j,k,id,buf_id;
  
  for (k=0; k<H.nz_real; k++) {
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
	id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
	buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
	destination[buf_id] = source[id];
      }
    }
  }
}

void WriteVizField(Header H, hid_t file_id, hid_t dataspace_id, Real* buffer, Real* source, const char* name)
{
  hid_t dataset_id;
  herr_t status;
  // Copy non-ghost parts of source to buffer
  CopyReal3D_CPU(source, buffer, H);

  // Create a dataset id 
  dataset_id = H5Dcreate(file_id, name, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the buffer to file
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  // Free the dataset id
  status = H5Dclose(dataset_id); 
  
}

void WriteViz3D(Grid3D &G, struct parameters P, hid_t file_id)
{
  Header H = G.H;
  int       nx_dset = H.nx_real;
  int       ny_dset = H.ny_real;
  int       nz_dset = H.nz_real;
  hsize_t   dims[3];
  // Create the data space for the datasets
  dims[0] = nx_dset;
  dims[1] = ny_dset;
  dims[2] = nz_dset;
  hid_t dataspace_id = H5Screate_simple(3, dims, NULL);  

  
  Real* dataset_buffer = (Real *) malloc(H.nx_real*H.ny_real*H.nz_real*sizeof(Real));


  if (P.outviz_density > 0) {
    WriteVizField(H, file_id, dataspace_id, dataset_buffer, G.C.density, "/density");
  }
  /* 
  // Just an example of extending this function to include other fields. 
  // Not implemented yet
  if (P.outviz_energy > 0) {
    WriteVizField(H, file_id, dataspace_id, dataset_buffer, C.Energy, "/energy");
  } 
  */ 
  
  
  free(dataset_buffer);
  herr_t status = H5Sclose(dataspace_id);  
}



void OutputViz3D(Grid3D &G, struct parameters P, int nfile)
{
  Header H = G.H;  
  // Do nothing in 1-D and 2-D case
  if (H.ny == 1) {
    return;
  }
  if (H.nz == 1) {
    return;
  }
  // Do nothing if nfile is not multiple of n_outviz
  if (nfile % P.n_outviz != 0) {
    return;
  }
  
  char filename[MAXLEN];
  char timestep[20];

  // create the filename
  sprintf(timestep, "%d", nfile);  
  strcpy(filename, P.outdir);
  strcat(filename, timestep);
  strcat(filename, ".viz3d.h5");
  #ifdef MPI_CHOLLA
  sprintf(filename,"%s.%d",filename,procID);
  #endif

  // create hdf5 file
  hid_t   file_id; /* file identifier */
  herr_t  status;

  // Create a new file using default properties.
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write the header (file attributes)
  G.Write_Header_HDF5(file_id);

  // write the conserved variables to the output file
  WriteViz3D(G, P, file_id);

  // close the file
  status = H5Fclose(file_id);

  if (status < 0) {printf("File write failed.\n"); exit(-1); }

}

#endif

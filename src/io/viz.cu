// Require HDF5
#ifdef HDF5

#include <hdf5.h>

#include "../grid/grid3D.h"

#include "../io/io.h" // To provide io.h with OutputViz3D

// Copy Real (non-ghost) cells from source to a double destination (for writing HDF5 in double precision)
__global__ void CopyReal3D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, double* destination, Real* source)
{

  int dest_id,source_id,id,i,j,k;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  
  k = id/(nx_real*ny_real);
  j = (id - k*nx_real*ny_real)/nx_real;
  i = id - j*nx_real - k*nx_real*ny_real;

  if (k >= nz_real) {
    return;
  }

  // This converts into HDF5 indexing that plays well with Python
  dest_id = k + j*nz_real + i*ny_real*nz_real;
  source_id = (i+n_ghost) + (j+n_ghost)*nx + (k+n_ghost)*nx*ny;

  destination[dest_id] = (double) source[source_id];
}

// Copy Real (non-ghost) cells from source to a float destination (for writing HDF5 in float precision)
__global__ void CopyReal3D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, float* destination, Real* source)
{

  int dest_id,source_id,id,i,j,k;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  
  k = id/(nx_real*ny_real);
  j = (id - k*nx_real*ny_real)/nx_real;
  i = id - j*nx_real - k*nx_real*ny_real;

  if (k >= nz_real) {
    return;
  }

  // This converts into HDF5 indexing that plays well with Python
  dest_id = k + j*nz_real + i*ny_real*nz_real;
  source_id = (i+n_ghost) + (j+n_ghost)*nx + (k+n_ghost)*nx*ny;

  destination[dest_id] = (float) source[source_id];
}

// When buffer is double, automatically use the double version of everything using function overloading
void WriteHDF5Field3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id, double* buffer, double* device_buffer, Real* device_source, const char* name)
{
  hid_t dataset_id;
  herr_t status;
  hsize_t dims[3];
  dims[0] = nx_real;
  dims[1] = ny_real;
  dims[2] = nz_real;
  hid_t dataspace_id = H5Screate_simple(3, dims, NULL);
  
  //Copy non-ghost parts of source to buffer
  dim3 dim1dGrid((nx_real*ny_real*nz_real+TPB-1)/TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1); 
  hipLaunchKernelGGL(CopyReal3D_GPU_Kernel,dim1dGrid,dim1dBlock,0,0,nx,ny,nx_real,ny_real,nz_real,n_ghost,device_buffer,device_source);
  CudaSafeCall(cudaMemcpy( buffer, device_buffer, nx_real*ny_real*nz_real*sizeof(double), cudaMemcpyDeviceToHost));

  // Create a dataset id 
  dataset_id = H5Dcreate(file_id, name, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the buffer to file
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  // Free the dataset id and dataspace id
  status = H5Dclose(dataset_id);
  status = H5Sclose(dataspace_id);                                                                                                                                                                           
}


// When buffer is float, automatically use the float version of everything using function overloading
void WriteHDF5Field3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id, float* buffer, float* device_buffer, Real* device_source, const char* name)
{

  hid_t dataset_id;
  herr_t status;
  hsize_t dims[3];
  dims[0] = nx_real;
  dims[1] = ny_real;
  dims[2] = nz_real;
  hid_t dataspace_id = H5Screate_simple(3, dims, NULL);
  
  //Copy non-ghost parts of source to buffer
  dim3 dim1dGrid((nx_real*ny_real*nz_real+TPB-1)/TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1); 
  hipLaunchKernelGGL(CopyReal3D_GPU_Kernel,dim1dGrid,dim1dBlock,0,0,nx,ny,nx_real,ny_real,nz_real,n_ghost,device_buffer,device_source);
  CudaSafeCall(cudaMemcpy( buffer, device_buffer, nx_real*ny_real*nz_real*sizeof(float), cudaMemcpyDeviceToHost));

  // Create a dataset id 
  dataset_id = H5Dcreate(file_id, name, H5T_IEEE_F32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the buffer to file
  status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  // Free the dataset id and dataspace id
  status = H5Dclose(dataset_id);
  status = H5Sclose(dataspace_id);                                                                                                                                                                           
}


void WriteViz3D(Grid3D &G, struct parameters P, hid_t file_id)
{
  Header H = G.H;
  int nx_real = H.nx_real;
  int ny_real = H.ny_real;
  int nz_real = H.nz_real;
  int n_ghost = H.n_ghost;
  int nx = H.nx;
  int ny = H.ny;

  float* dataset_buffer = (float *) malloc(H.nx_real*H.ny_real*H.nz_real*sizeof(Real));
  float* device_buffer;
  CudaSafeCall(cudaMalloc(&device_buffer,  nx_real*ny_real*nz_real*sizeof(float)));

  if (P.outviz_density > 0) {
    WriteHDF5Field3D(nx, ny, nx_real, ny_real, nz_real, n_ghost, file_id, dataset_buffer, device_buffer, G.C.d_density, "/density");
  }


  
  /* 
  // Just an example of extending this function to include other fields. 
  // Not implemented yet
  if (P.outviz_energy > 0) {
    WriteHDF5Field(H, file_id, dataspace_id, dataset_buffer, C.Energy, "/energy");
  } 
  */ 
  CudaSafeCall(cudaFree(device_buffer));  
  
  free(dataset_buffer);

}



void OutputViz3D(Grid3D &G, struct parameters P, int nfile)
{
  Header H = G.H;  
  // Do nothing in 1-D and 2-D case
  if (H.ny_real == 1) {
    return;
  }
  if (H.nz_real == 1) {
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

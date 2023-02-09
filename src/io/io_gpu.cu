// Require HDF5
#ifdef HDF5

  #include <hdf5.h>

  #include "../grid/grid3D.h"
  #include "../io/io.h"  // To provide io.h with OutputViz3D

// Note that the HDF5 file and buffer will have size nx_real * ny_real * nz_real
// whereas the conserved variables have size nx,ny,nz Note that magnetic fields
// add +1 to nx_real ny_real nz_real since an extra face needs to be output, but
// also has the same size nx ny nz For the magnetic field case, a different
// nx_real+1 ny_real+1 nz_real+1 n_ghost-1 are provided as inputs.

// Copy Real (non-ghost) cells from source to a double destination (for writing
// HDF5 in double precision)
__global__ void CopyReal3D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost,
                                      double* destination, Real* source)
{
  int dest_id, source_id, id, i, j, k;
  id = threadIdx.x + blockIdx.x * blockDim.x;

  k = id / (nx_real * ny_real);
  j = (id - k * nx_real * ny_real) / nx_real;
  i = id - j * nx_real - k * nx_real * ny_real;

  if (k >= nz_real) {
    return;
  }

  // This converts into HDF5 indexing that plays well with Python
  dest_id   = k + j * nz_real + i * ny_real * nz_real;
  source_id = (i + n_ghost) + (j + n_ghost) * nx + (k + n_ghost) * nx * ny;

  destination[dest_id] = (double)source[source_id];
}

// Copy Real (non-ghost) cells from source to a float destination (for writing
// HDF5 in float precision)
__global__ void CopyReal3D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost,
                                      float* destination, Real* source)
{
  int dest_id, source_id, id, i, j, k;
  id = threadIdx.x + blockIdx.x * blockDim.x;

  k = id / (nx_real * ny_real);
  j = (id - k * nx_real * ny_real) / nx_real;
  i = id - j * nx_real - k * nx_real * ny_real;

  if (k >= nz_real) {
    return;
  }

  // This converts into HDF5 indexing that plays well with Python
  dest_id   = k + j * nz_real + i * ny_real * nz_real;
  source_id = (i + n_ghost) + (j + n_ghost) * nx + (k + n_ghost) * nx * ny;

  destination[dest_id] = (float)source[source_id];
}

// When buffer is double, automatically use the double version of everything
// using function overloading
void WriteHDF5Field3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id, double* buffer,
                      double* device_buffer, Real* device_source, const char* name)
{
  herr_t status;
  hsize_t dims[3];
  dims[0]            = nx_real;
  dims[1]            = ny_real;
  dims[2]            = nz_real;
  hid_t dataspace_id = H5Screate_simple(3, dims, NULL);

  // Copy non-ghost parts of source to buffer
  dim3 dim1dGrid((nx_real * ny_real * nz_real + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(CopyReal3D_GPU_Kernel, dim1dGrid, dim1dBlock, 0, 0, nx, ny, nx_real, ny_real, nz_real, n_ghost,
                     device_buffer, device_source);
  CudaSafeCall(cudaMemcpy(buffer, device_buffer, nx_real * ny_real * nz_real * sizeof(double), cudaMemcpyDeviceToHost));

  // Write Buffer to HDF5
  status = Write_HDF5_Dataset(file_id, dataspace_id, buffer, name);

  status = H5Sclose(dataspace_id);
  if (status < 0) {
    printf("File write failed.\n");
  }
}

// When buffer is float, automatically use the float version of everything using
// function overloading
void WriteHDF5Field3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id, float* buffer,
                      float* device_buffer, Real* device_source, const char* name)
{
  herr_t status;
  hsize_t dims[3];
  dims[0]            = nx_real;
  dims[1]            = ny_real;
  dims[2]            = nz_real;
  hid_t dataspace_id = H5Screate_simple(3, dims, NULL);

  // Copy non-ghost parts of source to buffer
  dim3 dim1dGrid((nx_real * ny_real * nz_real + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(CopyReal3D_GPU_Kernel, dim1dGrid, dim1dBlock, 0, 0, nx, ny, nx_real, ny_real, nz_real, n_ghost,
                     device_buffer, device_source);
  CudaSafeCall(cudaMemcpy(buffer, device_buffer, nx_real * ny_real * nz_real * sizeof(float), cudaMemcpyDeviceToHost));

  // Write Buffer to HDF5
  status = Write_HDF5_Dataset(file_id, dataspace_id, buffer, name);

  status = H5Sclose(dataspace_id);
  if (status < 0) {
    printf("File write failed.\n");
  }
}

#endif  // HDF5

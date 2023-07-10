// Require HDF5
#ifdef HDF5

  #include <hdf5.h>

  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../utils/cuda_utilities.h"

// Note that the HDF5 file and buffer will have size nx_real * ny_real * nz_real
// whereas the conserved variables have size nx,ny,nz.

// Note that magnetic fields
// add +1 to nx_real ny_real nz_real since an extra face needs to be output, but
// also has the same size nx ny nz.

// For the magnetic field case, a different
// nx_real+1 ny_real+1 nz_real+1 n_ghost-1 are provided as inputs.

// 2D version of CopyReal3D_GPU_Kernel. Note that magnetic fields and float32 output are not enabled in 2-D so this is a
// simpler kernel
__global__ void CopyReal2D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost,
                                      Real* destination, Real* source)
{
  int const id = threadIdx.x + blockIdx.x * blockDim.x;

  int i, j, k;
  cuda_utilities::compute3DIndices(id, nx_real, ny_real, i, j, k);
  // i goes up to nx_real
  // j goes up to ny_real
  // for 2D, k should be 0
  if (k >= 1) {
    return;
  }

  // This converts into HDF5 indexing that plays well with Python
  int const dest_id   = j + i * ny_real;
  int const source_id = (i + n_ghost) + (j + n_ghost) * nx;

  destination[dest_id] = source[source_id];
}

// Copy Real (non-ghost) cells from source to a double destination (for writing
// HDF5 in double precision)
__global__ void CopyReal3D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost,
                                      double* destination, Real* source, int mhd_direction)
{
  int const id = threadIdx.x + blockIdx.x * blockDim.x;

  int i, j, k;
  cuda_utilities::compute3DIndices(id, nx_real, ny_real, i, j, k);

  if (k >= nz_real) {
    return;
  }

  // This converts into HDF5 indexing that plays well with Python
  int const dest_id   = k + j * nz_real + i * ny_real * nz_real;
  int const source_id = (i + n_ghost - int(mhd_direction == 0)) + (j + n_ghost - int(mhd_direction == 1)) * nx +
                        (k + n_ghost - int(mhd_direction == 2)) * nx * ny;

  destination[dest_id] = (double)source[source_id];
}

// Copy Real (non-ghost) cells from source to a float destination (for writing
// HDF5 in float precision)
__global__ void CopyReal3D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost,
                                      float* destination, Real* source, int mhd_direction)
{
  int const id = threadIdx.x + blockIdx.x * blockDim.x;

  int i, j, k;
  cuda_utilities::compute3DIndices(id, nx_real, ny_real, i, j, k);

  if (k >= nz_real) {
    return;
  }

  // This converts into HDF5 indexing that plays well with Python.
  // The `int(mhd_direction == NUM)` sections provide appropriate shifts for writing out the magnetic fields since they
  // need an extra cell in the same direction as the field
  int const dest_id   = k + j * nz_real + i * ny_real * nz_real;
  int const source_id = (i + n_ghost - int(mhd_direction == 0)) + (j + n_ghost - int(mhd_direction == 1)) * nx +
                        (k + n_ghost - int(mhd_direction == 2)) * nx * ny;

  destination[dest_id] = (float)source[source_id];
}

// When buffer is double, automatically use the double version of everything
// using function overloading
void Write_HDF5_Field_3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id,
                         double* buffer, double* device_buffer, Real* device_source, const char* name,
                         int mhd_direction)
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
                     device_buffer, device_source, mhd_direction);
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
void Write_HDF5_Field_3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id,
                         float* buffer, float* device_buffer, Real* device_source, const char* name, int mhd_direction)
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
                     device_buffer, device_source, mhd_direction);
  CudaSafeCall(cudaMemcpy(buffer, device_buffer, nx_real * ny_real * nz_real * sizeof(float), cudaMemcpyDeviceToHost));

  // Write Buffer to HDF5
  status = Write_HDF5_Dataset(file_id, dataspace_id, buffer, name);

  status = H5Sclose(dataspace_id);
  if (status < 0) {
    printf("File write failed.\n");
  }
}
void Fill_HDF5_Buffer_From_Grid_GPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                    Real* hdf5_buffer, Real* device_hdf5_buffer, Real* device_grid_buffer)
{
  int mhd_direction = -1;

  // 3D case
  if (nx > 1 && ny > 1 && nz > 1) {
    dim3 dim1dGrid((nx_real * ny_real * nz_real + TPB - 1) / TPB, 1, 1);
    dim3 dim1dBlock(TPB, 1, 1);
    hipLaunchKernelGGL(CopyReal3D_GPU_Kernel, dim1dGrid, dim1dBlock, 0, 0, nx, ny, nx_real, ny_real, nz_real, n_ghost,
                       device_hdf5_buffer, device_grid_buffer, mhd_direction);
    CudaSafeCall(cudaMemcpy(hdf5_buffer, device_hdf5_buffer, nx_real * ny_real * nz_real * sizeof(Real),
                            cudaMemcpyDeviceToHost));
    return;
  }

  // 2D case
  if (nx > 1 && ny > 1 && nz == 1) {
    dim3 dim1dGrid((nx_real * ny_real + TPB - 1) / TPB, 1, 1);
    dim3 dim1dBlock(TPB, 1, 1);
    hipLaunchKernelGGL(CopyReal2D_GPU_Kernel, dim1dGrid, dim1dBlock, 0, 0, nx, ny, nx_real, ny_real, nz_real, n_ghost,
                       device_hdf5_buffer, device_grid_buffer);
    CudaSafeCall(cudaMemcpy(hdf5_buffer, device_hdf5_buffer, nx_real * ny_real * sizeof(Real), cudaMemcpyDeviceToHost));
    return;
  }

  // 1D case
  if (nx > 1 && ny == 1 && nz == 1) {
    CudaSafeCall(cudaMemcpy(hdf5_buffer, device_grid_buffer + n_ghost, nx_real * sizeof(Real), cudaMemcpyDeviceToHost));
    return;
  }
}

#endif  // HDF5

// Routines for using Parallel HDF5 to read/write from single file
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../utils/error_handling.h"

#if defined(HDF5) && defined(MPI_CHOLLA)
  #include <hdf5.h>

  #include "../mpi/mpi_routines.h"
  #include "../utils/timing_functions.h"  // provides ScopedTimer

// Warning: H5Sselect_hyperslab expects its pointer args to be arrays of same size as the rank of the dataspace
// file_space_id
void Read_HDF5_Selection_3D(hid_t file_id, hsize_t* offset, hsize_t* count, double* buffer, const char* name)
{
  hid_t dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  // Select the requested subset of data
  hid_t file_space_id = H5Dget_space(dataset_id);
  hid_t mem_space_id  = H5Screate_simple(3, count, NULL);

  // Notes on hyperslab call:

  // First NULL is stride, setting to NULL is like setting to 1, contiguous

  // Second NULL is block, setting to NULL sets block size to 1.

  // Count is the number of blocks in each dimension:

  // since our block size is 1, Count is the number of voxels in each dimension

  herr_t status = H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
  // Read in the data subset
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, mem_space_id, file_space_id, H5P_DEFAULT, buffer);

  // Free the ids
  status = H5Sclose(mem_space_id);
  status = H5Sclose(file_space_id);
  status = H5Dclose(dataset_id);
}

// Alwin: I'm only writing a 3D version of this because that's what is practical.
// Read from concatenated HDF5 file
void Read_Grid_Cat_HDF5_Field(hid_t file_id, Real* dataset_buffer, Header H, hsize_t* offset, hsize_t* count,
                              Real* grid_buffer, const char* name)
{
  Read_HDF5_Selection_3D(file_id, offset, count, dataset_buffer, name);
  Fill_Grid_From_HDF5_Buffer(H.nx, H.ny, H.nz, H.nx_real, H.ny_real, H.nz_real, H.n_ghost, dataset_buffer, grid_buffer);
}

void Read_Grid_Cat_HDF5_Field_Magnetic(hid_t file_id, Real* dataset_buffer, Header H, hsize_t* offset, hsize_t* count,
                                       Real* grid_buffer, const char* name)
{
  Read_HDF5_Selection_3D(file_id, offset, count, dataset_buffer, name);
  Fill_Grid_From_HDF5_Buffer(H.nx, H.ny, H.nz, H.nx_real + 1, H.ny_real + 1, H.nz_real + 1, H.n_ghost - 1,
                             dataset_buffer, grid_buffer);
}

/*! \brief Read in grid data from a single concatenated output file. */
void Grid3D::Read_Grid_Cat(struct parameters P)
{
  ScopedTimer timer("Read_Grid_Cat");
  herr_t status;
  char filename[100];

  sprintf(filename, "%s%d.h5", P.indir, P.nfile);

  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

  if (file_id < 0) {
    printf("Unable to open input file: %s\n", filename);
    exit(0);
  }

  // TODO (written by Alwin, for anyone to do) :
  // Consider using collective calls if this part is slow at scale
  hid_t attribute_id;
  attribute_id = H5Aopen(file_id, "t", H5P_DEFAULT);
  status       = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &H.t);
  status       = H5Aclose(attribute_id);
  attribute_id = H5Aopen(file_id, "n_step", H5P_DEFAULT);
  status       = H5Aread(attribute_id, H5T_NATIVE_INT, &H.n_step);
  status       = H5Aclose(attribute_id);

  // Offsets are global variables from mpi_routines.h
  hsize_t offset[3];
  offset[0] = nx_local_start;
  offset[1] = ny_local_start;
  offset[2] = nz_local_start;

  // This is really dims but I name it count because that's what HDF5 names it
  hsize_t count[3];
  count[0] = H.nx_real;
  count[1] = H.ny_real;
  count[2] = H.nz_real;

  #ifdef MHD
  Real* dataset_buffer = (Real*)malloc((H.nz_real + 1) * (H.ny_real + 1) * (H.nx_real + 1) * sizeof(Real));
  #else
  Real* dataset_buffer = (Real*)malloc((H.nz_real) * (H.ny_real) * (H.nx_real) * sizeof(Real));
  #endif

  Read_Grid_Cat_HDF5_Field(file_id, dataset_buffer, H, offset, count, C.density, "/density");
  Read_Grid_Cat_HDF5_Field(file_id, dataset_buffer, H, offset, count, C.momentum_x, "/momentum_x");
  Read_Grid_Cat_HDF5_Field(file_id, dataset_buffer, H, offset, count, C.momentum_y, "/momentum_y");
  Read_Grid_Cat_HDF5_Field(file_id, dataset_buffer, H, offset, count, C.momentum_z, "/momentum_z");
  Read_Grid_Cat_HDF5_Field(file_id, dataset_buffer, H, offset, count, C.Energy, "/Energy");
  #ifdef DE
  Read_Grid_Cat_HDF5_Field(file_id, dataset_buffer, H, offset, count, C.GasEnergy, "/GasEnergy");
  #endif  // DE

  #ifdef SCALAR
    #ifdef BASIC_SCALAR
  Read_Grid_Cat_HDF5_Field(file_id, dataset_buffer, H, offset, count, C.basic_scalar, "/scalar0");
    #endif
    #ifdef DUST
  Read_Grid_Cat_HDF5_Field(file_id, dataset_buffer, H, offset, count, C.dust_density, "/dust_density");
    #endif
  #endif
  // TODO (Alwin) : add scalar stuff

  #ifdef MHD
  Read_Grid_Cat_HDF5_Field_Magnetic(file_id, dataset_buffer, H, offset, count, C.magnetic_x, "/magnetic_x");
  Read_Grid_Cat_HDF5_Field_Magnetic(file_id, dataset_buffer, H, offset, count, C.magnetic_y, "/magnetic_y");
  Read_Grid_Cat_HDF5_Field_Magnetic(file_id, dataset_buffer, H, offset, count, C.magnetic_z, "/magnetic_z");
  #endif

  free(dataset_buffer);
  status = H5Fclose(file_id);
}

#else

void Grid3D::Read_Grid_Cat(struct parameters P)
{
  chprintf("Warning: Read_Grid_Cat does nothing without MPI_CHOLLA and HDF5\n");
  chexit(-1);
  return;
  // Does nothing without HDF5 and MPI_CHOLLA
}

#endif

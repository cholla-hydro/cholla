// Special functions needed to make restart (init=Read_Grid) consistent with
// running continuously

#include <cstdio>

#ifdef GRAVITY
  #include "../gravity/grav3D.h"
  #include "../io/io.h"
#endif

#ifdef MPI_CHOLLA
// provides procID
  #include "../mpi/mpi_routines.h"
#endif  // MPI_CHOLLA

#ifdef HDF5
  #include <hdf5.h>
#endif

void Gravity_Restart_Filename(char* filename, char* dirname, int nfile)
{
#ifdef MPI_CHOLLA
  sprintf(filename, "%s%d_gravity.h5.%d", dirname, nfile, procID);
#else
  sprintf(filename, "%s%d_gravity.h5", dirname, nfile);
#endif
}

#if defined(GRAVITY) && defined(HDF5)
void Grav3D::Read_Restart_HDF5(struct parameters* P, int nfile)
{
  H5open();
  char filename[MAXLEN];
  Gravity_Restart_Filename(filename, P->indir, nfile);
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read dt_now
  hid_t attribute_id = H5Aopen(file_id, "dt_now", H5P_DEFAULT);
  herr_t status      = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &dt_now);
  status             = H5Aclose(attribute_id);

  // Read potential and copy to device to be used as potential n-1
  Read_HDF5_Dataset(file_id, F.potential_1_h, "/potential");
  #ifdef GRAVITY_GPU
  CudaSafeCall(cudaMemcpy(F.potential_1_d, F.potential_1_h,
                          n_cells_potential * sizeof(Real),
                          cudaMemcpyHostToDevice));
  #endif

  H5Fclose(file_id);
  H5close();

  // Set INITIAL to false
  INITIAL = false;
}

void Grav3D::Write_Restart_HDF5(struct parameters* P, int nfile)
{
  H5open();
  char filename[MAXLEN];
  Gravity_Restart_Filename(filename, P->outdir, nfile);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write dt_now
  hsize_t attr_dims  = 1;
  hid_t dataspace_id = H5Screate_simple(1, &attr_dims, NULL);

  hid_t attribute_id = H5Acreate(file_id, "dt_now", H5T_IEEE_F64BE,
                                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status      = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &dt_now);
  status             = H5Aclose(attribute_id);

  status = H5Sclose(dataspace_id);

  // Copy device to host if needed
  #ifdef GRAVITY_GPU
  CudaSafeCall(cudaMemcpy(F.potential_1_h, F.potential_1_d,
                          n_cells_potential * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  #endif

  // Write potential
  hsize_t dims[1];
  dims[0] = n_cells_potential;

  dataspace_id = H5Screate_simple(1, dims, NULL);
  HDF5_Dataset(file_id, dataspace_id, F.potential_1_h, "/potential");
  H5Sclose(dataspace_id);

  H5Fclose(file_id);

  H5close();
}

#elif defined(GRAVITY)
// Do nothing
void Grav3D::Read_Restart_HDF5(struct parameters* P, int nfile)
{
  chprintf("WARNING from file %s line %d: Read_Restart_HDF5 did nothing",
           __FILE__, __LINE__);
}

void Grav3D::Write_Restart_HDF5(struct parameters* P, int nfile) {}
chprintf("WARNING from file %s line %d: Write_Restart_HDF5 did nothing",
         __FILE__, __LINE__);
#endif

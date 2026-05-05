#include <cstdio>

#ifdef RT
  #include "../radiation/radiation.h"
  #include "../io/io.h"
#endif

#ifdef MPI_CHOLLA
// provides procID
  #include "../mpi/mpi_routines.h"
#endif  // MPI_CHOLLA

#ifdef HDF5
  #include <hdf5.h>
#endif

void Radiation_Restart_Filename(char* filename, char* dirname, int nfile)
{
#ifdef MPI_CHOLLA
  sprintf(filename, "%s%d_rt.h5.%d", dirname, nfile, procID);
#else
  sprintf(filename, "%s%d_rt.h5", dirname, nfile);
#endif
}

#if defined(RT) && defined(HDF5)
void Rad3D::Read_Restart_HDF5(struct Parameters* P, int nfile)
{
  H5open();
  char filename[MAXLEN];
  Radiation_Restart_Filename(filename, P->indir, nfile);
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read dt_now
  hid_t attribute_id = H5Aopen(file_id, "dt_now", H5P_DEFAULT);
  herr_t status      = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &dt_now);
  status             = H5Aclose(attribute_id);

  // Read source and copy to device
  Read_HDF5_Dataset(file_id, F.rs, "/source");
  GPU_Error_Check(cudaMemcpy(F.dev_rs, F.rs, grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));


  // Read radiation fiels and copy to device
  Read_HDF5_Dataset(file_id, F.rf, "/source");
  GPU_Error_Check(cudaMemcpy(F.dev_rf, F.rf, n_rf * grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));

  H5Fclose(file_id);
  H5close();

}

void Rad3D::Write_Restart_HDF5(struct Parameters* P, int nfile, const FnameTemplate& fname_template)
{
  H5open();
  hsize_t dims[1];
  std::string filename = fname_template.format_fname(nfile, "_rt");
  hid_t file_id        = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write dt_now
  hsize_t attr_dims  = 1;
  hid_t dataspace_id = H5Screate_simple(1, &attr_dims, NULL);

  hid_t attribute_id = H5Acreate(file_id, "dt_now", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status      = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &dt_now);
  status             = H5Aclose(attribute_id);

  status = H5Sclose(dataspace_id);

  // Source field

  // Copy device to host
  GPU_Error_Check(cudaMemcpy(rtFields.rs, rtFields.dev_rs, grid.n_cells * sizeof(Real), cudaMemcpyDeviceToHost));

  // Write source field
  dims[0] = grid.n_cells;

  dataspace_id = H5Screate_simple(1, dims, NULL);
  Write_HDF5_Dataset(file_id, dataspace_id, rtFields.rs, "/source");
  H5Sclose(dataspace_id);


  // Radiation fields

  // Copy device to host 
  GPU_Error_Check(cudaMemcpy(rtFields.rf, rtFields.dev_rf, n_rf * grid.n_cells * sizeof(Real), cudaMemcpyDeviceToHost));

  // Write radiation fields
  dims[0] = n_rf * grid.n_cells;

  dataspace_id = H5Screate_simple(1, dims, NULL);
  Write_HDF5_Dataset(file_id, dataspace_id, rtFields.rf, "/radiation");
  H5Sclose(dataspace_id);

  // close the file
  H5Fclose(file_id);

  H5close();
}

#elif defined(RT)
// Do nothing
void Rad3D::Read_Restart_HDF5(struct Parameters* P, int nfile)
{
  chprintf("WARNING from file %s line %d: Rad3D::Read_Restart_HDF5 did nothing", __FILE__, __LINE__);
}

void Rad3D::Write_Restart_HDF5(struct Parameters* P, int nfile)
{
  chprintf("WARNING from file %s line %d: Rad3D::Write_Restart_HDF5 did nothing", __FILE__, __LINE__);
}
#endif
#include <cstdio>

#ifdef RT
  #include "../io/io.h"
  #include "../radiation/radiation.h"
#endif

#ifdef MPI_CHOLLA
// provides procID
  #include "../mpi/mpi_routines.h"
#endif  // MPI_CHOLLA

#ifdef HDF5
  #include <hdf5.h>
#endif

#ifdef RT
void Rad3D::Radiation_Restart_Filename(char* filename, char* dirname, int nfile)
{
  #ifdef MPI_CHOLLA
  const std::string base_fname = dirname + (std::to_string(nfile) + "_rt.h5." + std::to_string(procID));
  #else
  const std::string base_fname = dirname + (std::to_string(nfile) + "_rt.h5." + std::to_string(procID));
  #endif

  std::strcpy(filename, base_fname.c_str());
}
#endif

#if defined(RT) && defined(HDF5)

herr_t Rad3D::Write_HDF5_Attribute(hid_t file_id, hid_t dataspace_id, int* attribute, const char* name)
{
  hid_t attribute_id = H5Acreate(file_id, name, H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status      = H5Awrite(attribute_id, H5T_NATIVE_INT, attribute);
  status             = H5Aclose(attribute_id);
  return status;
}
herr_t Rad3D::Write_HDF5_Attribute(hid_t file_id, hid_t dataspace_id, double* attribute, const char* name)
{
  hid_t attribute_id = H5Acreate(file_id, name, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status      = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, attribute);
  status             = H5Aclose(attribute_id);
  return status;
}

void Rad3D::Read_Restart_HDF5(Parameters* P, int nfile)
{
  H5open();
  hsize_t dims[1];
  char filename[MAXLEN];
  Radiation_Restart_Filename(filename, P->indir, nfile);
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  int rt_mode;

  Real* rtptr;  // pointer to radiation fields

  // Read dt_now
  //  hid_t attribute_id = H5Aopen(file_id, "dt_now", H5P_DEFAULT);
  //  herr_t status      = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &dt_now);
  //  status             = H5Aclose(attribute_id);

  hid_t attribute_id = H5Aopen(file_id, "rt_mode", H5P_DEFAULT);
  herr_t status      = H5Aread(attribute_id, H5T_NATIVE_INT, &rt_mode);
  status             = H5Aclose(attribute_id);

  #ifdef COSMOLOGY
  // Read in the current scale factor, required for
  // conversion to comoving coordinates
  double restart_a = 1.0;
  attribute_id     = H5Aopen(file_id, "current_a", H5P_DEFAULT);
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &restart_a);
  H5Aclose(attribute_id);

  // set the current scale factor from the restart
  current_a = (Real)restart_a;
  #endif

  // Read source and copy to device
  Read_HDF5_Dataset(file_id, rtFields.rs, "/rf_sources");
  GPU_Error_Check(cudaMemcpy(rtFields.dev_rs, rtFields.rs, grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));

  // Read radiation fields and copy to device
  Read_HDF5_Dataset(file_id, rtFields.rf, "/radiation");
  GPU_Error_Check(cudaMemcpy(rtFields.dev_rf, rtFields.rf, n_rf * grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));

  #ifdef RT_OTVET
  const char* rt_dset_names[7] = {"/rf_intensity", "/rf_HI_near", "/rf_HeI_near", "/rf_HeII_near",
                                  "/rf_HI_far",    "/rf_HeI_far", "/rf_HeII_far"};
  #endif
  #ifdef RT_M1
  const char* rt_dset_names[16] = {
      "/rf_HI",   "/rf_HI_Mx",   "/rf_HI_My",   "/rf_HI_Mz",   "/rf_HeI", "/rf_HeI_Mx", "/rf_HeI_My", "/rf_HeI_Mz",
      "/rf_HeII", "/rf_HeII_Mx", "/rf_HeII_My", "/rf_HeII_Mz", "/rf_0",   "/rf_0_Mx",   "/rf_0_My",   "/rf_0_Mz",
  };
  #endif

  // Read rf fields from the HDF5 file
  dims[0] = grid.n_cells;
  for (int n = 0; n < n_rf; n++) {
    rtptr = &rtFields.rf[n * grid.n_cells];
    Read_HDF5_Dataset(file_id, rtptr, rt_dset_names[n]);
  }
  H5Fclose(file_id);
  H5close();

  // copy radiation fields to the device
  // GPU_Error_Check(cudaMemcpy(rtFields.dev_rf, rtFields.rf, n_rf * grid.n_cells * sizeof(Real),
  // cudaMemcpyHostToDevice));
}

void Rad3D::Write_Restart_HDF5(Parameters* P, int nfile, const FnameTemplate& fname_template)
{
  H5open();
  hsize_t dims[1];
  int int_data[3];
  Real Real_data[3];
  Real* rtptr;

  std::string filename = fname_template.format_fname(nfile, "_rt");
  hid_t file_id        = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write rt_mode, dims
  hsize_t attr_dims  = 1;
  hid_t dataspace_id = H5Screate_simple(1, &attr_dims, NULL);

  //  hid_t attribute_id = H5Acreate(file_id, "dt_now", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  // herr_t status      = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &dt_now);
  //  status             = H5Aclose(attribute_id);
  int rt_mode = 0;  // default RT_OTVET
  #ifdef RT_M1
  rt_mode = 1;  // RT_M1
  #endif

  hid_t attribute_id = H5Acreate(file_id, "rt_mode", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status      = H5Awrite(attribute_id, H5T_NATIVE_INT, &rt_mode);
  status             = H5Aclose(attribute_id);

  attribute_id = H5Acreate(file_id, "n_ghost", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status       = H5Awrite(attribute_id, H5T_NATIVE_INT, &grid.n_ghost);
  status       = H5Aclose(attribute_id);

  attribute_id = H5Acreate(file_id, "n_rf", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status       = H5Awrite(attribute_id, H5T_NATIVE_INT, &n_rf);
  status       = H5Aclose(attribute_id);

  status = H5Sclose(dataspace_id);

  // Now 3D attributes
  attr_dims = 3;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);

  #ifndef MPI_CHOLLA
  int_data[0] = grid.nx_real;
  int_data[1] = grid.ny_real;
  int_data[2] = grid.nz_real;
  #endif
  #ifdef MPI_CHOLLA
  int_data[0] = nx_global;
  int_data[1] = ny_global;
  int_data[2] = nz_global;
  #endif

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "dims");

  #ifdef MPI_CHOLLA
  int_data[0] = grid.nx_real;
  int_data[1] = grid.ny_real;
  int_data[2] = grid.nz_real;

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "dims_local");

  int_data[0] = nx_local_start;
  int_data[1] = ny_local_start;
  int_data[2] = nz_local_start;

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "offset");

  int_data[0] = nproc_x;
  int_data[1] = nproc_y;
  int_data[2] = nproc_z;

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "nprocs");
  #endif

  Real_data[0] = grid.xbound;
  Real_data[1] = grid.ybound;
  Real_data[2] = grid.zbound;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "bounds");

  Real_data[0] = grid.xdglobal;
  Real_data[1] = grid.ydglobal;
  Real_data[2] = grid.zdglobal;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "domain");

  Real_data[0] = grid.dx;
  Real_data[1] = grid.dy;
  Real_data[2] = grid.dz;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "dx");

  // Close the dataspace
  status = H5Sclose(dataspace_id);

  #ifdef COSMOLOGY
  // We will save the RT data in physical units
  // Store the scale factor so we can convert from
  // physical to cosmological units after reading
  // from the restart file.
  attr_dims          = 1;
  dataspace_id       = H5Screate_simple(1, &attr_dims, NULL);
  double current_a_d = (double)current_a;  // current scale factor
  attribute_id       = H5Acreate(file_id, "current_a", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status             = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &current_a_d);
  status             = H5Aclose(attribute_id);
  H5Sclose(dataspace_id);
  #endif

  // Source field

  // Copy device to host (the host array only exists if the ICs created it)
  if (rtFields.rs == nullptr) rtFields.rs = (Real*)malloc(grid.n_cells * sizeof(Real));

  // Write source field
  dims[0]      = grid.n_cells;
  dataspace_id = H5Screate_simple(1, dims, NULL);
  Write_HDF5_Dataset(file_id, dataspace_id, rtFields.rs, "/rf_sources");
  H5Sclose(dataspace_id);

  // Radiation fields

  // Copy device to host
  GPU_Error_Check(cudaMemcpy(rtFields.rf, rtFields.dev_rf, n_rf * grid.n_cells * sizeof(Real), cudaMemcpyDeviceToHost));

  // Write radiation fields

  #ifdef RT_OTVET
  const char* rt_dset_names[7] = {"/rf_intensity", "/rf_HI_near", "/rf_HeI_near", "/rf_HeII_near",
                                  "/rf_HI_far",    "/rf_HeI_far", "/rf_HeII_far"};
  #endif
  #ifdef RT_M1
  const char* rt_dset_names[16] = {
      "/rf_HI",   "/rf_HI_Mx",   "/rf_HI_My",   "/rf_HI_Mz",   "/rf_HeI", "/rf_HeI_Mx", "/rf_HeI_My", "/rf_HeI_Mz",
      "/rf_HeII", "/rf_HeII_Mx", "/rf_HeII_My", "/rf_HeII_Mz", "/rf_0",   "/rf_0_Mx",   "/rf_0_My",   "/rf_0_Mz",
  };
  #endif

  dims[0] = grid.n_cells;
  for (int n = 0; n < n_rf; n++) {
    dataspace_id = H5Screate_simple(1, dims, NULL);
    rtptr        = &rtFields.rf[n * grid.n_cells];
    Write_HDF5_Dataset(file_id, dataspace_id, rtptr, rt_dset_names[n]);
    H5Sclose(dataspace_id);
  }

  // Source catalog of this rank (rt_n_pops > 0)
  if (sources.Active()) sources.Write_Restart_HDF5(file_id);

  // close the file
  H5Fclose(file_id);

  H5close();
}

#elif defined(RT)
// Do nothing
void Rad3D::Read_Restart_HDF5(Parameters* P, int nfile)
{
  chprintf("WARNING from file %s line %d: Rad3D::Read_Restart_HDF5 did nothing", __FILE__, __LINE__);
  CHOLLA_ERROR("Radiation currently requires HDF5 for restarting.\n");
}

void Rad3D::Write_Restart_HDF5(Parameters* P, int nfile, const FnameTemplate& fname_template)
{
  chprintf("WARNING from file %s line %d: Rad3D::Write_Restart_HDF5 did nothing", __FILE__, __LINE__);
  CHOLLA_ERROR("Radiation currently requires HDF5 for checkpointing.\n");
}
#endif

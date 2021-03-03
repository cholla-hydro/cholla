#ifdef ANALYSIS

#include <iostream>
#include <fstream>
#include "analysis.h"
#include "../io.h"
#include "../grid3D.h"

// #define OUTPUT_SKEWER

using namespace std;



void Grid3D::Output_Analysis( struct parameters *P ){
  
  FILE *out;
  char filename[180];
  char timestep[20];

  // create the filename
  strcpy(filename, P->analysisdir);
  sprintf(timestep, "%d", Analysis.n_file);
  strcat(filename,timestep);
  // a binary file is created for each process
  // only one HDF5 file is created
  strcat(filename,"_analysis");
  strcat(filename,".h5");
  
  
  chprintf("Writing Analysis File: %d\n", Analysis.n_file);
  
  hid_t   file_id;
  herr_t  status;
  
  // Create a new file collectively
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  Write_Analysis_Header_HDF5( file_id );
  Write_Analysis_Data_HDF5( file_id );
  
  // Close the file
  status = H5Fclose(file_id);
  
  chprintf("Saved Analysis File.\n\n");  
  
}


void Grid3D::Write_Analysis_Header_HDF5( hid_t file_id ){
  hid_t     attribute_id, dataspace_id;
  herr_t    status;
  hsize_t   attr_dims;
  int       int_data[3];
  Real      Real_data[3];
  
  Real H0 = Cosmo.cosmo_h*100;

  // Single attributes first
  attr_dims = 1;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);
  #ifdef COSMOLOGY
  attribute_id = H5Acreate(file_id, "current_a", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.current_a);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "current_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.current_z);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "H0", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &H0);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "Omega_M", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.Omega_M);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "Omega_L", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.Omega_L);
  status = H5Aclose(attribute_id);
  #endif
  
  #ifdef LYA_STATISTICS  
  attribute_id = H5Acreate(file_id, "n_step", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &Analysis.Computed_Flux_Power_Spectrum);
  status = H5Aclose(attribute_id);
  
  #endif
   
  status = H5Sclose(dataspace_id);
  
  // 3D atributes now
  attr_dims = 3;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);
  
  Real_data[0] = Analysis.Lbox_x;
  Real_data[1] = Analysis.Lbox_y;
  Real_data[2] = Analysis.Lbox_z;
  
  attribute_id = H5Acreate(file_id, "Lbox", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, Real_data);
  status = H5Aclose(attribute_id);
  
  
  
  status = H5Sclose(dataspace_id);
  
  
  
}

void Grid3D::Write_Analysis_Data_HDF5( hid_t file_id ){
  
  
  herr_t    status;
  hid_t     dataset_id, dataspace_id, group_id, attribute_id;
  hsize_t   dims2d[2];
  hsize_t   attr_dims;
  int nx_dset, ny_dset, j, i, id, buf_id;
  
  #ifdef PHASE_DIAGRAM
  nx_dset = Analysis.n_temp;
  ny_dset = Analysis.n_dens;
  float *dataset_buffer = (float *) malloc(nx_dset*ny_dset*sizeof(Real));

  
  // Create the data space for the datasets
  dims2d[0] = nx_dset;
  dims2d[1] = ny_dset;
  dataspace_id = H5Screate_simple(2, dims2d, NULL);
  
  group_id = H5Gcreate(file_id, "/phase_diagram", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  for (j=0; j<ny_dset; j++) {
    for (i=0; i<nx_dset; i++) {
      id = i + j*nx_dset; 
      buf_id = j + i*ny_dset;
      dataset_buffer[buf_id] = Analysis.phase_diagram[id];
    }
  }
  dataset_id = H5Dcreate(group_id, "data", H5T_IEEE_F32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);
  
  attr_dims = 1;
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);
  attribute_id = H5Acreate(group_id, "n_temp", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &Analysis.n_temp);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(group_id, "n_dens", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &Analysis.n_dens);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(group_id, "temp_min", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Analysis.temp_min);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(group_id, "temp_max", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Analysis.temp_max);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(group_id, "dens_min", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Analysis.dens_min);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(group_id, "dens_max", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Analysis.dens_max);
  status = H5Aclose(attribute_id);
  
  
  free(dataset_buffer);  
  status = H5Gclose(group_id);
  
  #endif//PHASE_DIAGRAM
  
  
  #ifdef LYA_STATISTICS
  
  group_id = H5Gcreate(file_id, "/lya_statistics", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  
  
  attribute_id = H5Acreate(group_id, "n_skewers", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &Analysis.n_skewers_processed);
  status = H5Aclose(attribute_id);
  
  
  attribute_id = H5Acreate(group_id, "Flux_mean_HI", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Analysis.Flux_mean_HI);
  status = H5Aclose(attribute_id);
  
  attribute_id = H5Acreate(group_id, "Flux_mean_HeII", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Analysis.Flux_mean_HeII);
  status = H5Aclose(attribute_id);
  
  if ( Analysis.Computed_Flux_Power_Spectrum == 1 ){

    
    hid_t ps_group, dataspace_id_ps;
    hsize_t   dims1d_ps[1];
    int n_bins = Analysis.n_hist_edges_x - 1;
    dims1d_ps[0] = n_bins;
    dataspace_id_ps = H5Screate_simple(1, dims1d_ps, NULL);
    
    ps_group = H5Gcreate(group_id, "power_spectrum", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    Real *buffer_ps = (Real *) malloc(n_bins*sizeof(Real));
    
    for ( int bin_id=0; bin_id<n_bins; bin_id++ ){
      buffer_ps[bin_id] = Analysis.k_ceters[bin_id];
    }
    dataset_id = H5Dcreate(ps_group, "k_vals", H5T_IEEE_F64BE, dataspace_id_ps, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_ps);
    status = H5Dclose(dataset_id);
    
    for ( int bin_id=0; bin_id<n_bins; bin_id++ ){
      buffer_ps[bin_id] = Analysis.ps_mean[bin_id];
    }
    dataset_id = H5Dcreate(ps_group, "p(k)", H5T_IEEE_F64BE, dataspace_id_ps, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_ps);
    status = H5Dclose(dataset_id);
    
    free( buffer_ps );
    status = H5Gclose(ps_group);
    
    
  }
  
  
  #ifdef OUTPUT_SKEWERS
  int n_global_x, n_global_y, n_global_z;
  int n_los_x, n_los_y, n_los_z;
  n_global_x = Analysis.n_skewers_processed_x;
  n_global_y = Analysis.n_skewers_processed_y;
  n_global_z = Analysis.n_skewers_processed_z;
  n_los_x = Analysis.nx_total;
  n_los_y = Analysis.ny_total;
  n_los_z = Analysis.nz_total;
  
  Real *dataset_buffer_x;
  Real *dataset_buffer_y;
  Real *dataset_buffer_z;
  
  int data_id, buffer_id;
  
  
  //Write Skerwes X
  dataset_buffer_x = (Real *) malloc(n_global_x*n_los_x*sizeof(Real));
  hsize_t  dims_x[2];
  dims_x[0] = n_global_x;
  dims_x[1] = n_los_x;
  hid_t skewers_group_x, dataspace_id_skewers_x;
  dataspace_id_skewers_x = H5Screate_simple(2, dims_x, NULL);
  skewers_group_x        = H5Gcreate(group_id, "skewers_x", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  
  
  for ( int skewer_id=0; skewer_id<n_global_x; skewer_id++ ){
    for ( int los_id=0; los_id<n_los_x; los_id ++ ){
      data_id   = skewer_id * n_los_x + los_id; 
      buffer_id = skewer_id * n_los_x + los_id;
      dataset_buffer_x[buffer_id] = Analysis.skewers_transmitted_flux_HI_x_global[data_id];
    }
  }
  dataset_id = H5Dcreate(skewers_group_x, "los_transmitted_flux_HI", H5T_IEEE_F64BE, dataspace_id_skewers_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_x );
  status = H5Dclose(dataset_id);
  
  for ( int skewer_id=0; skewer_id<n_global_x; skewer_id++ ){
    for ( int los_id=0; los_id<n_los_x; los_id ++ ){
      data_id   = skewer_id * n_los_x + los_id;
      buffer_id = skewer_id * n_los_x + los_id; 
      dataset_buffer_x[buffer_id] = Analysis.skewers_transmitted_flux_HeII_x_global[data_id];
    }
  }
  dataset_id = H5Dcreate(skewers_group_x, "los_transmitted_flux_HeII", H5T_IEEE_F64BE, dataspace_id_skewers_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_x );
  status = H5Dclose(dataset_id);
  free( dataset_buffer_x );
  
  
  //Write Skerwes Y 
  dataset_buffer_y = (Real *) malloc(n_global_y*n_los_y*sizeof(Real));
  hsize_t  dims_y[2];
  dims_y[0] = n_global_y;
  dims_y[1] = n_los_y;
  hid_t skewers_group_y, dataspace_id_skewers_y;
  dataspace_id_skewers_y = H5Screate_simple(2, dims_y, NULL);
  skewers_group_y        = H5Gcreate(group_id, "skewers_y", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  
  for ( int skewer_id=0; skewer_id<n_global_y; skewer_id++ ){
    for ( int los_id=0; los_id<n_los_y; los_id ++ ){
      data_id   = skewer_id * n_los_y + los_id;
      buffer_id = skewer_id * n_los_y + los_id; 
      dataset_buffer_y[buffer_id] = Analysis.skewers_transmitted_flux_HI_y_global[data_id];
    }
  }
  dataset_id = H5Dcreate(skewers_group_y, "los_transmitted_flux_HI", H5T_IEEE_F64BE, dataspace_id_skewers_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_y );
  status = H5Dclose(dataset_id);
  
  for ( int skewer_id=0; skewer_id<n_global_y; skewer_id++ ){
    for ( int los_id=0; los_id<n_los_y; los_id ++ ){
      data_id   = skewer_id * n_los_y + los_id; 
      buffer_id = skewer_id * n_los_y + los_id;
      dataset_buffer_y[buffer_id] = Analysis.skewers_transmitted_flux_HeII_y_global[data_id];
    }
  }
  dataset_id = H5Dcreate(skewers_group_y, "los_transmitted_flux_HeII", H5T_IEEE_F64BE, dataspace_id_skewers_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_y );
  status = H5Dclose(dataset_id);
  free( dataset_buffer_y );
  
  //Write Skerwes Z 
  dataset_buffer_z = (Real *) malloc(n_global_z*n_los_z*sizeof(Real));
  hsize_t  dims_z[2];
  dims_z[0] = n_global_z;
  dims_z[1] = n_los_z;
  hid_t skewers_group_z, dataspace_id_skewers_z;
  dataspace_id_skewers_z = H5Screate_simple(2, dims_z, NULL);
  skewers_group_z        = H5Gcreate(group_id, "skewers_z", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  
  for ( int skewer_id=0; skewer_id<n_global_z; skewer_id++ ){
    for ( int los_id=0; los_id<n_los_z; los_id ++ ){
      data_id   = skewer_id * n_los_z + los_id; 
      buffer_id = skewer_id * n_los_z + los_id;
      dataset_buffer_z[buffer_id] = Analysis.skewers_transmitted_flux_HI_z_global[data_id];
    }
  }
  dataset_id = H5Dcreate(skewers_group_z, "los_transmitted_flux_HI", H5T_IEEE_F64BE, dataspace_id_skewers_z, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_z );
  status = H5Dclose(dataset_id);
  
  for ( int skewer_id=0; skewer_id<n_global_z; skewer_id++ ){
    for ( int los_id=0; los_id<n_los_z; los_id ++ ){
      data_id   = skewer_id * n_los_z + los_id;
      buffer_id = skewer_id * n_los_z + los_id; 
      dataset_buffer_z[buffer_id] = Analysis.skewers_transmitted_flux_HeII_z_global[data_id];
    }
  }
  dataset_id = H5Dcreate(skewers_group_z, "los_transmitted_flux_HeII", H5T_IEEE_F64BE, dataspace_id_skewers_z, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_z );
  status = H5Dclose(dataset_id);
  free( dataset_buffer_z );

  int n_ghost;
  n_ghost = Analysis.n_ghost_skewer;
  
  hid_t dataspace_id_skewer_x;
  hsize_t   dims1d_x[1];
  dims1d_x[0] = n_los_x;
  dataspace_id_skewer_x = H5Screate_simple(1, dims1d_x, NULL);
  Real *buffer_skewer_x = (Real *) malloc(n_los_x*sizeof(Real));
  for ( int los_id=0; los_id<n_los_x; los_id++ ){
    buffer_skewer_x[los_id] = Analysis.full_vel_Hubble_x[los_id+n_ghost] / 1e5 ; //km/s
  }
  dataset_id = H5Dcreate(skewers_group_x, "vel_Hubble", H5T_IEEE_F64BE, dataspace_id_skewer_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer_x);
  status = H5Dclose(dataset_id);
    
  hid_t dataspace_id_skewer_y;
  hsize_t   dims1d_y[1];
  dims1d_y[0] = n_los_y;
  dataspace_id_skewer_y = H5Screate_simple(1, dims1d_y, NULL);
  Real *buffer_skewer_y = (Real *) malloc(n_los_y*sizeof(Real));
  for ( int los_id=0; los_id<n_los_y; los_id++ ){
    buffer_skewer_y[los_id] = Analysis.full_vel_Hubble_y[los_id+n_ghost] / 1e5 ; //km/s
  }
  dataset_id = H5Dcreate(skewers_group_y, "vel_Hubble", H5T_IEEE_F64BE, dataspace_id_skewer_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer_y);
  status = H5Dclose(dataset_id);

  hid_t dataspace_id_skewer_z;
  hsize_t   dims1d_z[1];
  dims1d_z[0] = n_los_z;
  dataspace_id_skewer_z = H5Screate_simple(1, dims1d_z, NULL);
  Real *buffer_skewer_z = (Real *) malloc(n_los_z*sizeof(Real));
  for ( int los_id=0; los_id<n_los_z; los_id++ ){
    buffer_skewer_z[los_id] = Analysis.full_vel_Hubble_z[los_id+n_ghost] / 1e5 ; //km/s
  }
  dataset_id = H5Dcreate(skewers_group_z, "vel_Hubble", H5T_IEEE_F64BE, dataspace_id_skewer_z, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer_z);
  status = H5Dclose(dataset_id);
  
  free( buffer_skewer_x );
  free( buffer_skewer_y );
  free( buffer_skewer_z );

  status = H5Gclose(skewers_group_x);
  status = H5Gclose(skewers_group_y);
  status = H5Gclose(skewers_group_z);  

  
  #endif
  
  #ifdef OUTPUT_SKEWER
  int nx, n_ghost;
  nx = Analysis.nx_total;
  n_ghost = Analysis.n_ghost_skewer;
  // printf( "N Ghost: %d\n", n_ghost );
  
  
  hid_t skewer_group, dataspace_id_skewer;
  hsize_t   dims1d[1];
  dims1d[0] = nx;
  dataspace_id_skewer = H5Screate_simple(1, dims1d, NULL);
  
  skewer_group = H5Gcreate(group_id, "skewer", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  
  
  Real *buffer_skewer = (Real *) malloc(nx*sizeof(Real));
  
  for ( int los_id=0; los_id<nx; los_id++ ){
    buffer_skewer[los_id] = Analysis.full_HI_density_x[los_id+n_ghost ];
  }
  dataset_id = H5Dcreate(skewer_group, "HI_density", H5T_IEEE_F64BE, dataspace_id_skewer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer);
  status = H5Dclose(dataset_id);
  
  
  for ( int los_id=0; los_id<nx; los_id++ ){
    buffer_skewer[los_id] = Analysis.full_velocity_x[los_id+n_ghost];
  }
  dataset_id = H5Dcreate(skewer_group, "velocity", H5T_IEEE_F64BE, dataspace_id_skewer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer);
  status = H5Dclose(dataset_id);
  
  
  for ( int los_id=0; los_id<nx; los_id++ ){
    buffer_skewer[los_id] = Analysis.full_temperature_x[los_id+n_ghost];
  }
  dataset_id = H5Dcreate(skewer_group, "temperature", H5T_IEEE_F64BE, dataspace_id_skewer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer);
  status = H5Dclose(dataset_id);
  
  
  for ( int los_id=0; los_id<nx; los_id++ ){
    buffer_skewer[los_id] = Analysis.full_optical_depth_x[los_id+n_ghost];
  }
  dataset_id = H5Dcreate(skewer_group, "optical_depth", H5T_IEEE_F64BE, dataspace_id_skewer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer);
  status = H5Dclose(dataset_id);
  
  for ( int los_id=0; los_id<nx; los_id++ ){
    buffer_skewer[los_id] = Analysis.full_vel_Hubble_x[los_id+n_ghost];
    // printf("%f\n", Analysis.full_vel_Hubble_x[los_id+n_ghost] );
  }
  dataset_id = H5Dcreate(skewer_group, "vel_Hubble", H5T_IEEE_F64BE, dataspace_id_skewer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer);
  status = H5Dclose(dataset_id);
  
  
  for ( int los_id=0; los_id<nx; los_id++ ){
    buffer_skewer[los_id] = Analysis.transmitted_flux_x[los_id];
    // printf("%f\n", Analysis.full_vel_Hubble_x[los_id+n_ghost] );
  }
  dataset_id = H5Dcreate(skewer_group, "transmitted_flux", H5T_IEEE_F64BE, dataspace_id_skewer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer);
  status = H5Dclose(dataset_id);
  
  
  
  status = H5Gclose(skewer_group);
  
  
  #endif
  
  
  
  status = H5Gclose(group_id);
  
  #endif
  
  
  
}

void Analysis_Module::Load_Scale_Outputs( struct parameters *P ) {

  char filename_1[100];
  strcpy(filename_1, P->analysis_scale_outputs_file);
  chprintf( " Loading Analysis Scale_Factor Outpus: %s\n", filename_1);
  
  ifstream file_out ( filename_1 );
  string line;
  Real a_value, current_a;
  if (file_out.is_open()){
    while ( getline (file_out,line) ){
      a_value = atof( line.c_str() );
      scale_outputs.push_back( a_value );
      n_outputs += 1;
      // chprintf("%f\n", a_value);
    }
    file_out.close();
    n_outputs = scale_outputs.size();
    next_output_indx = 0;
    chprintf("  Loaded %d scale outputs \n", n_outputs);
  }
  else{
    chprintf("  Error: Unable to open cosmology outputs file\n");
    exit(1);
  }
  
  chprintf(" Setting next analysis output\n");
  
  int scale_indx = next_output_indx;
  current_a = 1. / ( 1 + current_z );
  a_value = scale_outputs[scale_indx];
  
  while ( (current_a - a_value) > 1e-4  ){
    // chprintf( "%f   %f\n", a_value, current_a);
    scale_indx += 1;
    a_value = scale_outputs[scale_indx];
  }
  next_output_indx = scale_indx;
  next_output = a_value;
  chprintf("  Next output scale index: %d  \n", next_output_indx );
  chprintf("  Next output scale value: %f  \n", next_output);
  
  if ( fabs(current_a - next_output) > 1e-4 ) Output_Now = false;
  else Output_Now = true;
  
  n_file = next_output_indx;
  
}

void Analysis_Module::Set_Next_Scale_Output(  ){

  int scale_indx = next_output_indx;
  Real a_value, current_a;
  current_a = 1. / ( 1 + current_z ); 
  a_value = scale_outputs[scale_indx];
  if  ( ( scale_indx == 0 ) && ( abs(a_value - current_a )<1e-5 ) )scale_indx = 1;
  else scale_indx += 1;
  a_value = scale_outputs[scale_indx];

  next_output_indx = scale_indx;
  next_output = a_value;
  n_file = next_output_indx;
  
  // chprintf("Next Analysis Output: z=%f \n", 1./next_output - 1);
}

#endif

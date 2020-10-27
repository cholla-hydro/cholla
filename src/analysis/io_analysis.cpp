#ifdef ANALYSIS

#include <iostream>
#include <fstream>
#include "analysis.h"
#include "../io.h"
#include "../grid3D.h"

#define OUTPUT_SKEWER

using namespace std;



void Grid3D::Output_Analysis( struct parameters *P ){
  
  FILE *out;
  char filename[100];
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
    buffer_skewer[los_id] = Analysis.skewers_HI_density_root_x[los_id];
  }
  dataset_id = H5Dcreate(skewer_group, "HI_density", H5T_IEEE_F64BE, dataspace_id_skewer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer);
  status = H5Dclose(dataset_id);
  
  
  for ( int los_id=0; los_id<nx; los_id++ ){
    buffer_skewer[los_id] = Analysis.skewers_velocity_root_x[los_id];
  }
  dataset_id = H5Dcreate(skewer_group, "velocity", H5T_IEEE_F64BE, dataspace_id_skewer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_skewer);
  status = H5Dclose(dataset_id);
  
  
  for ( int los_id=0; los_id<nx; los_id++ ){
    buffer_skewer[los_id] = Analysis.skewers_temperature_root_x[los_id];
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
  
  while ( (current_a - a_value) > 1e-3  ){
    // chprintf( "%f   %f\n", a_value, current_a);
    scale_indx += 1;
    a_value = scale_outputs[scale_indx];
  }
  next_output_indx = scale_indx;
  next_output = a_value;
  chprintf("  Next output scale index: %d  \n", next_output_indx );
  chprintf("  Next output scale value: %f  \n", next_output);
  
  if ( fabs(current_a - next_output) > 1e-4 ) output_now = false;
  else output_now = true;
  
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

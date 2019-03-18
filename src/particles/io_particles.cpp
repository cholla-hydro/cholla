#ifdef PARTICLES
#include <unistd.h>
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include"../global.h"
#include"../grid3D.h"
#include"../io.h"
#include"particles_3D.h"

#ifdef HDF5
#include<hdf5.h>
#endif
#ifdef MPI_CHOLLA
#include"../mpi_routines.h"
#endif


void Grid3D::WriteData_Particles( struct parameters P, int nfile)
{
  // Write the particles data to file
  OutputData_Particles( P, nfile);
}


#ifdef HDF5
/*! \fn void Write_Header_HDF5(hid_t file_id)
 *  \brief Write the relevant header info to the HDF5 file. */
void Grid3D::Write_Particles_Header_HDF5( hid_t file_id){
  hid_t     attribute_id, dataspace_id;
  herr_t    status;
  hsize_t   attr_dims;
  int       int_data[3];
  Real      Real_data[3];

  // Single attributes first
  attr_dims = 1;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);
  // Create a group attribute
  attribute_id = H5Acreate(file_id, "t_particles", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  // Write the attribute data
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.t);
  // Close the attribute
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "dt_particles", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.dt);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "n_particles_local", H5T_STD_I64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_ULONG, &Particles.n_local);
  status = H5Aclose(attribute_id);
  #ifdef COSMOLOGY
  attribute_id = H5Acreate(file_id, "current_a", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.current_a);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "current_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.current_z);
  status = H5Aclose(attribute_id);
  #endif

  #ifdef SINGLE_PARTICLE_MASS
  attribute_id = H5Acreate(file_id, "particle_mass", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.particle_mass);
  status = H5Aclose(attribute_id);
  #endif

  status = H5Sclose(dataspace_id);
}


void Grid3D::Write_Particles_Data_HDF5( hid_t file_id){
  part_int_t i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_id;
  Real      *dataset_buffer;
  part_int_t  *dataset_buffer_IDs;
  herr_t    status;
  part_int_t n_local = Particles.n_local;
  // int       nx_dset = H.nx_real;
  hsize_t   dims[1];
  dataset_buffer = (Real *) malloc(n_local*sizeof(Real));


  // Create the data space for the datasets
  dims[0] = n_local;
  dataspace_id = H5Screate_simple(1, dims, NULL);

  // Copy the pos_x vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.pos_x[i];
  dataset_id = H5Dcreate(file_id, "/pos_x", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the pos_y vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.pos_y[i];
  dataset_id = H5Dcreate(file_id, "/pos_y", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the pos_z vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.pos_z[i];
  dataset_id = H5Dcreate(file_id, "/pos_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the vel_x vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.vel_x[i];
  dataset_id = H5Dcreate(file_id, "/vel_x", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the vel_y vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.vel_y[i];
  dataset_id = H5Dcreate(file_id, "/vel_y", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the vel_z vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.vel_z[i];
  dataset_id = H5Dcreate(file_id, "/vel_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  #ifndef SINGLE_PARTICLE_MASS
  // Copy the mass vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.mass[i];
  dataset_id = H5Dcreate(file_id, "/mass", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);
  #endif

  #ifdef PARTICLE_IDS
  dataset_buffer_IDs = (part_int_t *) malloc(n_local*sizeof(part_int_t));
  for ( i=0; i<n_local; i++) dataset_buffer_IDs[i] = Particles.partIDs[i];
  dataset_id = H5Dcreate(file_id, "/particle_IDs", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_IDs);
  status = H5Dclose(dataset_id);
  free(dataset_buffer_IDs);
  #endif

  // 3D case
  int       nx_dset = Particles.G.nx_local;
  int       ny_dset = Particles.G.ny_local;
  int       nz_dset = Particles.G.nz_local;
  hsize_t   dims3d[3];
  dataset_buffer = (Real *) malloc(Particles.G.nz_local*Particles.G.ny_local*Particles.G.nx_local*sizeof(Real));

  // Create the data space for the datasets
  dims3d[0] = nx_dset;
  dims3d[1] = ny_dset;
  dims3d[2] = nz_dset;
  dataspace_id = H5Screate_simple(3, dims3d, NULL);

  // Copy the density array to the memory buffer
  int nGHST = Particles.G.n_ghost_particles_grid;
  for (k=0; k<Particles.G.nz_local; k++) {
    for (j=0; j<Particles.G.ny_local; j++) {
      for (i=0; i<Particles.G.nx_local; i++) {
        id = (i+nGHST) + (j+nGHST)*(Particles.G.nx_local+2*nGHST) + (k+nGHST)*(Particles.G.nx_local+2*nGHST)*(Particles.G.ny_local+2*nGHST);
        buf_id = k + j*Particles.G.nz_local + i*Particles.G.nz_local*Particles.G.ny_local;
        dataset_buffer[buf_id] = Particles.G.density[id];
      }
    }
  }

  dataset_id = H5Dcreate(file_id, "/density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  #ifdef OUTPUT_POTENTIAL
  // Copy the potential array to the memory buffer
  for (k=0; k<Grav.nz_local; k++) {
    for (j=0; j<Grav.ny_local; j++) {
      for (i=0; i<Grav.nx_local; i++) {
        id = (i+N_GHOST_POTENTIAL) + (j+N_GHOST_POTENTIAL)*(Grav.nx_local+2*N_GHOST_POTENTIAL) + (k+N_GHOST_POTENTIAL)*(Grav.nx_local+2*N_GHOST_POTENTIAL)*(Grav.ny_local+2*N_GHOST_POTENTIAL);
        buf_id = k + j*Grav.nz_local + i*Grav.nz_local*Grav.ny_local;
        dataset_buffer[buf_id] = Grav.F.potential_h[id];

      }
    }
  }
  dataset_id = H5Dcreate(file_id, "/grav_potential", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);
  #endif //OUTPUT_POTENTIAL


  free(dataset_buffer);
}
#endif//HDF5



void Grid3D::OutputData_Particles( struct parameters P, int nfile)
{
  FILE *out;
  char filename[80];
  char timestep[20];

  // create the filename
  strcpy(filename, P.outdir);
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);
  // a binary file is created for each process
  #if defined BINARY
  chprintf("\nERROR: Particles only support HDF5 outputs\n")
  return;
  // only one HDF5 file is created
  #elif defined HDF5
  strcat(filename,"_particles");
  strcat(filename,".h5");
  #ifdef MPI_CHOLLA
  sprintf(filename,"%s.%d",filename,procID);
  #endif
  #endif

  #if defined HDF5
  hid_t   file_id;
  herr_t  status;

  // Create a new file collectively
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  Write_Header_HDF5(file_id);
  Write_Particles_Header_HDF5( file_id);
  Write_Particles_Data_HDF5( file_id);

  // Close the file
  status = H5Fclose(file_id);
  #endif
}



#endif
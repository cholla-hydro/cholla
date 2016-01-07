#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include<hdf5.h>
#include"io.h"
#include"grid3D.h"
#ifdef MPI_CHOLLA
#include"mpi_routines.h"
#endif

/* Write the initial conditions */
void WriteData(Grid3D G, struct parameters P, int nfile)
{
#ifndef  MPI_CHOLLA
  /*just use the data output routine*/
  OutputData(G,P,nfile);
#else  /*MPI_CHOLLA*/
  OutputDataMPI(G,P,nfile);
#endif /*MPI_CHOLLA*/
}


/* Output the grid data to file. */
void OutputData(Grid3D G, struct parameters P, int nfile)
{
  FILE *out;
  char filename[80];
  char timestep[20];

  // create the filename
  strcpy(filename, P.outdir); 
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);   
  #if defined BINARY
  strcat(filename,".bin");
  #elif defined HDF5
  strcat(filename,".h5");
  #endif

  // open the file for binary writes
  #if defined BINARY
  out = fopen(filename, "w");
  if(out == NULL) {printf("Error opening output file.\n"); exit(0); }

  // write the header to the output file
  G.Write_Header_Binary(out);

  // write the conserved variables to the output file
  G.Write_Grid_Binary(out);

  // close the output file
  fclose(out);
  
  // create the file for hdf5 writes
  #elif defined HDF5
  hid_t   file_id; /* file identifier */
  herr_t  status;

  // Create a new file using default properties.
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);  

  // Write the header (file attributes)
  G.Write_Header_HDF5(file_id);

  // write the conserved variables to the output file
  G.Write_Grid_HDF5(file_id);

  // close the file
  status = H5Fclose(file_id);
  #endif
}


/*! \fn void Write_Header_Binary(FILE *fp)
 *  \brief Write the relevant header info to a binary output file. */
void Grid3D::Write_Header_Binary(FILE *fp)
{

  // Write the header info to the output file
  //fwrite(&H, sizeof(H), 1, fp);  
  fwrite(&H.n_cells, sizeof(int), 1, fp); 
  fwrite(&H.n_ghost, sizeof(int), 1, fp); 
  fwrite(&H.nx, sizeof(int), 1, fp); 
  fwrite(&H.ny, sizeof(int), 1, fp); 
  fwrite(&H.nz, sizeof(int), 1, fp); 
  fwrite(&H.nx_real, sizeof(int), 1, fp); 
  fwrite(&H.ny_real, sizeof(int), 1, fp); 
  fwrite(&H.nz_real, sizeof(int), 1, fp); 
  fwrite(&H.xbound, sizeof(Real), 1, fp); 
  fwrite(&H.ybound, sizeof(Real), 1, fp); 
  fwrite(&H.zbound, sizeof(Real), 1, fp); 
  fwrite(&H.domlen_x, sizeof(Real), 1, fp); 
  fwrite(&H.domlen_y, sizeof(Real), 1, fp); 
  fwrite(&H.domlen_z, sizeof(Real), 1, fp); 
  fwrite(&H.xblocal, sizeof(Real), 1, fp); 
  fwrite(&H.yblocal, sizeof(Real), 1, fp); 
  fwrite(&H.zblocal, sizeof(Real), 1, fp); 
  fwrite(&H.xdglobal, sizeof(Real), 1, fp); 
  fwrite(&H.ydglobal, sizeof(Real), 1, fp); 
  fwrite(&H.zdglobal, sizeof(Real), 1, fp); 
  fwrite(&H.dx, sizeof(Real), 1, fp); 
  fwrite(&H.dy, sizeof(Real), 1, fp); 
  fwrite(&H.dz, sizeof(Real), 1, fp); 
  fwrite(&H.t, sizeof(Real), 1, fp); 
  fwrite(&H.dt, sizeof(Real), 1, fp); 
  fwrite(&H.t_wall, sizeof(Real), 1, fp); 
  fwrite(&H.n_step, sizeof(int), 1, fp); 
  
}


/*! \fn void Write_Header_HDF5(hid_t file_id)
 *  \brief Write the relevant header info to the HDF5 file. */
void Grid3D::Write_Header_HDF5(hid_t file_id)
{
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
  attribute_id = H5Acreate(file_id, "gamma", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  // Write the attribute data
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &gama);
  // Close the attribute
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "t", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &H.t);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "dt", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &H.dt);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "n_step", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &H.n_step);
  status = H5Aclose(attribute_id);
  // Close the dataspace
  status = H5Sclose(dataspace_id);

  // Now 3D attributes
  attr_dims = 3;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);

  #ifndef MPI_CHOLLA
  int_data[0] = H.nx_real;
  int_data[1] = H.ny_real;
  int_data[2] = H.nz_real;
  #endif
  #ifdef MPI_CHOLLA
  int_data[0] = nx_global_real;
  int_data[1] = ny_global_real;
  int_data[2] = nz_global_real;
  #endif

  attribute_id = H5Acreate(file_id, "dims", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);

  Real_data[0] = H.xbound;
  Real_data[1] = H.ybound;
  Real_data[2] = H.zbound;

  attribute_id = H5Acreate(file_id, "bounds", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, Real_data);
  status = H5Aclose(attribute_id);

  Real_data[0] = H.xdglobal;
  Real_data[1] = H.ydglobal;
  Real_data[2] = H.zdglobal;

  attribute_id = H5Acreate(file_id, "domain", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, Real_data);
  status = H5Aclose(attribute_id);

  Real_data[0] = H.dx;
  Real_data[1] = H.dy;
  Real_data[2] = H.dz;

  attribute_id = H5Acreate(file_id, "dx", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, Real_data);
  status = H5Aclose(attribute_id);
  
  // Close the dataspace
  status = H5Sclose(dataspace_id);

}


/*! \fn void Write_Grid_Binary(FILE *fp)
 *  \brief Write the conserved quantities to a binary output file. */
void Grid3D::Write_Grid_Binary(FILE *fp)
{
  int id, i, j, k;

  // Write the conserved quantities to the output file

  // 1D case
  if (H.nx>1 && H.ny==1 && H.nz==1) {

    id = H.n_ghost;

    fwrite(&(C.density[id]),    sizeof(Real), H.nx_real, fp);
    fwrite(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    fwrite(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    fwrite(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    fwrite(&(C.Energy[id]),     sizeof(Real), H.nx_real, fp);
    #ifdef DE
    fwrite(&(C.GasEnergy[id]),     sizeof(Real), H.nx_real, fp);
    #endif //DE
  }

  // 2D case
  else if (H.nx>1 && H.ny>1 && H.nz==1) {

    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fwrite(&(C.density[id]), sizeof(Real), H.nx_real, fp);
    }    
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fwrite(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fwrite(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fwrite(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fwrite(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
    }
    #ifdef DE
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fwrite(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
    }  
    #endif //DE

  }

  // 3D case
  else {
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fwrite(&(C.density[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fwrite(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fwrite(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fwrite(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fwrite(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    #ifdef DE
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fwrite(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
      }
    }    
    #endif //DE
  }    

}





/*! \fn void Write_Grid_HDF5(hid_t file_id)
 *  \brief Write the grid to a file, at the current simulation time. */
void Grid3D::Write_Grid_HDF5(hid_t file_id)
{
  int i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_id; 
  Real      *dataset_buffer;
  herr_t    status;
  #ifdef MPI_CHOLLA
  hid_t     memspace_id;
  hid_t     plist_id; // property list identifier
  // Create property list for collective dataset write
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  #endif


  // 1D case
  if (H.nx>1 && H.ny==1 && H.nz==1) {

    #ifndef MPI_CHOLLA
    int       nx_dset = H.nx_real;
    #endif
    #ifdef MPI_CHOLLA
    int       nx_dset = nx_global_real;
    hsize_t   count[1];
    hsize_t   offset[1];
    count[0]  = H.nx_real;  //x dimension for this process
    offset[0] = nx_local_start;
    // create the local process memory space (dimensions of the local grid)
    memspace_id = H5Screate_simple(1, count, NULL);    
    #endif

    hsize_t   dims[1];
    dataset_buffer = (Real *) malloc(H.nx_real*sizeof(Real));

    // Create the data space for the datasets
    dims[0] = nx_dset;
    dataspace_id = H5Screate_simple(1, dims, NULL);

    // Copy the density array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.density[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for density
    dataset_id = H5Dcreate(file_id, "/density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the density array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the x momentum array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.momentum_x[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for x momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_x", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the x momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the x momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the y momentum array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.momentum_y[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for y momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_y", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the y momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the y momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the z momentum array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.momentum_z[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for z momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the z momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the z momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif   
    // Free the dataset id
    status = H5Dclose(dataset_id);
   
    // Copy the Energy array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.Energy[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for Energy 
    dataset_id = H5Dcreate(file_id, "/Energy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the Energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the Energy array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    #ifdef DE
    // Copy the internal energy array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.GasEnergy[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for internal energy 
    dataset_id = H5Dcreate(file_id, "/GasEnergy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the internal energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the internal energy array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);
  }


  // 2D case
  if (H.nx>1 && H.ny>1 && H.nz==1) {

    #ifndef MPI_CHOLLA
    int       nx_dset = H.nx_real;
    int       ny_dset = H.ny_real;
    #endif
    #ifdef MPI_CHOLLA
    int       nx_dset = nx_global_real;
    int       ny_dset = ny_global_real;
    hsize_t   count[2];
    hsize_t   offset[2];
    count[0]  = H.nx_real;  //x dimension for this process
    count[1]  = H.ny_real;  //y dimension for this process
    offset[0] = nx_local_start;
    offset[1] = ny_local_start;
    // create the local process memory space (dimensions of the local grid)
    memspace_id = H5Screate_simple(2, count, NULL);    
    #endif

    hsize_t   dims[2];
    dataset_buffer = (Real *) malloc(H.ny_real*H.nx_real*sizeof(Real));

    // Create the data space for the datasets
    dims[0] = nx_dset;
    dims[1] = ny_dset;
    dataspace_id = H5Screate_simple(2, dims, NULL);

    // Copy the density array to the memory buffer
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        dataset_buffer[buf_id] = C.density[id];
      }
    }

    // Create a dataset id for density
    dataset_id = H5Dcreate(file_id, "/density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the density array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the x momentum array to the memory buffer
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        dataset_buffer[buf_id] = C.momentum_x[id];
      }
    }
    // Create a dataset id for x momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_x", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the x momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the x momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the y momentum array to the memory buffer
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        dataset_buffer[buf_id] = C.momentum_y[id];
      }
    }

    // Create a dataset id for y momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_y", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the y momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the y momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the z momentum array to the memory buffer
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        dataset_buffer[buf_id] = C.momentum_z[id];
      }
    }

    // Create a dataset id for z momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the z momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the z momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);
   
    // Copy the Energy array to the memory buffer
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        dataset_buffer[buf_id] = C.Energy[id];
      }
    }

    // Create a dataset id for Energy 
    dataset_id = H5Dcreate(file_id, "/Energy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the Energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the Energy array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    #ifdef DE
    // Copy the internal energy array to the memory buffer
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        dataset_buffer[buf_id] = C.GasEnergy[id];
      }
    }

    // Create a dataset id for internal energy 
    dataset_id = H5Dcreate(file_id, "/GasEnergy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the internal energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the internal energy array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);
  }

  // 3D case
  if (H.nx>1 && H.ny>1 && H.nz>1) {


    #ifndef MPI_CHOLLA
    int       nx_dset = H.nx_real;
    int       ny_dset = H.ny_real;
    int       nz_dset = H.nz_real;
    #endif
    #ifdef MPI_CHOLLA
    int       nx_dset = nx_global_real;
    int       ny_dset = ny_global_real;
    int       nz_dset = nz_global_real;
    hsize_t   count[3];
    hsize_t   offset[3];
    count[0]  = H.nx_real;  //x dimension for this process
    count[1]  = H.ny_real;  //y dimension for this process
    count[2]  = H.nz_real;  //y dimension for this process
    offset[0] = nx_local_start;
    offset[1] = ny_local_start;
    offset[2] = nz_local_start;
    // create the local process memory space (dimensions of the local grid)
    memspace_id = H5Screate_simple(3, count, NULL);    
    #endif

    hsize_t   dims[3];
    //Real      dataset_buffer[H.nz_real][H.ny_real][H.nx_real];
    dataset_buffer = (Real *) malloc(H.nz_real*H.ny_real*H.nx_real*sizeof(Real));

    // Create the data space for the datasets
    dims[0] = nx_dset;
    dims[1] = ny_dset;
    dims[2] = nz_dset;
    dataspace_id = H5Screate_simple(3, dims, NULL);

    // Copy the density array to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = C.density[id];
        }
      }
    }

    // Create a dataset id for density
    dataset_id = H5Dcreate(file_id, "/density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the density array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the x momentum array to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = C.momentum_x[id];
        }
      }
    }

    // Create a dataset id for x momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_x", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the x momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the x momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the y momentum array to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = C.momentum_y[id];
        }
      }
    }

    // Create a dataset id for y momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_y", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the y momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the y momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the z momentum array to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = C.momentum_z[id];
        }
      }
    }

    // Create a dataset id for z momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the z momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the z momentum array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);
   
    // Copy the energy array to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = C.Energy[id];
        }
      }
    }

    // Create a dataset id for Energy 
    dataset_id = H5Dcreate(file_id, "/Energy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the Energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the Energy array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);

    #ifdef DE
    // Copy the internal energy array to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = C.GasEnergy[id];
        }
      }
    }    

    // Create a dataset id for internal energy 
    dataset_id = H5Dcreate(file_id, "/GasEnergy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    #ifndef MPI_CHOLLA
    // Write the internal energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    #endif
    #ifdef MPI_CHOLLA
    // Select hyperslab in dataset
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // Write the internal eneryg array to file
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, dataset_buffer);
    #endif    
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);

  }
  free(dataset_buffer);

}


/* MPI-safe printf routine */
int chprintf(const char * __restrict sdata, ...)
{
  int code = 0;
#ifdef MPI_CHOLLA
  /*limit printf to root process only*/
  if(procID==root)
  {
#endif /*MPI_CHOLLA*/

  va_list ap;
  va_start(ap, sdata);
  code = vfprintf(stdout, sdata, ap);
  va_end(ap);

#ifdef MPI_CHOLLA
  }
#endif /*MPI_CHOLLA*/

  return code;
}

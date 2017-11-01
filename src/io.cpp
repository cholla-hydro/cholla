#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include<math.h>
#ifdef HDF5
#include<hdf5.h>
#endif
#include"io.h"
#include"grid3D.h"
#ifdef MPI_CHOLLA
#include"mpi_routines.h"
#endif
#include"error_handling.h"

/* Write the initial conditions */
void WriteData(Grid3D G, struct parameters P, int nfile)
{
#ifndef  MPI_CHOLLA
  /*just use the data output routine*/
  OutputData(G,P,nfile);
#else  /*MPI_CHOLLA*/
  OutputDataMPI(G,P,nfile);
#endif /*MPI_CHOLLA*/
#ifdef PROJECTION
  OutputProjectedData(G,P,nfile);
#endif /*PROJECTION*/
#ifdef ROTATED_PROJECTION
  OutputRotatedProjectedData(G,P,nfile);
#endif /*PROJECTION*/
}


/* Output the grid data to file. */
void OutputData(Grid3D G, struct parameters P, int nfile)
{
  char filename[100];
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
  FILE *out;
  out = fopen(filename, "w");
  if(out == NULL) {printf("Error opening output file.\n"); exit(-1); }

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

  if (status < 0) {printf("File write failed.\n"); exit(-1); }
  #endif
}

/* Output a projection of the grid data to file. */
void OutputProjectedData(Grid3D G, struct parameters P, int nfile)
{
  char filename[100];
  char timestep[20];
  hid_t   file_id;
  herr_t  status;

  // create the filename
  strcpy(filename, P.outdir); 
  sprintf(timestep, "%d_proj", nfile);
  strcat(filename,timestep);   
  strcat(filename,".h5");

#ifdef MPI_CHOLLA
  sprintf(filename,"%s.%d",filename,procID);
#endif /*MPI_CHOLLA*/

  // Create a new file 
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Write the density and temperature projections to the output file
  G.Write_Projection_HDF5(file_id);

  // Close the file
  status = H5Fclose(file_id);

  if (status < 0) {printf("OutputProjectedData: File write failed. ProcID: %d\n", procID); chexit(-1); }
}

/* Output a rotated projection of the grid data to file. */
void OutputRotatedProjectedData(Grid3D G, struct parameters P, int nfile)
{
  char filename[100];
  char timestep[20];
  hid_t   file_id;
  herr_t  status;

  // create the filename
  strcpy(filename, P.outdir); 
  sprintf(timestep, "%d_rot_proj", nfile);
  strcat(filename,timestep);   
  strcat(filename,".h5");

#ifdef MPI_CHOLLA
  sprintf(filename,"%s.%d",filename,procID);
#endif /*MPI_CHOLLA*/

  //Need to do something about delta index here
  if(G.R.flag_delta==1)
  {
    //if flag_delta==1, then we are just outputting a
    //bunch of rotations of the same snapshot
    int i_delta;
    char fname[200];

    for(i_delta=0;i_delta<G.R.n_delta;i_delta++)
    {
      sprintf(fname,"%s.%d",filename,G.R.i_delta);
      chprintf("Outputting rotated projection %s.\n",fname);

      //determine delta about z by output index
      G.R.delta = 2.0*M_PI*((double) i_delta)/((double) G.R.n_delta);

      // Create a new file 
      file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

      // Write header (file attributes)
      G.Write_Header_Rotated_HDF5(file_id);

      // Write the density and temperature projections to the output file
      G.Write_Rotated_Projection_HDF5(file_id);

      // Close the file
      status = H5Fclose(file_id);
      if (status < 0) {printf("OutputRotatedProjectedData: File write failed. ProcID: %d\n", procID); chexit(-1); }

      //iterate G.R.i_delta
      G.R.i_delta++;
    }

  }
  else if (G.R.flag_delta == 2) {

    // case 2 -- outputting at a rotating delta 
    // rotation rate given in the parameter file
    G.R.delta = fmod(nfile*G.R.ddelta_dt*2.0*PI , (2.0*PI));

    // Create a new file 
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write header (file attributes)
    G.Write_Header_Rotated_HDF5(file_id);

    // Write the density and temperature projections to the output file
    G.Write_Rotated_Projection_HDF5(file_id);

    // Close the file
    status = H5Fclose(file_id);
  }
  else {

    //case 0 -- just output at the delta given in the parameter file

    // Create a new file 
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write header (file attributes)
    G.Write_Header_Rotated_HDF5(file_id);

    // Write the density and temperature projections to the output file
    G.Write_Rotated_Projection_HDF5(file_id);

    // Close the file
    status = H5Fclose(file_id);
  }

  if (status < 0) {printf("OutputRotatedProjectedData: File write failed. ProcID: %d\n", procID); chexit(-1); }
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

#ifdef HDF5
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

  #ifdef MPI_CHOLLA
  int_data[0] = H.nx_real;
  int_data[1] = H.ny_real;
  int_data[2] = H.nz_real;

  attribute_id = H5Acreate(file_id, "dims_local", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);

  int_data[0] = nx_local_start;
  int_data[1] = ny_local_start;
  int_data[2] = nz_local_start;

  attribute_id = H5Acreate(file_id, "offset", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);
  #endif

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

/*! \fn void Write_Header_Rotated_HDF5(hid_t file_id)
 *  \brief Write the relevant header info to the HDF5 file for rotated projection. */
void Grid3D::Write_Header_Rotated_HDF5(hid_t file_id)
{
  hid_t     attribute_id, dataspace_id;
  herr_t    status;
  hsize_t   attr_dims;
  int       int_data[3];
  Real      Real_data[3];
  Real      delta, theta, phi;

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

  //Rotation data
  chprintf("Outputing rotation data with delta = %e, theta = %e, phi = %e, Lx = %f, Lz = %f\n",R.delta,R.theta,R.phi,R.Lx,R.Lz);
  attribute_id = H5Acreate(file_id, "nxr", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &R.nx);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "nzr", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &R.nz);
  status = H5Aclose(attribute_id);
  delta = 180.*R.delta/M_PI;
  attribute_id = H5Acreate(file_id, "delta", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &delta);
  status = H5Aclose(attribute_id);
  theta = 180.*R.theta/M_PI;
  attribute_id = H5Acreate(file_id, "theta", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &theta);
  status = H5Aclose(attribute_id);
  phi = 180.*R.phi/M_PI;
  attribute_id = H5Acreate(file_id, "phi", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &phi);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "Lx", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &R.Lx);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "Lz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &R.Lz);
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

  #ifdef MPI_CHOLLA
  int_data[0] = H.nx_real;
  int_data[1] = H.ny_real;
  int_data[2] = H.nz_real;

  attribute_id = H5Acreate(file_id, "dims_local", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);

  int_data[0] = nx_local_start;
  int_data[1] = ny_local_start;
  int_data[2] = nz_local_start;

  attribute_id = H5Acreate(file_id, "offset", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);
  #endif

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
#endif


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




#ifdef HDF5
/*! \fn void Write_Grid_HDF5(hid_t file_id)
 *  \brief Write the grid to a file, at the current simulation time. */
void Grid3D::Write_Grid_HDF5(hid_t file_id)
{
  int i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_id; 
  Real      *dataset_buffer;
  herr_t    status;


  // 1D case
  if (H.nx>1 && H.ny==1 && H.nz==1) {

    int       nx_dset = H.nx_real;
    hsize_t   dims[1];
    dataset_buffer = (Real *) malloc(H.nx_real*sizeof(Real));

    // Create the data space for the datasets
    dims[0] = nx_dset;
    dataspace_id = H5Screate_simple(1, dims, NULL);

    // Copy the density array to the memory buffer
    // (Remapping to play nicely with python)
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.density[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for density
    dataset_id = H5Dcreate(file_id, "/density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the x momentum array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.momentum_x[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for x momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_x", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the x momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the y momentum array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.momentum_y[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for y momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_y", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the y momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the z momentum array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.momentum_z[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for z momentum 
    dataset_id = H5Dcreate(file_id, "/momentum_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the z momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);
   
    // Copy the Energy array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.Energy[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for Energy 
    dataset_id = H5Dcreate(file_id, "/Energy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the Energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    #ifdef DE
    // Copy the internal energy array to the memory buffer
    id = H.n_ghost;
    memcpy(&dataset_buffer[0], &(C.GasEnergy[id]), H.nx_real*sizeof(Real));    

    // Create a dataset id for internal energy 
    dataset_id = H5Dcreate(file_id, "/GasEnergy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the internal energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);
  }


  // 2D case
  if (H.nx>1 && H.ny>1 && H.nz==1) {

    int       nx_dset = H.nx_real;
    int       ny_dset = H.ny_real;
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
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the x momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the y momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the z momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the Energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the internal energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);
  }

  // 3D case
  if (H.nx>1 && H.ny>1 && H.nz>1) {

    int       nx_dset = H.nx_real;
    int       ny_dset = H.ny_real;
    int       nz_dset = H.nz_real;
    hsize_t   dims[3];
    dataset_buffer = (Real *) malloc(H.nx_real*H.ny_real*H.ny_real*sizeof(Real));

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
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the x momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the y momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the z momentum array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the Energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
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
    // Write the internal energy array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);

  }
  free(dataset_buffer);

}

/*! \fn void Write_Projection_HDF5(hid_t file_id)
 *  \brief Write projected density and temperature data to a file, at the current simulation time. */
void Grid3D::Write_Projection_HDF5(hid_t file_id)
{
  int i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_xy_id, dataspace_xz_id; 
  Real      *dataset_buffer_dxy, *dataset_buffer_dxz;
  Real      *dataset_buffer_Txy, *dataset_buffer_Txz;
  herr_t    status;
  Real dxy, dxz, Txy, Txz, n, T;


  n = T = 0;
  Real mu = 0.6;

  // 3D 
  if (H.nx>1 && H.ny>1 && H.nz>1) {

    int       nx_dset = H.nx_real;
    int       ny_dset = H.ny_real;
    int       nz_dset = H.nz_real;
    hsize_t   dims[2];
    dataset_buffer_dxy = (Real *) malloc(H.nx_real*H.ny_real*sizeof(Real));
    dataset_buffer_dxz = (Real *) malloc(H.nx_real*H.nz_real*sizeof(Real));
    dataset_buffer_Txy = (Real *) malloc(H.nx_real*H.ny_real*sizeof(Real));
    dataset_buffer_Txz = (Real *) malloc(H.nx_real*H.nz_real*sizeof(Real));

    // Create the data space for the datasets
    dims[0] = nx_dset;
    dims[1] = ny_dset;
    dataspace_xy_id = H5Screate_simple(2, dims, NULL);
    dims[1] = nz_dset;
    dataspace_xz_id = H5Screate_simple(2, dims, NULL);

    // Copy the xy density and temperature projections to the memory buffer
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        dxy = 0;
        Txy = 0;
        // for each xy element, sum over the z column
        for (k=0; k<H.nz_real; k++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          // sum density
          dxy += C.density[id]*H.dz;
          // calculate number density
          n = C.density[id]*DENSITY_UNIT/(mu*MP);
          // calculate temperature
          #ifdef DE
          T = C.GasEnergy[id]*PRESSURE_UNIT*(gama-1.0) / (n*KB);
          #endif
          Txy += T*C.density[id]*H.dz;
        }
        buf_id = j + i*H.ny_real;
        dataset_buffer_dxy[buf_id] = dxy;
        dataset_buffer_Txy[buf_id] = Txy;
      }
    }

    // Copy the xz density and temperature projections to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (i=0; i<H.nx_real; i++) {
        dxz = 0;
        Txz = 0;
        // for each xz element, sum over the y column
        for (j=0; j<H.ny_real; j++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          // sum density
          dxz += C.density[id]*H.dy;
          // calculate number density
          n = C.density[id]*DENSITY_UNIT/(mu*MP);
          // calculate temperature
          #ifdef DE
          T = C.GasEnergy[id]*PRESSURE_UNIT*(gama-1.0) / (n*KB);
          #endif
          Txz += T*C.density[id]*H.dy;
        }
        buf_id = k + i*H.nz_real;
        dataset_buffer_dxz[buf_id] = dxz;
        dataset_buffer_Txz[buf_id] = Txz;
      }
    }
    

    // Create a dataset id for projected xy density
    dataset_id = H5Dcreate(file_id, "/d_xy", H5T_IEEE_F64BE, dataspace_xy_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_dxy); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Create a dataset id for projected xz density
    dataset_id = H5Dcreate(file_id, "/d_xz", H5T_IEEE_F64BE, dataspace_xz_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_dxz); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Create a dataset id for projected xy temperature 
    dataset_id = H5Dcreate(file_id, "/T_xy", H5T_IEEE_F64BE, dataspace_xy_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected temperature array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_Txy); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Create a dataset id for projected xz density
    dataset_id = H5Dcreate(file_id, "/T_xz", H5T_IEEE_F64BE, dataspace_xz_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected temperature array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_Txz); 
    // Free the dataset id
    status = H5Dclose(dataset_id);    

  }
  else printf("Projection write only implemented for 3D data.\n");

  free(dataset_buffer_dxy);
  free(dataset_buffer_dxz);
  free(dataset_buffer_Txy);
  free(dataset_buffer_Txz);

}

/*! \fn void Write_Rotated_Projection_HDF5(hid_t file_id)
 *  \brief Write rotated projected data to a file, at the current simulation time. */
void Grid3D::Write_Rotated_Projection_HDF5(hid_t file_id)
{
  int i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_xzr_id;
  Real      *dataset_buffer_dxzr;
  Real      *dataset_buffer_Txzr;
  Real      *dataset_buffer_vxxzr;
  Real      *dataset_buffer_vyxzr;
  Real      *dataset_buffer_vzxzr;

  herr_t    status;
  Real d, dxy, dxz, Txy, Txz, n, T;
  Real vx, vy, vz;

  Real x, y, z;     //cell positions
  Real xp, yp, zp;  //rotated positions
  Real alpha, beta; //projected positions
  int  ix, iz;      //projected index positions
  Real cd,sd,cp,sp,ct,st; //sines and cosines
  Real a00, a01, a02;     //rotation matrix elements
  Real a10, a11, a12;
  Real a20, a21, a22;

  n = T = 0;
  Real mu = 0.6;

  //compute trig functions of rotation angles
  cd = cos(R.delta);
  sd = sin(R.delta);
  cp = cos(R.phi);
  sp = sin(R.phi);
  ct = cos(R.theta);
  st = sin(R.theta);

  //compute the rotation matrix elements
  /*a00 =       cosp*cosd - sinp*cost*sind;
  a01 = -1.0*(cosp*sind + sinp*cost*cosd);
  a02 =       sinp*sint;
  
  a10 =       sinp*cosd + cosp*cost*sind;
  a11 =      (cosp*cost*cosd - sint*sind);
  a12 = -1.0* cosp*sint;

  a20 =       sint*sind;
  a21 =       sint*cosd;
  a22 =       cost;*/
  a00 = (cp*cd - sp*ct*sd);
  a01 = -1.0*(cp*sd+sp*ct*cd);
  a02 = sp*st;
  a10 = (sp*cd + cp*ct*sd);
  a11 = (cp*ct*cd -st*sd);
  a12 = cp*st;
  a20 = st*sd;
  a21 = st*cd;
  a22 = ct;

  srand(137);     //initialize a random number
  Real eps = 0.1; //randomize cell centers slightly to combat aliasing

  // 3D 
  if (H.nx>1 && H.ny>1 && H.nz>1) {

    int       nx_dset = R.nx;
    int       nz_dset = R.nz;
    Real      Lx = R.Lx; //projected box size in x dir
    Real      Lz = R.Lz; //projected box size in z dir

    hsize_t   dims[2];

    //allocate and initialize zero
    dataset_buffer_dxzr  = (Real *) calloc(nx_dset*nz_dset,sizeof(Real));
    dataset_buffer_Txzr  = (Real *) calloc(nx_dset*nz_dset,sizeof(Real));
    dataset_buffer_vxxzr = (Real *) calloc(nx_dset*nz_dset,sizeof(Real));
    dataset_buffer_vyxzr = (Real *) calloc(nx_dset*nz_dset,sizeof(Real));
    dataset_buffer_vzxzr = (Real *) calloc(nx_dset*nz_dset,sizeof(Real));

    // Create the data space for the datasets
    dims[0] = nx_dset;
    dims[1] = nz_dset;
    dataspace_xzr_id = H5Screate_simple(2, dims, NULL);

    // Copy the xz rotated projection to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (i=0; i<H.nx_real; i++) {
        for (j=0; j<H.ny_real; j++) {

          //get cell index
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;

          //get cell positions
          Get_Position(i+H.n_ghost, j+H.n_ghost, k+H.n_ghost, &x, &y, &z);

          //add very slight noise to locations
          x += eps*H.dx * (drand48() - 0.5);
          y += eps*H.dy * (drand48() - 0.5);
          z += eps*H.dz * (drand48() - 0.5);

          //rotate cell positions
          xp = a00*x + a01*y + a02*z;
          yp = a10*x + a11*y + a12*z;
          zp = a20*x + a21*y + a22*z;


          //find projected locations
          //assumes box centered at [0,0,0]
          alpha = (nx_dset*(xp+0.5*Lx)/Lx);
          beta  = (nz_dset*(zp+0.5*Lz)/Lz);
          ix = (int) round(alpha);
          iz = (int) round(beta);

          //project density
          if((ix>=0)&&(ix<nx_dset)&&(iz>=0)&&(iz<nz_dset))
          {
            buf_id = iz + ix*nz_dset;
            d = C.density[id];
            dataset_buffer_dxzr[buf_id] += d*H.dy;
            // calculate number density
            n = d*DENSITY_UNIT/(mu*MP);
            // calculate temperature
            #ifdef DE
            T = C.GasEnergy[id]*PRESSURE_UNIT*(gama-1.0) / (n*KB);
            #endif
            Txz = T*d*H.dy;
            dataset_buffer_Txzr[buf_id] += Txz;

            //compute velocities
            vx = C.momentum_x[id];
            dataset_buffer_vxxzr[buf_id] += vx*H.dy;
            vy = C.momentum_y[id];
            dataset_buffer_vyxzr[buf_id] += vy*H.dy;
            vz = C.momentum_z[id];
            dataset_buffer_vzxzr[buf_id] += vz*H.dy;
          }
        }
      }
    }


    // Create a dataset id for projected xy density
    dataset_id = H5Dcreate(file_id, "/d_xzr", H5T_IEEE_F64BE, dataspace_xzr_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_dxzr); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Create a dataset id for projected xz density
    dataset_id = H5Dcreate(file_id, "/T_xzr", H5T_IEEE_F64BE, dataspace_xzr_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_Txzr); 
    // Free the dataset id
    status = H5Dclose(dataset_id);


    // Create a dataset id for projected xz density
    dataset_id = H5Dcreate(file_id, "/vx_xzr", H5T_IEEE_F64BE, dataspace_xzr_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_vxxzr); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Create a dataset id for projected xz density
    dataset_id = H5Dcreate(file_id, "/vy_xzr", H5T_IEEE_F64BE, dataspace_xzr_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_vyxzr); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Create a dataset id for projected xz density
    dataset_id = H5Dcreate(file_id, "/vz_xzr", H5T_IEEE_F64BE, dataspace_xzr_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the projected density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_vzxzr); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

  }
  else printf("Rotated projection write only implemented for 3D data.\n");

  //free the data
  free(dataset_buffer_dxzr);
  free(dataset_buffer_Txzr);
  free(dataset_buffer_vxxzr);
  free(dataset_buffer_vyxzr);
  free(dataset_buffer_vzxzr);

}

#endif


/*! \fn void Read_Grid(struct parameters P)
 *  \brief Read in grid data from an output file. */
void Grid3D::Read_Grid(struct parameters P) {

  char filename[100];
  char timestep[20];
  int nfile = P.nfile; //output step you want to read from

  // create the filename to read from
  // assumes your data is in the outdir specified in the input file
  strcpy(filename, P.outdir);
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);
  #if defined BINARY
  strcat(filename,".bin");
  #elif defined HDF5
  strcat(filename,".h5");
  #endif
  // for now assumes you will run on the same number of processors
  #ifdef MPI_CHOLLA
  sprintf(filename,"%s.%d",filename,procID);
  #endif

  #if defined BINARY
  FILE *fp;
  // open the file
  fp = fopen(filename, "r");
  if (!fp) {
    printf("Unable to open input file.\n");
    exit(0);
  }

  // read in grid data
  Read_Grid_Binary(fp);

  // close the file
  fclose(fp);

  #elif defined HDF5
  hid_t  file_id;
  herr_t  status;

  // open the file
  file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    printf("Unable to open input file.\n");
    exit(0);
  }

  // read in grid data
  Read_Grid_HDF5(file_id);

  // close the file
  status = H5Fclose(file_id);
  #endif


}


/*! \fn Read_Grid_Binary(FILE *fp)
 *  \brief Read in grid data from a binary file. */
void Grid3D::Read_Grid_Binary(FILE *fp)
{
  int id, i, j, k;

  // Read in the header data
  fread(&H.n_cells, sizeof(int), 1, fp); 
  fread(&H.n_ghost, sizeof(int), 1, fp); 
  fread(&H.nx, sizeof(int), 1, fp); 
  fread(&H.ny, sizeof(int), 1, fp); 
  fread(&H.nz, sizeof(int), 1, fp); 
  fread(&H.nx_real, sizeof(int), 1, fp); 
  fread(&H.ny_real, sizeof(int), 1, fp); 
  fread(&H.nz_real, sizeof(int), 1, fp); 
  fread(&H.xbound, sizeof(Real), 1, fp); 
  fread(&H.ybound, sizeof(Real), 1, fp); 
  fread(&H.zbound, sizeof(Real), 1, fp); 
  fread(&H.domlen_x, sizeof(Real), 1, fp); 
  fread(&H.domlen_y, sizeof(Real), 1, fp); 
  fread(&H.domlen_z, sizeof(Real), 1, fp); 
  fread(&H.xblocal, sizeof(Real), 1, fp); 
  fread(&H.yblocal, sizeof(Real), 1, fp); 
  fread(&H.zblocal, sizeof(Real), 1, fp); 
  fread(&H.xdglobal, sizeof(Real), 1, fp); 
  fread(&H.ydglobal, sizeof(Real), 1, fp); 
  fread(&H.zdglobal, sizeof(Real), 1, fp); 
  fread(&H.dx, sizeof(Real), 1, fp); 
  fread(&H.dy, sizeof(Real), 1, fp); 
  fread(&H.dz, sizeof(Real), 1, fp); 
  fread(&H.t, sizeof(Real), 1, fp); 
  fread(&H.dt, sizeof(Real), 1, fp); 
  fread(&H.t_wall, sizeof(Real), 1, fp); 
  fread(&H.n_step, sizeof(int), 1, fp); 
  //fread(&H, 1, 184, fp);


  // Read in the conserved quantities from the input file
  #ifdef WITH_GHOST
  fread(&(C.density[id]),    sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_x[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_y[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_z[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.Energy[id]),     sizeof(Real), H.n_cells, fp); 
  #endif //WITH_GHOST

  #ifdef NO_GHOST
  // 1D case
  if (H.nx>1 && H.ny==1 && H.nz==1) {

    id = H.n_ghost;

    fread(&(C.density[id]),    sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.Energy[id]),     sizeof(Real), H.nx_real, fp);
    #ifdef DE
    fread(&(C.GasEnergy[id]),  sizeof(Real), H.nx_real, fp);
    #endif
  }

  // 2D case
  else if (H.nx>1 && H.ny>1 && H.nz==1) {
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.density[id]), sizeof(Real), H.nx_real, fp);
    }    
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    }     
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
    }     
    #ifdef DE
    for (j=0; j<H.ny_real; j++) {
      id = H.n_ghost + (j+H.n_ghost)*H.nx;
      fread(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
    }     
    #endif
  }

  // 3D case
  else {
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.density[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    #ifdef DE
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        fread(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
      }
    }  
    #endif

  }    
  #endif

}



#ifdef HDF5
/*! \fn void Read_Grid_HDF5(hid_t file_id)
 *  \brief Read in grid data from an hdf5 file. */
void Grid3D::Read_Grid_HDF5(hid_t file_id)
{
  int i, j, k, id, buf_id;
  hid_t     attribute_id, dataset_id; 
  Real      *dataset_buffer;
  herr_t    status;

  // Read in header values not set by grid initialization
  attribute_id = H5Aopen(file_id, "gamma", H5P_DEFAULT); 
  status = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &gama);
  status = H5Aclose(attribute_id);
  attribute_id = H5Aopen(file_id, "t", H5P_DEFAULT); 
  status = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &H.t);
  status = H5Aclose(attribute_id);
  attribute_id = H5Aopen(file_id, "dt", H5P_DEFAULT); 
  status = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &H.dt);
  status = H5Aclose(attribute_id);
  attribute_id = H5Aopen(file_id, "n_step", H5P_DEFAULT); 
  status = H5Aread(attribute_id, H5T_NATIVE_INT, &H.n_step);
  status = H5Aclose(attribute_id);

  // 1D case
  if (H.nx>1 && H.ny==1 && H.nz==1) {

    // need a dataset buffer to remap fastest index
    dataset_buffer = (Real *) malloc(H.nx_real*sizeof(Real));


    // Open the density dataset
    dataset_id = H5Dopen(file_id, "/density", H5P_DEFAULT);
    // Read the density array into the dataset buffer // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the density array to the grid 
    id = H.n_ghost;
    memcpy(&(C.density[id]), &dataset_buffer[0], H.nx_real*sizeof(Real));  


    // Open the x momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_x", H5P_DEFAULT);
    // Read the x momentum array into the dataset buffer // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the x momentum array to the grid 
    id = H.n_ghost;
    memcpy(&(C.momentum_x[id]), &dataset_buffer[0], H.nx_real*sizeof(Real));    


    // Open the y momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_y", H5P_DEFAULT);
    // Read the x momentum array into the dataset buffer // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the y momentum array to the grid 
    id = H.n_ghost;
    memcpy(&(C.momentum_y[id]), &dataset_buffer[0], H.nx_real*sizeof(Real));   


    // Open the z momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_x", H5P_DEFAULT);
    // Read the x momentum array into the dataset buffer // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the z momentum array to the grid 
    id = H.n_ghost;
    memcpy(&(C.momentum_z[id]), &dataset_buffer[0], H.nx_real*sizeof(Real));    


    // Open the Energy dataset
    dataset_id = H5Dopen(file_id, "/Energy", H5P_DEFAULT);
    // Read the Energy array into the dataset buffer // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the Energy array to the grid 
    id = H.n_ghost;
    memcpy(&(C.Energy[id]), &dataset_buffer[0], H.nx_real*sizeof(Real)); 


    #ifdef DE
    // Open the internal energy dataset
    dataset_id = H5Dopen(file_id, "/GasEnergy", H5P_DEFAULT);
    // Read the Energy array into the dataset buffer // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the internal energy array to the grid 
    id = H.n_ghost;
    memcpy(&(C.GasEnergy[id]), &dataset_buffer[0], H.nx_real*sizeof(Real));    
    #endif

  }


  // 2D case
  if (H.nx>1 && H.ny>1 && H.nz==1) {

    // need a dataset buffer to remap fastest index
    dataset_buffer = (Real *) malloc(H.ny_real*H.nx_real*sizeof(Real));


    // Open the density dataset
    dataset_id = H5Dopen(file_id, "/density", H5P_DEFAULT);
    // Read the density array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the density array to the grid 
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        C.density[id] = dataset_buffer[buf_id];
      }
    }


    // Open the x momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_x", H5P_DEFAULT);
    // Read the x momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the x momentum array to the grid 
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        C.momentum_x[id] = dataset_buffer[buf_id];
      }
    }


    // Open the y momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_y", H5P_DEFAULT);
    // Read the y momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the y momentum array to the grid 
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        C.momentum_y[id] = dataset_buffer[buf_id];
      }
    }


    // Open the z momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_z", H5P_DEFAULT);
    // Read the z momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the z momentum array to the grid 
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        C.momentum_z[id] = dataset_buffer[buf_id];
      }
    }


    // Open the Energy dataset
    dataset_id = H5Dopen(file_id, "/Energy", H5P_DEFAULT);
    // Read the Energy array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the Energy array to the grid 
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        C.Energy[id] = dataset_buffer[buf_id];
      }
    }


    #ifdef DE
    // Open the internal energy dataset
    dataset_id = H5Dopen(file_id, "/GasEnergy", H5P_DEFAULT);
    // Read the internal energy array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the internal energy array to the grid 
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        buf_id = j + i*H.ny_real;
        C.GasEnergy[id] = dataset_buffer[buf_id];
      }
    }    
    #endif

  }

  // 3D case
  if (H.nx>1 && H.ny>1 && H.nz>1) {

    // need a dataset buffer to remap fastest index
    dataset_buffer = (Real *) malloc(H.nz_real*H.ny_real*H.nx_real*sizeof(Real));


    // Open the density dataset
    dataset_id = H5Dopen(file_id, "/density", H5P_DEFAULT);
    // Read the density array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the density array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.density[id] = dataset_buffer[buf_id];
        }
      }
    }


    // Open the x momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_x", H5P_DEFAULT);
    // Read the x momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the x momentum array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.momentum_x[id] = dataset_buffer[buf_id];
        }
      }
    }


    // Open the y momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_y", H5P_DEFAULT);
    // Read the y momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the y momentum array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.momentum_y[id] = dataset_buffer[buf_id];
        }
      }
    }


    // Open the z momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_z", H5P_DEFAULT);
    // Read the z momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the z momentum array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.momentum_z[id] = dataset_buffer[buf_id];
        }
      }
    }


    // Open the Energy dataset
    dataset_id = H5Dopen(file_id, "/Energy", H5P_DEFAULT);
    // Read the Energy array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the Energy array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.Energy[id] = dataset_buffer[buf_id];
        }
      }
    }


    #ifdef DE
    // Open the internal Energy dataset
    dataset_id = H5Dopen(file_id, "/GasEnergy", H5P_DEFAULT);
    // Read the internal Energy array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    // Copy the internal Energy array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.GasEnergy[id] = dataset_buffer[buf_id];
        }
      }
    }    
    #endif

  }
  free(dataset_buffer);

}
#endif



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

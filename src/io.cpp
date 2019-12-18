#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include <iostream>
#include <fstream>
#include<math.h>
#include<algorithm>
#include<ctime>
#ifdef HDF5
#include<hdf5.h>
#endif
#include"io.h"
#include"grid3D.h"
#ifdef MPI_CHOLLA
#include"mpi_routines.h"
#endif
#include"error_handling.h"

#ifdef COSMOLOGY
#include "cosmology/cosmology.h"
#endif

using namespace std;

/* function used to rotate points about an axis in 3D for the rotated projection output routine */
void rotate_point(Real x, Real y, Real z, Real delta, Real phi, Real theta, Real *xp, Real *yp, Real *zp);

void Create_Log_File( struct parameters P ){
  
  #ifdef MPI_CHOLLA 
  if ( procID != 0 ) return;
  #endif
    
  string file_name ( LOG_FILE_NAME );
  chprintf( "\nCreating Log File: %s \n\n", file_name.c_str() );
  
  bool file_exists = false;
  if (FILE *file = fopen(file_name.c_str(), "r")){
    file_exists = true;
    chprintf( "  File exists, appending values: %s \n\n", file_name.c_str() );
    fclose( file );
  } 
  
  // current date/time based on current system
  time_t now = time(0);
  // convert now to string form
  char* dt = ctime(&now);
  
  ofstream out_file;
  out_file.open(file_name.c_str(), ios::app);
  out_file << "\n";
  out_file << "Run date: " << dt;
  out_file.close();
  
}

void Write_Message_To_Log_File( const char* message ){
  
    #ifdef MPI_CHOLLA
    if ( procID != 0 ) return;
    #endif
    
    
    string file_name ( LOG_FILE_NAME );
    ofstream out_file;
    out_file.open(file_name.c_str(), ios::app);
    out_file << message << endl;
    out_file.close();
}

/* Write the initial conditions */
void WriteData(Grid3D &G, struct parameters P, int nfile)
{
  

  #ifdef COSMOLOGY
  chprintf( "\nSaving Snapshot: %d \n", nfile );
  G.Change_Cosmological_Frame_Sytem( false );
  #endif
  
  #ifndef ONLY_PARTICLES
  /*call the data output routine*/
  OutputData(G,P,nfile);
  #endif
  
  #ifdef PROJECTION
  OutputProjectedData(G,P,nfile);
  #endif /*PROJECTION*/
  #ifdef ROTATED_PROJECTION
  OutputRotatedProjectedData(G,P,nfile);
  #endif /*ROTATED_PROJECTION*/
  #ifdef SLICES
  OutputSlices(G,P,nfile);
  #endif /*SLICES*/
  
  #ifdef PARTICLES
  G.WriteData_Particles( P, nfile );
  #endif
  
  #ifdef COSMOLOGY
  if ( G.H.OUTPUT_SCALE_FACOR || G.H.Output_Initial){
    G.Cosmo.Set_Next_Scale_Output();
    chprintf( " Saved Snapshot: %d     a:%f   next_output: %f\n", nfile, G.Cosmo.current_a, G.Cosmo.next_output );
    G.H.Output_Initial = false;
  }
  else chprintf( " Saved Snapshot: %d     a:%f     z:%f\n", nfile, G.Cosmo.current_a, G.Cosmo.current_z );
  G.Change_Cosmological_Frame_Sytem( true );
  chprintf( "\n" );
  G.H.Output_Now = false;
  #endif
}


/* Output the grid data to file. */
void OutputData(Grid3D &G, struct parameters P, int nfile)
{
  char filename[100];
  char timestep[20];
  int flag = 0;

  // status of flag determines whether to output the full grid
  // (use nfull = 1 if not using projection or slice output routines)
  if (nfile % P.nfull != 0) flag = 1;

  if (flag == 0) {
    // create the filename
    strcpy(filename, P.outdir); 
    sprintf(timestep, "%d", nfile/P.nfull);
    strcat(filename,timestep);   
    #if defined BINARY
    strcat(filename,".bin");
    #elif defined HDF5
    strcat(filename,".h5");
    #endif
    #ifdef MPI_CHOLLA
    sprintf(filename,"%s.%d",filename,procID);
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
}


/* Output a projection of the grid data to file. */
void OutputProjectedData(Grid3D &G, struct parameters P, int nfile)
{
  char filename[100];
  char timestep[20];
  #ifdef HDF5
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

  #ifdef MPI_CHOLLA
  if (status < 0) {printf("OutputProjectedData: File write failed. ProcID: %d\n", procID); chexit(-1); }
  #else
  if (status < 0) {printf("OutputProjectedData: File write failed.\n"); exit(-1); }
  #endif

  #else
  printf("OutputProjected Data only defined for hdf5 writes.\n");
  #endif //HDF5
}


/* Output a rotated projection of the grid data to file. */
void OutputRotatedProjectedData(Grid3D &G, struct parameters P, int nfile)
{
  char filename[100];
  char timestep[20];
  #ifdef HDF5
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
      #ifdef MPI_CHOLLA
      if (status < 0) {printf("OutputRotatedProjectedData: File write failed. ProcID: %d\n", procID); chexit(-1); }
      #else
      if (status < 0) {printf("OutputRotatedProjectedData: File write failed.\n"); exit(-1); }
      #endif

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

  #ifdef MPI_CHOLLA
  if (status < 0) {printf("OutputRotatedProjectedData: File write failed. ProcID: %d\n", procID); chexit(-1); }
  #else
  if (status < 0) {printf("OutputRotatedProjectedData: File write failed.\n"); exit(-1); }
  #endif

  #else
  printf("OutputRotatedProjectedData only defined for HDF5 writes.\n");
  #endif
}


/* Output xy, xz, and yz slices of the grid data. */
void OutputSlices(Grid3D &G, struct parameters P, int nfile)
{
  char filename[100];
  char timestep[20];
  #ifdef HDF5
  hid_t   file_id;
  herr_t  status;

  // create the filename
  strcpy(filename, P.outdir); 
  sprintf(timestep, "%d_slice", nfile);
  strcat(filename,timestep);   
  strcat(filename,".h5");

  #ifdef MPI_CHOLLA
  sprintf(filename,"%s.%d",filename,procID);
  #endif /*MPI_CHOLLA*/

  // Create a new file 
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Write slices of all variables to the output file
  G.Write_Slices_HDF5(file_id);

  // Close the file
  status = H5Fclose(file_id);

  #ifdef MPI_CHOLLA
  if (status < 0) {printf("OutputSlices: File write failed. ProcID: %d\n", procID); chexit(-1); }
  #else
  if (status < 0) {printf("OutputSlices: File write failed.\n"); exit(-1); }
  #endif
  #else
  printf("OutputSlices only defined for hdf5 writes.\n");
  #endif //HDF5
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
  attribute_id = H5Acreate(file_id, "n_fields", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &H.n_fields);
  status = H5Aclose(attribute_id);
  
  #ifdef COSMOLOGY
  attribute_id = H5Acreate(file_id, "H0", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.H0);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "Omega_M", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.Omega_M);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "Omega_L", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.Omega_L);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "Current_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.current_z);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "Current_a", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.current_a);
  status = H5Aclose(attribute_id);
  #endif
  
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
  int_data[0] = nx_global;
  int_data[1] = ny_global;
  int_data[2] = nz_global;
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

  #ifdef MPI_CHOLLA
  // determine the size of the projection to output for this subvolume
  Real x, y, z, xp, yp, zp;
  Real alpha, beta;
  int ix, iz;
  R.nx_min = R.nx;
  R.nx_max = 0;
  R.nz_min = R.nz;
  R.nz_max = 0;
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
        // find the corners of this domain in the rotated position
        Get_Position(H.n_ghost+i*(H.nx-2*H.n_ghost), H.n_ghost+j*(H.ny-2*H.n_ghost), H.n_ghost+k*(H.nz-2*H.n_ghost), &x, &y, &z);
        // rotate cell position
        rotate_point(x, y, z, R.delta, R.phi, R.theta, &xp, &yp, &zp);
        //find projected location
        //assumes box centered at [0,0,0]
        alpha = (R.nx*(xp+0.5*R.Lx)/R.Lx);
        beta  = (R.nz*(zp+0.5*R.Lz)/R.Lz);
        ix = (int) round(alpha);
        iz = (int) round(beta);
        R.nx_min = std::min(ix, R.nx_min);
        R.nx_max = std::max(ix, R.nx_max);
        R.nz_min = std::min(iz, R.nz_min);
        R.nz_max = std::max(iz, R.nz_max);
      }
    }
  }
  // if the corners aren't within the chosen projection area
  // take the input projection edge as the edge of this piece of the projection
  R.nx_min = std::max(R.nx_min, 0);
  R.nx_max = std::min(R.nx_max, R.nx);
  R.nz_min = std::max(R.nz_min, 0);
  R.nz_max = std::min(R.nz_max, R.nz);
  #endif

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
  attribute_id = H5Acreate(file_id, "n_fields", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &H.n_fields);
  status = H5Aclose(attribute_id);  

  //Rotation data
  attribute_id = H5Acreate(file_id, "nxr", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &R.nx);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "nzr", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &R.nz);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "nx_min", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &R.nx_min);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "nz_min", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &R.nz_min);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "nx_max", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &R.nx_max);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "nz_max", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); 
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, &R.nz_max);
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
  int_data[0] = nx_global;
  int_data[1] = ny_global;
  int_data[2] = nz_global;
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

  chprintf("Outputting rotation data with delta = %e, theta = %e, phi = %e, Lx = %f, Lz = %f\n",R.delta,R.theta,R.phi,R.Lx,R.Lz);

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

    #ifdef SCALAR
    for (int s=0; s<NSCALARS; s++) {
      // create the name of the dataset
      char dataset[100]; 
      char number[10];
      strcpy(dataset, "/scalar"); 
      sprintf(number, "%d", s);
      strcat(dataset,number);        
      // Copy the scalar array to the memory buffer
      id = H.n_ghost;
      memcpy(&dataset_buffer[0], &(C.scalar[id+s*H.n_cells]), H.nx_real*sizeof(Real));    

      // Create a dataset id for the scalar
      dataset_id = H5Dcreate(file_id, dataset, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      // Write the scalar array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
      status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
      // Free the dataset id
      status = H5Dclose(dataset_id);
    }
    #endif

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


    #ifdef SCALAR
    for (int s=0; s<NSCALARS; s++) {
      // create the name of the dataset
      char dataset[100]; 
      char number[10];
      strcpy(dataset, "/scalar"); 
      sprintf(number, "%d", s);
      strcat(dataset,number);        
      // Copy the scalar array to the memory buffer
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
          buf_id = j + i*H.ny_real;
          dataset_buffer[buf_id] = C.scalar[id+s*H.n_cells];
        }
      }
      // Create a dataset id for the scalar
      dataset_id = H5Dcreate(file_id, dataset, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      // Write the scalar array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
      status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
      // Free the dataset id
      status = H5Dclose(dataset_id);
    }
    #endif


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
    dataset_buffer = (Real *) malloc(H.nx_real*H.ny_real*H.nz_real*sizeof(Real));

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

    #ifdef SCALAR
    #ifndef COOLING_GRACKLE // Dont write scalars when using grackle
    for (int s=0; s<NSCALARS; s++) {
      // create the name of the dataset
      char dataset[100]; 
      char number[10];
      strcpy(dataset, "/scalar"); 
      sprintf(number, "%d", s);
      strcat(dataset,number);        
      // Copy the scalar array to the memory buffer
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            dataset_buffer[buf_id] = C.scalar[id+s*H.n_cells];
          }
        }
      }
      // Create a dataset id for the scalar
      dataset_id = H5Dcreate(file_id, dataset, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      // Write the scalar array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
      status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
      // Free the dataset id
      status = H5Dclose(dataset_id);
    }
    #else // Write Chemistry when using GRACKLE
    #ifdef OUTPUT_CHEMISTRY
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = Cool.fields.HI_density[id];
        }
      }
    }
    dataset_id = H5Dcreate(file_id, "/HI_density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    status = H5Dclose(dataset_id);

    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = Cool.fields.HII_density[id];
        }
      }
    }
    dataset_id = H5Dcreate(file_id, "/HII_density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    status = H5Dclose(dataset_id);

    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = Cool.fields.HeI_density[id];
        }
      }
    }
    dataset_id = H5Dcreate(file_id, "/HeI_density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    status = H5Dclose(dataset_id);

    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = Cool.fields.HeII_density[id];
        }
      }
    }
    dataset_id = H5Dcreate(file_id, "/HeII_density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    status = H5Dclose(dataset_id);

    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = Cool.fields.HeIII_density[id];
        }
      }
    }
    dataset_id = H5Dcreate(file_id, "/HeIII_density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    status = H5Dclose(dataset_id);

    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = Cool.fields.e_density[id];
        }
      }
    }
    dataset_id = H5Dcreate(file_id, "/e_density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    status = H5Dclose(dataset_id);

    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = Cool.fields.metal_density[id];
        }
      }
    }
    dataset_id = H5Dcreate(file_id, "/metal_density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    status = H5Dclose(dataset_id);

    #endif //OUTPUT_CHEMISTRY

    #ifdef OUTPUT_TEMPERATURE
    // Copy the internal energy array to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = Cool.temperature[id];
        }
      }
    }
    // Create a dataset id for density
    dataset_id = H5Dcreate(file_id, "/temperature", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif //OUTPUT_TEMPERATURE
    
    #ifdef OUTPUT_DUAL_ENERGY_FLAGS
    // Copy the internal energy array to the memory buffer
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          dataset_buffer[buf_id] = (Real) Cool.flags_DE[id];
        }
      }
    }
    // Create a dataset id for density
    dataset_id = H5Dcreate(file_id, "/flags_DE", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif //OUTPUT_DUAL_ENERGY_FLAGS

    #endif //COOLING_GRACKLE
    #endif //SCALAR

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
    
    #ifdef GRAVITY
    #ifdef OUTPUT_POTENTIAL
    // Copy the potential array to the memory buffer
    for (k=0; k<Grav.nz_local; k++) {
      for (j=0; j<Grav.ny_local; j++) {
        for (i=0; i<Grav.nx_local; i++) {
          // id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          // buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          id = (i+N_GHOST_POTENTIAL) + (j+N_GHOST_POTENTIAL)*(Grav.nx_local+2*N_GHOST_POTENTIAL) + (k+N_GHOST_POTENTIAL)*(Grav.nx_local+2*N_GHOST_POTENTIAL)*(Grav.ny_local+2*N_GHOST_POTENTIAL);
          buf_id = k + j*Grav.nz_local + i*Grav.nz_local*Grav.ny_local;
          dataset_buffer[buf_id] = Grav.F.potential_h[id];
        }
      }
    }
    // Create a dataset id for density
    dataset_id = H5Dcreate(file_id, "/potential", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    // Free the dataset id
    status = H5Dclose(dataset_id);
    #endif//OUTPUT_POTENTIAL
    #endif//GRAVITY

    // Free the dataspace id
    status = H5Sclose(dataspace_id);

  }
  free(dataset_buffer);

}
#endif //HDF5


#ifdef HDF5
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

    // Free the dataspace ids
    status = H5Sclose(dataspace_xz_id);
    status = H5Sclose(dataspace_xy_id);
  }
  else printf("Projection write only works for 3D data.\n");

  free(dataset_buffer_dxy);
  free(dataset_buffer_dxz);
  free(dataset_buffer_Txy);
  free(dataset_buffer_Txz);

}
#endif //HDF5


#ifdef HDF5
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
  Real dxy, dxz, Txy, Txz;
  Real d, n, T, vx, vy, vz;

  Real x, y, z;     //cell positions
  Real xp, yp, zp;  //rotated positions
  Real alpha, beta; //projected positions
  int  ix, iz;      //projected index positions

  n = T = 0;
  Real mu = 0.6;

  srand(137);     //initialize a random number
  Real eps = 0.1; //randomize cell centers slightly to combat aliasing

  // 3D 
  if (H.nx>1 && H.ny>1 && H.nz>1) {

    Real      Lx = R.Lx; //projected box size in x dir
    Real      Lz = R.Lz; //projected box size in z dir
    int nx_dset = R.nx;
    int nz_dset = R.nz;

    // set the projected dataset size for this process to capture
    // this piece of the simulation volume
    // min and max values were set in the header write
    int nx_min, nx_max, nz_min, nz_max;
    nx_min = R.nx_min;
    nx_max = R.nx_max;
    nz_min = R.nz_min;
    nz_max = R.nz_max;
    nx_dset = nx_max-nx_min;
    nz_dset = nz_max-nz_min;

    hsize_t   dims[2];

    // allocate the buffers for the projected dataset
    // and initialize to zero
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
          rotate_point(x, y, z, R.delta, R.phi, R.theta, &xp, &yp, &zp);

          //find projected locations
          //assumes box centered at [0,0,0]
          alpha = (R.nx*(xp+0.5*R.Lx)/R.Lx);
          beta  = (R.nz*(zp+0.5*R.Lz)/R.Lz);
          ix = (int) round(alpha);
          iz = (int) round(beta);
          #ifdef MPI_CHOLLA
          ix = ix - nx_min;
          iz = iz - nz_min;
          #endif

          if((ix>=0)&&(ix<nx_dset)&&(iz>=0)&&(iz<nz_dset))
          {
            buf_id = iz + ix*nz_dset;
            d = C.density[id];
            // project density
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

    // Free the dataspace id
    status = H5Sclose(dataspace_xzr_id);

  }
  else printf("Rotated projection write only implemented for 3D data.\n");

  //free the data
  free(dataset_buffer_dxzr);
  free(dataset_buffer_Txzr);
  free(dataset_buffer_vxxzr);
  free(dataset_buffer_vyxzr);
  free(dataset_buffer_vzxzr);

}
#endif //HDF5


#ifdef HDF5
/*! \fn void Write_Slices_HDF5(hid_t file_id)
 *  \brief Write centered xy, xz, and yz slices of all variables to a file, 
     at the current simulation time. */
void Grid3D::Write_Slices_HDF5(hid_t file_id)
{
  int i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_id; 
  Real      *dataset_buffer_d;
  Real      *dataset_buffer_mx;
  Real      *dataset_buffer_my;
  Real      *dataset_buffer_mz;
  Real      *dataset_buffer_E;
  #ifdef DE
  Real      *dataset_buffer_GE;
  #endif
  #ifdef SCALAR
  Real      *dataset_buffer_scalar;
  #endif
  herr_t    status;
  int xslice, yslice, zslice;
  xslice = H.nx/2;
  yslice = H.ny/2;
  zslice = H.nz/2;
  #ifdef MPI_CHOLLA
  xslice = nx_global/2;
  yslice = ny_global/2;
  zslice = nz_global/2;
  #endif


  // 3D 
  if (H.nx>1 && H.ny>1 && H.nz>1) {

    int       nx_dset = H.nx_real;
    int       ny_dset = H.ny_real;
    int       nz_dset = H.nz_real;
    hsize_t   dims[2];


    // Create the xy data space for the datasets
    dims[0] = nx_dset;
    dims[1] = ny_dset;
    dataspace_id = H5Screate_simple(2, dims, NULL);

    // Allocate memory for the xy slices
    dataset_buffer_d  = (Real *) malloc(H.nx_real*H.ny_real*sizeof(Real));
    dataset_buffer_mx = (Real *) malloc(H.nx_real*H.ny_real*sizeof(Real));
    dataset_buffer_my = (Real *) malloc(H.nx_real*H.ny_real*sizeof(Real));
    dataset_buffer_mz = (Real *) malloc(H.nx_real*H.ny_real*sizeof(Real));
    dataset_buffer_E  = (Real *) malloc(H.nx_real*H.ny_real*sizeof(Real));
    #ifdef DE
    dataset_buffer_GE = (Real *) malloc(H.nx_real*H.ny_real*sizeof(Real));
    #endif
    #ifdef SCALAR
    dataset_buffer_scalar = (Real *) malloc(NSCALARS*H.nx_real*H.ny_real*sizeof(Real));
    #endif

    // Copy the xy slices to the memory buffers
    for (j=0; j<H.ny_real; j++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + zslice*H.nx*H.ny;
        buf_id = j + i*H.ny_real;
        #ifdef MPI_CHOLLA
        // When there are multiple processes, check whether this slice is in your domain
        if (zslice >= nz_local_start && zslice < nz_local_start+nz_local) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (zslice-nz_local_start+H.n_ghost)*H.nx*H.ny;
        #endif //MPI_CHOLLA
          dataset_buffer_d[buf_id]  = C.density[id];
          dataset_buffer_mx[buf_id] = C.momentum_x[id];
          dataset_buffer_my[buf_id] = C.momentum_y[id];
          dataset_buffer_mz[buf_id] = C.momentum_z[id];
          dataset_buffer_E[buf_id]  = C.Energy[id];
          #ifdef DE
          dataset_buffer_GE[buf_id] = C.GasEnergy[id];
          #endif
          #ifdef SCALAR
          for (int ii=0; ii<NSCALARS; ii++) {
            dataset_buffer_scalar[buf_id+ii*H.nx*H.ny] = C.scalar[id+ii*H.n_cells];
          }
          #endif
        #ifdef MPI_CHOLLA
        }
        // if the slice isn't in your domain, just write out zeros
        else {
          dataset_buffer_d[buf_id]  = 0;
          dataset_buffer_mx[buf_id] = 0;
          dataset_buffer_my[buf_id] = 0;
          dataset_buffer_mz[buf_id] = 0;
          dataset_buffer_E[buf_id]  = 0;
          #ifdef DE
          dataset_buffer_GE[buf_id] = 0;
          #endif
          #ifdef SCALAR
          for (int ii=0; ii<NSCALARS; ii++) {
            dataset_buffer_scalar[buf_id+ii*H.nx*H.ny] = 0;
          }
          #endif
        } 
        #endif // MPI_CHOLLA
      }
    }

    // Write out the xy datasets for each variable 
    dataset_id = H5Dcreate(file_id, "/d_xy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_d); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/mx_xy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_mx); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/my_xy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_my); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/mz_xy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_mz); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/E_xy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_E); 
    status = H5Dclose(dataset_id);
    #ifdef DE
    dataset_id = H5Dcreate(file_id, "/GE_xy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_GE); 
    status = H5Dclose(dataset_id);
    #endif
    #ifdef SCALAR
    dataset_id = H5Dcreate(file_id, "/scalar_xy", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_scalar); 
    status = H5Dclose(dataset_id);
    #endif
    // Free the dataspace id
    status = H5Sclose(dataspace_id);

    // free the dataset buffers
    free(dataset_buffer_d);
    free(dataset_buffer_mx);
    free(dataset_buffer_my);
    free(dataset_buffer_mz);
    free(dataset_buffer_E);
    #ifdef DE
    free(dataset_buffer_GE);
    #endif
    #ifdef SCALAR
    free(dataset_buffer_scalar);
    #endif
    

    // Create the xz data space for the datasets
    dims[0] = nx_dset;
    dims[1] = nz_dset;
    dataspace_id = H5Screate_simple(2, dims, NULL);

    // allocate the memory for the xz slices
    dataset_buffer_d  = (Real *) malloc(H.nx_real*H.nz_real*sizeof(Real));
    dataset_buffer_mx = (Real *) malloc(H.nx_real*H.nz_real*sizeof(Real));
    dataset_buffer_my = (Real *) malloc(H.nx_real*H.nz_real*sizeof(Real));
    dataset_buffer_mz = (Real *) malloc(H.nx_real*H.nz_real*sizeof(Real));
    dataset_buffer_E  = (Real *) malloc(H.nx_real*H.nz_real*sizeof(Real));
    #ifdef DE
    dataset_buffer_GE = (Real *) malloc(H.nx_real*H.nz_real*sizeof(Real));
    #endif
    #ifdef SCALAR
    dataset_buffer_scalar = (Real *) malloc(NSCALARS*H.nx_real*H.nz_real*sizeof(Real));
    #endif
    

    // Copy the xz slices to the memory buffers
    for (k=0; k<H.nz_real; k++) {
      for (i=0; i<H.nx_real; i++) {
        id = (i+H.n_ghost) + yslice*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        buf_id = k + i*H.nz_real;
        #ifdef MPI_CHOLLA
        // When there are multiple processes, check whether this slice is in your domain
        if (yslice >= ny_local_start && yslice < ny_local_start+ny_local) {
          id = (i+H.n_ghost) + (yslice-ny_local_start+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        #endif //MPI_CHOLLA
        dataset_buffer_d[buf_id]  = C.density[id];
        dataset_buffer_mx[buf_id] = C.momentum_x[id];
        dataset_buffer_my[buf_id] = C.momentum_y[id];
        dataset_buffer_mz[buf_id] = C.momentum_z[id];
        dataset_buffer_E[buf_id]  = C.Energy[id];
        #ifdef DE
        dataset_buffer_GE[buf_id] = C.GasEnergy[id];
        #endif
        #ifdef SCALAR
        for (int ii=0; ii<NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id+ii*H.nx*H.nz] = C.scalar[id+ii*H.n_cells];
        }
        #endif
        #ifdef MPI_CHOLLA
        }
        // if the slice isn't in your domain, just write out zeros
        else {
          dataset_buffer_d[buf_id]  = 0;
          dataset_buffer_mx[buf_id] = 0;
          dataset_buffer_my[buf_id] = 0;
          dataset_buffer_mz[buf_id] = 0;
          dataset_buffer_E[buf_id]  = 0;
          #ifdef DE
          dataset_buffer_GE[buf_id] = 0;
          #endif
          #ifdef SCALAR
          for (int ii=0; ii<NSCALARS; ii++) {
            dataset_buffer_scalar[buf_id+ii*H.nx*H.nz] = 0;
          }
          #endif
        } 
        #endif // MPI_CHOLLA
      }
    }

    // Write out the xz datasets for each variable 
    dataset_id = H5Dcreate(file_id, "/d_xz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_d); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/mx_xz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_mx); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/my_xz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_my); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/mz_xz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_mz); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/E_xz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_E); 
    status = H5Dclose(dataset_id);
    #ifdef DE
    dataset_id = H5Dcreate(file_id, "/GE_xz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_GE); 
    status = H5Dclose(dataset_id);
    #endif
    #ifdef SCALAR
    dataset_id = H5Dcreate(file_id, "/scalar_xz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_scalar); 
    status = H5Dclose(dataset_id);
    #endif
    // Free the dataspace id
    status = H5Sclose(dataspace_id);

    // free the dataset buffers 
    free(dataset_buffer_d);
    free(dataset_buffer_mx);
    free(dataset_buffer_my);
    free(dataset_buffer_mz);
    free(dataset_buffer_E);
    #ifdef DE
    free(dataset_buffer_GE);
    #endif
    #ifdef SCALAR
    free(dataset_buffer_scalar);
    #endif


    // Create the yz data space for the datasets
    dims[0] = ny_dset;
    dims[1] = nz_dset;
    dataspace_id = H5Screate_simple(2, dims, NULL);

    // allocate the memory for the yz slices
    dataset_buffer_d  = (Real *) malloc(H.ny_real*H.nz_real*sizeof(Real));
    dataset_buffer_mx = (Real *) malloc(H.ny_real*H.nz_real*sizeof(Real));
    dataset_buffer_my = (Real *) malloc(H.ny_real*H.nz_real*sizeof(Real));
    dataset_buffer_mz = (Real *) malloc(H.ny_real*H.nz_real*sizeof(Real));
    dataset_buffer_E  = (Real *) malloc(H.ny_real*H.nz_real*sizeof(Real));
    #ifdef DE
    dataset_buffer_GE = (Real *) malloc(H.ny_real*H.nz_real*sizeof(Real));
    #endif
    #ifdef SCALAR
    dataset_buffer_scalar = (Real *) malloc(NSCALARS*H.ny_real*H.nz_real*sizeof(Real));
    #endif
    

    // Copy the yz slices to the memory buffers
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        id = xslice + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        buf_id = k + j*H.nz_real;
        #ifdef MPI_CHOLLA
        // When there are multiple processes, check whether this slice is in your domain
        if (xslice >= nx_local_start && xslice < nx_local_start+nx_local) {
          id = (xslice-nx_local_start) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        #endif //MPI_CHOLLA
        dataset_buffer_d[buf_id]  = C.density[id];
        dataset_buffer_mx[buf_id] = C.momentum_x[id];
        dataset_buffer_my[buf_id] = C.momentum_y[id];
        dataset_buffer_mz[buf_id] = C.momentum_z[id];
        dataset_buffer_E[buf_id]  = C.Energy[id];
        #ifdef DE
        dataset_buffer_GE[buf_id] = C.GasEnergy[id];
        #endif
        #ifdef SCALAR
        for (int ii=0; ii<NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id+ii*H.ny*H.nz] = C.scalar[id+ii*H.n_cells];
        }
        #endif
        #ifdef MPI_CHOLLA
        }
        // if the slice isn't in your domain, just write out zeros
        else {
          dataset_buffer_d[buf_id]  = 0;
          dataset_buffer_mx[buf_id] = 0;
          dataset_buffer_my[buf_id] = 0;
          dataset_buffer_mz[buf_id] = 0;
          dataset_buffer_E[buf_id]  = 0;
          #ifdef DE
          dataset_buffer_GE[buf_id] = 0;
          #endif
          #ifdef SCALAR
          for (int ii=0; ii<NSCALARS; ii++) {
            dataset_buffer_scalar[buf_id+ii*H.ny*H.nz] = 0;
          }
          #endif
        } 
        #endif // MPI_CHOLLA
      }
    }

    // Write out the yz datasets for each variable 
    dataset_id = H5Dcreate(file_id, "/d_yz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_d); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/mx_yz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_mx); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/my_yz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_my); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/mz_yz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_mz); 
    status = H5Dclose(dataset_id);
    dataset_id = H5Dcreate(file_id, "/E_yz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_E); 
    status = H5Dclose(dataset_id);
    #ifdef DE
    dataset_id = H5Dcreate(file_id, "/GE_yz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_GE); 
    status = H5Dclose(dataset_id);
    #endif
    #ifdef SCALAR
    dataset_id = H5Dcreate(file_id, "/scalar_yz", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_scalar); 
    status = H5Dclose(dataset_id);
    #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);

    // free the dataset buffers
    free(dataset_buffer_d);
    free(dataset_buffer_mx);
    free(dataset_buffer_my);
    free(dataset_buffer_mz);
    free(dataset_buffer_E);
    #ifdef DE
    free(dataset_buffer_GE);
    #endif
    #ifdef SCALAR
    free(dataset_buffer_scalar);
    #endif


  }
  else printf("Slice write only works for 3D data.\n");

}
#endif //HDF5


/*! \fn void Read_Grid(struct parameters P)
 *  \brief Read in grid data from an output file. */
void Grid3D::Read_Grid(struct parameters P) {

  char filename[100];
  char timestep[20];
  int nfile = P.nfile; //output step you want to read from

  // create the filename to read from
  // assumes your data is in the outdir specified in the input file
  // strcpy(filename, P.outdir);
  // Changed to read initial conditions from indir
  strcpy(filename, P.indir);
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);
  #if defined BINARY
  strcat(filename,".bin");
  #elif defined HDF5
  strcat(filename,".h5");
  #endif
  // for now assumes you will run on the same number of processors
  #ifdef MPI_CHOLLA
  #ifdef TILED_INITIAL_CONDITIONS
  sprintf(filename,"%s",filename); //Everyone reads the same file
  #else
  sprintf(filename,"%s.%d",filename,procID);
  #endif //TILED_INITIAL_CONDITIONS
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
    printf("Unable to open input file: %s\n", filename);
    exit(0);
  }

  // read in grid data
  Read_Grid_HDF5(file_id, P);

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
void Grid3D::Read_Grid_HDF5(hid_t file_id, struct parameters P)
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

    #ifdef SCALAR
    for (int s=0; s<NSCALARS; s++) {
      // create the name of the dataset
      char dataset[100]; 
      char number[10];
      strcpy(dataset, "/scalar"); 
      sprintf(number, "%d", s);
      strcat(dataset,number);        
    
      // Open the passive scalar dataset
      dataset_id = H5Dopen(file_id, dataset, H5P_DEFAULT);
      // Read the scalar array into the dataset buffer // NOTE: NEED TO FIX FOR FLOAT REAL!!!
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
      // Free the dataset id
      status = H5Dclose(dataset_id);

      // Copy the scalar array to the grid 
      id = H.n_ghost;
      memcpy(&(C.scalar[id + s*H.n_cells]), &dataset_buffer[0], H.nx_real*sizeof(Real));    
    }
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


    #ifdef SCALAR
    for (int s=0; s<NSCALARS; s++) {
      // create the name of the dataset
      char dataset[100]; 
      char number[10];
      strcpy(dataset, "/scalar"); 
      sprintf(number, "%d", s);
      strcat(dataset,number);        

      // Open the scalar dataset
      dataset_id = H5Dopen(file_id, dataset, H5P_DEFAULT);
      // Read the scalar array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
      // Free the dataset id
      status = H5Dclose(dataset_id);

      // Copy the scalar array to the grid 
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
          buf_id = j + i*H.ny_real;
          C.scalar[id+s*H.n_cells] = dataset_buffer[buf_id];
        }
      }    
    }
    #endif

  }

  // 3D case
  if (H.nx>1 && H.ny>1 && H.nz>1) {

    // Compute Statistic of Initial data
    Real mean_l, min_l, max_l;
    Real mean_g, min_g, max_g;
    
    // need a dataset buffer to remap fastest index
    dataset_buffer = (Real *) malloc(H.nz_real*H.ny_real*H.nx_real*sizeof(Real));


    // Open the density dataset
    dataset_id = H5Dopen(file_id, "/density", H5P_DEFAULT);
    // Read the density array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);
    

    mean_l = 0;
    min_l = 1e65;
    max_l = -1;

    // Copy the density array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.density[id] = dataset_buffer[buf_id];
          mean_l += C.density[id];
          if ( C.density[id] > max_l ) max_l = C.density[id];
          if ( C.density[id] < min_l ) min_l = C.density[id];
        }
      }
    }
    mean_l /= ( H.nz_real * H.ny_real * H.nx_real );

    #if MPI_CHOLLA
    mean_g = ReduceRealAvg( mean_l );
    max_g = ReduceRealMax( max_l );
    min_g = ReduceRealMin( min_l );
    mean_l = mean_g;
    max_l = max_g;
    min_l = min_g;
    #endif

    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY) 
    chprintf( " Density  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3] \n", mean_l, min_l, max_l );
    #endif


    // Open the x momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_x", H5P_DEFAULT);
    // Read the x momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);
    
    mean_l = 0;
    min_l = 1e65;
    max_l = -1;
    // Copy the x momentum array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.momentum_x[id] = dataset_buffer[buf_id];
          mean_l += fabs(C.momentum_x[id]);
          if ( fabs(C.momentum_x[id]) > max_l ) max_l = fabs(C.momentum_x[id]);
          if ( fabs(C.momentum_x[id]) < min_l ) min_l = fabs(C.momentum_x[id]);
        }
      }
    }
    mean_l /= ( H.nz_real * H.ny_real * H.nx_real );
    
    #if MPI_CHOLLA
    mean_g = ReduceRealAvg( mean_l );
    max_g = ReduceRealMax( max_l );
    min_g = ReduceRealMin( min_l );
    mean_l = mean_g;
    max_l = max_g;
    min_l = min_g;
    #endif
    
    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY) 
    chprintf( " abs(Momentum X)  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", mean_l, min_l, max_l );
    #endif

    // Open the y momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_y", H5P_DEFAULT);
    // Read the y momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    mean_l = 0;
    min_l = 1e65;
    max_l = -1;
    // Copy the y momentum array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.momentum_y[id] = dataset_buffer[buf_id];
          mean_l += fabs(C.momentum_y[id]);
          if ( fabs(C.momentum_y[id]) > max_l ) max_l = fabs(C.momentum_y[id]);
          if ( fabs(C.momentum_y[id]) < min_l ) min_l = fabs(C.momentum_y[id]);
        }
      }
    }
    mean_l /= ( H.nz_real * H.ny_real * H.nx_real );
    
    #if MPI_CHOLLA
    mean_g = ReduceRealAvg( mean_l );
    max_g = ReduceRealMax( max_l );
    min_g = ReduceRealMin( min_l );
    mean_l = mean_g;
    max_l = max_g;
    min_l = min_g;
    #endif
    
    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY) 
    chprintf( " abs(Momentum Y)  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", mean_l, min_l, max_l );
    #endif


    // Open the z momentum dataset
    dataset_id = H5Dopen(file_id, "/momentum_z", H5P_DEFAULT);
    // Read the z momentum array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    mean_l = 0;
    min_l = 1e65;
    max_l = -1;
    // Copy the z momentum array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.momentum_z[id] = dataset_buffer[buf_id];
          mean_l += fabs(C.momentum_z[id]);
          if ( fabs(C.momentum_z[id]) > max_l ) max_l = fabs(C.momentum_z[id]);
          if ( fabs(C.momentum_z[id]) < min_l ) min_l = fabs(C.momentum_z[id]);
        }
      }
    }
    mean_l /= ( H.nz_real * H.ny_real * H.nx_real );
    
    #if MPI_CHOLLA
    mean_g = ReduceRealAvg( mean_l );
    max_g = ReduceRealMax( max_l );
    min_g = ReduceRealMin( min_l );
    mean_l = mean_g;
    max_l = max_g;
    min_l = min_g;
    #endif
    
    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY) 
    chprintf( " abs(Momentum Z)  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", mean_l, min_l, max_l );
    #endif


    // Open the Energy dataset
    dataset_id = H5Dopen(file_id, "/Energy", H5P_DEFAULT);
    // Read the Energy array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);

    mean_l = 0;
    min_l = 1e65;
    max_l = -1;
    // Copy the Energy array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.Energy[id] = dataset_buffer[buf_id];
          mean_l += C.Energy[id];
          if ( C.Energy[id] > max_l ) max_l = C.Energy[id];
          if ( C.Energy[id] < min_l ) min_l = C.Energy[id];
        }
      }
    }
    mean_l /= ( H.nz_real * H.ny_real * H.nx_real );
    
    #if MPI_CHOLLA
    mean_g = ReduceRealAvg( mean_l );
    max_g = ReduceRealMax( max_l );
    min_g = ReduceRealMin( min_l );
    mean_l = mean_g;
    max_l = max_g;
    min_l = min_g;
    #endif
    
    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY) 
    chprintf( " Energy  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km^2 s^-2 ] \n", mean_l, min_l, max_l );
    #endif


    #ifdef DE
    // Open the internal Energy dataset
    dataset_id = H5Dopen(file_id, "/GasEnergy", H5P_DEFAULT);
    // Read the internal Energy array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
    // Free the dataset id
    status = H5Dclose(dataset_id);
    
    Real temp, temp_max_l, temp_min_l, temp_mean_l;
    Real temp_min_g, temp_max_g, temp_mean_g;
    temp_mean_l = 0;
    temp_min_l = 1e65;
    temp_max_l = -1;
    mean_l = 0;
    min_l = 1e65;
    max_l = -1;
    // Copy the internal Energy array to the grid 
    for (k=0; k<H.nz_real; k++) {
      for (j=0; j<H.ny_real; j++) {
        for (i=0; i<H.nx_real; i++) {
          id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
          C.GasEnergy[id] = dataset_buffer[buf_id];
          mean_l += C.GasEnergy[id];
          if ( C.GasEnergy[id] > max_l ) max_l = C.GasEnergy[id];
          if ( C.GasEnergy[id] < min_l ) min_l = C.GasEnergy[id];
          temp = C.GasEnergy[id] / C.density[id] * ( gama - 1 ) * MP / KB * 1e10 ;
          temp_mean_l += temp;
          // chprintf( "%f\n", temp);
          if ( temp > temp_max_l ) temp_max_l = temp;
          if ( temp < temp_min_l ) temp_min_l = temp;
        }
      }
    }
    mean_l /= ( H.nz_real * H.ny_real * H.nx_real );
    temp_mean_l /= ( H.nz_real * H.ny_real * H.nx_real );
    
    #if MPI_CHOLLA
    mean_g = ReduceRealAvg( mean_l );
    max_g = ReduceRealMax( max_l );
    min_g = ReduceRealMin( min_l );
    mean_l = mean_g;
    max_l = max_g;
    min_l = min_g;
    temp_mean_g = ReduceRealAvg( temp_mean_l );
    temp_max_g = ReduceRealMax( temp_max_l );
    temp_min_g = ReduceRealMin( temp_min_l );
    temp_mean_l = temp_mean_g;
    temp_max_l = temp_max_g;
    temp_min_l = temp_min_g;
    #endif
    
    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY) 
    chprintf( " GasEnergy  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km^2 s^-2 ] \n", mean_l, min_l, max_l );
    chprintf( " Temperature  Mean: %f   Min: %f   Max: %f      [ K ] \n", temp_mean_l, temp_min_l, temp_max_l );
    #endif
    
    #endif//DE

    #ifdef SCALAR
    #ifndef COOLING_GRACKLE  // Dont Load scalars when using grackle
    for (int s=0; s<NSCALARS; s++) {
      // create the name of the dataset
      char dataset[100]; 
      char number[10];
      strcpy(dataset, "/scalar"); 
      sprintf(number, "%d", s);
      strcat(dataset,number);        

      // Open the scalar dataset
      dataset_id = H5Dopen(file_id, dataset, H5P_DEFAULT);
      // Read the scalar array into the dataset buffer  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer); 
      // Free the dataset id
      status = H5Dclose(dataset_id);

      // Copy the scalar array to the grid 
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            C.scalar[id+s*H.n_cells] = dataset_buffer[buf_id];
          }
        }
      }    
    }
    #else //Load Chemistry when using GRACKLE
    if (P.nfile == 0){
      Real dens;
      Real HI_frac = INITIAL_FRACTION_HI;
      Real HII_frac = INITIAL_FRACTION_HII;
      Real HeI_frac = INITIAL_FRACTION_HEI;
      Real HeII_frac = INITIAL_FRACTION_HEII;
      Real HeIII_frac = INITIAL_FRACTION_HEIII;
      Real e_frac = INITIAL_FRACTION_ELECTRON;
      Real metal_frac = INITIAL_FRACTION_METAL;
      chprintf( " Initial HI Fraction:    %e \n", HI_frac);
      chprintf( " Initial HII Fraction:   %e \n", HII_frac);
      chprintf( " Initial HeI Fraction:   %e \n", HeI_frac);
      chprintf( " Initial HeII Fraction:  %e \n", HeII_frac);
      chprintf( " Initial HeIII Fraction: %e \n", HeIII_frac);
      chprintf( " Initial elect Fraction: %e \n", e_frac);
      chprintf( " Initial metal Fraction: %e \n", metal_frac);
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            dens = C.density[id];
            C.scalar[0*H.n_cells + id] = HI_frac * dens;
            C.scalar[1*H.n_cells + id] = HII_frac * dens;
            C.scalar[2*H.n_cells + id] = HeI_frac * dens;
            C.scalar[3*H.n_cells + id] = HeII_frac * dens;
            C.scalar[4*H.n_cells + id] = HeIII_frac * dens;
            C.scalar[5*H.n_cells + id] = e_frac * dens;
            C.scalar[6*H.n_cells + id] = metal_frac * dens;
          }
        }
      }
    }
    else{
      dataset_id = H5Dopen(file_id, "/HI_density", H5P_DEFAULT);
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
      status = H5Dclose(dataset_id);
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            C.scalar[0*H.n_cells + id] = dataset_buffer[buf_id];
            // chprintf("%f \n",  C.scalar[0*H.n_cells + id] / C.density[id]);
          }
        }
      }
      dataset_id = H5Dopen(file_id, "/HII_density", H5P_DEFAULT);
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
      status = H5Dclose(dataset_id);
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            C.scalar[1*H.n_cells + id] = dataset_buffer[buf_id];
          }
        }
      }
      dataset_id = H5Dopen(file_id, "/HeI_density", H5P_DEFAULT);
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
      status = H5Dclose(dataset_id);
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            C.scalar[2*H.n_cells + id] = dataset_buffer[buf_id];
          }
        }
      }
      dataset_id = H5Dopen(file_id, "/HeII_density", H5P_DEFAULT);
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
      status = H5Dclose(dataset_id);
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            C.scalar[3*H.n_cells + id] = dataset_buffer[buf_id];
          }
        }
      }
      dataset_id = H5Dopen(file_id, "/HeIII_density", H5P_DEFAULT);
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
      status = H5Dclose(dataset_id);
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            C.scalar[4*H.n_cells + id] = dataset_buffer[buf_id];
          }
        }
      }
      dataset_id = H5Dopen(file_id, "/e_density", H5P_DEFAULT);
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
      status = H5Dclose(dataset_id);
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            C.scalar[5*H.n_cells + id] = dataset_buffer[buf_id];
          }
        }
      }
      dataset_id = H5Dopen(file_id, "/metal_density", H5P_DEFAULT);
      status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
      status = H5Dclose(dataset_id);
      for (k=0; k<H.nz_real; k++) {
        for (j=0; j<H.ny_real; j++) {
          for (i=0; i<H.nx_real; i++) {
            id = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            buf_id = k + j*H.nz_real + i*H.nz_real*H.ny_real;
            C.scalar[6*H.n_cells + id] = dataset_buffer[buf_id];
          }
        }
      }
    }
    #endif//COOLING_GRACKLE
    #endif//SCALAR
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


void rotate_point(Real x, Real y, Real z, Real delta, Real phi, Real theta, Real *xp, Real *yp, Real *zp) {

  Real cd,sd,cp,sp,ct,st; //sines and cosines
  Real a00, a01, a02;     //rotation matrix elements
  Real a10, a11, a12;
  Real a20, a21, a22;

  //compute trig functions of rotation angles
  cd = cos(delta);
  sd = sin(delta);
  cp = cos(phi);
  sp = sin(phi);
  ct = cos(theta);
  st = sin(theta);

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

  *xp = a00*x + a01*y + a02*z;
  *yp = a10*x + a11*y + a12*z;
  *zp = a20*x + a21*y + a22*z;

} 

#pragma once

#include "../global/global.h"
#include "../grid/grid3D.h"
#include <iostream>


/* Write the data */
void WriteData(Grid3D &G, struct parameters P, int nfile);

/* Output the grid data to file. */
void OutputData(Grid3D &G, struct parameters P, int nfile);

/* Output the grid data to file as 32-bit floats. */
void OutputFloat32(Grid3D &G, struct parameters P, int nfile);

/* Output a projection of the grid data to file. */
void OutputProjectedData(Grid3D &G, struct parameters P, int nfile);

/* Output a rotated projection of the grid data to file. */
void OutputRotatedProjectedData(Grid3D &G, struct parameters P, int nfile);

/* Output xy, xz, and yz slices of the grid data to file. */
void OutputSlices(Grid3D &G, struct parameters P, int nfile);

/* MPI-safe printf routine */
int chprintf(const char * __restrict sdata, ...);

void Create_Log_File( struct parameters P );

void Write_Message_To_Log_File( const char* message );

void write_debug ( Real *Value, const char *fname, int nValues, int iProc );

#ifdef HDF5
// From io/io.cpp
herr_t HDF5_Dataset(hid_t file_id, hid_t dataspace_id, double* dataset_buffer, const char* name);
herr_t HDF5_Dataset(hid_t file_id, hid_t dataspace_id, float* dataset_buffer, const char* name);

// From io/io_gpu.cu
// Use GPU to pack source -> device_buffer, then copy device_buffer -> buffer, then write HDF5 field 
void WriteHDF5Field3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id, float* buffer, float* device_buffer, Real* source, const char* name);
void WriteHDF5Field3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id, double* buffer, double* device_buffer, Real* source, const char* name);  
#endif

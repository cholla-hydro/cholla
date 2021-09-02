#ifndef IO_CHOLLA_H
#define IO_CHOLLA_H

#include "../global/global.h"
#include "../grid/grid3D.h"
#include <iostream>


/* Write the data */
void WriteData(Grid3D &G, struct parameters P, int nfile);

/* Output the grid data to file. */
void OutputData(Grid3D &G, struct parameters P, int nfile);

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
#endif /*IO_CHOLLA_H*/

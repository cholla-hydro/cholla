#ifndef IO_CHOLLA_H
#define IO_CHOLLA_H

#include"global.h"
#include"grid3D.h"


/* Write the data */
void WriteData(Grid3D G, struct parameters P, int nfile);

/* Output the grid data to file. */
void OutputData(Grid3D G, struct parameters P, int nfile);

#ifdef   MPI_CHOLLA
/* Output the grid data to file. */
void OutputDataMPI(Grid3D G, struct parameters P, int nfile);
#endif /*MPI_CHOLLA*/

/* MPI-safe printf routine */
int chprintf(const char * __restrict sdata, ...);

#endif /*IO_CHOLLA_H*/

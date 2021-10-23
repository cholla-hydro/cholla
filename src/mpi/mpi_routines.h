#ifdef MPI_CHOLLA
#ifndef  MPI_ROUTINES_H
#define  MPI_ROUTINES_H
#include <mpi.h>
#include <stddef.h>
#include "../grid/grid3D.h"
#include "../global/global.h"

#ifdef FFTW
#include "fftw3.h"
#include "fftw3-mpi.h"
#endif /*FFTW*/

/*Global MPI Variables*/
extern int procID; /*process rank*/
extern int nproc;  /*number of processes in global comm*/
extern int root;   /*rank of root process*/
extern int procID_node; /*process rank on node*/
extern int nproc_node;  /*number of MPI processes on node*/

extern MPI_Comm world;	/*global communicator*/
extern MPI_Comm node;	/*communicator for each node*/

extern MPI_Datatype MPI_CHREAL; /*data type describing float precision*/

#ifdef PARTICLES
extern MPI_Datatype MPI_PART_INT; /*data type describing interger for particles precision*/
#endif

//extern MPI_Request send_request[6];
//extern MPI_Request recv_request[6];
extern MPI_Request *send_request;
extern MPI_Request *recv_request;

//MPI destinations and sources
extern int dest[6];
extern int source[6];

/* Decomposition flag */
extern int flag_decomp;

#define SLAB_DECOMP  1 	//slab  decomposition flag
#define BLOCK_DECOMP 2	//block decomposition flag

//Communication buffers
// For SLAB
extern Real *send_buffer_0;
extern Real *send_buffer_1;
extern Real *recv_buffer_0;
extern Real *recv_buffer_1;

// For BLOCK
extern Real *d_send_buffer_x0;
extern Real *d_send_buffer_x1;
extern Real *d_send_buffer_y0;
extern Real *d_send_buffer_y1;
extern Real *d_send_buffer_z0;
extern Real *d_send_buffer_z1;
extern Real *d_recv_buffer_x0;
extern Real *d_recv_buffer_x1;
extern Real *d_recv_buffer_y0;
extern Real *d_recv_buffer_y1;
extern Real *d_recv_buffer_z0;
extern Real *d_recv_buffer_z1;

extern Real *h_send_buffer_x0;
extern Real *h_send_buffer_x1;
extern Real *h_send_buffer_y0;
extern Real *h_send_buffer_y1;
extern Real *h_send_buffer_z0;
extern Real *h_send_buffer_z1;
extern Real *h_recv_buffer_x0;
extern Real *h_recv_buffer_x1;
extern Real *h_recv_buffer_y0;
extern Real *h_recv_buffer_y1;
extern Real *h_recv_buffer_z0;
extern Real *h_recv_buffer_z1;

#ifdef PARTICLES
//Buffers for particles transfers
extern Real *d_send_buffer_x0_particles;
extern Real *d_send_buffer_x1_particles;
extern Real *d_send_buffer_y0_particles;
extern Real *d_send_buffer_y1_particles;
extern Real *d_send_buffer_z0_particles;
extern Real *d_send_buffer_z1_particles;
extern Real *d_recv_buffer_x0_particles;
extern Real *d_recv_buffer_x1_particles;
extern Real *d_recv_buffer_y0_particles;
extern Real *d_recv_buffer_y1_particles;
extern Real *d_recv_buffer_z0_particles;
extern Real *d_recv_buffer_z1_particles;

extern Real *h_send_buffer_x0_particles;
extern Real *h_send_buffer_x1_particles;
extern Real *h_send_buffer_y0_particles;
extern Real *h_send_buffer_y1_particles;
extern Real *h_send_buffer_z0_particles;
extern Real *h_send_buffer_z1_particles;
extern Real *h_recv_buffer_x0_particles;
extern Real *h_recv_buffer_x1_particles;
extern Real *h_recv_buffer_y0_particles;
extern Real *h_recv_buffer_y1_particles;
extern Real *h_recv_buffer_z0_particles;
extern Real *h_recv_buffer_z1_particles;

// Size of the buffers for particles transfers
extern int buffer_length_particles_x0_send;
extern int buffer_length_particles_x0_recv;
extern int buffer_length_particles_x1_send;
extern int buffer_length_particles_x1_recv;
extern int buffer_length_particles_y0_send;
extern int buffer_length_particles_y0_recv;
extern int buffer_length_particles_y1_send;
extern int buffer_length_particles_y1_recv;
extern int buffer_length_particles_z0_send;
extern int buffer_length_particles_z0_recv;
extern int buffer_length_particles_z1_send;
extern int buffer_length_particles_z1_recv;

// Request for Number Of Particles to be transferred
extern MPI_Request *send_request_n_particles;
extern MPI_Request *recv_request_n_particles;
// Request for Particles Transfer
extern MPI_Request *send_request_particles_transfer;
extern MPI_Request *recv_request_particles_transfer;
#endif//PARTICLES


extern int send_buffer_length;
extern int recv_buffer_length;
extern int x_buffer_length;
extern int y_buffer_length;
extern int z_buffer_length;

/*local domain sizes*/
/*none of these include ghost cells!*/
extern ptrdiff_t nx_global;
extern ptrdiff_t ny_global;
extern ptrdiff_t nz_global;
extern ptrdiff_t nx_local;
extern ptrdiff_t ny_local;
extern ptrdiff_t nz_local;
extern ptrdiff_t nx_local_start;
extern ptrdiff_t ny_local_start;
extern ptrdiff_t nz_local_start;

#ifdef   FFTW
extern ptrdiff_t n_local_complex;
#endif /*FFTW*/

/*number of MPI procs in each dimension*/
extern int nproc_x;
extern int nproc_y;
extern int nproc_z;

/*\fn void InitializeChollaMPI(void) */
/* Routine to initialize MPI */
void InitializeChollaMPI(int *pargc, char **pargv[]);

/* Perform domain decomposition */
void DomainDecomposition(struct parameters *P, struct Header *H, int nx_global, int ny_global, int nz_global);
void DomainDecompositionSLAB(struct parameters *P, struct Header *H, int nx_global, int ny_global, int nz_global);
void DomainDecompositionBLOCK(struct parameters *P, struct Header *H, int nx_global, int ny_global, int nz_global);

/*tile MPI processes in a block decomposition*/
void TileBlockDecomposition(void);

/* MPI reduction wrapper for max(Real)*/
Real ReduceRealMax(Real x);

/* MPI reduction wrapper for min(Real)*/
Real ReduceRealMin(Real x);

/* MPI reduction wrapper for avg(Real)*/
Real ReduceRealAvg(Real x);

/* MPI reduction wrapper for sum(Real)*/
Real ReduceRealSum(Real x);
#ifdef PARTICLES
/* MPI reduction wrapper for sum(part_int)*/
Real ReducePartIntSum(part_int_t x);

// Function that checks if the buffer size For the particles transfer is large enough,
// and grows the buffer if needed.
void Check_and_Grow_Particles_Buffer( Real **part_buffer, int *current_size_ptr, int new_size );
#endif

/* Set the domain properties */
void Set_Parallel_Domain(Real xmin_global, Real ymin_global, Real zmin_global, Real xlen_global, Real ylen_global, Real zlen_global, struct Header *H);

/* Print information about the domain properties */
void Print_Domain_Properties(struct Header H);

/* Allocate MPI communication buffers*/
void Allocate_MPI_Buffers(struct Header *H);

/* Allocate MPI communication buffers for a SLAB decomposition */
void Allocate_MPI_Buffers_SLAB(struct Header *H);

/* Allocate MPI communication buffers for a BLOCK decomposition */
void Allocate_MPI_Buffers_BLOCK(struct Header *H);

/* Allocate MPI communication GPU buffers for a BLOCK decomposition */
void Allocate_MPI_DeviceBuffers_BLOCK(struct Header *H);

/* find the greatest prime factor of an integer */
int greatest_prime_factor(int n);


/*! \fn int ***three_dimensional_int_array(int n, int l, int m)
 *  *  \brief Allocate a three dimensional (n x l x m) int array
 *   */
int ***three_dimensional_int_array(int n, int l, int m);

/*! \fn void deallocate_three_int_dimensional_array(int ***x, int n, int l, int m)
 *  \brief De-allocate a three dimensional (n x l x m) int array.
 *   */
void deallocate_three_dimensional_int_array(int ***x, int n, int l, int m);

/* Copy MPI receive buffers on Host to their device locations */
void copyHostToDeviceReceiveBuffer ( int direction );

#endif /*MPI_ROUTINES_H*/
#endif /*MPI_CHOLLA*/

/*! /file global_parallel.h
 *  /brief Declarations of global variables related to parallelism
 *
 *  While most of these variables were originally defined in mpi_routines.h,
 *  there are cases where it can be useful to have them defined (with sensible
 *  defaults) even when compiled without MPI.
 */

#ifndef GLOBAL_PARALLEL_H
#define GLOBAL_PARALLEL_H

// NOTE: it would be nice to put the following in a namespace, but that would
//       involve a bunch of refactoring

extern int procID;      /*process rank*/
extern int nproc;       /*number of processes executing simulation*/
extern int root;        /*rank of root process*/
extern int procID_node; /*process rank on node*/
extern int nproc_node;  /*number of processes on node*/

#endif /* GLOBAL_PARALLEL_H */
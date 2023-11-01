#include "../global/global_parallel.h"

#include "../utils/error_handling.h"

// global parallelism related variables

int procID; /*process rank*/
int nproc;  /*number of processes in global comm*/
int root;   /*rank of root process*/

void Init_Global_Parallel_Vars_No_MPI()
{
#ifdef MPI_CHOLLA
  CHOLLA_ERROR("This function should not be executed when compiled with MPI");
#endif
  procID = 0;
  nproc  = 1;
  root   = 0;
}
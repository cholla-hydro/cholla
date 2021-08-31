#ifndef CUDA_MPI_ROUTINES
#define CUDA_MPI_ROUTINES 

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

/*! \fn int initialize_cuda_mpi(int myid, int nprocs);
 *  \brief CUDA initialization within MPI. */
int initialize_cuda_mpi(int myid, int nprocs);

#ifdef __cplusplus
}
#endif //__cplusplus


#endif //CUDA_MPI_ROUTINES 

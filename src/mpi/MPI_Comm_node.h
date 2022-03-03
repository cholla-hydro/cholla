#ifndef MPI_COMM_NODE
#define MPI_COMM_NODE

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus
MPI_Comm MPI_Comm_node(int *pid, int *np);
#ifdef __cplusplus
}
#endif //__cplusplus

#endif //MPI_COMM_NODE

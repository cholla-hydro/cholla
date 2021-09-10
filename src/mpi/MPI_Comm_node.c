#ifdef MPI_CHOLLA
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "../mpi/MPI_Comm_node.h"


/*! \fn int djb2_hash(char *str)
 *  \brief Simple hash function by Dan Bernstein */
int djb2_hash(char *str);

/*! \fn MPI_Comm MPI_Comm_node(void)
 *  \brief Returns an MPI_Comm for processes on each node.*/
MPI_Comm MPI_Comm_node(int *myid_node, int *nproc_node)
{
  int  myid;				    //global rank
  int  nproc;				    //global rank
  char pname[MPI_MAX_PROCESSOR_NAME];	    //node hostname
  int  pname_length;			    //length of node hostname
  int hash;				    //hash of node hostname

  MPI_Comm node_comm;			    //communicator for the procs on each node

  //get the global process rank
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);


  //if we're the only process, then just return
  //the global rank, size, and comm
  if(nproc==1)
  {
    *myid_node  = myid;
    *nproc_node = nproc;
    return MPI_COMM_WORLD;
  }

  //get the hostname of the node
  MPI_Get_processor_name(pname, &pname_length);

  //hash the name of the node
  hash = abs(djb2_hash(pname));

  //printf("hash %d\n",hash);

  //split the communicator
  MPI_Comm_split(MPI_COMM_WORLD, hash, myid, &node_comm);

  //get size and rank
  MPI_Comm_rank(node_comm,myid_node);
  MPI_Comm_size(node_comm,nproc_node);

  //return the communicator for processors on the node
  return node_comm;
}

/*! \fn int djb2_hash(char *str)
 *  \brief Simple hash function by Dan Bernstein */
int djb2_hash(char *str)
{
  int hash = 5381;
  int c;
  while((c = *str++))
    hash = ((hash<<5) + hash) + c; /*hash*33 + c*/
  return hash;
}
#endif /*MPI_CHOLLA*/

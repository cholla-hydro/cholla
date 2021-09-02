#include "../utils/error_handling.h"
#ifdef   MPI_CHOLLA
#include <mpi.h>
void chexit(int code)
{
  
  if(code==0)
  {
    /*exit normally*/
    MPI_Finalize();
    exit(code);

  }else{

    /*exit with non-zero error code*/
    MPI_Abort(MPI_COMM_WORLD,code);
    exit(code);

  }
}
#else  /*MPI_CHOLLA*/
void chexit(int code)
{
  /*exit using code*/
  exit(code);
}
#endif /*MPI_CHOLLA*/

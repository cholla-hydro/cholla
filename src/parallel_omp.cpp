#ifdef PARALLEL_OMP

#include "parallel_omp.h"

void Get_OMP_Grid_Indxs(  int n_grid_cells, int n_omp_procs, int omp_proc_id, int *omp_gridIndx_start, int *omp_gridIndx_end  ){
  
  int grid_reminder, n_grid_omp, g_start, g_end;
  grid_reminder = n_grid_cells % n_omp_procs;
  n_grid_omp = n_grid_cells / n_omp_procs;
  
  g_start = 0;
  int counter = 0;
  while ( counter < omp_proc_id ){
    g_start += n_grid_omp;
    if ( counter < grid_reminder )  g_start += 1;
    counter += 1;
  }
  g_end = g_start + n_grid_omp;
  if ( omp_proc_id < grid_reminder )  g_end += 1;

  *omp_gridIndx_start = g_start;
  *omp_gridIndx_end = g_end;
  
}








#endif
#ifdef PARALLEL_OMP

  #include "../utils/parallel_omp.h"

void Get_OMP_Grid_Indxs(int n_grid_cells, int n_omp_procs, int omp_proc_id, int *omp_gridIndx_start,
                        int *omp_gridIndx_end)
{
  int grid_reminder, n_grid_omp, g_start, g_end;
  grid_reminder = n_grid_cells % n_omp_procs;
  n_grid_omp    = n_grid_cells / n_omp_procs;

  g_start     = 0;
  int counter = 0;
  while (counter < omp_proc_id) {
    g_start += n_grid_omp;
    if (counter < grid_reminder) g_start += 1;
    counter += 1;
  }
  g_end = g_start + n_grid_omp;
  if (omp_proc_id < grid_reminder) g_end += 1;

  *omp_gridIndx_start = g_start;
  *omp_gridIndx_end   = g_end;
}

  #ifdef PARTICLES
void Get_OMP_Particles_Indxs(part_int_t n_parts_local, int n_omp_procs, int omp_proc_id, part_int_t *omp_pIndx_start,
                             part_int_t *omp_pIndx_end)
{
  part_int_t n_parts_omp, parts_reminder, p_start, p_end;

  parts_reminder = n_parts_local % n_omp_procs;
  n_parts_omp    = n_parts_local / n_omp_procs;

  p_start     = 0;
  int counter = 0;
  while (counter < omp_proc_id) {
    p_start += n_parts_omp;
    if (counter < parts_reminder) p_start += 1;
    counter += 1;
  }
  p_end = p_start + n_parts_omp;
  if (omp_proc_id < parts_reminder) p_end += 1;

  *omp_pIndx_start = p_start;
  *omp_pIndx_end   = p_end;
}
  #endif

#endif

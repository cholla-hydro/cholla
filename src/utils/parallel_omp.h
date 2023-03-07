#ifdef PARALLEL_OMP

  #ifndef PARALLEL_OMP_H
    #define PARALLEL_OMP_H

    #include <omp.h>
    #include <stdio.h>
    #include <stdlib.h>

    #include <iostream>

    #include "../global/global.h"
    #include "math.h"

void Get_OMP_Grid_Indxs(int n_grid_cells, int n_omp_procs, int omp_proc_id, int *omp_gridIndx_start,
                        int *omp_gridIndx_end);

    #ifdef PARTICLES
void Get_OMP_Particles_Indxs(part_int_t n_parts_local, int n_omp_procs, int omp_proc_id, part_int_t *omp_pIndx_start,
                             part_int_t *omp_pIndx_end);
    #endif

  #endif
#endif

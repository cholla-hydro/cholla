#ifdef PARALLEL_OMP

#ifndef PARALLEL_OMP_H
#define PARALLEL_OMP_H

#include<stdio.h>
#include<stdlib.h>
#include"math.h"
#include"global.h"
#include <iostream>
#include <omp.h>

#ifdef PARTICLES
void Get_OMP_Indxs( part_int_t n_parts_local, int n_omp_procs, int omp_proc_id, int nGrid, part_int_t *omp_pIndx_start, part_int_t *omp_pIndx_end, int *omp_gridIndx_start, int *omp_gridIndx_end);
#endif

void Get_OMP_Grid_Indxs(  int n_omp_procs, int omp_proc_id, int nGrid,  int *omp_gridIndx_start, int *omp_gridIndx_end  );
#endif
#endif
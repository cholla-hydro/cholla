#ifdef ANALYSIS

#include <stdio.h>      /* printf */
#include <math.h> 
#include "analysis.h"
#include "../io.h"

#ifdef MPI_CHOLLA
#include "../mpi_routines.h"
#endif

void Grid3D::Compute_Phase_Diagram(){
  
  int n_temp, n_dens;
  Real temp_min, temp_max, dens_min, dens_max;
  Real log_temp_min, log_temp_max, log_dens_min, log_dens_max; 
  Real log_delta_dens, log_delta_temp;
  
  n_dens = Analysis.n_dens;
  n_temp = Analysis.n_temp;
  dens_min = Analysis.dens_min;
  dens_max = Analysis.dens_max;
  temp_min = Analysis.temp_min;
  temp_max = Analysis.temp_max;
  
  log_dens_min = log10( dens_min );
  log_dens_max = log10( dens_max );
  log_temp_min = log10( temp_min );
  log_temp_max = log10( temp_max );
  
  log_delta_dens = ( log_dens_max - log_dens_min ) / n_dens;
  log_delta_temp = ( log_temp_max - log_temp_min ) / n_temp;
  
  
  int nx_local, ny_local, nz_local, n_ghost;
  int nx_grid, ny_grid, nz_grid; 
  nx_local = Analysis.nx_local;
  ny_local = Analysis.ny_local;
  nz_local = Analysis.nz_local;
  n_ghost = Analysis.n_ghost;
  nx_grid = nx_local + 2*n_ghost;
  ny_grid = ny_local + 2*n_ghost;
  nz_grid = nz_local + 2*n_ghost;
  
  
  
  Real dens, log_dens, temp, log_temp;
  int k, j, i, id_grid;
  int indx_dens, indx_temp, indx_phase;
  
  
  //Clear Phase Dikagram
  for (indx_phase=0; indx_phase<n_temp*n_dens; indx_phase++) Analysis.phase_diagram[indx_phase] = 0;
  
  for ( k=0; k<nz_local; k++ ){
    for ( j=0; j<ny_local; j++ ){
      for ( i=0; i<nx_local; i++ ){
          id_grid = (i+n_ghost) + (j+n_ghost)*nx_grid + (k+n_ghost)*nx_grid*ny_grid;
          dens = C.density[id_grid] * Cosmo.rho_0_gas / Cosmo.rho_mean_baryon; // Baryionic overdensity
          // chprintf( "%f %f \n", dens, temp);
          #ifdef COOLING_GRACKLE
          temp = Cool.temperature[id_grid];
          #endif
          if ( dens < dens_min || dens > dens_max || temp < temp_min || temp > temp_max ){
            printf("%f   %f\n", dens, temp );
            continue;
          }
          log_dens = log10(dens);
          log_temp = log10(temp);
          indx_dens = ( log_dens - log_dens_min ) / log_delta_dens;
          indx_temp = ( log_temp - log_temp_min ) / log_delta_temp;
          
          indx_phase = indx_temp + indx_dens*n_temp;
          Analysis.phase_diagram[indx_phase] += 1;
          
      }
    }
  }
  
  // Real phase_sum_local = 0;
  // for (indx_phase=0; indx_phase<n_temp*n_dens; indx_phase++) phase_sum_local += Analysis.phase_diagram[indx_phase];
  // printf(" Phase Diagram Sum Local: %f\n", phase_sum_local );
  
  #ifdef MPI_CHOLLA
  MPI_Reduce( Analysis.phase_diagram, Analysis.phase_diagram_global, n_temp*n_dens,  MPI_FLOAT,  MPI_SUM, 0,  world );
  if ( procID == 0) for (indx_phase=0; indx_phase<n_temp*n_dens; indx_phase++) Analysis.phase_diagram[indx_phase] = Analysis.phase_diagram_global[indx_phase];
  #endif
  
  //Compute the sum for normalization
  Real phase_sum = 0;
  for (indx_phase=0; indx_phase<n_temp*n_dens; indx_phase++) phase_sum += Analysis.phase_diagram[indx_phase];
  chprintf(" Phase Diagram Sum Global: %f\n", phase_sum );
  
  //Normalize the Phase Diagram
  for (indx_phase=0; indx_phase<n_temp*n_dens; indx_phase++) Analysis.phase_diagram[indx_phase] /= phase_sum;
  
  
  
}


void Analysis_Module::Initialize_Phase_Diagram( struct parameters *P ){
  
  //Size of the diagram
  n_dens = 1000;
  n_temp = 1000;
  dens_min = 1e-3;
  dens_max = 1e6;
  temp_min = 1e0;
  temp_max = 1e8;
  
  phase_diagram = (float *) malloc(n_dens*n_temp*sizeof(float));
  
  #ifdef MPI_CHOLLA
  if (procID == 0) phase_diagram_global = (float *) malloc(n_dens*n_temp*sizeof(float));
  #endif
  chprintf(" Phase Diagram Initialized.\n");
  
}













#endif
#ifdef PARTICLES

#include<stdio.h>
#include<stdlib.h>
#include"math.h"
#include <iostream>
#include"../global.h"
#include"../grid3D.h"
#include"particles_3D.h"

#ifdef PARALLEL_OMP
#include"../parallel_omp.h"
#endif

void Grid3D::Get_Gravity_Field_Particles(){

  #ifndef PARALLEL_OMP
  Get_Gravity_Field_Particles_function( 0, Particles.G.nz_local + 2*Particles.G.n_ghost_particles_grid);
  #else

  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();

    Get_OMP_Grid_Indxs( Particles.G.nz_local + 2*Particles.G.n_ghost_particles_grid, N_OMP_THREADS, omp_id,  &g_start, &g_end );

    Get_Gravity_Field_Particles_function( g_start, g_end);  
  }
  #endif
}

void Grid3D::Get_Gravity_Field_Particles_function( int g_start, int g_end ){
  
  int nx_grav, ny_grav, nz_grav, nGHST_grav;
  nGHST_grav = Particles.G.n_ghost_particles_grid;
  nx_grav = Particles.G.nx_local + 2*nGHST_grav;
  ny_grav = Particles.G.ny_local + 2*nGHST_grav;
  nz_grav = Particles.G.nz_local + 2*nGHST_grav;

  int nx_grid, ny_grid, nz_grid, nGHST_grid;
  Real *potential;
  
  #ifdef GRAVITY_COUPLE_CPU
  potential = Grav.F.potential_h;
  nGHST_grid = N_GHOST_POTENTIAL;
  #endif
  #ifdef GRAVITY_COUPLE_GPU
  potential = C.Grav_potential;
  nGHST_grid = H.n_ghost;
  #endif
  
  nx_grid = Grav.nx_local + 2*nGHST_grid;
  ny_grid = Grav.ny_local + 2*nGHST_grid;
  nz_grid = Grav.nz_local + 2*nGHST_grid;

  int nGHST = nGHST_grid - nGHST_grav;
  
  Real dx, dy, dz;
  dx = Particles.G.dx;
  dy = Particles.G.dy;
  dz = Particles.G.dz;
  
  Real phi_l, phi_r;
  int k, j, i, id_l, id_r, id;
  for ( k=g_start; k<g_end; k++ ){
    for ( j=0; j<ny_grav; j++ ){
      for ( i=0; i<nx_grav; i++ ){
        id   = (i) + (j)*nx_grav + (k)*ny_grav*nx_grav;
        id_l = (i-1 + nGHST) + (j + nGHST)*nx_grid + (k + nGHST)*ny_grid*nx_grid;
        id_r = (i+1 + nGHST) + (j + nGHST)*nx_grid + (k + nGHST)*ny_grid*nx_grid;
        phi_l = potential[id_l];
        phi_r = potential[id_r];
        Particles.G.gravity_x[id] = -0.5 * ( phi_r - phi_l ) / dx;
      }
    }
  }

  for ( k=g_start; k<g_end; k++ ){
    for ( j=0; j<ny_grav; j++ ){
      for ( i=0; i<nx_grav; i++ ){
        id   = (i) + (j)*nx_grav + (k)*ny_grav*nx_grav;
        id_l = (i + nGHST) + (j-1 + nGHST)*nx_grid + (k + nGHST)*ny_grid*nx_grid;
        id_r = (i + nGHST) + (j+1 + nGHST)*nx_grid + (k + nGHST)*ny_grid*nx_grid;
        phi_l = potential[id_l];
        phi_r = potential[id_r];
        Particles.G.gravity_y[id] = -0.5 * ( phi_r - phi_l ) / dy;
      }
    }
  }

  for ( k=g_start; k<g_end; k++ ){
    for ( j=0; j<ny_grav; j++ ){
      for ( i=0; i<nx_grav; i++ ){
        id   = (i) + (j)*nx_grav + (k)*ny_grav*nx_grav;
        id_l = (i + nGHST) + (j + nGHST)*nx_grid + (k-1 + nGHST)*ny_grid*nx_grid;
        id_r = (i + nGHST) + (j + nGHST)*nx_grid + (k+1 + nGHST)*ny_grid*nx_grid;
        phi_l = potential[id_l];
        phi_r = potential[id_r];
        Particles.G.gravity_z[id] = -0.5 * ( phi_r - phi_l ) / dz;
      }
    }
  }

}

































#endif//PARTICLES
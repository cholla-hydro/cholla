#ifdef PARTICLES

#include<stdio.h>
#include<stdlib.h>
#include"math.h"
#include <iostream>
#include"../global.h"
#include"../grid3D.h"
#include"particles_3D.h"
#include"density_CIC.h"


#ifdef PARALLEL_OMP
#include"../parallel_omp.h"
#endif

//Get the Gravitational Field from the potential: g=-gradient(potential)
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

//Compute the gradient of the potential
void Grid3D::Get_Gravity_Field_Particles_function( int g_start, int g_end ){
  
  int nx_grav, ny_grav, nz_grav, nGHST_grav;
  nGHST_grav = Particles.G.n_ghost_particles_grid;
  nx_grav = Particles.G.nx_local + 2*nGHST_grav;
  ny_grav = Particles.G.ny_local + 2*nGHST_grav;
  nz_grav = Particles.G.nz_local + 2*nGHST_grav;

  int nx_grid, ny_grid, nz_grid, nGHST_grid;
  Real *potential;
  
  potential = Grav.F.potential_h;
  nGHST_grid = N_GHOST_POTENTIAL;
  
  nx_grid = Grav.nx_local + 2*nGHST_grid;
  ny_grid = Grav.ny_local + 2*nGHST_grid;
  nz_grid = Grav.nz_local + 2*nGHST_grid;

  int nGHST = nGHST_grid - nGHST_grav;
  
  Real dx, dy, dz;
  dx = Particles.G.dx;
  dy = Particles.G.dy;
  dz = Particles.G.dz;
  
  #ifdef GRAVITY_5_POINTS_GRADIENT
  Real phi_ll, phi_rr;
  int id_ll, id_rr;
  #endif  
  
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
        #ifdef GRAVITY_5_POINTS_GRADIENT
        id_ll = (i-2 + nGHST) + (j + nGHST)*nx_grid + (k + nGHST)*ny_grid*nx_grid;
        id_rr = (i+2 + nGHST) + (j + nGHST)*nx_grid + (k + nGHST)*ny_grid*nx_grid;  
        phi_ll = potential[id_ll];
        phi_rr = potential[id_rr];
        Particles.G.gravity_x[id] = -1 * ( -phi_rr + 8*phi_r - 8*phi_l + phi_ll) / (12*dx);
        #else
        Particles.G.gravity_x[id] = -0.5 * ( phi_r - phi_l ) / dx;
        #endif
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
        #ifdef GRAVITY_5_POINTS_GRADIENT
        id_ll = (i + nGHST) + (j-2 + nGHST)*nx_grid + (k + nGHST)*ny_grid*nx_grid;
        id_rr = (i + nGHST) + (j+2 + nGHST)*nx_grid + (k + nGHST)*ny_grid*nx_grid;
        phi_ll = potential[id_ll];
        phi_rr = potential[id_rr];
        Particles.G.gravity_y[id] = -1 * ( -phi_rr + 8*phi_r - 8*phi_l + phi_ll) / (12*dy);
        #else
        Particles.G.gravity_y[id] = -0.5 * ( phi_r - phi_l ) / dy;
        #endif
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
        #ifdef GRAVITY_5_POINTS_GRADIENT
        id_ll = (i + nGHST) + (j + nGHST)*nx_grid + (k-2 + nGHST)*ny_grid*nx_grid;
        id_rr = (i + nGHST) + (j + nGHST)*nx_grid + (k+2 + nGHST)*ny_grid*nx_grid;
        phi_ll = potential[id_ll];
        phi_rr = potential[id_rr];
        Particles.G.gravity_z[id] = -1 * ( -phi_rr + 8*phi_r - 8*phi_l + phi_ll) / (12*dz);
        #else
        Particles.G.gravity_z[id] = -0.5 * ( phi_r - phi_l ) / dz;
        #endif
      }
    }
  }

}

//Get the CIC interpolation of the Gravitational field at the particles positions
void Grid3D::Get_Gravity_CIC(){
  
  #ifndef PARALLEL_OMP
  Get_Gravity_CIC_function( 0, Particles.n_local );
  #else
  
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    
    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );
    
    Get_Gravity_CIC_function( p_start, p_end );
  }
  #endif
}

//Get the CIC interpolation of the Gravitational field at the particles positions
void Grid3D::Get_Gravity_CIC_function( part_int_t p_start, part_int_t p_end ){
  
  int nx_g, ny_g, nz_g, nGHST;
  nGHST = Particles.G.n_ghost_particles_grid;
  nx_g = Particles.G.nx_local + 2*nGHST;
  ny_g = Particles.G.ny_local + 2*nGHST;
  nz_g = Particles.G.nz_local + 2*nGHST;

  Real xMin, yMin, zMin, dx, dy, dz;
  xMin = Particles.G.xMin;
  yMin = Particles.G.yMin;
  zMin = Particles.G.zMin;
  dx = Particles.G.dx;
  dy = Particles.G.dy;
  dz = Particles.G.dz;

  part_int_t pIndx;
  int indx_x, indx_y, indx_z, indx;
  Real x_pos, y_pos, z_pos;
  Real cell_center_x, cell_center_y, cell_center_z;
  Real delta_x, delta_y, delta_z;
  Real g_x_bl, g_x_br, g_x_bu, g_x_bru, g_x_tl, g_x_tr, g_x_tu, g_x_tru;
  Real g_y_bl, g_y_br, g_y_bu, g_y_bru, g_y_tl, g_y_tr, g_y_tu, g_y_tru;
  Real g_z_bl, g_z_br, g_z_bu, g_z_bru, g_z_tl, g_z_tr, g_z_tu, g_z_tru;
  Real g_x, g_y, g_z;
  bool ignore, in_local;
  for ( pIndx=p_start; pIndx < p_end; pIndx++ ){
    ignore = false;
    in_local = true;
    
    x_pos = Particles.pos_x[pIndx];
    y_pos = Particles.pos_y[pIndx];
    z_pos = Particles.pos_z[pIndx];
    Get_Indexes_CIC( xMin, yMin, zMin, dx, dy, dz, x_pos, y_pos, z_pos, indx_x, indx_y, indx_z );
    if ( indx_x < -1 ) ignore = true;
    if ( indx_y < -1 ) ignore = true;
    if ( indx_z < -1 ) ignore = true;
    if ( indx_x > nx_g-3  ) ignore = true;
    if ( indx_y > ny_g-3  ) ignore = true;
    if ( indx_y > nz_g-3  ) ignore = true;
    if ( x_pos < Particles.G.xMin || x_pos >= Particles.G.xMax ) in_local = false;
    if ( y_pos < Particles.G.yMin || y_pos >= Particles.G.yMax ) in_local = false;
    if ( z_pos < Particles.G.zMin || z_pos >= Particles.G.zMax ) in_local = false;
    if ( ! in_local  ) {
      std::cout << " Gravity CIC Error:" << std::endl;
      #ifdef PARTICLE_IDS
      std::cout << " Particle outside Loacal  domain    pID: " << pID << std::endl;
      #else
      std::cout << " Particle outside Loacal  domain " << std::endl;
      #endif
      std::cout << "  Domain X: " << Particles.G.xMin <<  "  " << Particles.G.xMax << std::endl;
      std::cout << "  Domain Y: " << Particles.G.yMin <<  "  " << Particles.G.yMax << std::endl;
      std::cout << "  Domain Z: " << Particles.G.zMin <<  "  " << Particles.G.zMax << std::endl;
      std::cout << "  Particle X: " << x_pos << std::endl;
      std::cout << "  Particle Y: " << y_pos << std::endl;
      std::cout << "  Particle Z: " << z_pos << std::endl;
      // Particles.grav_x[pIndx] = 0;
      // Particles.grav_y[pIndx] = 0;
      // Particles.grav_z[pIndx] = 0;
      continue;
    }
    if ( ignore ){
      #ifdef PARTICLE_IDS
      std::cout << "ERROR GRAVITY_CIC Index    pID: " << Particles.partIDs[pIndx] << std::endl;
      #else
      std::cout << "ERROR GRAVITY_CIC Index " << std::endl;
      #endif
      std::cout << "Negative xIndx: " << x_pos << "  " << indx_x << std::endl;
      std::cout << "Negative zIndx: " << z_pos << "  " << indx_z << std::endl;
      std::cout << "Negative yIndx: " << y_pos << "  " << indx_y << std::endl;
      std::cout << "Excess xIndx: " << x_pos << "  " << indx_x << std::endl;
      std::cout << "Excess yIndx: " << y_pos << "  " << indx_y << std::endl;
      std::cout << "Excess zIndx: " << z_pos << "  " << indx_z << std::endl;
      std::cout << std::endl;
      continue;
    }
    
    cell_center_x = xMin + indx_x*dx + 0.5*dx;
    cell_center_y = yMin + indx_y*dy + 0.5*dy;
    cell_center_z = zMin + indx_z*dz + 0.5*dz;
    delta_x = 1 - ( x_pos - cell_center_x ) / dx;
    delta_y = 1 - ( y_pos - cell_center_y ) / dy;
    delta_z = 1 - ( z_pos - cell_center_z ) / dz;
    indx_x += nGHST;
    indx_y += nGHST;
    indx_z += nGHST;

    indx = indx_x + indx_y*nx_g + indx_z*nx_g*ny_g;
    g_x_bl = Particles.G.gravity_x[indx];
    g_y_bl = Particles.G.gravity_y[indx];
    g_z_bl = Particles.G.gravity_z[indx];

    indx = (indx_x+1) + (indx_y)*nx_g + (indx_z)*nx_g*ny_g;
    g_x_br = Particles.G.gravity_x[indx];
    g_y_br = Particles.G.gravity_y[indx];
    g_z_br = Particles.G.gravity_z[indx];

    indx = (indx_x) + (indx_y+1)*nx_g + (indx_z)*nx_g*ny_g;
    g_x_bu = Particles.G.gravity_x[indx];
    g_y_bu = Particles.G.gravity_y[indx];
    g_z_bu = Particles.G.gravity_z[indx];

    indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z)*nx_g*ny_g;
    g_x_bru = Particles.G.gravity_x[indx];
    g_y_bru = Particles.G.gravity_y[indx];
    g_z_bru = Particles.G.gravity_z[indx];

    indx = (indx_x) + (indx_y)*nx_g + (indx_z+1)*nx_g*ny_g;
    g_x_tl = Particles.G.gravity_x[indx];
    g_y_tl = Particles.G.gravity_y[indx];
    g_z_tl = Particles.G.gravity_z[indx];

    indx = (indx_x+1) + (indx_y)*nx_g + (indx_z+1)*nx_g*ny_g;
    g_x_tr = Particles.G.gravity_x[indx];
    g_y_tr = Particles.G.gravity_y[indx];
    g_z_tr = Particles.G.gravity_z[indx];

    indx = (indx_x) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    g_x_tu = Particles.G.gravity_x[indx];
    g_y_tu = Particles.G.gravity_y[indx];
    g_z_tu = Particles.G.gravity_z[indx];

    indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    g_x_tru = Particles.G.gravity_x[indx];
    g_y_tru = Particles.G.gravity_y[indx];
    g_z_tru = Particles.G.gravity_z[indx];

    g_x = g_x_bl*(delta_x)*(delta_y)*(delta_z)     + g_x_br*(1-delta_x)*(delta_y)*(delta_z) +
          g_x_bu*(delta_x)*(1-delta_y)*(delta_z  ) + g_x_bru*(1-delta_x)*(1-delta_y)*(delta_z) +
          g_x_tl*(delta_x)*(delta_y)*(1-delta_z)   + g_x_tr*(1-delta_x)*(delta_y)*(1-delta_z) +
          g_x_tu*(delta_x)*(1-delta_y)*(1-delta_z) + g_x_tru*(1-delta_x)*(1-delta_y)*(1-delta_z);

    g_y = g_y_bl*(delta_x)*(delta_y)*(delta_z)     + g_y_br*(1-delta_x)*(delta_y)*(delta_z) +
          g_y_bu*(delta_x)*(1-delta_y)*(delta_z)   + g_y_bru*(1-delta_x)*(1-delta_y)*(delta_z) +
          g_y_tl*(delta_x)*(delta_y)*(1-delta_z)   + g_y_tr*(1-delta_x)*(delta_y)*(1-delta_z) +
          g_y_tu*(delta_x)*(1-delta_y)*(1-delta_z) + g_y_tru*(1-delta_x)*(1-delta_y)*(1-delta_z);

    g_z = g_z_bl*(delta_x)*(delta_y)*(delta_z)     + g_z_br*(1-delta_x)*(delta_y)*(delta_z) +
          g_z_bu*(delta_x)*(1-delta_y)*(delta_z)   + g_z_bru*(1-delta_x)*(1-delta_y)*(delta_z) +
          g_z_tl*(delta_x)*(delta_y)*(1-delta_z)   + g_z_tr*(1-delta_x)*(delta_y)*(1-delta_z) +
          g_z_tu*(delta_x)*(1-delta_y)*(1-delta_z) + g_z_tru*(1-delta_x)*(1-delta_y)*(1-delta_z);

    Particles.grav_x[pIndx] = g_x;
    Particles.grav_y[pIndx] = g_y;
    Particles.grav_z[pIndx] = g_z;
  }
}


































#endif//PARTICLES
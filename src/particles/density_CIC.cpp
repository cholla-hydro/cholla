#ifdef PARTICLES

#include<stdio.h>
#include<stdlib.h>
#include"math.h"
#include <iostream>
#include"../global.h"
#include"particles_3D.h"
#include "../grid3D.h"

#ifdef PARALLEL_OMP
#include"../parallel_omp.h"
#endif

void Grid3D::Copy_Particles_Density_to_Gravity(){
  
  // Step 1: Get Partcles Density
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif
  
  Particles.Clear_Density();
  
  #ifdef PARALLEL_OMP
  Particles.Get_Density_CIC_OMP();
  #else
  Particles.Get_Density_CIC_Serial();
  #endif
  
  #ifdef CPU_TIME
  Timer.End_and_Record_Time( 4 );
  #endif
  
  // Step 2: Transfer Particles CIC density Boundaries
  
  
  
  //Step 3: Copy Particles density to Gravity array
  
  
  
}


void::Particles_3D::Clear_Density(){
  for( int i=0; i<G.n_cells; i++ ) G.density[i] = 0;  
}

void Particles_3D::Get_Density_CIC(){
  
  
}

void Get_Indexes_CIC( Real xMin, Real yMin, Real zMin, Real dx, Real dy, Real dz, Real pos_x, Real pos_y, Real pos_z, int &indx_x, int &indx_y, int &indx_z ){
  indx_x = (int) floor( ( pos_x - xMin - 0.5*dx ) / dx );
  indx_y = (int) floor( ( pos_y - yMin - 0.5*dy ) / dy );
  indx_z = (int) floor( ( pos_z - zMin - 0.5*dz ) / dz );
}


void Particles_3D::Get_Density_CIC_Serial( ){
  int nGHST = G.n_ghost_particles_grid;
  int nx_g = G.nx_local + 2*nGHST;
  int ny_g = G.ny_local + 2*nGHST;
  int nz_g = G.nz_local + 2*nGHST;

  Real xMin, yMin, zMin, dx, dy, dz;
  xMin = G.xMin;
  yMin = G.yMin;
  zMin = G.zMin;
  dx = G.dx;
  dy = G.dy;
  dz = G.dz;

  part_int_t pIndx;
  int indx_x, indx_y, indx_z, indx;
  Real pMass, x_pos, y_pos, z_pos;

  Real cell_center_x, cell_center_y, cell_center_z;
  Real delta_x, delta_y, delta_z;
  Real dV_inv = 1./(G.dx*G.dy*G.dz);
  bool ignore;

  for ( pIndx=0; pIndx < n_local; pIndx++ ){
    ignore = false;

    #ifdef SINGLE_PARTICLE_MASS
    pMass = particle_mass * dV_inv;
    #else
    pMass = mass[pIndx] * dV_inv;
    #endif
    x_pos = pos_x[pIndx];
    y_pos = pos_y[pIndx];
    z_pos = pos_z[pIndx];
    Get_Indexes_CIC( xMin, yMin, zMin, dx, dy, dz, x_pos, y_pos, z_pos, indx_x, indx_y, indx_z );
    if ( indx_x < -1 ) ignore = true;
    if ( indx_y < -1 ) ignore = true;
    if ( indx_z < -1 ) ignore = true;
    if ( indx_x > nx_g-3  ) ignore = true;
    if ( indx_y > ny_g-3  ) ignore = true;
    if ( indx_y > nz_g-3  ) ignore = true;
    if ( ignore ){
      #ifdef PARTICLE_IDS
      std::cout << "ERROR CIC Index    pID: " << partIDs[pIndx] << std::endl;
      #else
      std::cout << "ERROR CIC Index " << std::endl;
      #endif
      std::cout << "Negative xIndx: " << x_pos << "  " << indx_x << std::endl;
      std::cout << "Negative zIndx: " << z_pos << "  " << indx_z << std::endl;
      std::cout << "Negative yIndx: " << y_pos << "  " << indx_y << std::endl;
      std::cout << "Excess xIndx: " << x_pos << "  " << indx_x << std::endl;
      std::cout << "Excess yIndx: " << y_pos << "  " << indx_y << std::endl;
      std::cout << "Excess zIndx: " << z_pos << "  " << indx_z << std::endl;
      std::cout << std::endl;
      // exit(-1);
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
    G.density[indx] += pMass  * delta_x * delta_y * delta_z;

    indx = (indx_x+1) + indx_y*nx_g + indx_z*nx_g*ny_g;
    G.density[indx] += pMass  * (1-delta_x) * delta_y * delta_z;

    indx = indx_x + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
    G.density[indx] += pMass  * delta_x * (1-delta_y) * delta_z;

    indx = indx_x + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
    G.density[indx] += pMass  * delta_x * delta_y * (1-delta_z);

    indx = (indx_x+1) + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
    G.density[indx] += pMass  * (1-delta_x) * (1-delta_y) * delta_z;

    indx = (indx_x+1) + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
    G.density[indx] += pMass  * (1-delta_x) * delta_y * (1-delta_z);

    indx = indx_x + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    G.density[indx] += pMass  * delta_x * (1-delta_y) * (1-delta_z);

    indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    G.density[indx] += pMass * (1-delta_x) * (1-delta_y) * (1-delta_z);
  }
}



#ifdef PARALLEL_OMP
void Particles_3D::Get_Density_CIC_OMP( ){



  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id;
    int g_start, g_end;
    int n_omp_procs;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();

    int nGHST = G.n_ghost_particles_grid;
    int nx_g = G.nx_local + 2*nGHST;
    int ny_g = G.ny_local + 2*nGHST;
    int nz_g = G.nz_local + 2*nGHST;

    Real xMin, yMin, zMin, dx, dy, dz;
    xMin = G.xMin;
    yMin = G.yMin;
    zMin = G.zMin;
    dx = G.dx;
    dy = G.dy;
    dz = G.dz;
    Real dV_inv = 1./(G.dx*G.dy*G.dz);


    Get_OMP_Grid_Indxs( nz_g, n_omp_procs, omp_id,  &g_start, &g_end );

    part_int_t pIndx;
    int indx_x, indx_y, indx_z, indx;
    Real pMass, x_pos, y_pos, z_pos;

    Real cell_center_x, cell_center_y, cell_center_z;
    Real delta_x, delta_y, delta_z;
    bool ignore;
    bool add_1, add_2;

    for ( pIndx=0; pIndx < n_local; pIndx++ ){
      add_1 = false;
      add_2 = false;

      z_pos = pos_z[pIndx];
      indx_z = (int) floor( ( z_pos - zMin - 0.5*dz ) / dz );
      indx_z += nGHST;
      if ( (indx_z >= g_start) && (indx_z < g_end) ) add_1 = true;
      if ( ((indx_z+1) >= g_start) && ((indx_z+1) < g_end) ) add_2 = true;
      if (!( add_1 || add_2) ) continue;

      ignore = false;
      x_pos = pos_x[pIndx];
      y_pos = pos_y[pIndx];

      indx_x = (int) floor( ( x_pos - xMin - 0.5*dx ) / dx );
      indx_y = (int) floor( ( y_pos - yMin - 0.5*dy ) / dy );
      indx_z -= nGHST;

      if ( indx_x < -1 ) ignore = true;
      if ( indx_y < -1 ) ignore = true;
      if ( indx_z < -1 ) ignore = true;
      if ( indx_x > nx_g-3  ) ignore = true;
      if ( indx_y > ny_g-3  ) ignore = true;
      if ( indx_y > nz_g-3  ) ignore = true;
      if ( ignore ){
        #ifdef PARTICLE_IDS
        std::cout << "ERROR CIC Index    pID: " << partIDs[pIndx] << std::endl;
        #else
        std::cout << "ERROR CIC Index " << std::endl;
        #endif
        std::cout << "Negative xIndx: " << x_pos << "  " << indx_x << std::endl;
        std::cout << "Negative zIndx: " << z_pos << "  " << indx_z << std::endl;
        std::cout << "Negative yIndx: " << y_pos << "  " << indx_y << std::endl;
        std::cout << "Excess xIndx: " << x_pos << "  " << indx_x << std::endl;
        std::cout << "Excess yIndx: " << y_pos << "  " << indx_y << std::endl;
        std::cout << "Excess zIndx: " << z_pos << "  " << indx_z << std::endl;
        std::cout << std::endl;
        // exit(-1);
        continue;
      }

      #ifdef SINGLE_PARTICLE_MASS
      pMass = particle_mass * dV_inv;
      #else
      pMass = mass[pIndx] * dV_inv;
      #endif

      cell_center_x = xMin + indx_x*dx + 0.5*dx;
      cell_center_y = yMin + indx_y*dy + 0.5*dy;
      cell_center_z = zMin + indx_z*dz + 0.5*dz;
      delta_x = 1 - ( x_pos - cell_center_x ) / dx;
      delta_y = 1 - ( y_pos - cell_center_y ) / dy;
      delta_z = 1 - ( z_pos - cell_center_z ) / dz;
      indx_x += nGHST;
      indx_y += nGHST;
      indx_z += nGHST;

      if ( add_1 ){
        indx = indx_x + indx_y*nx_g + indx_z*nx_g*ny_g;
        G.density[indx] += pMass  * delta_x * delta_y * delta_z;

        indx = (indx_x+1) + indx_y*nx_g + indx_z*nx_g*ny_g;
        G.density[indx] += pMass  * (1-delta_x) * delta_y * delta_z;

        indx = indx_x + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
        G.density[indx] += pMass  * delta_x * (1-delta_y) * delta_z;

        indx = (indx_x+1) + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
        G.density[indx] += pMass  * (1-delta_x) * (1-delta_y) * delta_z;
      }

      if ( add_2 ){
        indx = indx_x + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
        G.density[indx] += pMass  * delta_x * delta_y * (1-delta_z);

        indx = (indx_x+1) + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
        G.density[indx] += pMass  * (1-delta_x) * delta_y * (1-delta_z);

        indx = indx_x + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
        G.density[indx] += pMass  * delta_x * (1-delta_y) * (1-delta_z);

        indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
        G.density[indx] += pMass * (1-delta_x) * (1-delta_y) * (1-delta_z);
      }
    }
  }
}

#endif















#endif
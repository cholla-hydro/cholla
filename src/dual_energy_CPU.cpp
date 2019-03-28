#ifdef DE

#include"global.h"
#include"grid3D.h"
#include<stdio.h>
#include<stdlib.h>
#include"math.h"
#include <iostream>

#ifdef PARALLEL_OMP
#include"parallel_omp.h"
#endif

#ifdef MPI_CHOLLA
#include"mpi_routines.h"
#endif


int Grid3D::Select_Internal_Energy_From_DE( Real E, Real U_total, Real U_advected ){

  Real eta = DE_LIMIT;
  
  if( U_total / E > eta ) return 0;
  else return 1;  
}

void Grid3D::Sync_Energies_3D_CPU(){
  #ifndef PARALLEL_OMP
  Sync_Energies_3D_CPU_function( 0, H.nz_real );
  #ifdef TEMPERATURE_FLOOR
  Apply_Temperature_Floor_CPU_function( 0, H.nz_real );
  #endif
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs( H.nz_real, n_omp_procs, omp_id, &g_start, &g_end  );

    Sync_Energies_3D_CPU_function( g_start, g_end );
    #ifdef TEMPERATURE_FLOOR
    #pragma omp barrier
    Apply_Temperature_Floor_CPU_function( g_start, g_end );
    #endif
  }
  #endif
  
  
}

void Grid3D::Sync_Energies_3D_CPU_function( int g_start, int g_end ){
  
  int nx_grid, ny_grid, nz_grid, nGHST_grid;
  nGHST_grid = H.n_ghost;
  nx_grid = H.nx;
  ny_grid = H.ny;
  nz_grid = H.nz;

  int nx, ny, nz;
  nx = H.nx_real;
  ny = H.ny_real;
  nz = H.nz_real;

  int nGHST = nGHST_grid ;
  Real d, d_inv, vx, vy, vz, E, Ek, ge_total, ge_advected, Emax;
  int k, j, i, id;
  int k_g, j_g, i_g;
  
  Real eta = DE_LIMIT;


  int imo, ipo, jmo, jpo, kmo, kpo;
  for ( k_g=g_start; k_g<g_end; k_g++ ){
    for ( j_g=0; j_g<ny; j_g++ ){
      for ( i_g=0; i_g<nx; i_g++ ){

        i = i_g + nGHST;
        j = j_g + nGHST;
        k = k_g + nGHST;

        id  = (i) + (j)*nx_grid + (k)*ny_grid*nx_grid;
        imo = (i-1) + (j)*nx_grid + (k)*ny_grid*nx_grid;
        ipo = (i+1) + (j)*nx_grid + (k)*ny_grid*nx_grid;
        jmo = (i) + (j-1)*nx_grid + (k)*ny_grid*nx_grid;
        jpo = (i) + (j+1)*nx_grid + (k)*ny_grid*nx_grid;
        kmo = (i) + (j)*nx_grid + (k-1)*ny_grid*nx_grid;
        kpo = (i) + (j)*nx_grid + (k+1)*ny_grid*nx_grid;

        d = C.density[id];
        d_inv = 1/d;
        vx = C.momentum_x[id] * d_inv;
        vy = C.momentum_y[id] * d_inv;
        vz = C.momentum_z[id] * d_inv;
        E = C.Energy[id];

        // if (E < 0.0 || E != E) continue; // BUG: This leads to negative Energy
        Ek = 0.5*d*(vx*vx + vy*vy + vz*vz);
        ge_total = E - Ek;
        ge_advected = C.GasEnergy[id];

        //Dont Change Internal energies based on first condition, 
        //This condition is used only to compute pressure and Intenal energy for cooling step
        // #ifdef LIMIT_DE_EKINETIC
        // if (ge2 > 0.0 && E > 0.0 && ge2/E > eta && Ek/H.Ekin_avrg > 0.4 ){
        // #else
        // if (ge2 > 0.0 && E > 0.0 && ge2/E > eta ) {
        // #endif          
        //   C.GasEnergy[id] = ge2;
        //   ge1 = ge2;
        // }
        
        //Syncronize advected internal energy with total internal energy when using total internal energy for dynamical purposes 
        if (ge_total > 0.0 && E > 0.0 && ge_total/E > eta ) C.GasEnergy[id] = ge_total;   

        //Syncronize advected internal energy with total internal energy when using total internal energy based on local maxEnergy condition
        //find the max nearby total energy
        Emax = E;
        Emax = std::max(C.Energy[imo], E);
        Emax = std::max(Emax, C.Energy[ipo]);
        Emax = std::max(Emax, C.Energy[jmo]);
        Emax = std::max(Emax, C.Energy[jpo]);
        Emax = std::max(Emax, C.Energy[kmo]);
        Emax = std::max(Emax, C.Energy[kpo]);
        if (ge_total/Emax > 0.1 && ge_total > 0.0 && Emax > 0.0) {
          C.GasEnergy[id] = ge_total;
        }
        
        //Dont Change total energy  
        // // sync the total energy with the internal energy
        // else {
        //   if (ge1 > 0.0) C.Energy[id] += ge1 - ge2;
        //   else C.GasEnergy[id] = ge2;
        // }
      }
    }
  }
}

#ifdef TEMPERATURE_FLOOR
void Grid3D::Apply_Temperature_Floor_CPU_function( int g_start, int g_end ){

  Real U_floor = H.temperature_floor / (gama - 1) / MP * KB * 1e-10;

  #ifdef COSMOLOGY
  U_floor /=  Cosmo.v_0_gas * Cosmo.v_0_gas / Cosmo.current_a / Cosmo.current_a;
  #endif

  int nx_grid, ny_grid, nz_grid, nGHST_grid;
  nGHST_grid = H.n_ghost;
  nx_grid = H.nx;
  ny_grid = H.ny;
  nz_grid = H.nz;

  int nx, ny, nz;
  nx = H.nx_real;
  ny = H.ny_real;
  nz = H.nz_real;

  int nGHST = nGHST_grid ;
  Real d, vx, vy, vz, Ekin, E, U, GE;
  int k, j, i, id;
  for ( k=g_start; k<g_end; k++ ){
    for ( j=0; j<ny; j++ ){
      for ( i=0; i<nx; i++ ){
        id  = (i+nGHST) + (j+nGHST)*nx_grid + (k+nGHST)*ny_grid*nx_grid;

        d = C.density[id];
        vx = C.momentum_x[id] / d;
        vy = C.momentum_y[id] / d;
        vz = C.momentum_z[id] / d;
        Ekin = 0.5 * d * (vx*vx + vy*vy + vz*vz);
        E = C.Energy[id];
        
        U = ( E - Ekin ) / d;
        if ( U < U_floor ) C.Energy[id] = Ekin + d*U_floor;
        
        #ifdef DE
        GE = C.GasEnergy[id];
        U = GE / d;
        if ( U < U_floor ) C.GasEnergy[id] = d*U_floor;
        #endif
      }
    }
  }
}
#endif //TEMPERATURE_FLOOR


Real Grid3D::Get_Average_Kinetic_Energy_function( int g_start, int g_end ){

  int nx_grid, ny_grid, nz_grid, nGHST;
  nGHST = H.n_ghost;
  nx_grid = H.nx;
  ny_grid = H.ny;
  nz_grid = H.nz;

  int nx, ny, nz;
  nx = H.nx_real;
  ny = H.ny_real;
  nz = H.nz_real;

  Real Ek_sum = 0;
  Real d, d_inv, vx, vy, vz, E, Ek;

  int k, j, i, id;
  for ( k=g_start; k<g_end; k++ ){
    for ( j=0; j<ny; j++ ){
      for ( i=0; i<nx; i++ ){

        id  = (i+nGHST) + (j+nGHST)*nx_grid + (k+nGHST)*ny_grid*nx_grid;

        d = C.density[id];
        d_inv = 1/d;
        vx = C.momentum_x[id] * d_inv;
        vy = C.momentum_y[id] * d_inv;
        vz = C.momentum_z[id] * d_inv;
        E = C.Energy[id];
        Ek = 0.5*d*(vx*vx + vy*vy + vz*vz);
        Ek_sum += Ek;
      }
    }
  }
  return Ek_sum;
}

void Grid3D::Get_Average_Kinetic_Energy(){
  Real Ek_sum;

  #ifndef PARALLEL_OMP
  Ek_sum = Get_Average_Kinetic_Energy_function(  0, H.nz_real );
  #else
  Ek_sum = 0;
  Real Ek_sum_all[N_OMP_THREADS];
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs( H.nz_real, n_omp_procs, omp_id,  &g_start, &g_end  );
    Ek_sum_all[omp_id] = Get_Average_Kinetic_Energy_function(  g_start, g_end );

  }
  for ( int i=0; i<N_OMP_THREADS; i++ ){
    Ek_sum += Ek_sum_all[i];
  }
  #endif
  
  #ifdef MPI_CHOLLA
  Ek_sum /=  ( H.nx_real * H.ny_real * H.nz_real);
  H.Ekin_avrg = ReduceRealAvg(Ek_sum);
  #else
  H.Ekin_avrg = Ek_sum / ( H.nx_real * H.ny_real * H.nz_real);
  #endif
  
  
  
  
}








#endif
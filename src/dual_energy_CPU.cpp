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
  Real d, d_inv, vx, vy, vz, E, Ek, ge_total, ge_advected, Emax, U;
  int k, j, i, id;
  int k_g, j_g, i_g;
  int flag_DE;
  
  Real eta = DE_LIMIT;
  Real Beta_DE = BETA_DUAL_ENERGY;

  Real v_l, v_r, delta_vx, delta_vy, delta_vz, delta_v2, ge_trunc;


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

        Ek = 0.5*d*(vx*vx + vy*vy + vz*vz);
        ge_total = E - Ek;
        ge_advected = C.GasEnergy[id];

        //New Dual Enegy Condition from Teyssier 2015
        //Get delta velocity
        
        //X direcction
        v_l = C.momentum_x[imo] / C.density[imo];
        v_r = C.momentum_x[ipo] / C.density[ipo];
        delta_vx = v_r - v_l;
        
        //Y direcction
        v_l = C.momentum_y[jmo] / C.density[jmo];
        v_r = C.momentum_y[jpo] / C.density[jpo];
        delta_vy = v_r - v_l;
        
        //Z direcction
        v_l = C.momentum_z[kmo] / C.density[kmo];
        v_r = C.momentum_z[kpo] / C.density[kpo];
        delta_vz = v_r - v_l;
        
        delta_v2 = delta_vx*delta_vx + delta_vy*delta_vy + delta_vz*delta_vz; 
        
        //Get the truncation error (Teyssier 2015)
        ge_trunc = 0.5 * d * delta_v2;
                
        if (ge_total > 0.0 && E > 0.0 && ge_total/E > eta && ge_total > 0.5*ge_advected && ( ge_total > Beta_DE * ge_trunc ) ){
          U = ge_total;
          flag_DE = 0;
        }            
        else{
          U = ge_advected;
          flag_DE = 1;
        }
        
        // //Syncronize advected internal energy with total internal energy when using total internal energy based on local maxEnergy condition
        // //find the max nearby total energy
        // Emax = E;
        // Emax = std::max(C.Energy[imo], E);
        // Emax = std::max(Emax, C.Energy[ipo]);
        // Emax = std::max(Emax, C.Energy[jmo]);
        // Emax = std::max(Emax, C.Energy[jpo]);
        // Emax = std::max(Emax, C.Energy[kmo]);
        // Emax = std::max(Emax, C.Energy[kpo]);
        // if (ge_total/Emax > 0.1 && ge_total > 0.0 && Emax > 0.0){
        //   U = ge_total;
        //   flag_DE = 0;
        // }
        
        //Set the Internal Energy
        C.Energy[id] = Ek + U;
        C.GasEnergy[id] = U;
        
        //Set the flag for which internal energy was used
        #ifdef COOLING_GRACKLE
        Cool.flags_DE[id] = flag_DE;
        #endif
                 
      }
    }
  }
}

#ifdef TEMPERATURE_FLOOR
void Grid3D::Apply_Temperature_Floor_CPU_function( int g_start, int g_end ){

  Real temp_floor = H.temperature_floor;
  
  #ifdef COOLING_GRACKLE
  if ( Cosmo.current_a > Cool.scale_factor_UVB_on ) temp_floor = 3e2;
  #endif 
  
  Real U_floor = temp_floor / (gama - 1) / MP * KB * 1e-10;
  
  

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
  
  Real U_floor_local, mu;

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
        
        #ifdef COOLING_GRACKLE
        mu = Cool.Get_Mean_Molecular_Weight( id );
        U_floor_local = U_floor / mu ;
        #else
        U_floor_local = U_floor;
        #endif
        
        U = ( E - Ekin ) / d;
        if ( U < U_floor_local ) C.Energy[id] = Ekin + d*U_floor_local;
        
        #ifdef DE
        GE = C.GasEnergy[id];
        U = GE / d;
        if ( U < U_floor_local ) C.GasEnergy[id] = d*U_floor_local;
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
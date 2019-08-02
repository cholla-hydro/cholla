#ifdef DE

#include"global.h"
#include"grid3D.h"
#include<stdio.h>
#include<stdlib.h>
#include"math.h"
#include <iostream>
#include"io.h"
#include<stdarg.h>
#include<string.h>
#include <iostream>
#include <fstream>
#include<ctime>

using namespace std;

#ifdef PARALLEL_OMP
#include"parallel_omp.h"
#endif

#ifdef MPI_CHOLLA
#include"mpi_routines.h"
#endif


#ifdef JEANS_CONDITION
void Grid3D::Apply_Jeans_Length_Condition( ){

  #ifndef PARALLEL_OMP
  Apply_Jeans_Length_Condition_function( 0, H.nz_real );
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs( H.nz_real, n_omp_procs, omp_id, &g_start, &g_end  );

    Apply_Jeans_Length_Condition_function( g_start, g_end );
    
  }
  #endif
    
}



void Grid3D::Apply_Jeans_Length_Condition_function( int g_start, int g_end ){
  
  
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
  Real d, d_inv, vx, vy, vz, E, Ek, p;
  int k, j, i, id;
  int k_g, j_g, i_g;
  
  Real dens_conv, dx, G, pressure_conv, p_jeans, K, rho, mu;
  // dens_conv = Cosmo.rho_0_gas / Cosmo.current_a / Cosmo.current_a / Cosmo.current_;
  dens_conv = Cosmo.rho_0_gas;
  G = G_COSMO;  //G_COSMO
  dx = fmax( H.dx, H.dy );
  dx = fmax( H.dz, dx );
  // dx *= Cosmo.current_a;
  
  mu = 1.22;
  K = 100;
  
  for ( k_g=g_start; k_g<g_end; k_g++ ){
    for ( j_g=0; j_g<ny; j_g++ ){
      for ( i_g=0; i_g<nx; i_g++ ){

        i = i_g + nGHST;
        j = j_g + nGHST;
        k = k_g + nGHST;
        id  = (i) + (j)*nx_grid + (k)*ny_grid*nx_grid;        
        
        d = C.density[id];
        d_inv = 1/d;
        vx = C.momentum_x[id] * d_inv;
        vy = C.momentum_y[id] * d_inv;
        vz = C.momentum_z[id] * d_inv;
        E = C.Energy[id];
        Ek = 0.5*d*(vx*vx + vy*vy + vz*vz);
        // ge_advected = C.GasEnergy[id];
        // ge_total = E - Ek;
        p = (gama - 1) * ( E - Ek );
        
        //Convert to physical units
        rho = d * Cosmo.rho_0_gas; 
        p_jeans = K * G * rho * rho * dx * dx / mu;
        p_jeans *= 1. / Cosmo.rho_0_gas / Cosmo.v_0_gas / Cosmo.v_0_gas; //Convert to internal code units; 
        p_jeans *= Cosmo.current_a * Cosmo.current_a; //Convert to comuving internal units
        // chprintf( "%f \n",  p/p_jeans );
        
        // if ( p_jeans > p ) Cool.flags_DE[id] = 2; 
        
      }
    }
  }
  
}
#endif


int Grid3D::Select_Internal_Energy_From_DE( Real E, Real U_total, Real U_advected ){

  Real eta = DE_ETA_1;
  
  if( U_total / E > eta ) return 0;
  else return 1;  
}

void Grid3D::Sync_Energies_3D_CPU(){
  #ifndef PARALLEL_OMP
  Sync_Energies_3D_CPU_function( 0, H.nz_real );
  Update_Total_Energy_After_Dual_Energy_function( 0, H.nz_real );
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
    
    #pragma omp barrier
    Update_Total_Energy_After_Dual_Energy_function( g_start, g_end );
    
    #ifdef TEMPERATURE_FLOOR
    #pragma omp barrier
    Apply_Temperature_Floor_CPU_function( g_start, g_end );
    #endif
  }
  #endif
  
  
}

Real Grid3D::Get_Pressure_From_Energy( int indx ){
  
  Real d, d_inv, vx, vy, vz, E, Ek, p;
  d = C.density[indx];
  d_inv = 1/d;
  vx = C.momentum_x[indx] * d_inv;
  vy = C.momentum_y[indx] * d_inv;
  vz = C.momentum_z[indx] * d_inv;
  E = C.Energy[indx];
  Ek = 0.5*d*(vx*vx + vy*vy + vz*vz);
  p = (gama - 1) * ( E - Ek );
  return p;  
}

bool Get_Pressure_Jump( Real gamma, Real rho_l, Real rho_r, Real p_l, Real p_r ){
  bool pressure_jump = false;
  if ( ( fabs( p_r - p_l ) / fmin( p_r, p_l) ) > ( 10.0 * gamma *  fabs( rho_r - rho_l ) / fmin( rho_r, rho_l)  ) ) pressure_jump = true;
  if ( ( fabs( rho_r - rho_l ) / fmin( rho_r, rho_l) ) < 0.01 ) pressure_jump = false; 
  return pressure_jump;
}

Real Get_Second_Derivative( int i, int j, int k, int direction, int nx, int ny, int nz, Real dx, Real dy, Real dz, Real *field ){
  
  Real delta_x;
  int id_c, id_l, id_r;
  id_c = (i) + (j)*nx + (k)*ny*nx;
  if ( direction == 0 ){
    id_l = (i-1) + (j)*nx + (k)*ny*nx;
    id_r = (i+1) + (j)*nx + (k)*ny*nx; 
    delta_x = dx; 
  }
  if ( direction == 1 ){
    id_l = (i) + (j-1)*nx + (k)*ny*nx;
    id_r = (i) + (j+1)*nx + (k)*ny*nx;
    delta_x = dy;  
  }
  if ( direction == 2 ){
    id_l = (i) + (j)*nx + (k-1)*ny*nx;
    id_r = (i) + (j)*nx + (k+1)*ny*nx;
    delta_x = dz;  
  }
  
  Real val_c, val_l, val_r, d2_val;
  val_c = field[id_c];
  val_l = field[id_l];
  val_r = field[id_r];
  
  //Finite Difference First Order Second Derivative:
  d2_val = ( val_r - 2*val_c + val_l ) / ( delta_x * delta_x );
  return d2_val;  
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
  
  Real eta_2 = DE_ETA_2;
  
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

        //Syncronize advected internal energy with total internal energy when using total internal energy based on local maxEnergy condition
        //find the max nearby total energy
        Emax = E;
        Emax = std::max(Emax, C.Energy[imo]);
        Emax = std::max(Emax, C.Energy[ipo]);
        Emax = std::max(Emax, C.Energy[jmo]);
        Emax = std::max(Emax, C.Energy[jpo]);
        Emax = std::max(Emax, C.Energy[kmo]);
        Emax = std::max(Emax, C.Energy[kpo]);
        if (ge_total/Emax > eta_2 ){
          U = ge_total;
          flag_DE = 0;
        }
        else{
          U = ge_advected;
          flag_DE = 1;
        }
                
        //Set the Internal Energy
        C.GasEnergy[id] = U;
        //Update the total energy after the Dual Energy Condition finished
        // C.Energy[id] = Ek + U;
        
        //Set the flag for which internal energy was used
        #if defined(OUTPUT_DUAL_ENERGY_FLAGS) && defined(COOLING_GRACKLE) 
        Cool.flags_DE[id] = flag_DE;
        #endif
                 
      }
    }
  }
}

void Grid3D::Update_Total_Energy_After_Dual_Energy_function( int g_start, int g_end ){
  
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
  
  for ( k_g=g_start; k_g<g_end; k_g++ ){
    for ( j_g=0; j_g<ny; j_g++ ){
      for ( i_g=0; i_g<nx; i_g++ ){

        i = i_g + nGHST;
        j = j_g + nGHST;
        k = k_g + nGHST;

        id  = (i) + (j)*nx_grid + (k)*ny_grid*nx_grid;

        d = C.density[id];
        d_inv = 1/d;
        vx = C.momentum_x[id] * d_inv;
        vy = C.momentum_y[id] * d_inv;
        vz = C.momentum_z[id] * d_inv;
        E = C.Energy[id];

        Ek = 0.5*d*(vx*vx + vy*vy + vz*vz);
        ge_advected = C.GasEnergy[id];

                
        //Set the Internal Energy in Total Energy
        C.Energy[id] = Ek + ge_advected;
        
      }
    }
  }
}


#ifdef TEMPERATURE_FLOOR
void Grid3D::Apply_Temperature_Floor_CPU_function( int g_start, int g_end ){

  Real temp_floor = H.temperature_floor;
  
  #ifdef COOLING_GRACKLE
  if ( Cosmo.current_a > Cool.scale_factor_UVB_on ) temp_floor = 1;
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
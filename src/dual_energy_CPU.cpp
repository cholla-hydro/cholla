#ifdef DE

#include"global.h"
#include"grid3D.h"

#ifdef PARALLEL_OMP
#include"parallel_omp.h"
#endif


bool Grid3D::Select_Internal_Energy_From_DE( Real E, Real U_total, Real U_advected ){

  Real eta = DE_LIMIT;
  
  if( U_total / E > eta ) return 0;
  else return 1;  
}

void Grid3D::Sync_Energies_3D_CPU(){
  #ifndef PARALLEL_OMP
  Sync_Energies_3D_CPU_function( 0, H.nz_real );
    Apply_Temperature_Floor_CPU_function( 0, H.nz_real );
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
  Real d, d_inv, vx, vy, vz, E, Ek, ge1, ge2, Emax;
  int k, j, i, id;
  int k_g, j_g, i_g;


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
        ge1 = C.GasEnergy[id];
        ge2 = E - Ek;

        // if (ge2 > 0.0 && E > 0.0 && ge2/E > 0.001 ) {        
        //   C.GasEnergy[id] = ge2;
        //   ge1 = ge2;
        // }

        //find the max nearby total energy
        Emax = E;
        Emax = std::max(C.Energy[imo], E);
        Emax = std::max(Emax, C.Energy[ipo]);
        Emax = std::max(Emax, C.Energy[jmo]);
        Emax = std::max(Emax, C.Energy[jpo]);
        Emax = std::max(Emax, C.Energy[kmo]);
        Emax = std::max(Emax, C.Energy[kpo]);

        if (ge2/Emax > 0.1 && ge2 > 0.0 && Emax > 0.0) {
          C.GasEnergy[id] = ge2;
        }
        // sync the total energy with the internal energy
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







#endif
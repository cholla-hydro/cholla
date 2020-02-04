#ifdef COOLING_GRACKLE

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../io.h"
#include"cool_grackle.h"

#ifdef PARALLEL_OMP
#include"../parallel_omp.h"
#endif





void Grid3D::Initialize_Fields_Grackle(){
  
  int nx_g, ny_g, nz_g, nx, ny, nz, nGHST;
  nx_g = H.nx;
  ny_g = H.ny;
  nz_g = H.nz;
  nx = H.nx_real;
  ny = H.ny_real;
  nz = H.nz_real;
  nGHST = H.n_ghost;
  
  Real d, vx, vy, vz, E, Ekin, GE, U;
  bool flag_DE;
  int i, j, k, i_g, j_g, k_g, id;
  for (k=0; k<nz_g; k++) {
    for (j=0; j<ny_g; j++) {
      for (i=0; i<nx_g; i++) {
        id = i + j*nx_g + k*nx_g*ny_g;
        Cool.fields.x_velocity[id] = 0.0;
        Cool.fields.y_velocity[id] = 0.0;
        Cool.fields.z_velocity[id] = 0.0;

        Cool.fields.internal_energy[id] = C.GasEnergy[id]  / C.density[id] * Cool.energy_conv / Cosmo.current_a / Cosmo.current_a ;

      }
    }
  }

  #ifdef OUTPUT_TEMPERATURE
  if (calculate_temperature(&Cool.units, &Cool.fields,  Cool.temperature) == 0) {
    chprintf( "GRACKLE: Error in calculate_temperature.\n");
    return ;
  }
  Real temp_avrg = 0 ;
  for (k=0; k<nz; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        id = (i+nGHST) + (j+nGHST)*nx_g + (k+nGHST)*nx_g*ny_g;
        temp_avrg += Cool.temperature[id];
      }
    }
  }
  temp_avrg /= nz*ny*nx;
  chprintf("Average Temperature = %le K.\n", temp_avrg);
  #endif
  
  
}


void Grid3D::Copy_Fields_To_Grackle(){
  #ifndef PARALLEL_OMP
  Copy_Fields_To_Grackle_function( 0, H.nz_real );
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs( H.nz_real, n_omp_procs, omp_id, &g_start, &g_end  );
    
    Copy_Fields_To_Grackle_function( g_start, g_end );
  }
  #endif
}

void Grid3D::Update_Internal_Energy(){
  #ifndef PARALLEL_OMP
  Update_Internal_Energy_function( 0, H.nz_real );
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
    Get_OMP_Grid_Indxs( H.nz_real, n_omp_procs, omp_id, &g_start, &g_end  );
    
    Update_Internal_Energy_function( g_start, g_end );
  }
  #endif
}

void Grid3D::Copy_Fields_To_Grackle_function( int g_start, int g_end ){
  
  int nx_g, ny_g, nz_g, nx, ny, nz, nGHST;
  nx_g = H.nx;
  ny_g = H.ny;
  nz_g = H.nz;
  nx = H.nx_real;
  ny = H.ny_real;
  nz = H.nz_real;
  nGHST = H.n_ghost;
  
  Real d, vx, vy, vz, E, Ekin, GE, U;
  int flag_DE;
  int k, j, i, id;
  for (k=g_start; k<g_end; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        id = (i+nGHST) + (j+nGHST)*nx_g + (k+nGHST)*nx_g*ny_g;    
        d = C.density[id];
        // vx = C.momentum_x[id] / d;
        // vy = C.momentum_y[id] / d;
        // vz = C.momentum_z[id] / d;
        // E = C.Energy[id];
        // Ekin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );
        GE = C.GasEnergy[id];
        
        //The Flag for Dual Energy Is set on the Sync_Energies_3D step before cooling step
        // flag_DE = Select_Internal_Energy_From_DE( E, E - Ekin, GE );
        // Cool.flags_DE[id] = flag_DE;
        
        // if ( flag_DE ) U = GE;  
        // else U = E - Ekin;
        U = GE;
        Cool.fields.internal_energy[id] = U / d * Cool.energy_conv / Cosmo.current_a / Cosmo.current_a ;
      }
    }
  }
}
  
void Grid3D::Update_Internal_Energy_function( int g_start, int g_end ){
  
  
  int nx_g, ny_g, nz_g, nx, ny, nz, nGHST;
  nx_g = H.nx;
  ny_g = H.ny;
  nz_g = H.nz;
  nx = H.nx_real;
  ny = H.ny_real;
  nz = H.nz_real;
  nGHST = H.n_ghost;
  // Real ge_0, ge_1, delta_ge;
  // Real dens;
  Real dens, vx, vy, vz, E, Ekin, GE, U_0, U_1, delta_U;
  int flag_DE;
  int k, j, i, id;
  for (k=g_start; k<g_end; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        id = (i+nGHST) + (j+nGHST)*nx_g + (k+nGHST)*nx_g*ny_g;
        dens = C.density[id];
        // vx = C.momentum_x[id] / dens;
        // vy = C.momentum_y[id] / dens;
        // vz = C.momentum_z[id] / dens;
        // E = C.Energy[id];
        // Ekin = 0.5 * dens * ( vx*vx + vy*vy + vz*vz );
        GE = C.GasEnergy[id];
        
        // flag_DE = Cool.flags_DE[id];
        // // PRESSURE_DE
        // if ( flag_DE == 0 ) U_0 = E - Ekin;
        // else if ( flag_DE == 1 ) U_0 = GE;
        // else std::cout << " ### Frag_DE ERROR: Flag_DE: " << flag_DE << std::endl;
        U_0 = GE;
        U_1 = Cool.fields.internal_energy[id] * dens / Cool.energy_conv  * Cosmo.current_a * Cosmo.current_a;
        delta_U = U_1 - U_0;
        C.GasEnergy[id] += delta_U ;
        C.Energy[id] += delta_U ;
        

      }
    }
  }
}

void Grid3D::Do_Cooling_Step_Grackle(){
  
  Real kpc_cgs = KPC_CGS;
  // Upfate the units conversion
  Cool.units.a_value = Cosmo.current_a / Cool.units.a_units;
  Cool.units.density_units = Cool.dens_to_CGS  / Cosmo.current_a / Cosmo.current_a / Cosmo.current_a ;
  Cool.units.length_units = kpc_cgs / Cosmo.cosmo_h * Cosmo.current_a;

  
  Copy_Fields_To_Grackle();
  
    
  Real dt_cool = Cosmo.dt_secs;
  chprintf( " dt_cool: %e s\n", dt_cool );
  if (solve_chemistry(&Cool.units, &Cool.fields, dt_cool / Cool.units.time_units ) == 0) {
    chprintf( "GRACKLE: Error in solve_chemistry.\n");
    return ;
  }
  
  #ifdef OUTPUT_TEMPERATURE
  if (calculate_temperature(&Cool.units, &Cool.fields,  Cool.temperature) == 0) {
    chprintf( "GRACKLE: Error in calculate_temperature.\n");
    return ;
  }
  #endif
  
  Update_Internal_Energy(); 
  
}

Real Cool_GK::Get_Mean_Molecular_Weight( int cell_id ){
  
  Real mu, dens, HI_dens, HII_dens, HeI_dens, HeII_dens, HeIII_dens;
  
  dens = fields.density[cell_id];
  HI_dens = fields.HI_density[cell_id];
  HII_dens = fields.HII_density[cell_id];
  HeI_dens = fields.HeI_density[cell_id];
  HeII_dens = fields.HeII_density[cell_id];
  HeIII_dens = fields.HeIII_density[cell_id];
  
  mu = dens / ( HI_dens + 2*HII_dens + ( HeI_dens + 2*HeII_dens + 3*HeIII_dens) / 4 );
  return mu;
  
}











#endif

#ifdef COOLING_GRACKLE

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../io.h"
#include"cool_grackle.h"




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
  Real temp_avrg;
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
  
  Real d, vx, vy, vz, E, Ekin, GE, U;
  bool flag_DE;
  for (int id = 0;id < Cool.field_size;id++) {
    
    d = C.density[id];
    vx = C.momentum_x[id] / d;
    vy = C.momentum_y[id] / d;
    vz = C.momentum_z[id] / d;
    E = C.Energy[id];
    GE = C.GasEnergy[id];
    Ekin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );
    // PRESSURE_DE
    flag_DE = Select_Internal_Energy_From_DE( E, E - Ekin, GE );
    Cool.flags_DE[id] = flag_DE;
    
    if ( flag_DE ) U = GE;  
    else U = E - Ekin;
    Cool.fields.internal_energy[id] = U / d * Cool.energy_conv / Cosmo.current_a / Cosmo.current_a ;
    // Cool.fields.internal_energy[id] = C.GasEnergy[id]  / C.density[id] * Cool.energy_conv / Cosmo.current_a / Cosmo.current_a ;
  }
}
  
void Grid3D::Update_Internal_Energy(){
  
  
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
  bool flag_DE;
  int k, j, i, id;
  for (k=0; k<nz; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        id = (i+nGHST) + (j+nGHST)*nx_g + (k+nGHST)*nx_g*ny_g;
        dens = C.density[id];
        vx = C.momentum_x[id] / dens;
        vy = C.momentum_y[id] / dens;
        vz = C.momentum_z[id] / dens;
        E = C.Energy[id];
        GE = C.GasEnergy[id];
        Ekin = 0.5 * dens * ( vx*vx + vy*vy + vz*vz );
        
        flag_DE = Cool.flags_DE[id];
        // PRESSURE_DE
        if ( flag_DE ) U_0 = GE;
        else U_0 = E - Ekin;
        
        U_1 = Cool.fields.internal_energy[id] * dens / Cool.energy_conv  * Cosmo.current_a * Cosmo.current_a;
        delta_U = U_1 - U_0;
        C.GasEnergy[id] += delta_U ;
        C.Energy[id] += delta_U ;
        
        // ge_0 = C.GasEnergy[id];
        // ge_1 = Cool.fields.internal_energy[id] * dens / Cool.energy_conv  * Cosmo.current_a * Cosmo.current_a;
        // delta_ge = ge_1 - ge_0;
        // C.GasEnergy[id] += delta_ge ;
        // C.Energy[id] += delta_ge ;

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













#endif

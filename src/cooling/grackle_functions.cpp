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
  
  
  
}

void Grid3D::Do_Cooling_Step_Grackle(){
  
  
}













#endif

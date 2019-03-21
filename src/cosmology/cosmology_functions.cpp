#ifdef COSMOLOGY


#include"../grid3D.h"
#include"../global.h"
#include "../io.h"



void Grid3D::Initialize_Cosmology( struct parameters *P ){
  
  chprintf( "Initializing Cosmology... \n");
  Cosmo.Initialize( P, Grav, Particles );
  
  // Change to comoving Cosmological System
  Change_Cosmological_Frame_Sytem( true );
  
  if ( fabs( Cosmo.current_a - Cosmo.next_output ) < 1e-5 ) H.Output_Now = true; 
  
  chprintf( "Cosmology Successfully Initialized. \n\n");
  
}

Real Cosmology::Get_da_from_dt( Real dt ){
  Real a2 = current_a * current_a;
  Real a_dot = sqrt( Omega_M/current_a + a2*Omega_L + Omega_K ) * H0 ;
  return a_dot * dt;
}

Real Cosmology::Get_dt_from_da( Real da ){
  Real a2 = current_a * current_a;
  Real a_dot = sqrt( Omega_M/current_a + a2*Omega_L + Omega_K ) * H0 ;
  return da / a_dot;
}

Real Cosmology::Get_da_from_dt_hydro( Real dt ){
  Real a2, a_dot, da;
  a2 = current_a * current_a;
  a_dot = sqrt( Omega_M/current_a + a2*Omega_L + Omega_K );
  da = dt * a_dot * current_a * current_a;
  return da;
}

Real Cosmology::Get_dt_from_da_hydro( Real da ){
  Real a2, a_dot, dt;
  a2 = current_a * current_a;
  a_dot = sqrt( Omega_M/current_a + a2*Omega_L + Omega_K );
  dt = da / a_dot / current_a / current_a;
  return dt;
}


Real Cosmology::Scale_Function( Real a, Real Omega_M, Real Omega_L, Real Omega_K ){
  Real a3 = a * a * a;
  Real factor = ( Omega_M + a*Omega_K + a3*Omega_L ) / a;
  return 1./sqrt(factor);
}

void Grid3D::Change_Cosmological_Frame_Sytem( bool forward ){
  
  if (forward) chprintf( " Converting to Cosmological Comoving System\n");
  else chprintf( " Converting to Cosmological Physical System\n");
  
  Change_DM_Frame_System( forward );
  #ifndef ONLY_PARTICLES
  Change_GAS_Frame_System( forward );
  #endif
}
void Grid3D::Change_DM_Frame_System( bool forward ){
  
  part_int_t pIndx;
  Real vel_factor;
  if (forward ) vel_factor = Cosmo.current_a ;
  else vel_factor =  1./Cosmo.current_a;
  for ( pIndx=0; pIndx<Particles.n_local; pIndx++ ){
    Particles.vel_x[pIndx] *= vel_factor;
    Particles.vel_y[pIndx] *= vel_factor;
    Particles.vel_z[pIndx] *= vel_factor;
  }
}

void Grid3D::Change_GAS_Frame_System( bool forward ){
  
  Real dens_factor, momentum_factor, energy_factor;
  if ( forward ){
    dens_factor = 1 / Cosmo.rho_0_gas;
    momentum_factor = 1 / Cosmo.rho_0_gas / Cosmo.v_0_gas * Cosmo.current_a;
    energy_factor = 1 / Cosmo.rho_0_gas / Cosmo.v_0_gas / Cosmo.v_0_gas * Cosmo.current_a * Cosmo.current_a;
  }
  else{
    dens_factor = Cosmo.rho_0_gas;
    momentum_factor =  Cosmo.rho_0_gas * Cosmo.v_0_gas / Cosmo.current_a;
    energy_factor =  Cosmo.rho_0_gas * Cosmo.v_0_gas * Cosmo.v_0_gas / Cosmo.current_a / Cosmo.current_a;
  }
  int k, j, i, id;
  for (k=0; k<H.nz; k++) {
    for (j=0; j<H.ny; j++) {
      for (i=0; i<H.nx; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;
        C.density[id] = C.density[id] * dens_factor ;
        C.momentum_x[id] = C.momentum_x[id] *  momentum_factor ;
        C.momentum_y[id] = C.momentum_y[id] *  momentum_factor ;
        C.momentum_z[id] = C.momentum_z[id] *  momentum_factor ;
        C.Energy[id] = C.Energy[id] * energy_factor ;

        #ifdef DE
        C.GasEnergy[id] = C.GasEnergy[id]  * energy_factor ;
        #endif

        #ifdef COOLING_GRACKLE
        // if ( Grav.INITIAL ) continue;
        C.scalar[0*H.n_cells + id] *= dens_factor;
        C.scalar[1*H.n_cells + id] *= dens_factor;
        C.scalar[2*H.n_cells + id] *= dens_factor;
        C.scalar[3*H.n_cells + id] *= dens_factor;
        C.scalar[4*H.n_cells + id] *= dens_factor;
        C.scalar[5*H.n_cells + id] *= dens_factor;
        C.scalar[6*H.n_cells + id] *= dens_factor;
        #endif
      }
    }
  }
}




#endif
/*! \file grid3D.cpp
 *  \brief Definitions of the Grid3D class */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef HDF5
#include <hdf5.h>
#endif
#include "global.h"
#include "grid3D.h"
#include "io.h"
#include "error_handling.h"

#ifdef PARTICLES

Real Grid3D::Update_Grid_and_Particles_DKD(  struct parameters P ){
  
  Real dti;
  
  #ifdef PARTICLES
  //Advance the particles KDK( first step )
  Advance_Particles( 1 );
  //Transfer the particles boundaries
  Transfer_Particles_Boundaries(P);    
  #endif
  
  // Advance the grid by one timestep
  dti = Update_Hydro_Grid();
  
  #ifdef COSMOLOGY
  Cosmo.current_a += 0.5 *Cosmo.delta_a;
  Cosmo.current_z = 1./Cosmo.current_a - 1;
  Particles.current_a = Cosmo.current_a;
  Particles.current_z = Cosmo.current_z;
  Grav.current_a = Cosmo.current_a;  
  #endif //COSMOLOGY  
      
  #ifdef GRAVITY
  //Compute Gravitational potential for next step
  Compute_Gravitational_Potential( &P);
  #endif

  // add one to the timestep count
  H.n_step++;
  
  // update the simulation time ( t += dt )
  Update_Time();

  // set boundary conditions for next time step 
  Set_Boundary_Conditions_All(P);
  
  #ifdef PARTICLES
  //Advance the particles KDK( second step )
  Advance_Particles( 2 );
  //Transfer the particles boundaries
  Transfer_Particles_Boundaries(P); 
  #endif
  
  
  return dti;
}


Real Grid3D::Update_Grid_and_Particles_KDK(  struct parameters P ){
  
  Real dti;
  
  #ifdef PARTICLES
  //Advance the particles KDK( first step )
  Advance_Particles( 1 );   
  //Transfer the particles boundaries
  Transfer_Particles_Boundaries(P); 
  #endif
  
  // Advance the grid by one timestep
  dti = Update_Hydro_Grid();
  
  // update the simulation time ( t += dt )
  Update_Time();
  
      
  #ifdef GRAVITY
  //Compute Gravitational potential for next step
  Compute_Gravitational_Potential( &P);
  #endif

  // add one to the timestep count
  H.n_step++;

  // set boundary conditions for next time step 
  Set_Boundary_Conditions_All(P);
  
  #ifdef PARTICLES
  //Advance the particles KDK( second step )
  Advance_Particles( 2 );
  #endif
  
  return dti;
}

#endif

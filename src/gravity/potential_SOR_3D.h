#if defined(GRAVITY) && defined(SOR)

#ifndef POTENTIAL_SOR_3D_H
#define POTENTIAL_SOR_3D_H

#include "../global.h"
// #include "cuda.h"



class Potential_SOR_3D{
  public:
    
  Real Lbox_x;
  Real Lbox_y;
  Real Lbox_z;

  grav_int_t nx_total;
  grav_int_t ny_total;
  grav_int_t nz_total;

  int nx_local;
  int ny_local;
  int nz_local;
  
  int nx_pot;
  int ny_pot;
  int nz_pot;
  
  int n_ghost;

  Real dx;
  Real dy;
  Real dz;
  grav_int_t n_cells_local;
  grav_int_t n_cells_potential;
  grav_int_t n_cells_total;

  bool potential_initialized;

  struct Fields
  {

  Real *output_h;
  
  Real *input_d;
  // Real *output_d;
  Real *density_d;
  Real *potential_d;
  
  bool *converged_d;
  
  } F;

  Potential_SOR_3D( void );

  void Initialize( Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx, Real dy, Real dz );
  
  void AllocateMemory_CPU( void );
  void AllocateMemory_GPU( void );
  void FreeMemory_GPU( void );
  void Reset( void );
  void Copy_Input( Real *input_density, Real Grav_Constant, Real dens_avrg, Real current_a );
  void Copy_Output( Real *output_potential );
  Real Get_Potential( Real *input_density,  Real *output_potential, Real Grav_Constant, Real dens_avrg, Real current_a );
  void Set_Boundaries(  );
  
};




#endif //POTENTIAL_SOR_H
#endif //GRAVITY
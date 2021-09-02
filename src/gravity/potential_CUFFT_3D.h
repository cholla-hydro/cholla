#ifdef GRAVITY
#ifdef CUFFT

#ifndef POTENTIAL_CUFFT_3D_H
#define POTENTIAL_CUFFT_3D_H

#include "../global/global.h"
#include "../utils/gpu.hpp"


#if PRECISION == 1
typedef cufftReal Real_cufft;
typedef cufftComplex Complex_cufft;
#endif

#if PRECISION == 2
typedef cufftDoubleReal Real_cufft;
typedef cufftDoubleComplex Complex_cufft;
#endif


class Potential_CUFFT_3D{
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

  Real dx;
  Real dy;
  Real dz;
  grav_int_t n_cells_total;
  grav_int_t n_cells_local;

  cufftHandle plan_cufft_fwd;
  cufftHandle plan_cufft_bwd;

  int threads_per_block;
  int blocks_per_grid;

  struct Fields
  {

    // Real_cufft *input_d;
    // Real_cufft *output_d;
    Real_cufft *input_real_d;
    Complex_cufft *input_d;
    Complex_cufft *output_d;
    Complex_cufft *transform_d;

    Complex_cufft *output_h;

    // Real *k_fft_x;
    // Real *k_fft_y;
    // Real *k_fft_z;

    Real *G_h;
    Real *G_d;

  } F;

  Potential_CUFFT_3D( void );

  void Initialize( Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx, Real dy, Real dz );

  void AllocateMemory_CPU( void );
  void AllocateMemory_GPU( void );
  void FreeMemory_GPU( void );
  void Reset( void );
  void Copy_Input( Real *input_density, Real Grav_Constant, Real dens_avrg, Real current_a );
  void Copy_Output( Real *output_potential );
  void Get_K_for_Green_function( void );
  Real Get_Potential( Real *input_density,  Real *output_potential, Real Grav_Constant, Real dens_avrg, Real current_a );

};




#endif //POTENTIAL_CUFFT_H
#endif //CUFFT
#endif //GRAVITY

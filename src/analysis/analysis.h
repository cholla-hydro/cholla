#ifdef ANALYSIS

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include"../global.h"

class Analysis_Module{
public:
  
  Real Lbox_x;
  Real Lbox_y;
  Real Lbox_z;
  
  Real xMin;
  Real yMin;
  Real zMin;
  
  Real dx;
  Real dy;
  Real dz;
  
  int nx_total;
  int ny_total;
  int nz_total;
  
  int nx_local;
  int ny_local;
  int nz_local;
  int n_ghost;
  
  int n_outputs;
  int next_output_indx;
  real_vector_t scale_outputs;
  Real next_output;
  bool output_now;
  int n_file;
  
  #ifdef COSMOLOGY
  Real current_z;
  #endif
  
  
  
  #ifdef PHASE_DIAGRAM
  int n_dens;
  int n_temp;
  Real temp_min;
  Real temp_max;
  Real dens_min;
  Real dens_max;
  float *phase_diagram;
  #ifdef MPI_CHOLLA
  float *phase_diagram_global;
  #endif
  #endif
  
  
  #ifdef LYA_STATISTICS
  
  #endif
  
  
  Analysis_Module( void );
  void Initialize( Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx_real, Real dy_real, Real dz_real, int n_ghost_hydro, Real z_now, struct parameters *P );
  void Reset(void);
  
  void Load_Scale_Outputs( struct parameters *P );
  void Set_Next_Scale_Output(  );

  
  
  #ifdef PHASE_DIAGRAM
  void Initialize_Phase_Diagram( struct parameters *P );
  #endif
  
  #ifdef LYA_STATISTICS
  void Initialize_Lya_Statistics( struct parameters *P );
  #endif
};



#endif
#endif
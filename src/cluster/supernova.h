#ifndef SUPERNOVA_H
#define SUPERNOVA_H
#include "../global/global.h" //defines Real
#include "../grid/grid3D.h" // defines Header
namespace Supernova {
  extern int n_cluster;

  //extern Real cluster_data[];
  
  //extern Real *d_cluster_array;
  //extern Real *d_omega_array;
  //extern bool *d_flags_array;
  extern Real *d_hydro_array;
  extern Real *d_tracker;
  extern Real h_tracker[];

  extern Real xMin;
  extern Real yMin;
  extern Real zMin;
  
  extern Real xMax;
  extern Real yMax;
  extern Real zMax;

  extern Real dx;
  extern Real dy;
  extern Real dz;
 
  extern int nx;
  extern int ny;
  extern int nz;
  
  extern int pnx;
  extern int pny;
  extern int pnz;

  extern int n_tracker;
  extern int n_cells;
  extern int n_fields;
  extern int n_ghost;
  extern int supernova_e;

  extern Real R_cl;
  extern Real SFR;
  void Initialize(Grid3D G, struct Parameters *P);
  void Initialize_GPU(void);
  void InitializeS99(void);
  void Calc_Omega(void);//Real *cluster_array, Real *omega_array, int n_cluster);
  void Calc_Flags(Real time);
  void Copy_Tracker(void);
  void Print_Tracker(Grid3D G);

  Real Feedback(Real density, Real energy, Real time, Real dt);
  Real Update_Grid(Grid3D G, Real old_dti);
  
}
#endif


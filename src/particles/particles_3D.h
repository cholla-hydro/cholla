#ifdef PARTICLES

#ifndef PARTICLES_H
#define PARTICLES_H

#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <cstdlib>
#include <string.h>
#include"../global.h"
#include "../gravity/grav3D.h"



/*! \class Part3D
 *  \brief Class to create a set of particles in 3D space. */
class Particles_3D
{
  public:

  part_int_t n_local;

  part_int_t n_total;
  part_int_t n_total_initial;

  Real dt;
  Real t;
  Real max_dt;

  Real C_cfl;

  bool INITIAL;

  #ifdef SINGLE_PARTICLE_MASS
  Real particle_mass;
  #endif

  #ifdef COSMOLOGY
  Real current_z;
  Real current_a;
  #endif

  #ifdef PARTICLE_IDS
  int_vector_t partIDs;
  #endif
  #ifndef SINGLE_PARTICLE_MASS
  real_vector_t mass;
  #endif
  real_vector_t pos_x;
  real_vector_t pos_y;
  real_vector_t pos_z;
  real_vector_t vel_x;
  real_vector_t vel_y;
  real_vector_t vel_z;
  real_vector_t grav_x;
  real_vector_t grav_y;
  real_vector_t grav_z;


  #ifdef MPI_CHOLLA
  int_vector_t out_indxs_vec_x0;
  int_vector_t out_indxs_vec_x1;
  int_vector_t out_indxs_vec_y0;
  int_vector_t out_indxs_vec_y1;
  int_vector_t out_indxs_vec_z0;
  int_vector_t out_indxs_vec_z1;


  part_int_t n_transfer_x0;
  part_int_t n_transfer_x1;
  part_int_t n_transfer_y0;
  part_int_t n_transfer_y1;
  part_int_t n_transfer_z0;
  part_int_t n_transfer_z1;

  part_int_t n_send_x0;
  part_int_t n_send_x1;
  part_int_t n_send_y0;
  part_int_t n_send_y1;
  part_int_t n_send_z0;
  part_int_t n_send_z1;

  part_int_t n_recv_x0;
  part_int_t n_recv_x1;
  part_int_t n_recv_y0;
  part_int_t n_recv_y1;
  part_int_t n_recv_z0;
  part_int_t n_recv_z1;

  part_int_t n_in_buffer_x0;
  part_int_t n_in_buffer_x1;
  part_int_t n_in_buffer_y0;
  part_int_t n_in_buffer_y1;
  part_int_t n_in_buffer_z0;
  part_int_t n_in_buffer_z1;
  #endif //MPI_CHOLLA

  bool TRANSFER_DENSITY_BOUNDARIES;
  bool TRANSFER_PARTICLES_BOUNDARIES;

  
  struct Grid
  {

    int nx_local, ny_local, nz_local;
    int nx_total, ny_total, nz_total;

    Real xMin, yMin, zMin;
    Real xMax, yMax, zMax;
    Real dx, dy, dz;

    Real domainMin_x, domainMax_x;
    Real domainMin_y, domainMax_y;
    Real domainMin_z, domainMax_z;

    int n_ghost_particles_grid;
    int n_cells;

    Real *density;
    Real *gravity_x;
    Real *gravity_y;
    Real *gravity_z;


  } G;
  
  Particles_3D(void);

  void Initialize( struct parameters *P, Grav3D &Grav,  Real xbound, Real ybound, Real zbound, Real xdglobal, Real ydglobal, Real zdglobal  );
  
  void Allocate_Memory();
  
  void Initialize_Grid_Values();
  
  void Initialize_Sphere();
  
  void Initialize_Zeldovich_Pancake( struct parameters *P );
  
  void Load_Particles_Data( struct parameters *P );
  
  void Free_Memory();
  
  void Reset();
  
  void Clear_Density();
  
  void Get_Density_CIC_Serial( );
  
  #ifdef HDF5
  void Load_Particles_Data_HDF5( hid_t file_id, int nfile, struct parameters *P );
  #endif
  
  #ifdef PARALLEL_OMP
  void Get_Density_CIC_OMP( );
  #endif
  
  #ifdef MPI_CHOLLA
  void Clear_Vectors_For_Transfers( void );
  void Clear_Particles_For_Transfer( void );
  void Select_Particles_to_Transfer( int dir );
  void Select_Particles_to_Transfer_All( int *flags );
  void Add_Particle_To_Buffer( Real *buffer, part_int_t n_in_buffer, int buffer_length, Real pId, Real pMass,
                              Real pPos_x, Real pPos_y, Real pPos_z, Real pVel_x, Real pVel_y, Real pVel_z);
  void Add_Particle_To_Vectors( Real pId, Real pMass, Real pPos_x, Real pPos_y, Real pPos_z, Real pVel_x, Real pVel_y, Real pVel_z );
  void Load_Particles_to_Buffer( int direction, int side, Real *send_buffer, int buffer_length  );
  void Unload_Particles_from_Buffer( int direction, int side, Real *recv_buffer, part_int_t n_recv,
        Real *send_buffer_y0, Real *send_buffer_y1, Real *send_buffer_z0, Real *send_buffer_z1, int buffer_length_y0, int buffer_length_y1, int buffer_length_z0, int buffer_length_z1);
  void Remove_Transfered_Particles();
  #endif
};











#endif //PARTICLES_H
#endif //PARTICLES
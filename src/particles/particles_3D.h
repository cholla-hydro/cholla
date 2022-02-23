#ifdef PARTICLES

#ifndef PARTICLES_H
#define PARTICLES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstdlib>
#include <string.h>
#include "../global/global.h"
#include "../gravity/grav3D.h"

#ifdef PARTICLES_GPU
#define TPB_PARTICLES 1024
// #define PRINT_GPU_MEMORY
#endif



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

  Real particle_mass;

  #ifdef COSMOLOGY
  Real current_z;
  Real current_a;
  #endif


  #ifdef PARTICLES_CPU
  #ifdef PARTICLE_IDS
  int_vector_t partIDs;
  #endif
  #ifndef SINGLE_PARTICLE_MASS
  real_vector_t mass;
  #endif
  #ifdef PARTICLE_AGE
  real_vector_t age;
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
  #endif //PARTICLES_CPU

  #ifdef PARTICLES_GPU
  part_int_t particles_array_size;
  #ifdef PARTICLE_IDS
  part_int_t *partIDs_dev;
  #endif
  Real *mass_dev;
  Real *pos_x_dev;
  Real *pos_y_dev;
  Real *pos_z_dev;
  Real *vel_x_dev;
  Real *vel_y_dev;
  Real *vel_z_dev;
  Real *grav_x_dev;
  Real *grav_y_dev;
  Real *grav_z_dev;


  #endif //PARTICLES_GPU


  #ifdef MPI_CHOLLA

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


  #ifdef PARTICLES_CPU
  int_vector_t out_indxs_vec_x0;
  int_vector_t out_indxs_vec_x1;
  int_vector_t out_indxs_vec_y0;
  int_vector_t out_indxs_vec_y1;
  int_vector_t out_indxs_vec_z0;
  int_vector_t out_indxs_vec_z1;
  #endif //PARTICLES_CPU


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

    int boundary_type_x0, boundary_type_x1;
    int boundary_type_y0, boundary_type_y1;
    int boundary_type_z0, boundary_type_z1;

    int n_ghost_particles_grid;
    int n_cells;
    #ifdef PARTICLES_GPU
    Real gpu_allocation_factor;
    part_int_t size_blocks_array;
    int n_cells_potential;
    #endif

    Real *density;
    #ifdef PARTICLES_CPU
    Real *gravity_x;
    Real *gravity_y;
    Real *gravity_z;
    #endif

    #ifdef PARTICLES_GPU
    Real *density_dev;
    Real *potential_dev;
    Real *gravity_x_dev;
    Real *gravity_y_dev;
    Real *gravity_z_dev;
    Real *dti_array_dev;
    Real *dti_array_host;

    #ifdef MPI_CHOLLA
    bool *transfer_particles_flags_d;
    int *transfer_particles_indices_d;
    int *replace_particles_indices_d;
    int *transfer_particles_prefix_sum_d;
    int *transfer_particles_prefix_sum_blocks_d;
    int *n_transfer_d;
    int *n_transfer_h;

    int send_buffer_size_x0;
    int send_buffer_size_x1;
    int send_buffer_size_y0;
    int send_buffer_size_y1;
    int send_buffer_size_z0;
    int send_buffer_size_z1;
    Real *send_buffer_x0_d;
    Real *send_buffer_x1_d;
    Real *send_buffer_y0_d;
    Real *send_buffer_y1_d;
    Real *send_buffer_z0_d;
    Real *send_buffer_z1_d;

    int recv_buffer_size_x0;
    int recv_buffer_size_x1;
    int recv_buffer_size_y0;
    int recv_buffer_size_y1;
    int recv_buffer_size_z0;
    int recv_buffer_size_z1;
    Real *recv_buffer_x0_d;
    Real *recv_buffer_x1_d;
    Real *recv_buffer_y0_d;
    Real *recv_buffer_y1_d;
    Real *recv_buffer_z0_d;
    Real *recv_buffer_z1_d;

    #endif // MPI_CHOLLA

    #endif //PARTICLES_GPU


  } G;

  Particles_3D(void);

  void Initialize( struct parameters *P, Grav3D &Grav,  Real xbound, Real ybound, Real zbound, Real xdglobal, Real ydglobal, Real zdglobal  );

  #ifdef PARTICLES_GPU

  void Free_GPU_Array_Real( Real *array );
  void Free_GPU_Array_int( int *array );
  void Free_GPU_Array_bool( bool *array );
  void Allocate_Memory_GPU();
  void Allocate_Particles_GPU_Array_Real( Real **array_dev, part_int_t size );
  void Allocate_Particles_GPU_Array_bool( bool **array_dev, part_int_t size );
  void Allocate_Particles_GPU_Array_int( int **array_dev, part_int_t size );
  void Allocate_Particles_Grid_Field_Real( Real **array_dev, int size );
  void Copy_Particles_Array_Real_Host_to_Device( Real *array_host, Real *array_dev, part_int_t size);
  void Copy_Particles_Array_Real_Device_to_Host( Real *array_dev, Real *array_host, part_int_t size);
  void Set_Particles_Array_Real( Real value, Real *array_dev, part_int_t size);
  void Free_Memory_GPU();
  void Initialize_Grid_Values_GPU();
  void Get_Density_CIC_GPU();
  void Get_Density_CIC_GPU_function(part_int_t n_local, Real particle_mass,  Real xMin, Real xMax, Real yMin, Real yMax, Real zMin, Real zMax, Real dx, Real dy, Real dz, int nx_local, int ny_local, int nz_local, int n_ghost_particles_grid, int n_cells, Real *density_h, Real *density_dev, Real *pos_x_dev, Real *pos_y_dev , Real *pos_z_dev, Real *mass_dev);
  void Clear_Density_GPU();
  void Clear_Density_GPU_function( Real *density_dev, int n_cells);
  void Copy_Potential_To_GPU( Real *potential_host, Real *potential_dev, int n_cells_potential );
  void Get_Gravity_Field_Particles_GPU( Real *potential_host );
  void Get_Gravity_Field_Particles_GPU_function( int nx_local, int ny_local, int nz_local, int n_ghost_particles_grid, int n_cells_potential, Real dx, Real dy, Real dz,  Real *potential_host, Real *potential_dev, Real *gravity_x_dev, Real *gravity_y_dev, Real *gravity_z_dev  );
  void Get_Gravity_CIC_GPU();
  void Get_Gravity_CIC_GPU_function( part_int_t n_local, int nx_local, int ny_local, int nz_local, int n_ghost_particles_grid, Real xMin, Real xMax, Real yMin, Real yMax, Real zMin,  Real zMax, Real dx, Real dy, Real dz,   Real *pos_x_dev, Real *pos_y_dev, Real *pos_z_dev, Real *grav_x_dev,  Real *grav_y_dev,  Real *grav_z_dev, Real *gravity_x_dev, Real *gravity_y_dev, Real *gravity_z_dev );
  Real Calc_Particles_dt_GPU_function( int ngrid, part_int_t n_local, Real dx, Real dy, Real dz, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *dti_array_host, Real *dti_array_dev );
  void Advance_Particles_KDK_Step1_GPU_function( part_int_t n_local, Real dt, Real *pos_x_dev, Real *pos_y_dev, Real *pos_z_dev, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev  );
  void Advance_Particles_KDK_Step1_Cosmo_GPU_function( part_int_t n_local, Real delta_a, Real *pos_x_dev, Real *pos_y_dev, Real *pos_z_dev, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev, Real current_a, Real H0, Real cosmo_h, Real Omega_M, Real Omega_L, Real Omega_K  );
  void Advance_Particles_KDK_Step2_GPU_function( part_int_t n_local, Real dt, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev  );
  void Advance_Particles_KDK_Step2_Cosmo_GPU_function( part_int_t n_local, Real delta_a,  Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev, Real current_a, Real H0, Real cosmo_h, Real Omega_M, Real Omega_L, Real Omega_K  );
  part_int_t Compute_Particles_GPU_Array_Size( part_int_t n );
  int Select_Particles_to_Transfer_GPU( int direction, int side );
  void Copy_Transfer_Particles_to_Buffer_GPU(int n_transfer, int direction, int side, Real *send_buffer, int buffer_length );
  void Replace_Tranfered_Particles_GPU( int n_transfer );
  void Unload_Particles_from_Buffer_GPU( int direction, int side , Real *recv_buffer_h, int n_recv );
  void Copy_Transfer_Particles_from_Buffer_GPU(int n_recv, Real *recv_buffer_d );
  #endif //PARTICLES_GPU



  void Allocate_Memory();

  void Initialize_Grid_Values();

  void Initialize_Sphere(struct parameters *P);

  void Initialize_Disk_Stellar_Clusters(struct parameters *P);

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

  void Get_Density_CIC();

  #ifdef MPI_CHOLLA
  void Clear_Particles_For_Transfer( void );
  void Select_Particles_to_Transfer_All( int *flags );
  void Add_Particle_To_Buffer( Real *buffer, part_int_t n_in_buffer, int buffer_length, Real pId, Real pMass, Real pAge,
                              Real pPos_x, Real pPos_y, Real pPos_z, Real pVel_x, Real pVel_y, Real pVel_z);
  void Remove_Transfered_Particles();

  #ifdef PARTICLES_CPU
  void Clear_Vectors_For_Transfers( void );
  void Add_Particle_To_Vectors( Real pId, Real pMass, Real pAge, Real pPos_x, Real pPos_y, Real pPos_z, Real pVel_x, Real pVel_y, Real pVel_z, int *flags );
  void Select_Particles_to_Transfer_All_CPU( int *flags );
  void Load_Particles_to_Buffer_CPU( int direction, int side, Real *send_buffer, int buffer_length  );
  void Unload_Particles_from_Buffer_CPU( int direction, int side, Real *recv_buffer, part_int_t n_recv,
        Real *send_buffer_y0, Real *send_buffer_y1, Real *send_buffer_z0, Real *send_buffer_z1, int buffer_length_y0, int buffer_length_y1, int buffer_length_z0, int buffer_length_z1, int *flags);
  #endif//PARTICLES_CPU


  #ifdef PARTICLES_GPU
  void Allocate_Memory_GPU_MPI();
  void ReAllocate_Memory_GPU_MPI();
  void Load_Particles_to_Buffer_GPU( int direction, int side, Real *send_buffer, int buffer_length  );
  #endif //PARTICLES_GPU
  #endif

};


#endif //PARTICLES_H
#endif //PARTICLES

/*! \file grid3D.h
 *  \brief Declarations of the Grid3D class. */

#ifndef GRID3D_H
#define GRID3D_H

#ifdef   MPI_CHOLLA
#include"mpi_routines.h"
#endif /*MPI_CHOLLA*/

#include<stdio.h>
#include"global.h"

#ifdef HDF5
#include<hdf5.h>
#endif

#ifdef GRAVITY
#include"gravity/grav3D.h"
#endif

#ifdef PARTICLES
#include "particles/particles_3D.h"
#endif

#ifdef COSMOLOGY
#include"cosmology/cosmology.h"
#endif

#ifdef COOLING_GRACKLE
#include "cooling/cool_grackle.h"
#endif

#ifdef CPU_TIME
#include "timing_functions.h"
#endif

struct Rotation
{
  /*! \var nx
  *   \brief Number of pixels in x-dir of rotated, projected image*/
  int nx;

  /*! \var nz
  *   \brief Number of pixels in z-dir of rotated, projected image*/
  int nz;

  /*! \var nx_min
  *   \brief Left most point in the projected image for this subvolume*/
  int nx_min;

  /*! \var nx_max
  *   \brief Right most point in the projected image for this subvolume*/
  int nx_max;

  /*! \var nz_min
  *   \brief Bottom most point in the projected image for this subvolume*/
  int nz_min;

  /*! \var nz_max
  *   \brief Top most point in the projected image for this subvolume*/
  int nz_max;

  /*! \var delta
  *   \brief Rotation angle about z axis in simulation frame*/
  Real delta;

  /*! \var theta
  *   \brief Rotation angle about x axis in simulation frame*/
  Real theta;

  /*! \var phi
  *   \brief Rotation angle about y axis in simulation frame*/
  Real phi;

  /*! \var Lx
  *   \brief Physical x-dir size of projected image*/
  Real Lx;

  /*! \var Lz
  *   \brief Physical z-dir size of projected image*/
  Real Lz;

  /*! \var i_delta
  *   \brief number of output projection for delta rotation*/
  int i_delta;

  /*! \var n_delta
  *   \brief total number of output projection for delta rotation*/
  Real n_delta;

  /*! \var ddelta_dt
  *   \brief rate of delta rotation*/
  Real ddelta_dt;

  /*! \var flag_delta
   *  \brief output mode for box rotation*/
  int flag_delta;
};

struct Header
{
  /*! \var n_cells 
  *  \brief Total number of cells in the grid (including ghost cells) */
  int n_cells;

  /*! \var n_ghost
  *  \brief Number of ghost cells on each side of the grid */
  int n_ghost;

  /*! \var nx
  *  \brief Total number of cells in the x-dimension */
  int nx;

  /*! \var ny
  *  \brief Total number of cells in the y-dimension */
  int ny;

  /*! \var nz
  *  \brief Total number of cells in the z-dimension */
  int nz;

  /*! \var nx_real
  *  \brief Number of real cells in the x-dimension */
  int nx_real;

  /*! \var ny
  *  \brief Number of real cells in the y-dimension */
  int ny_real;

  /*! \var nz
  *  \brief Number of real cells in the z-dimension */
  int nz_real;

  /*! \var xbound */
  /*  \brief Global domain x-direction minimum */
  Real xbound;

  /*! \var ybound */
  /*  \brief Global domain y-direction minimum */
  Real ybound;

  /*! \var zbound */
  /*  \brief Global domain z-direction minimum */
  Real zbound;

  /*! \var domlen_x */
  /*  \brief Local domain length in x-direction */
  Real domlen_x;

  /*! \var domlen_y */
  /*  \brief Local domain length in y-direction */
  Real domlen_y;
   
  /*! \var domlen_z */
  /*  \brief Local domain length in z-direction */
  Real domlen_z;

  /*! \var xblocal */
  /*  \brief Local domain x-direction minimum */
  Real xblocal;

  /*! \var yblocal */
  /*  \brief Local domain y-direction minimum */
  Real yblocal;

  /*! \var zblocal*/
  /*  \brief Local domain z-direction minimum */
  Real zblocal;

  /*! \var xdglobal */
  /*  \brief Global domain length in x-direction */
  Real xdglobal;

  /*! \var ydglobal */
  /*  \brief Global domain length in y-direction */
  Real ydglobal;

  /*! \var zdglobal */
  /*  \brief Global domain length in z-direction */
  Real zdglobal;

  /*! \var dx
  *  \brief x-width of cells */
  Real dx;

  /*! \var dy
  *  \brief y-width of cells */
  Real dy;

  /*! \var dz
  *  \brief z-width of cells */
  Real dz;
      
  /*! \var t
  *  \brief Simulation time */
  Real t;

  /*! \var dt
  *  \brief Length of the current timestep */
  Real dt;

  /*! \var t_wall
  *  \brief Wall time */
  Real t_wall;

  /*! \var n_step
  *  \brief Number of timesteps taken */
  int n_step;

  /*! \var n_fields
  *  \brief Number of fields (conserved variables, scalars, etc.) */
  int n_fields;
  
  // Values for lower limit for density and temperature
  Real density_floor;
  Real temperature_floor;
  
  Real Ekin_avrg;
  
  //Flag to indicate when to transfer the Conserved boundaries
  bool TRANSFER_HYDRO_BOUNDARIES;
  
  #ifdef GRAVITY
  /*! \var n_ghost_potential_offset
  *  \brief Number of offset betewen hydro_ghost_cells and potential_ghost_cells */
  int n_ghost_potential_offset;
  #endif
  
  /*! \var Output_Now
  *  \brief Flag set to true when data has to be written to file */
  bool Output_Now;

};

/*! \class Grid3D
 *  \brief Class to create a 3D grid of cells. */
class Grid3D
{
  public:

    /*! \var flag_init
     *  \brief Initialization flag */
    int flag_init;

    /*! \var gflag
     *  \brief Flag that determines which buffer contains updated conserved variables */
    int gflag;

    /*! \var struct Header H
     *  \brief Header for the grid */
    struct Header H;

    /*! \var struct Rotation R
     *  \brief Rotation struct for data projections */
    struct Rotation R;

    /*! \var buffer0
     *  \brief Buffer to hold conserved variable arrays */
    Real *buffer0;

    /*! \var buffer1
     *  \brief Buffer to hold conserved variable arrays */
    Real *buffer1;
    
    #ifdef GRAVITY
    // Object that contains data for gravity
    Grav3D Grav;
    #endif
    
    #ifdef PARTICLES
    // Object that contains data for particles
    Particles_3D Particles;
    #endif
    
    #ifdef COSMOLOGY
    // Object that contains data for cosmology
    Cosmology Cosmo;
    #endif
    
    #ifdef COOLING_GRACKLE
    // Object that contains data for Grackle cooling
    Cool_GK Cool;
    #endif
    
    #ifdef CPU_TIME
    Time Timer;
    #endif

    struct Conserved
    {
      /*! \var density
       *  \brief Array containing the density of each cell in the grid */
      Real *density;

      /*! \var momentum_x 
       *  \brief Array containing the momentum in the x direction of each cell in the grid */
      Real *momentum_x;

      /*! \var momentum_y 
       *  \brief Array containing the momentum in the y direction of each cell in the grid */
      Real *momentum_y;

      /*! \var momentum_z 
       *  \brief Array containing the momentum in the z direction of each cell in the grid */
      Real *momentum_z;			

      /*! \var Energy 
       *  \brief Array containing the total Energy of each cell in the grid */
      Real *Energy;
      
      #ifdef DE
      /*! \var GasEnergy
       *  \brief Array containing the internal energy of each cell, only tracked separately when using
           the dual-energy formalism. */
      Real *GasEnergy;
      #endif

      #ifdef SCALAR
      /*! \var scalar
       *  \brief Array containing the values of the passive scalar variable(s). */
      Real *scalar;
      #endif
      
      #ifdef GRAVITY
      #ifdef GRAVITY_COUPLE_GPU
      /*! \var Grav_potential
       *  \brief Array containing the gravitational potential of each cell, only tracked separately when using
            GRAVITY. */
      Real *Grav_potential;
      #endif
      #ifdef GRAVITY_COUPLE_CPU
      //Arrays for conserved variables at the begining of the timesteps
      Real *density_0;
      Real *momentum_x_0;
      Real *momentum_y_0;
      Real *momentum_z_0;
      #endif
      #endif//GRAVITY

    } C;


    /*! \fn Grid3D(void)
     *  \brief Constructor for the grid */
    Grid3D(void);

    /*! \fn void Initialize(int nx_in, int ny_in, int nz_in)
     *  \brief Initialize the grid. */
    void Initialize(struct parameters *P);
 
    /*! \fn void AllocateMemory(void)
     *  \brief Allocate memory for the d, m, E arrays. */
    void AllocateMemory(void);

    /*! \fn void Set_Initial_Conditions(parameters P)
     *  \brief Set the initial conditions based on info in the parameters structure. */
    void Set_Initial_Conditions(parameters P);

    /*! \fn void Get_Position(long i, long j, long k, Real *xpos, Real *ypos, Real *zpos)
     *  \brief Get the cell-centered position based on cell index */ 
    void Get_Position(long i, long j, long k, Real *xpos, Real *ypos, Real *zpos);

    /*! \fn void Set_Domain_Properties(struct parameters P)
     *  \brief Set local domain properties */
    void Set_Domain_Properties(struct parameters P);

    /*! \fn void set_dt(Real dti)
     *  \brief Calculate the timestep. */
    void set_dt(Real dti);
    
    #ifdef GRAVITY
    /*! \fn void set_dt(Real dti)
     *  \brief Calculate the timestep for Gravity. */
    void set_dt_Gravity();
    #endif
    
    /*! \fn Real calc_dti_CPU_1D()
     *  \brief Calculate the maximum inverse timestep on 1D, according to the CFL condition (Toro 6.17). */
    Real calc_dti_CPU_1D();
    
    /*! \fn Real calc_dti_CPU_2D()
     *  \brief Calculate the maximum inverse timestep on 2D, according to the CFL condition (Toro 6.17). */
    Real calc_dti_CPU_2D();

    /*! \fn Real calc_dti_CPU_3D_function()
     *  \brief Calculate the maximum inverse timestep on 3D using openMP, according to the CFL condition (Toro 6.17). */
    Real calc_dti_CPU_3D_function( int g_start, int g_end );
        
    /*! \fn Real calc_dti_CPU_3D()
     *  \brief Calculate the maximum inverse timestep on 3D, according to the CFL condition (Toro 6.17). */
    Real calc_dti_CPU_3D();

    /*! \fn Real calc_dti_CPU()
     *  \brief Calculate the maximum inverse timestep, according to the CFL condition (Toro 6.17). */ 
    Real calc_dti_CPU();

    /*! \fn void Update_Grid(void)
     *  \brief Update the conserved quantities in each cell. */
    Real Update_Grid(void);
    
    /*! \fn void Update_Hydro_Grid(void)
     *  \brief Do all steps to update the hydro. */
    Real Update_Hydro_Grid(void);
    
    void Update_Time();
    
    #ifdef DE
    void Sync_Energies_3D_CPU_function( int g_start, int g_end );
    void Sync_Energies_3D_CPU();
    bool Select_Internal_Energy_From_DE( Real E, Real U_total, Real U_advected );
    Real Get_Average_Kinetic_Energy_function( int g_start, int g_end );
    void Get_Average_Kinetic_Energy();
    #endif
    void Apply_Temperature_Floor_CPU_function( int g_start, int g_end );

    /*! \fn void Write_Header_Binary(FILE *fp)
     *  \brief Write the relevant header info to a binary output file. */
    void Write_Header_Binary(FILE *fp);

    /*! \fn void Write_Grid_Binary(FILE *fp)
     *  \brief Write the grid to a file, at the current simulation time. */
    void Write_Grid_Binary(FILE *fp);

#ifdef HDF5
    /*! \fn void Write_Header_HDF5(hid_t file_id)
     *  \brief Write the relevant header info to the HDF5 file. */
    void Write_Header_HDF5(hid_t file_id);

    /*! \fn void Write_Grid_HDF5(hid_t file_id)
     *  \brief Write the grid to a file, at the current simulation time. */
    void Write_Grid_HDF5(hid_t file_id);

    /*! \fn void Write_Projection_HDF5(hid_t file_id)
     *  \brief Write projected density and temperature data to a file. */
    void Write_Projection_HDF5(hid_t file_id);    

    /*! \fn void Write_Header_Rotated_HDF5(hid_t file_id)
     *  \brief Write the relevant header info to the HDF5 file for rotated projection. */
    void Write_Header_Rotated_HDF5(hid_t file_id);

    /*! \fn void Write_Rotated_Projection_HDF5(hid_t file_id)
     *  \brief Write rotated projected data to a file, at the current simulation time. */
    void Write_Rotated_Projection_HDF5(hid_t file_id);   

    /*! \fn void Write_Slices_HDF5(hid_t file_id)
     *  \brief Write xy, xz, and yz slices of all data to a file. */
    void Write_Slices_HDF5(hid_t file_id);    

#endif

    /*! \fn void Read_Grid(struct parameters P)
     *  \brief Read in grid data from an output file. */
    void Read_Grid(struct parameters P);

    /*! \fn Read_Grid_Binary(FILE *fp)
     *  \brief Read in grid data from a binary file. */
    void Read_Grid_Binary(FILE *fp);
    
#ifdef HDF5
    /*! \fn void Read_Grid_HDF5(hid_t file_id)
     *  \brief Read in grid data from an hdf5 file. */
    void Read_Grid_HDF5(hid_t file_id);
#endif

    /*! \fn void Reset(void)
     *  \brief Reset the Grid3D class. */
    void Reset(void);

    /*! \fn void FreeMemory(void)
     *  \brief Free the memory for the density array. */
    void FreeMemory(void);

    /*! \fn void Constant(Real rho, Real vx, Real vy, Real vz, Real P)
     *  \brief Constant gas properties. */
    void Constant(Real rho, Real vx, Real vy, Real vz, Real P);

    /*! \fn void Sound_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
     *  \brief Sine wave perturbation. */
    void Sound_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A);

    /*! \fn void Square_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
     *  \brief Square wave density perturbation with amplitude A*rho in pressure equilibrium. */
    void Square_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A);

    /*! \fn void Riemann(Real rho_l, Real v_l, Real P_l, Real rho_r, Real v_r, Real P_r, Real diaph)
     *  \brief Initialize the grid with a Riemann problem. */
    void Riemann(Real rho_l, Real v_l, Real P_l, Real rho_r, Real v_r, Real P_r, Real diaph);

    /*! \fn void Shu_Osher()
     *  \brief Initialize the grid with the Shu-Osher shock tube problem. See Stone 2008, Section 8.1 */
    void Shu_Osher();

    /*! \fn void Blast_1D()
     *  \brief Initialize the grid with two interacting blast waves. See Stone 2008, Section 8.1.*/
    void Blast_1D();

    /*! \fn void KH()
    *  \brief Initialize the grid with a Kelvin-Helmholtz instability with a discontinuous interface. */
    void KH();

    /*! \fn void KH_res_ind()
     *  \brief Initialize the grid with a Kelvin-Helmholtz instability whose modes are resolution independent. */
    void KH_res_ind();

    /*! \fn void Rayleigh_Taylor()
    *  \brief Initialize the grid with a 2D Rayleigh-Taylor instability. */
    void Rayleigh_Taylor();

    /*! \fn void Gresho()
     *  \brief Initialize the grid with the 2D Gresho problem described in LW03. */
    void Gresho();    

    /*! \fn void Implosion_2D()
     *  \brief Implosion test described in Liska, 2003. */
    void Implosion_2D();

    /*! \fn void Explosion_2D()
     *  \brief Explosion test described in Liska, 2003. */
    void Explosion_2D();

    /*! \fn void Noh_2D()
     *  \brief Noh test described in Liska, 2003. */
    void Noh_2D();

    /*! \fn void Noh_3D()
     *  \brief Noh test described in Stone, 2008. */
    void Noh_3D();

    /*! \fn void Disk_2D()
     *  \brief Initialize the grid with a 2D disk following a Kuzmin profile. */
    void Disk_2D();    

    /*! \fn void Disk_3D(parameters P)
     *  \brief Initialize the grid with a 3D disk following a Miyamoto-Nagai profile. */
    void Disk_3D(parameters P);    

    /*! \fn void Set_Boundary_Conditions(parameters P)
     *  \brief Set the boundary conditions based on info in the parameters structure. */
    void Set_Boundary_Conditions(parameters P);
    
    /*! \fn void Set_Boundary_Conditions_All(parameters P)
     *  \brief Set the boundary conditions for all componentes based on info in the parameters structure. */
    void Set_Boundary_Conditions_All( parameters P);

    /*! \fn int Check_Custom_Boundary(int *flags, struct parameters P)
     *  \brief Check for custom boundary conditions */
    int Check_Custom_Boundary(int *flags, struct parameters P);

    /*! \fn void Set_Boundaries(int dir, int flags[])
     *  \brief Apply boundary conditions to the grid. */
    void Set_Boundaries(int dir, int flags[]);

    /*! \fn Set_Boundary_Extents(int dir, int *imin, int *imax)
     *  \brief Set the extents of the ghost region we are initializing. */
    void Set_Boundary_Extents(int dir, int *imin, int *imax);

    /*! \fn Set_Boundary_Mapping(int ig, int jg, int kg, int flags[], Real *a)
     *  \brief Given the i,j,k index of a ghost cell, return the index of the
        corresponding real cell, and reverse the momentum if necessary. */
    int  Set_Boundary_Mapping(int ig, int jg, int kg, int flags[], Real *a);

    /*! \fn int Find_Index(int ig, int nx, int flag, int face, Real *a)
     *  \brief Given a ghost cell index and boundary flag, 
        return the index of the corresponding real cell. */
    int  Find_Index(int ig, int nx, int flag, int face, Real *a);

    /*! \fn void Custom_Boundary(char bcnd[MAXLEN])
     *  \brief Select appropriate custom boundary function. */
    void Custom_Boundary(char bcnd[MAXLEN]);

    /*! \fn void Noh_Boundary()
     *  \brief Apply analytic boundary conditions to +x, +y (and +z) faces, 
        as per the Noh problem in Liska, 2003, or in Stone, 2008. */
    void Noh_Boundary();
    
    
    void Spherical_Overpressure_3D();
    
    void Spherical_Overdensity_3D();
    
    void Uniform_Grid();


#ifdef   MPI_CHOLLA
    void Set_Boundaries_MPI(struct parameters P);
    void Set_Boundaries_MPI_SLAB(int *flags, struct parameters P);
    void Set_Boundaries_MPI_BLOCK(int *flags, struct parameters P);
    void Set_Edge_Boundaries(int dir, int *flags);
    void Set_Edge_Boundary_Extents(int dir, int edge, int *imin, int *imax);
    void Load_and_Send_MPI_Comm_Buffers(int dir, int *flags);
    void Load_and_Send_MPI_Comm_Buffers_SLAB(int *flags);
    void Load_and_Send_MPI_Comm_Buffers_BLOCK(int dir, int *flags);
    void Wait_and_Unload_MPI_Comm_Buffers_SLAB(int *flags);
    void Wait_and_Unload_MPI_Comm_Buffers_BLOCK(int dir, int *flags);
    void Unload_MPI_Comm_Buffers(int index);
    void Unload_MPI_Comm_Buffers_SLAB(int index);
    void Unload_MPI_Comm_Buffers_BLOCK(int index);
    int Load_Hydro_Buffer_X0();
    int Load_Hydro_Buffer_X1();
    int Load_Hydro_Buffer_Y0();
    int Load_Hydro_Buffer_Y1();
    int Load_Hydro_Buffer_Z0();
    int Load_Hydro_Buffer_Z1();
#endif /*MPI_CHOLLA*/

  #ifdef GRAVITY
  void Initialize_Gravity( struct parameters *P );
  void Compute_Gravitational_Potential( struct parameters *P );
  void Copy_Hydro_Density_to_Gravity_Function( int g_start, int g_end);
  void Copy_Hydro_Density_to_Gravity();
  void Copy_Potential_to_Hydro_Grid_Function( int g_start, int g_end ); 
  void Copy_Potential_to_Hydro_Grid();
  void Extrapolate_Grav_Potential_Function( int g_start, int g_end );
  void Extrapolate_Grav_Potential();
  int Load_Gravity_Potential_To_Buffer( int direction, int side, Real *buffer, int buffer_start  );
  void Unload_Gravity_Potential_from_Buffer( int direction, int side, Real *buffer, int buffer_start  );
  #ifdef GRAVITY_COUPLE_CPU
  void Get_Gravitational_Field_Function( int g_start, int g_end );
  void Get_Gravitational_Field();
  void Add_Gavity_To_Hydro_Function( int g_start, int g_end );
  void Add_Gavity_To_Hydro();
  #endif//GRAVITY_COUPLE_CPU
  #endif//GRAVITY 
  
  #ifdef PARTICLES
  void Initialize_Particles( struct parameters *P );
  void Copy_Particles_Density_function( int g_start, int g_end );
  void Copy_Particles_Density();
  void Copy_Particles_Density_to_Gravity(struct parameters P);
  #ifdef MPI_CHOLLA
  int Load_Particles_Density_Boundary_to_Buffer( int direction, int side, Real *buffer );
  void Unload_Particles_Density_Boundary_From_Buffer( int direction, int side, Real *buffer );
  void Transfer_Particles_Density_Boundaries_MPI( struct parameters P );
  // void Transfer_Particles_Boundaries_MPI( struct parameters P );
  void Load_and_Send_Particles_X0( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_X1( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_Y0( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_Y1( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_Z0( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_Z1( int ireq_n_particles, int ireq_particles_transfer );
  void Unload_Particles_from_Buffer_X0();
  void Unload_Particles_from_Buffer_X1();
  void Unload_Particles_from_Buffer_Y0();
  void Unload_Particles_from_Buffer_Y1();
  void Unload_Particles_from_Buffer_Z0();
  void Unload_Particles_from_Buffer_Z1();
  void Wait_and_Recv_Particles_Transfer_BLOCK(int dir, int *flags);
  void Load_N_Particles_Transfer(int index, int *ireq_particles_transfer);
  void Wait_and_Unload_MPI_Comm_Particles_Buffers_BLOCK(int dir, int *flags);
  void Unload_Particles_From_Buffers_BLOCK(int index);
  void Finish_Particles_Transfer();
  #endif//MPI_CHOLLA
  void Transfer_Particles_Density_Boundaries( struct parameters P );
  // void Transfer_Particles_Boundaries( struct parameters P );
  void WriteData_Particles(  struct parameters P, int nfile);
  void OutputData_Particles(  struct parameters P, int nfile);
  void Load_Particles_Data(  struct parameters P);
  #ifdef HDF5
  void Write_Particles_Header_HDF5( hid_t file_id);
  void Write_Particles_Data_HDF5( hid_t file_id);
  void Load_Particles_Data_HDF5(hid_t file_id, int nfile);
  #endif//HDF5
  void Get_Gravity_Field_Particles_function( int g_start, int g_end );
  void Get_Gravity_Field_Particles();
  void Get_Gravity_CIC_function( part_int_t p_start, part_int_t p_end );
  void Get_Gravity_CIC();
  void Advance_Particles_KDK_Step1_function( part_int_t p_start, part_int_t p_end );
  void Advance_Particles_KDK_Step1( );
  void Advance_Particles_KDK_Step2_function( part_int_t p_start, part_int_t p_end );
  void Advance_Particles_KDK_Step2( );
  void Get_Particles_Accelration();
  void Advance_Particles( int N_KDK_step );
  Real Calc_Particles_dt_function( part_int_t p_start, part_int_t p_end );
  Real Calc_Particles_dt();
  #endif//PARTICLES
  
  #ifdef COSMOLOGY
  void Initialize_Cosmology( struct parameters *P );
  void Change_DM_Frame_System( bool forward );
  void Change_GAS_Frame_System( bool forward );
  void Change_Cosmological_Frame_Sytem( bool forward );
  void Advance_Particles_KDK_Cosmo_Step1_function( part_int_t p_start, part_int_t p_end );
  void Advance_Particles_KDK_Cosmo_Step2_function( part_int_t p_start, part_int_t p_end );
  Real Calc_Particles_dt_Cosmo_function( part_int_t p_start, part_int_t p_end );
  Real Calc_Particles_dt_Cosmo();
  #endif
  
  #ifdef COOLING_GRACKLE
  void Initialize_Grackle( struct parameters *P );
  void Allocate_Memory_Grackle();
  void Initialize_Fields_Grackle();
  void Copy_Fields_To_Grackle_function( int g_start, int g_end );
  void Copy_Fields_To_Grackle();
  void Update_Internal_Energy_function( int g_start, int g_end );
  void Update_Internal_Energy();
  void Do_Cooling_Step_Grackle();
  #endif
  
};



#endif //GRID3D_H

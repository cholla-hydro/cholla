/*! \file grid3D.h
 *  \brief Declarations of the Grid3D class. */

#ifndef GRID3D_H
#define GRID3D_H

#ifdef   MPI_CHOLLA
#include "../mpi/mpi_routines.h"
#endif /*MPI_CHOLLA*/

#include <stdio.h>
#include "../global/global.h"
#include "../global/global_cuda.h"

#ifdef HDF5
#include <hdf5.h>
#endif

#ifdef GRAVITY
#include "../gravity/grav3D.h"
#endif

#ifdef PARTICLES
#include "../particles/particles_3D.h"
#endif

#include "../model/disk_galaxy.h"

#ifdef COSMOLOGY
#include "../cosmology/cosmology.h"
#endif

#ifdef COOLING_GRACKLE
#include "../cooling_grackle/cool_grackle.h"
#endif

#ifdef CPU_TIME
#include "../utils/timing_functions.h"
#endif

#ifdef CHEMISTRY_GPU
#include "chemistry_gpu/chemistry_gpu.h"
#endif

#ifdef ANALYSIS
#include "../analysis/analysis.h"
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

   /*! \var xblocal */
  /*  \brief Local domain x-direction minimum */
  Real xblocal;

  /*! \var yblocal */
  /*  \brief Local domain y-direction minimum */
  Real yblocal;

  /*! \var zblocal */
  /*  \brief Local domain z-direction minimum */
  Real zblocal;

  /*! \var xblocal_max */
  /*  \brief Local domain x-direction maximum */
  Real xblocal_max;

  /*! \var yblocal_max */
  /*  \brief Local domain y-direction maximum */
  Real yblocal_max;

  /*! \var zblocal_max */
  /*  \brief Local domain z-direction maximum */
  Real zblocal_max;

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

  #ifdef AVERAGE_SLOW_CELLS
  Real min_dt_slow;
  #endif

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

  //Parameters For Spherical Colapse Problem
  Real sphere_density;
  Real sphere_radius;
  Real sphere_background_density;
  Real sphere_center_x;
  Real sphere_center_y;
  Real sphere_center_z;


  #ifdef GRAVITY
  /*! \var n_ghost_potential_offset
  *  \brief Number of offset betewen hydro_ghost_cells and potential_ghost_cells */
  int n_ghost_potential_offset;
  #endif

  #ifdef COSMOLOGY
  bool OUTPUT_SCALE_FACOR;
  #endif

  /*! \var Output_Now
  *  \brief Flag set to true when data has to be written to file */
  bool Output_Now;
  bool Output_Initial;

  /*! \var Output_Complete_Data
  *  \brief Flag set to true when all the data will  be written to file (Restart File ) */
  bool Output_Complete_Data;


};

/*! \class Grid3D
 *  \brief Class to create a 3D grid of cells. */
class Grid3D
{
  public:

    /*! \var flag_init
     *  \brief Initialization flag */
    int flag_init;

    /*! \var struct Header H
     *  \brief Header for the grid */
    struct Header H;

    /*! \var struct Rotation R
     *  \brief Rotation struct for data projections */
    struct Rotation R;

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

    #ifdef CHEMISTRY_GPU
    // Object that contains data for the GPU chemistry solver
    Chem_GPU Chem;
    #endif

    #ifdef ANALYSIS
    Analysis_Module Analysis;
    #endif

    struct Conserved
    {
      /*! pointer to conserved variable array on the host */
      Real *host;

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

      #ifdef SCALAR
      /*! \var scalar
       *  \brief Array containing the values of the passive scalar variable(s). */
      Real *scalar;
      #endif  // SCALAR

      #ifdef MHD
      /*! \var magnetic_x \brief Array containing the magnetic field in the x
       *  direction of each cell in the grid. Note that this is the magnetic
       *  field at the x+1/2 face of the cell since constrained transport
       *  requires face centered, not cell centered, magnetic fields */
      Real *magnetic_x;

      /*! \var magnetic_y \brief Array containing the magnetic field in the y
       *  direction of each cell in the grid. Note that this is the magnetic
       *  field at the y+1/2 face of the cell since constrained transport
       *  requires face centered, not cell centered, magnetic fields */
      Real *magnetic_y;

      /*! \var magnetic_z \brief Array containing the magnetic field in the z
       *  direction of each cell in the grid. Note that this is the magnetic
       *  field at the z+1/2 face of the cell since constrained transport
       *  requires face centered, not cell centered, magnetic fields */
      Real *magnetic_z;
      #endif  // MHD

      #ifdef DE
      /*! \var GasEnergy
       *  \brief Array containing the internal energy of each cell, only tracked separately when using
           the dual-energy formalism. */
      Real *GasEnergy;
      #endif  // DE

      /*! \var grav_potential
      *  \brief Array containing the gravitational potential of each cell, only tracked separately when using  GRAVITY. */
      Real *Grav_potential;

      #ifdef CHEMISTRY_GPU
      Real *HI_density;
      Real *HII_density;
      Real *HeI_density;
      Real *HeII_density;
      Real *HeIII_density;
      Real *e_density;
      #endif


      /*! pointer to conserved variable on device */
      Real *device;
      Real *d_density, *d_momentum_x, *d_momentum_y, *d_momentum_z,
           *d_Energy, *d_scalar, *d_magnetic_x, *d_magnetic_y, *d_magnetic_z,
           *d_GasEnergy;

       /*! pointer to gravitational potential on device */
      Real *d_Grav_potential;
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

     /*! \fn void Write_Header_Text(FILE *fp)
     *  \brief Write the relevant header info to a text output file. */
    void Write_Header_Text(FILE *fp);

    /*! \fn void Write_Grid_Text(FILE *fp)
     *  \brief Write the grid to a file, at the current simulation time. */
    void Write_Grid_Text(FILE *fp);

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
    void Read_Grid_HDF5(hid_t file_id, struct parameters P);
#endif

    /*! \fn void Reset(void)
     *  \brief Reset the Grid3D class. */
    void Reset(void);

    /*! \fn void FreeMemory(void)
     *  \brief Free the memory for the density array. */
    void FreeMemory(void);

    /*! \fn void Constant(Real rho, Real vx, Real vy, Real vz, Real P)
     *  \brief Constant gas properties. */
    void Constant(Real rho, Real vx, Real vy, Real vz, Real P, Real Bx, Real By, Real Bz);

    /*! \fn void Sound_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
     *  \brief Sine wave perturbation. */
    void Sound_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A);

    /*! \fn void Square_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A)
     *  \brief Square wave density perturbation with amplitude A*rho in pressure equilibrium. */
    void Square_Wave(Real rho, Real vx, Real vy, Real vz, Real P, Real A);

    /*! \fn void Riemann(Real rho_l, Real vx_l, Real vy_l, Real vz_l, Real P_l, Real Bx_l, Real By_l, Real Bz_l,
                         Real rho_r, Real vx_r, Real vy_r, Real vz_r, Real P_r, Real Bx_r, Real By_r, Real Bz_r,
                         Real diaph)
     *  \brief Initialize the grid with a Riemann problem. */
    void Riemann(Real rho_l, Real vx_l, Real vy_l, Real vz_l, Real P_l, Real Bx_l, Real By_l, Real Bz_l,
                 Real rho_r, Real vx_r, Real vy_r, Real vz_r, Real P_r, Real Bx_r, Real By_r, Real Bz_r,
                 Real diaph);

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

    /*! \fn void Set_Boundary_Conditions_Grid(parameters P)
     *  \brief Set the boundary conditions for all components based on info in the parameters structure. */
    void Set_Boundary_Conditions_Grid( parameters P);

    /*! \fn int Check_Custom_Boundary(int *flags, struct parameters P)
     *  \brief Check for custom boundary conditions */
    int Check_Custom_Boundary(int *flags, struct parameters P);

    /*! \fn void Set_Boundaries(int dir, int flags[])
     *  \brief Apply boundary conditions to the grid. */
    void Set_Boundaries(int dir, int flags[]);

    /*! \fn Set_Boundary_Extents(int dir, int *imin, int *imax)
     *  \brief Set the extents of the ghost region we are initializing. */
    void Set_Boundary_Extents(int dir, int *imin, int *imax);

    /*! \fn void Custom_Boundary(char bcnd[MAXLEN])
     *  \brief Select appropriate custom boundary function. */
    void Custom_Boundary(char bcnd[MAXLEN]);

    /*! \fn void Noh_Boundary()
     *  \brief Apply analytic boundary conditions to +x, +y (and +z) faces,
        as per the Noh problem in Liska, 2003, or in Stone, 2008. */
    void Noh_Boundary();

    /*! \fn void Spherical_Overpressure_3D()
     *  \brief Initialize the grid with a 3D spherical overdensity and overpressue. */
    void Spherical_Overpressure_3D();

    /*! \fn void Spherical_Overpressure_3D()
     *  \brief Initialize the grid with a 3D spherical overdensity for gravitational collapse */
    void Spherical_Overdensity_3D();

    void Uniform_Grid();

    void Zeldovich_Pancake( struct parameters P );

    void Chemistry_Test( struct parameters P );


#ifdef   MPI_CHOLLA
    void Set_Boundaries_MPI(struct parameters P);
    void Set_Boundaries_MPI_BLOCK(int *flags, struct parameters P);
    void Load_and_Send_MPI_Comm_Buffers(int dir, int *flags);
    void Wait_and_Unload_MPI_Comm_Buffers(int dir, int *flags);
    void Unload_MPI_Comm_Buffers(int index);

    int Load_Hydro_DeviceBuffer_X0(Real *buffer);
    int Load_Hydro_DeviceBuffer_X1(Real *buffer);
    int Load_Hydro_DeviceBuffer_Y0(Real *buffer);
    int Load_Hydro_DeviceBuffer_Y1(Real *buffer);
    int Load_Hydro_DeviceBuffer_Z0(Real *buffer);
    int Load_Hydro_DeviceBuffer_Z1(Real *buffer);

    void Unload_Hydro_DeviceBuffer_X0(Real *buffer);
    void Unload_Hydro_DeviceBuffer_X1(Real *buffer);
    void Unload_Hydro_DeviceBuffer_Y0(Real *buffer);
    void Unload_Hydro_DeviceBuffer_Y1(Real *buffer);
    void Unload_Hydro_DeviceBuffer_Z0(Real *buffer);
    void Unload_Hydro_DeviceBuffer_Z1(Real *buffer);
#endif /*MPI_CHOLLA*/

  #ifdef GRAVITY
  void Initialize_Gravity( struct parameters *P );
  void Compute_Gravitational_Potential( struct parameters *P );
  void Copy_Hydro_Density_to_Gravity_Function( int g_start, int g_end);
  void Copy_Hydro_Density_to_Gravity();
  void Extrapolate_Grav_Potential_Function( int g_start, int g_end );
  void Extrapolate_Grav_Potential();
  void Set_Potential_Boundaries_Periodic( int direction, int side, int *flags );
  int Load_Gravity_Potential_To_Buffer( int direction, int side, Real *buffer, int buffer_start  );
  void Unload_Gravity_Potential_from_Buffer( int direction, int side, Real *buffer, int buffer_start  );
  void Set_Potential_Boundaries_Isolated( int direction, int side, int *flags );
  void Compute_Potential_Boundaries_Isolated( int dir, struct parameters *P );
  void Compute_Potential_Isolated_Boundary( int direction, int side, int bc_potential_type );
  #ifdef SOR
  void Get_Potential_SOR( Real Grav_Constant, Real dens_avrg, Real current_a, struct parameters *P );
  int Load_Poisson_Boundary_To_Buffer( int direction, int side, Real *buffer  );
  void Unload_Poisson_Boundary_From_Buffer( int direction, int side, Real *buffer_host  );
  #endif
  #ifdef GRAVITY_GPU
  void Copy_Hydro_Density_to_Gravity_GPU();
  void Extrapolate_Grav_Potential_GPU();
  int Load_Gravity_Potential_To_Buffer_GPU( int direction, int side, Real *buffer, int buffer_start  );
  void Unload_Gravity_Potential_from_Buffer_GPU( int direction, int side, Real *buffer, int buffer_start  );
  void Set_Potential_Boundaries_Isolated_GPU( int direction, int side, int *flags );
  void Set_Potential_Boundaries_Periodic_GPU( int direction, int side, int *flags );
  #endif

  #endif//GRAVITY

  #ifdef GRAVITY_ANALYTIC_COMP
  void Add_Analytic_Potential(struct parameters *P);
  void Add_Analytic_Galaxy_Potential(int g_start, int g_end, DiskGalaxy& gal);
  #endif //GRAVITY_ANALYTIC_COMP

  #ifdef PARTICLES
  void Initialize_Particles( struct parameters *P );
  void Initialize_Uniform_Particles();
  void Copy_Particles_Density_function( int g_start, int g_end );
  void Copy_Particles_Density();
  void Copy_Particles_Density_to_Gravity(struct parameters P);
  void Set_Particles_Density_Boundaries_Periodic( int direction, int side );
  void Transfer_Particles_Boundaries( struct parameters P );
  Real Update_Grid_and_Particles_KDK( struct parameters P );
  void Set_Particles_Boundary( int dir, int side);
  void Set_Particles_Open_Boundary(int dir, int side);
  #ifdef MPI_CHOLLA
  int Load_Particles_Density_Boundary_to_Buffer( int direction, int side, Real *buffer );
  void Unload_Particles_Density_Boundary_From_Buffer( int direction, int side, Real *buffer );
  void Load_and_Send_Particles_X0( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_X1( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_Y0( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_Y1( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_Z0( int ireq_n_particles, int ireq_particles_transfer );
  void Load_and_Send_Particles_Z1( int ireq_n_particles, int ireq_particles_transfer );
  void Unload_Particles_from_Buffer_X0( int *flags );
  void Unload_Particles_from_Buffer_X1( int *flags );
  void Unload_Particles_from_Buffer_Y0( int *flags );
  void Unload_Particles_from_Buffer_Y1( int *flags );
  void Unload_Particles_from_Buffer_Z0( int *flags );
  void Unload_Particles_from_Buffer_Z1( int *flags );
  void Wait_NTransfer_and_Request_Recv_Particles_Transfer_BLOCK(int dir, int *flags);
  void Load_NTtransfer_and_Request_Receive_Particles_Transfer(int index, int *ireq_particles_transfer);
  void Wait_and_Unload_MPI_Comm_Particles_Buffers_BLOCK(int dir, int *flags);
  void Unload_Particles_From_Buffers_BLOCK(int index, int *flags );
  void Finish_Particles_Transfer();
  #endif//MPI_CHOLLA
  void Transfer_Particles_Density_Boundaries( struct parameters P );
  void Copy_Particles_Density_Buffer_Device_to_Host( int direction, int side, Real *buffer_d, Real *buffer_h );
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
  void Advance_Particles_KDK_Step1( );
  void Advance_Particles_KDK_Step2( );
  void Advance_Particles_KDK_Step1_function( part_int_t p_start, part_int_t p_end );
  void Advance_Particles_KDK_Step2_function( part_int_t p_start, part_int_t p_end );
  void Get_Particles_Acceleration();
  void Advance_Particles( int N_KDK_step );
  Real Calc_Particles_dt_function( part_int_t p_start, part_int_t p_end );
  Real Calc_Particles_dt();
  #ifdef PARTICLES_GPU
  Real Calc_Particles_dt_GPU();
  void Advance_Particles_KDK_Step1_GPU();
  void Advance_Particles_KDK_Step2_GPU();
  void Set_Particles_Boundary_GPU( int dir, int side);
  void Set_Particles_Density_Boundaries_Periodic_GPU( int direction, int side );
  #endif//PARTICLES_GPU
  #ifdef GRAVITY_GPU
  void Copy_Potential_From_GPU();
  void Copy_Particles_Density_to_GPU();
  void Copy_Particles_Density_GPU();
  int Load_Particles_Density_Boundary_to_Buffer_GPU( int direction, int side, Real *buffer  );
  void Unload_Particles_Density_Boundary_From_Buffer_GPU( int direction, int side, Real *buffer  );
  #endif//GRAVITY_GPU
  #endif//PARTICLES

  #ifdef COSMOLOGY
  void Initialize_Cosmology( struct parameters *P );
  void Change_DM_Frame_System( bool forward );
  void Change_GAS_Frame_System( bool forward );
  void Change_GAS_Frame_System_GPU( bool forward );
  void Change_Cosmological_Frame_Sytem( bool forward );
  void Advance_Particles_KDK_Cosmo_Step1_function( part_int_t p_start, part_int_t p_end );
  void Advance_Particles_KDK_Cosmo_Step2_function( part_int_t p_start, part_int_t p_end );
  Real Calc_Particles_dt_Cosmo_function( part_int_t p_start, part_int_t p_end );
  Real Calc_Particles_dt_Cosmo();
  #ifdef PARTICLES_GPU
  void Advance_Particles_KDK_Cosmo_Step1_GPU();
  void Advance_Particles_KDK_Cosmo_Step2_GPU();
  #endif//PARTICLES_GPU
  #endif//COSMOLOGY

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

  #ifdef CHEMISTRY_GPU
  void Initialize_Chemistry( struct parameters *P );
  void Compute_Gas_Temperature(  Real *temperature, bool convert_cosmo_units  );
  void Update_Chemistry();
  #endif

  #ifdef ANALYSIS
  void Initialize_Analysis_Module( struct parameters *P );
  void Compute_and_Output_Analysis( struct parameters *P );
  void Output_Analysis( struct parameters *P );
  void Write_Analysis_Header_HDF5( hid_t file_id );
  void Write_Analysis_Data_HDF5( hid_t file_id );

  #ifdef PHASE_DIAGRAM
  void Compute_Phase_Diagram();
  #endif

  #ifdef LYA_STATISTICS
  void Populate_Lya_Skewers_Local( int axis );
  void Compute_Transmitted_Flux_Skewer( int skewer_id, int axis );
  void Compute_Lya_Statistics( );
  void Compute_Flux_Power_Spectrum_Skewer( int skewer_id, int axis );
  void Initialize_Power_Spectrum_Measurements( int axis );
  #ifdef OUTPUT_SKEWERS
  void Output_Skewers_File( struct parameters *P );
  void Write_Skewers_Header_HDF5( hid_t file_id );
  void Write_Skewers_Data_HDF5( hid_t file_id );
  #endif
  #endif//LYA_STATISTICS
  #endif//ANALYSIS

  #ifdef PARTICLES
  #ifdef DE
  #ifdef PARTICLE_AGE
  void Cluster_Feedback();
  void Cluster_Feedback_Function(part_int_t p_start, part_int_t p_end);
  #endif
  #endif
  #endif

};

// typedef for Grid3D_PointerMemberFunction
typedef void (Grid3D::*Grid3D_PMF_UnloadHydroBuffer)(Real *);
typedef void (Grid3D::*Grid3D_PMF_UnloadGravityPotential)
               (int, int, Real *, int);
typedef void (Grid3D::*Grid3D_PMF_UnloadParticleDensity)
               (int, int, Real *);

#endif //GRID3D_H

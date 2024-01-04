#ifndef GRAV3D_H
#define GRAV3D_H

#include <stdio.h>

#include "../global/global.h"

#ifdef SOR
  #include "../gravity/potential_SOR_3D.h"
#endif

#ifdef PARIS
  #include "../gravity/potential_paris_3D.h"
#endif

#ifdef PARIS_GALACTIC
  #include "../gravity/potential_paris_galactic.h"
#endif

#ifdef HDF5
  #include <hdf5.h>
#endif

#define GRAV_ISOLATED_BOUNDARY_X
#define GRAV_ISOLATED_BOUNDARY_Y
#define GRAV_ISOLATED_BOUNDARY_Z

#define TPB_GRAV  1024
#define TPBX_GRAV 16
#define TPBY_GRAV 8
#define TPBZ_GRAV 8

/*! \class Grid3D
 *  \brief Class to create a the gravity object. */
class Grav3D
{
 public:
  Real Lbox_x;
  Real Lbox_y;
  Real Lbox_z;

  Real xMin;
  Real yMin;
  Real zMin;
  Real xMax;
  Real yMax;
  Real zMax;
  /*! \var nx
   *  \brief Total number of cells in the x-dimension */
  int nx_total;
  /*! \var ny
   *  \brief Total number of cells in the y-dimension */
  int ny_total;
  /*! \var nz
   *  \brief Total number of cells in the z-dimension */
  int nz_total;

  /*! \var nx_local
   *  \brief Local number of cells in the x-dimension */
  int nx_local;
  /*! \var ny_local
   *  \brief Local number of cells in the y-dimension */
  int ny_local;
  /*! \var nz_local
   *  \brief Local number of cells in the z-dimension */
  int nz_local;

  /*! \var dx
   *  \brief x-width of cells */
  Real dx;
  /*! \var dy
   *  \brief y-width of cells */
  Real dy;
  /*! \var dz
   *  \brief z-width of cells */
  Real dz;

#ifdef COSMOLOGY
  Real current_a;
#endif

  Real dens_avrg;

  int n_cells;
  int n_cells_potential;

  bool INITIAL;

  Real dt_prev;
  Real dt_now;

  Real Gconst;

  bool TRANSFER_POTENTIAL_BOUNDARIES;

  bool BC_FLAGS_SET;
  int *boundary_flags;

#ifdef SOR
  Potential_SOR_3D Poisson_solver;
#endif

#ifdef PARIS
  PotentialParis3D Poisson_solver;
#endif

#ifdef PARIS_GALACTIC
  #ifdef SOR
    #define PARIS_GALACTIC_TEST
  PotentialParisGalactic Poisson_solver_test;
  #else
  PotentialParisGalactic Poisson_solver;
  #endif
#endif

  struct Fields {
    /*! \var density_h
     *  \brief Array containing the density of each cell in the grid */
    Real *density_h;

    /*! \var potential_h
     *  \brief Array containing the gravitational potential of each cell in the
     * grid */
    Real *potential_h;

    /*! \var potential_h
     *  \brief Array containing the gravitational potential of each cell in the
     * grid at the previous time step */
    Real *potential_1_h;

#ifdef GRAVITY_ANALYTIC_COMP
    Real *analytic_potential_h;
#endif

#ifdef GRAVITY_GPU

    /*! \var density_d
     *  \brief Device Array containing the density of each cell in the grid */
    Real *density_d;

    /*! \var potential_d
     *  \brief Device Array containing the gravitational potential of each cell
     * in the grid */
    Real *potential_d;

    /*! \var potential_d
     *  \brief Device Array containing the gravitational potential of each cell
     * in the grid at the previous time step */
    Real *potential_1_d;

  #ifdef GRAVITY_ANALYTIC_COMP
    Real *analytic_potential_d;
  #endif

#endif  // GRAVITY_GPU

// Arrays for computing the potential values in isolated boundaries
#ifdef GRAV_ISOLATED_BOUNDARY_X
    Real *pot_boundary_x0;
    Real *pot_boundary_x1;
#endif
#ifdef GRAV_ISOLATED_BOUNDARY_Y
    Real *pot_boundary_y0;
    Real *pot_boundary_y1;
#endif
#ifdef GRAV_ISOLATED_BOUNDARY_Z
    Real *pot_boundary_z0;
    Real *pot_boundary_z1;
#endif

#ifdef GRAVITY_GPU
  #ifdef GRAV_ISOLATED_BOUNDARY_X
    Real *pot_boundary_x0_d;
    Real *pot_boundary_x1_d;
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
    Real *pot_boundary_y0_d;
    Real *pot_boundary_y1_d;
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
    Real *pot_boundary_z0_d;
    Real *pot_boundary_z1_d;
  #endif
#endif  // GRAVITY_GPU

  } F;

  /*! \fn Grav3D(void)
   *  \brief Constructor for the gravity class */
  Grav3D(void);

  /*! \fn void Initialize(int nx_in, int ny_in, int nz_in)
   *  \brief Initialize the grid. */
  void Initialize(Real x_min, Real y_min, Real z_min, Real x_max, Real y_max, Real z_max, Real Lx, Real Ly, Real Lz,
                  int nx_total, int ny_total, int nz_total, int nx_real, int ny_real, int nz_real, Real dx_real,
                  Real dy_real, Real dz_real, int n_ghost_pot_offset, struct Parameters *P);

  void AllocateMemory_CPU(void);
  void Initialize_values_CPU();
  void FreeMemory_CPU(void);

  void Read_Restart_HDF5(struct Parameters *P, int nfile);
  void Write_Restart_HDF5(struct Parameters *P, int nfile);

  Real Get_Average_Density();
  Real Get_Average_Density_function(int g_start, int g_end);

  void Set_Boundary_Flags(int *flags);

#ifdef SOR
  void Copy_Isolated_Boundary_To_GPU_buffer(Real *isolated_boundary_h, Real *isolated_boundary_d, int boundary_size);
  void Copy_Isolated_Boundaries_To_GPU(struct Parameters *P);
#endif

#ifdef GRAVITY_GPU
  void AllocateMemory_GPU(void);
  void FreeMemory_GPU(void);
#endif
};

#endif  // GRAV3D_H

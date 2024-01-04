#if defined(GRAVITY) && defined(SOR)

  #ifndef POTENTIAL_SOR_3D_H
    #define POTENTIAL_SOR_3D_H

    #include <stdlib.h>

    #include "../global/global.h"

// #define TIME_SOR
// #define HALF_SIZE_BOUNDARIES

class Potential_SOR_3D
{
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

  int n_ghost_transfer;
  int size_buffer_x;
  int size_buffer_y;
  int size_buffer_z;

  bool TRANSFER_POISSON_BOUNDARIES;

  int iteration_parity;

  bool potential_initialized;

  struct Fields {
    Real *output_h;

    Real *input_d;
    // Real *output_d;
    Real *density_d;
    Real *potential_d;

    bool *converged_d;

    bool *converged_h;

    Real *boundaries_buffer_x0_d;
    Real *boundaries_buffer_x1_d;
    Real *boundaries_buffer_y0_d;
    Real *boundaries_buffer_y1_d;
    Real *boundaries_buffer_z0_d;
    Real *boundaries_buffer_z1_d;

    Real *boundary_isolated_x0_d;
    Real *boundary_isolated_x1_d;
    Real *boundary_isolated_y0_d;
    Real *boundary_isolated_y1_d;
    Real *boundary_isolated_z0_d;
    Real *boundary_isolated_z1_d;

    #ifdef MPI_CHOLLA
    Real *recv_boundaries_buffer_x0_d;
    Real *recv_boundaries_buffer_x1_d;
    Real *recv_boundaries_buffer_y0_d;
    Real *recv_boundaries_buffer_y1_d;
    Real *recv_boundaries_buffer_z0_d;
    Real *recv_boundaries_buffer_z1_d;
    #endif

  } F;

  Potential_SOR_3D(void);

  void Initialize(Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real,
                  int ny_real, int nz_real, Real dx, Real dy, Real dz);

  void AllocateMemory_CPU(void);
  void AllocateMemory_GPU(void);
  void FreeMemory_GPU(void);
  void Reset(void);
  void Copy_Input(int n_cells, Real *input_d, Real *input_density_h, Real Grav_Constant, Real dens_avrg,
                  Real current_a);

  void Copy_Output(Real *output_potential);
  void Copy_Potential_From_Host(Real *output_potential);

  void Set_Boundaries();
  // Real Get_Potential( Real *input_density,  Real *output_potential, Real
  // Grav_Constant, Real dens_avrg, Real current_a ); void
  // Copy_Potential_From_Host( Real *potential_host );

  void Allocate_Array_GPU_Real(Real **array_dev, grav_int_t size);
  void Allocate_Array_GPU_bool(bool **array_dev, grav_int_t size);
  void Free_Array_GPU_Real(Real *array_dev);
  void Free_Array_GPU_bool(bool *array_dev);

  void Initialize_Potential(int nx, int ny, int nz, int n_ghost_potential, Real *potential_d, Real *density_d);
  void Copy_Input_And_Initialize(Real *input_density, const Real *input_potential, Real Grav_Constant, Real dens_avrg,
                                 Real current_a);

  void Poisson_iteration(int n_cells, int nx, int ny, int nz, int n_ghost_potential, Real dx, Real dy, Real dz,
                         Real omega, Real epsilon, Real *density_d, Real *potential_d, bool *converged_h,
                         bool *converged_d);
  void Poisson_iteration_Patial_1(int n_cells, int nx, int ny, int nz, int n_ghost_potential, Real dx, Real dy, Real dz,
                                  Real omega, Real epsilon, Real *density_d, Real *potential_d, bool *converged_h,
                                  bool *converged_d);
  void Poisson_iteration_Patial_2(int n_cells, int nx, int ny, int nz, int n_ghost_potential, Real dx, Real dy, Real dz,
                                  Real omega, Real epsilon, Real *density_d, Real *potential_d, bool *converged_h,
                                  bool *converged_d);
  void Poisson_Partial_Iteration(int n_step, Real omega, Real epsilon);

  void Load_Transfer_Buffer_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
                                int n_ghost_potential, Real *potential_d, Real *transfer_buffer_d);
  void Load_Transfer_Buffer_Half_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
                                     int n_ghost_potential, Real *potential_d, Real *transfer_buffer_d);
  void Load_Transfer_Buffer_GPU_x0();
  void Load_Transfer_Buffer_GPU_x1();
  void Load_Transfer_Buffer_GPU_y0();
  void Load_Transfer_Buffer_GPU_y1();
  void Load_Transfer_Buffer_GPU_z0();
  void Load_Transfer_Buffer_GPU_z1();
  void Unload_Transfer_Buffer_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
                                  int n_ghost_potential, Real *potential_d, Real *transfer_buffer_d);
  void Unload_Transfer_Buffer_Half_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
                                       int n_ghost_potential, Real *potential_d, Real *transfer_buffer_d);
  void Unload_Transfer_Buffer_GPU_x0();
  void Unload_Transfer_Buffer_GPU_x1();
  void Unload_Transfer_Buffer_GPU_y0();
  void Unload_Transfer_Buffer_GPU_y1();
  void Unload_Transfer_Buffer_GPU_z0();
  void Unload_Transfer_Buffer_GPU_z1();

  void Copy_Poisson_Boundary_Periodic(int direction, int side);

  void Copy_Poisson_Boundary_Open(int direction, int side);

  // void Load_Transfer_Buffer_GPU_All();
  // void Unload_Transfer_Buffer_GPU_All();

  void Copy_Transfer_Buffer_To_Host(int size_buffer, Real *transfer_bufer_h, Real *transfer_buffer_d);
  void Copy_Transfer_Buffer_To_Device(int size_buffer, Real *transfer_bufer_h, Real *transfer_buffer_d);

  void Set_Isolated_Boundary_Conditions(int *boundary_flags, struct Parameters *P);
  void Set_Isolated_Boundary_GPU(int direction, int side, Real *boundary_d);

    #ifdef MPI_CHOLLA
  bool Get_Global_Converged(bool converged_local);
    #endif
};

  #endif  // POTENTIAL_SOR_H
#endif    // GRAVITY

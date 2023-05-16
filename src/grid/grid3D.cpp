/*! \file grid3D.cpp
 *  \brief Definitions of the Grid3D class */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef HDF5
  #include <hdf5.h>
#endif
#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../grid/grid_enum.h"    // provides grid_enum
#include "../hydro/hydro_cuda.h"  // provides Calc_dt_GPU
#include "../integrators/VL_1D_cuda.h"
#include "../integrators/VL_2D_cuda.h"
#include "../integrators/VL_3D_cuda.h"
#include "../integrators/simple_1D_cuda.h"
#include "../integrators/simple_2D_cuda.h"
#include "../integrators/simple_3D_cuda.h"
#include "../io/io.h"
#include "../utils/error_handling.h"
#include "../utils/ran.h"
#ifdef MPI_CHOLLA
  #include <mpi.h>
  #ifdef HDF5
    #include <H5FDmpio.h>
  #endif
  #include "../mpi/mpi_routines.h"
#endif
#include <stdio.h>
#ifdef CLOUDY_COOL
  #include "../cooling/load_cloudy_texture.h"  // provides Load_Cuda_Textures and Free_Cuda_Textures
#endif

#ifdef PARALLEL_OMP
  #include "../utils/parallel_omp.h"
#endif

#ifdef COOLING_GPU
  #include "../cooling/cooling_cuda.h"  // provides Cooling_Update
#endif

#ifdef DUST
  #include "../dust/dust_cuda.h"  // provides Dust_Update
#endif

/*! \fn Grid3D(void)
 *  \brief Constructor for the Grid. */
Grid3D::Grid3D(void)
{
  // set initialization flag to 0
  flag_init = 0;

// set number of ghost cells
#ifdef PCM
  H.n_ghost = 2;
#endif  // PCM
#ifdef PLMP
  H.n_ghost = 3;
#endif  // PLMP
#ifdef PLMC
  H.n_ghost = 3;
#endif  // PLMC
#ifdef PPMP
  H.n_ghost = 4;
#endif  // PPMP
#ifdef PPMC
  H.n_ghost = 4;
#endif  // PPMC

#ifdef GRAVITY
  H.n_ghost_potential_offset = H.n_ghost - N_GHOST_POTENTIAL;
#endif

#ifdef MHD
  // Set the number of ghost cells high enough for MHD
  if (H.n_ghost < 3) {
    chprintf(
        "Insufficient number of ghost cells for MHD. H.n_ghost was %i, setting "
        "to 3.\n",
        H.n_ghost);
    H.n_ghost = 3;
  }
#endif  // MHD
}

/*! \fn void Get_Position(long i, long j, long k, Real *xpos, Real *ypos, Real
 * *zpos) \brief Get the cell-centered position based on cell index */
void Grid3D::Get_Position(long i, long j, long k, Real *x_pos, Real *y_pos, Real *z_pos)
{
#ifndef MPI_CHOLLA

  *x_pos = H.xbound + H.dx * (i - H.n_ghost) + 0.5 * H.dx;
  *y_pos = H.ybound + H.dy * (j - H.n_ghost) + 0.5 * H.dy;
  *z_pos = H.zbound + H.dz * (k - H.n_ghost) + 0.5 * H.dz;

#else /*MPI_CHOLLA*/

  /* position relative to local xyz bounds */
  /* This approach was replaced because it is less consistent for multiple
  cores. Since distributive property does not perfectly hold for floating point
  operations

  > Global_bound + global_i * dx

  is more consistent than

  >local_bound + local_i*dx = (global_bound + (global_i-local_i)*dx) +
  local_i*dx.

  *x_pos = H.xblocal + H.dx*(i-H.n_ghost) + 0.5*H.dx;
  *y_pos = H.yblocal + H.dy*(j-H.n_ghost) + 0.5*H.dy;
  *z_pos = H.zblocal + H.dz*(k-H.n_ghost) + 0.5*H.dz;
  */

  *x_pos = H.xbound + (nx_local_start + i - H.n_ghost) * H.dx + 0.5 * H.dx;
  *y_pos = H.ybound + (ny_local_start + j - H.n_ghost) * H.dy + 0.5 * H.dy;
  *z_pos = H.zbound + (nz_local_start + k - H.n_ghost) * H.dz + 0.5 * H.dz;

#endif /*MPI_CHOLLA*/
}

Real Grid3D::Calc_Inverse_Timestep()
{
  // ==Calculate the next inverse time step using Calc_dt_GPU from
  // hydro/hydro_cuda.h==
  return Calc_dt_GPU(C.device, H.nx, H.ny, H.nz, H.n_ghost, H.n_cells, H.dx, H.dy, H.dz, gama);
}

/*! \fn void Initialize(int nx_in, int ny_in, int nz_in)
 *  \brief Initialize the grid. */
void Grid3D::Initialize(struct parameters *P)
{
  // number of fields to track (default 5 is # of conserved variables)
  H.n_fields = 5;

// if including passive scalars increase the number of fields
#ifdef SCALAR
  H.n_fields += NSCALARS;
#endif

// if including magnetic fields increase the number of fields
#ifdef MHD
  H.n_fields += 3;
#endif  // MHD

// if using dual energy formalism must track internal energy - always the last
// field!
#ifdef DE
  H.n_fields++;
#endif

  int nx_in = P->nx;
  int ny_in = P->ny;
  int nz_in = P->nz;

  // Set the CFL coefficient (a global variable)
  C_cfl = 0.3;

#ifdef AVERAGE_SLOW_CELLS
  H.min_dt_slow = 1e-100;  // Initialize the minumum dt to a tiny number
#endif                     // AVERAGE_SLOW_CELLS

#ifndef MPI_CHOLLA

  // set grid dimensions
  H.nx      = nx_in + 2 * H.n_ghost;
  H.nx_real = nx_in;
  if (ny_in == 1)
    H.ny = 1;
  else
    H.ny = ny_in + 2 * H.n_ghost;
  H.ny_real = ny_in;
  if (nz_in == 1)
    H.nz = 1;
  else
    H.nz = nz_in + 2 * H.n_ghost;
  H.nz_real = nz_in;

  // set total number of cells
  H.n_cells = H.nx * H.ny * H.nz;

#else /*MPI_CHOLLA*/

  /* perform domain decomposition
   * and set grid dimensions
   * and allocate comm buffers */
  DomainDecomposition(P, &H, nx_in, ny_in, nz_in);

#endif /*MPI_CHOLLA*/

  // failsafe
  if (H.n_cells <= 0) {
    chprintf("Error initializing grid: H.n_cells = %d\n", H.n_cells);
    chexit(-1);
  }

  // check for initialization
  if (flag_init) {
    chprintf("Already initialized. Please reset.\n");
    return;
  } else {
    // mark that we are initializing
    flag_init = 1;
  }

  // Set header variables for time within the simulation
  H.t = 0.0;
  // and the number of timesteps taken
  H.n_step = 0;
  // and the wall time
  H.t_wall = 0.0;
  // and initialize the timestep
  H.dt = 0.0;

  // Set Transfer flag to false, only set to true before Conserved boundaries
  // are transferred
  H.TRANSFER_HYDRO_BOUNDARIES = false;

  // Set output to true when data has to be written to file;
  H.Output_Now = false;

  // allocate memory
  AllocateMemory();

#ifdef ROTATED_PROJECTION
  // x-dir pixels in projection
  R.nx = P->nxr;
  // z-dir pixels in projection
  R.nz = P->nzr;
  // minimum x location to project
  R.nx_min = 0;
  // minimum z location to project
  R.nz_min = 0;
  // maximum x location to project
  R.nx_max = R.nx;
  // maximum z location to project
  R.nz_max = R.nz;
  // rotation angle about z direction
  R.delta = M_PI * (P->delta / 180.);  // convert to radians
  // rotation angle about x direction
  R.theta = M_PI * (P->theta / 180.);  // convert to radians
  // rotation angle about y direction
  R.phi = M_PI * (P->phi / 180.);  // convert to radians
  // x-dir physical size of projection
  R.Lx = P->Lx;
  // z-dir physical size of projection
  R.Lz = P->Lz;
  // initialize a counter for rotated outputs
  R.i_delta = 0;
  // number of rotated outputs in a complete revolution
  R.n_delta = P->n_delta;
  // rate of rotation between outputs, for an actual simulation
  R.ddelta_dt = P->ddelta_dt;
  // are we not rotating about z(0)?
  // are we outputting multiple rotations(1)? or rotating during a
  // simulation(2)?
  R.flag_delta = P->flag_delta;
#endif /*ROTATED_PROJECTION*/

// Values for lower limit for density and temperature
#ifdef DENSITY_FLOOR
  H.density_floor = DENS_FLOOR;
#else
  H.density_floor     = 0.0;
#endif

#ifdef TEMPERATURE_FLOOR
  H.temperature_floor = TEMP_FLOOR;
#else
  H.temperature_floor = 0.0;
#endif

#ifdef COSMOLOGY
  if (P->scale_outputs_file[0] == '\0') {
    H.OUTPUT_SCALE_FACOR = false;
  } else {
    H.OUTPUT_SCALE_FACOR = true;
  }
#endif

  H.Output_Initial = true;
}

/*! \fn void AllocateMemory(void)
 *  \brief Allocate memory for the arrays. */
void Grid3D::AllocateMemory(void)
{
  // allocate memory for the conserved variable arrays
  // allocate all the memory to density, to insure contiguous memory
  CudaSafeCall(cudaHostAlloc((void **)&C.host, H.n_fields * H.n_cells * sizeof(Real), cudaHostAllocDefault));

  // point conserved variables to the appropriate locations
  C.density    = &(C.host[grid_enum::density * H.n_cells]);
  C.momentum_x = &(C.host[grid_enum::momentum_x * H.n_cells]);
  C.momentum_y = &(C.host[grid_enum::momentum_y * H.n_cells]);
  C.momentum_z = &(C.host[grid_enum::momentum_z * H.n_cells]);
  C.Energy     = &(C.host[grid_enum::Energy * H.n_cells]);
#ifdef SCALAR
  C.scalar = &(C.host[H.n_cells * grid_enum::scalar]);
  #ifdef BASIC_SCALAR
  C.basic_scalar = &(C.host[H.n_cells * grid_enum::basic_scalar]);
  #endif
  #ifdef DUST
  C.dust_density = &(C.host[H.n_cells * grid_enum::dust_density]);
  #endif
#endif  // SCALAR
#ifdef MHD
  C.magnetic_x = &(C.host[grid_enum::magnetic_x * H.n_cells]);
  C.magnetic_y = &(C.host[grid_enum::magnetic_y * H.n_cells]);
  C.magnetic_z = &(C.host[grid_enum::magnetic_z * H.n_cells]);
#endif  // MHD
#ifdef DE
  C.GasEnergy = &(C.host[(H.n_fields - 1) * H.n_cells]);
#endif  // DE

  // allocate memory for the conserved variable arrays on the device
  CudaSafeCall(cudaMalloc((void **)&C.device, H.n_fields * H.n_cells * sizeof(Real)));
  cuda_utilities::initGpuMemory(C.device, H.n_fields * H.n_cells * sizeof(Real));
  C.d_density    = C.device;
  C.d_momentum_x = &(C.device[H.n_cells]);
  C.d_momentum_y = &(C.device[2 * H.n_cells]);
  C.d_momentum_z = &(C.device[3 * H.n_cells]);
  C.d_Energy     = &(C.device[4 * H.n_cells]);
#ifdef SCALAR
  C.d_scalar = &(C.device[H.n_cells * grid_enum::scalar]);
  #ifdef BASIC_SCALAR
  C.d_basic_scalar = &(C.device[H.n_cells * grid_enum::basic_scalar]);
  #endif
  #ifdef DUST
  C.d_dust_density = &(C.device[H.n_cells * grid_enum::dust_density]);
  #endif
#endif  // SCALAR
#ifdef MHD
  C.d_magnetic_x = &(C.device[(grid_enum::magnetic_x)*H.n_cells]);
  C.d_magnetic_y = &(C.device[(grid_enum::magnetic_y)*H.n_cells]);
  C.d_magnetic_z = &(C.device[(grid_enum::magnetic_z)*H.n_cells]);
#endif  // MHD
#ifdef DE
  C.d_GasEnergy = &(C.device[(H.n_fields - 1) * H.n_cells]);
#endif  // DE

#if defined(GRAVITY)
  CudaSafeCall(cudaHostAlloc(&C.Grav_potential, H.n_cells * sizeof(Real), cudaHostAllocDefault));
  CudaSafeCall(cudaMalloc((void **)&C.d_Grav_potential, H.n_cells * sizeof(Real)));
#else
  C.Grav_potential    = NULL;
  C.d_Grav_potential  = NULL;
#endif

#ifdef CHEMISTRY_GPU
  C.HI_density    = &C.host[H.n_cells * grid_enum::HI_density];
  C.HII_density   = &C.host[H.n_cells * grid_enum::HII_density];
  C.HeI_density   = &C.host[H.n_cells * grid_enum::HeI_density];
  C.HeII_density  = &C.host[H.n_cells * grid_enum::HeII_density];
  C.HeIII_density = &C.host[H.n_cells * grid_enum::HeIII_density];
  C.e_density     = &C.host[H.n_cells * grid_enum::e_density];
#endif

  // initialize host array
  for (int i = 0; i < H.n_fields * H.n_cells; i++) {
    C.host[i] = 0.0;
  }

#ifdef CLOUDY_COOL
  Load_Cuda_Textures();
#endif  // CLOUDY_COOL
}

/*! \fn void set_dt(Real dti)
 *  \brief Set the timestep. */
void Grid3D::set_dt(Real dti)
{
  Real max_dti;

#ifdef CPU_TIME
  Timer.Calc_dt.Start();
#endif

#ifdef ONLY_PARTICLES
  // If only solving particles the time for hydro is set to a  large value,
  // that way the minimum dt is the one corresponding to particles
  H.dt = 1e10;

#else  // NOT ONLY_PARTICLES

  // dti is calculated before first loop and at the end of Update_Grid
  max_dti = dti;

  #ifdef MPI_CHOLLA
  // Note that this is the MPI_Allreduce for every iteration of the loop, not
  // just the first one
  max_dti = ReduceRealMax(max_dti);
  #endif /*MPI_CHOLLA*/

  H.dt = C_cfl / max_dti;

#endif  // ONLY_PARTICLES

#ifdef GRAVITY
  // Set dt for hydro and particles
  set_dt_Gravity();
#endif  // GRAVITY

#ifdef CPU_TIME
  Timer.Calc_dt.End();
#endif
}

/*! \fn void Update_Grid(void)
 *  \brief Update the conserved quantities in each cell. */
Real Grid3D::Update_Grid(void)
{
  Real max_dti = 0;
  int x_off, y_off, z_off;

  // set x, y, & z offsets of local CPU volume to pass to GPU
  // so global position on the grid is known
  x_off = y_off = z_off = 0;
#ifdef MPI_CHOLLA
  x_off = nx_local_start;
  y_off = ny_local_start;
  z_off = nz_local_start;
#endif

  // Set the lower limit for density and temperature (Internal Energy)
  Real U_floor, density_floor;
  density_floor = H.density_floor;
  // Minimum of internal energy from minumum of temperature
  U_floor = H.temperature_floor * KB / (gama - 1) / MU / MP / SP_ENERGY_UNIT;
#ifdef COSMOLOGY
  U_floor = H.temperature_floor / (gama - 1) / MP * KB * 1e-10;  // ( km/s )^2
  U_floor /= Cosmo.v_0_gas * Cosmo.v_0_gas / Cosmo.current_a / Cosmo.current_a;
#endif

#ifdef CPU_TIME
  Timer.Hydro_Integrator.Start();
#endif  // CPU_TIME

  // Run the hydro integrator on the grid
  if (H.nx > 1 && H.ny == 1 && H.nz == 1)  // 1D
  {
#ifdef CUDA
  #ifdef VL
    VL_Algorithm_1D_CUDA(C.device, H.nx, x_off, H.n_ghost, H.dx, H.xbound, H.dt, H.n_fields);
  #endif  // VL
  #ifdef SIMPLE
    Simple_Algorithm_1D_CUDA(C.device, H.nx, x_off, H.n_ghost, H.dx, H.xbound, H.dt, H.n_fields);
  #endif                                         // SIMPLE
#endif                                           // CUDA
  } else if (H.nx > 1 && H.ny > 1 && H.nz == 1)  // 2D
  {
#ifdef CUDA
  #ifdef VL
    VL_Algorithm_2D_CUDA(C.device, H.nx, H.ny, x_off, y_off, H.n_ghost, H.dx, H.dy, H.xbound, H.ybound, H.dt,
                         H.n_fields);
  #endif  // VL
  #ifdef SIMPLE
    Simple_Algorithm_2D_CUDA(C.device, H.nx, H.ny, x_off, y_off, H.n_ghost, H.dx, H.dy, H.xbound, H.ybound, H.dt,
                             H.n_fields);
  #endif                                        // SIMPLE
#endif                                          // CUDA
  } else if (H.nx > 1 && H.ny > 1 && H.nz > 1)  // 3D
  {
#ifdef CUDA
  #ifdef VL
    VL_Algorithm_3D_CUDA(C.device, C.d_Grav_potential, H.nx, H.ny, H.nz, x_off, y_off, z_off, H.n_ghost, H.dx, H.dy,
                         H.dz, H.xbound, H.ybound, H.zbound, H.dt, H.n_fields, density_floor, U_floor,
                         C.Grav_potential);
  #endif  // VL
  #ifdef SIMPLE
    Simple_Algorithm_3D_CUDA(C.device, C.d_Grav_potential, H.nx, H.ny, H.nz, x_off, y_off, z_off, H.n_ghost, H.dx, H.dy,
                             H.dz, H.xbound, H.ybound, H.zbound, H.dt, H.n_fields, density_floor, U_floor,
                             C.Grav_potential);
  #endif  // SIMPLE
#endif
  } else {
    chprintf("Error: Grid dimensions nx: %d  ny: %d  nz: %d  not supported.\n", H.nx, H.ny, H.nz);
    chexit(-1);
  }

#ifdef CPU_TIME
  Timer.Hydro_Integrator.End();
#endif  // CPU_TIME

#ifdef CUDA

  #ifdef COOLING_GPU
    #ifdef CPU_TIME
  Timer.Cooling_GPU.Start();
    #endif
  // ==Apply Cooling from cooling/cooling_cuda.h==
  Cooling_Update(C.device, H.nx, H.ny, H.nz, H.n_ghost, H.n_fields, H.dt, gama);
    #ifdef CPU_TIME
  Timer.Cooling_GPU.End();
    #endif

  #endif  // COOLING_GPU

  #ifdef DUST
  // ==Apply dust from dust/dust_cuda.h==
  Dust_Update(C.device, H.nx, H.ny, H.nz, H.n_ghost, H.n_fields, H.dt, gama);
  #endif  // DUST

  // Update the H and He ionization fractions and apply cooling and photoheating
  #ifdef CHEMISTRY_GPU
  Update_Chemistry();
    #ifdef CPU_TIME
  Timer.Chemistry.RecordTime(Chem.H.runtime_chemistry_step);
    #endif
  #endif

  #ifdef AVERAGE_SLOW_CELLS
  // Set the min_delta_t for averaging a slow cell
  Real max_dti_slow;
  max_dti_slow = 1 / H.min_dt_slow;
  Average_Slow_Cells(C.device, H.nx, H.ny, H.nz, H.n_ghost, H.n_fields, H.dx, H.dy, H.dz, gama, max_dti_slow);
  #endif  // AVERAGE_SLOW_CELLS

  // ==Calculate the next time step using Calc_dt_GPU from hydro/hydro_cuda.h==
  max_dti = Calc_Inverse_Timestep();

#endif  // CUDA

#ifdef COOLING_GRACKLE
  Cool.fields.density       = C.density;
  Cool.fields.HI_density    = &C.host[H.n_cells * grid_enum::HI_density];
  Cool.fields.HII_density   = &C.host[H.n_cells * grid_enum::HII_density];
  Cool.fields.HeI_density   = &C.host[H.n_cells * grid_enum::HeI_density];
  Cool.fields.HeII_density  = &C.host[H.n_cells * grid_enum::HeII_density];
  Cool.fields.HeIII_density = &C.host[H.n_cells * grid_enum::HeIII_density];
  Cool.fields.e_density     = &C.host[H.n_cells * grid_enum::e_density];

  #ifdef GRACKLE_METALS
  Cool.fields.metal_density = &C.host[H.n_cells * grid_enum::metal_density];
  #endif
#endif

#ifdef CHEMISTRY_GPU
  C.HI_density    = &C.host[H.n_cells * grid_enum::HI_density];
  C.HII_density   = &C.host[H.n_cells * grid_enum::HII_density];
  C.HeI_density   = &C.host[H.n_cells * grid_enum::HeI_density];
  C.HeII_density  = &C.host[H.n_cells * grid_enum::HeII_density];
  C.HeIII_density = &C.host[H.n_cells * grid_enum::HeIII_density];
  C.e_density     = &C.host[H.n_cells * grid_enum::e_density];
#endif

  return max_dti;
}

/*! \fn void Update_Hydro_Grid(void)
 *  \brief Do all steps to update the hydro. */
Real Grid3D::Update_Hydro_Grid()
{
#ifdef ONLY_PARTICLES
  // Don't integrate the Hydro when only solving for particles
  return 1e-10;
#endif  // ONLY_PARTICLES

  Real dti;

#ifdef CPU_TIME
  Timer.Hydro.Start();
#endif  // CPU_TIME

#ifdef GRAVITY
  // Extrapolate gravitational potential for hydro step
  Extrapolate_Grav_Potential();
#endif  // GRAVITY

  dti = Update_Grid();

#ifdef CPU_TIME
  #ifdef CHEMISTRY_GPU
  Timer.Hydro.Subtract(Chem.H.runtime_chemistry_step);
  // Subtract the time spent on the Chemical Update
  #endif  // CHEMISTRY_GPU
  Timer.Hydro.End();
#endif  // CPU_TIME

#ifdef COOLING_GRACKLE
  #ifdef CPU_TIME
  Timer.Cooling_Grackle.Start();
  #endif  // CPU_TIME
  Do_Cooling_Step_Grackle();
  #ifdef CPU_TIME
  Timer.Cooling_Grackle.End();
  #endif  // CPU_TIME
#endif    // COOLING_GRACKLE

  return dti;
}

void Grid3D::Update_Time()
{
  // update the time
  H.t += H.dt;

#ifdef PARTICLES
  Particles.t = H.t;

  #ifdef COSMOLOGY
  Cosmo.current_a += Cosmo.delta_a;
  Cosmo.current_z     = 1. / Cosmo.current_a - 1;
  Particles.current_a = Cosmo.current_a;
  Particles.current_z = Cosmo.current_z;
  Grav.current_a      = Cosmo.current_a;
  #endif  // COSMOLOGY
#endif    // PARTICLES

#if defined(ANALYSIS) && defined(COSMOLOGY)
  Analysis.current_z = Cosmo.current_z;
#endif
}

/*! \fn void Reset(void)
 *  \brief Reset the Grid3D class. */
void Grid3D::Reset(void)
{
  // free the memory
  FreeMemory();

  // reset the initialization flag
  flag_init = 0;
}

/*! \fn void FreeMemory(void)
 *  \brief Free the memory allocated by the Grid3D class. */
void Grid3D::FreeMemory(void)
{
  // free the conserved variable arrays
  CudaSafeCall(cudaFreeHost(C.host));

#ifdef GRAVITY
  CudaSafeCall(cudaFreeHost(C.Grav_potential));
  CudaSafeCall(cudaFree(C.d_Grav_potential));
#endif

// If memory is single allocated, free the memory at the end of the simulation.
#ifdef VL
  if (H.nx > 1 && H.ny == 1 && H.nz == 1) {
    Free_Memory_VL_1D();
  }
  if (H.nx > 1 && H.ny > 1 && H.nz == 1) {
    Free_Memory_VL_2D();
  }
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    Free_Memory_VL_3D();
  }
#endif  // VL
#ifdef SIMPLE
  if (H.nx > 1 && H.ny == 1 && H.nz == 1) {
    Free_Memory_Simple_1D();
  }
  if (H.nx > 1 && H.ny > 1 && H.nz == 1) {
    Free_Memory_Simple_2D();
  }
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    Free_Memory_Simple_3D();
  }
#endif  // SIMPLE

#ifdef GRAVITY
  Grav.FreeMemory_CPU();
  #ifdef GRAVITY_GPU
  Grav.FreeMemory_GPU();
  #endif
#endif

#ifdef PARTICLES
  Particles.Reset();
#endif

#ifdef COOLING_GRACKLE
  Cool.Free_Memory();
#endif

#ifdef COOLING_GPU
  #ifdef CLOUDY_COOL
  Free_Cuda_Textures();
  #endif
#endif

#ifdef CHEMISTRY_GPU
  Chem.Reset();
#endif

#ifdef ANALYSIS
  Analysis.Reset();
#endif
}

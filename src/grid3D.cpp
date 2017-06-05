/*! \file grid3D.cpp
 *  \brief Definitions of the Grid3D class */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef HDF5
#include <hdf5.h>
#endif
#include "global.h"
#include "grid3D.h"
#include "CTU_1D.h"
#include "CTU_2D.h"
#include "CTU_3D.h"
#include "CTU_1D_cuda.h"
#include "CTU_2D_cuda.h"
#include "CTU_3D_cuda.h"
#include "VL_1D_cuda.h"
#include "VL_2D_cuda.h"
#include "VL_3D_cuda.h"
#include "io.h"
#include "error_handling.h"
#ifdef MPI_CHOLLA
#include <mpi.h>
#ifdef HDF5
#include <H5FDmpio.h>
#endif
#include "mpi_routines.h"
#endif
#include <stdio.h>



/*! \fn Grid3D(void)
 *  \brief Constructor for the Grid. */
Grid3D::Grid3D(void)
{
  // set initialization flag to 0
  flag_init = 0;

  // set number of ghost cells
  #ifdef PCM
  H.n_ghost = 2;
  #endif //PCM
  #ifdef PLMP
  H.n_ghost = 3;
  #endif //PLMP
  #ifdef PLMC
  H.n_ghost = 3;
  #endif //PLMC
  #ifdef PPMP
  H.n_ghost = 4;
  #endif //PPMP
  #ifdef PPMC
  H.n_ghost=4;
  #endif //PPMC

}

/*! \fn void Get_Position(long i, long j, long k, Real *xpos, Real *ypos, Real *zpos)
 *  \brief Get the cell-centered position based on cell index */ 
void Grid3D::Get_Position(long i, long j, long k, Real *x_pos, Real *y_pos, Real *z_pos)
{

#ifndef   MPI_CHOLLA

  *x_pos = H.xbound + H.dx*(i-H.n_ghost) + 0.5*H.dx;
  *y_pos = H.ybound + H.dy*(j-H.n_ghost) + 0.5*H.dy;
  *z_pos = H.zbound + H.dz*(k-H.n_ghost) + 0.5*H.dz;

#else   /*MPI_CHOLLA*/

  /* position relative to local xyz bounds */
  *x_pos = H.xblocal + H.dx*(i-H.n_ghost) + 0.5*H.dx;
  *y_pos = H.yblocal + H.dy*(j-H.n_ghost) + 0.5*H.dy;
  *z_pos = H.zblocal + H.dz*(k-H.n_ghost) + 0.5*H.dz;
  

#endif  /*MPI_CHOLLA*/

}


/*! \fn void Initialize(int nx_in, int ny_in, int nz_in)
 *  \brief Initialize the grid. */
void Grid3D::Initialize(struct parameters *P)
{
  int nx_in = P->nx;
  int ny_in = P->ny;
  int nz_in = P->nz;

#ifndef MPI_CHOLLA

  // set grid dimensions
  H.nx = nx_in+2*H.n_ghost;
  H.nx_real = nx_in;
  if (ny_in == 1) H.ny = 1;
  else H.ny = ny_in+2*H.n_ghost;
  H.ny_real = ny_in;
  if (nz_in == 1) H.nz = 1;
  else H.nz = nz_in+2*H.n_ghost;
  H.nz_real = nz_in;

  // set total number of cells
  H.n_cells = H.nx * H.ny * H.nz;

#else  /*MPI_CHOLLA*/

  /* perform domain decomposition
   * and set grid dimensions      
   * and allocate comm buffers */ 
  DomainDecomposition(P, &H, nx_in, ny_in, nz_in); 
  
#endif /*MPI_CHOLLA*/

  // failsafe
  if(H.n_cells<=0)
  {
    chprintf("Error initializing grid: H.n_cells = %d\n", H.n_cells);
    chexit(-1);
  }

  // check for initilization
  if(flag_init)
  {
    chprintf("Already initialized. Please reset.\n");
    return;
  }
  else
  {
    // mark that we are initializing
    flag_init = 1;
  }

  // Set header variables for time within the simulation
  H.t = 0.0;
  // and the number of timesteps taken
  H.n_step = 0;
  // and the wall time
  H.t_wall = 0.0;
  // and inialize the timestep
  H.dt = 0.0;

  // allocate memory
  AllocateMemory();

}


/*! \fn void AllocateMemory(void)
 *  \brief Allocate memory for the arrays. */
void Grid3D::AllocateMemory(void)
{
  // number of fields to track (default 5 is # of conserved variables)
  int fields;
  fields = 5;

  // if using dual energy formalism must track internal energy
  #ifdef DE
  fields = 6;
  #endif

  // allocate memory for the conserved variable arrays
  // allocate all the memory to density, to insure contiguous memory
  C.density  = (Real *) malloc(fields*H.n_cells*sizeof(Real));
  // point momentum and Energy to the appropriate locations in density array
  C.momentum_x = &(C.density[H.n_cells]);
  C.momentum_y = &(C.density[2*H.n_cells]);
  C.momentum_z = &(C.density[3*H.n_cells]);
  C.Energy   = &(C.density[4*H.n_cells]);
  #ifdef DE
  C.GasEnergy = &(C.density[5*H.n_cells]);
  #endif

  // initialize array
  for (int i=0; i<fields*H.n_cells; i++)
  {
    C.density[i] = 0.0;
  }

}


/*! \fn void set_dt(Real C_cfl, Real dti)
 *  \brief Set the timestep. */
 void Grid3D::set_dt(Real C_cfl, Real dti)
{
  Real max_dti;

  if (H.n_step == 0) {
    max_dti = calc_dti_CPU(C_cfl);
  }
  else {
    #ifndef CUDA
    max_dti = calc_dti_CPU(C_cfl);
    #endif /*NO_CUDA*/
    #ifdef CUDA
    max_dti = dti;
    #endif /*CUDA*/
  }

  #ifdef   MPI_CHOLLA
  max_dti = ReduceRealMax(max_dti);
  #endif /*MPI_CHOLLA*/
  
  // new timestep
  H.dt = C_cfl / max_dti;

}


/*! \fn Real calc_dti_CPU(Real C_cfl)
 *  \brief Calculate the maximum inverse timestep, according to the CFL condition (Toro 6.17). */
Real Grid3D::calc_dti_CPU(Real C_cfl)
{
  int i, j, k, id;
  Real d_inv, vx, vy, vz, P, cs;
  Real max_vx, max_vy, max_vz;
  Real max_dti = 0.0;
  max_vx = max_vy = max_vz = 0.0;

  // 1D
  if (H.nx > 1 && H.ny == 1 && H.nz == 1) {
    //Find the maximum wave speed in the grid
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      id = i;
      d_inv = 1.0 / C.density[id];
      vx = d_inv * C.momentum_x[id];
      vy = d_inv * C.momentum_y[id];
      vz = d_inv * C.momentum_z[id];
      P = fmax((C.Energy[id] - 0.5*C.density[id]*(vx*vx + vy*vy + vz*vz) )*(gama-1.0), TINY_NUMBER);
      cs = sqrt(d_inv * gama * P);
      // compute maximum cfl velocity
      max_vx = fmax(max_vx, fabs(vx) + cs);
    }
    // compute max inverse of dt
    max_dti = max_vx / H.dx;
  }
  // 2D
  else if (H.nx > 1 && H.ny > 1 && H.nz == 1) {
    // Find the maximum wave speed in the grid
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {
      for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
        id = i + j*H.nx;
        d_inv = 1.0 / C.density[id];
        vx = d_inv * C.momentum_x[id];
        vy = d_inv * C.momentum_y[id];
        vz = d_inv * C.momentum_z[id];
        P = fmax((C.Energy[id] - 0.5*C.density[id]*(vx*vx + vy*vy + vz*vz) )*(gama-1.0), TINY_NUMBER);
        cs = sqrt(d_inv * gama * P);
        // compute maximum cfl velocity
        max_vx = fmax(max_vx, fabs(vx) + cs);
        max_vy = fmax(max_vy, fabs(vy) + cs);
      }
    }
    // compute max inverse of dt
    max_dti = max_vx / H.dx;
    max_dti = fmax(max_dti, max_vy / H.dy);
  }
  // 3D
  else if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    // Find the maximum wave speed in the grid
    for (i=0; i<H.nx-H.n_ghost; i++) {
      for (j=0; j<H.ny-H.n_ghost; j++) {
        for (k=0; k<H.nz-H.n_ghost; k++) {
          id = i + j*H.nx + k*H.nx*H.ny;
          d_inv = 1.0 / C.density[id];
          vx = d_inv * C.momentum_x[id];
          vy = d_inv * C.momentum_y[id];
          vz = d_inv * C.momentum_z[id];
          P = fmax((C.Energy[id] - 0.5*C.density[id]*(vx*vx + vy*vy + vz*vz) )*(gama-1.0), TINY_NUMBER);
          cs = sqrt(d_inv * gama * P);
          // compute maximum cfl velocity
          max_vx = fmax(max_vx, fabs(vx) + cs);
          max_vy = fmax(max_vy, fabs(vy) + cs);
          max_vz = fmax(max_vz, fabs(vz) + cs);
        }
      }
    }
    // compute max inverse of dt
    max_dti = max_vx / H.dx;
    max_dti = fmax(max_dti, max_vy / H.dy);
    max_dti = fmax(max_dti, max_vz / H.dy);
  } 
  else {
    chprintf("Invalid grid dimensions. Failed to compute dt.\n");
    chexit(0);
  }

  return max_dti;

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

  // Pass the structure of conserved variables to the CTU update functions
  // The function returns the updated variables
  if (H.nx > 1 && H.ny == 1 && H.nz == 1) //1D
  {
    #ifndef CUDA
    #ifndef VL
    CTU_Algorithm_1D(&(C.density[0]), H.nx, H.n_ghost, H.dx, H.dt);
    #endif //not_VL
    #ifdef VL
    chprintf("VL algorithm not implemented in non-cuda version.");
    chexit(1);
    #endif //VL
    #endif //not_CUDA

    #ifdef CUDA
    #ifndef VL
    max_dti = CTU_Algorithm_1D_CUDA(&(C.density[0]), H.nx, x_off, H.n_ghost, H.dx, H.xbound, H.dt);
    #endif //not_VL
    #ifdef VL
    max_dti = VL_Algorithm_1D_CUDA(&(C.density[0]), H.nx, x_off, H.n_ghost, H.dx, H.xbound, H.dt);
    #endif //VL
    #endif //CUDA
  }
  else if (H.nx > 1 && H.ny > 1 && H.nz == 1) //2D
  {
    #ifndef CUDA
    #ifndef VL
    CTU_Algorithm_2D(&(C.density[0]), H.nx, H.ny, H.n_ghost, H.dx, H.dy, H.dt);
    #endif //not_VL
    #ifdef VL
    chprintf("VL algorithm not implemented in non-cuda version.");
    chexit(1);    
    #endif //VL
    #endif //not_CUDA

    #ifdef CUDA
    #ifndef VL
    max_dti = CTU_Algorithm_2D_CUDA(&(C.density[0]), H.nx, H.ny, x_off, y_off, H.n_ghost, H.dx, H.dy, H.xbound, H.ybound, H.dt);
    #endif //not_VL
    #ifdef VL
    max_dti = VL_Algorithm_2D_CUDA(&(C.density[0]), H.nx, H.ny, x_off, y_off, H.n_ghost, H.dx, H.dy, H.xbound, H.ybound, H.dt);
    #endif //VL
    #endif //CUDA
  }
  else if (H.nx > 1 && H.ny > 1 && H.nz > 1) //3D
  {
    #ifndef CUDA
    #ifndef VL
    CTU_Algorithm_3D(&(C.density[0]), H.nx, H.ny, H.nz, H.n_ghost, H.dx, H.dy, H.dz, H.dt);
    #endif //not_VL
    #ifdef VL
    chprintf("VL algorithm not implemented in non-cuda version.");
    chexit(1);    
    #endif //VL
    #endif //not_CUDA

    #ifdef CUDA
    #ifndef VL
    max_dti = CTU_Algorithm_3D_CUDA(&(C.density[0]), H.nx, H.ny, H.nz, x_off, y_off, z_off, H.n_ghost, H.dx, H.dy, H.dz, H.xbound, H.ybound, H.zbound, H.dt);
    #endif //not_VL
    #ifdef VL
    max_dti = VL_Algorithm_3D_CUDA(&(C.density[0]), H.nx, H.ny, H.nz, x_off, y_off, z_off, H.n_ghost, H.dx, H.dy, H.dz, H.xbound, H.ybound, H.zbound, H.dt);
    #endif //VL
    #endif    
  }
  else
  {
    chprintf("Error: Grid dimensions nx: %d  ny: %d  nz: %d  not supported.\n", H.nx, H.ny, H.nz);
    chexit(1);
  }


  return max_dti;
}



void Grid3D::Add_Supernovae(void)
{
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r, R_s, f;
  Real M_dot, E_dot, V, rho_dot, Ed_dot;
  R_s = 0.3; // starburst radius, in kpc
  M_dot = 2.0e3; // mass input rate, in M_sun / kyr
  E_dot = 1.0e42; // energy input rate, in erg/s
  E_dot = E_dot*TIME_UNIT/(MASS_UNIT*VELOCITY_UNIT*VELOCITY_UNIT); // convert to code units
  V = (4.0/3.0)*PI*R_s*R_s*R_s;
  f = H.dx*H.dy*H.dz / V;
  rho_dot = f * M_dot / V;
  Ed_dot = f * E_dot / V;
  //printf("rho_dot: %f  Ed_dot: %f\n", rho_dot, Ed_dot);

  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;

        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate spherical radius
        r = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);

        // within starburst radius, inject energy and momentum
        if (r < R_s) {
          C.density[id] += rho_dot * H.dt;
          C.Energy[id] += Ed_dot * H.dt;
        }
      }
    }
  }


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
  // free the conserved variable array
  free(C.density);

}

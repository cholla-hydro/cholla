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
#include <H5FDmpio.h>
#include "mpi_routines.h"
#endif
#include <stdio.h>
#ifdef COOLING_GPU
#include "cooling_wrapper.h"
#endif
//#include "rng.h"




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
  
  //printf("i %d io %d xbl %e dx %e xp %e\n",i,i-H.n_ghost,H.xblocal,H.dx,*x_pos);
  //fflush(stdout);

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

  #ifdef COOLING_CPU
  // Load cooling tables
  Load_Cooling_Tables();
  #endif
  #ifdef COOLING_GPU
  Load_Cuda_Textures();
  #endif

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
    max_dti = CTU_Algorithm_1D_CUDA(&(C.density[0]), H.nx, H.n_ghost, H.dx, H.dt);
    #endif //not_VL
    #ifdef VL
    max_dti = VL_Algorithm_1D_CUDA(&(C.density[0]), H.nx, H.n_ghost, H.dx, H.dt);
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
    max_dti = CTU_Algorithm_2D_CUDA(&(C.density[0]), H.nx, H.ny, H.n_ghost, H.dx, H.dy, H.dt);
    #endif //not_VL
    #ifdef VL
    max_dti = VL_Algorithm_2D_CUDA(&(C.density[0]), H.nx, H.ny, H.n_ghost, H.dx, H.dy, H.dt);
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
    max_dti = CTU_Algorithm_3D_CUDA(&(C.density[0]), H.nx, H.ny, H.nz, H.n_ghost, H.dx, H.dy, H.dz, H.dt);
    #endif //not_VL
    #ifdef VL
    max_dti = VL_Algorithm_3D_CUDA(&(C.density[0]), H.nx, H.ny, H.nz, H.n_ghost, H.dx, H.dy, H.dz, H.dt);
    #endif //VL
    #endif    
  }
  else
  {
    chprintf("Error: Grid dimensions nx: %d  ny: %d  nz: %d  not supported.\n", H.nx, H.ny, H.nz);
    chexit(1);
  }

  #ifdef COOLING_CPU
  // Use Cloudy tables to apply cooling
  Cool_CPU();
  #endif //COOLING_CPU

  return max_dti;
}



/*! \fn Apply_Forcing(void)
 *  \brief Apply a forcing field to continuously generate turbulence. */
void Grid3D::Apply_Forcing(void)
{
 /* 
  int i, j, k, id, fid;
  int n_cells = H.nx_real * H.ny_real * H.nz_real;  
  int istart, jstart, kstart, iend, jend, kend;
  Real x_pos, y_pos, z_pos;
  Real *A1, *A2, *A3, *A4, *B1, *B2, *B3, *B4;
  Real *vxp, *vyp, *vzp;
  Real *p;
  Real mxp, myp, mzp;
  Real vx_av, vy_av, vz_av, v_av;
  Real vxsq, vysq, vzsq;
  Real mxsq, mysq, mzsq;
  Real mx, my, mz;
  Real M, d_tot;  
  A1 = rng_direction(3);
  A2 = rng_direction(3);
  A3 = rng_direction(3);
  A4 = rng_direction(3);
  B1 = rng_direction(3);
  B2 = rng_direction(3);
  B3 = rng_direction(3);
  B4 = rng_direction(3);
  p = rng_direction(3);
  vxp = (Real *) malloc(n_cells*sizeof(Real));
  vyp = (Real *) malloc(n_cells*sizeof(Real));
  vzp = (Real *) malloc(n_cells*sizeof(Real));
  
  //printf("%f %f %f\n", A1[0], A1[1], A1[2]);
  //printf("%f %f %f\n", B1[0], B1[1], B1[2]);
  //printf("%f %f %f\n", A2[0], A2[1], A2[2]);
  //printf("%f %f %f\n", B2[0], B2[1], B2[2]);

  //set the desired Mach number
  M = 2.0;
  d_tot = 0;


  istart = H.n_ghost;
  iend   = H.nx-H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny-H.n_ghost;
  }
  else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz-H.n_ghost;
  }
  else {
    kstart = 0;
    kend   = H.nz;
  }

  mxp = myp = mzp = 0.0;
  mx = my = mz = 0.0;
  vx_av = vy_av = vz_av = 0.0;
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;
        fid = (i-H.n_ghost) + (j-H.n_ghost)*H.nx_real + (k-H.n_ghost)*H.nx_real*H.ny_real;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate velocity perturbations
        vxp[fid] = B1[0]*sin( (2*PI/sqrt(A1[0]*A1[0] + A1[1]*A1[1] + A1[2]*A1[2])) * (A1[0]*(x_pos+p[0]) + A1[1]*(y_pos+p[1]) + A1[2]*(z_pos+p[2])));
        vyp[fid] = B1[1]*sin( (2*PI/sqrt(A1[0]*A1[0] + A1[1]*A1[1] + A1[2]*A1[2])) * (A1[0]*(x_pos+p[0]) + A1[1]*(y_pos+p[1]) + A1[2]*(z_pos+p[2])));
        vzp[fid] = B1[2]*sin( (2*PI/sqrt(A1[0]*A1[0] + A1[1]*A1[1] + A1[2]*A1[2])) * (A1[0]*(x_pos+p[0]) + A1[1]*(y_pos+p[1]) + A1[2]*(z_pos+p[2])));
        vxp[fid] += B2[0]*sin( (4*PI/sqrt(A2[0]*A2[0] + A2[1]*A2[1] + A2[2]*A2[2])) * (A2[0]*(x_pos+p[0]) + A2[1]*(y_pos+p[1]) + A2[2]*(z_pos+p[2])));
        vyp[fid] += B2[1]*sin( (4*PI/sqrt(A2[0]*A2[0] + A2[1]*A2[1] + A2[2]*A2[2])) * (A2[0]*(x_pos+p[0]) + A2[1]*(y_pos+p[1]) + A2[2]*(z_pos+p[2])));
        vzp[fid] += B2[2]*sin( (4*PI/sqrt(A2[0]*A2[0] + A2[1]*A2[1] + A2[2]*A2[2])) * (A2[0]*(x_pos+p[0]) + A2[1]*(y_pos+p[1]) + A2[2]*(z_pos+p[2])));

        vxp[fid] += B3[0]*cos( (2*PI/sqrt(A3[0]*A3[0] + A3[1]*A3[1] + A3[2]*A3[2])) * (A3[0]*(x_pos+p[0]) + A3[1]*(y_pos+p[1]) + A3[2]*(z_pos+p[2])));
        vyp[fid] += B3[1]*cos( (2*PI/sqrt(A3[0]*A3[0] + A3[1]*A3[1] + A3[2]*A3[2])) * (A3[0]*(x_pos+p[0]) + A3[1]*(y_pos+p[1]) + A3[2]*(z_pos+p[2])));
        vzp[fid] += B3[2]*cos( (2*PI/sqrt(A3[0]*A3[0] + A3[1]*A3[1] + A3[2]*A3[2])) * (A3[0]*(x_pos+p[0]) + A3[1]*(y_pos+p[1]) + A3[2]*(z_pos+p[2])));
        vxp[fid] += B4[0]*cos( (4*PI/sqrt(A4[0]*A4[0] + A4[1]*A4[1] + A4[2]*A4[2])) * (A4[0]*(x_pos+p[0]) + A4[1]*(y_pos+p[1]) + A4[2]*(z_pos+p[2])));
        vyp[fid] += B4[1]*cos( (4*PI/sqrt(A4[0]*A4[0] + A4[1]*A4[1] + A4[2]*A4[2])) * (A4[0]*(x_pos+p[0]) + A4[1]*(y_pos+p[1]) + A4[2]*(z_pos+p[2])));
        vzp[fid] += B4[2]*cos( (4*PI/sqrt(A4[0]*A4[0] + A4[1]*A4[1] + A4[2]*A4[2])) * (A4[0]*(x_pos+p[0]) + A4[1]*(y_pos+p[1]) + A4[2]*(z_pos+p[2])));
        
        // calculate momentum of forcing field
        mxp = C.density[id] * vxp[fid];
        myp = C.density[id] * vyp[fid];
        mzp = C.density[id] * vzp[fid];

        // track total momentum
        mx += mxp;
        my += myp;
        mz += mzp;

        d_tot += C.density[id];
      }
    }
  }
  // calculate density weighted average velocity of forcing field
  vx_av = mx / d_tot;
  vy_av = my / d_tot;
  vz_av = mz / d_tot;
  //printf("%f %f %f\n", vx_av, vy_av, vz_av);

  mxsq = mysq = mzsq = 0.0;
  mx = my = mz = 0.0;
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        //get cell index
        id = i + j*H.nx + k*H.nx*H.ny;
        fid = (i-H.n_ghost) + (j-H.n_ghost)*H.nx_real + (k-H.n_ghost)*H.nx_real*H.ny_real;

        // get cell-centered position
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

        // subtract off average velocity to create a field with zero net momentum
        vxp[fid] -= vx_av; 
        vyp[fid] -= vy_av; 
        vzp[fid] -= vz_av; 

        // calculate momentum of forcing field
        mxp = C.density[id]*vxp[fid];
        myp = C.density[id]*vyp[fid];
        mzp = C.density[id]*vzp[fid];

        // track total momentum
        mx += C.momentum_x[id] + mxp;
        my += C.momentum_y[id] + myp;
        mz += C.momentum_z[id] + mzp;

        // calculate <v^2> for each direction
        mxsq += mxp*mxp/C.density[id];
        mysq += myp*myp/C.density[id];
        mzsq += mzp*mzp/C.density[id];

      }
    }
  }
  vx_av = sqrt(mxsq / d_tot);
  vy_av = sqrt(mysq / d_tot);
  vz_av = sqrt(mzsq / d_tot);
  v_av = sqrt(vx_av*vx_av + vy_av*vy_av + vz_av*vz_av);
  //printf("%f %f %f %f\n", vx_av, vy_av, vz_av, v_av);
  //printf("%f %f %f\n", mx, my, mz);   

  mx = my = mz = 0.0;
  mxp = myp = mzp = 0.0;
  mxsq = mysq = mzsq = 0.0;
  for(k=kstart; k<kend; k++) {
    for(j=jstart; j<jend; j++) {
      for(i=istart; i<iend; i++) {

        // get cell index
        id = i + j*H.nx + k*H.nx*H.ny;
        fid = (i-H.n_ghost) + (j-H.n_ghost)*H.nx_real + (k-H.n_ghost)*H.nx_real*H.ny_real;
        
        // rescale velocities to get desired Mach number
        // only apply a tenth of initial energy since forcing is 
        // applied every tenth of a crossing time
        vxp[fid] *= sqrt(0.1*M*M/3.0) / vx_av;
        vyp[fid] *= sqrt(0.1*M*M/3.0) / vy_av;
        vzp[fid] *= sqrt(0.1*M*M/3.0) / vz_av;

        // calculate momentum perturbations and apply
        mxp = C.density[id]*vxp[fid];
        myp = C.density[id]*vyp[fid];
        mzp = C.density[id]*vzp[fid];
        C.momentum_x[id] += mxp;
        C.momentum_y[id] += myp;
        C.momentum_z[id] += mzp;
        C.Energy[id] += 0.5*(mxp*mxp + myp*myp + mzp*mzp)/C.density[id];        

        // track total momentum
        mx += C.momentum_x[id];
        my += C.momentum_y[id];
        mz += C.momentum_z[id];

        // calcultate <v^2> for each direction
        mxsq += C.momentum_x[id]*C.momentum_x[id]/C.density[id];
        mysq += C.momentum_y[id]*C.momentum_y[id]/C.density[id];
        mzsq += C.momentum_z[id]*C.momentum_z[id]/C.density[id];        
        
      }
    }
  }
  vx_av = sqrt(mxsq / d_tot);
  vy_av = sqrt(mysq / d_tot);
  vz_av = sqrt(mzsq / d_tot);
  v_av = sqrt(vx_av*vx_av + vy_av*vy_av + vz_av*vz_av);
  printf("%f %f %f %f\n", vx_av, vy_av, vz_av, v_av);
  //printf("%f %f %f\n", mx, my, mz);  


  free(vxp);
  free(vyp);
  free(vzp);
  free(A1);
  free(A2);
  free(A3);
  free(A4);
  free(B1);
  free(B2);
  free(B3);
  free(B4);
  free(p);

*/
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

  #ifdef COOLING_CPU
  Free_Cooling_Tables();
  #endif
  #ifdef COOLING_GPU
  Free_Cuda_Textures();
  #endif

}

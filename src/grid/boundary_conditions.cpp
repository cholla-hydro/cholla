/*! \file boundary_conditions.cpp
 *  \brief Definitions of the boundary conditions for various tests.
           Functions are members of the Grid3D class. */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../utils/error_handling.h"
#include "../mpi/mpi_routines.h"

#include "../grid/cuda_boundaries.h" // provides SetGhostCells


/*! \fn void Set_Boundary_Conditions_Grid(parameters P)
 *  \brief Set the boundary conditions for all components based on info in the parameters structure. */
void Grid3D::Set_Boundary_Conditions_Grid( parameters P){

  #ifndef ONLY_PARTICLES
  // Dont transfer Hydro boundaries when only doing particles

  // Transfer Hydro Conserved boundaries
  #ifdef CPU_TIME
  Timer.Boundaries.Start();
  #endif //CPU_TIME
  H.TRANSFER_HYDRO_BOUNDARIES = true;
  Set_Boundary_Conditions(P);
  H.TRANSFER_HYDRO_BOUNDARIES = false;
  #ifdef CPU_TIME
  Timer.Boundaries.End();
  #endif //CPU_TIME
  #endif //ONLY_PARTICLES

  // If the Gravity coupling is on the CPU, the potential is not in the Conserved arrays,
  // and its boundaries need to be transferred separately
  #ifdef GRAVITY
  #ifdef CPU_TIME
  Timer.Pot_Boundaries.Start();
  #endif
  Grav.TRANSFER_POTENTIAL_BOUNDARIES = true;
  Set_Boundary_Conditions(P);
  Grav.TRANSFER_POTENTIAL_BOUNDARIES = false;
  #ifdef CPU_TIME
  Timer.Pot_Boundaries.End();
  #endif
  #endif
}

/*! \fn void Set_Boundary_Conditions(parameters P)
 *  \brief Set the boundary conditions based on info in the parameters structure. */
void Grid3D::Set_Boundary_Conditions(parameters P) {

  //Check Only one boundary type id being transferred
  int n_bounds = 0;
  n_bounds += (int) H.TRANSFER_HYDRO_BOUNDARIES;
  #ifdef GRAVITY
  n_bounds += (int) Grav.TRANSFER_POTENTIAL_BOUNDARIES;
  #ifdef SOR
  n_bounds += (int) Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES;
  #endif //SOR
  #endif //GRAVITY
  #ifdef PARTICLES
  n_bounds += (int) Particles.TRANSFER_PARTICLES_BOUNDARIES;
  n_bounds += (int) Particles.TRANSFER_DENSITY_BOUNDARIES;
  #endif  //PARTICLES

  if ( n_bounds > 1 ){
    printf("ERROR: More than one boundary type for transfer. N boundary types: %d\n", n_bounds );
    printf(" Boundary Hydro: %d\n", (int) H.TRANSFER_HYDRO_BOUNDARIES );
    #ifdef GRAVITY
    printf(" Boundary Potential: %d\n", (int) Grav.TRANSFER_POTENTIAL_BOUNDARIES );
    #ifdef SOR
    printf(" Boundary Poisson: %d\n", (int) Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES );
    #endif //SOR
    #endif //GRAVITY
    #ifdef PARTICLES
    printf(" Boundary Particles: %d\n", (int) Particles.TRANSFER_PARTICLES_BOUNDARIES );
    printf(" Boundary Particles Density: %d\n", (int) Particles.TRANSFER_DENSITY_BOUNDARIES );
    #endif //PARTICLES
    exit(-1);
  }

  // If no boundaries are set to be transferred then exit;
  if ( n_bounds == 0 ){
     printf( " Warning: No boundary type for transfer \n");
     return;
  }


#ifndef MPI_CHOLLA

  int flags[6] = {0,0,0,0,0,0};

  // Check for custom boundary conditions and set boundary flags
  if(Check_Custom_Boundary(&flags[0], P))
  {
    Custom_Boundary(P.custom_bcnd);
  }

  // set regular boundaries
  if(H.nx>1) {
    Set_Boundaries(0, flags);
    Set_Boundaries(1, flags);
  }
  if(H.ny>1) {
    Set_Boundaries(2, flags);
    Set_Boundaries(3, flags);
  }
  if(H.nz>1) {
    Set_Boundaries(4, flags);
    Set_Boundaries(5, flags);
  }

  #ifdef GRAVITY
  Grav.Set_Boundary_Flags( flags );
  #endif  //Gravity

#else  /*MPI_CHOLLA*/

  /*Set boundaries, including MPI exchanges*/

  Set_Boundaries_MPI(P);

#endif /*MPI_CHOLLA*/
}


/*! \fn int Check_Custom_Boundary(int *flags, struct parameters P)
 *  \brief Check for custom boundary conditions and set boundary flags. */
int Grid3D::Check_Custom_Boundary(int *flags, struct parameters P)
{

  /*check if any boundary is a custom boundary*/
  /*if yes, then return 1*/
  /*if no, then return 0*/
  /*additionally, set a flag for each boundary*/

  if(H.nx>1)
  {
    *(flags+0) = P.xl_bcnd;
    *(flags+1) = P.xu_bcnd;
  }
  if(H.ny>1)
  {
    *(flags+2) = P.yl_bcnd;
    *(flags+3) = P.yu_bcnd;
  }
  if(H.nz>1)
  {
    *(flags+4) = P.zl_bcnd;
    *(flags+5) = P.zu_bcnd;
  }

  for (int i=0; i<6; i++)
  {
    if (!( (flags[i]>=0)&&(flags[i]<=5) ) )
    {
      chprintf("Invalid boundary conditions. Must select between 1 (periodic), 2 (reflective), 3 (transmissive), 4 (custom), 5 (mpi).\n");
      chexit(-1);
    }
    if (flags[i] == 4) {
      /*custom boundaries*/
      return 1;
    }
  }
  /*no custom boundaries*/
  return 0;
}



/*! \fn void Set_Boundaries(int dir, int flags[])
 *  \brief Apply boundary conditions to the grid. */
void Grid3D::Set_Boundaries(int dir, int flags[])
{
  int i, j, k;
  int imin[3] = {0,0,0};
  int imax[3] = {H.nx,H.ny,H.nz};
  Real a[3]   = {1,1,1};  //sign of momenta
  int idx;    //index of a real cell
  int gidx;   //index of a ghost cell

  int nPB, nBoundaries;
  int *iaBoundary, *iaCell;

  /*if the cell face is an custom boundary, exit */
  if(flags[dir]==4)
    return;

#ifdef   MPI_CHOLLA
  /*if the cell face is an mpi boundary, exit */
  if(flags[dir]==5)
    return;
#endif /*MPI_CHOLLA*/



  #ifdef GRAVITY

  if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
    if ( flags[dir] == 1 ){
      // Set Periodic Boundaries for the ghost cells.
      #ifdef GRAVITY_GPU
      if ( dir == 0 ) Set_Potential_Boundaries_Periodic_GPU( 0, 0, flags );
      if ( dir == 1 ) Set_Potential_Boundaries_Periodic_GPU( 0, 1, flags );
      if ( dir == 2 ) Set_Potential_Boundaries_Periodic_GPU( 1, 0, flags );
      if ( dir == 3 ) Set_Potential_Boundaries_Periodic_GPU( 1, 1, flags );
      if ( dir == 4 ) Set_Potential_Boundaries_Periodic_GPU( 2, 0, flags );
      if ( dir == 5 ) Set_Potential_Boundaries_Periodic_GPU( 2, 1, flags );
      #else
      if ( dir == 0 ) Set_Potential_Boundaries_Periodic( 0, 0, flags );
      if ( dir == 1 ) Set_Potential_Boundaries_Periodic( 0, 1, flags );
      if ( dir == 2 ) Set_Potential_Boundaries_Periodic( 1, 0, flags );
      if ( dir == 3 ) Set_Potential_Boundaries_Periodic( 1, 1, flags );
      if ( dir == 4 ) Set_Potential_Boundaries_Periodic( 2, 0, flags );
      if ( dir == 5 ) Set_Potential_Boundaries_Periodic( 2, 1, flags );
      #endif
    }
    if ( flags[dir] == 3 ){

      #ifdef GRAVITY_GPU
      if ( dir == 0 ) Set_Potential_Boundaries_Isolated_GPU( 0, 0, flags );
      if ( dir == 1 ) Set_Potential_Boundaries_Isolated_GPU( 0, 1, flags );
      if ( dir == 2 ) Set_Potential_Boundaries_Isolated_GPU( 1, 0, flags );
      if ( dir == 3 ) Set_Potential_Boundaries_Isolated_GPU( 1, 1, flags );
      if ( dir == 4 ) Set_Potential_Boundaries_Isolated_GPU( 2, 0, flags );
      if ( dir == 5 ) Set_Potential_Boundaries_Isolated_GPU( 2, 1, flags );
      #else
      if ( dir == 0 ) Set_Potential_Boundaries_Isolated( 0, 0, flags );
      if ( dir == 1 ) Set_Potential_Boundaries_Isolated( 0, 1, flags );
      if ( dir == 2 ) Set_Potential_Boundaries_Isolated( 1, 0, flags );
      if ( dir == 3 ) Set_Potential_Boundaries_Isolated( 1, 1, flags );
      if ( dir == 4 ) Set_Potential_Boundaries_Isolated( 2, 0, flags );
      if ( dir == 5 ) Set_Potential_Boundaries_Isolated( 2, 1, flags );
      #endif//GRAVITY_GPU
    }
    return;
  }
  #ifdef SOR
  if ( Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES ){
    if ( flags[dir] ==1 ){
      if ( dir == 0 ) Grav.Poisson_solver.Copy_Poisson_Boundary_Periodic( 0, 0 );
      if ( dir == 1 ) Grav.Poisson_solver.Copy_Poisson_Boundary_Periodic( 0, 1 );
      if ( dir == 2 ) Grav.Poisson_solver.Copy_Poisson_Boundary_Periodic( 1, 0 );
      if ( dir == 3 ) Grav.Poisson_solver.Copy_Poisson_Boundary_Periodic( 1, 1 );
      if ( dir == 4 ) Grav.Poisson_solver.Copy_Poisson_Boundary_Periodic( 2, 0 );
      if ( dir == 5 ) Grav.Poisson_solver.Copy_Poisson_Boundary_Periodic( 2, 1 );
    }
    return;
  }
  #endif //SOR
  #endif //GRAVITY

  #ifdef PARTICLES
  if ( Particles.TRANSFER_DENSITY_BOUNDARIES ){
    if ( flags[dir] ==1 ){
      // Set Periodic Boundaries for the particles density.
      #ifdef PARTICLES_GPU
      if ( dir == 0 ) Set_Particles_Density_Boundaries_Periodic_GPU( 0, 0 );
      if ( dir == 1 ) Set_Particles_Density_Boundaries_Periodic_GPU( 0, 1 );
      if ( dir == 2 ) Set_Particles_Density_Boundaries_Periodic_GPU( 1, 0 );
      if ( dir == 3 ) Set_Particles_Density_Boundaries_Periodic_GPU( 1, 1 );
      if ( dir == 4 ) Set_Particles_Density_Boundaries_Periodic_GPU( 2, 0 );
      if ( dir == 5 ) Set_Particles_Density_Boundaries_Periodic_GPU( 2, 1 );
      #endif
      #ifdef PARTICLES_CPU
      if ( dir == 0 ) Set_Particles_Density_Boundaries_Periodic( 0, 0 );
      if ( dir == 1 ) Set_Particles_Density_Boundaries_Periodic( 0, 1 );
      if ( dir == 2 ) Set_Particles_Density_Boundaries_Periodic( 1, 0 );
      if ( dir == 3 ) Set_Particles_Density_Boundaries_Periodic( 1, 1 );
      if ( dir == 4 ) Set_Particles_Density_Boundaries_Periodic( 2, 0 );
      if ( dir == 5 ) Set_Particles_Density_Boundaries_Periodic( 2, 1 );
      #endif
    }
    return;
  }
  #endif  //PARTICLES

  #ifdef PARTICLES
  if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ){
    if ( flags[dir] ==1 ){
      #ifdef PARTICLES_CPU
      if ( dir == 0 ) Set_Particles_Boundary( 0, 0 );
      if ( dir == 1 ) Set_Particles_Boundary( 0, 1 );
      if ( dir == 2 ) Set_Particles_Boundary( 1, 0 );
      if ( dir == 3 ) Set_Particles_Boundary( 1, 1 );
      if ( dir == 4 ) Set_Particles_Boundary( 2, 0 );
      if ( dir == 5 ) Set_Particles_Boundary( 2, 1 );
      #endif//PARTICLES_CPU

      #ifdef PARTICLES_GPU
      if ( dir == 0 ) Set_Particles_Boundary_GPU( 0, 0 );
      if ( dir == 1 ) Set_Particles_Boundary_GPU( 0, 1 );
      if ( dir == 2 ) Set_Particles_Boundary_GPU( 1, 0 );
      if ( dir == 3 ) Set_Particles_Boundary_GPU( 1, 1 );
      if ( dir == 4 ) Set_Particles_Boundary_GPU( 2, 0 );
      if ( dir == 5 ) Set_Particles_Boundary_GPU( 2, 1 );
      #endif//PARTICLES_GPU


      } else if (flags[dir] == 3) {
        #ifdef PARTICLES_CPU
        Set_Particles_Open_Boundary(dir/2, dir%2);
        #endif  //PARTICLES_CPU
    }
    return;
  }
  #endif//PARTICLES

  //get the extents of the ghost region we are initializing
  Set_Boundary_Extents(dir, &imin[0], &imax[0]);

  // from grid/cuda_boundaries.cu
  SetGhostCells(C.device,
		 H.nx, H.ny, H.nz, H.n_fields, H.n_cells, H.n_ghost, flags,
		 imax[0]-imin[0], imax[1]-imin[1], imax[2]-imin[2],
		 imin[0], imin[1], imin[2], dir);
}

/*! \fn Set_Boundary_Extents(int dir, int *imin, int *imax)
 *  \brief Set the extents of the ghost region we are initializing. */
void Grid3D::Set_Boundary_Extents(int dir, int *imin, int *imax)
{
  int il, iu, jl, ju, kl, ku;
  jl = 0;
  ju = H.ny;
  kl = 0;
  ku = H.nz;
  if (H.ny > 1) {
    jl = H.n_ghost;
    ju = H.ny-H.n_ghost;
  }
  if (H.nz > 1) {
    kl = H.n_ghost;
    ku = H.nz-H.n_ghost;
  }

  il = 0;
  iu = H.n_ghost;
  /*lower x face*/
  if(dir==0)
  {
    *(imin) = il;
    *(imax) = iu;
    *(imin+1) = jl;
    *(imax+1) = ju;
    *(imin+2) = kl;
    *(imax+2) = ku;
  }
  il = H.nx-H.n_ghost;
  iu = H.nx;
  /*upper x face*/
  if(dir==1)
  {
    *(imin) = il;
    *(imax) = iu;
    *(imin+1) = jl;
    *(imax+1) = ju;
    *(imin+2) = kl;
    *(imax+2) = ku;
  }
  il = 0;
  iu = H.nx;
  jl = 0;
  ju = H.n_ghost;
  /*lower y face*/
  if(dir==2)
  {
    *(imin) = il;
    *(imax) = iu;
    *(imin+1) = jl;
    *(imax+1) = ju;
    *(imin+2) = kl;
    *(imax+2) = ku;
  }
  jl = H.ny-H.n_ghost;
  ju = H.ny;
  /*upper y face*/
  if(dir==3)
  {
    *(imin) = il;
    *(imax) = iu;
    *(imin+1) = jl;
    *(imax+1) = ju;
    *(imin+2) = kl;
    *(imax+2) = ku;
  }
  jl = 0;
  ju = H.ny;
  kl = 0;
  ku = H.n_ghost;
  /*lower z face*/
  if(dir==4)
  {
    *(imin) = il;
    *(imax) = iu;
    *(imin+1) = jl;
    *(imax+1) = ju;
    *(imin+2) = kl;
    *(imax+2) = ku;
  }
  kl = H.nz-H.n_ghost;
  ku = H.nz;
  /*upper z face*/
  if(dir==5)
  {
    *(imin) = il;
    *(imax) = iu;
    *(imin+1) = jl;
    *(imax+1) = ju;
    *(imin+2) = kl;
    *(imax+2) = ku;
  }
}



/*! \fn void Custom_Boundary(char bcnd[MAXLEN])
 *  \brief Select appropriate custom boundary function. */
void Grid3D::Custom_Boundary(char bcnd[MAXLEN])
{
  if (strcmp(bcnd, "noh")==0) {
    // from grid/cuda_boundaries.cu
    Noh_Boundary();
  }
  if (strcmp(bcnd, "wind")==0) {
    // from grid/cuda_boundaries.cu
    Wind_Boundary();
  }
  else {
    printf("ABORT: %s -> Unknown custom boundary condition.\n", bcnd);
    exit(0);
  }
}



/*! \fn void Wind_Boundary()
 *  \brief Apply wind boundary */
void Grid3D::Wind_Boundary()
{

  int x_off, y_off, z_off;
  // set x, y, & z offsets of local CPU volume to pass to GPU
  // so global position on the grid is known
  x_off = y_off = z_off = 0;
  #ifdef MPI_CHOLLA
  x_off = nx_local_start;
  y_off = ny_local_start;
  z_off = nz_local_start;
  #endif

  Wind_Boundary_CUDA(C.device, H.nx, H.ny, H.nz, H.n_cells, H.n_ghost,
         x_off, y_off, z_off, H.dx, H.dy, H.dz, 
         H.xbound, H.ybound, H.zbound, gama, H.t);
}

/*! \fn void Noh_Boundary()
 *  \brief Apply analytic boundary conditions to +x, +y (and +z) faces,
    as per the Noh problem in Liska, 2003, or in Stone, 2008. */
void Grid3D::Noh_Boundary()
{
  // This is now a wrapper function -- the actual boundary setting
  // functions are in grid/cuda_boundaries.cu

  int x_off, y_off, z_off;
  // set x, y, & z offsets of local CPU volume to pass to GPU
  // so global position on the grid is known
  x_off = y_off = z_off = 0;
  #ifdef MPI_CHOLLA
  x_off = nx_local_start;
  y_off = ny_local_start;
  z_off = nz_local_start;
  #endif

  Noh_Boundary_CUDA(C.device, H.nx, H.ny, H.nz, H.n_cells, H.n_ghost,
                    x_off, y_off, z_off, H.dx, H.dy, H.dz, 
                    H.xbound, H.ybound, H.zbound, gama, H.t);

/*
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r;
  Real vx, vy, vz, d_0, P_0, P;
  d_0 = 1.0;
  P_0 = 1.0e-6;
  // set exact boundaries on the +x face
  for (k=0; k<H.nz; k++) {
    for (j=0; j<H.ny; j++) {
      for (i=H.nx-H.n_ghost; i<H.nx; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;
        // get the (centered) x, y, and z positions at (x,y,z)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        if (H.nz > 1) r = sqrt(x_pos*x_pos + y_pos*y_pos+ z_pos*z_pos);
        else r = sqrt(x_pos*x_pos + y_pos*y_pos);
        // set the velocities
        vx = -x_pos / r;
        vy = -y_pos / r;
        if (H.nz > 1) vz = -z_pos / r;
        else vz = 0;
        // set the conserved quantities
        if (H.nz > 1) C.density[id] = d_0*(1.0 + H.t/r)*(1.0 + H.t/r);
        else C.density[id]    = d_0*(1.0 + H.t/r);
        C.momentum_x[id] = vx*C.density[id];
        C.momentum_y[id] = vy*C.density[id];
        C.momentum_z[id] = vz*C.density[id];
        C.Energy[id]     = P_0/(gama-1.0) + 0.5*C.density[id];
      }
    }
  }
  // set exact boundaries on the +y face
  for (k=0; k<H.nz; k++) {
    for (j=H.ny-H.n_ghost; j<H.ny; j++) {
      for (i=0; i<H.nx; i++) {
        id = i + j*H.nx + k*H.nx*H.ny;
        // get the (centered) x, y, and z positions at (x,y,z)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        if (H.nz > 1) r = sqrt(x_pos*x_pos + y_pos*y_pos+ z_pos*z_pos);
        else r = sqrt(x_pos*x_pos + y_pos*y_pos);
        // set the velocities
        vx = -x_pos / r;
        vy = -y_pos / r;
        if (H.nz > 1) vz = -z_pos / r;
        else vz = 0;
        // set the conserved quantities
        if (H.nz > 1) C.density[id] = d_0*(1.0 + H.t/r)*(1.0 + H.t/r);
        else C.density[id]    = d_0*(1.0 + H.t/r);
        C.momentum_x[id] = vx*C.density[id];
        C.momentum_y[id] = vy*C.density[id];
        C.momentum_z[id] = vz*C.density[id];
        C.Energy[id]     = P_0/(gama-1.0) + 0.5*C.density[id];
      }
    }
  }
  // set exact boundaries on the +z face
  if (H.nz > 1) {
    for (k=H.nz-H.n_ghost; k<H.nz; k++) {
      for (j=0; j<H.ny; j++) {
        for (i=0; i<H.nx; i++) {
          id = i + j*H.nx + k*H.nx*H.ny;
          // get the (centered) x, y, and z positions at (x,y,z)
          Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
          r = sqrt(x_pos*x_pos + y_pos*y_pos+ z_pos*z_pos);
          // set the velocities
          vx = -x_pos / r;
          vy = -y_pos / r;
          vz = -z_pos / r;
          // set the conserved quantities
          C.density[id]    = d_0*(1.0 + H.t/r)*(1.0 + H.t/r);
          C.momentum_x[id] = vx*C.density[id];
          C.momentum_y[id] = vy*C.density[id];
          C.momentum_z[id] = vz*C.density[id];
          C.Energy[id]     = P_0/(gama-1.0) + 0.5*C.density[id];
        }
      }
    }
  }
*/
}

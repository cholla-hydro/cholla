/*! \file boundary_conditions.cpp
 *  \brief Definitions of the boundary conditions for various tests.
           Functions are members of the Grid3D class. */

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"grid3D.h"
#include"io.h"
#include"error_handling.h"
#include"mpi_routines.h"

#include "nvtx.h"

/*! \fn void Set_Boundary_Conditions_Grid(parameters P)
 *  \brief Set the boundary conditions for all componentes based on info in the parameters structure. */
void Grid3D::Set_Boundary_Conditions_Grid( parameters P){
  
  #ifndef ONLY_PARTICLES
  // Dont transfer Hydro boundaries when only doing particles
  
  // Transfer Hydro Conserved boundaries 
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif //CPU_TIME
  H.TRANSFER_HYDRO_BOUNDARIES = true;
  Set_Boundary_Conditions(P);
  H.TRANSFER_HYDRO_BOUNDARIES = false;
  #ifdef CPU_TIME
  Timer.End_and_Record_Time( 2 );
  #endif //CPU_TIME
  #endif //ONLY_PARTICLES

  // If the Gravity cuopling is on the CPU, the potential is not in the Conserved arrays,
  // and its boundaries need to be transfered separately
  #ifdef GRAVITY
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif
  Grav.TRANSFER_POTENTIAL_BOUNDARIES = true;
  Set_Boundary_Conditions(P);
  Grav.TRANSFER_POTENTIAL_BOUNDARIES = false;
  #ifdef CPU_TIME
  Timer.End_and_Record_Time( 9 );
  #endif
  #endif
}

/*! \fn void Set_Boundary_Conditions(parameters P)
 *  \brief Set the boundary conditions based on info in the parameters structure. */
void Grid3D::Set_Boundary_Conditions(parameters P) {

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
  nvtx_raii _nvtx(__FUNCTION__, __LINE__);
  int i, j, k;
  int imin[3] = {0,0,0};
  int imax[3] = {H.nx,H.ny,H.nz};
  Real a[3]   = {1,1,1};  //sign of momenta
  int idx;    //index of a real cell
  int gidx;   //index of a ghost cell

#ifdef   MPI_CHOLLA
  /*if the cell face is an mpi boundary, exit */
  if(flags[dir]==5)
    return;
#endif /*MPI_CHOLLA*/

  #if( defined(GRAVITY) )
  if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
    if ( flags[dir] ==1 ){
      if ( dir == 0 ) Copy_Potential_Boundaries( 0, 0 );
      if ( dir == 1 ) Copy_Potential_Boundaries( 0, 1 );
      if ( dir == 2 ) Copy_Potential_Boundaries( 1, 0 );
      if ( dir == 3 ) Copy_Potential_Boundaries( 1, 1 );
      if ( dir == 4 ) Copy_Potential_Boundaries( 2, 0 );
      if ( dir == 5 ) Copy_Potential_Boundaries( 2, 1 );
    }
    return; 
  }
  #endif
  
  #ifdef PARTICLES
  if ( Particles.TRANSFER_DENSITY_BOUNDARIES ){
    if ( flags[dir] ==1 ){
      if ( dir == 0 ) Copy_Particles_Density_Boundaries( 0, 0 );
      if ( dir == 1 ) Copy_Particles_Density_Boundaries( 0, 1 );
      if ( dir == 2 ) Copy_Particles_Density_Boundaries( 1, 0 );
      if ( dir == 3 ) Copy_Particles_Density_Boundaries( 1, 1 );
      if ( dir == 4 ) Copy_Particles_Density_Boundaries( 2, 0 );
      if ( dir == 5 ) Copy_Particles_Density_Boundaries( 2, 1 );
    }
    return; 
  }
  #endif
  
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
      #endif//PARTICLES_CPU
      
      
    }
    return; 
  }
  #endif

  //get the extents of the ghost region we are initializing
  Set_Boundary_Extents(dir, &imin[0], &imax[0]);

  nvtx_raii _set_ghost_cells("set_ghost_cells", __LINE__);
  /*set ghost cells*/
  #ifdef DEVICE_COMM
  Set_Ghost_Cells_Cuda(&imin[0], &imax[0], &a[0], &flags[0], dir);
  #else
  Set_Ghost_Cells(&imin[0], &imax[0], &a[0], &flags[0], dir);
  #endif
}

void Grid3D::Set_Ghost_Cells(int imin[3], int imax[3], Real a[3], int flags[6], int dir) {

  for (int k=imin[2]; k<imax[2]; k++) {
    for (int j=imin[1]; j<imax[1]; j++) {
      for (int i=imin[0]; i<imax[0]; i++) {

        //reset sign of momenta
        a[0] = 1.;
        a[1] = 1.;
        a[2] = 1.;

        //find the ghost cell index
        int gidx = i + j*H.nx + k*H.nx*H.ny;

        //find the corresponding real cell index and momenta signs
        int idx  = Set_Boundary_Mapping(i,j,k,flags,&a[0]);

        
        //idx will be >= 0 if the boundary mapping function has
        //not set this ghost cell by hand, for instance for analytical 
        //boundary conditions
        //
        //Otherwise, the boundary mapping function will set idx<0
        //if this ghost cell has been set by hand
        if(idx>=0)
        {
          //set the ghost cell value
          C.density[gidx]    = C.density[idx];
          C.momentum_x[gidx] = C.momentum_x[idx]*a[0];
          C.momentum_y[gidx] = C.momentum_y[idx]*a[1];
          C.momentum_z[gidx] = C.momentum_z[idx]*a[2];
          C.Energy[gidx]     = C.Energy[idx];
          #ifdef DE
          C.GasEnergy[gidx]  = C.GasEnergy[idx];
          #endif
          #ifdef SCALAR 
          for (int ii=0; ii<NSCALARS; ii++) {
            C.scalar[gidx + ii*H.n_cells]  = C.scalar[idx + ii*H.n_cells];
          }
          #endif

          //for outflow boundaries, set momentum to restrict inflow
          if (flags[dir] == 3) {
            // first subtract kinetic energy from total
            C.Energy[gidx] -= 0.5*(C.momentum_x[gidx]*C.momentum_x[gidx] + C.momentum_y[gidx]*C.momentum_y[gidx] + C.momentum_z[gidx]*C.momentum_z[gidx])/C.density[gidx];
            if (dir == 0) {
              C.momentum_x[gidx] = fmin(C.momentum_x[gidx], 0.0);
            }
            if (dir == 1) {
              C.momentum_x[gidx] = fmax(C.momentum_x[gidx], 0.0);
            }
            if (dir == 2) {
              C.momentum_y[gidx] = fmin(C.momentum_y[gidx], 0.0);
            }
            if (dir == 3) {
              C.momentum_y[gidx] = fmax(C.momentum_y[gidx], 0.0);
            }
            if (dir == 4) {
              C.momentum_z[gidx] = fmin(C.momentum_z[gidx], 0.0);
            }
            if (dir == 5) {
              C.momentum_z[gidx] = fmax(C.momentum_z[gidx], 0.0);
            }
            // now re-add the new kinetic energy
            C.Energy[gidx] += 0.5*(C.momentum_x[gidx]*C.momentum_x[gidx] + C.momentum_y[gidx]*C.momentum_y[gidx] + C.momentum_z[gidx]*C.momentum_z[gidx])/C.density[gidx];
          }

          
        }
      }
    }
  }
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


/*! \fn Set_Boundary_Mapping(int ig, int jg, int kg, int flags[], Real *a)
 *  \brief Given the i,j,k index of a ghost cell, return the index of the
    corresponding real cell, and reverse the momentum if necessary. */
int Grid3D::Set_Boundary_Mapping(int ig, int jg, int kg, int flags[], Real *a)
{
  // index of real cell we're mapping to
  int ir, jr, kr, idx;
  ir = jr = kr = idx = 0;

  /* 1D */
  if (H.nx>1) {

    // set index on -x face
    if (ig < H.n_ghost) {
      ir = Find_Index(ig, H.nx, flags[0], 0, &a[0]);
    }
    // set index on +x face
    else if (ig >= H.nx-H.n_ghost) {
      ir = Find_Index(ig, H.nx, flags[1], 1, &a[0]);
    }
    // set i index for multi-D problems
    else {
      ir = ig;
    }

    // if custom x boundaries are needed, set index to -1 and return
    if (ir < 0) {
      return idx = -1;
    }

    // otherwise add i index to ghost cell mapping
    idx += ir;

  }

  /* 2D */
  if (H.ny > 1) {

    // set index on -y face
    if (jg < H.n_ghost) {
      jr = Find_Index(jg, H.ny, flags[2], 0, &a[1]);
    }
    // set index on +y face
    else if (jg >= H.ny-H.n_ghost) {
      jr = Find_Index(jg, H.ny, flags[3], 1, &a[1]);
    }
    // set j index for multi-D problems
    else {
      jr = jg;
    }

    // if custom y boundaries are needed, set index to -1 and return
    if (jr < 0) {
      return idx = -1;
    }
    
    // otherwise add j index to ghost cell mapping
    idx += H.nx*jr;

  }

  /* 3D */
  if (H.nz > 1) {

    // set index on -z face
    if (kg < H.n_ghost) {
      kr = Find_Index(kg, H.nz, flags[4], 0, &a[2]);
    }
    // set index on +z face
    else if (kg >= H.nz-H.n_ghost) {
      kr = Find_Index(kg, H.nz, flags[5], 1, &a[2]);
    }
    // set k index for multi-D problems
    else {
      kr = kg;
    }

    // if custom z boundaries are needed, set index to -1 and return
    if (kr < 0) {
      return idx = -1;
    }

    // otherwise add k index to ghost cell mapping
    idx += H.nx*H.ny*kr;

  }

  return idx;
}

/*! \fn int Find_Index(int ig, int nx, int flag, int face, Real *a)
 *  \brief Given a ghost cell index and boundary flag, 
    return the index of the corresponding real cell. */
int Grid3D::Find_Index(int ig, int nx, int flag, int face, Real *a)
{
  int id;

  // lower face
  if (face==0) {
    switch(flag)
    {
      // periodic
      case 1: id = ig+nx-2*H.n_ghost;
        break;
      // reflective
      case 2: id = 2*H.n_ghost-ig-1;
        *(a) = -1.0;
        break;
      // transmissive
      case 3: id = H.n_ghost;
        break;
      // custom
      case 4: id = -1;
        break;
      // MPI
      case 5: id = ig;
        break;
      // default is periodic
      default: id = ig+nx-2*H.n_ghost;
    }
  }
  // upper face
  else {
    switch(flag)
    {
      // periodic
      case 1: id = ig-nx+2*H.n_ghost;
        break;
      // reflective
      case 2: id = 2*(nx-H.n_ghost)-ig-1;
        *(a) = -1.0;
        break;
      // transmissive
      case 3: id = nx-H.n_ghost-1;
        break;
      // custom
      case 4: id = -1;
        break;
      // MPI
      case 5: id = ig;
        break;
      // default is periodic
      default: id = ig-nx+2*H.n_ghost;
    }
  }

  return id;
}


/*! \fn void Custom_Boundary(char bcnd[MAXLEN])
 *  \brief Select appropriate custom boundary function. */
void Grid3D::Custom_Boundary(char bcnd[MAXLEN])
{
  if (strcmp(bcnd, "noh")==0) {
    Noh_Boundary();
  }
  else {
    printf("ABORT: %s -> Unknown custom boundary condition.\n", bcnd);
    exit(0);
  }
}



/*! \fn void Noh_Boundary()
 *  \brief Apply analytic boundary conditions to +x, +y (and +z) faces, 
    as per the Noh problem in Liska, 2003, or in Stone, 2008. */
void Grid3D::Noh_Boundary()
{
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

}



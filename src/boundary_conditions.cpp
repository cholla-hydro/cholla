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

  //get the extents of the ghost region we are initializing
  Set_Boundary_Extents(dir, &imin[0], &imax[0]);

  /*set ghost cells*/
  for (k=imin[2]; k<imax[2]; k++) {
    for (j=imin[1]; j<imax[1]; j++) {
      for (i=imin[0]; i<imax[0]; i++) {

        //reset sign of momenta
        a[0] = 1.;
        a[1] = 1.;
        a[2] = 1.;

        //find the ghost cell index
        gidx = i + j*H.nx + k*H.nx*H.ny; 

        //find the corresponding real cell index and momenta signs
        idx  = Set_Boundary_Mapping(i,j,k,flags,&a[0]);

        
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
  else if (strcmp(bcnd, "disk_2D")==0) {
    Disk_Boundary_2D();
  }  
  else if (strcmp(bcnd, "disk_3D")==0) {
    Disk_Boundary_3D();
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


/*! \fn void Disk_Boundary_2D()
 *  \brief Apply analytic boundary conditions to x and y faces, 
    as per the disk setup in the ICs. */
void Grid3D::Disk_Boundary_2D()
{
  int i, j, id;
  Real x_pos, y_pos, z_pos, xc, yc, r, phi;
  Real d, n, a, a_d, a_h, v, vx, vy, P, T_d, x;
  Real M_vir, M_h, M_d, c_vir, R_vir, R_h, R_d, Sigma;

  M_vir = 1.0e12; // viral mass of MW in M_sun
  M_d = 6.5e10; // mass of disk in M_sun (assume all gas)
  M_h = M_vir - M_d; // halo mass in M_sun
  R_vir = 261; // viral radius in kpc
  c_vir = 20; // halo concentration
  R_h = R_vir / c_vir; // halo scale length in kpc
  R_d = 3.5; // disk scale length in kpc
  T_d = 10000; // disk temperature, 10^4K

  // set exact boundaries on the -x face
  for (j=0; j<H.ny; j++) {
    for (i=0; i<H.n_ghost; i++) {

      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);
      
      // calculate centered radial position and phi
      r = sqrt(x_pos*x_pos + y_pos*y_pos);
      phi = atan2(y_pos, x_pos);

      // Disk surface density [M_sun / kpc^2]
      // Assume gas surface density is exponential with scale length 2*R_d and
      // mass 0.25*M_d
      Sigma = 0.25*M_d * exp(-r/(2*R_d)) / (2*PI*2*R_d) ;
      d = Sigma; // just use sigma for mass density since height is arbitrary
      n = d * DENSITY_UNIT / MP; // number density, cgs
      P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

      // radial acceleration due to Kuzmin disk + NFW halo
      x = r / R_h;
      a_d = GN * M_d * r * pow(r*r + R_d*R_d, -1.5);
      a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
      a = a_d + a_h;

      // circular velocity 
      v = sqrt(r*a);
      vx = -sin(phi)*v;
      vy = cos(phi)*v;

      // set values of conserved variables   
      C.density[id] = d;
      C.momentum_x[id] = d*vx;
      C.momentum_y[id] = d*vy;
      C.momentum_z[id] = 0.0;
      C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy);
      
    }
  }

  // set exact boundaries on the +x face
  for (j=0; j<H.ny; j++) {
    for (i=H.nx-H.n_ghost; i<H.nx; i++) {

      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);
      
      // calculate centered radial position and phi
      r = sqrt(x_pos*x_pos + y_pos*y_pos);
      phi = atan2(y_pos, x_pos);

      // Disk surface density [M_sun / kpc^2]
      // Assume gas surface density is exponential with scale length 2*R_d and
      // mass 0.25*M_d
      Sigma = 0.25*M_d * exp(-r/(2*R_d)) / (2*PI*2*R_d) ;
      d = Sigma; // just use sigma for mass density since height is arbitrary
      n = d * DENSITY_UNIT / MP; // number density, cgs
      P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

      // radial acceleration due to Kuzmin disk + NFW halo
      x = r / R_h;
      a_d = GN * M_d * r * pow(r*r + R_d*R_d, -1.5);
      a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
      a = a_d + a_h;

      // circular velocity 
      v = sqrt(r*a);
      vx = -sin(phi)*v;
      vy = cos(phi)*v;

      // set values of conserved variables   
      C.density[id] = d;
      C.momentum_x[id] = d*vx;
      C.momentum_y[id] = d*vy;
      C.momentum_z[id] = 0.0;
      C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy);
      
    }
  }
  
  // set exact boundaries on the -y face
  for (j=0; j<H.n_ghost; j++) {
    for (i=0; i<H.nx; i++) {

       id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);
      
      // calculate centered radial position and phi
      r = sqrt(x_pos*x_pos + y_pos*y_pos);
      phi = atan2(y_pos, x_pos);

      // Disk surface density [M_sun / kpc^2]
      // Assume gas surface density is exponential with scale length 2*R_d and
      // mass 0.25*M_d
      Sigma = 0.25*M_d * exp(-r/(2*R_d)) / (2*PI*2*R_d) ;
      d = Sigma; // just use sigma for mass density since height is arbitrary
      n = d * DENSITY_UNIT / MP; // number density, cgs
      P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

      // radial acceleration due to Kuzmin disk + NFW halo
      x = r / R_h;
      a_d = GN * M_d * r * pow(r*r + R_d*R_d, -1.5);
      a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
      a = a_d + a_h;

      // circular velocity 
      v = sqrt(r*a);
      vx = -sin(phi)*v;
      vy = cos(phi)*v;

      // set values of conserved variables   
      C.density[id] = d;
      C.momentum_x[id] = d*vx;
      C.momentum_y[id] = d*vy;
      C.momentum_z[id] = 0.0;
      C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy);     

    }
  }

  // set exact boundaries on the +y face
  for (j=H.ny-H.n_ghost; j<H.ny; j++) {
    for (i=0; i<H.nx; i++) {

      id = i + j*H.nx;
      // get the centered x and y positions
      Get_Position(i, j, H.n_ghost, &x_pos, &y_pos, &z_pos);
      
      // calculate centered radial position and phi
      r = sqrt(x_pos*x_pos + y_pos*y_pos);
      phi = atan2(y_pos, x_pos);

      // Disk surface density [M_sun / kpc^2]
      // Assume gas surface density is exponential with scale length 2*R_d and
      // mass 0.25*M_d
      Sigma = 0.25*M_d * exp(-r/(2*R_d)) / (2*PI*2*R_d) ;
      d = Sigma; // just use sigma for mass density since height is arbitrary
      n = d * DENSITY_UNIT / MP; // number density, cgs
      P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

      // radial acceleration due to Kuzmin disk + NFW halo
      x = r / R_h;
      a_d = GN * M_d * r * pow(r*r + R_d*R_d, -1.5);
      a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
      a = a_d + a_h;

      // circular velocity 
      v = sqrt(r*a);
      vx = -sin(phi)*v;
      vy = cos(phi)*v;

      // set values of conserved variables   
      C.density[id] = d;
      C.momentum_x[id] = d*vx;
      C.momentum_y[id] = d*vy;
      C.momentum_z[id] = 0.0;
      C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy);
      
    }
  }
  

}


/*! \fn void Disk_Boundary_3D()
 *  \brief Apply analytic boundary conditions to x, y, and z faces, 
    as per the disk setup in the ICs. */
void Grid3D::Disk_Boundary_3D()
{
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r, phi;
  Real d, n, a, a_d, a_h, v, vx, vy, vz, P, T_d, x;
  Real M_vir, M_h, M_d, c_vir, R_vir, R_s, R_d, z_d, Sigma;

  M_vir = 1.0e12; // viral mass of MW in M_sun
  M_d = 6.5e10; // mass of disk in M_sun (assume all gas)
  M_h = M_vir - M_d; // halo mass in M_sun
  R_vir = 261; // viral radius in kpc
  c_vir = 20; // halo concentration
  R_s = R_vir / c_vir; // halo scale length in kpc
  R_d = 3.5; // disk scale length in kpc
  z_d = 3.5/5.0; // disk scale height in kpc
  T_d = 10000; // disk temperature, 10^4K

  // set exact boundaries on the -x face
  for (k=0; k<H.nz; k++) {
    for (j=0; j<H.ny; j++) {
      for (i=0; i<H.n_ghost; i++) {
  
        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate radial position and phi (assumes disk is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos);
        phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)

        // mass density [M_sun / kpc^3]
        d = ((z_d*z_d*M_d)/(4*PI)) * (R_d*r*r + (R_d + 3*sqrt(z_pos*z_pos+z_d*z_d))*(R_d + sqrt(z_pos*z_pos+z_d*z_d))) / (pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d), 2),2.5) * pow(z_pos*z_pos + z_d*z_d, 1.5));
        n = d * DENSITY_UNIT / MP; // number density, cgs
        P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

        // radial acceleration due to Kuzmin disk + NFW halo
        x = r / R_s;
        a_d = GN * M_d * r * pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
        a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
        a = a_d + a_h;

        // radial velocity 
        v = sqrt(r*a);
        vx = -sin(phi)*v;
        vy = cos(phi)*v;

        vz =  0.0;

        // set values of conserved variables   
        C.density[id] = d;
        C.momentum_x[id] = d*vx;
        C.momentum_y[id] = d*vy;
        C.momentum_z[id] = d*vz;
        C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy + vz*vz);

      }
    }
  }

  // set exact boundaries on the +x face
  for (k=0; k<H.nz; k++) {
    for (j=0; j<H.ny; j++) {
      for (i=H.nx-H.n_ghost; i<H.nx; i++) {
  
        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate radial position and phi (assumes disk is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos);
        phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)

        // mass density [M_sun / kpc^3]
        d = ((z_d*z_d*M_d)/(4*PI)) * (R_d*r*r + (R_d + 3*sqrt(z_pos*z_pos+z_d*z_d))*(R_d + sqrt(z_pos*z_pos+z_d*z_d))) / (pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d), 2),2.5) * pow(z_pos*z_pos + z_d*z_d, 1.5));
        n = d * DENSITY_UNIT / MP; // number density, cgs
        P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

        // radial acceleration due to Kuzmin disk + NFW halo
        x = r / R_s;
        a_d = GN * M_d * r * pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
        a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
        a = a_d + a_h;

        // radial velocity 
        v = sqrt(r*a);
        vx = -sin(phi)*v;
        vy = cos(phi)*v;

        vz =  0.0;

        // set values of conserved variables   
        C.density[id] = d;
        C.momentum_x[id] = d*vx;
        C.momentum_y[id] = d*vy;
        C.momentum_z[id] = d*vz;
        C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy + vz*vz);
      }
    }
  }
  // set exact boundaries on the -y face
  for (k=0; k<H.nz; k++) {
    for (j=0; j<H.n_ghost; j++) {
      for (i=0; i<H.nx; i++) {
  
        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate radial position and phi (assumes disk is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos);
        phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)

        // mass density [M_sun / kpc^3]
        d = ((z_d*z_d*M_d)/(4*PI)) * (R_d*r*r + (R_d + 3*sqrt(z_pos*z_pos+z_d*z_d))*(R_d + sqrt(z_pos*z_pos+z_d*z_d))) / (pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d), 2),2.5) * pow(z_pos*z_pos + z_d*z_d, 1.5));
        n = d * DENSITY_UNIT / MP; // number density, cgs
        P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

        // radial acceleration due to Kuzmin disk + NFW halo
        x = r / R_s;
        a_d = GN * M_d * r * pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
        a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
        a = a_d + a_h;

        // radial velocity 
        v = sqrt(r*a);
        vx = -sin(phi)*v;
        vy = cos(phi)*v;

        vz =  0.0;

        // set values of conserved variables   
        C.density[id] = d;
        C.momentum_x[id] = d*vx;
        C.momentum_y[id] = d*vy;
        C.momentum_z[id] = d*vz;
        C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy + vz*vz);
      }
    }
  }

  // set exact boundaries on the +y face
  for (k=0; k<H.nz; k++) {
    for (j=H.ny-H.n_ghost; j<H.ny; j++) {
      for (i=0; i<H.nx; i++) {
  
        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate radial position and phi (assumes disk is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos);
        phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)

        // mass density [M_sun / kpc^3]
        d = ((z_d*z_d*M_d)/(4*PI)) * (R_d*r*r + (R_d + 3*sqrt(z_pos*z_pos+z_d*z_d))*(R_d + sqrt(z_pos*z_pos+z_d*z_d))) / (pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d), 2),2.5) * pow(z_pos*z_pos + z_d*z_d, 1.5));
        n = d * DENSITY_UNIT / MP; // number density, cgs
        P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

        // radial acceleration due to Kuzmin disk + NFW halo
        x = r / R_s;
        a_d = GN * M_d * r * pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
        a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
        a = a_d + a_h;

        // radial velocity 
        v = sqrt(r*a);
        vx = -sin(phi)*v;
        vy = cos(phi)*v;

        vz =  0.0;

        // set values of conserved variables   
        C.density[id] = d;
        C.momentum_x[id] = d*vx;
        C.momentum_y[id] = d*vy;
        C.momentum_z[id] = d*vz;
        C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy + vz*vz);
      }
    }
  }
  // set exact boundaries on the -z face
  for (k=0; k<H.n_ghost; k++) {
    for (j=0; j<H.ny; j++) {
      for (i=0; i<H.nx; i++) {
  
        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate radial position and phi (assumes disk is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos);
        phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)

        // mass density [M_sun / kpc^3]
        d = ((z_d*z_d*M_d)/(4*PI)) * (R_d*r*r + (R_d + 3*sqrt(z_pos*z_pos+z_d*z_d))*(R_d + sqrt(z_pos*z_pos+z_d*z_d))) / (pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d), 2),2.5) * pow(z_pos*z_pos + z_d*z_d, 1.5));
        n = d * DENSITY_UNIT / MP; // number density, cgs
        P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

        // radial acceleration due to Kuzmin disk + NFW halo
        x = r / R_s;
        a_d = GN * M_d * r * pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
        a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
        a = a_d + a_h;

        // radial velocity 
        v = sqrt(r*a);
        vx = -sin(phi)*v;
        vy = cos(phi)*v;

        vz =  0.0;

        // set values of conserved variables   
        C.density[id] = d;
        C.momentum_x[id] = d*vx;
        C.momentum_y[id] = d*vy;
        C.momentum_z[id] = d*vz;
        C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy + vz*vz);
      }
    }
  }

  // set exact boundaries on the +z face
  for (k=H.nz-H.n_ghost; k<H.nz; k++) {
    for (j=0; j<H.ny; j++) {
      for (i=0; i<H.nx; i++) {
  
        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate radial position and phi (assumes disk is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos);
        phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)

        // mass density [M_sun / kpc^3]
        d = ((z_d*z_d*M_d)/(4*PI)) * (R_d*r*r + (R_d + 3*sqrt(z_pos*z_pos+z_d*z_d))*(R_d + sqrt(z_pos*z_pos+z_d*z_d))) / (pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d), 2),2.5) * pow(z_pos*z_pos + z_d*z_d, 1.5));
        n = d * DENSITY_UNIT / MP; // number density, cgs
        P = n*KB*T_d / PRESSURE_UNIT; // disk pressure, code units

        // radial acceleration due to Kuzmin disk + NFW halo
        x = r / R_s;
        a_d = GN * M_d * r * pow(r*r + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
        a_h = GN * M_h * (log(1+x)- x / (1+x)) / ((log(1+c_vir) - c_vir / (1+c_vir)) * r*r);
        a = a_d + a_h;

        // radial velocity 
        v = sqrt(r*a);
        vx = -sin(phi)*v;
        vy = cos(phi)*v;

        vz =  0.0;

        // set values of conserved variables   
        C.density[id] = d;
        C.momentum_x[id] = d*vx;
        C.momentum_y[id] = d*vy;
        C.momentum_z[id] = d*vz;
        C.Energy[id] = P/(gama-1.0) + 0.5*d*(vx*vx + vy*vy + vz*vz);
      }
    }
  }  
}


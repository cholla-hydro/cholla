#ifndef __BOUNDARY_CONDITIONS_CUDA_H__
#define __BOUNDARY_CONDITIONS_CUDA_H__

#include "grid3D.h"

// This is a CUDA ported version of the `Set_Ghost_Cells` function

/*! \fn int Find_Index(int ig, int nx, int flag, int face, Real *a)
 *  \brief Given a ghost cell index and boundary flag,
    return the index of the corresponding real cell. */
__device__ __host__ inline int Find_Index(Header &H, int ig, int nx, int flag, int face, Real a[3])
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

/*! \fn Set_Boundary_Mapping(int ig, int jg, int kg, int flags[], Real *a)
 *  \brief Given the i,j,k index of a ghost cell, return the index of the
    corresponding real cell, and reverse the momentum if necessary. */
__device__ __host__ inline int Set_Boundary_Mapping(Header &H, int ig, int jg, int kg, int flags[6], Real a[3])
{
  // index of real cell we're mapping to
  int ir, jr, kr, idx;
  ir = jr = kr = idx = 0;

  /* 1D */
  if (H.nx>1) {

    // set index on -x face
    if (ig < H.n_ghost) {
      ir = Find_Index(H, ig, H.nx, flags[0], 0, &a[0]);
    }
    // set index on +x face
    else if (ig >= H.nx-H.n_ghost) {
      ir = Find_Index(H, ig, H.nx, flags[1], 1, &a[0]);
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
      jr = Find_Index(H, jg, H.ny, flags[2], 0, &a[1]);
    }
    // set index on +y face
    else if (jg >= H.ny-H.n_ghost) {
      jr = Find_Index(H, jg, H.ny, flags[3], 1, &a[1]);
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
      kr = Find_Index(H, kg, H.nz, flags[4], 0, &a[2]);
    }
    // set index on +z face
    else if (kg >= H.nz-H.n_ghost) {
      kr = Find_Index(H, kg, H.nz, flags[5], 1, &a[2]);
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

struct Set_Ghost_Cells_Cuda_Arg {

  int imin[3];
  int imax[3];

  int flags[6];

  int dir;

};

__global__ void Set_Ghost_Cells_Cuda_Kernel(Real *dev_conserved, Header H, Set_Ghost_Cells_Cuda_Arg arg) {

  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = thread_idx % (arg.imax[0] - arg.imin[0]) + arg.imin[0]; thread_idx /= (arg.imax[0] - arg.imin[0]);
  int j = thread_idx % (arg.imax[1] - arg.imin[1]) + arg.imin[1];
  int k = thread_idx / (arg.imax[1] - arg.imin[1]) + arg.imin[2];

  Real a[3];

  //reset sign of momenta
  a[0] = 1.;
  a[1] = 1.;
  a[2] = 1.;

  //find the ghost cell index
  int gidx = i + j*H.nx + k*H.nx*H.ny;

  //find the corresponding real cell index and momenta signs
  int idx  = Set_Boundary_Mapping(H, i,j,k,arg.flags,&a[0]);

  Real *density    = &dev_conserved[0*H.n_cells];
  Real *momentum_x = &dev_conserved[1*H.n_cells];
  Real *momentum_y = &dev_conserved[2*H.n_cells];
  Real *momentum_z = &dev_conserved[3*H.n_cells];
  Real *Energy     = &dev_conserved[4*H.n_cells];

  //idx will be >= 0 if the boundary mapping function has
  //not set this ghost cell by hand, for instance for analytical
  //boundary conditions
  //
  //Otherwise, the boundary mapping function will set idx<0
  //if this ghost cell has been set by hand
  if(idx>=0)
  {
    //set the ghost cell value
    density[gidx]    = density[idx];
    momentum_x[gidx] = momentum_x[idx]*a[0];
    momentum_y[gidx] = momentum_y[idx]*a[1];
    momentum_z[gidx] = momentum_z[idx]*a[2];
    Energy[gidx]     = Energy[idx];
#ifdef DE
    C.GasEnergy[gidx]  = C.GasEnergy[idx];
#endif
#ifdef SCALAR
    for (int ii=0; ii<NSCALARS; ii++) {
      C.scalar[gidx + ii*H.n_cells]  = C.scalar[idx + ii*H.n_cells];
    }
#endif

    //for outflow boundaries, set momentum to restrict inflow
    if (arg.flags[arg.dir] == 3) {
      // first subtract kinetic energy from total
      Energy[gidx] -= 0.5*(momentum_x[gidx]*momentum_x[gidx] + momentum_y[gidx]*momentum_y[gidx] + momentum_z[gidx]*momentum_z[gidx])/density[gidx];
      if (arg.dir == 0) {
        momentum_x[gidx] = fmin(momentum_x[gidx], 0.0);
      }
      if (arg.dir == 1) {
        momentum_x[gidx] = fmax(momentum_x[gidx], 0.0);
      }
      if (arg.dir == 2) {
        momentum_y[gidx] = fmin(momentum_y[gidx], 0.0);
      }
      if (arg.dir == 3) {
        momentum_y[gidx] = fmax(momentum_y[gidx], 0.0);
      }
      if (arg.dir == 4) {
        momentum_z[gidx] = fmin(momentum_z[gidx], 0.0);
      }
      if (arg.dir == 5) {
        momentum_z[gidx] = fmax(momentum_z[gidx], 0.0);
      }
      // now re-add the new kinetic energy
      Energy[gidx] += 0.5*(momentum_x[gidx]*momentum_x[gidx] + momentum_y[gidx]*momentum_y[gidx] + momentum_z[gidx]*momentum_z[gidx])/density[gidx];
    }


  }
}

#endif

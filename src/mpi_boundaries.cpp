#include"grid3D.h"
#include"mpi_routines.h"
#include"io.h"
#include"error_handling.h"
#include <iostream>

#include"gpu.hpp"
#include"global_cuda.h"//provides TPB
#include"cuda_pack_buffers.h"

#ifdef MPI_CHOLLA

void Grid3D::Set_Boundaries_MPI(struct parameters P)
{ 
  int flags[6] = {0,0,0,0,0,0};
  
  if(Check_Custom_Boundary(&flags[0],P))
  {
    //perform custom boundaries
    Custom_Boundary(P.custom_bcnd);
  }

  switch(flag_decomp)
  {
    case SLAB_DECOMP:
      Set_Boundaries_MPI_SLAB(flags,P);
      break;
    case BLOCK_DECOMP:
      Set_Boundaries_MPI_BLOCK(flags,P);
      break;
  }
  
  #ifdef GRAVITY
  Grav.Set_Boundary_Flags( flags );
  #endif
  
}


void Grid3D::Set_Boundaries_MPI_SLAB(int *flags, struct parameters P)
{
  //perform boundary conditions around
  //edges of communication region on x faces

  if(flags[0]==5)
    Set_Edge_Boundaries(0,flags);
  if(flags[1]==5)
    Set_Edge_Boundaries(1,flags);

  //1) load and post comm for buffers
  Load_and_Send_MPI_Comm_Buffers(0, flags);

  //2) perform any additional boundary conditions
  //including whether the x face is non-MPI
  if(H.nx>1)
  {
    Set_Boundaries(0,flags);
    Set_Boundaries(1,flags);
  }
  if(H.ny>1)
  {
    Set_Boundaries(2,flags);
    Set_Boundaries(3,flags);
  }
  if(H.nz>1)
  {
    Set_Boundaries(4,flags);
    Set_Boundaries(5,flags);
  }

  //3) wait for sends and receives to finish
  //   and then load ghost cells from comm buffers
  if(flags[0]==5 || flags[1]==5)
    Wait_and_Unload_MPI_Comm_Buffers_SLAB(flags);
}


void Grid3D::Set_Boundaries_MPI_BLOCK(int *flags, struct parameters P)
{
  #ifdef PARTICLES
  // Clear the vectors that contain the particles IDs to be transfred
  if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ){
    Particles.Clear_Particles_For_Transfer();
    Particles.Select_Particles_to_Transfer_All( flags );
  }  
  #endif
  
  if (H.nx > 1) {

    /* Step 1 - Send MPI x-boundaries */
    if (flags[0]==5 || flags[1]==5) {
      Load_and_Send_MPI_Comm_Buffers(0, flags);
    }
    
    /* Step 2 - Set non-MPI x-boundaries */
    Set_Boundaries(0, flags);
    Set_Boundaries(1, flags);
    
    /* Step 3 - Receive MPI x-boundaries */
    
    if (flags[0]==5 || flags[1]==5) {
      Wait_and_Unload_MPI_Comm_Buffers_BLOCK(0, flags);
      #ifdef PARTICLES
      // Unload Particles buffers when transfering Particles
      if (Particles.TRANSFER_PARTICLES_BOUNDARIES) Wait_and_Unload_MPI_Comm_Particles_Buffers_BLOCK(0, flags);
      #endif
    }
    
  }
  MPI_Barrier(world);
  if (H.ny > 1) {

    /* Step 4 - Send MPI y-boundaries */
    if (flags[2]==5 || flags[3]==5) {
      Load_and_Send_MPI_Comm_Buffers(1, flags);
    }

    /* Step 5 - Set non-MPI y-boundaries */
    Set_Boundaries(2, flags);
    Set_Boundaries(3, flags);
    
    /* Step 6 - Receive MPI y-boundaries */
    if (flags[2]==5 || flags[3]==5) {
      Wait_and_Unload_MPI_Comm_Buffers_BLOCK(1, flags);
      #ifdef PARTICLES
      // Unload Particles buffers when transfering Particles
      if (Particles.TRANSFER_PARTICLES_BOUNDARIES) Wait_and_Unload_MPI_Comm_Particles_Buffers_BLOCK(1, flags);
      #endif
    }
  }
  MPI_Barrier(world);
  if (H.nz > 1) {

    /* Step 7 - Send MPI z-boundaries */
    if (flags[4]==5 || flags[5]==5) {
      Load_and_Send_MPI_Comm_Buffers(2, flags);
    }

    /* Step 8 - Set non-MPI z-boundaries */
    Set_Boundaries(4, flags);
    Set_Boundaries(5, flags);
    
    /* Step 9 - Receive MPI z-boundaries */
    if (flags[4]==5 || flags[5]==5) {
      Wait_and_Unload_MPI_Comm_Buffers_BLOCK(2, flags);
      #ifdef PARTICLES
      // Unload Particles buffers when transfering Particles
      if (Particles.TRANSFER_PARTICLES_BOUNDARIES) Wait_and_Unload_MPI_Comm_Particles_Buffers_BLOCK(2, flags);
      #endif
    }
  }
  
  #ifdef PARTICLES
  if ( Particles.TRANSFER_PARTICLES_BOUNDARIES)  Finish_Particles_Transfer();
  #endif

}


void Grid3D::Set_Edge_Boundaries(int dir, int *flags)
{
  int iedge;
  int i, j, k;
  int imin[3] = {0,0,0};
  int imax[3] = {H.nx,H.ny,H.nz};
  Real a[3]   = {1,1,1};  //sign of momenta
  int idx;    //index of a real cell
  int gidx;   //index of a ghost cell
  
  int nedge = 0;

  if(H.ny>1)
    nedge = 2;
  if(H.ny>1 && H.nz>1)
    nedge = 8;

  for(iedge=0;iedge<nedge;iedge++)
  {
    //set the edge or corner extents
    Set_Edge_Boundary_Extents(dir, iedge, &imin[0], &imax[0]);

    /*set ghost cells*/
    for (i=imin[0]; i<imax[0]; i++) {
      for (j=imin[1]; j<imax[1]; j++) {
        for (k=imin[2]; k<imax[2]; k++) {

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
              C.scalar[gidx + ii*H.n_cells]  = C.scalar[idx+ii*H.n_cells];
            }
            #endif
          }
        }
      }
    }
  }
}


/*! \fn Set_Edge_Boundary_Extents(int dir, int edge, int *imin, int *imax)
 *  \brief Set the extents of the edge and corner ghost regions we want to
 *         initialize first, so we can then do communication */
void Grid3D::Set_Edge_Boundary_Extents(int dir, int edge, int *imin, int *imax)
{

  int i, j, k;
  int ni, nj, nk;


  //two D case
  if(H.ny>1 && H.nz==1)
  {
    if(dir==0 || dir==1)
    {
      ni = H.nx;
      i  = 0;
      nj = H.ny;
      j  = 1;
      nk = 1;
      k  = 2;
    }

    if(dir==2 || dir==3)
    {
      ni = H.ny;
      i  = 1;
      nj = H.nx;
      j  = 0;
      nk = 1;
      k  = 2;
    }

    //upper or lower face?
    if(!(dir%2))
    {
      *(imin+i) = 0;
      *(imax+i) = 2*H.n_ghost;
  

    }else{
      *(imin+i) = ni-2*H.n_ghost;
      *(imax+i) = ni;
    }

    if(edge==0)
    {
      //lower square
      *(imin+j) = 0;
      *(imax+j) = 2*H.n_ghost;
    }else{
      //upper square
      *(imin+j) = nj-2*H.n_ghost;
      *(imax+j) = nj;
    }
    return;
  }

  //three D case
  if(H.ny>1 && H.nz>1)
  {

    if(dir==0 || dir==1)
    {
      ni = H.nx;
      i  = 0;
      nj = H.ny;
      j  = 1;
      nk = H.nz;
      k  = 2;
    }

    if(dir==2 || dir==3)
    {
      ni = H.ny;
      i  = 1;
      nj = H.nz;
      j  = 2;
      nk = H.nx;
      k  = 0;
    }

    if(dir==4 || dir==5)
    {
      ni = H.nz;
      i  = 2;
      nj = H.nx;
      j  = 0;
      nk = H.ny;
      k  = 1;
    }


    //upper or lower face?
    if(!(dir%2))
    {
      *(imin+i) = H.n_ghost;
      *(imax+i) = 2*H.n_ghost;
    }else{
      *(imin+i) = ni-2*H.n_ghost;
      *(imax+i) = ni-H.n_ghost;
    }

    //edges and corners clockwise from lower left corner
    switch(edge)
    {
      //lower left corner
      case 0:
        *(imin+j) = 0;
        *(imax+j) = H.n_ghost;
        *(imin+k) = 0;
        *(imax+k) = H.n_ghost;
        break;

      //left edge
      case 1:
        *(imin+j) = H.n_ghost;
        *(imax+j) = nj-H.n_ghost;
        *(imin+k) = 0;
        *(imax+k) = H.n_ghost;
        break;

      //upper left corner
      case 2:
        *(imin+j) = nj-H.n_ghost;
        *(imax+j) = nj;
        *(imin+k) = 0;
        *(imax+k) = H.n_ghost;
        break;

      //upper edge
      case 3:
        *(imin+j) = nj-H.n_ghost; 
        *(imax+j) = nj;
        *(imin+k) = H.n_ghost;
        *(imax+k) = nk-H.n_ghost;
        break;

      //upper right corner
      case 4:
        *(imin+j) = nj-H.n_ghost; 
        *(imax+j) = nj;
        *(imin+k) = nk-H.n_ghost;
        *(imax+k) = nk;
        break;

      //right edge
      case 5:
        *(imin+j) = H.n_ghost;
        *(imax+j) = nj-H.n_ghost;
        *(imin+k) = nk-H.n_ghost;
        *(imax+k) = nk;
        break;

      //lower right corner
      case 6:
        *(imin+j) = 0;
        *(imax+j) = H.n_ghost;
        *(imin+k) = nk-H.n_ghost;
        *(imax+k) = nk;
        break;

      //lower edge
      case 7:
        *(imin+j) = 0;
        *(imax+j) = H.n_ghost;
        *(imin+k) = H.n_ghost;
        *(imax+k) = nk-H.n_ghost;
        break;
    }
  }
}




void Grid3D::Load_and_Send_MPI_Comm_Buffers(int dir, int *flags)
{

  switch(flag_decomp)
  {
    case SLAB_DECOMP:
      /*load communication buffers*/
      Load_and_Send_MPI_Comm_Buffers_SLAB(flags);
      break;
    case BLOCK_DECOMP:
      /*load communication buffers*/
      Load_and_Send_MPI_Comm_Buffers_BLOCK(dir, flags);
      break;
  }

}

void Grid3D::Load_and_Send_MPI_Comm_Buffers_SLAB(int *flags)
{

  int i, j, k, ii;
  int gidx;
  int idx;
  int ireq = 0;

  int offset = H.n_ghost*H.ny*H.nz;

  /*check left side*/
  if(flags[0]==5)
  {
    //load left x communication buffer
    for(i=0;i<H.n_ghost;i++)
    {
      for(j=0;j<H.ny;j++)
      {
        for(k=0;k<H.nz;k++)
        {
          idx  = (i+H.n_ghost) + j*H.nx      + k*H.nx*H.ny;
          gidx = i             + j*H.n_ghost + k*H.n_ghost*H.ny;

          for (ii=0; ii<H.n_fields; ii++) {
            *(send_buffer_0 + gidx + ii*offset) = C.density[ii*H.n_cells + idx];
          }
        }
      }
    }


    //post non-blocking receive left x communication buffer
    MPI_Irecv(recv_buffer_0, recv_buffer_length, MPI_CHREAL, source[0], 0, world, &recv_request[ireq]);

    //non-blocking send left x communication buffer
    MPI_Isend(send_buffer_0, send_buffer_length, MPI_CHREAL, dest[0],   1,    world, &send_request[0]);
    MPI_Request_free(send_request);

    //remember how many recv's this proc expects
    ireq++;
  }

  /*check right side*/
  if(flags[1]==5)
  {
    //load right x communication buffer
    for(i=0;i<H.n_ghost;i++)
    {
      for(j=0;j<H.ny;j++)
      {
        for(k=0;k<H.nz;k++)
        {
          idx  = (i+H.nx-2*H.n_ghost) + j*H.nx      + k*H.nx*H.ny;
          gidx = i                    + j*H.n_ghost + k*H.n_ghost*H.ny;

          for (ii=0; ii<H.n_fields; ii++) {
            *(send_buffer_1 + gidx + ii*offset) = C.density[ii*H.n_cells + idx];
          }
        }
      }
    }


    //post non-blocking receive right x communication buffer
    MPI_Irecv(recv_buffer_1, recv_buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request[ireq]);

    //non-blocking send right x communication buffer
    MPI_Isend(send_buffer_1, send_buffer_length, MPI_CHREAL, dest[1], 0, world, &send_request[1]);
    MPI_Request_free(send_request+1);

    //remember how many recv's this proc expects
    ireq++;
  }

  //done!
}


void Grid3D::Load_Hydro_DeviceBuffer3D( Real * send_buffer_3d, int axis, int side){
  int i,j,k,ii;
  int isize,jsize,ksize,ioffset,joffset,koffset;
  int gidx,idx;
  Real *c_head;

  c_head = (Real *)C.device;
  ioffset = 0;
  joffset = H.n_ghost;
  koffset = H.n_ghost;
  
  switch (axis){
      case 0:
	isize = H.n_ghost;
	jsize = H.ny-2*H.n_ghost;
	ksize = H.nz-2*H.n_ghost;
	if (side == 0){
	  ioffset = H.n_ghost;
	} else {
	  ioffset = H.nx-2*H.n_ghost;
	}
	break;
      case 1:
	isize = H.nx;
	jsize = H.n_ghost;
	ksize = H.nz-2*H.n_ghost;
	if (side == 1){
	  joffset = H.ny-2*H.n_ghost;
	}
	break;
      case 2:
	isize = H.nx;
	jsize = H.ny;
	ksize = H.n_ghost;
	joffset = 0;
	if (side == 1){
	  koffset = H.nz-2*H.n_ghost;
	}
	break;
    
  }
  /*
  // X axis
  if (axis == 0){
    isize = H.n_ghost;
    jsize = H.ny-2*H.n_ghost;
    ksize = H.nz-2*H.n_ghost;
    if (side == 0){
      ioffset = H.n_ghost;
    } else {
      ioffset = H.nx-2*H.n_ghost;
    }
  }
  
  // Y axis
  else if (axis == 1){
    isize = H.nx;
    jsize = H.n_ghost;
    ksize = H.nz-2*H.n_ghost;
    if (side == 1){
      joffset = H.ny-2*H.n_ghost;
    }
  }
  // Z axis
  else if (axis == 2){
    isize = H.nx;
    jsize = H.ny;
    ksize = H.n_ghost;
    joffset = 0;
    if (side == 1){
      koffset = H.nz-2*H.n_ghost;
    }
  }
  */

  int offset = isize*jsize*ksize;
  int idxoffset = ioffset + joffset*H.nx + koffset*H.nx*H.ny;
  dim3 dim1dGrid((isize*jsize*ksize+TPB-1)/TPB,1,1);
  dim3 dim1dBlock(TPB, 1, 1);                                                                                              
  
  hipLaunchKernelGGL(PackBuffers3D,dim1dGrid,dim1dBlock,0,0,send_buffer_3d,c_head,isize,jsize,ksize,H.nx,H.ny,idxoffset,offset,H.n_fields,H.n_cells)

    /*
    #pragma omp target teams distribute parallel for collapse ( 3 ) \
            private ( idx, gidx ) \
      firstprivate ( offset , idxoffset, isize, jsize, ksize)	\
            is_device_ptr ( send_buffer_3d, c_head )
    for(i=0;i<isize;i++)
    {
      for(j=0;j<jsize;j++)
      {
        for(k=0;k<ksize;k++)
        {
          idx  = i + (j+k*H.ny)*H.nx + idxoffset;//(i+ioffset) + (j+joffset)*H.nx + (k+koffset)*H.nx*H.ny;
          gidx = i+(j+k*jsize)*isize;
          //gidx = i + j*isize + k*ijsize;//i+(j+k*jsize)*isize
          for (ii=0; ii<H.n_fields; ii++) {
            *(send_buffer_3d + gidx + ii*offset) = c_head[idx + ii*H.n_cells];
          }
        }
      }
    }
    */

    /*
// need to figure out G and B 

     */

    
}


// load left x communication buffer
int Grid3D::Load_Hydro_Buffer_X0( Real * send_buffer_x0 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;

  
  // 1D
  if (H.ny == 1 && H.nz == 1) {
    offset = H.n_ghost;
    for (i=0;i<H.n_ghost;i++) {
      idx = (i+H.n_ghost);
      gidx = i;
      for (ii=0; ii<H.n_fields; ii++) {
        *(send_buffer_x0 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
      }
    }
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    offset = H.n_ghost*(H.ny-2*H.n_ghost);
    for (i=0;i<H.n_ghost;i++) {
      for (j=0;j<H.ny-2*H.n_ghost;j++) {
        idx = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        gidx = i + j*H.n_ghost;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_x0 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
        } 
      }
    }
  }
  // 3D
  if (H.ny > 1 && H.nz > 1) { 
    offset = H.n_ghost*(H.ny-2*H.n_ghost)*(H.nz-2*H.n_ghost);
    for(i=0;i<H.n_ghost;i++)
    {
      for(j=0;j<H.ny-2*H.n_ghost;j++)
      {
        for(k=0;k<H.nz-2*H.n_ghost;k++)
        {
          idx  = (i+H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          gidx = i + j*H.n_ghost + k*H.n_ghost*(H.ny-2*H.n_ghost);
          for (ii=0; ii<H.n_fields; ii++) {
            *(send_buffer_x0 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
          }
        }
      }
    }
  }

  return x_buffer_length;  
}


int Grid3D::Load_Hydro_DeviceBuffer_X0 ( Real *send_buffer_x0 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  Real *c_head;
  
  c_head = (Real *)C.device;
  
  // 1D
  if (H.ny == 1 && H.nz == 1) {
    offset = H.n_ghost;
    #pragma omp target teams distribute parallel for \
            private ( idx, gidx ) \
            firstprivate ( offset ) \
            is_device_ptr ( send_buffer_x0, c_head )
    for (i=0;i<H.n_ghost;i++) {
      idx = (i+H.n_ghost);
      gidx = i;
      for (ii=0; ii<H.n_fields; ii++) {
        *(send_buffer_x0 + gidx + ii*offset) = c_head[idx + ii*H.n_cells];
      }
    }
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    offset = H.n_ghost*(H.ny-2*H.n_ghost);
    #pragma omp target teams distribute parallel for collapse ( 2 ) \
            private ( idx, gidx ) \
            firstprivate ( offset ) \
            is_device_ptr ( send_buffer_x0, c_head )
    for (i=0;i<H.n_ghost;i++) {
      for (j=0;j<H.ny-2*H.n_ghost;j++) {
        idx = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
        gidx = i + j*H.n_ghost;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_x0 + gidx + ii*offset) = c_head[idx + ii*H.n_cells];
        } 
      }
    }
  }
  // 3D
  if (H.ny > 1 && H.nz > 1) {
    Load_Hydro_DeviceBuffer3D(send_buffer_x0,0,0);
  }

  return x_buffer_length;  
}


// load right x communication buffer
int Grid3D::Load_Hydro_Buffer_X1 ( Real *send_buffer_x1 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  
  // 1D
  if (H.ny == 1 && H.nz == 1) {
    offset = H.n_ghost;
    for (i=0;i<H.n_ghost;i++) {
      idx = (i+H.nx-2*H.n_ghost);
      gidx = i;
      for (ii=0; ii<H.n_fields; ii++) {
        *(send_buffer_x1 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
      }
    }
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    offset = H.n_ghost*(H.ny-2*H.n_ghost);
    for (i=0;i<H.n_ghost;i++) {
      for (j=0;j<H.ny-2*H.n_ghost;j++) {
        idx = (i+H.nx-2*H.n_ghost) + (j+H.n_ghost)*H.nx;
        gidx = i + j*H.n_ghost;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_x1 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
        }
      }
    }
  }
  // 3D
  if (H.ny > 1 && H.nz > 1) { 
    offset = H.n_ghost*(H.ny-2*H.n_ghost)*(H.nz-2*H.n_ghost);
    for(i=0;i<H.n_ghost;i++)
    {
      for(j=0;j<H.ny-2*H.n_ghost;j++)
      {
        for(k=0;k<H.nz-2*H.n_ghost;k++)
        {
          idx  = (i+H.nx-2*H.n_ghost) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          gidx = i + j*H.n_ghost + k*H.n_ghost*(H.ny-2*H.n_ghost);
          for (ii=0; ii<H.n_fields; ii++) {
            *(send_buffer_x1 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
          }
        }
      }
    }
  }
  return x_buffer_length;
}


// load right x communication buffer
int Grid3D::Load_Hydro_DeviceBuffer_X1 ( Real *send_buffer_x1 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  Real *c_head;
  
  c_head = (Real *)C.device;
  
  // 1D
  if (H.ny == 1 && H.nz == 1) {
    offset = H.n_ghost;
    #pragma omp target teams distribute parallel for \
            private ( idx, gidx ) \
            firstprivate ( offset ) \
            is_device_ptr ( send_buffer_x1, c_head )
    for (i=0;i<H.n_ghost;i++) {
      idx = (i+H.nx-2*H.n_ghost);
      gidx = i;
      for (ii=0; ii<H.n_fields; ii++) {
        *(send_buffer_x1 + gidx + ii*offset) = c_head[idx + ii*H.n_cells];
      }
    }
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    offset = H.n_ghost*(H.ny-2*H.n_ghost);
    #pragma omp target teams distribute parallel for collapse ( 2 ) \
            private ( idx, gidx ) \
            firstprivate ( offset ) \
            is_device_ptr ( send_buffer_x1, c_head )
    for (i=0;i<H.n_ghost;i++) {
      for (j=0;j<H.ny-2*H.n_ghost;j++) {
        idx = (i+H.nx-2*H.n_ghost) + (j+H.n_ghost)*H.nx;
        gidx = i + j*H.n_ghost;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_x1 + gidx + ii*offset) = c_head[idx + ii*H.n_cells];
        }
      }
    }
  }
  // 3D
  if (H.ny > 1 && H.nz > 1) {
    Load_Hydro_DeviceBuffer3D(send_buffer_x1,0,1);
  }
  
  return x_buffer_length;
}

// load left y communication buffer
int Grid3D::Load_Hydro_Buffer_Y0 ( Real *send_buffer_y0 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  // 2D
  if (H.nz == 1) {
    offset = H.n_ghost*H.nx;
    for (i=0;i<H.nx;i++) {
      for (j=0;j<H.n_ghost;j++) {
        idx = i + (j+H.n_ghost)*H.nx;
        gidx = i + j*H.nx;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_y0 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
        }
      }
    }
  }
  // 3D
  if (H.nz > 1) { 
    offset = H.n_ghost*H.nx*(H.nz-2*H.n_ghost);
    for(i=0;i<H.nx;i++)
    {
      for(j=0;j<H.n_ghost;j++)
      {
        for(k=0;k<H.nz-2*H.n_ghost;k++)
        {
          idx  = i + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          gidx = i + j*H.nx + k*H.nx*H.n_ghost;
          for (ii=0; ii<H.n_fields; ii++) {
            *(send_buffer_y0 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
          }
        }
      }
    }
  }  
  return y_buffer_length;
}


// load left y communication buffer
int Grid3D::Load_Hydro_DeviceBuffer_Y0 ( Real *send_buffer_y0 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  Real *c_head;
  
  c_head = (Real *)C.device;
  
  // 2D
  if (H.nz == 1) {
    offset = H.n_ghost*H.nx;
    #pragma omp target teams distribute parallel for collapse ( 2 ) \
            private ( idx, gidx ) \
            firstprivate ( offset ) \
            is_device_ptr ( send_buffer_y0, c_head )
    for (i=0;i<H.nx;i++) {
      for (j=0;j<H.n_ghost;j++) {
        idx = i + (j+H.n_ghost)*H.nx;
        gidx = i + j*H.nx;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_y0 + gidx + ii*offset) = c_head[idx + ii*H.n_cells];
        }
      }
    }
  }
  // 3D
  if (H.nz > 1) {
    Load_Hydro_DeviceBuffer3D(send_buffer_y0,1,0);
  }
  return y_buffer_length;
}


// load right y communication buffer
int Grid3D::Load_Hydro_Buffer_Y1 ( Real *send_buffer_y1 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  // 2D
  if (H.nz == 1) {
    offset = H.n_ghost*H.nx;
    for (i=0;i<H.nx;i++) {
      for (j=0;j<H.n_ghost;j++) {
        idx = i + (j+H.ny-2*H.n_ghost)*H.nx;
        gidx = i + j*H.nx;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_y1 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
        }
      }
    }
  }
  // 3D
  if (H.nz > 1) { 
    offset = H.n_ghost*H.nx*(H.nz-2*H.n_ghost);
    for(i=0;i<H.nx;i++)
    {
      for(j=0;j<H.n_ghost;j++)
      {
        for(k=0;k<H.nz-2*H.n_ghost;k++)
        {
          idx  = i + (j+H.ny-2*H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
          gidx = i + j*H.nx + k*H.nx*H.n_ghost;
          for (ii=0; ii<H.n_fields; ii++) {
            *(send_buffer_y1 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
          }
        }
      }
    }
  }
  return y_buffer_length;
}


int Grid3D::Load_Hydro_DeviceBuffer_Y1 ( Real *send_buffer_y1 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  Real *c_head;
  
  c_head = (Real *)C.device;
  
  // 2D
  if (H.nz == 1) {
    offset = H.n_ghost*H.nx;
    #pragma omp target teams distribute parallel for collapse ( 2 ) \
            private ( idx, gidx ) \
            firstprivate ( offset ) \
            is_device_ptr ( send_buffer_y1, c_head )
    for (i=0;i<H.nx;i++) {
      for (j=0;j<H.n_ghost;j++) {
        idx = i + (j+H.ny-2*H.n_ghost)*H.nx;
        gidx = i + j*H.nx;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_y1 + gidx + ii*offset) = c_head[idx + ii*H.n_cells];
        }
      }
    }
  }
  // 3D
  if (H.nz > 1) {
    Load_Hydro_DeviceBuffer3D(send_buffer_y1,1,1);
  }
  return y_buffer_length;
}

// load left z communication buffer
int Grid3D::Load_Hydro_Buffer_Z0 ( Real *send_buffer_z0 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  // 3D
  offset = H.n_ghost*H.nx*H.ny;
  for(i=0;i<H.nx;i++)
  {
    for(j=0;j<H.ny;j++)
    {
      for(k=0;k<H.n_ghost;k++)
      {
        idx  = i + j*H.nx + (k+H.n_ghost)*H.nx*H.ny;
        gidx = i + j*H.nx + k*H.nx*H.ny;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_z0 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
        }
      }
    }
  }
  
  return z_buffer_length;
}

// load left z communication buffer
int Grid3D::Load_Hydro_DeviceBuffer_Z0 ( Real *send_buffer_z0 ){
  // 3D
  Load_Hydro_DeviceBuffer3D(send_buffer_z0,2,0);
  return z_buffer_length;
}

// load right z communication buffer
int Grid3D::Load_Hydro_Buffer_Z1 ( Real *send_buffer_z1 ){
  int i, j, k, ii;
  int gidx;
  int idx;
  int offset;
  offset = H.n_ghost*H.nx*H.ny;
  for(i=0;i<H.nx;i++)
  {
    for(j=0;j<H.ny;j++)
    {
      for(k=0;k<H.n_ghost;k++)
      {
        idx  = i + j*H.nx + (k+H.nz-2*H.n_ghost)*H.nx*H.ny;
        gidx = i + j*H.nx + k*H.nx*H.ny;
        for (ii=0; ii<H.n_fields; ii++) {
          *(send_buffer_z1 + gidx + ii*offset) = C.density[idx + ii*H.n_cells];
        }
      }
    }
  }
  return z_buffer_length;
}


int Grid3D::Load_Hydro_DeviceBuffer_Z1 ( Real *send_buffer_z1 ){
  Load_Hydro_DeviceBuffer3D(send_buffer_z1,2,1);
  return z_buffer_length;
}


void Grid3D::Unload_Hydro_Buffer_X0 ( Real *recv_buffer_x0 ) {
  
  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.density;
  
  // 1D
  if (ny == 1 && nz == 1) {
    offset = n_ghost;
    for(i=0;i<n_ghost;i++) {
      idx  = i;
      gidx = i;
      for (ii=0; ii<n_fields; ii++) { 
        c_head[idx + ii*n_cells] = *(recv_buffer_x0 + gidx + ii*offset);
      }
    }
  }
  // 2D
  if (ny > 1 && nz == 1) {
    offset = n_ghost*(ny-2*n_ghost);
    for(i=0;i<n_ghost;i++) {
      for (j=0;j<ny-2*n_ghost;j++) {
        idx  = i + (j+n_ghost)*nx;
        gidx = i + j*n_ghost;
        for (ii=0; ii<n_fields; ii++) { 
          c_head[idx + ii*n_cells] = *(recv_buffer_x0 + gidx + ii*offset);
        }
      }
    }
  }
  // 3D
  if (nz > 1) {
    offset = n_ghost*(ny-2*n_ghost)*(nz-2*n_ghost);
    for(i=0;i<n_ghost;i++) {
      for(j=0;j<ny-2*n_ghost;j++) {
        for(k=0;k<nz-2*n_ghost;k++) {
          idx  = i + (j+n_ghost)*nx + (k+n_ghost)*nx*ny;
          gidx = i + j*n_ghost + k*n_ghost*(ny-2*n_ghost);
          for (ii=0; ii<n_fields; ii++) {
            c_head[idx + ii*n_cells] = *(recv_buffer_x0 + gidx + ii*offset);
          }
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_DeviceBuffer_X0 ( Real *recv_buffer_x0 ) {
  
  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.device;
  
  // 1D
  if (ny == 1 && nz == 1) {
    offset = n_ghost;
    #pragma omp target teams distribute parallel for \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_x0, c_head )
    for(i=0;i<n_ghost;i++) {
      idx  = i;
      gidx = i;
      for (ii=0; ii<n_fields; ii++) { 
        c_head[idx + ii*n_cells] = *(recv_buffer_x0 + gidx + ii*offset);
      }
    }
  }
  // 2D
  if (ny > 1 && nz == 1) {
    offset = n_ghost*(ny-2*n_ghost);
    #pragma omp target teams distribute parallel for collapse ( 2 ) \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_x0, c_head )
    for(i=0;i<n_ghost;i++) {
      for (j=0;j<ny-2*n_ghost;j++) {
        idx  = i + (j+n_ghost)*nx;
        gidx = i + j*n_ghost;
        for (ii=0; ii<n_fields; ii++) { 
          c_head[idx + ii*n_cells] = *(recv_buffer_x0 + gidx + ii*offset);
        }
      }
    }
  }
  // 3D
  if (nz > 1) {
    offset = n_ghost*(ny-2*n_ghost)*(nz-2*n_ghost);
    #pragma omp target teams distribute parallel for collapse ( 3 ) \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_x0, c_head )
    for(i=0;i<n_ghost;i++) {
      for(j=0;j<ny-2*n_ghost;j++) {
        for(k=0;k<nz-2*n_ghost;k++) {
          idx  = i + (j+n_ghost)*nx + (k+n_ghost)*nx*ny;
          gidx = i + j*n_ghost + k*n_ghost*(ny-2*n_ghost);
          for (ii=0; ii<n_fields; ii++) {
            c_head[idx + ii*n_cells] = *(recv_buffer_x0 + gidx + ii*offset);
          }
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_Buffer_X1 ( Real *recv_buffer_x1 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.density;

  // 1D
  if (ny == 1 && nz == 1) {
    offset = n_ghost;
    for(i=0;i<n_ghost;i++) {
      idx  = i+nx-n_ghost;
      gidx = i;
      for (ii=0; ii<n_fields; ii++) {
        c_head[idx + ii*n_cells] = *(recv_buffer_x1 + gidx + ii*offset);
      }
    }
  }
  // 2D
  if (ny > 1 && nz == 1) {
    offset = n_ghost*(ny-2*n_ghost);
    for(i=0;i<n_ghost;i++) {
      for (j=0;j<ny-2*n_ghost;j++) {
        idx  = i+nx-n_ghost + (j+n_ghost)*nx;
        gidx = i + j*n_ghost;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_x1 + gidx + ii*offset);
        }
      }
    }
  }
  // 3D
  if (nz > 1) {
    offset = n_ghost*(ny-2*n_ghost)*(nz-2*n_ghost);
    for(i=0;i<n_ghost;i++) {
      for(j=0;j<ny-2*n_ghost;j++) {
        for(k=0;k<nz-2*n_ghost;k++) {
          idx  = i+nx-n_ghost + (j+n_ghost)*nx + (k+n_ghost)*nx*ny;
          gidx = i + j*n_ghost + k*n_ghost*(ny-2*n_ghost);
          for (ii=0; ii<n_fields; ii++) {
            c_head[idx + ii*n_cells] = *(recv_buffer_x1 + gidx + ii*offset);
            
          }
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_DeviceBuffer_X1 ( Real *recv_buffer_x1 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.device;

  // 1D
  if (ny == 1 && nz == 1) {
    offset = n_ghost;
    #pragma omp target teams distribute parallel for \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_x1, c_head )
    for(i=0;i<n_ghost;i++) {
      idx  = i+nx-n_ghost;
      gidx = i;
      for (ii=0; ii<n_fields; ii++) {
        c_head[idx + ii*n_cells] = *(recv_buffer_x1 + gidx + ii*offset);
      }
    }
  }
  // 2D
  if (ny > 1 && nz == 1) {
    offset = n_ghost*(ny-2*n_ghost);
    #pragma omp target teams distribute parallel for collapse ( 2 ) \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_x1, c_head )
    for(i=0;i<n_ghost;i++) {
      for (j=0;j<ny-2*n_ghost;j++) {
        idx  = i+nx-n_ghost + (j+n_ghost)*nx;
        gidx = i + j*n_ghost;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_x1 + gidx + ii*offset);
        }
      }
    }
  }
  // 3D
  if (nz > 1) {
    offset = n_ghost*(ny-2*n_ghost)*(nz-2*n_ghost);
    #pragma omp target teams distribute parallel for collapse ( 3 ) \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_x1, c_head )
    for(i=0;i<n_ghost;i++) {
      for(j=0;j<ny-2*n_ghost;j++) {
        for(k=0;k<nz-2*n_ghost;k++) {
          idx  = i+nx-n_ghost + (j+n_ghost)*nx + (k+n_ghost)*nx*ny;
          gidx = i + j*n_ghost + k*n_ghost*(ny-2*n_ghost);
          for (ii=0; ii<n_fields; ii++) {
            c_head[idx + ii*n_cells] = *(recv_buffer_x1 + gidx + ii*offset);
            
          }
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_Buffer_Y0 ( Real *recv_buffer_y0 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.density;

  // 2D
  if (nz == 1) {
    offset = n_ghost*nx;
    for(i=0;i<nx;i++) {
      for (j=0;j<n_ghost;j++) {
        idx  = i + j*nx;
        gidx = i + j*nx;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_y0 + gidx + ii*offset);
        }
      }
    }
  }
  // 3D
  if (nz > 1) {
    offset = n_ghost*nx*(nz-2*n_ghost);
    for(i=0;i<nx;i++) {
      for(j=0;j<n_ghost;j++) {
        for(k=0;k<nz-2*n_ghost;k++) {
          idx  = i + j*nx + (k+n_ghost)*nx*ny;
          gidx = i + j*nx + k*nx*n_ghost;
          for (ii=0; ii<n_fields; ii++) {
            c_head[idx + ii*n_cells] = *(recv_buffer_y0 + gidx + ii*offset);
          }
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_DeviceBuffer_Y0 ( Real *recv_buffer_y0 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.device;

  // 2D
  if (nz == 1) {
    offset = n_ghost*nx;
    #pragma omp target teams distribute parallel for collapse ( 2 ) \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_y0, c_head )
    for(i=0;i<nx;i++) {
      for (j=0;j<n_ghost;j++) {
        idx  = i + j*nx;
        gidx = i + j*nx;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_y0 + gidx + ii*offset);
        }
      }
    }
  }
  // 3D
  if (nz > 1) {
    offset = n_ghost*nx*(nz-2*n_ghost);
    #pragma omp target teams distribute parallel for collapse ( 3 ) \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_y0, c_head )
    for(i=0;i<nx;i++) {
      for(j=0;j<n_ghost;j++) {
        for(k=0;k<nz-2*n_ghost;k++) {
          idx  = i + j*nx + (k+n_ghost)*nx*ny;
          gidx = i + j*nx + k*nx*n_ghost;
          for (ii=0; ii<n_fields; ii++) {
            c_head[idx + ii*n_cells] = *(recv_buffer_y0 + gidx + ii*offset);
          }
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_Buffer_Y1 ( Real *recv_buffer_y1 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.density;

  // 2D
  if (nz == 1) {
    offset = n_ghost*nx;
    for(i=0;i<nx;i++) {
      for (j=0;j<n_ghost;j++) {
        idx  = i + (j+ny-n_ghost)*nx;
        gidx = i + j*nx;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_y1 + gidx + ii*offset);
        }
      }
    }
  }
  // 3D
  if (nz > 1) {
    offset = n_ghost*nx*(nz-2*n_ghost);
    for(i=0;i<nx;i++) {
      for(j=0;j<n_ghost;j++) {
        for(k=0;k<nz-2*n_ghost;k++) {
          idx  = i + (j+ny-n_ghost)*nx + (k+n_ghost)*nx*ny;
          gidx = i + j*nx + k*nx*n_ghost;
          for (ii=0; ii<n_fields; ii++) {
            c_head[idx + ii*n_cells] = *(recv_buffer_y1 + gidx + ii*offset);
          }
        }
      }
    }
  }
  
}


void Grid3D::Unload_Hydro_DeviceBuffer_Y1 ( Real *recv_buffer_y1 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.device;

  // 2D
  if (nz == 1) {
    offset = n_ghost*nx;
    #pragma omp target teams distribute parallel for collapse ( 2 ) \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_y1, c_head )
    for(i=0;i<nx;i++) {
      for (j=0;j<n_ghost;j++) {
        idx  = i + (j+ny-n_ghost)*nx;
        gidx = i + j*nx;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_y1 + gidx + ii*offset);
        }
      }
    }
  }
  // 3D
  if (nz > 1) {
    offset = n_ghost*nx*(nz-2*n_ghost);
    #pragma omp target teams distribute parallel for collapse ( 3 ) \
        private ( idx, gidx, ii ) \
        firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
        is_device_ptr ( recv_buffer_y1, c_head )
    for(i=0;i<nx;i++) {
      for(j=0;j<n_ghost;j++) {
        for(k=0;k<nz-2*n_ghost;k++) {
          idx  = i + (j+ny-n_ghost)*nx + (k+n_ghost)*nx*ny;
          gidx = i + j*nx + k*nx*n_ghost;
          for (ii=0; ii<n_fields; ii++) {
            c_head[idx + ii*n_cells] = *(recv_buffer_y1 + gidx + ii*offset);
          }
        }
      }
    }
  }
  
}


void Grid3D::Unload_Hydro_Buffer_Z0 ( Real *recv_buffer_z0 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.density;

  offset = n_ghost*nx*ny;
  for(i=0;i<nx;i++) {
    for(j=0;j<ny;j++) {
      for(k=0;k<n_ghost;k++) {
        idx  = i + j*nx + k*nx*ny;
        gidx = i + j*nx + k*nx*ny;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_z0 + gidx + ii*offset);
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_DeviceBuffer_Z0 ( Real *recv_buffer_z0 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.device;

  offset = n_ghost*nx*ny;
  #pragma omp target teams distribute parallel for collapse ( 3 ) \
      private ( idx, gidx, ii ) \
      firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
      is_device_ptr ( recv_buffer_z0, c_head )
  for(i=0;i<nx;i++) {
    for(j=0;j<ny;j++) {
      for(k=0;k<n_ghost;k++) {
        idx  = i + j*nx + k*nx*ny;
        gidx = i + j*nx + k*nx*ny;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_z0 + gidx + ii*offset);
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_Buffer_Z1 ( Real *recv_buffer_z1 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.density;

  offset = n_ghost*nx*ny;
  for(i=0;i<nx;i++) {
    for(j=0;j<ny;j++) {
      for(k=0;k<n_ghost;k++) {
        idx  = i + j*nx + (k+nz-n_ghost)*nx*ny;
        gidx = i + j*nx + k*nx*ny;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_z1 + gidx + ii*offset);
        }
      }
    }
  }

}


void Grid3D::Unload_Hydro_DeviceBuffer_Z1 ( Real *recv_buffer_z1 ) {

  int i, j, k, ii;
  int idx;
  int gidx;
  int offset;
  int n_ghost, nx, ny, nz, n_fields, n_cells;
  Real *c_head;
  
  n_ghost   = H.n_ghost;
  nx        = H.nx;
  ny        = H.ny;
  nz        = H.nz;
  n_fields  = H.n_fields;	
  n_cells   = H.n_cells;
  
  c_head = (Real *) C.device;

  offset = n_ghost*nx*ny;
  #pragma omp target teams distribute parallel for collapse ( 3 ) \
      private ( idx, gidx, ii ) \
      firstprivate ( offset, n_ghost, nx, ny, nz, n_fields, n_cells ) \
      is_device_ptr ( recv_buffer_z1, c_head )
  for(i=0;i<nx;i++) {
    for(j=0;j<ny;j++) {
      for(k=0;k<n_ghost;k++) {
        idx  = i + j*nx + (k+nz-n_ghost)*nx*ny;
        gidx = i + j*nx + k*nx*ny;
        for (ii=0; ii<n_fields; ii++) {
          c_head[idx + ii*n_cells] = *(recv_buffer_z1 + gidx + ii*offset);
        }
      }
    }
  }

}

void Grid3D::Load_and_Send_MPI_Comm_Buffers_BLOCK(int dir, int *flags)
{
  
  #ifdef PARTICLES
  // Select wich particles need to be transfred for this direction
  // if ( Particles.TRANSFER_PARTICLES_BOUNDARIES) Particles.Select_Particles_to_Transfer( dir );
  
  // Initialize MPI requests for particles transfers
  int ireq_n_particles, ireq_particles_transfer;
  ireq_n_particles = 0;
  ireq_particles_transfer = 0;
  #endif
  
  int ireq;
  ireq = 0;
  
  int xbsize = x_buffer_length,
      ybsize = y_buffer_length,
      zbsize = z_buffer_length;

  int buffer_length;
  
  // Flag to ommit the transfer of the main buffer when tharnsfering the particles buffer
  bool transfer_main_buffer = true;
  
  /* x boundaries */
  if(dir == 0)
  {
    if (flags[0]==5) { 
      
      // load left x communication buffer
      if ( H.TRANSFER_HYDRO_BOUNDARIES ) 
        {
        #ifdef HYDRO_GPU
        buffer_length = Load_Hydro_DeviceBuffer_X0(d_send_buffer_x0);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_x0, d_send_buffer_x0, xbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Hydro_Buffer_X0(h_send_buffer_x0);
        #endif
        }
      
      #ifdef GRAVITY 
      if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
        #ifdef GRAVITY_GPU
        buffer_length = Load_Gravity_Potential_To_Buffer_GPU( 0, 0, d_send_buffer_x0, 0 );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_x0, d_send_buffer_x0, xbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Gravity_Potential_To_Buffer( 0, 0, h_send_buffer_x0, 0 );
        #endif
        
      }
      #ifdef SOR
      if ( Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES )  buffer_length = Load_Poisson_Boundary_To_Buffer( 0, 0, h_send_buffer_x0 );
      #endif //SOR
      #endif //GRAVITY
      
      #ifdef PARTICLES
      if ( Particles.TRANSFER_DENSITY_BOUNDARIES) {
        #ifdef PARTICLES_GPU
        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU( 0, 0, d_send_buffer_x0  );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_x0, d_send_buffer_x0, xbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Particles_Density_Boundary_to_Buffer( 0, 0, h_send_buffer_x0  );
        #endif
      } 
      else if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ){
        Load_and_Send_Particles_X0( ireq_n_particles, ireq_particles_transfer );
        transfer_main_buffer = false;
        ireq_n_particles ++;
        ireq_particles_transfer ++;
      }
      #endif
   
      if ( transfer_main_buffer ){
        #ifdef MPI_GPU
        //post non-blocking receive left x communication buffer
        MPI_Irecv(d_recv_buffer_x0, buffer_length, MPI_CHREAL, source[0], 0, 
                  world, &recv_request[ireq]);

        //non-blocking send left x communication buffer
        MPI_Isend(d_send_buffer_x0, buffer_length, MPI_CHREAL, dest[0], 1, 
                  world, &send_request[0]);
        #else
        //post non-blocking receive left x communication buffer
        MPI_Irecv(h_recv_buffer_x0, buffer_length, MPI_CHREAL, source[0], 0, 
                  world, &recv_request[ireq]);

        //non-blocking send left x communication buffer
        MPI_Isend(h_send_buffer_x0, buffer_length, MPI_CHREAL, dest[0], 1, 
                  world, &send_request[0]);
        #endif
        MPI_Request_free(send_request);

        //keep track of how many sends and receives are expected
        ireq++;
      }
    }

    if(flags[1]==5)
    {
      // load right x communication buffer
      if ( H.TRANSFER_HYDRO_BOUNDARIES ) 
        {
        #ifdef HYDRO_GPU
        buffer_length = Load_Hydro_DeviceBuffer_X1(d_send_buffer_x1);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_x1, d_send_buffer_x1, xbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Hydro_Buffer_X1(h_send_buffer_x1);
        #endif
        //printf("X1 len: %d\n", buffer_length);
        }
      
      #ifdef GRAVITY
      if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
        #ifdef GRAVITY_GPU
        buffer_length = Load_Gravity_Potential_To_Buffer_GPU( 0, 1, d_send_buffer_x1, 0 );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_x1, d_send_buffer_x1, xbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Gravity_Potential_To_Buffer( 0, 1, h_send_buffer_x1, 0 );
        #endif
      }
      #ifdef SOR
      if ( Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES )  buffer_length = Load_Poisson_Boundary_To_Buffer( 0, 1, h_send_buffer_x1 );
      #endif //SOR
      #endif //GRAVITY
      
      #ifdef PARTICLES
      if ( Particles.TRANSFER_DENSITY_BOUNDARIES) {
        #ifdef PARTICLES_GPU
        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU( 0, 1, d_send_buffer_x1  );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_x1, d_send_buffer_x1, xbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Particles_Density_Boundary_to_Buffer( 0, 1, h_send_buffer_x1  );
        #endif
      }
      else if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ){
        Load_and_Send_Particles_X1( ireq_n_particles, ireq_particles_transfer );
        transfer_main_buffer = false;
        ireq_n_particles ++;
        ireq_particles_transfer ++;
      }
      #endif
      
      if ( transfer_main_buffer ){
        #ifdef MPI_GPU
        //post non-blocking receive right x communication buffer
        MPI_Irecv(d_recv_buffer_x1, buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request[ireq]);

        //non-blocking send right x communication buffer
        MPI_Isend(d_send_buffer_x1, buffer_length, MPI_CHREAL, dest[1],   0, world, &send_request[1]);
        #else
        //post non-blocking receive right x communication buffer
        MPI_Irecv(h_recv_buffer_x1, buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request[ireq]);

        //non-blocking send right x communication buffer
        MPI_Isend(h_send_buffer_x1, buffer_length, MPI_CHREAL, dest[1],   0, world, &send_request[1]);
        #endif
        
        MPI_Request_free(send_request+1);

        //keep track of how many sends and receives are expected
        ireq++;
      }
    }
    // Receive the number of particles transfer for X
    #ifdef PARTICLES
    if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ) Wait_NTransfer_and_Request_Recv_Particles_Transfer_BLOCK( dir, flags );
    #endif

  }

  /* y boundaries */
  if (dir==1) {
    if(flags[2] == 5)
    {
      // load left y communication buffer
      if ( H.TRANSFER_HYDRO_BOUNDARIES ) 
        {
        #ifdef HYDRO_GPU
        buffer_length = Load_Hydro_DeviceBuffer_Y0(d_send_buffer_y0);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_y0, d_send_buffer_y0, ybsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Hydro_Buffer_Y0(h_send_buffer_y0);
        #endif
        //printf("Y0 len: %d\n", buffer_length);
        }
      
      #ifdef GRAVITY
      if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
        #ifdef GRAVITY_GPU
        buffer_length = Load_Gravity_Potential_To_Buffer_GPU( 1, 0, d_send_buffer_y0, 0 );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_y0, d_send_buffer_y0, ybsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Gravity_Potential_To_Buffer( 1, 0, h_send_buffer_y0, 0 );
        #endif
      }
      #ifdef SOR
      if ( Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES )  buffer_length = Load_Poisson_Boundary_To_Buffer( 1, 0, h_send_buffer_y0 );
      #endif //SOR
      #endif //GRAVITY
      
      #ifdef PARTICLES
      if ( Particles.TRANSFER_DENSITY_BOUNDARIES) {
        #ifdef PARTICLES_GPU
        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU( 1, 0, d_send_buffer_y0  );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_y0, d_send_buffer_y0, ybsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Particles_Density_Boundary_to_Buffer( 1, 0, h_send_buffer_y0  );
        #endif
      }
      else if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ){
        Load_and_Send_Particles_Y0( ireq_n_particles, ireq_particles_transfer );
        transfer_main_buffer = false;
        ireq_n_particles ++;
        ireq_particles_transfer ++;
      }
      #endif
      
      if ( transfer_main_buffer ){
        #ifdef MPI_GPU
        //post non-blocking receive left y communication buffer
        MPI_Irecv(d_recv_buffer_y0, buffer_length, MPI_CHREAL, source[2], 2, world, &recv_request[ireq]);

        //non-blocking send left y communication buffer
        MPI_Isend(d_send_buffer_y0, buffer_length, MPI_CHREAL, dest[2],   3, world, &send_request[0]);
        #else
        //post non-blocking receive left y communication buffer
        MPI_Irecv(h_recv_buffer_y0, buffer_length, MPI_CHREAL, source[2], 2, world, &recv_request[ireq]);

        //non-blocking send left y communication buffer
        MPI_Isend(h_send_buffer_y0, buffer_length, MPI_CHREAL, dest[2],   3, world, &send_request[0]);
        #endif
        
        MPI_Request_free(send_request);

        //keep track of how many sends and receives are expected
        ireq++;
      }
    }

    if(flags[3]==5)
    {
      // load right y communication buffer
      if ( H.TRANSFER_HYDRO_BOUNDARIES ) 
        {
        #ifdef HYDRO_GPU
        buffer_length = Load_Hydro_DeviceBuffer_Y1(d_send_buffer_y1);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_y1, d_send_buffer_y1, ybsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Hydro_Buffer_Y1(h_send_buffer_y1);
        #endif
        //printf("Y1 len: %d\n", buffer_length);
        }
        
      
      #ifdef GRAVITY
      if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
        #ifdef GRAVITY_GPU
        buffer_length = Load_Gravity_Potential_To_Buffer_GPU( 1, 1, d_send_buffer_y1, 0 );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_y1, d_send_buffer_y1, ybsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Gravity_Potential_To_Buffer( 1, 1, h_send_buffer_y1, 0 );
        #endif
      }
      #ifdef SOR
      if ( Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES )  buffer_length = Load_Poisson_Boundary_To_Buffer( 1, 1, h_send_buffer_y1 );
      #endif //SOR
      #endif //GRAVITY
      
      #ifdef PARTICLES
      if ( Particles.TRANSFER_DENSITY_BOUNDARIES) {
        #ifdef PARTICLES_GPU
        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU( 1, 1, d_send_buffer_y1  );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_y1, d_send_buffer_y1, ybsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Particles_Density_Boundary_to_Buffer( 1, 1, h_send_buffer_y1  );
        #endif
      }
      else if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ){
        Load_and_Send_Particles_Y1( ireq_n_particles, ireq_particles_transfer );
        transfer_main_buffer = false;
        ireq_n_particles ++;
        ireq_particles_transfer ++;
      }
      #endif
      
      if ( transfer_main_buffer ){      
        #ifdef MPI_GPU
        //post non-blocking receive right y communication buffer
        MPI_Irecv(d_recv_buffer_y1, buffer_length, MPI_CHREAL, source[3], 3, world, &recv_request[ireq]);

        //non-blocking send right y communication buffer
        MPI_Isend(d_send_buffer_y1, buffer_length, MPI_CHREAL, dest[3],   2, world, &send_request[1]);
        #else
        //post non-blocking receive right y communication buffer
        MPI_Irecv(h_recv_buffer_y1, buffer_length, MPI_CHREAL, source[3], 3, world, &recv_request[ireq]);

        //non-blocking send right y communication buffer
        MPI_Isend(h_send_buffer_y1, buffer_length, MPI_CHREAL, dest[3],   2, world, &send_request[1]);
        #endif
        MPI_Request_free(send_request+1);

        //keep track of how many sends and receives are expected
        ireq++;
      }
    }
    // Receive the number of particles transfer for Y
    #ifdef PARTICLES
    if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ) Wait_NTransfer_and_Request_Recv_Particles_Transfer_BLOCK( dir, flags );
    #endif

  }

  /* z boundaries */
  if (dir==2) {

    if(flags[4]==5)
    {
      // left z communication buffer
      if ( H.TRANSFER_HYDRO_BOUNDARIES ) 
        {
        #ifdef HYDRO_GPU
        buffer_length = Load_Hydro_DeviceBuffer_Z0(d_send_buffer_z0);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_z0, d_send_buffer_z0, zbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Hydro_Buffer_Z0(h_send_buffer_z0);
        #endif
        //printf("Z0 len: %d\n", buffer_length);
        }
      
      #ifdef GRAVITY
      if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
        #ifdef GRAVITY_GPU
        buffer_length = Load_Gravity_Potential_To_Buffer_GPU( 2, 0, d_send_buffer_z0, 0 );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_z0, d_send_buffer_z0, zbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Gravity_Potential_To_Buffer( 2, 0, h_send_buffer_z0, 0 );
        #endif
      }
      #ifdef SOR
      if ( Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES )  buffer_length = Load_Poisson_Boundary_To_Buffer( 2, 0, h_send_buffer_z0 );
      #endif //SOR
      #endif //GRAVITY
      
      #ifdef PARTICLES
      if ( Particles.TRANSFER_DENSITY_BOUNDARIES) {
        #ifdef PARTICLES_GPU
        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU( 2, 0, d_send_buffer_z0  );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_z0, d_send_buffer_z0, zbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Particles_Density_Boundary_to_Buffer( 2, 0, h_send_buffer_z0  );
        #endif
      }
      else if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ){
        Load_and_Send_Particles_Z0( ireq_n_particles, ireq_particles_transfer );
        transfer_main_buffer = false;
        ireq_n_particles ++;
        ireq_particles_transfer ++;
      }
      #endif
      
      if ( transfer_main_buffer ){
      
        #ifdef MPI_GPU
        //post non-blocking receive left z communication buffer
        MPI_Irecv(d_recv_buffer_z0, buffer_length, MPI_CHREAL, source[4], 4, world, &recv_request[ireq]);
        //non-blocking send left z communication buffer
        MPI_Isend(d_send_buffer_z0, buffer_length, MPI_CHREAL, dest[4],   5, world, &send_request[0]);
        #else
        //post non-blocking receive left z communication buffer
        MPI_Irecv(h_recv_buffer_z0, buffer_length, MPI_CHREAL, source[4], 4, world, &recv_request[ireq]);

        //non-blocking send left z communication buffer
        MPI_Isend(h_send_buffer_z0, buffer_length, MPI_CHREAL, dest[4],   5, world, &send_request[0]);
        #endif
        
        MPI_Request_free(send_request);

        //keep track of how many sends and receives are expected
        ireq++;
      }
    }

    if(flags[5]==5)
    {
      // load right z communication buffer
      if ( H.TRANSFER_HYDRO_BOUNDARIES ) 
        {
        #ifdef HYDRO_GPU
        buffer_length = Load_Hydro_DeviceBuffer_Z1(d_send_buffer_z1);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_z1, d_send_buffer_z1, zbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Hydro_Buffer_Z1(h_send_buffer_z1);
        #endif
        //printf("Z1 len: %d\n", buffer_length);
        }
              
      #ifdef GRAVITY
      if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
        #ifdef GRAVITY_GPU
        buffer_length = Load_Gravity_Potential_To_Buffer_GPU( 2, 1, d_send_buffer_z1, 0 );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_z1, d_send_buffer_z1, zbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Gravity_Potential_To_Buffer( 2, 1, h_send_buffer_z1, 0 );
        #endif
      }
      #ifdef SOR
      if ( Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES )  buffer_length = Load_Poisson_Boundary_To_Buffer( 2, 1, h_send_buffer_z1 );
      #endif //SOR
      #endif //GRAVITY
      
      #ifdef PARTICLES
      if ( Particles.TRANSFER_DENSITY_BOUNDARIES) {
        #ifdef PARTICLES_GPU
        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU( 2, 1, d_send_buffer_z1  );
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_z1, d_send_buffer_z1, zbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
          #endif
        #else
        buffer_length = Load_Particles_Density_Boundary_to_Buffer( 2, 1, h_send_buffer_z1  );
        #endif
      }
      else if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ){
        Load_and_Send_Particles_Z1( ireq_n_particles, ireq_particles_transfer );
        transfer_main_buffer = false;
        ireq_n_particles ++;
        ireq_particles_transfer ++;
      }
      #endif
      
      if ( transfer_main_buffer ){
        #ifdef MPI_GPU
        //post non-blocking receive right x communication buffer
        MPI_Irecv(d_recv_buffer_z1, buffer_length, MPI_CHREAL, source[5], 5, world, &recv_request[ireq]);

        //non-blocking send right x communication buffer
        MPI_Isend(d_send_buffer_z1, buffer_length, MPI_CHREAL, dest[5],   4, world, &send_request[1]);
        #else
        //post non-blocking receive right x communication buffer
        MPI_Irecv(h_recv_buffer_z1, buffer_length, MPI_CHREAL, source[5], 5, world, &recv_request[ireq]);

        //non-blocking send right x communication buffer
        MPI_Isend(h_send_buffer_z1, buffer_length, MPI_CHREAL, dest[5],   4, world, &send_request[1]);
        #endif
        MPI_Request_free(send_request+1);

        //keep track of how many sends and receives are expected
        ireq++;
      }
    }
    // Receive the number of particles transfer for Z
      #ifdef PARTICLES
      if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ) Wait_NTransfer_and_Request_Recv_Particles_Transfer_BLOCK( dir, flags );
      #endif
  }

}


void Grid3D::Wait_and_Unload_MPI_Comm_Buffers_SLAB(int *flags)
{
  int iwait;
  int index = 0;
  int wait_max=0;
  MPI_Status status;

  //find out how many recvs we need to wait for
  for(iwait=0;iwait<6;iwait++)
    if(flags[iwait] == 5) //there is communication on this face
      wait_max++;   //so we'll need to wait for its comm

  //wait for any receives to complete
  for(iwait=0;iwait<wait_max;iwait++)
  {
    //wait for recv completion
    MPI_Waitany(wait_max,recv_request,&index,&status);

    //depending on which face arrived, load the buffer into the ghost grid
    Unload_MPI_Comm_Buffers(status.MPI_TAG);
  }

}


void Grid3D::Wait_and_Unload_MPI_Comm_Buffers_BLOCK(int dir, int *flags)
{
  
  #ifdef PARTICLES
  // If we are transfering the particles buffers we dont need to unload the main buffers
  if ( Particles.TRANSFER_PARTICLES_BOUNDARIES ) return;
  #endif
  
  int iwait;
  int index = 0;
  int wait_max=0;
  MPI_Status status;

  //find out how many recvs we need to wait for
  if (dir==0) {
    if(flags[0] == 5) //there is communication on this face
      wait_max++;   //so we'll need to wait for its comm
    if(flags[1] == 5) //there is communication on this face
      wait_max++;   //so we'll need to wait for its comm
  }
  if (dir==1) {
    if(flags[2] == 5) //there is communication on this face
      wait_max++;   //so we'll need to wait for its comm
    if(flags[3] == 5) //there is communication on this face
      wait_max++;   //so we'll need to wait for its comm
  }
  if (dir==2) {
    if(flags[4] == 5) //there is communication on this face
      wait_max++;   //so we'll need to wait for its comm
    if(flags[5] == 5) //there is communication on this face
      wait_max++;   //so we'll need to wait for its comm
  }

  //wait for any receives to complete
  for(iwait=0;iwait<wait_max;iwait++)
  {
    //wait for recv completion
    MPI_Waitany(wait_max,recv_request,&index,&status);
    //if (procID==1) MPI_Get_count(&status, MPI_CHREAL, &count);
    //if (procID==1) printf("Process 1 unloading direction %d, source %d, index %d, length %d.\n", status.MPI_TAG, status.MPI_SOURCE, index, count);
    //depending on which face arrived, load the buffer into the ghost grid
    Unload_MPI_Comm_Buffers(status.MPI_TAG);
  }
}



void Grid3D::Unload_MPI_Comm_Buffers(int index)
{
  switch(flag_decomp)
  {
    case SLAB_DECOMP:
      Unload_MPI_Comm_Buffers_SLAB(index);
      break;
    case BLOCK_DECOMP:
      Unload_MPI_Comm_Buffers_BLOCK(index);
      break;
  }
}


void Grid3D::Unload_MPI_Comm_Buffers_SLAB(int index)
{
  int i, j, k, ii;
  int idx;
  int gidx;
  int offset = H.n_ghost*H.ny*H.nz;

  //left face
  if(index==0)
  {
    //load left x communication buffer
    for(i=0;i<H.n_ghost;i++) {
      for(j=0;j<H.ny;j++) {
        for(k=0;k<H.nz;k++)
        {
          idx  = i + j*H.nx + k*H.nx*H.ny;
          gidx = i + j*H.n_ghost + k*H.n_ghost*H.ny;
          for (ii=0; ii<H.n_fields; ii++) {
            C.density[idx + ii*H.n_cells] = *(recv_buffer_0 + gidx + ii*offset);
          }
        }
      }
    }
  }

  //right face
  if(index==1)
  {
    //load left x communication buffer
    for(i=0;i<H.n_ghost;i++) {
      for(j=0;j<H.ny;j++) {
        for(k=0;k<H.nz;k++)
        {
          idx  = (i + H.nx - H.n_ghost) + j*H.nx + k*H.nx*H.ny;
          gidx = i                      + j*H.n_ghost + k*H.n_ghost*H.ny;
          for (ii=0; ii<H.n_fields; ii++) {
            C.density[idx + ii*H.n_cells] = *(recv_buffer_1 + gidx + ii*offset);
          }
        }
      }
    }
  }

  //done unloading
}


void Grid3D::Unload_MPI_Comm_Buffers_BLOCK(int index)
{
  
  // local recv buffers
  Real *l_recv_buffer_x0, *l_recv_buffer_x1, *l_recv_buffer_y0,
       *l_recv_buffer_y1, *l_recv_buffer_z0, *l_recv_buffer_z1;
       
  Grid3D_PMF_UnloadHydroBuffer Fptr_Unload_Hydro_Buffer_X0,
                               Fptr_Unload_Hydro_Buffer_X1,
                               Fptr_Unload_Hydro_Buffer_Y0,
                               Fptr_Unload_Hydro_Buffer_Y1,
                               Fptr_Unload_Hydro_Buffer_Z0,
                               Fptr_Unload_Hydro_Buffer_Z1;
  
  Grid3D_PMF_UnloadGravityPotential Fptr_Unload_Gravity_Potential;
  Grid3D_PMF_UnloadParticleDensity Fptr_Unload_Particle_Density;
       
  if ( H.TRANSFER_HYDRO_BOUNDARIES ) {
    #ifdef HYDRO_GPU
      #ifndef MPI_GPU
      copyHostToDeviceReceiveBuffer ( index );
      #endif 
    l_recv_buffer_x0 = d_recv_buffer_x0;
    l_recv_buffer_x1 = d_recv_buffer_x1;
    l_recv_buffer_y0 = d_recv_buffer_y0;
    l_recv_buffer_y1 = d_recv_buffer_y1;
    l_recv_buffer_z0 = d_recv_buffer_z0;
    l_recv_buffer_z1 = d_recv_buffer_z1;
    
    Fptr_Unload_Hydro_Buffer_X0 = &Grid3D::Unload_Hydro_DeviceBuffer_X0;
    Fptr_Unload_Hydro_Buffer_X1 = &Grid3D::Unload_Hydro_DeviceBuffer_X1;
    Fptr_Unload_Hydro_Buffer_Y0 = &Grid3D::Unload_Hydro_DeviceBuffer_Y0;
    Fptr_Unload_Hydro_Buffer_Y1 = &Grid3D::Unload_Hydro_DeviceBuffer_Y1;
    Fptr_Unload_Hydro_Buffer_Z0 = &Grid3D::Unload_Hydro_DeviceBuffer_Z0;
    Fptr_Unload_Hydro_Buffer_Z1 = &Grid3D::Unload_Hydro_DeviceBuffer_Z1;
    
    #else
    
    l_recv_buffer_x0 = h_recv_buffer_x0;
    l_recv_buffer_x1 = h_recv_buffer_x1;
    l_recv_buffer_y0 = h_recv_buffer_y0;
    l_recv_buffer_y1 = h_recv_buffer_y1;
    l_recv_buffer_z0 = h_recv_buffer_z0;
    l_recv_buffer_z1 = h_recv_buffer_z1;    
    
    Fptr_Unload_Hydro_Buffer_X0 = &Grid3D::Unload_Hydro_Buffer_X0;
    Fptr_Unload_Hydro_Buffer_X1 = &Grid3D::Unload_Hydro_Buffer_X1;
    Fptr_Unload_Hydro_Buffer_Y0 = &Grid3D::Unload_Hydro_Buffer_Y0;
    Fptr_Unload_Hydro_Buffer_Y1 = &Grid3D::Unload_Hydro_Buffer_Y1;
    Fptr_Unload_Hydro_Buffer_Z0 = &Grid3D::Unload_Hydro_Buffer_Z0;
    Fptr_Unload_Hydro_Buffer_Z1 = &Grid3D::Unload_Hydro_Buffer_Z1;
    
    #endif // HYDRO_GPU
    
    switch ( index ) {
    case ( 0 ): (this->*Fptr_Unload_Hydro_Buffer_X0) ( l_recv_buffer_x0 ); break;
    case ( 1 ): (this->*Fptr_Unload_Hydro_Buffer_X1) ( l_recv_buffer_x1 ); break;
    case ( 2 ): (this->*Fptr_Unload_Hydro_Buffer_Y0) ( l_recv_buffer_y0 ); break;
    case ( 3 ): (this->*Fptr_Unload_Hydro_Buffer_Y1) ( l_recv_buffer_y1 ); break;
    case ( 4 ): (this->*Fptr_Unload_Hydro_Buffer_Z0) ( l_recv_buffer_z0 ); break;
    case ( 5 ): (this->*Fptr_Unload_Hydro_Buffer_Z1) ( l_recv_buffer_z1 ); break;
    }
  }
  
  #ifdef GRAVITY
  if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
    #ifdef GRAVITY_GPU
     #ifndef MPI_GPU
     copyHostToDeviceReceiveBuffer ( index );
     #endif // MPI_GPU
    
    l_recv_buffer_x0 = d_recv_buffer_x0;
    l_recv_buffer_x1 = d_recv_buffer_x1;
    l_recv_buffer_y0 = d_recv_buffer_y0;
    l_recv_buffer_y1 = d_recv_buffer_y1;
    l_recv_buffer_z0 = d_recv_buffer_z0;
    l_recv_buffer_z1 = d_recv_buffer_z1;
    
    Fptr_Unload_Gravity_Potential 
      = &Grid3D::Unload_Gravity_Potential_from_Buffer_GPU;
    
    #else

    l_recv_buffer_x0 = h_recv_buffer_x0;
    l_recv_buffer_x1 = h_recv_buffer_x1;
    l_recv_buffer_y0 = h_recv_buffer_y0;
    l_recv_buffer_y1 = h_recv_buffer_y1;
    l_recv_buffer_z0 = h_recv_buffer_z0;
    l_recv_buffer_z1 = h_recv_buffer_z1;
    
    Fptr_Unload_Gravity_Potential 
      = &Grid3D::Unload_Gravity_Potential_from_Buffer;
    
    #endif // GRAVITY_GPU
    
    if ( index == 0 ) (this->*Fptr_Unload_Gravity_Potential)( 0, 0, l_recv_buffer_x0, 0  );
    if ( index == 1 ) (this->*Fptr_Unload_Gravity_Potential)( 0, 1, l_recv_buffer_x1, 0  );
    if ( index == 2 ) (this->*Fptr_Unload_Gravity_Potential)( 1, 0, l_recv_buffer_y0, 0  );
    if ( index == 3 ) (this->*Fptr_Unload_Gravity_Potential)( 1, 1, l_recv_buffer_y1, 0  );
    if ( index == 4 ) (this->*Fptr_Unload_Gravity_Potential)( 2, 0, l_recv_buffer_z0, 0  );
    if ( index == 5 ) (this->*Fptr_Unload_Gravity_Potential)( 2, 1, l_recv_buffer_z1, 0  );
  }
  
  #ifdef SOR
  if ( Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES ){
    l_recv_buffer_x0 = h_recv_buffer_x0;
    l_recv_buffer_x1 = h_recv_buffer_x1;
    l_recv_buffer_y0 = h_recv_buffer_y0;
    l_recv_buffer_y1 = h_recv_buffer_y1;
    l_recv_buffer_z0 = h_recv_buffer_z0;
    l_recv_buffer_z1 = h_recv_buffer_z1;
    
    if ( index == 0 ) Unload_Poisson_Boundary_From_Buffer( 0, 0, l_recv_buffer_x0 );
    if ( index == 1 ) Unload_Poisson_Boundary_From_Buffer( 0, 1, l_recv_buffer_x1 );
    if ( index == 2 ) Unload_Poisson_Boundary_From_Buffer( 1, 0, l_recv_buffer_y0 );
    if ( index == 3 ) Unload_Poisson_Boundary_From_Buffer( 1, 1, l_recv_buffer_y1 );
    if ( index == 4 ) Unload_Poisson_Boundary_From_Buffer( 2, 0, l_recv_buffer_z0 );
    if ( index == 5 ) Unload_Poisson_Boundary_From_Buffer( 2, 1, l_recv_buffer_z1 );
  }
  #endif //SOR
  
  #endif  //GRAVITY
  
  
  #ifdef PARTICLES
  if (  Particles.TRANSFER_DENSITY_BOUNDARIES ){
    #ifdef PARTICLES_GPU
      #ifndef MPI_GPU
      copyHostToDeviceReceiveBuffer ( index );
      #endif
    
    l_recv_buffer_x0 = d_recv_buffer_x0;
    l_recv_buffer_x1 = d_recv_buffer_x1;
    l_recv_buffer_y0 = d_recv_buffer_y0;
    l_recv_buffer_y1 = d_recv_buffer_y1;
    l_recv_buffer_z0 = d_recv_buffer_z0;
    l_recv_buffer_z1 = d_recv_buffer_z1;
    
    Fptr_Unload_Particle_Density 
      = &Grid3D::Unload_Particles_Density_Boundary_From_Buffer_GPU;
    
    #else
    
    l_recv_buffer_x0 = d_recv_buffer_x0;
    l_recv_buffer_x1 = d_recv_buffer_x1;
    l_recv_buffer_y0 = d_recv_buffer_y0;
    l_recv_buffer_y1 = d_recv_buffer_y1;
    l_recv_buffer_z0 = d_recv_buffer_z0;
    l_recv_buffer_z1 = d_recv_buffer_z1;
    
    Fptr_Unload_Particle_Density 
      = &Grid3D::Unload_Particles_Density_Boundary_From_Buffer;
    
    #endif // PARTICLES_GPU
    
    if ( index == 0 ) (this->*Fptr_Unload_Particle_Density)( 0, 0, l_recv_buffer_x0 );
    if ( index == 1 ) (this->*Fptr_Unload_Particle_Density)( 0, 1, l_recv_buffer_x1 );
    if ( index == 2 ) (this->*Fptr_Unload_Particle_Density)( 1, 0, l_recv_buffer_y0 );
    if ( index == 3 ) (this->*Fptr_Unload_Particle_Density)( 1, 1, l_recv_buffer_y1 );
    if ( index == 4 ) (this->*Fptr_Unload_Particle_Density)( 2, 0, l_recv_buffer_z0 );
    if ( index == 5 ) (this->*Fptr_Unload_Particle_Density)( 2, 1, l_recv_buffer_z1 );
  }
  
  #endif  //PARTICLES

}

#endif /*MPI_CHOLLA*/

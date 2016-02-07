#include"grid3D.h"
#include"mpi_routines.h"
#include"io.h"
#include"error_handling.h"
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
}


void Grid3D::Set_Boundaries_MPI_SLAB(int *flags, struct parameters P)
{
  Real tcomm_start, tcomm_end;
  Real tother_start, tother_end;

  t_comm = t_other = 0;

  //perform boundary conditions around
  //edges of communication region on x faces

  tother_start = get_time();
  if(flags[0]==5)
    Set_Edge_Boundaries(0,flags);
  if(flags[1]==5)
    Set_Edge_Boundaries(1,flags);
  tother_end = get_time();
  t_other += tother_end - tother_start;

  /*

  550011111111112233 223344444444445500
  5500          2233 2233    5500
  5500          2233 2233    5500
  5500          2233 2233    5500
  5500          2233 2233    5500
  550011111111112233 223344444444445500

  */

  //printf("proc %d about to load and send MPI buffers\n",procID);
  //fflush(stdout);
  tcomm_start = get_time();
  //1) load and post comm for buffers
  Load_and_Send_MPI_Comm_Buffers(0, flags);
  tcomm_end = get_time();
  t_comm = tcomm_end - tcomm_start;

  //printf("proc %d about to perform other non-MPI boundary conditions\n",procID);

  //3) perform any additional boundary conditions
  //including whether the x face is non-MPI
  tother_start = get_time();
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
  tother_end = get_time();
  t_other += tother_end - tother_start;
  //printf("proc %d other boundaries: %9.4f\n", procID, t_other);
  //fflush(stdout);

  tcomm_start = get_time();
  //4) wait for sends and receives to finish
  //   and then load ghost cells from comm buffers
  if(flags[0]==5 || flags[1]==5)
    Wait_and_Unload_MPI_Comm_Buffers_SLAB(flags);
  tcomm_end = get_time();
  t_comm += tcomm_end - tcomm_start;
  //printf("proc %d communications: %9.4f\n", procID, t_comm);
  //fflush(stdout);
    //Set_Boundaries(2,flags);
    //Set_Boundaries(3,flags);
}


void Grid3D::Set_Boundaries_MPI_BLOCK(int *flags, struct parameters P)
{
  Real tcomm_start, tcomm_end;
  Real tother_start, tother_end;

  t_comm = t_other = 0;

  if (H.nx > 1) {

    /* Step 1 - Send MPI x-boundaries */
    if (flags[0]==5 || flags[1]==5) {
      tcomm_start = get_time();
      Load_and_Send_MPI_Comm_Buffers(0, flags);
      tcomm_end = get_time();
      t_comm += tcomm_end - tcomm_start;
    }

    /* Step 2 - Set non-MPI x-boundaries */
    tother_start = get_time();
    Set_Boundaries(0, flags);
    Set_Boundaries(1, flags);
    tother_end = get_time();
    t_other += tother_end - tother_start;

    /* Step 3 - Receive MPI x-boundaries */
    if (flags[0]==5 || flags[1]==5) {
      tcomm_start = get_time();
      Wait_and_Unload_MPI_Comm_Buffers_BLOCK(0, flags);
      tcomm_end = get_time();
      t_comm += tcomm_end - tcomm_start;
    }
  }
  MPI_Barrier(world);
  if (H.ny > 1) {

    /* Step 4 - Send MPI y-boundaries */
    if (flags[2]==5 || flags[3]==5) {
      tcomm_start = get_time();
      Load_and_Send_MPI_Comm_Buffers(1, flags);
      tcomm_end = get_time();
      t_comm += tcomm_end - tcomm_start;
    }

    /* Step 5 - Set non-MPI y-boundaries */
    tother_start = get_time();
    Set_Boundaries(2, flags);
    Set_Boundaries(3, flags);
    tother_end = get_time();
    t_other += tother_end - tother_start;

    /* Step 6 - Receive MPI y-boundaries */
    if (flags[2]==5 || flags[3]==5) {
      tcomm_start = get_time();
      Wait_and_Unload_MPI_Comm_Buffers_BLOCK(1, flags);
      tcomm_end = get_time();
      t_comm += tcomm_end - tcomm_start;
    }
  }
  MPI_Barrier(world);
  if (H.nz > 1) {

    /* Step 7 - Send MPI z-boundaries */
    if (flags[4]==5 || flags[5]==5) {
      tcomm_start = get_time();
      Load_and_Send_MPI_Comm_Buffers(2, flags);
      tcomm_end = get_time();
      t_comm += tcomm_end - tcomm_start;
    }

    /* Step 8 - Set non-MPI z-boundaries */
    tother_start = get_time();
    Set_Boundaries(4, flags);
    Set_Boundaries(5, flags);
    tother_end = get_time();
    t_other += tother_end - tother_start;

    /* Step 9 - Receive MPI z-boundaries */
    if (flags[4]==5 || flags[5]==5) {
      tcomm_start = get_time();
      Wait_and_Unload_MPI_Comm_Buffers_BLOCK(2, flags);
      tcomm_end = get_time();
      t_comm += tcomm_end - tcomm_start;
    }
  }

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
          }
        }
      }
    }

    //printf("proc %d dir %d iedge %d imin %d %d %d imax %d %d %d\n",procID,dir,iedge,imin[0],imin[1],imin[2],imax[0],imax[1],imax[2]);
    //fflush(stdout);
  }

  //MPI_Finalize();
  //exit(0);
}


/*! \fn Set_Edge_Boundary_Extents(int dir, int edge, int *imin, int *imax)
 *  \brief Set the extents of the edge and corner ghost regions we want to
 *         initialize first, so we can then do communication */
void Grid3D::Set_Edge_Boundary_Extents(int dir, int edge, int *imin, int *imax)
{

  int i, j, k;
  int ni, nj, nk;
  int nx, ny, nz;


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

  int i, j, k;
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

          *(send_buffer_0 + gidx + 0*offset) = C.density[idx];
          *(send_buffer_0 + gidx + 1*offset) = C.momentum_x[idx];
          *(send_buffer_0 + gidx + 2*offset) = C.momentum_y[idx];
          *(send_buffer_0 + gidx + 3*offset) = C.momentum_z[idx];
          *(send_buffer_0 + gidx + 4*offset) = C.Energy[idx];
          #ifdef DE
          *(send_buffer_0 + gidx + 5*offset) = C.GasEnergy[idx];
          #endif
        }
      }
    }


    //post non-blocking receive left x communication buffer
    MPI_Irecv(recv_buffer_0, recv_buffer_length, MPI_CHREAL, source[0], 0, world, &recv_request[ireq]);

    //non-blocking send left x communication buffer
    MPI_Isend(send_buffer_0, send_buffer_length, MPI_CHREAL, dest[0],   1,    world, &send_request[0]);

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
  
          *(send_buffer_1 + gidx + 0*offset) = C.density[idx];
          *(send_buffer_1 + gidx + 1*offset) = C.momentum_x[idx];
          *(send_buffer_1 + gidx + 2*offset) = C.momentum_y[idx];
          *(send_buffer_1 + gidx + 3*offset) = C.momentum_z[idx];
          *(send_buffer_1 + gidx + 4*offset) = C.Energy[idx];
          #ifdef DE
          *(send_buffer_1 + gidx + 5*offset) = C.GasEnergy[idx];
          #endif
        }
      }
    }


    //post non-blocking receive right x communication buffer
    MPI_Irecv(recv_buffer_1, recv_buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request[ireq]);

    //non-blocking send right x communication buffer
    MPI_Isend(send_buffer_1, send_buffer_length, MPI_CHREAL, dest[1], 0, world, &send_request[1]);

    //remember how many recv's this proc expects
    ireq++;
  }

  //done!
}


void Grid3D::Load_and_Send_MPI_Comm_Buffers_BLOCK(int dir, int *flags)
{
  int i, j, k;
  int gidx;
  int idx;
  int offset;
  int ireq;
  ireq = 0;

  /* x boundaries */
  if(dir == 0)
  {
    if (flags[0]==5) { 
      // load left x communication buffer
      // 1D
      if (H.ny == 1 && H.nz == 1) {
        offset = H.n_ghost;
        for (i=0;i<H.n_ghost;i++) {
          idx = (i+H.n_ghost);
          gidx = i;
          *(send_buffer_x0 + gidx + 0*offset) = C.density[idx];
          *(send_buffer_x0 + gidx + 1*offset) = C.momentum_x[idx];
          *(send_buffer_x0 + gidx + 2*offset) = C.momentum_y[idx];
          *(send_buffer_x0 + gidx + 3*offset) = C.momentum_z[idx];
          *(send_buffer_x0 + gidx + 4*offset) = C.Energy[idx];
          #ifdef DE
          *(send_buffer_x0 + gidx + 5*offset) = C.GasEnergy[idx];
          #endif
        }
      }
      // 2D
      if (H.ny > 1 && H.nz == 1) {
        offset = H.n_ghost*(H.ny-2*H.n_ghost);
        for (i=0;i<H.n_ghost;i++) {
          for (j=0;j<H.ny-2*H.n_ghost;j++) {
            idx = (i+H.n_ghost) + (j+H.n_ghost)*H.nx;
            gidx = i + j*H.n_ghost;
            *(send_buffer_x0 + gidx + 0*offset) = C.density[idx];
            *(send_buffer_x0 + gidx + 1*offset) = C.momentum_x[idx];
            *(send_buffer_x0 + gidx + 2*offset) = C.momentum_y[idx];
            *(send_buffer_x0 + gidx + 3*offset) = C.momentum_z[idx];
            *(send_buffer_x0 + gidx + 4*offset) = C.Energy[idx];
            #ifdef DE
            *(send_buffer_x0 + gidx + 5*offset) = C.GasEnergy[idx];
            #endif
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
              *(send_buffer_x0 + gidx + 0*offset) = C.density[idx];
              *(send_buffer_x0 + gidx + 1*offset) = C.momentum_x[idx];
              *(send_buffer_x0 + gidx + 2*offset) = C.momentum_y[idx];
              *(send_buffer_x0 + gidx + 3*offset) = C.momentum_z[idx];
              *(send_buffer_x0 + gidx + 4*offset) = C.Energy[idx];
              #ifdef DE
              *(send_buffer_x0 + gidx + 5*offset) = C.GasEnergy[idx];
              #endif
            }
          }
        }
      }

   
      //post non-blocking receive left x communication buffer
      MPI_Irecv(recv_buffer_x0, x_buffer_length, MPI_CHREAL, source[0], 0, world, &recv_request[ireq]);

      //non-blocking send left x communication buffer
      MPI_Isend(send_buffer_x0, x_buffer_length, MPI_CHREAL, dest[0],   1, world, &send_request[0]);

      //keep track of how many sends and receives are expected
      ireq++;
    }

    if(flags[1]==5)
    {
      // load right x communication buffer
      // 1D
      if (H.ny == 1 && H.nz == 1) {
        offset = H.n_ghost;
        for (i=0;i<H.n_ghost;i++) {
          idx = (i+H.nx-2*H.n_ghost);
          gidx = i;
          *(send_buffer_x1 + gidx + 0*offset) = C.density[idx];
          *(send_buffer_x1 + gidx + 1*offset) = C.momentum_x[idx];
          *(send_buffer_x1 + gidx + 2*offset) = C.momentum_y[idx];
          *(send_buffer_x1 + gidx + 3*offset) = C.momentum_z[idx];
          *(send_buffer_x1 + gidx + 4*offset) = C.Energy[idx];
          #ifdef DE
          *(send_buffer_x1 + gidx + 5*offset) = C.GasEnergy[idx];
          #endif
        }
      }
      // 2D
      if (H.ny > 1 && H.nz == 1) {
        offset = H.n_ghost*(H.ny-2*H.n_ghost);
        for (i=0;i<H.n_ghost;i++) {
          for (j=0;j<H.ny-2*H.n_ghost;j++) {
            idx = (i+H.nx-2*H.n_ghost) + (j+H.n_ghost)*H.nx;
            gidx = i + j*H.n_ghost;
            *(send_buffer_x1 + gidx + 0*offset) = C.density[idx];
            *(send_buffer_x1 + gidx + 1*offset) = C.momentum_x[idx];
            *(send_buffer_x1 + gidx + 2*offset) = C.momentum_y[idx];
            *(send_buffer_x1 + gidx + 3*offset) = C.momentum_z[idx];
            *(send_buffer_x1 + gidx + 4*offset) = C.Energy[idx];
            #ifdef DE
            *(send_buffer_x1 + gidx + 5*offset) = C.GasEnergy[idx];
            #endif
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
              *(send_buffer_x1 + gidx + 0*offset) = C.density[idx];
              *(send_buffer_x1 + gidx + 1*offset) = C.momentum_x[idx];
              *(send_buffer_x1 + gidx + 2*offset) = C.momentum_y[idx];
              *(send_buffer_x1 + gidx + 3*offset) = C.momentum_z[idx];
              *(send_buffer_x1 + gidx + 4*offset) = C.Energy[idx];
              #ifdef DE
              *(send_buffer_x1 + gidx + 5*offset) = C.GasEnergy[idx];
              #endif
            }
          }
        }
      }

      //post non-blocking receive right x communication buffer
      MPI_Irecv(recv_buffer_x1, x_buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request[ireq]);

      //non-blocking send right x communication buffer
      MPI_Isend(send_buffer_x1, x_buffer_length, MPI_CHREAL, dest[1],   0, world, &send_request[1]);

      //keep track of how many sends and receives are expected
      ireq++;
    }

  }

  /* y boundaries */
  if (dir==1) {
    // load left y communication buffer
    if(flags[2] == 5)
    {
      // 2D
      if (H.nz == 1) {
        offset = H.n_ghost*H.nx;
        for (i=0;i<H.nx;i++) {
          for (j=0;j<H.n_ghost;j++) {
            idx = i + (j+H.n_ghost)*H.nx;
            gidx = i + j*H.nx;
            *(send_buffer_y0 + gidx + 0*offset) = C.density[idx];
            *(send_buffer_y0 + gidx + 1*offset) = C.momentum_x[idx];
            *(send_buffer_y0 + gidx + 2*offset) = C.momentum_y[idx];
            *(send_buffer_y0 + gidx + 3*offset) = C.momentum_z[idx];
            *(send_buffer_y0 + gidx + 4*offset) = C.Energy[idx];
            #ifdef DE
            *(send_buffer_y0 + gidx + 5*offset) = C.GasEnergy[idx];
            #endif
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
              *(send_buffer_y0 + gidx + 0*offset) = C.density[idx];
              *(send_buffer_y0 + gidx + 1*offset) = C.momentum_x[idx];
              *(send_buffer_y0 + gidx + 2*offset) = C.momentum_y[idx];
              *(send_buffer_y0 + gidx + 3*offset) = C.momentum_z[idx];
              *(send_buffer_y0 + gidx + 4*offset) = C.Energy[idx];
              #ifdef DE
              *(send_buffer_y0 + gidx + 5*offset) = C.GasEnergy[idx];
              #endif
            }
          }
        }
      }
      
      //post non-blocking receive left y communication buffer
      MPI_Irecv(recv_buffer_y0, y_buffer_length, MPI_CHREAL, source[2], 2, world, &recv_request[ireq]);

      //non-blocking send left y communication buffer
      MPI_Isend(send_buffer_y0, y_buffer_length, MPI_CHREAL, dest[2],   3, world, &send_request[0]);

      //keep track of how many sends and receives are expected
      ireq++;
    }

    // load right y communication buffer
    if(flags[3]==5)
    {
      // 2D
      if (H.nz == 1) {
        offset = H.n_ghost*H.nx;
        for (i=0;i<H.nx;i++) {
          for (j=0;j<H.n_ghost;j++) {
            idx = i + (j+H.ny-2*H.n_ghost)*H.nx;
            gidx = i + j*H.nx;
            *(send_buffer_y1 + gidx + 0*offset) = C.density[idx];
            *(send_buffer_y1 + gidx + 1*offset) = C.momentum_x[idx];
            *(send_buffer_y1 + gidx + 2*offset) = C.momentum_y[idx];
            *(send_buffer_y1 + gidx + 3*offset) = C.momentum_z[idx];
            *(send_buffer_y1 + gidx + 4*offset) = C.Energy[idx];
            #ifdef DE
            *(send_buffer_y1 + gidx + 5*offset) = C.GasEnergy[idx];
            #endif
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
              *(send_buffer_y1 + gidx + 0*offset) = C.density[idx];
              *(send_buffer_y1 + gidx + 1*offset) = C.momentum_x[idx];
              *(send_buffer_y1 + gidx + 2*offset) = C.momentum_y[idx];
              *(send_buffer_y1 + gidx + 3*offset) = C.momentum_z[idx];
              *(send_buffer_y1 + gidx + 4*offset) = C.Energy[idx];
              #ifdef DE
              *(send_buffer_y1 + gidx + 5*offset) = C.GasEnergy[idx];
              #endif
            }
          }
        }
      }
      
      //post non-blocking receive right y communication buffer
      MPI_Irecv(recv_buffer_y1, y_buffer_length, MPI_CHREAL, source[3], 3, world, &recv_request[ireq]);

      //non-blocking send right y communication buffer
      MPI_Isend(send_buffer_y1, y_buffer_length, MPI_CHREAL, dest[3],   2, world, &send_request[1]);

      //keep track of how many sends and receives are expected
      ireq++;
    }
  }

  /* z boundaries */
  if (dir==2) {

    // left z communication buffer
    if(flags[4]==5)
    {
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
            *(send_buffer_z0 + gidx + 0*offset) = C.density[idx];
            *(send_buffer_z0 + gidx + 1*offset) = C.momentum_x[idx];
            *(send_buffer_z0 + gidx + 2*offset) = C.momentum_y[idx];
            *(send_buffer_z0 + gidx + 3*offset) = C.momentum_z[idx];
            *(send_buffer_z0 + gidx + 4*offset) = C.Energy[idx];
            #ifdef DE
            *(send_buffer_z0 + gidx + 5*offset) = C.GasEnergy[idx];
            #endif
          }
        }
      }
      
      //post non-blocking receive left z communication buffer
      MPI_Irecv(recv_buffer_z0, z_buffer_length, MPI_CHREAL, source[4], 4, world, &recv_request[ireq]);

      //non-blocking send left z communication buffer
      MPI_Isend(send_buffer_z0, z_buffer_length, MPI_CHREAL, dest[4],   5, world, &send_request[0]);

      //keep track of how many sends and receives are expected
      ireq++;
    }

    // load right z communication buffer
    if(flags[5]==5)
    {
      offset = H.n_ghost*H.nx*H.ny;
      for(i=0;i<H.nx;i++)
      {
        for(j=0;j<H.ny;j++)
        {
          for(k=0;k<H.n_ghost;k++)
          {
            idx  = i + j*H.nx + (k+H.nz-2*H.n_ghost)*H.nx*H.ny;
            gidx = i + j*H.nx + k*H.nx*H.ny;
            *(send_buffer_z1 + gidx + 0*offset) = C.density[idx];
            *(send_buffer_z1 + gidx + 1*offset) = C.momentum_x[idx];
            *(send_buffer_z1 + gidx + 2*offset) = C.momentum_y[idx];
            *(send_buffer_z1 + gidx + 3*offset) = C.momentum_z[idx];
            *(send_buffer_z1 + gidx + 4*offset) = C.Energy[idx];
            #ifdef DE
            *(send_buffer_z1 + gidx + 5*offset) = C.GasEnergy[idx];
            #endif
          }
        }
      }
      
      //post non-blocking receive right x communication buffer
      MPI_Irecv(recv_buffer_z1, z_buffer_length, MPI_CHREAL, source[5], 5, world, &recv_request[ireq]);

      //non-blocking send right x communication buffer
      MPI_Isend(send_buffer_z1, z_buffer_length, MPI_CHREAL, dest[5],   4, world, &send_request[1]);

      //keep track of how many sends and receives are expected
      ireq++;
    }
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
  int iwait;
  int index = 0;
  int count = 0;
  int wait_max=0;
  MPI_Status status;
  int error = 0;

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
  int i, j, k;
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
  
          C.density[idx]    = *(recv_buffer_0 + gidx + 0*offset);
          C.momentum_x[idx] = *(recv_buffer_0 + gidx + 1*offset);
          C.momentum_y[idx] = *(recv_buffer_0 + gidx + 2*offset);
          C.momentum_z[idx] = *(recv_buffer_0 + gidx + 3*offset);
          C.Energy[idx]     = *(recv_buffer_0 + gidx + 4*offset);
          #ifdef DE
          C.GasEnergy[idx]  = *(recv_buffer_0 + gidx + 5*offset);
          #endif
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

          C.density[idx]    = *(recv_buffer_1 + gidx + 0*offset);
          C.momentum_x[idx] = *(recv_buffer_1 + gidx + 1*offset);
          C.momentum_y[idx] = *(recv_buffer_1 + gidx + 2*offset);
          C.momentum_z[idx] = *(recv_buffer_1 + gidx + 3*offset);
          C.Energy[idx]     = *(recv_buffer_1 + gidx + 4*offset);
          #ifdef DE
          C.GasEnergy[idx]  = *(recv_buffer_1 + gidx + 5*offset);
          #endif
        }
      }
    }
  }

  //done unloading
}


void Grid3D::Unload_MPI_Comm_Buffers_BLOCK(int index)
{
  int i, j, k;
  int idx;
  int gidx;
  int offset;

  //unload left x communication buffer
  if(index==0)
  {
    // 1D
    if (H.ny == 1 && H.nz == 1) {
      offset = H.n_ghost;
      for(i=0;i<H.n_ghost;i++) {
        idx  = i;
        gidx = i;
  
        C.density[idx]    = *(recv_buffer_x0 + gidx + 0*offset);
        C.momentum_x[idx] = *(recv_buffer_x0 + gidx + 1*offset);
        C.momentum_y[idx] = *(recv_buffer_x0 + gidx + 2*offset);
        C.momentum_z[idx] = *(recv_buffer_x0 + gidx + 3*offset);
        C.Energy[idx]     = *(recv_buffer_x0 + gidx + 4*offset);
        #ifdef DE
        C.GasEnergy[idx]  = *(recv_buffer_x0 + gidx + 5*offset);
        #endif
      }
    }
    // 2D
    if (H.ny > 1 && H.nz == 1) {
      offset = H.n_ghost*(H.ny-2*H.n_ghost);
      for(i=0;i<H.n_ghost;i++) {
        for (j=0;j<H.ny-2*H.n_ghost;j++) {
          idx  = i + (j+H.n_ghost)*H.nx;
          gidx = i + j*H.n_ghost;
  
          C.density[idx]    = *(recv_buffer_x0 + gidx + 0*offset);
          C.momentum_x[idx] = *(recv_buffer_x0 + gidx + 1*offset);
          C.momentum_y[idx] = *(recv_buffer_x0 + gidx + 2*offset);
          C.momentum_z[idx] = *(recv_buffer_x0 + gidx + 3*offset);
          C.Energy[idx]     = *(recv_buffer_x0 + gidx + 4*offset);
          #ifdef DE
          C.GasEnergy[idx]  = *(recv_buffer_x0 + gidx + 5*offset);
          #endif
        }
      }
    }
    // 3D
    if (H.nz > 1) {
      offset = H.n_ghost*(H.ny-2*H.n_ghost)*(H.nz-2*H.n_ghost);
      for(i=0;i<H.n_ghost;i++) {
        for(j=0;j<H.ny-2*H.n_ghost;j++) {
          for(k=0;k<H.nz-2*H.n_ghost;k++) {
            idx  = i + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            gidx = i + j*H.n_ghost + k*H.n_ghost*(H.ny-2*H.n_ghost);
 
            C.density[idx]    = *(recv_buffer_x0 + gidx + 0*offset);
            C.momentum_x[idx] = *(recv_buffer_x0 + gidx + 1*offset);
            C.momentum_y[idx] = *(recv_buffer_x0 + gidx + 2*offset);
            C.momentum_z[idx] = *(recv_buffer_x0 + gidx + 3*offset);
            C.Energy[idx]     = *(recv_buffer_x0 + gidx + 4*offset);
            #ifdef DE
            C.GasEnergy[idx]  = *(recv_buffer_x0 + gidx + 5*offset);
            #endif
          }
        }
      }
    }
  }

  //unload right x communication buffer
  if(index==1)
  {
    // 1D
    if (H.ny == 1 && H.nz == 1) {
      offset = H.n_ghost;
      for(i=0;i<H.n_ghost;i++) {
        idx  = i+H.nx-H.n_ghost;
        gidx = i;
  
        C.density[idx]    = *(recv_buffer_x1 + gidx + 0*offset);
        C.momentum_x[idx] = *(recv_buffer_x1 + gidx + 1*offset);
        C.momentum_y[idx] = *(recv_buffer_x1 + gidx + 2*offset);
        C.momentum_z[idx] = *(recv_buffer_x1 + gidx + 3*offset);
        C.Energy[idx]     = *(recv_buffer_x1 + gidx + 4*offset);
        #ifdef DE
        C.GasEnergy[idx]  = *(recv_buffer_x1 + gidx + 5*offset);
        #endif
      }
    }
    // 2D
    if (H.ny > 1 && H.nz == 1) {
      offset = H.n_ghost*(H.ny-2*H.n_ghost);
      for(i=0;i<H.n_ghost;i++) {
        for (j=0;j<H.ny-2*H.n_ghost;j++) {
          idx  = i+H.nx-H.n_ghost + (j+H.n_ghost)*H.nx;
          gidx = i + j*H.n_ghost;
  
          C.density[idx]    = *(recv_buffer_x1 + gidx + 0*offset);
          C.momentum_x[idx] = *(recv_buffer_x1 + gidx + 1*offset);
          C.momentum_y[idx] = *(recv_buffer_x1 + gidx + 2*offset);
          C.momentum_z[idx] = *(recv_buffer_x1 + gidx + 3*offset);
          C.Energy[idx]     = *(recv_buffer_x1 + gidx + 4*offset);
          #ifdef DE
          C.GasEnergy[idx]  = *(recv_buffer_x1 + gidx + 5*offset);
          #endif
        }
      }
    }
    // 3D
    if (H.nz > 1) {
      offset = H.n_ghost*(H.ny-2*H.n_ghost)*(H.nz-2*H.n_ghost);
      for(i=0;i<H.n_ghost;i++) {
        for(j=0;j<H.ny-2*H.n_ghost;j++) {
          for(k=0;k<H.nz-2*H.n_ghost;k++) {
            idx  = i+H.nx-H.n_ghost + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            gidx = i + j*H.n_ghost + k*H.n_ghost*(H.ny-2*H.n_ghost);
 
            C.density[idx]    = *(recv_buffer_x1 + gidx + 0*offset);
            C.momentum_x[idx] = *(recv_buffer_x1 + gidx + 1*offset);
            C.momentum_y[idx] = *(recv_buffer_x1 + gidx + 2*offset);
            C.momentum_z[idx] = *(recv_buffer_x1 + gidx + 3*offset);
            C.Energy[idx]     = *(recv_buffer_x1 + gidx + 4*offset);
            #ifdef DE
            C.GasEnergy[idx]  = *(recv_buffer_x1 + gidx + 5*offset);
            #endif
          }
        }
      }
    }
  }


  //unload left y communication buffer
  if(index==2)
  {
    // 2D
    if (H.nz == 1) {
      offset = H.n_ghost*H.nx;
      for(i=0;i<H.nx;i++) {
        for (j=0;j<H.n_ghost;j++) {
          idx  = i + j*H.nx;
          gidx = i + j*H.nx;
  
          C.density[idx]    = *(recv_buffer_y0 + gidx + 0*offset);
          C.momentum_x[idx] = *(recv_buffer_y0 + gidx + 1*offset);
          C.momentum_y[idx] = *(recv_buffer_y0 + gidx + 2*offset);
          C.momentum_z[idx] = *(recv_buffer_y0 + gidx + 3*offset);
          C.Energy[idx]     = *(recv_buffer_y0 + gidx + 4*offset);
          #ifdef DE
          C.GasEnergy[idx]  = *(recv_buffer_y0 + gidx + 5*offset);
          #endif
        }
      }
    }
    // 3D
    if (H.nz > 1) {
      offset = H.n_ghost*H.nx*(H.nz-2*H.n_ghost);
      for(i=0;i<H.nx;i++) {
        for(j=0;j<H.n_ghost;j++) {
          for(k=0;k<H.nz-2*H.n_ghost;k++) {
            idx  = i + j*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            gidx = i + j*H.nx + k*H.nx*H.n_ghost;
 
            C.density[idx]    = *(recv_buffer_y0 + gidx + 0*offset);
            C.momentum_x[idx] = *(recv_buffer_y0 + gidx + 1*offset);
            C.momentum_y[idx] = *(recv_buffer_y0 + gidx + 2*offset);
            C.momentum_z[idx] = *(recv_buffer_y0 + gidx + 3*offset);
            C.Energy[idx]     = *(recv_buffer_y0 + gidx + 4*offset);
            #ifdef DE
            C.GasEnergy[idx]  = *(recv_buffer_y0 + gidx + 5*offset);
            #endif
          }
        }
      }
    }
  }

  //unload right y communication buffer
  if(index==3)
  {
    // 2D
    if (H.nz == 1) {
      offset = H.n_ghost*H.nx;
      for(i=0;i<H.nx;i++) {
        for (j=0;j<H.n_ghost;j++) {
          idx  = i + (j+H.ny-H.n_ghost)*H.nx;
          gidx = i + j*H.nx;
  
          C.density[idx]    = *(recv_buffer_y1 + gidx + 0*offset);
          C.momentum_x[idx] = *(recv_buffer_y1 + gidx + 1*offset);
          C.momentum_y[idx] = *(recv_buffer_y1 + gidx + 2*offset);
          C.momentum_z[idx] = *(recv_buffer_y1 + gidx + 3*offset);
          C.Energy[idx]     = *(recv_buffer_y1 + gidx + 4*offset);
          #ifdef DE
          C.GasEnergy[idx]  = *(recv_buffer_y1 + gidx + 5*offset);
          #endif
        }
      }
    }
    // 3D
    if (H.nz > 1) {
      offset = H.n_ghost*H.nx*(H.nz-2*H.n_ghost);
      for(i=0;i<H.nx;i++) {
        for(j=0;j<H.n_ghost;j++) {
          for(k=0;k<H.nz-2*H.n_ghost;k++) {
            idx  = i + (j+H.ny-H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
            gidx = i + j*H.nx + k*H.nx*H.n_ghost;
 
            C.density[idx]    = *(recv_buffer_y1 + gidx + 0*offset);
            C.momentum_x[idx] = *(recv_buffer_y1 + gidx + 1*offset);
            C.momentum_y[idx] = *(recv_buffer_y1 + gidx + 2*offset);
            C.momentum_z[idx] = *(recv_buffer_y1 + gidx + 3*offset);
            C.Energy[idx]     = *(recv_buffer_y1 + gidx + 4*offset);
            #ifdef DE
            C.GasEnergy[idx]  = *(recv_buffer_y1 + gidx + 5*offset);
            #endif
          }
        }
      }
    }
  }

  //unload left z communication buffer
  if(index==4)
  {
    offset = H.n_ghost*H.nx*H.ny;
    for(i=0;i<H.nx;i++) {
      for(j=0;j<H.ny;j++) {
        for(k=0;k<H.n_ghost;k++) {
          idx  = i + j*H.nx + k*H.nx*H.ny;
          gidx = i + j*H.nx + k*H.nx*H.ny;
 
          C.density[idx]    = *(recv_buffer_z0 + gidx + 0*offset);
          C.momentum_x[idx] = *(recv_buffer_z0 + gidx + 1*offset);
          C.momentum_y[idx] = *(recv_buffer_z0 + gidx + 2*offset);
          C.momentum_z[idx] = *(recv_buffer_z0 + gidx + 3*offset);
          C.Energy[idx]     = *(recv_buffer_z0 + gidx + 4*offset);
          #ifdef DE
          C.GasEnergy[idx]  = *(recv_buffer_z0 + gidx + 5*offset);
          #endif
        }
      }
    }
  }

  //unload right z communication buffer
  if(index==5)
  {
    offset = H.n_ghost*H.nx*H.ny;
    for(i=0;i<H.nx;i++) {
      for(j=0;j<H.ny;j++) {
        for(k=0;k<H.n_ghost;k++) {
          idx  = i + j*H.nx + (k+H.nz-H.n_ghost)*H.nx*H.ny;
          gidx = i + j*H.nx + k*H.nx*H.ny;
 
          C.density[idx]    = *(recv_buffer_z1 + gidx + 0*offset);
          C.momentum_x[idx] = *(recv_buffer_z1 + gidx + 1*offset);
          C.momentum_y[idx] = *(recv_buffer_z1 + gidx + 2*offset);
          C.momentum_z[idx] = *(recv_buffer_z1 + gidx + 3*offset);
          C.Energy[idx]     = *(recv_buffer_z1 + gidx + 4*offset);
          #ifdef DE
          C.GasEnergy[idx]  = *(recv_buffer_z1 + gidx + 5*offset);
          #endif
        }
      }
    }
  }

}


#endif /*MPI_CHOLLA*/

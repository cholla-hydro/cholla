#include "../grid/grid3D.h"
#include "../mpi/mpi_routines.h"
#include "../io/io.h"
#include "../utils/error_handling.h"
#include <iostream>

#include "../utils/gpu.hpp"
#include "../global/global_cuda.h"//provides TPB
#include "../grid/cuda_boundaries.h"// provides PackBuffers3D and UnpackBuffers3D

#ifdef MPI_CHOLLA

void Grid3D::Set_Boundaries_MPI(struct parameters P)
{
  int flags[6] = {0,0,0,0,0,0};

  if(Check_Custom_Boundary(&flags[0],P))
  {
    //perform custom boundaries
    Custom_Boundary(P.custom_bcnd);
  }

  Set_Boundaries_MPI_BLOCK(flags,P);

  #ifdef GRAVITY
  Grav.Set_Boundary_Flags( flags );
  #endif

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
      Wait_and_Unload_MPI_Comm_Buffers(0, flags);
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
      Wait_and_Unload_MPI_Comm_Buffers(1, flags);
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
      Wait_and_Unload_MPI_Comm_Buffers(2, flags);
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


int Grid3D::Load_Hydro_DeviceBuffer_X0 ( Real *send_buffer_x0 ){

  // 1D
  if (H.ny == 1 && H.nz == 1) {
    int idxoffset = H.n_ghost;
    PackBuffers3D(send_buffer_x0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,1,1);
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    int idxoffset = H.n_ghost + H.n_ghost*H.nx;
    PackBuffers3D(send_buffer_x0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,H.ny-2*H.n_ghost,1);
  }
  // 3D
  if (H.ny > 1 && H.nz > 1) {
    int idxoffset = H.n_ghost + H.n_ghost*H.nx + H.n_ghost*H.nx*H.ny;
    PackBuffers3D(send_buffer_x0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,H.ny-2*H.n_ghost,H.nz-2*H.n_ghost);
  }

  return x_buffer_length;
}


// load right x communication buffer
int Grid3D::Load_Hydro_DeviceBuffer_X1 ( Real *send_buffer_x1 ){

  // 1D
  if (H.ny == 1 && H.nz == 1) {
    int idxoffset = H.nx-2*H.n_ghost;
    PackBuffers3D(send_buffer_x1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,1,1);
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    int idxoffset = H.nx-2*H.n_ghost + H.n_ghost*H.nx;
    PackBuffers3D(send_buffer_x1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,H.ny-2*H.n_ghost,1);
  }
  // 3D
  if (H.ny > 1 && H.nz > 1) {
    int idxoffset = H.nx-2*H.n_ghost + H.n_ghost*H.nx + H.n_ghost*H.nx*H.ny;
    PackBuffers3D(send_buffer_x1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,H.ny-2*H.n_ghost,H.nz-2*H.n_ghost);
  }

  return x_buffer_length;
}

// load left y communication buffer
int Grid3D::Load_Hydro_DeviceBuffer_Y0 ( Real *send_buffer_y0 ){

  // 2D
  if (H.nz == 1) {
    int idxoffset = H.n_ghost*H.nx;
    PackBuffers3D(send_buffer_y0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.n_ghost,1);
  }
  // 3D
  if (H.nz > 1) {
    int idxoffset = H.n_ghost*H.nx + H.n_ghost*H.nx*H.ny;
    PackBuffers3D(send_buffer_y0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.n_ghost,H.nz-2*H.n_ghost);
  }

  return y_buffer_length;
}

int Grid3D::Load_Hydro_DeviceBuffer_Y1 ( Real *send_buffer_y1 ){

  // 2D
  if (H.nz == 1) {
    int idxoffset = (H.ny-2*H.n_ghost)*H.nx;
    PackBuffers3D(send_buffer_y1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.n_ghost,1);
  }
  // 3D
  if (H.nz > 1) {
    int idxoffset = (H.ny-2*H.n_ghost)*H.nx + H.n_ghost*H.nx*H.ny;
    PackBuffers3D(send_buffer_y1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.n_ghost,H.nz-2*H.n_ghost);
  }

  return y_buffer_length;

}

// load left z communication buffer
int Grid3D::Load_Hydro_DeviceBuffer_Z0 ( Real *send_buffer_z0 ){

  // 3D
  int idxoffset = H.n_ghost*H.nx*H.ny;
  PackBuffers3D(send_buffer_z0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.ny,H.n_ghost);

  return z_buffer_length;
}

int Grid3D::Load_Hydro_DeviceBuffer_Z1 ( Real *send_buffer_z1 ){

  // 3D
  int idxoffset = (H.nz-2*H.n_ghost)*H.nx*H.ny;
  PackBuffers3D(send_buffer_z1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.ny,H.n_ghost);

  return z_buffer_length;
}

void Grid3D::Unload_Hydro_DeviceBuffer_X0 ( Real *recv_buffer_x0 ) {

  // 1D
  if (H.ny == 1 && H.nz == 1) {
    int idxoffset = 0;
    UnpackBuffers3D(recv_buffer_x0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,1,1);
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    int idxoffset = H.n_ghost*H.nx;
    UnpackBuffers3D(recv_buffer_x0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,H.ny-2*H.n_ghost,1);
  }
  // 3D
  if (H.nz > 1) {
    int idxoffset = H.n_ghost*(H.nx+H.nx*H.ny);
    UnpackBuffers3D(recv_buffer_x0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,H.ny-2*H.n_ghost,H.nz-2*H.n_ghost);
  }

}

void Grid3D::Unload_Hydro_DeviceBuffer_X1 ( Real *recv_buffer_x1 ) {

  // 1D
  if (H.ny == 1 && H.nz == 1) {
    int idxoffset = H.nx - H.n_ghost;
    UnpackBuffers3D(recv_buffer_x1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,1,1);
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    int idxoffset = H.nx - H.n_ghost + H.n_ghost*H.nx;
    UnpackBuffers3D(recv_buffer_x1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,H.ny-2*H.n_ghost,1);
  }
  // 3D
  if (H.nz > 1) {
    int idxoffset = H.nx - H.n_ghost + H.n_ghost*(H.nx+H.nx*H.ny);
    UnpackBuffers3D(recv_buffer_x1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.n_ghost,H.ny-2*H.n_ghost,H.nz-2*H.n_ghost);
  }

}


void Grid3D::Unload_Hydro_DeviceBuffer_Y0 ( Real *recv_buffer_y0 ) {

  // 2D
  if (H.nz == 1) {
    int idxoffset = 0;
    UnpackBuffers3D(recv_buffer_y0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.n_ghost,1);
  }
  // 3D
  if (H.nz > 1) {
    int idxoffset = H.n_ghost*H.nx*H.ny;
    UnpackBuffers3D(recv_buffer_y0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.n_ghost,H.nz-2*H.n_ghost);
  }

}


void Grid3D::Unload_Hydro_DeviceBuffer_Y1 ( Real *recv_buffer_y1 ) {

  // 2D
  if (H.nz == 1) {
    int idxoffset = (H.ny-H.n_ghost)*H.nx;
    UnpackBuffers3D(recv_buffer_y1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.n_ghost,1);
  }
  // 3D
  if (H.nz > 1) {
    int idxoffset = (H.ny-H.n_ghost)*H.nx + H.n_ghost*H.nx*H.ny;
    UnpackBuffers3D(recv_buffer_y1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.n_ghost,H.nz-2*H.n_ghost);
  }

}



void Grid3D::Unload_Hydro_DeviceBuffer_Z0 ( Real *recv_buffer_z0 ) {

  // 3D
  int idxoffset = 0;
  UnpackBuffers3D(recv_buffer_z0,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.ny,H.n_ghost);
}


void Grid3D::Unload_Hydro_DeviceBuffer_Z1 ( Real *recv_buffer_z1 ) {

  // 3D
  int idxoffset = (H.nz-H.n_ghost)*H.nx*H.ny;
  UnpackBuffers3D(recv_buffer_z1,C.device,H.nx,H.ny,H.n_fields,H.n_cells,idxoffset,H.nx,H.ny,H.n_ghost);
}

void Grid3D::Load_and_Send_MPI_Comm_Buffers(int dir, int *flags)
{

  #ifdef PARTICLES
  // Select which particles need to be transfred for this direction
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

  // Flag to omit the transfer of the main buffer when tranferring the particles buffer
  bool transfer_main_buffer = true;

  /* x boundaries */
  if(dir == 0)
  {
    if (flags[0]==5) {

      // load left x communication buffer
      if ( H.TRANSFER_HYDRO_BOUNDARIES )
        {
        buffer_length = Load_Hydro_DeviceBuffer_X0(d_send_buffer_x0);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_x0, d_send_buffer_x0, xbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
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
          #ifndef MPI_GPU 
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 0, 0, h_send_buffer_x0  );
          #else
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 0, 0, h_send_buffer_x0_particles  );
          cudaMemcpy(d_send_buffer_x0, h_send_buffer_x0_particles, buffer_length*sizeof(Real), cudaMemcpyHostToDevice);
          #endif
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
        #if defined(MPI_GPU)
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
        buffer_length = Load_Hydro_DeviceBuffer_X1(d_send_buffer_x1);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_x1, d_send_buffer_x1, xbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
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
          #ifndef MPI_GPU 
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 0, 1, h_send_buffer_x1  );
          #else
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 0, 1, h_send_buffer_x1_particles  );
          cudaMemcpy(d_send_buffer_x1, h_send_buffer_x1_particles, buffer_length*sizeof(Real), cudaMemcpyHostToDevice);
          #endif
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
	#if defined(MPI_GPU)
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
        buffer_length = Load_Hydro_DeviceBuffer_Y0(d_send_buffer_y0);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_y0, d_send_buffer_y0, ybsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
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
          #ifndef MPI_GPU 
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 1, 0, h_send_buffer_y0  );
          #else
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 1, 0, h_send_buffer_y0_particles  );
          cudaMemcpy(d_send_buffer_y0, h_send_buffer_y0_particles, buffer_length*sizeof(Real), cudaMemcpyHostToDevice);
          #endif
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
	#if defined(MPI_GPU)
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
        buffer_length = Load_Hydro_DeviceBuffer_Y1(d_send_buffer_y1);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_y1, d_send_buffer_y1, ybsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
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
          #ifndef MPI_GPU 
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 1, 1, h_send_buffer_y1  );
          #else
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 1, 1, h_send_buffer_y1_particles  );
          cudaMemcpy(d_send_buffer_y1, h_send_buffer_y1_particles, buffer_length*sizeof(Real), cudaMemcpyHostToDevice);
          #endif
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
        #if defined(MPI_GPU)
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
        buffer_length = Load_Hydro_DeviceBuffer_Z0(d_send_buffer_z0);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_z0, d_send_buffer_z0, zbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
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
          #ifndef MPI_GPU 
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 2, 0, h_send_buffer_z0  );
          #else
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 2, 0, h_send_buffer_z0_particles  );
          cudaMemcpy(d_send_buffer_z0, h_send_buffer_z0_particles, buffer_length*sizeof(Real), cudaMemcpyHostToDevice);
          #endif
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
        #if defined(MPI_GPU)
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
        buffer_length = Load_Hydro_DeviceBuffer_Z1(d_send_buffer_z1);
          #ifndef MPI_GPU
          cudaMemcpy(h_send_buffer_z1, d_send_buffer_z1, zbsize*sizeof(Real),
                     cudaMemcpyDeviceToHost);
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
          #ifndef MPI_GPU 
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 2, 1, h_send_buffer_z1  );
          #else
          buffer_length = Load_Particles_Density_Boundary_to_Buffer( 2, 1, h_send_buffer_z1_particles  );
          cudaMemcpy(d_send_buffer_z1, h_send_buffer_z1_particles, buffer_length*sizeof(Real), cudaMemcpyHostToDevice);
          #endif
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
        #if defined(MPI_GPU)
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

void Grid3D::Wait_and_Unload_MPI_Comm_Buffers(int dir, int *flags)
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
    
    #ifdef MPI_GPU 
    if ( index == 0 ) Copy_Particles_Density_Buffer_Device_to_Host( 0, 0, d_recv_buffer_x0, h_recv_buffer_x0_particles );
    if ( index == 1 ) Copy_Particles_Density_Buffer_Device_to_Host( 0, 1, d_recv_buffer_x1, h_recv_buffer_x1_particles );
    if ( index == 2 ) Copy_Particles_Density_Buffer_Device_to_Host( 1, 0, d_recv_buffer_y0, h_recv_buffer_y0_particles );
    if ( index == 3 ) Copy_Particles_Density_Buffer_Device_to_Host( 1, 1, d_recv_buffer_y1, h_recv_buffer_y1_particles );
    if ( index == 4 ) Copy_Particles_Density_Buffer_Device_to_Host( 2, 0, d_recv_buffer_z0, h_recv_buffer_z0_particles );
    if ( index == 5 ) Copy_Particles_Density_Buffer_Device_to_Host( 2, 1, d_recv_buffer_z1, h_recv_buffer_z1_particles );
    l_recv_buffer_x0 = h_recv_buffer_x0_particles;
    l_recv_buffer_x1 = h_recv_buffer_x1_particles;
    l_recv_buffer_y0 = h_recv_buffer_y0_particles;
    l_recv_buffer_y1 = h_recv_buffer_y1_particles;
    l_recv_buffer_z0 = h_recv_buffer_z0_particles;
    l_recv_buffer_z1 = h_recv_buffer_z1_particles;
    #else
    l_recv_buffer_x0 = h_recv_buffer_x0;
    l_recv_buffer_x1 = h_recv_buffer_x1;
    l_recv_buffer_y0 = h_recv_buffer_y0;
    l_recv_buffer_y1 = h_recv_buffer_y1;
    l_recv_buffer_z0 = h_recv_buffer_z0;
    l_recv_buffer_z1 = h_recv_buffer_z1;
    #endif //MPI_GPU
    
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

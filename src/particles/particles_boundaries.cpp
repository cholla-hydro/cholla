#ifdef PARTICLES

#include <unistd.h>
#include <algorithm>
#include <iostream>
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../particles/particles_3D.h"

#ifdef MPI_CHOLLA
#include "../mpi/mpi_routines.h"
#ifdef PARTICLES_GPU
#include "../particles/particles_boundaries_gpu.h"
#endif//PARTICLES_GPU
#endif//MPI_CHOLLA


//Transfer the particles that moved outside the local domain
void Grid3D::Transfer_Particles_Boundaries( struct parameters P ){


  //Transfer Particles Boundaries
  Particles.TRANSFER_PARTICLES_BOUNDARIES = true;
  #ifdef CPU_TIME
  Timer.Start_Timer();
  #endif
  Set_Boundary_Conditions(P);
  #ifdef CPU_TIME
  Timer.End_and_Record_Time( 8 );
  #endif
  Particles.TRANSFER_PARTICLES_BOUNDARIES = false;

}

#ifdef MPI_CHOLLA
//Remove the particles that were transfered outside the local domain
void Grid3D::Finish_Particles_Transfer( void ){

  #ifdef PARTICLES_CPU
  Particles.Remove_Transfered_Particles();
  #endif

}


//Wait for the MPI request and unload the transfered particles
void Grid3D::Wait_and_Unload_MPI_Comm_Particles_Buffers_BLOCK(int dir, int *flags)
{

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
    MPI_Waitany(wait_max,recv_request_particles_transfer,&index,&status);
    //depending on which face arrived, load the buffer into the ghost grid
    Unload_Particles_From_Buffers_BLOCK(status.MPI_TAG, flags );
  }
}

//Unload the particles after MPI tranfer for a single index ( axis and side )
void Grid3D::Unload_Particles_From_Buffers_BLOCK(int index, int *flags ){

  // Make sure not to unload when not transfering particles
  if ( Particles.TRANSFER_DENSITY_BOUNDARIES ) return;
  if ( H.TRANSFER_HYDRO_BOUNDARIES ) return;
  if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ) return;

  if( index == 0) Unload_Particles_from_Buffer_X0( flags );
  if( index == 1) Unload_Particles_from_Buffer_X1( flags );
  if( index == 2) Unload_Particles_from_Buffer_Y0( flags );
  if( index == 3) Unload_Particles_from_Buffer_Y1( flags );
  if( index == 4) Unload_Particles_from_Buffer_Z0( flags );
  if( index == 5) Unload_Particles_from_Buffer_Z1( flags );
}


//Wait for the Number of particles that will be transferd, and request the MPI_Recv to receive the MPI buffer
void Grid3D::Wait_NTransfer_and_Request_Recv_Particles_Transfer_BLOCK(int dir, int *flags)
{
  #ifdef PARTICLES
  if ( !Particles.TRANSFER_PARTICLES_BOUNDARIES ) return;
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

  int ireq_particles_transfer = 0;
  //wait for any receives to complete
  for(iwait=0;iwait<wait_max;iwait++)
  {
    //wait for recv completion
    MPI_Waitany(wait_max,recv_request_n_particles,&index,&status);
    //depending on which face arrived, load the buffer into the ghost grid
    Load_NTtransfer_and_Request_Receive_Particles_Transfer(status.MPI_TAG, &ireq_particles_transfer);
  }
}

//Load the Number of particles that will be recieved (Particles.n_recv) and make the MPI_Irecv request for that buffer size
void Grid3D::Load_NTtransfer_and_Request_Receive_Particles_Transfer(int index, int *ireq_particles_transfer){

  int buffer_length;

  #ifdef PARTICLES_GPU
    #ifdef MPI_GPU
    Real *recv_buffer_x0_particles = d_recv_buffer_x0_particles;
    Real *recv_buffer_x1_particles = d_recv_buffer_x1_particles;
    Real *recv_buffer_y0_particles = d_recv_buffer_y0_particles;
    Real *recv_buffer_y1_particles = d_recv_buffer_y1_particles;
    Real *recv_buffer_z0_particles = d_recv_buffer_z0_particles;
    Real *recv_buffer_z1_particles = d_recv_buffer_z1_particles;
    #else
    Real *recv_buffer_x0_particles = h_recv_buffer_x0_particles;
    Real *recv_buffer_x1_particles = h_recv_buffer_x1_particles;
    Real *recv_buffer_y0_particles = h_recv_buffer_y0_particles;
    Real *recv_buffer_y1_particles = h_recv_buffer_y1_particles;
    Real *recv_buffer_z0_particles = h_recv_buffer_z0_particles;
    Real *recv_buffer_z1_particles = h_recv_buffer_z1_particles;
    #endif
  #endif

  #ifdef PARTICLES_CPU
  Real *recv_buffer_x0_particles = h_recv_buffer_x0_particles;
  Real *recv_buffer_x1_particles = h_recv_buffer_x1_particles;
  Real *recv_buffer_y0_particles = h_recv_buffer_y0_particles;
  Real *recv_buffer_y1_particles = h_recv_buffer_y1_particles;
  Real *recv_buffer_z0_particles = h_recv_buffer_z0_particles;
  Real *recv_buffer_z1_particles = h_recv_buffer_z1_particles;
  #endif

  if ( index == 0){
    buffer_length = Particles.n_recv_x0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_x0_particles , &buffer_length_particles_x0_recv, buffer_length );
    // if ( Particles.n_recv_x0 > 0 ) std::cout << " Recv X0: " << Particles.n_recv_x0 << std::endl;
    MPI_Irecv(recv_buffer_x0_particles, buffer_length, MPI_CHREAL, source[0], 0, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
  }
  if ( index == 1){
    buffer_length = Particles.n_recv_x1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_x1_particles , &buffer_length_particles_x1_recv, buffer_length );
    // if ( Particles.n_recv_x1 > 0 ) if ( Particles.n_recv_x1 > 0 ) std::cout << " Recv X1:  " << Particles.n_recv_x1 <<  "  " << procID <<  "  from "  <<  source[1] <<  std::endl;
    MPI_Irecv(recv_buffer_x1_particles, buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
  }
  if ( index == 2){
    buffer_length = Particles.n_recv_y0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_y0_particles , &buffer_length_particles_y0_recv, buffer_length );
    // if ( Particles.n_recv_y0 > 0 ) std::cout << " Recv Y0: " << Particles.n_recv_y0 << std::endl;
    MPI_Irecv(recv_buffer_y0_particles, buffer_length, MPI_CHREAL, source[2], 2, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
  }
  if ( index == 3){
    buffer_length = Particles.n_recv_y1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_y1_particles , &buffer_length_particles_y1_recv, buffer_length );
    // if ( Particles.n_recv_y1 > 0 ) std::cout << " Recv Y1: " << Particles.n_recv_y1 << std::endl;
    MPI_Irecv(recv_buffer_y1_particles, buffer_length, MPI_CHREAL, source[3], 3, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
  }
  if ( index == 4){
    buffer_length = Particles.n_recv_z0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_z0_particles , &buffer_length_particles_z0_recv, buffer_length );
    // if ( Particles.n_recv_z0 > 0 ) std::cout << " Recv Z0: " << Particles.n_recv_z0 << std::endl;
    MPI_Irecv(recv_buffer_z0_particles, buffer_length, MPI_CHREAL, source[4], 4, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
  }
  if ( index == 5){
    buffer_length = Particles.n_recv_z1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_z1_particles , &buffer_length_particles_z1_recv, buffer_length );
    // if ( Particles.n_recv_z1 >0 ) std::cout << " Recv Z1: " << Particles.n_recv_z1 << std::endl;
    MPI_Irecv(recv_buffer_z1_particles, buffer_length, MPI_CHREAL, source[5], 5, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
  }

  *ireq_particles_transfer += 1;
}


//Make Send and Recieve request for the number of particles that will be transfered, and then load and send the transfer particles
void Grid3D::Load_and_Send_Particles_X0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  Real *send_buffer_x0_particles;

  #ifdef PARTICLES_GPU
  send_buffer_x0_particles = d_send_buffer_x0_particles;
  Particles.Load_Particles_to_Buffer_GPU(0, 0, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_GPU

  MPI_Irecv(&Particles.n_recv_x0, 1, MPI_PART_INT, source[0], 0, world, &recv_request_n_particles[ireq_n_particles]);
  MPI_Isend(&Particles.n_send_x0, 1, MPI_PART_INT, dest[0],   1, world, &send_request_n_particles[0]);
  MPI_Request_free(send_request_n_particles);
  // if ( Particles.n_send_x0 > 0 )   if ( Particles.n_send_x0 > 0 ) std::cout << " Sent X0:  " << Particles.n_send_x0 <<  "  " << procID <<  "  to  "  <<  dest[0] <<  std::endl;
  buffer_length = Particles.n_send_x0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_x0_particles , &buffer_length_particles_x0_send, buffer_length );
  #ifdef PARTICLES_CPU
  send_buffer_x0_particles = h_send_buffer_x0_particles;
  Particles.Load_Particles_to_Buffer_CPU( 0, 0, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_CPU

  #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
  cudaMemcpy(h_send_buffer_x0_particles, d_send_buffer_x0_particles,
             buffer_length*sizeof(Real), cudaMemcpyDeviceToHost);
  send_buffer_x0_particles = h_send_buffer_x0_particles;
  #endif

  MPI_Isend(send_buffer_x0_particles, buffer_length, MPI_CHREAL, dest[0],   1, world, &send_request_particles_transfer[ireq_particles_transfer]);
  MPI_Request_free(send_request_particles_transfer+ireq_particles_transfer);
}

void Grid3D::Load_and_Send_Particles_X1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  Real *send_buffer_x1_particles;

  #ifdef PARTICLES_GPU
  send_buffer_x1_particles = d_send_buffer_x1_particles;
  Particles.Load_Particles_to_Buffer_GPU(0, 1, send_buffer_x1_particles,  buffer_length_particles_x1_send );
  #endif //PARTICLES_GPU

  MPI_Irecv(&Particles.n_recv_x1, 1, MPI_PART_INT, source[1], 1, world, &recv_request_n_particles[ireq_n_particles]);
  MPI_Isend(&Particles.n_send_x1, 1, MPI_PART_INT, dest[1],   0, world, &send_request_n_particles[1]);
  MPI_Request_free(send_request_n_particles+1);
  // if ( Particles.n_send_x1 > 0 )  std::cout << " Sent X1: " << Particles.n_send_x1 << std::endl;
  buffer_length = Particles.n_send_x1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_x1_particles , &buffer_length_particles_x1_send, buffer_length );
  #ifdef PARTICLES_CPU
  send_buffer_x1_particles = h_send_buffer_x1_particles;
  Particles.Load_Particles_to_Buffer_CPU( 0, 1, send_buffer_x1_particles,  buffer_length_particles_x1_send  );
  #endif //PARTICLES_CPU

  #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
  cudaMemcpy(h_send_buffer_x1_particles, d_send_buffer_x1_particles,
             buffer_length*sizeof(Real), cudaMemcpyDeviceToHost);
  send_buffer_x1_particles = h_send_buffer_x1_particles;
  #endif

  MPI_Isend(send_buffer_x1_particles, buffer_length, MPI_CHREAL, dest[1],   0, world, &send_request_particles_transfer[ireq_particles_transfer]);\
  MPI_Request_free(send_request_particles_transfer+ireq_particles_transfer);
}

void Grid3D::Load_and_Send_Particles_Y0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  Real *send_buffer_y0_particles;

  #ifdef PARTICLES_GPU
  send_buffer_y0_particles = d_send_buffer_y0_particles;
  Particles.Load_Particles_to_Buffer_GPU(1, 0, send_buffer_y0_particles,  buffer_length_particles_y0_send );
  #endif //PARTICLES_GPU

  MPI_Isend(&Particles.n_send_y0, 1, MPI_PART_INT, dest[2],   3, world, &send_request_n_particles[0]);
  MPI_Request_free(send_request_n_particles);
  MPI_Irecv(&Particles.n_recv_y0, 1, MPI_PART_INT, source[2], 2, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_y0 > 0 )   std::cout << " Sent Y0: " << Particles.n_send_y0 << std::endl;
  buffer_length = Particles.n_send_y0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_y0_particles , &buffer_length_particles_y0_send, buffer_length );
  #ifdef PARTICLES_CPU
  send_buffer_y0_particles = h_send_buffer_y0_particles;
  Particles.Load_Particles_to_Buffer_CPU( 1, 0, send_buffer_y0_particles,  buffer_length_particles_y0_send  );
  #endif //PARTICLES_CPU

  #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
  cudaMemcpy(h_send_buffer_y0_particles, d_send_buffer_y0_particles,
             buffer_length*sizeof(Real), cudaMemcpyDeviceToHost);
  send_buffer_y0_particles = h_send_buffer_y0_particles;
  #endif

  MPI_Isend(send_buffer_y0_particles, buffer_length, MPI_CHREAL, dest[2],   3, world, &send_request_particles_transfer[ireq_particles_transfer]);
  MPI_Request_free(send_request_particles_transfer+ireq_particles_transfer);
}

void Grid3D::Load_and_Send_Particles_Y1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  Real *send_buffer_y1_particles;

  #ifdef PARTICLES_GPU
  send_buffer_y1_particles = d_send_buffer_y1_particles;
  Particles.Load_Particles_to_Buffer_GPU(1, 1, send_buffer_y1_particles,  buffer_length_particles_y1_send );
  #endif //PARTICLES_GPU

  MPI_Isend(&Particles.n_send_y1, 1, MPI_PART_INT, dest[3],   2, world, &send_request_n_particles[1]);
  MPI_Request_free(send_request_n_particles+1);
  MPI_Irecv(&Particles.n_recv_y1, 1, MPI_PART_INT, source[3], 3, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_y1 > 0 )  std::cout << " Sent Y1: " << Particles.n_send_y1 << std::endl;
  buffer_length = Particles.n_send_y1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_y1_particles , &buffer_length_particles_y1_send, buffer_length );
  #ifdef PARTICLES_CPU
  send_buffer_y1_particles = h_send_buffer_y1_particles;
  Particles.Load_Particles_to_Buffer_CPU( 1, 1, send_buffer_y1_particles,  buffer_length_particles_y1_send  );
  #endif //PARTICLES_CPU

  #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
  cudaMemcpy(h_send_buffer_y1_particles, d_send_buffer_y1_particles,
             buffer_length*sizeof(Real), cudaMemcpyDeviceToHost);
  send_buffer_y1_particles = h_send_buffer_y1_particles;
  #endif

  MPI_Isend(send_buffer_y1_particles, buffer_length, MPI_CHREAL, dest[3],   2, world, &send_request_particles_transfer[ireq_particles_transfer]);
  MPI_Request_free(send_request_particles_transfer+ireq_particles_transfer);
}

void Grid3D::Load_and_Send_Particles_Z0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  Real *send_buffer_z0_particles;

  #ifdef PARTICLES_GPU
  send_buffer_z0_particles = d_send_buffer_z0_particles;
  Particles.Load_Particles_to_Buffer_GPU(2, 0, send_buffer_z0_particles,  buffer_length_particles_z0_send );
  #endif //PARTICLES_GPU

  MPI_Isend(&Particles.n_send_z0, 1, MPI_PART_INT, dest[4],   5, world, &send_request_n_particles[0]);
  MPI_Request_free(send_request_n_particles);
  MPI_Irecv(&Particles.n_recv_z0, 1, MPI_PART_INT, source[4], 4, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_z0 > 0 )   std::cout << " Sent Z0: " << Particles.n_send_z0 << std::endl;
  buffer_length = Particles.n_send_z0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_z0_particles , &buffer_length_particles_z0_send, buffer_length );
  #ifdef PARTICLES_CPU
  send_buffer_z0_particles = h_send_buffer_z0_particles;
  Particles.Load_Particles_to_Buffer_CPU( 2, 0, send_buffer_z0_particles,  buffer_length_particles_z0_send  );
  #endif //PARTICLES_CPU

  #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
  cudaMemcpy(h_send_buffer_z0_particles, d_send_buffer_z0_particles,
             buffer_length*sizeof(Real), cudaMemcpyDeviceToHost);
  send_buffer_z0_particles = h_send_buffer_z0_particles;
  #endif

  MPI_Isend(send_buffer_z0_particles, buffer_length, MPI_CHREAL, dest[4],   5, world, &send_request_particles_transfer[ireq_particles_transfer]);
  MPI_Request_free(send_request_particles_transfer+ireq_particles_transfer);
}

void Grid3D::Load_and_Send_Particles_Z1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  Real *send_buffer_z1_particles;

  #ifdef PARTICLES_GPU
  send_buffer_z1_particles = d_send_buffer_z1_particles;
  Particles.Load_Particles_to_Buffer_GPU(2, 1, send_buffer_z1_particles,  buffer_length_particles_z1_send );
  #endif //PARTICLES_GPU

  MPI_Isend(&Particles.n_send_z1, 1, MPI_CHREAL, dest[5],   4, world, &send_request_n_particles[1]);
  MPI_Request_free(send_request_n_particles+1);
  MPI_Irecv(&Particles.n_recv_z1, 1, MPI_CHREAL, source[5], 5, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_z1 > 0 )   std::cout << " Sent Z1: " << Particles.n_send_z1 << std::endl;
  buffer_length = Particles.n_send_z1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_z1_particles , &buffer_length_particles_z1_send, buffer_length );
  #ifdef PARTICLES_CPU
  send_buffer_z1_particles = h_send_buffer_z1_particles;
  Particles.Load_Particles_to_Buffer_CPU( 2, 1, send_buffer_z1_particles,  buffer_length_particles_z1_send  );
  #endif //PARTICLES_CPU

  #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
  cudaMemcpy(h_send_buffer_z1_particles, d_send_buffer_z1_particles,
             buffer_length*sizeof(Real), cudaMemcpyDeviceToHost);
  send_buffer_z1_particles = h_send_buffer_z1_particles;
  #endif

  MPI_Isend(send_buffer_z1_particles, buffer_length, MPI_CHREAL, dest[5],   4, world, &send_request_particles_transfer[ireq_particles_transfer]);
  MPI_Request_free(send_request_particles_transfer+ireq_particles_transfer);
}

//Unload the Transfered particles from the MPI_buffer, after buffer was received
void Grid3D::Unload_Particles_from_Buffer_X0( int *flags ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 0, 0, h_recv_buffer_x0_particles, Particles.n_recv_x0,
      h_send_buffer_y0_particles, h_send_buffer_y1_particles, h_send_buffer_z0_particles,
      h_send_buffer_z1_particles, buffer_length_particles_y0_send, buffer_length_particles_y1_send,
      buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
  #ifdef PARTICLES_GPU
  #ifndef MPI_GPU
  cudaMemcpy(d_recv_buffer_x0_particles, h_recv_buffer_x0_particles,
             buffer_length_particles_x0_recv*sizeof(Real),
             cudaMemcpyHostToDevice);
  #endif
  Particles.Unload_Particles_from_Buffer_GPU( 0, 0, d_recv_buffer_x0_particles, Particles.n_recv_x0 );
  #endif//PARTICLES_GPU
}

void Grid3D::Unload_Particles_from_Buffer_X1( int *flags  ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 0, 1, h_recv_buffer_x1_particles, Particles.n_recv_x1,
      h_send_buffer_y0_particles, h_send_buffer_y1_particles, h_send_buffer_z0_particles,
      h_send_buffer_z1_particles, buffer_length_particles_y0_send, buffer_length_particles_y1_send,
      buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
  #ifdef PARTICLES_GPU
  #ifndef MPI_GPU
  cudaMemcpy(d_recv_buffer_x1_particles, h_recv_buffer_x1_particles,
             buffer_length_particles_x1_recv*sizeof(Real),
             cudaMemcpyHostToDevice);
  #endif
  Particles.Unload_Particles_from_Buffer_GPU( 0, 1, d_recv_buffer_x1_particles, Particles.n_recv_x1 );
  #endif//PARTICLES_GPU
}

void Grid3D::Unload_Particles_from_Buffer_Y0( int *flags ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 1, 0, h_recv_buffer_y0_particles, Particles.n_recv_y0,
      h_send_buffer_y0_particles, h_send_buffer_y1_particles, h_send_buffer_z0_particles,
      h_send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send,
      buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
  #ifdef PARTICLES_GPU
  #ifndef MPI_GPU
  cudaMemcpy(d_recv_buffer_y0_particles, h_recv_buffer_y0_particles,
             buffer_length_particles_y0_recv*sizeof(Real),
             cudaMemcpyHostToDevice);
  #endif
  Particles.Unload_Particles_from_Buffer_GPU( 1, 0, d_recv_buffer_y0_particles, Particles.n_recv_y0 );
  #endif//PARTICLES_GPU
}

void Grid3D::Unload_Particles_from_Buffer_Y1( int *flags  ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 1, 1, h_recv_buffer_y1_particles, Particles.n_recv_y1,
      h_send_buffer_y0_particles, h_send_buffer_y1_particles, h_send_buffer_z0_particles,
      h_send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send,
      buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
  #ifdef PARTICLES_GPU
  #ifndef MPI_GPU
  cudaMemcpy(d_recv_buffer_y1_particles, h_recv_buffer_y1_particles,
             buffer_length_particles_y1_recv*sizeof(Real),
             cudaMemcpyHostToDevice);
  #endif
  Particles.Unload_Particles_from_Buffer_GPU( 1, 1, d_recv_buffer_y1_particles, Particles.n_recv_y1 );
  #endif//PARTICLES_GPU
}

void Grid3D::Unload_Particles_from_Buffer_Z0( int *flags ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 2, 0, h_recv_buffer_z0_particles, Particles.n_recv_z0,
      h_send_buffer_y0_particles, h_send_buffer_y1_particles, h_send_buffer_z0_particles,
      h_send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send,
      buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
  #ifdef PARTICLES_GPU
  #ifndef MPI_GPU
  cudaMemcpy(d_recv_buffer_z0_particles, h_recv_buffer_z0_particles,
             buffer_length_particles_z0_recv*sizeof(Real),
             cudaMemcpyHostToDevice);
  #endif
  Particles.Unload_Particles_from_Buffer_GPU( 2, 0, d_recv_buffer_z0_particles, Particles.n_recv_z0 );
  #endif//PARTICLES_GPU
}

void Grid3D::Unload_Particles_from_Buffer_Z1( int *flags ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 2, 1, h_recv_buffer_z1_particles, Particles.n_recv_z1,
      h_send_buffer_y0_particles, h_send_buffer_y1_particles, h_send_buffer_z0_particles,
      h_send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send,
      buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
  #ifdef PARTICLES_GPU
  #ifndef MPI_GPU
  cudaMemcpy(d_recv_buffer_z1_particles, h_recv_buffer_z1_particles,
             buffer_length_particles_z1_recv*sizeof(Real),
             cudaMemcpyHostToDevice);
  #endif
  Particles.Unload_Particles_from_Buffer_GPU( 2, 1, d_recv_buffer_z1_particles, Particles.n_recv_z1 );
  #endif//PARTICLES_GPU
}


//Find the particles that moved outside the local domain in order to transfer them.
void Particles_3D::Select_Particles_to_Transfer_All( int *flags ){

  #ifdef PARTICLES_CPU
  Select_Particles_to_Transfer_All_CPU( flags );
  #endif//PARTICLES_CPU

  // When using PARTICLES_GPU the particles that need to be Transfered
  // are seleted on the Load_Buffer_GPU functions

}


void Particles_3D::Clear_Particles_For_Transfer( void ){

  //Set the number of transfered particles to 0.
  n_transfer_x0 = 0;
  n_transfer_x1 = 0;
  n_transfer_y0 = 0;
  n_transfer_y1 = 0;
  n_transfer_z0 = 0;
  n_transfer_z1 = 0;

  //Set the number of send particles to 0.
  n_send_x0 = 0;
  n_send_x1 = 0;
  n_send_y0 = 0;
  n_send_y1 = 0;
  n_send_z0 = 0;
  n_send_z1 = 0;

  //Set the number of recieved particles to 0.
  n_recv_x0 = 0;
  n_recv_x1 = 0;
  n_recv_y0 = 0;
  n_recv_y1 = 0;
  n_recv_z0 = 0;
  n_recv_z1 = 0;

  //Set the number of particles in transfer buffers to 0.
  n_in_buffer_x0 = 0;
  n_in_buffer_x1 = 0;
  n_in_buffer_y0 = 0;
  n_in_buffer_y1 = 0;
  n_in_buffer_z0 = 0;
  n_in_buffer_z1 = 0;


  #ifdef PARTICLES_CPU
  //Clear the particles indices that were transfered during the previour timestep
  Clear_Vectors_For_Transfers();
  #endif //PARTICLES_CPU

}

#ifdef PARTICLES_GPU

int Particles_3D::Select_Particles_to_Transfer_GPU( int direction, int side ){

  int n_transfer;
  Real *pos;
  Real domainMin, domainMax;

  if ( direction == 0 ){
    pos = pos_x_dev;
    domainMax = G.xMax;
    domainMin = G.xMin;
  }
  if ( direction == 1 ){
    pos = pos_y_dev;
    domainMax = G.yMax;
    domainMin = G.yMin;
  }
  if ( direction == 2 ){
    pos = pos_z_dev;
    domainMax = G.zMax;
    domainMin = G.zMin;
  }

  //Set the number of particles that will be sent and load the particles data into the transfer buffers
  n_transfer = Select_Particles_to_Transfer_GPU_function(  n_local, side, domainMin, domainMax, pos, G.n_transfer_d, G.n_transfer_h, G.transfer_particles_flags_d, G.transfer_particles_indices_d, G.replace_particles_indices_d, G.transfer_particles_prefix_sum_d, G.transfer_particles_prefix_sum_blocks_d  );
  CHECK(cudaDeviceSynchronize());

  return n_transfer;
}

void Particles_3D::Copy_Transfer_Particles_to_Buffer_GPU(int n_transfer, int direction, int side, Real *send_buffer_h, int buffer_length  ){

  part_int_t *n_send;
  int n_fields_to_transfer, buffer_size;
  Real *pos, *send_buffer_d;
  Real domainMin, domainMax;
  int bt_pos_x, bt_pos_y, bt_pos_z, bt_non_pos;

  bt_pos_x = -1;
  bt_pos_y = -1;
  bt_pos_z = -1;
  bt_non_pos = -1;

  if ( direction == 0 ){
    pos = pos_x_dev;
    domainMin = G.domainMin_x;
    domainMax = G.domainMax_x;
    if ( side == 0 ){
      n_send = &n_send_x0;
      buffer_size = G.send_buffer_size_x0;
      send_buffer_d = G.send_buffer_x0_d;
      bt_pos_x = G.boundary_type_x0;
    }
    if ( side == 1 ){
      n_send = &n_send_x1;
      buffer_size = G.send_buffer_size_x1;
      send_buffer_d = G.send_buffer_x1_d;
      bt_pos_x = G.boundary_type_x1;
    }
  }
  if ( direction == 1 ){
    pos = pos_y_dev;
    domainMin = G.domainMin_y;
    domainMax = G.domainMax_y;
    if ( side == 0 ){
      n_send = &n_send_y0;
      buffer_size = G.send_buffer_size_y0;
      send_buffer_d = G.send_buffer_y0_d;
      bt_pos_y = G.boundary_type_y0;
    }
    if ( side == 1 ){
      n_send = &n_send_y1;
      buffer_size = G.send_buffer_size_y1;
      send_buffer_d = G.send_buffer_y1_d;
      bt_pos_y = G.boundary_type_y1;
    }
  }
  if ( direction == 2 ){
    pos = pos_z_dev;
    domainMin = G.domainMin_z;
    domainMax = G.domainMax_z;
    if ( side == 0 ){
      n_send = &n_send_z0;
      buffer_size = G.send_buffer_size_z0;
      send_buffer_d = G.send_buffer_z0_d;
      bt_pos_z = G.boundary_type_z0;
    }
    if ( side == 1 ){
      n_send = &n_send_z1;
      buffer_size = G.send_buffer_size_z1;
      send_buffer_d = G.send_buffer_z1_d;
      bt_pos_z = G.boundary_type_z1;
    }
  }




  if ( (*n_send + n_transfer)*N_DATA_PER_PARTICLE_TRANSFER > buffer_size  ){
      printf("ERROR:Transfer Buffer is not large enough\n" );
      exit(-1);
  }

  // Load the particles that will be transfered into the buffers
  n_fields_to_transfer = N_DATA_PER_PARTICLE_TRANSFER;
  Load_Particles_to_Transfer_GPU_function( n_transfer, 0, n_fields_to_transfer, pos_x_dev, G.transfer_particles_indices_d, send_buffer_d, domainMin, domainMax, bt_pos_x );
  Load_Particles_to_Transfer_GPU_function( n_transfer, 1, n_fields_to_transfer, pos_y_dev, G.transfer_particles_indices_d, send_buffer_d, domainMin, domainMax, bt_pos_y );
  Load_Particles_to_Transfer_GPU_function( n_transfer, 2, n_fields_to_transfer, pos_z_dev, G.transfer_particles_indices_d, send_buffer_d, domainMin, domainMax, bt_pos_z );
  Load_Particles_to_Transfer_GPU_function( n_transfer, 3, n_fields_to_transfer, vel_x_dev, G.transfer_particles_indices_d, send_buffer_d, domainMin, domainMax, bt_non_pos );
  Load_Particles_to_Transfer_GPU_function( n_transfer, 4, n_fields_to_transfer, vel_y_dev, G.transfer_particles_indices_d, send_buffer_d, domainMin, domainMax, bt_non_pos );
  Load_Particles_to_Transfer_GPU_function( n_transfer, 5, n_fields_to_transfer, vel_z_dev, G.transfer_particles_indices_d, send_buffer_d, domainMin, domainMax, bt_non_pos );

  CHECK(cudaDeviceSynchronize());

  *n_send += n_transfer;
  // if ( *n_send > 0 ) printf( "###Transfered %ld  particles\n", *n_send);


}


void Particles_3D::Replace_Tranfered_Particles_GPU( int n_transfer ){

  // Replace the particles that were transfered
  Replace_Transfered_Particles_GPU_function( n_transfer, pos_x_dev, G.transfer_particles_indices_d, G.replace_particles_indices_d, false );
  Replace_Transfered_Particles_GPU_function( n_transfer, pos_y_dev, G.transfer_particles_indices_d, G.replace_particles_indices_d, false );
  Replace_Transfered_Particles_GPU_function( n_transfer, pos_z_dev, G.transfer_particles_indices_d, G.replace_particles_indices_d, false );
  Replace_Transfered_Particles_GPU_function( n_transfer, vel_x_dev, G.transfer_particles_indices_d, G.replace_particles_indices_d, false );
  Replace_Transfered_Particles_GPU_function( n_transfer, vel_y_dev, G.transfer_particles_indices_d, G.replace_particles_indices_d, false );
  Replace_Transfered_Particles_GPU_function( n_transfer, vel_z_dev, G.transfer_particles_indices_d, G.replace_particles_indices_d, false );

  CHECK(cudaDeviceSynchronize());
  // Update the local number of particles
  n_local -= n_transfer;

}


void Particles_3D::Load_Particles_to_Buffer_GPU( int direction, int side, Real *send_buffer_h, int buffer_length ){

  int n_transfer;

  n_transfer = Select_Particles_to_Transfer_GPU( direction, side );

  Copy_Transfer_Particles_to_Buffer_GPU( n_transfer, direction, side, send_buffer_h, buffer_length );

  Replace_Tranfered_Particles_GPU( n_transfer );

}


void Particles_3D::Copy_Transfer_Particles_from_Buffer_GPU(int n_recv, Real *recv_buffer_d ){

  int n_fields_to_transfer;

  part_int_t n_local_after = n_local + n_recv;
  if ( n_local_after > particles_array_size ){
    particles_array_size = Compute_Particles_GPU_Array_Size( n_local_after );
    printf("Reallocating GPU particles arrays \n" );
    Reallocate_and_Copy_Partciles_Array_Real( &pos_x_dev, n_local, particles_array_size  );
    Reallocate_and_Copy_Partciles_Array_Real( &pos_y_dev, n_local, particles_array_size  );
    Reallocate_and_Copy_Partciles_Array_Real( &pos_z_dev, n_local, particles_array_size  );
    Reallocate_and_Copy_Partciles_Array_Real( &vel_x_dev, n_local, particles_array_size  );
    Reallocate_and_Copy_Partciles_Array_Real( &vel_y_dev, n_local, particles_array_size  );
    Reallocate_and_Copy_Partciles_Array_Real( &vel_z_dev, n_local, particles_array_size  );
  }

  // Unload the particles that were transfered from the buffers
  n_fields_to_transfer = N_DATA_PER_PARTICLE_TRANSFER;
  Unload_Particles_to_Transfer_GPU_function( n_local, n_recv, 0, n_fields_to_transfer, pos_x_dev, recv_buffer_d  );
  Unload_Particles_to_Transfer_GPU_function( n_local, n_recv, 1, n_fields_to_transfer, pos_y_dev, recv_buffer_d  );
  Unload_Particles_to_Transfer_GPU_function( n_local, n_recv, 2, n_fields_to_transfer, pos_z_dev, recv_buffer_d  );
  Unload_Particles_to_Transfer_GPU_function( n_local, n_recv, 3, n_fields_to_transfer, vel_x_dev, recv_buffer_d  );
  Unload_Particles_to_Transfer_GPU_function( n_local, n_recv, 4, n_fields_to_transfer, vel_y_dev, recv_buffer_d  );
  Unload_Particles_to_Transfer_GPU_function( n_local, n_recv, 5, n_fields_to_transfer, vel_z_dev, recv_buffer_d  );
  //
  n_local += n_recv;
  // if ( n_recv > 0 ) printf( "###Unloaded %d  particles\n", n_recv );


}



void Particles_3D::Unload_Particles_from_Buffer_GPU( int direction, int side , Real *recv_buffer_h, int n_recv ){

  int buffer_size;
  Real domainMin, domainMax;
  Real *recv_buffer_d;

  if ( direction == 0 ){
    domainMin = G.domainMin_x;
    domainMin = G.domainMax_x;
    if ( side == 0 ){
      buffer_size   = G.recv_buffer_size_x0;
      recv_buffer_d = G.recv_buffer_x0_d;
    }
    if ( side == 1 ){
      buffer_size   = G.recv_buffer_size_x1;
      recv_buffer_d = G.recv_buffer_x1_d;
    }
  }
  if ( direction == 1 ){
    domainMin = G.domainMin_y;
    domainMin = G.domainMax_y;
    if ( side == 0 ){
      buffer_size   = G.recv_buffer_size_y0;
      recv_buffer_d = G.recv_buffer_y0_d;
    }
    if ( side == 1 ){
      buffer_size   = G.recv_buffer_size_y1;
      recv_buffer_d = G.recv_buffer_y1_d;
    }
  }
  if ( direction == 2 ){
    domainMin = G.domainMin_z;
    domainMin = G.domainMax_z;
    if ( side == 0 ){
      buffer_size   = G.recv_buffer_size_z0;
      recv_buffer_d = G.recv_buffer_z0_d;
    }
    if ( side == 1 ){
      buffer_size   = G.recv_buffer_size_z1;
      recv_buffer_d = G.recv_buffer_z1_d;
    }
  }

  CudaCheckError();

  Copy_Transfer_Particles_from_Buffer_GPU( n_recv, recv_buffer_d );

}




#endif //PARTICLES_GPU






#endif //MPI_CHOLLA
#endif //PARTICLES

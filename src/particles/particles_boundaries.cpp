#ifdef PARTICLES

#include <unistd.h>
#include <algorithm>
#include <iostream>
#include "../grid3D.h"
#include "../io.h"
#include "particles_3D.h"

#ifdef MPI_CHOLLA
#include "../mpi_routines.h"
#ifdef PARTICLES_GPU
#include "particles_boundaries_gpu.h"
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
  
  #ifdef PARTICLES_GPU
  Particles.Load_Particles_to_Buffer_GPU(0, 0, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_GPU

  MPI_Irecv(&Particles.n_recv_x0, 1, MPI_PART_INT, source[0], 0, world, &recv_request_n_particles[ireq_n_particles]);
  MPI_Isend(&Particles.n_send_x0, 1, MPI_PART_INT, dest[0],   1, world, &send_request_n_particles[0]);
  // if ( Particles.n_send_x0 > 0 )   if ( Particles.n_send_x0 > 0 ) std::cout << " Sent X0:  " << Particles.n_send_x0 <<  "  " << procID <<  "  to  "  <<  dest[0] <<  std::endl;
  buffer_length = Particles.n_send_x0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_x0_particles , &buffer_length_particles_x0_send, buffer_length );
  #ifdef PARTICLES_CPU
  Particles.Load_Particles_to_Buffer_CPU( 0, 0, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_CPU
  MPI_Isend(send_buffer_x0_particles, buffer_length, MPI_CHREAL, dest[0],   1, world, &send_request_particles_transfer[ireq_particles_transfer]);
}

void Grid3D::Load_and_Send_Particles_X1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  
  #ifdef PARTICLES_GPU
  Particles.Load_Particles_to_Buffer_GPU(0, 1, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_GPU

  MPI_Irecv(&Particles.n_recv_x1, 1, MPI_PART_INT, source[1], 1, world, &recv_request_n_particles[ireq_n_particles]);
  MPI_Isend(&Particles.n_send_x1, 1, MPI_PART_INT, dest[1],   0, world, &send_request_n_particles[1]);
  // if ( Particles.n_send_x1 > 0 )  std::cout << " Sent X1: " << Particles.n_send_x1 << std::endl;
  buffer_length = Particles.n_send_x1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_x1_particles , &buffer_length_particles_x1_send, buffer_length );
  #ifdef PARTICLES_CPU
  Particles.Load_Particles_to_Buffer_CPU( 0, 1, send_buffer_x1_particles,  buffer_length_particles_x1_send  );
  #endif //PARTICLES_CPU
  MPI_Isend(send_buffer_x1_particles, buffer_length, MPI_CHREAL, dest[1],   0, world, &send_request_particles_transfer[ireq_particles_transfer]);\
}

void Grid3D::Load_and_Send_Particles_Y0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  
  #ifdef PARTICLES_GPU
  Particles.Load_Particles_to_Buffer_GPU(1, 0, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_GPU

  MPI_Isend(&Particles.n_send_y0, 1, MPI_PART_INT, dest[2],   3, world, &send_request_n_particles[0]);
  MPI_Irecv(&Particles.n_recv_y0, 1, MPI_PART_INT, source[2], 2, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_y0 > 0 )   std::cout << " Sent Y0: " << Particles.n_send_y0 << std::endl;
  buffer_length = Particles.n_send_y0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_y0_particles , &buffer_length_particles_y0_send, buffer_length );
  #ifdef PARTICLES_CPU
  Particles.Load_Particles_to_Buffer_CPU( 1, 0, send_buffer_y0_particles,  buffer_length_particles_y0_send  );
  #endif //PARTICLES_CPU
  MPI_Isend(send_buffer_y0_particles, buffer_length, MPI_CHREAL, dest[2],   3, world, &send_request_particles_transfer[ireq_particles_transfer]);
}

void Grid3D::Load_and_Send_Particles_Y1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  
  #ifdef PARTICLES_GPU
  Particles.Load_Particles_to_Buffer_GPU(1, 1, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_GPU

  MPI_Isend(&Particles.n_send_y1, 1, MPI_PART_INT, dest[3],   2, world, &send_request_n_particles[1]);
  MPI_Irecv(&Particles.n_recv_y1, 1, MPI_PART_INT, source[3], 3, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_y1 > 0 )  std::cout << " Sent Y1: " << Particles.n_send_y1 << std::endl;
  buffer_length = Particles.n_send_y1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_y1_particles , &buffer_length_particles_y1_send, buffer_length );
  #ifdef PARTICLES_CPU
  Particles.Load_Particles_to_Buffer_CPU( 1, 1, send_buffer_y1_particles,  buffer_length_particles_y1_send  );
  #endif //PARTICLES_CPU
  MPI_Isend(send_buffer_y1_particles, buffer_length, MPI_CHREAL, dest[3],   2, world, &send_request_particles_transfer[ireq_particles_transfer]);
}

void Grid3D::Load_and_Send_Particles_Z0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  
  #ifdef PARTICLES_GPU
  Particles.Load_Particles_to_Buffer_GPU(2, 0, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_GPU

  MPI_Isend(&Particles.n_send_z0, 1, MPI_PART_INT, dest[4],   5, world, &send_request_n_particles[0]);
  MPI_Irecv(&Particles.n_recv_z0, 1, MPI_PART_INT, source[4], 4, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_z0 > 0 )   std::cout << " Sent Z0: " << Particles.n_send_z0 << std::endl;
  buffer_length = Particles.n_send_z0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_z0_particles , &buffer_length_particles_z0_send, buffer_length );
  #ifdef PARTICLES_CPU
  Particles.Load_Particles_to_Buffer_CPU( 2, 0, send_buffer_z0_particles,  buffer_length_particles_z0_send  );
  #endif //PARTICLES_CPU
  MPI_Isend(send_buffer_z0_particles, buffer_length, MPI_CHREAL, dest[4],   5, world, &send_request_particles_transfer[ireq_particles_transfer]);
}

void Grid3D::Load_and_Send_Particles_Z1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  
  #ifdef PARTICLES_GPU
  Particles.Load_Particles_to_Buffer_GPU(2, 1, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  #endif //PARTICLES_GPU

  MPI_Isend(&Particles.n_send_z1, 1, MPI_CHREAL, dest[5],   4, world, &send_request_n_particles[1]);
  MPI_Irecv(&Particles.n_recv_z1, 1, MPI_CHREAL, source[5], 5, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_z1 > 0 )   std::cout << " Sent Z1: " << Particles.n_send_z1 << std::endl;
  buffer_length = Particles.n_send_z1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_z1_particles , &buffer_length_particles_z1_send, buffer_length );
  #ifdef PARTICLES_CPU
  Particles.Load_Particles_to_Buffer_CPU( 2, 1, send_buffer_z1_particles,  buffer_length_particles_z1_send  );
  #endif //PARTICLES_CPU
  MPI_Isend(send_buffer_z1_particles, buffer_length, MPI_CHREAL, dest[5],   4, world, &send_request_particles_transfer[ireq_particles_transfer]);
}

//Unload the Transfered particles from the MPI_buffer, after buffer was received
void Grid3D::Unload_Particles_from_Buffer_X0( int *flags ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 0, 0, recv_buffer_x0_particles, Particles.n_recv_x0, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
}

void Grid3D::Unload_Particles_from_Buffer_X1( int *flags  ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 0, 1, recv_buffer_x1_particles, Particles.n_recv_x1, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
}

void Grid3D::Unload_Particles_from_Buffer_Y0( int *flags ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 1, 0, recv_buffer_y0_particles, Particles.n_recv_y0, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
}

void Grid3D::Unload_Particles_from_Buffer_Y1( int *flags  ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 1, 1, recv_buffer_y1_particles, Particles.n_recv_y1, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
}

void Grid3D::Unload_Particles_from_Buffer_Z0( int *flags ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 2, 0, recv_buffer_z0_particles, Particles.n_recv_z0, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
}

void Grid3D::Unload_Particles_from_Buffer_Z1( int *flags ){
  #ifdef PARTICLES_CPU
  Particles.Unload_Particles_from_Buffer_CPU( 2, 1, recv_buffer_z1_particles, Particles.n_recv_z1, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send, flags);
  #endif//PARTICLES_CPU
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



void Particles_3D::Load_Particles_to_Buffer_GPU( int direction, int side, Real *send_buffer, int buffer_length  ){
  
  part_int_t *n_send;
  Real *pos;
  Real domainMin, domainMax;
  
  if ( direction == 0 ){
    pos = pos_x_dev;
    domainMax = G.xMax;
    domainMin = G.xMin;
    if ( side == 0 ) n_send = &n_send_x0;
    if ( side == 1 ) n_send = &n_send_x1;
  }  
  if ( direction == 1 ){
    pos = pos_y_dev;
    domainMax = G.yMax;
    domainMin = G.yMin;
    if ( side == 0 ) n_send = &n_send_y0;
    if ( side == 1 ) n_send = &n_send_y1;
  }  
  if ( direction == 2 ){
    pos = pos_z_dev;
    domainMax = G.zMax;
    domainMin = G.zMin;
    if ( side == 0 ) n_send = &n_send_z0;
    if ( side == 1 ) n_send = &n_send_z1;
  }
  
  // Check if size_block_array is large enough to fit the blocks
  
  //Set the number of particles that will be sent
  *n_send += Select_Particles_to_Transfer_GPU_function(  n_local, side, domainMin, domainMax, pos, G.n_transfer_d, G.n_transfer_h, G.transfer_particles_flags_d, G.transfer_particles_indxs_d, G.transfer_particles_partial_sum_d, G.transfer_particles_sum_d  ); 
  
  if ( *n_send > 0 ) printf( "###Transfered %d  particles\n", *n_send);
  
}



#endif //PARTICLES_GPU






#endif //MPI_CHOLLA
#endif //PARTICLES
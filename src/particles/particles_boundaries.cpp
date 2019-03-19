#ifdef PARTICLES

#include <unistd.h>
#include <algorithm>
#include <iostream>
#include "../grid3D.h"
#include "particles_3D.h"

#ifdef MPI_CHOLLA
#include "../mpi_routines.h"
#endif

void Grid3D::Transfer_Particles_Boundaries( struct parameters P ){
  
  
}


void Grid3D::Transfer_Particles_Boundaries_MPI( struct parameters P ){
  
  
}

Real Get_and_Remove_Real( part_int_t indx, real_vector_t &vec ){
  Real value = vec[indx];
  vec[indx] = vec.back();
  vec.pop_back();
  return value;
}

void Remove_Real( part_int_t indx, real_vector_t &vec ){
  vec[indx] = vec.back();
  vec.pop_back();
}

Real Get_and_Remove_partID( part_int_t indx, int_vector_t &vec ){
  Real value = (Real) vec[indx];
  vec[indx] = vec.back();
  vec.pop_back();
  return value;
}

part_int_t Real_to_part_int( Real inVal ){
  part_int_t outVal = (part_int_t) inVal;
  if ( (inVal - outVal) > 0.1 ) outVal += 1;
  if ( fabs(outVal - inVal) > 0.5 ) outVal -= 1;
  return outVal;
}

#ifdef MPI_CHOLLA


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
    Unload_Particles_From_Buffers_BLOCK(status.MPI_TAG);
  }
}

void Grid3D::Unload_Particles_From_Buffers_BLOCK(int index){

  // Make sure not to unload when not transferin particles
  if ( Particles.TRANSFER_DENSITY_BOUNDARIES ) return;
  if ( H.TRANSFER_HYDRO_BOUNDARIES ) return;
  #ifdef GRAVITY
  if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ) return;
  #endif

  if( index == 0) {
    Unload_Particles_from_Buffer_X0(  );
  }

  if( index == 1){
    Unload_Particles_from_Buffer_X1();
  }

  if( index == 2){
    Unload_Particles_from_Buffer_Y0();
  }

  if( index == 3){
    Unload_Particles_from_Buffer_Y1();
  }

  if( index == 4){
    Unload_Particles_from_Buffer_Z0();
  }

  if( index == 5){
    Unload_Particles_from_Buffer_Z1();
  }

}



void Grid3D::Wait_and_Recv_Particles_Transfer_BLOCK(int dir, int *flags)
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
    Load_N_Particles_Transfer(status.MPI_TAG, &ireq_particles_transfer);
  }
}


void Grid3D::Load_N_Particles_Transfer(int index, int *ireq_particles_transfer){

  int buffer_length;
  // std::cout << "ireq: " << *ireq_particles_transfer << std::endl;
  if ( index == 0){
    buffer_length = Particles.n_recv_x0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_x0_particles , &buffer_length_particles_x0_recv, buffer_length );
    // if ( Particles.n_recv_x0 ){
      // std::cout << " Recv X0: " << Particles.n_recv_x0 << std::endl;
      MPI_Irecv(recv_buffer_x0_particles, buffer_length, MPI_CHREAL, source[0], 0, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
    // }
  }
  if ( index == 1){
    buffer_length = Particles.n_recv_x1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_x1_particles , &buffer_length_particles_x1_recv, buffer_length );
    // if ( Particles.n_recv_x1 ){
      // if ( Particles.n_recv_x1 > 0 ) std::cout << " Recv X1:  " << Particles.n_recv_x1 <<  "  " << procID <<  "  from "  <<  source[1] <<  std::endl;
      MPI_Irecv(recv_buffer_x1_particles, buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
    // }
  }
  if ( index == 2){
    buffer_length = Particles.n_recv_y0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_y0_particles , &buffer_length_particles_y0_recv, buffer_length );
    // if ( Particles.n_recv_y0 ){
      // std::cout << " Recv Y0: " << Particles.n_recv_y0 << std::endl;
      MPI_Irecv(recv_buffer_y0_particles, buffer_length, MPI_CHREAL, source[2], 2, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
    // }
  }
  if ( index == 3){
    buffer_length = Particles.n_recv_y1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_y1_particles , &buffer_length_particles_y1_recv, buffer_length );
    // if ( Particles.n_recv_y1 ){
      // std::cout << " Recv Y1: " << Particles.n_recv_y1 << std::endl;
      MPI_Irecv(recv_buffer_y1_particles, buffer_length, MPI_CHREAL, source[3], 3, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
    // }
  }
  if ( index == 4){
    buffer_length = Particles.n_recv_z0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_z0_particles , &buffer_length_particles_z0_recv, buffer_length );
    // if ( Particles.n_recv_z0 ){
      // std::cout << " Recv Z0: " << Particles.n_recv_z0 << std::endl;
      MPI_Irecv(recv_buffer_z0_particles, buffer_length, MPI_CHREAL, source[4], 4, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
    // }
  }
  if ( index == 5){
    buffer_length = Particles.n_recv_z1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &recv_buffer_z1_particles , &buffer_length_particles_z1_recv, buffer_length );
    // if ( Particles.n_recv_z1 ){
      // std::cout << " Recv Z1: " << Particles.n_recv_z1 << std::endl;
      MPI_Irecv(recv_buffer_z1_particles, buffer_length, MPI_CHREAL, source[5], 5, world, &recv_request_particles_transfer[*ireq_particles_transfer]);
    // }
  }

  *ireq_particles_transfer += 1;
}



void Grid3D::Load_and_Send_Particles_X0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;

  MPI_Irecv(&Particles.n_recv_x0, 1, MPI_PART_INT, source[0], 0, world, &recv_request_n_particles[ireq_n_particles]);
  MPI_Isend(&Particles.n_send_x0, 1, MPI_PART_INT, dest[0],   1, world, &send_request_n_particles[0]);
  // if ( Particles.n_send_x0 > 0 ){
  //   if ( Particles.n_send_x0 > 0 ) std::cout << " Sent X0:  " << Particles.n_send_x0 <<  "  " << procID <<  "  to  "  <<  dest[0] <<  std::endl;
    buffer_length = Particles.n_send_x0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &send_buffer_x0_particles , &buffer_length_particles_x0_send, buffer_length );
    Particles.Load_Particles_to_Buffer( 0, 0, send_buffer_x0_particles,  buffer_length_particles_x0_send );
    MPI_Isend(send_buffer_x0_particles, buffer_length, MPI_CHREAL, dest[0],   1, world, &send_request_particles_transfer[ireq_particles_transfer]);
  // }
}

void Grid3D::Load_and_Send_Particles_X1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Irecv(&Particles.n_recv_x1, 1, MPI_PART_INT, source[1], 1, world, &recv_request_n_particles[ireq_n_particles]);
  MPI_Isend(&Particles.n_send_x1, 1, MPI_PART_INT, dest[1],   0, world, &send_request_n_particles[1]);
  // if ( Particles.n_send_x1 > 0 ){
  //   std::cout << " Sent X1: " << Particles.n_send_x1 << std::endl;
    buffer_length = Particles.n_send_x1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &send_buffer_x1_particles , &buffer_length_particles_x1_send, buffer_length );
    Particles.Load_Particles_to_Buffer( 0, 1, send_buffer_x1_particles,  buffer_length_particles_x1_send  );
    MPI_Isend(send_buffer_x1_particles, buffer_length, MPI_CHREAL, dest[1],   0, world, &send_request_particles_transfer[ireq_particles_transfer]);\
  // }
}

void Grid3D::Load_and_Send_Particles_Y0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Isend(&Particles.n_send_y0, 1, MPI_PART_INT, dest[2],   3, world, &send_request_n_particles[0]);
  MPI_Irecv(&Particles.n_recv_y0, 1, MPI_PART_INT, source[2], 2, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_y0 > 0 ){
  //   std::cout << " Sent Y0: " << Particles.n_send_y0 << std::endl;
    buffer_length = Particles.n_send_y0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &send_buffer_y0_particles , &buffer_length_particles_y0_send, buffer_length );
    Particles.Load_Particles_to_Buffer( 1, 0, send_buffer_y0_particles,  buffer_length_particles_y0_send  );
    MPI_Isend(send_buffer_y0_particles, buffer_length, MPI_CHREAL, dest[2],   3, world, &send_request_particles_transfer[ireq_particles_transfer]);
  // }

}
void Grid3D::Load_and_Send_Particles_Y1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Isend(&Particles.n_send_y1, 1, MPI_PART_INT, dest[3],   2, world, &send_request_n_particles[1]);
  MPI_Irecv(&Particles.n_recv_y1, 1, MPI_PART_INT, source[3], 3, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_y1 > 0 ){
  //   std::cout << " Sent Y1: " << Particles.n_send_y1 << std::endl;
    buffer_length = Particles.n_send_y1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &send_buffer_y1_particles , &buffer_length_particles_y1_send, buffer_length );
    Particles.Load_Particles_to_Buffer( 1, 1, send_buffer_y1_particles,  buffer_length_particles_y1_send  );
    MPI_Isend(send_buffer_y1_particles, buffer_length, MPI_CHREAL, dest[3],   2, world, &send_request_particles_transfer[ireq_particles_transfer]);
  // }

}
void Grid3D::Load_and_Send_Particles_Z0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Isend(&Particles.n_send_z0, 1, MPI_PART_INT, dest[4],   5, world, &send_request_n_particles[0]);
  MPI_Irecv(&Particles.n_recv_z0, 1, MPI_PART_INT, source[4], 4, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_z0 > 0 ){
  //   std::cout << " Sent Z0: " << Particles.n_send_z0 << std::endl;
    buffer_length = Particles.n_send_z0 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &send_buffer_z0_particles , &buffer_length_particles_z0_send, buffer_length );
    Particles.Load_Particles_to_Buffer( 2, 0, send_buffer_z0_particles,  buffer_length_particles_z0_send  );
    MPI_Isend(send_buffer_z0_particles, buffer_length, MPI_CHREAL, dest[4],   5, world, &send_request_particles_transfer[ireq_particles_transfer]);
  // }

}
void Grid3D::Load_and_Send_Particles_Z1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Isend(&Particles.n_send_z1, 1, MPI_CHREAL, dest[5],   4, world, &send_request_n_particles[1]);
  MPI_Irecv(&Particles.n_recv_z1, 1, MPI_CHREAL, source[5], 5, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_z1 > 0 ){
  //   std::cout << " Sent Z1: " << Particles.n_send_z1 << std::endl;
    buffer_length = Particles.n_send_z1 * N_DATA_PER_PARTICLE_TRANSFER;
    Check_and_Grow_Particles_Buffer( &send_buffer_z1_particles , &buffer_length_particles_z1_send, buffer_length );
    Particles.Load_Particles_to_Buffer( 2, 1, send_buffer_z1_particles,  buffer_length_particles_z1_send  );
    MPI_Isend(send_buffer_z1_particles, buffer_length, MPI_CHREAL, dest[5],   4, world, &send_request_particles_transfer[ireq_particles_transfer]);
  // }
}

void Grid3D::Unload_Particles_from_Buffer_X0(){
  Particles.Unload_Particles_from_Buffer( 0, 0, recv_buffer_x0_particles, Particles.n_recv_x0, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send);
}

void Grid3D::Unload_Particles_from_Buffer_X1(){
  Particles.Unload_Particles_from_Buffer( 0, 1, recv_buffer_x1_particles, Particles.n_recv_x1, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send);
}

void Grid3D::Unload_Particles_from_Buffer_Y0(){
  Particles.Unload_Particles_from_Buffer( 1, 0, recv_buffer_y0_particles, Particles.n_recv_y0, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send);
}

void Grid3D::Unload_Particles_from_Buffer_Y1(){
  Particles.Unload_Particles_from_Buffer( 1, 1, recv_buffer_y1_particles, Particles.n_recv_y1, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send);
}

void Grid3D::Unload_Particles_from_Buffer_Z0(){
  Particles.Unload_Particles_from_Buffer( 2, 0, recv_buffer_z0_particles, Particles.n_recv_z0, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send);
}

void Grid3D::Unload_Particles_from_Buffer_Z1(){
  Particles.Unload_Particles_from_Buffer( 2, 1, recv_buffer_z1_particles, Particles.n_recv_z1, send_buffer_y0_particles, send_buffer_y1_particles, send_buffer_z0_particles, send_buffer_z1_particles, buffer_length_particles_y0_send , buffer_length_particles_y1_send, buffer_length_particles_z0_send, buffer_length_particles_z1_send);
}

void Particles_3D::Load_Particles_to_Buffer( int direction, int side, Real *send_buffer, int buffer_length  ){

  part_int_t n_out;
  part_int_t n_send;
  int_vector_t *out_indxs_vec;
  part_int_t *n_in_buffer;

  if ( direction == 0 ){
    if ( side == 0 ){
      out_indxs_vec = &out_indxs_vec_x0;
      n_send = n_send_x0;
      n_in_buffer = &n_in_buffer_x0;
    }
    if ( side == 1 ){
      out_indxs_vec = &out_indxs_vec_x1;
      n_send = n_send_x1;
      n_in_buffer = &n_in_buffer_x1;
    }
  }
  if ( direction == 1 ){
    if ( side == 0 ){
      out_indxs_vec = &out_indxs_vec_y0;
      n_send = n_send_y0;
      n_in_buffer = &n_in_buffer_y0;
    }
    if ( side == 1 ){
      out_indxs_vec = &out_indxs_vec_y1;
      n_send = n_send_y1;
      n_in_buffer = &n_in_buffer_y1;
    }
  }
  if ( direction == 2 ){
    if ( side == 0 ){
      out_indxs_vec = &out_indxs_vec_z0;
      n_send = n_send_z0;
      n_in_buffer = &n_in_buffer_z0;
    }
    if ( side == 1 ){
      out_indxs_vec = &out_indxs_vec_z1;
      n_send = n_send_z1;
      n_in_buffer = &n_in_buffer_z1;
    }
  }

  part_int_t offset, offset_extra;

  n_out = out_indxs_vec->size();
  offset = *n_in_buffer*N_DATA_PER_PARTICLE_TRANSFER;

  part_int_t indx, pIndx;
  for ( indx=0; indx<n_out; indx++ ){
    pIndx = out_indxs_vec->back();
    send_buffer[ offset + 0 ] = Get_and_Remove_Real( pIndx, pos_x );
    send_buffer[ offset + 1 ] = Get_and_Remove_Real( pIndx, pos_y );
    send_buffer[ offset + 2 ] = Get_and_Remove_Real( pIndx, pos_z );
    send_buffer[ offset + 3 ] = Get_and_Remove_Real( pIndx, vel_x );
    send_buffer[ offset + 4 ] = Get_and_Remove_Real( pIndx, vel_y );
    send_buffer[ offset + 5 ] = Get_and_Remove_Real( pIndx, vel_z );

    offset_extra = offset + 5;
    #ifndef SINGLE_PARTICLE_MASS
    offset_extra += 1;
    send_buffer[ offset_extra ] = Get_and_Remove_Real( pIndx, mass );
    #endif
    #ifdef PARTICLE_IDS
    offset_extra += 1;
    send_buffer[ offset_extra ] = (Real) Get_and_Remove_partID( pIndx, partIDs );
    #endif

    Remove_Real( pIndx, grav_x );
    Remove_Real( pIndx, grav_y );
    Remove_Real( pIndx, grav_z );
    // *n_send += 1;
    n_local -= 1;
    *n_in_buffer += 1;
    out_indxs_vec->pop_back();
    offset += N_DATA_PER_PARTICLE_TRANSFER;
    if ( offset > buffer_length ) std::cout << "ERROR: Buffer length exceeded on particles transfer" << std::endl;
  }

  if (out_indxs_vec->size() > 0 ) std::cout << "ERROR: Particle output vector not empty after transfer " << std::endl;
}


void Particles_3D::Add_Particle_To_Buffer( Real *buffer, part_int_t n_in_buffer, int buffer_length, Real pId, Real pMass,
                            Real pPos_x, Real pPos_y, Real pPos_z, Real pVel_x, Real pVel_y, Real pVel_z){

  int offset, offset_extra;
  offset = n_in_buffer * N_DATA_PER_PARTICLE_TRANSFER;

  if (offset > buffer_length ) if ( offset > buffer_length ) std::cout << "ERROR: Buffer length exceeded on particles transfer" << std::endl;
  buffer[offset + 0] = pPos_x;
  buffer[offset + 1] = pPos_y;
  buffer[offset + 2] = pPos_z;
  buffer[offset + 3] = pVel_x;
  buffer[offset + 4] = pVel_y;
  buffer[offset + 5] = pVel_z;

  offset_extra = offset + 5;
  #ifndef SINGLE_PARTICLE_MASS
  offset_extra += 1;
  buffer[ offset_extra ] = pMass;
  #endif
  #ifdef PARTICLE_IDS
  offset_extra += 1;
  buffer[offset_extra] = pId;
  #endif


}


void Particles_3D::Add_Particle_To_Vectors( Real pId, Real pMass,
                            Real pPos_x, Real pPos_y, Real pPos_z,
                            Real pVel_x, Real pVel_y, Real pVel_z ){
  pos_x.push_back( pPos_x );
  pos_y.push_back( pPos_y );
  pos_z.push_back( pPos_z );
  vel_x.push_back( pVel_x );
  vel_y.push_back( pVel_y );
  vel_z.push_back( pVel_z );
  #ifndef SINGLE_PARTICLE_MASS
  mass.push_back( pMass );
  #endif
  #ifdef PARTICLE_IDS
  partIDs.push_back( Real_to_part_int(pId) );
  #endif
  grav_x.push_back(0);
  grav_y.push_back(0);
  grav_z.push_back(0);

  n_local += 1;
}

void Particles_3D::Unload_Particles_from_Buffer( int direction, int side, Real *recv_buffer, part_int_t n_recv,
      Real *send_buffer_y0, Real *send_buffer_y1, Real *send_buffer_z0, Real *send_buffer_z1, int buffer_length_y0, int buffer_length_y1, int buffer_length_z0, int buffer_length_z1){

  int offset_buff, offset_extra;
  part_int_t pId;
  Real pMass, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z;

  offset_buff = 0;
  part_int_t indx;
  for ( indx=0; indx<n_recv; indx++ ){
    pPos_x = recv_buffer[ offset_buff + 0 ];
    pPos_y = recv_buffer[ offset_buff + 1 ];
    pPos_z = recv_buffer[ offset_buff + 2 ];
    pVel_x = recv_buffer[ offset_buff + 3 ];
    pVel_y = recv_buffer[ offset_buff + 4 ];
    pVel_z = recv_buffer[ offset_buff + 5 ];

    offset_extra = offset_buff + 5;
    #if SINGLE_PARTICLE_MASS
    pMass = particle_mass;
    #else
    offset_extra += 1;
    pMass  = recv_buffer[ offset_extra ];
    #endif
    #ifdef PARTICLE_IDS
    offset_extra += 1;
    pId    = recv_buffer[ offset_extra ];
    #else
    pId = 0;
    #endif

    offset_buff += N_DATA_PER_PARTICLE_TRANSFER;
    if ( pPos_x <  G.domainMin_x ) pPos_x += ( G.domainMax_x - G.domainMin_x );
    if ( pPos_x >= G.domainMax_x ) pPos_x -= ( G.domainMax_x - G.domainMin_x );
    if ( ( pPos_x < G.xMin ) || ( pPos_x >= G.xMax )  ){
      #ifdef PARTICLE_IDS
      std::cout << "ERROR Particle Transfer out of X domain    pID: " << pId << std::endl;
      #else
      std::cout << "ERROR Particle Transfer out of X domain" << std::endl;
      #endif
      std::cout << " posX: " << pPos_x << " velX: " << pVel_x << std::endl;
      std::cout << " posY: " << pPos_y << " velY: " << pVel_y << std::endl;
      std::cout << " posZ: " << pPos_z << " velZ: " << pVel_z << std::endl;
      std::cout << " Domain X: " << G.xMin << "  " << G.xMax << std::endl;
      std::cout << " Domain Y: " << G.yMin << "  " << G.yMax << std::endl;
      std::cout << " Domain Z: " << G.zMin << "  " << G.zMax << std::endl;
      continue;
    }

    if (  direction == 1 ){
      if ( pPos_y <  G.domainMin_y ) pPos_y += ( G.domainMax_y - G.domainMin_y );
      if ( pPos_y >= G.domainMax_y ) pPos_y -= ( G.domainMax_y - G.domainMin_y );
    }
    if ( (direction==1 || direction==2) && (( pPos_y < G.yMin ) || ( pPos_y >= G.yMax ))  ){
      #ifdef PARTICLE_IDS
      std::cout << "ERROR Particle Transfer out of Y domain    pID: " << pId << std::endl;
      #else
      std::cout << "ERROR Particle Transfer out of Y domain" << std::endl;
      #endif
      std::cout << " posX: " << pPos_x << " velX: " << pVel_x << std::endl;
      std::cout << " posY: " << pPos_y << " velY: " << pVel_y << std::endl;
      std::cout << " posZ: " << pPos_z << " velZ: " << pVel_z << std::endl;
      std::cout << " Domain X: " << G.xMin << "  " << G.xMax << std::endl;
      std::cout << " Domain Y: " << G.yMin << "  " << G.yMax << std::endl;
      std::cout << " Domain Z: " << G.zMin << "  " << G.zMax << std::endl;
      continue;
    }

    if (direction  == 0 ){
      if ( pPos_y < G.yMin ){
        // std::cout << "Added Y0" << std::endl;
        Add_Particle_To_Buffer( send_buffer_y0, n_in_buffer_y0, buffer_length_y0, pId, pMass, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
        n_send_y0 += 1;
        n_in_buffer_y0 += 1;
        continue;
      }
      if ( pPos_y >= G.yMax ){
        // std::cout << "Added Y1" << std::endl;
        Add_Particle_To_Buffer( send_buffer_y1, n_in_buffer_y1, buffer_length_y1, pId, pMass, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
        n_send_y1 += 1;
        n_in_buffer_y1 += 1;
        continue;
      }
    }

    if (  direction == 2 ){
      if ( pPos_z <  G.domainMin_z ) pPos_z += ( G.domainMax_z - G.domainMin_z );
      if ( pPos_z >= G.domainMax_z ) pPos_z -= ( G.domainMax_z - G.domainMin_z );
    }
    if ( (direction==2) && (( pPos_z < G.zMin ) || ( pPos_z >= G.zMax ))  ){
      #ifdef PARTICLE_IDS
      std::cout << "ERROR Particle Transfer out of Z domain    pID: " << pId << std::endl;
      #else
      std::cout << "ERROR Particle Transfer out of Z domain" << std::endl;
      #endif
      std::cout << " posX: " << pPos_x << " velX: " << pVel_x << std::endl;
      std::cout << " posY: " << pPos_y << " velY: " << pVel_y << std::endl;
      std::cout << " posZ: " << pPos_z << " velZ: " << pVel_z << std::endl;
      std::cout << " Domain X: " << G.xMin << "  " << G.xMax << std::endl;
      std::cout << " Domain Y: " << G.yMin << "  " << G.yMax << std::endl;
      std::cout << " Domain Z: " << G.zMin << "  " << G.zMax << std::endl;
      continue;
    }
  //
    if (direction  !=2 ){
      if ( pPos_z < G.zMin ){
        // std::cout << "Added Z0" << std::endl;
        Add_Particle_To_Buffer( send_buffer_z0, n_in_buffer_z0, buffer_length_z0, pId, pMass, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
        n_send_z0 += 1;
        n_in_buffer_z0 += 1;
        continue;
      }
      if ( pPos_z >= G.zMax ){
        // std::cout << "Added Z1" << std::endl;
        Add_Particle_To_Buffer( send_buffer_z1, n_in_buffer_z1, buffer_length_z1, pId, pMass, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
        n_send_z1 += 1;
        n_in_buffer_z1 += 1;
        continue;
      }
    }
    Add_Particle_To_Vectors( pId, pMass, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
  }
}









void Particles_3D::Select_Particles_to_Transfer( int dir ){

  int i;
  bool already_transfered, pID_in_transfers;
  part_int_t pIndx;
  if ( dir == 0 ){
    out_indxs_vec_x0.clear();
    out_indxs_vec_x1.clear();
    for ( pIndx=0; pIndx<n_local; pIndx++ ){
      if ( pos_x[pIndx] < G.xMin ){
        out_indxs_vec_x0.push_back( pIndx );
        continue;
      }
      if ( pos_x[pIndx] >= G.xMax ){
        out_indxs_vec_x1.push_back( pIndx );
        continue;
      }
    }
    std::sort(out_indxs_vec_x0.begin(), out_indxs_vec_x0.end());
    std::sort(out_indxs_vec_x1.begin(), out_indxs_vec_x1.end());
    n_send_x0 += out_indxs_vec_x0.size();
    n_send_x1 += out_indxs_vec_x1.size();
    return;
  }
  if ( dir == 1 ){
    out_indxs_vec_y0.clear();
    out_indxs_vec_y1.clear();
    for ( pIndx=0; pIndx<n_local; pIndx++ ){
      already_transfered = false;
      if ( pos_x[pIndx] < G.xMin || pos_x[pIndx] >= G.xMax ) already_transfered = true;
      if ( already_transfered ){
        continue;
      }
      if ( pos_y[pIndx] < G.yMin ){
        out_indxs_vec_y0.push_back( pIndx );
        continue;
      }
      if ( pos_y[pIndx] >= G.yMax ){
        out_indxs_vec_y1.push_back( pIndx );
        continue;
      }
    }
    std::sort(out_indxs_vec_y0.begin(), out_indxs_vec_y0.end());
    std::sort(out_indxs_vec_y1.begin(), out_indxs_vec_y1.end());
    n_send_y0 += out_indxs_vec_y0.size();
    n_send_y1 += out_indxs_vec_y1.size();
  }
  if ( dir == 2 ){
    out_indxs_vec_z0.clear();
    out_indxs_vec_z1.clear();
    for ( pIndx=0; pIndx<n_local; pIndx++ ){
      already_transfered = false;
      if ( pos_x[pIndx] < G.xMin || pos_x[pIndx] >= G.xMax || pos_y[pIndx] < G.yMin || pos_y[pIndx] >= G.yMax ) already_transfered = true;
      if ( already_transfered ){
        continue;
      }
      if ( pos_z[pIndx] < G.zMin ){
        out_indxs_vec_z0.push_back( pIndx );
        continue;
      }
      if ( pos_z[pIndx] >= G.zMax ){
        out_indxs_vec_z1.push_back( pIndx );
        continue;
      }
    }
    std::sort(out_indxs_vec_z0.begin(), out_indxs_vec_z0.end());
    std::sort(out_indxs_vec_z1.begin(), out_indxs_vec_z1.end());
    n_send_z0 += out_indxs_vec_z0.size();
    n_send_z1 += out_indxs_vec_z1.size();
  }
}


void Particles_3D::Clear_Particles_For_Transfer( void ){
  n_transfer_x0 = 0;
  n_transfer_x1 = 0;
  n_transfer_y0 = 0;
  n_transfer_y1 = 0;
  n_transfer_z0 = 0;
  n_transfer_z1 = 0;

  n_send_x0 = 0;
  n_send_x1 = 0;
  n_send_y0 = 0;
  n_send_y1 = 0;
  n_send_z0 = 0;
  n_send_z1 = 0;

  n_recv_x0 = 0;
  n_recv_x1 = 0;
  n_recv_y0 = 0;
  n_recv_y1 = 0;
  n_recv_z0 = 0;
  n_recv_z1 = 0;

  n_in_buffer_x0 = 0;
  n_in_buffer_x1 = 0;
  n_in_buffer_y0 = 0;
  n_in_buffer_y1 = 0;
  n_in_buffer_z0 = 0;
  n_in_buffer_z1 = 0;
  
  Clear_Vectors_For_Transfers();
}

void Particles_3D::Clear_Vectors_For_Transfers( void ){
  out_indxs_vec_x0.clear();
  out_indxs_vec_x1.clear();
  out_indxs_vec_y0.clear();
  out_indxs_vec_y1.clear();
  out_indxs_vec_z0.clear();
  out_indxs_vec_z1.clear();
}


#endif //MPI_CHOLLA
#endif //PARTICLES
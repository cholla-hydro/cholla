#ifdef PARTICLES

#include <unistd.h>
#include <algorithm>
#include <iostream>
#include "../grid3D.h"
#include "../io.h"
#include "particles_3D.h"

#ifdef MPI_CHOLLA
#include "../mpi_routines.h"
#endif


//Get and remove value at index on vector
Real Get_and_Remove_Real( part_int_t indx, real_vector_t &vec ){
  Real value = vec[indx]; 
  vec[indx] = vec.back(); //The item at the specified index is replaced by the last item in the vector
  vec.pop_back(); //The last item in the vector is discarded
  return value;
}

//Remove value at index on vector 
void Remove_Real( part_int_t indx, real_vector_t &vec ){
  vec[indx] = vec.back(); //The item at the specified index is replaced by the last item in the vector
  vec.pop_back(); //The last item in the vector is discarded
}


//Get and remove integer value at index on vector
Real Get_and_Remove_partID( part_int_t indx, int_vector_t &vec ){
  Real value = (Real) vec[indx];
  vec[indx] = vec.back();
  vec.pop_back();
  return value;
}


//Remove integer value at index on vector
void Remove_ID( part_int_t indx, int_vector_t &vec ){
  vec[indx] = vec.back();
  vec.pop_back();
}

//Convert Real to Integer
part_int_t Real_to_part_int( Real inVal ){
  part_int_t outVal = (part_int_t) inVal;
  if ( (inVal - outVal) > 0.1 ) outVal += 1;
  if ( fabs(outVal - inVal) > 0.5 ) outVal -= 1;
  return outVal;
}


//Set periodic boundaries for particles. Only when not using MPI
void Grid3D::Set_Particles_Boundary( int dir, int side ){
  
  Real d_min, d_max, L;
  
  if ( dir == 0 ){
    d_min = Particles.G.zMin;
    d_max = Particles.G.zMax;
  }
  if ( dir == 1 ){
    d_min = Particles.G.yMin;
    d_max = Particles.G.yMax;
  }
  if ( dir == 2 ){
    d_min = Particles.G.zMin;
    d_max = Particles.G.zMax;
  }
  
  L = d_max - d_min;
  
  bool changed_pos;
  Real pos;
  #ifdef PARALLEL_OMP
  #pragma omp parallel for private( pos, changed_pos) num_threads( N_OMP_THREADS )
  #endif
  for( int i=0; i<Particles.n_local; i++){
    
    if ( dir == 0 ) pos = Particles.pos_x[i];
    if ( dir == 1 ) pos = Particles.pos_y[i];
    if ( dir == 2 ) pos = Particles.pos_z[i];
    
    changed_pos = false;
    if ( side == 0 ){
      if ( pos < d_min ) pos += L; //When the position is on the left of the domain boundary, add the domain Length to the position
      changed_pos = true;
    }
    if ( side == 1 ){
      if ( pos >= d_max ) pos -= L;//When the position is on the right of the domain boundary, substract the domain Length to the position
      changed_pos = true;
    }
    
    if ( !changed_pos ) continue;
    //If the position was changed write the new position to the vectors
    if ( dir == 0 ) Particles.pos_x[i] = pos;
    if ( dir == 1 ) Particles.pos_y[i] = pos;
    if ( dir == 2 ) Particles.pos_z[i] = pos;
    
  }
}

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

  Particles.Remove_Transfered_Particles();

}

//Remove the particles that were transfered outside the local domain
void Particles_3D::Remove_Transfered_Particles( void ){
  
  //Get the number of particles to delete
  part_int_t n_delete = 0;
  n_delete += out_indxs_vec_x0.size();
  n_delete += out_indxs_vec_x1.size();
  n_delete += out_indxs_vec_y0.size();
  n_delete += out_indxs_vec_y1.size();
  n_delete += out_indxs_vec_z0.size();
  n_delete += out_indxs_vec_z1.size();
  // std::cout << "N to delete: " << n_delete << std::endl;
  
  //Concatenate the indices of all the particles that moved into a new vector (delete_indxs_vec)
  int_vector_t delete_indxs_vec;
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_x0.begin(), out_indxs_vec_x0.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_x1.begin(), out_indxs_vec_x1.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_y0.begin(), out_indxs_vec_y0.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_y1.begin(), out_indxs_vec_y1.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_z0.begin(), out_indxs_vec_z0.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_z1.begin(), out_indxs_vec_z1.end() );
  
  //Clear the vectors that stored the transfered indices for each direction. All these indices are now stored in delete_indxs_vec
  out_indxs_vec_x0.clear();
  out_indxs_vec_x1.clear();
  out_indxs_vec_y0.clear();
  out_indxs_vec_y1.clear();
  out_indxs_vec_z0.clear();
  out_indxs_vec_z1.clear();
  
  //Sor the indices that need to be deleted so that the particles are deleted from right to left
  std::sort(delete_indxs_vec.begin(), delete_indxs_vec.end());
  
  part_int_t indx, pIndx;
  for ( indx=0; indx<n_delete; indx++ ){
    //From right to left get the index of the particle that will be deleted
    pIndx = delete_indxs_vec.back();
    //Remove the particle data at the selected index 
    Remove_Real( pIndx, pos_x );
    Remove_Real( pIndx, pos_y );
    Remove_Real( pIndx, pos_z );
    Remove_Real( pIndx, vel_x );
    Remove_Real( pIndx, vel_y );
    Remove_Real( pIndx, vel_z );
    Remove_Real( pIndx, grav_x );
    Remove_Real( pIndx, grav_y );
    Remove_Real( pIndx, grav_z );
    #ifdef PARTICLE_IDS
    Remove_ID( pIndx, partIDs );
    #endif
    #ifndef SINGLE_PARTICLE_MASS
    Remove_Real( pIndx, mass );
    #endif
    
    delete_indxs_vec.pop_back(); //Discard the index of ther delted particle from the delete_indxs_vector
    n_local -= 1; //substract one to the local number of particles
  }
  
  //At the end the delete_indxs_vec must be empty
  if ( delete_indxs_vec.size() != 0 ) std::cout << "ERROR: Deleting Transfered Particles " << std::endl;

  //Check that the size of the particles data vectors is consistent with the local number of particles
  int  n_in_vectors;
  n_in_vectors =  pos_x.size() + pos_y.size() + pos_z.size() + vel_x.size() + vel_y.size() + vel_z.size() ;
  #ifndef SINGLE_PARTICLE_MASS
  n_in_vectors += mass.size();
  #endif
  #ifdef PARTICLE_IDS
  n_in_vectors += partIDs.size();
  #endif
  
  if ( n_in_vectors != n_local * N_DATA_PER_PARTICLE_TRANSFER ){
    std::cout << "ERROR PARTICLES TRANSFER: DATA IN VECTORS DIFFERENT FROM N_LOCAL###########" << std::endl;
    exit(-1);
  }
  

  //Check all particles are in local domain
  // std::cout << " Finish particles transfer" << std::endl;
  // Real x_pos, y_pos, z_pos;
  // // part_int_t pIndx;
  // bool in_local;
  // for ( pIndx=0; pIndx < n_local; pIndx++ ){
  //   in_local = true;
  //   x_pos = pos_x[pIndx];
  //   y_pos = pos_y[pIndx];
  //   z_pos = pos_z[pIndx];
  //   if ( x_pos < G.xMin || x_pos >= G.xMax ) in_local = false;
  //   if ( y_pos < G.yMin || y_pos >= G.yMax ) in_local = false;
  //   if ( z_pos < G.zMin || z_pos >= G.zMax ) in_local = false;
  //   if ( ! in_local  ) {
  //     std::cout << " Particle Transfer Error:  indx: " << pIndx  << "  procID: " << procID << std::endl;
  //     #ifdef PARTICLE_IDS
  //     std::cout << " Particle outside Loacal  domain    pID: " << pID << std::endl;
  //     #else
  //     std::cout << " Particle outside Loacal  domain " << std::endl;
  //     #endif
  //     std::cout << "  Domain X: " << G.xMin <<  "  " << G.xMax << std::endl;
  //     std::cout << "  Domain Y: " << G.yMin <<  "  " << G.yMax << std::endl;
  //     std::cout << "  Domain Z: " << G.zMin <<  "  " << G.zMax << std::endl;
  //     std::cout << "  Particle X: " << x_pos << std::endl;
  //     std::cout << "  Particle Y: " << y_pos << std::endl;
  //     std::cout << "  Particle Z: " << z_pos << std::endl;
  //   }
  // }
  
    
  //Check that the vectors for the transfered indices are empty
  // int n_in_out_vectors;
  // n_in_out_vectors = out_indxs_vec_x0.size() + out_indxs_vec_x1.size() + out_indxs_vec_y0.size() + out_indxs_vec_y1.size() + out_indxs_vec_z0.size() + out_indxs_vec_z1.size();
  // if ( n_in_out_vectors != 0 ){
  //   std::cout << "#################ERROR PARTICLES TRANSFER: OUPTUT VECTORS NOT EMPTY, N_IN_VECTORS: " << n_in_out_vectors << std::endl;
  //   part_int_t pId;
  //   if ( out_indxs_vec_x0.size()>0){
  //     std::cout << " In x0" << std::endl;
  //     pId = out_indxs_vec_x0[0];
  //   }
  //   if ( out_indxs_vec_x1.size()>0){
  //     std::cout << " In x1" << std::endl;
  //     pId = out_indxs_vec_x1[0];
  //   }
  //   if ( out_indxs_vec_y0.size()>0){
  //     std::cout << " In y0" << std::endl;
  //     pId = out_indxs_vec_y0[0];
  //   }
  //   if ( out_indxs_vec_y1.size()>0){
  //     std::cout << " In y1" << std::endl;
  //     pId = out_indxs_vec_y1[0];
  //   }
  //   if ( out_indxs_vec_z0.size()>0){
  //     std::cout << " In z0" << std::endl;
  //     pId = out_indxs_vec_z0[0];
  //   }
  //   if ( out_indxs_vec_z1.size()>0){
  //     std::cout << " In z1" << std::endl;
  //     pId = out_indxs_vec_z1[0];
  //   }
  //   std::cout  << "pos_x: " << pos_x[pId] << " x: " << G.xMin << "  " << G.xMax << std::endl;
  //   std::cout  << "pos_y: " << pos_y[pId] << " y: " << G.yMin << "  " << G.yMax << std::endl;
  //   std::cout  << "pos_z: " << pos_z[pId] << " z: " << G.zMin << "  " << G.zMax << std::endl;
  //   exit(-1);
  // }
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
    Unload_Particles_From_Buffers_BLOCK(status.MPI_TAG);
  }
}

//Unload the particles after MPI tranfer for a single index ( axis and side )
void Grid3D::Unload_Particles_From_Buffers_BLOCK(int index){

  // Make sure not to unload when not transfering particles
  if ( Particles.TRANSFER_DENSITY_BOUNDARIES ) return;
  if ( H.TRANSFER_HYDRO_BOUNDARIES ) return;
  if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ) return;

  if( index == 0) Unload_Particles_from_Buffer_X0();
  if( index == 1) Unload_Particles_from_Buffer_X1();
  if( index == 2) Unload_Particles_from_Buffer_Y0();
  if( index == 3) Unload_Particles_from_Buffer_Y1();
  if( index == 4) Unload_Particles_from_Buffer_Z0();
  if( index == 5) Unload_Particles_from_Buffer_Z1();
}


//Wait for the Number of particles that will be transferd
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


//Load the Number of particles that will be recieved (Particles.n_recv) and make the MPI_Irecv request for that buffer size
void Grid3D::Load_N_Particles_Transfer(int index, int *ireq_particles_transfer){

  int buffer_length;
  // std::cout << "ireq: " << *ireq_particles_transfer << std::endl;
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


//Send and Recieve request for the number of particles that will be transfered, and then load and send the transfer particles
void Grid3D::Load_and_Send_Particles_X0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;

  MPI_Irecv(&Particles.n_recv_x0, 1, MPI_PART_INT, source[0], 0, world, &recv_request_n_particles[ireq_n_particles]);
  MPI_Isend(&Particles.n_send_x0, 1, MPI_PART_INT, dest[0],   1, world, &send_request_n_particles[0]);
  // if ( Particles.n_send_x0 > 0 )   if ( Particles.n_send_x0 > 0 ) std::cout << " Sent X0:  " << Particles.n_send_x0 <<  "  " << procID <<  "  to  "  <<  dest[0] <<  std::endl;
  buffer_length = Particles.n_send_x0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_x0_particles , &buffer_length_particles_x0_send, buffer_length );
  Particles.Load_Particles_to_Buffer( 0, 0, send_buffer_x0_particles,  buffer_length_particles_x0_send );
  MPI_Isend(send_buffer_x0_particles, buffer_length, MPI_CHREAL, dest[0],   1, world, &send_request_particles_transfer[ireq_particles_transfer]);
}

void Grid3D::Load_and_Send_Particles_X1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Irecv(&Particles.n_recv_x1, 1, MPI_PART_INT, source[1], 1, world, &recv_request_n_particles[ireq_n_particles]);
  MPI_Isend(&Particles.n_send_x1, 1, MPI_PART_INT, dest[1],   0, world, &send_request_n_particles[1]);
  // if ( Particles.n_send_x1 > 0 )  std::cout << " Sent X1: " << Particles.n_send_x1 << std::endl;
  buffer_length = Particles.n_send_x1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_x1_particles , &buffer_length_particles_x1_send, buffer_length );
  Particles.Load_Particles_to_Buffer( 0, 1, send_buffer_x1_particles,  buffer_length_particles_x1_send  );
  MPI_Isend(send_buffer_x1_particles, buffer_length, MPI_CHREAL, dest[1],   0, world, &send_request_particles_transfer[ireq_particles_transfer]);\
}

void Grid3D::Load_and_Send_Particles_Y0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Isend(&Particles.n_send_y0, 1, MPI_PART_INT, dest[2],   3, world, &send_request_n_particles[0]);
  MPI_Irecv(&Particles.n_recv_y0, 1, MPI_PART_INT, source[2], 2, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_y0 > 0 )   std::cout << " Sent Y0: " << Particles.n_send_y0 << std::endl;
  buffer_length = Particles.n_send_y0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_y0_particles , &buffer_length_particles_y0_send, buffer_length );
  Particles.Load_Particles_to_Buffer( 1, 0, send_buffer_y0_particles,  buffer_length_particles_y0_send  );
  MPI_Isend(send_buffer_y0_particles, buffer_length, MPI_CHREAL, dest[2],   3, world, &send_request_particles_transfer[ireq_particles_transfer]);
}
void Grid3D::Load_and_Send_Particles_Y1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Isend(&Particles.n_send_y1, 1, MPI_PART_INT, dest[3],   2, world, &send_request_n_particles[1]);
  MPI_Irecv(&Particles.n_recv_y1, 1, MPI_PART_INT, source[3], 3, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_y1 > 0 )  std::cout << " Sent Y1: " << Particles.n_send_y1 << std::endl;
  buffer_length = Particles.n_send_y1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_y1_particles , &buffer_length_particles_y1_send, buffer_length );
  Particles.Load_Particles_to_Buffer( 1, 1, send_buffer_y1_particles,  buffer_length_particles_y1_send  );
  MPI_Isend(send_buffer_y1_particles, buffer_length, MPI_CHREAL, dest[3],   2, world, &send_request_particles_transfer[ireq_particles_transfer]);
}
void Grid3D::Load_and_Send_Particles_Z0( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Isend(&Particles.n_send_z0, 1, MPI_PART_INT, dest[4],   5, world, &send_request_n_particles[0]);
  MPI_Irecv(&Particles.n_recv_z0, 1, MPI_PART_INT, source[4], 4, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_z0 > 0 )   std::cout << " Sent Z0: " << Particles.n_send_z0 << std::endl;
  buffer_length = Particles.n_send_z0 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_z0_particles , &buffer_length_particles_z0_send, buffer_length );
  Particles.Load_Particles_to_Buffer( 2, 0, send_buffer_z0_particles,  buffer_length_particles_z0_send  );
  MPI_Isend(send_buffer_z0_particles, buffer_length, MPI_CHREAL, dest[4],   5, world, &send_request_particles_transfer[ireq_particles_transfer]);
}
void Grid3D::Load_and_Send_Particles_Z1( int ireq_n_particles, int ireq_particles_transfer ){
  int buffer_length;
  MPI_Isend(&Particles.n_send_z1, 1, MPI_CHREAL, dest[5],   4, world, &send_request_n_particles[1]);
  MPI_Irecv(&Particles.n_recv_z1, 1, MPI_CHREAL, source[5], 5, world, &recv_request_n_particles[ireq_n_particles]);
  // if ( Particles.n_send_z1 > 0 )   std::cout << " Sent Z1: " << Particles.n_send_z1 << std::endl;
  buffer_length = Particles.n_send_z1 * N_DATA_PER_PARTICLE_TRANSFER;
  Check_and_Grow_Particles_Buffer( &send_buffer_z1_particles , &buffer_length_particles_z1_send, buffer_length );
  Particles.Load_Particles_to_Buffer( 2, 1, send_buffer_z1_particles,  buffer_length_particles_z1_send  );
  MPI_Isend(send_buffer_z1_particles, buffer_length, MPI_CHREAL, dest[5],   4, world, &send_request_particles_transfer[ireq_particles_transfer]);
}

//Unload the Transfered particles from the MPI_buffer
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


//Load the particles that need to be transfered to the MPI buffer
void Particles_3D::Load_Particles_to_Buffer( int direction, int side, Real *send_buffer, int buffer_length  ){

  part_int_t n_out;
  part_int_t n_send;
  int_vector_t *out_indxs_vec;
  part_int_t *n_in_buffer;

  //Depending on the direction and side select the vector with the particle indices for the transfer
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
  n_out = out_indxs_vec->size();  //Number of particles to be transfered
  offset = *n_in_buffer*N_DATA_PER_PARTICLE_TRANSFER; //Offset in the array to take in to account the particles that already reside in the buffer array

  part_int_t indx, pIndx;
  for ( indx=0; indx<n_out; indx++ ){
    pIndx = (*out_indxs_vec)[indx]; // Index of the particle that will be transferd
    //Copy the particle data to the buffer array in the following order ( position, velocity )
    send_buffer[ offset + 0 ] = pos_x[pIndx];
    send_buffer[ offset + 1 ] = pos_y[pIndx];
    send_buffer[ offset + 2 ] = pos_z[pIndx];
    send_buffer[ offset + 3 ] = vel_x[pIndx];
    send_buffer[ offset + 4 ] = vel_y[pIndx];
    send_buffer[ offset + 5 ] = vel_z[pIndx];
    
    offset_extra = offset + 5;
    
    #ifndef SINGLE_PARTICLE_MASS
    //Copy the particle mass to the buffer array in the following order ( position, velocity, mass )
    offset_extra += 1;
    send_buffer[ offset_extra ] = mass[pIndx];
    #endif
    
    #ifdef PARTICLE_IDS
    offset_extra += 1;
    //Copy the particle mass to the buffer array in the following order ( position, velocity, mass, ID )
    send_buffer[ offset_extra ] = (Real) partIDs[pIndx];
    #endif
    
    *n_in_buffer += 1; // add one to the number of particles in the transfer_buffer
    offset += N_DATA_PER_PARTICLE_TRANSFER;
    
    //Check that the offset doesnt exceede the bufer size
    if ( offset > buffer_length ) std::cout << "ERROR: Buffer length exceeded on particles transfer" << std::endl;
  }

  // if (out_indxs_vec->size() > 0 ) std::cout << "ERROR: Particle output vector not empty after transfer " << std::endl;
}

//Add the data of a single particle to a transfer buffer
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


//After a particle was transfered, add the transfered particle data to the vectors that contain the data of the local particles
void Particles_3D::Add_Particle_To_Vectors( Real pId, Real pMass,
                            Real pPos_x, Real pPos_y, Real pPos_z,
                            Real pVel_x, Real pVel_y, Real pVel_z ){
  
  // Make sure that the particle position is inside the local domain
  bool in_local = true;                            
  if ( pPos_x < G.xMin || pPos_x >= G.xMax ) in_local = false;
  if ( pPos_y < G.yMin || pPos_y >= G.yMax ) in_local = false;
  if ( pPos_z < G.zMin || pPos_z >= G.zMax ) in_local = false;
  if ( ! in_local  ) {
    std::cout << " Adding particle out of local domain to vectors Error:" << std::endl;
    #ifdef PARTICLE_IDS
    std::cout << " Particle outside Loacal  domain    pID: " << pID << std::endl;
    #else
    std::cout << " Particle outside Loacal  domain " << std::endl;
    #endif
    std::cout << "  Domain X: " << G.xMin <<  "  " << G.xMax << std::endl;
    std::cout << "  Domain Y: " << G.yMin <<  "  " << G.yMax << std::endl;
    std::cout << "  Domain Z: " << G.zMin <<  "  " << G.zMax << std::endl;
    std::cout << "  Particle X: " << pPos_x << std::endl;
    std::cout << "  Particle Y: " << pPos_y << std::endl;
    std::cout << "  Particle Z: " << pPos_z << std::endl;
  }
                              
  //Append the particle data to the local data vectors                           
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

  //Add one to the local number of particles
  n_local += 1;
}

//After the MPI transfer, unload the data buffers
void Particles_3D::Unload_Particles_from_Buffer( int direction, int side, Real *recv_buffer, part_int_t n_recv,
      Real *send_buffer_y0, Real *send_buffer_y1, Real *send_buffer_z0, Real *send_buffer_z1, int buffer_length_y0, int buffer_length_y1, int buffer_length_z0, int buffer_length_z1){

  //Loop over the data in the recv_buffer, get the data for each particle and append the particle data to the local vecors

  int offset_buff, offset_extra;
  part_int_t pId;
  Real pMass, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z;
  
  offset_buff = 0;
  part_int_t indx;
  for ( indx=0; indx<n_recv; indx++ ){
    //Get the data for each transfered particle
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
    
    //PERIODIC BOUNDARIES: for the X direction
    if ( pPos_x <  G.domainMin_x ) pPos_x += ( G.domainMax_x - G.domainMin_x );
    if ( pPos_x >= G.domainMax_x ) pPos_x -= ( G.domainMax_x - G.domainMin_x );
    
    //If the particle x_position is outside the local domain there was an error
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
    
    // If the y_position at the X_Tansfer (direction=0) is outside the local domain, then the particles is added to the buffer for the Y_Transfer 
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
    
    // If the z_position at the X_Tansfer or Y_Transfer is outside the local domain, then the particles is added to the buffer for the Z_Transfer 
    if (direction != 2 ){
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
    

    if (  direction == 1 ){
      //PERIODIC BOUNDARIES: for the Y direction
      if ( pPos_y <  G.domainMin_y ) pPos_y += ( G.domainMax_y - G.domainMin_y );
      if ( pPos_y >= G.domainMax_y ) pPos_y -= ( G.domainMax_y - G.domainMin_y );
    }
    
    
    //If the particle y_position is outside the local domain after the X-Transfer, there was an error
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


    if (  direction == 2 ){
      //PERIODIC BOUNDARIES: for the Z direction
      if ( pPos_z <  G.domainMin_z ) pPos_z += ( G.domainMax_z - G.domainMin_z );
      if ( pPos_z >= G.domainMax_z ) pPos_z -= ( G.domainMax_z - G.domainMin_z );
    }
    
    //If the particle z_position is outside the local domain after the X-Transfer and Y-Transfer, there was an error
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
    
    //If the particles doesnt have to be transfered to the y_directtion or z_direction, then add the particle date to the local vectors
    Add_Particle_To_Vectors( pId, pMass, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
  }
}

//Find the indices of the particles that moved outside the local domain and need to be transfered
void Particles_3D::Select_Particles_to_Transfer_All( int *flags ){


  //Loop over the local particles and add the indices of those that need to be transfered to the out_indxs_vectors
  part_int_t pIndx;
  for ( pIndx=0; pIndx<n_local; pIndx++ ){
    
    if ( pos_x[pIndx] < G.xMin && (flags[0]==5) ){
      out_indxs_vec_x0.push_back( pIndx );
      continue;
    }
    if ( pos_x[pIndx] >= G.xMax && (flags[1]==5) ){
      out_indxs_vec_x1.push_back( pIndx );
      continue;
    }
    if ( pos_y[pIndx] < G.yMin && (flags[2]==5) ){
      out_indxs_vec_y0.push_back( pIndx );
      continue;
    }
    if ( pos_y[pIndx] >= G.yMax && (flags[3]==5) ){
      out_indxs_vec_y1.push_back( pIndx );
      continue;
    }
    if ( pos_z[pIndx] < G.zMin && (flags[4]==5) ){
        out_indxs_vec_z0.push_back( pIndx );
      continue;
    }
    if ( pos_z[pIndx] >= G.zMax && (flags[5]==5) ){
        out_indxs_vec_z1.push_back( pIndx );
      continue;
    }
  }

  std::sort(out_indxs_vec_x0.begin(), out_indxs_vec_x0.end());
  std::sort(out_indxs_vec_x1.begin(), out_indxs_vec_x1.end());
  std::sort(out_indxs_vec_y0.begin(), out_indxs_vec_y0.end());
  std::sort(out_indxs_vec_y1.begin(), out_indxs_vec_y1.end());
  std::sort(out_indxs_vec_z0.begin(), out_indxs_vec_z0.end());
  std::sort(out_indxs_vec_z1.begin(), out_indxs_vec_z1.end());
  
  //Compute the number os particles that will be transfered in each dierection
  n_send_x0 += out_indxs_vec_x0.size();
  n_send_x1 += out_indxs_vec_x1.size();
  n_send_y0 += out_indxs_vec_y0.size();
  n_send_y1 += out_indxs_vec_y1.size();
  n_send_z0 += out_indxs_vec_z0.size();
  n_send_z1 += out_indxs_vec_z1.size();

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
  
  //Clear the particles indices that were transfered during the previour timestep
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
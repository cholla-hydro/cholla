#if defined(PARTICLES) && defined(PARTICLES_CPU)

#include <unistd.h>
#include <algorithm>
#include <iostream>
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../particles/particles_3D.h"

#ifdef MPI_CHOLLA
#include "../mpi/mpi_routines.h"
#endif


//Get and remove Real value at index on vector
Real Get_and_Remove_Real( part_int_t indx, real_vector_t &vec ){
  Real value = vec[indx]; 
  vec[indx] = vec.back(); //The item at the specified index is replaced by the last item in the vector
  vec.pop_back(); //The last item in the vector is discarded
  return value;
}

//Remove Real value at index on vector 
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

//Convert Real to Integer for transfering particles IDs on Real buffer arrays
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
    d_min = Particles.G.xMin;
    d_max = Particles.G.xMax;
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
    
    //If the position was changed write the new position to the vectors
    if ( !changed_pos ) continue;
    if ( dir == 0 ) Particles.pos_x[i] = pos;
    if ( dir == 1 ) Particles.pos_y[i] = pos;
    if ( dir == 2 ) Particles.pos_z[i] = pos;
    
  }
}


//Set open boundaries for particles when not using MPI
void Grid3D::Set_Particles_Open_Boundary( int dir, int side ){
  Real d_min, d_max, L;

  if ( dir == 0 ){
    d_min = Particles.G.xMin;
    d_max = Particles.G.xMax;
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

  Real pos;
  int_vector_t removed_indices;

  #ifdef PARALLEL_OMP
  #pragma omp parallel for private(pos) num_threads( N_OMP_THREADS )
  #endif
  for( int i=0; i<Particles.n_local; i++){

    if ( dir == 0 ) pos = Particles.pos_x[i];
    if ( dir == 1 ) pos = Particles.pos_y[i];
    if ( dir == 2 ) pos = Particles.pos_z[i];

    //If the position is out of the region, remove.
    if (( side == 0 && pos < d_min ) || ( side == 1 && pos > d_max)) removed_indices.push_back(i);
  }
  std::sort(removed_indices.begin(), removed_indices.end());

  part_int_t indx, pIndx;
  part_int_t n_delete = removed_indices.size();
  for ( indx=0; indx<n_delete; indx++ ){
    //From right to left get the index of the particle that will be deleted
    pIndx = removed_indices.back();
    //Remove the particle data at the selected index
    Remove_Real( pIndx, Particles.pos_x );
    Remove_Real( pIndx, Particles.pos_y );
    Remove_Real( pIndx, Particles.pos_z );
    Remove_Real( pIndx, Particles.vel_x );
    Remove_Real( pIndx, Particles.vel_y );
    Remove_Real( pIndx, Particles.vel_z );
    Remove_Real( pIndx, Particles.grav_x );
    Remove_Real( pIndx, Particles.grav_y );
    Remove_Real( pIndx, Particles.grav_z );
    #ifdef PARTICLE_IDS
    Remove_ID( pIndx, Particles.partIDs );
    #endif
    #ifndef SINGLE_PARTICLE_MASS
    Remove_Real( pIndx, Particles.mass );
    #endif
    #ifdef PARTICLE_AGE
    Remove_Real(pIndx, Particles.age);
    #endif
    Particles.n_local -= 1;

  }
}


#ifdef MPI_CHOLLA


//Find the particles that moved outside the local domain in order to transfer them.
//The indices of selected particles are added to the out_indx_vectors
void Particles_3D::Select_Particles_to_Transfer_All_CPU( int *flags ){

  part_int_t pIndx;
  for ( pIndx=0; pIndx<n_local; pIndx++ ){
    
    if ( pos_x[pIndx] < G.xMin && flags[0]==5 ){
      out_indxs_vec_x0.push_back( pIndx );
      continue;
    }
    if ( pos_x[pIndx] >= G.xMax && flags[1]==5 ){
      out_indxs_vec_x1.push_back( pIndx );
      continue;
    }
    if ( pos_y[pIndx] < G.yMin && flags[2]==5 ){
      out_indxs_vec_y0.push_back( pIndx );
      continue;
    }
    if ( pos_y[pIndx] >= G.yMax && flags[3]==5 ){
      out_indxs_vec_y1.push_back( pIndx );
      continue;
    }
    if ( pos_z[pIndx] < G.zMin && flags[4]==5 ){
        out_indxs_vec_z0.push_back( pIndx );
      continue;
    }
    if ( pos_z[pIndx] >= G.zMax && flags[5]==5 ){
        out_indxs_vec_z1.push_back( pIndx );
      continue;
    }
  }

  //Sort the transfer Indices (NOT NEEDED: All indices are sorted at the end of the transfer before removing transfered particles )
  // std::sort(out_indxs_vec_x0.begin(), out_indxs_vec_x0.end());
  // std::sort(out_indxs_vec_x1.begin(), out_indxs_vec_x1.end());
  // std::sort(out_indxs_vec_y0.begin(), out_indxs_vec_y0.end());
  // std::sort(out_indxs_vec_y1.begin(), out_indxs_vec_y1.end());
  // std::sort(out_indxs_vec_z0.begin(), out_indxs_vec_z0.end());
  // std::sort(out_indxs_vec_z1.begin(), out_indxs_vec_z1.end());
  
  //Add the size of the out_vectors to the number of particles that will be send in each direction
  n_send_x0 += out_indxs_vec_x0.size();
  n_send_x1 += out_indxs_vec_x1.size();
  n_send_y0 += out_indxs_vec_y0.size();
  n_send_y1 += out_indxs_vec_y1.size();
  n_send_z0 += out_indxs_vec_z0.size();
  n_send_z1 += out_indxs_vec_z1.size();

}


//Load the particles that need to be transfered to the MPI buffer
void Particles_3D::Load_Particles_to_Buffer_CPU( int direction, int side, Real *send_buffer, int buffer_length  ){

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
    //Copy the particle mass to the buffer array in the following order ( position, velocity, mass, ID )
    offset_extra += 1;
    send_buffer[ offset_extra ] = (Real) partIDs[pIndx];
    #endif
    #ifdef PARTICLE_AGE
    //Copy the particle age to the buffer array in the following order (position, velocity, mass, ID, age)
    offset_extra += 1;
    send_buffer[offset_extra] = age[pIndx];
    #endif

    *n_in_buffer += 1; // add one to the number of particles in the transfer_buffer
    offset += N_DATA_PER_PARTICLE_TRANSFER;
    //Check that the offset doesnt exceede the bufer size
    if ( offset > buffer_length ) std::cout << "ERROR: Buffer length exceeded on particles transfer" << std::endl;
  }
}


//Add the data of a single particle to a transfer buffer
void Particles_3D::Add_Particle_To_Buffer( Real *buffer, part_int_t n_in_buffer, int buffer_length, Real pId, Real pMass, Real pAge,
                            Real pPos_x, Real pPos_y, Real pPos_z, Real pVel_x, Real pVel_y, Real pVel_z){

  int offset, offset_extra;
  offset = n_in_buffer * N_DATA_PER_PARTICLE_TRANSFER;

  if (offset > buffer_length ) std::cout << "ERROR: Buffer length exceeded on particles transfer" << std::endl;
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
  #ifdef PARTICLE_AGE
  offset_extra += 1;
  buffer[offset_extra] = pAge;
  #endif
}


//After a particle was transfered, add the transfered particle data to the vectors that contain the data of the local particles
void Particles_3D::Add_Particle_To_Vectors( Real pId, Real pMass, Real pAge,
                            Real pPos_x, Real pPos_y, Real pPos_z,
                            Real pVel_x, Real pVel_y, Real pVel_z, int *flags ){
  
  // Make sure that the particle position is inside the local domain
  bool in_local = true;                            
  if ( pPos_x < G.xMin || pPos_x >= G.xMax ) in_local = false;
  if ( ( pPos_y < G.yMin && flags[2]==5 ) || ( pPos_y >= G.yMax && flags[3]==5 ) ) in_local = false;
  if ( ( pPos_z < G.zMin && flags[4]==5 ) || ( pPos_z >= G.zMax && flags[4]==5 ) ) in_local = false;
  if ( ! in_local  ) {
    std::cout << " Adding particle out of local domain to vectors Error:" << std::endl;
    #ifdef PARTICLE_IDS
    std::cout << " Particle outside Local  domain    pID: " << pId << std::endl;
    #else
    std::cout << " Particle outside Local  domain " << std::endl;
    #endif
    std::cout << "  Domain X: " << G.xMin <<  "  " << G.xMax << std::endl;
    std::cout << "  Domain Y: " << G.yMin <<  "  " << G.yMax << std::endl;
    std::cout << "  Domain Z: " << G.zMin <<  "  " << G.zMax << std::endl;
    std::cout << "  Particle X: " << pPos_x << std::endl;
    std::cout << "  Particle Y: " << pPos_y << std::endl;
    std::cout << "  Particle Z: " << pPos_z << std::endl;
  }
  //TODO: is it good enough to log the error (but then go ahead and add it to the vector)?
                              
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
  #ifdef PARTICLE_AGE
  age.push_back(pAge);
  #endif
  grav_x.push_back(0);
  grav_y.push_back(0);
  grav_z.push_back(0);

  //Add one to the local number of particles
  n_local += 1;
}



//After the MPI transfer, unload the particles data from the buffers
void Particles_3D::Unload_Particles_from_Buffer_CPU( int direction, int side, Real *recv_buffer, part_int_t n_recv,
      Real *send_buffer_y0, Real *send_buffer_y1, Real *send_buffer_z0, Real *send_buffer_z1, int buffer_length_y0, int buffer_length_y1, int buffer_length_z0, int buffer_length_z1, int *flags){

  //Loop over the data in the recv_buffer, get the data for each particle and append the particle data to the local vecors
  
  int offset_buff, offset_extra;
  part_int_t pId;
  Real pMass, pAge, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z;

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
    #ifdef PARTICLE_AGE
    offset_extra += 1;
    pAge = recv_buffer[offset_extra];
    #else 
    pAge = 0.0;
    #endif

    offset_buff += N_DATA_PER_PARTICLE_TRANSFER;
    
    //GLOBAL PERIODIC BOUNDARIES: for the X direction
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
      if ( pPos_y < G.yMin  && flags[2]==5  ){
        Add_Particle_To_Buffer( send_buffer_y0, n_in_buffer_y0, buffer_length_y0, pId, pMass, pAge, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
        n_send_y0 += 1;
        n_in_buffer_y0 += 1;
        continue;
      }
      if ( pPos_y >= G.yMax && flags[3]==5 ){
        Add_Particle_To_Buffer( send_buffer_y1, n_in_buffer_y1, buffer_length_y1, pId, pMass, pAge, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
        n_send_y1 += 1;
        n_in_buffer_y1 += 1;
        continue;
      }
    }

    //PERIODIC BOUNDARIES: for the Y direction
    if (  direction == 1 ){
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
    
    // If the z_position at the X_Tansfer or Y_Transfer is outside the local domain, then the particles is added to the buffer for the Z_Transfer 
    if (direction  !=2 ){
      if ( pPos_z < G.zMin && flags[4]==5 ){
        Add_Particle_To_Buffer( send_buffer_z0, n_in_buffer_z0, buffer_length_z0, pId, pMass, pAge, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
        n_send_z0 += 1;
        n_in_buffer_z0 += 1;
        continue;
      }
      if ( pPos_z >= G.zMax && flags[5]==5  ){
        Add_Particle_To_Buffer( send_buffer_z1, n_in_buffer_z1, buffer_length_z1, pId, pMass, pAge, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z );
        n_send_z1 += 1;
        n_in_buffer_z1 += 1;
        continue;
      }
    }
    
    //GLOBAL PERIODIC BOUNDARIES: for the Z direction
    if (  direction == 2 ){
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
    
    //If the particle doesnt have to be transfered to the y_directtion or z_direction, then add the particle date to the local vectors
    Add_Particle_To_Vectors( pId, pMass, pAge, pPos_x, pPos_y, pPos_z, pVel_x, pVel_y, pVel_z, flags );
  }
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
  
  //Sort the indices that need to be deleted so that the particles are deleted from right to left
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
    #ifdef PARTICLE_AGE
    Remove_Real(pIndx, age);
    #endif
    
    delete_indxs_vec.pop_back(); //Discard the index of ther delted particle from the delete_indxs_vector
    n_local -= 1; //substract one to the local number of particles
  }

  //At the end the delete_indxs_vec must be empty
  if ( delete_indxs_vec.size() != 0 ) std::cout << "ERROR: Deleting Transfered Particles " << std::endl;
  

  //Check that the size of the particles data vectors is consistent with the local number of particles
  int n_in_out_vectors, n_in_vectors;
  n_in_vectors =  pos_x.size() + pos_y.size() + pos_z.size() + vel_x.size() + vel_y.size() + vel_z.size() ;
  #ifndef SINGLE_PARTICLE_MASS
  n_in_vectors += mass.size();
  #endif
  #ifdef PARTICLE_IDS
  n_in_vectors += partIDs.size();
  #endif
  #ifdef PARTICLE_AGE
  n_in_vectors += age.size();
  #endif

  if ( n_in_vectors != n_local * N_DATA_PER_PARTICLE_TRANSFER ){
    std::cout << "ERROR PARTICLES TRANSFER: DATA IN VECTORS DIFFERENT FROM N_LOCAL###########" << std::endl;
    exit(-1);
  }
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

#if defined(PARTICLES) && defined(PARTICLES_CPU)

#include <unistd.h>
#include <algorithm>
#include <iostream>
#include "../grid3D.h"
#include "../io.h"
#include "particles_3D.h"

#ifdef MPI_CHOLLA
#include "../mpi_routines.h"
#endif



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

void Remove_ID( part_int_t indx, int_vector_t &vec ){
  vec[indx] = vec.back();
  vec.pop_back();
}

part_int_t Real_to_part_int( Real inVal ){
  part_int_t outVal = (part_int_t) inVal;
  if ( (inVal - outVal) > 0.1 ) outVal += 1;
  if ( fabs(outVal - inVal) > 0.5 ) outVal -= 1;
  return outVal;
}


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
      if ( pos < d_min ) pos += L;
      changed_pos = true;
    }
    if ( side == 1 ){
      if ( pos >= d_max ) pos -= L;
      changed_pos = true;
    }
    
    if ( !changed_pos ) continue;
    
    if ( dir == 0 ) Particles.pos_x[i] = pos;
    if ( dir == 1 ) Particles.pos_y[i] = pos;
    if ( dir == 2 ) Particles.pos_z[i] = pos;
    
  }
}

#ifdef MPI_CHOLLA



void Particles_3D::Select_Particles_to_Transfer_All_CPU( void ){


  
  Clear_Vectors_For_Transfers();

  part_int_t pIndx;
  for ( pIndx=0; pIndx<n_local; pIndx++ ){
    // if (procID == 0) std::cout << pIndx << std::endl;
    if ( pos_x[pIndx] < G.xMin ){
      out_indxs_vec_x0.push_back( pIndx );
      continue;
    }
    if ( pos_x[pIndx] >= G.xMax ){
      out_indxs_vec_x1.push_back( pIndx );
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
    if ( pos_z[pIndx] < G.zMin ){
        out_indxs_vec_z0.push_back( pIndx );
      continue;
    }
    if ( pos_z[pIndx] >= G.zMax ){
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
  
  n_send_x0 += out_indxs_vec_x0.size();
  n_send_x1 += out_indxs_vec_x1.size();
  n_send_y0 += out_indxs_vec_y0.size();
  n_send_y1 += out_indxs_vec_y1.size();
  n_send_z0 += out_indxs_vec_z0.size();
  n_send_z1 += out_indxs_vec_z1.size();

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



void Particles_3D::Load_Particles_to_Buffer_CPU( int direction, int side, Real *send_buffer, int buffer_length  ){

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
    // pIndx = out_indxs_vec->back();
    pIndx = (*out_indxs_vec)[indx];
    send_buffer[ offset + 0 ] = pos_x[pIndx];
    send_buffer[ offset + 1 ] = pos_y[pIndx];
    send_buffer[ offset + 2 ] = pos_z[pIndx];
    send_buffer[ offset + 3 ] = vel_x[pIndx];
    send_buffer[ offset + 4 ] = vel_y[pIndx];
    send_buffer[ offset + 5 ] = vel_z[pIndx];
    
    offset_extra = offset + 5;
    #ifndef SINGLE_PARTICLE_MASS
    offset_extra += 1;
    send_buffer[ offset_extra ] = mass[pIndx];
    #endif
    #ifdef PARTICLE_IDS
    offset_extra += 1;
    send_buffer[ offset_extra ] = (Real) partIDs[pIndx];
    #endif

    *n_in_buffer += 1;
    offset += N_DATA_PER_PARTICLE_TRANSFER;
    if ( offset > buffer_length ) std::cout << "ERROR: Buffer length exceeded on particles transfer" << std::endl;
  }
}



void Particles_3D::Unload_Particles_from_Buffer_CPU( int direction, int side, Real *recv_buffer, part_int_t n_recv,
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


void Particles_3D::Remove_Transfered_Particles( void ){
  
  
  part_int_t n_delete = 0;
  n_delete += out_indxs_vec_x0.size();
  n_delete += out_indxs_vec_x1.size();
  n_delete += out_indxs_vec_y0.size();
  n_delete += out_indxs_vec_y1.size();
  n_delete += out_indxs_vec_z0.size();
  n_delete += out_indxs_vec_z1.size();
  
  // std::cout << "N to delete: " << n_delete << std::endl;
  int_vector_t delete_indxs_vec;
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_x0.begin(), out_indxs_vec_x0.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_x1.begin(), out_indxs_vec_x1.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_y0.begin(), out_indxs_vec_y0.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_y1.begin(), out_indxs_vec_y1.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_z0.begin(), out_indxs_vec_z0.end() );
  delete_indxs_vec.insert( delete_indxs_vec.end(), out_indxs_vec_z1.begin(), out_indxs_vec_z1.end() );
  
  out_indxs_vec_x0.clear();
  out_indxs_vec_x1.clear();
  out_indxs_vec_y0.clear();
  out_indxs_vec_y1.clear();
  out_indxs_vec_z0.clear();
  out_indxs_vec_z1.clear();
  
  
  std::sort(delete_indxs_vec.begin(), delete_indxs_vec.end());
  
  part_int_t indx, pIndx;
  for ( indx=0; indx<n_delete; indx++ ){
    pIndx = delete_indxs_vec.back();
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
    delete_indxs_vec.pop_back();
    n_local -= 1;
  }
  if ( delete_indxs_vec.size() != 0 ) std::cout << "ERROR: Deleting Transfered Particles " << std::endl;

  

  //Check all particles are in local domain
  // std::cout << " Finish particles transfer" << std::endl;
  Real x_pos, y_pos, z_pos;
  // part_int_t pIndx;
  bool in_local;
  for ( pIndx=0; pIndx < n_local; pIndx++ ){
    in_local = true;
    x_pos = pos_x[pIndx];
    y_pos = pos_y[pIndx];
    z_pos = pos_z[pIndx];
    if ( x_pos < G.xMin || x_pos >= G.xMax ) in_local = false;
    if ( y_pos < G.yMin || y_pos >= G.yMax ) in_local = false;
    if ( z_pos < G.zMin || z_pos >= G.zMax ) in_local = false;
    if ( ! in_local  ) {
      std::cout << " Particle Transfer Error:  indx: " << pIndx  << "  procID: " << procID << std::endl;
      #ifdef PARTICLE_IDS
      std::cout << " Particle outside Loacal  domain    pID: " << pID << std::endl;
      #else
      std::cout << " Particle outside Loacal  domain " << std::endl;
      #endif
      std::cout << "  Domain X: " << G.xMin <<  "  " << G.xMax << std::endl;
      std::cout << "  Domain Y: " << G.yMin <<  "  " << G.yMax << std::endl;
      std::cout << "  Domain Z: " << G.zMin <<  "  " << G.zMax << std::endl;
      std::cout << "  Particle X: " << x_pos << std::endl;
      std::cout << "  Particle Y: " << y_pos << std::endl;
      std::cout << "  Particle Z: " << z_pos << std::endl;
    }
  }
  
  
  int n_in_out_vectors, n_in_vectors;
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

  n_in_out_vectors = out_indxs_vec_x0.size() + out_indxs_vec_x1.size() + out_indxs_vec_y0.size() + out_indxs_vec_y1.size() + out_indxs_vec_z0.size() + out_indxs_vec_z1.size();
  if ( n_in_out_vectors != 0 ){
    std::cout << "#################ERROR PARTICLES TRANSFER: OUPTUT VECTORS NOT EMPTY, N_IN_VECTORS: " << n_in_out_vectors << std::endl;
    part_int_t pId;
    if ( out_indxs_vec_x0.size()>0){
      std::cout << " In x0" << std::endl;
      pId = out_indxs_vec_x0[0];
    }
    if ( out_indxs_vec_x1.size()>0){
      std::cout << " In x1" << std::endl;
      pId = out_indxs_vec_x1[0];
    }
    if ( out_indxs_vec_y0.size()>0){
      std::cout << " In y0" << std::endl;
      pId = out_indxs_vec_y0[0];
    }
    if ( out_indxs_vec_y1.size()>0){
      std::cout << " In y1" << std::endl;
      pId = out_indxs_vec_y1[0];
    }
    if ( out_indxs_vec_z0.size()>0){
      std::cout << " In z0" << std::endl;
      pId = out_indxs_vec_z0[0];
    }
    if ( out_indxs_vec_z1.size()>0){
      std::cout << " In z1" << std::endl;
      pId = out_indxs_vec_z1[0];
    }
    std::cout  << "pos_x: " << pos_x[pId] << " x: " << G.xMin << "  " << G.xMax << std::endl;
    std::cout  << "pos_y: " << pos_y[pId] << " y: " << G.yMin << "  " << G.yMax << std::endl;
    std::cout  << "pos_z: " << pos_z[pId] << " z: " << G.zMin << "  " << G.zMax << std::endl;
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
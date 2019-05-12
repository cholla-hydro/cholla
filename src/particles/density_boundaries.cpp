#ifdef PARTICLES

#include "../io.h"
#include "../grid3D.h"
#include "particles_3D.h"
#include <iostream>



void Grid3D::Copy_Particles_Density_Boundaries( int direction, int side ){
  
  int i, j, k, indx_src, indx_dst;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = Particles.G.n_ghost_particles_grid;
  nx_g = Particles.G.nx_local + 2*nGHST;
  ny_g = Particles.G.ny_local + 2*nGHST;
  nz_g = Particles.G.nz_local + 2*nGHST;
  
  //Copy X boundaries
  if (direction == 0){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nGHST; i++ ){
          if ( side == 0 ){
            indx_src = (i) + (j)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (nx_g - 2*nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            indx_src = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (i+nGHST) + (j)*nx_g + (k)*nx_g*ny_g;
          }
          Particles.G.density[indx_dst] += Particles.G.density[indx_src] ;
        }
      }
    }
  }
  
  //Copy Y boundaries
  if (direction == 1){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<nGHST; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ){
            indx_src = (i) + (j)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (i) + (ny_g - 2*nGHST + j)*nx_g + (k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            indx_src = (i) + (ny_g - nGHST + j)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (i) + (j+nGHST)*nx_g + (k)*nx_g*ny_g;
          }
          Particles.G.density[indx_dst] += Particles.G.density[indx_src] ;
        }
      }
    }
  }
    
  //Copy Z boundaries
  if (direction == 2){
    for ( k=0; k<nGHST; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ){
            indx_src = (i) + (j)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (i) + (j)*nx_g + (nz_g - 2*nGHST + k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            indx_src = (i) + (j)*nx_g + (nz_g - nGHST + k)*nx_g*ny_g;
            indx_dst = (i) + (j)*nx_g + (k+nGHST)*nx_g*ny_g;
          }
          Particles.G.density[indx_dst] += Particles.G.density[indx_src] ;
        }
      }
    }  
  }
  
}



#ifdef MPI_CHOLLA
void Grid3D::Transfer_Particles_Density_Boundaries_MPI( struct parameters P ){
  
  Particles.TRANSFER_DENSITY_BOUNDARIES = true;
  Set_Boundary_Conditions(P);
  Particles.TRANSFER_DENSITY_BOUNDARIES = false;
  
}
#endif

void Grid3D::Transfer_Particles_Density_Boundaries( struct parameters P ){
  
  #ifdef MPI_CHOLLA
  Transfer_Particles_Density_Boundaries_MPI(P);
  #else
  chprintf( " Error Partcles not implemented for non MPI");
  exit(-1);
  #endif  
}

int Grid3D::Load_Particles_Density_Boundary_to_Buffer( int direction, int side, Real *buffer  ){

  int i, j, k, indx, indx_buff, buffer_length;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = Particles.G.n_ghost_particles_grid;
  nx_g = Particles.G.nx_local + 2*nGHST;
  ny_g = Particles.G.ny_local + 2*nGHST;
  nz_g = Particles.G.nz_local + 2*nGHST;

  //Load Z boundaries
  if (direction == 2){
    for ( k=0; k<nGHST; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ) indx = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (i) + (j)*nx_g + (nz_g - nGHST + k)*nx_g*ny_g;
          indx_buff = i + j*nx_g + k*nx_g*ny_g ;
          buffer[indx_buff] = Particles.G.density[indx];
        }
      }
    }
    buffer_length = nGHST * nx_g * ny_g;
  }

  //Load Y boundaries
  if (direction == 1){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<nGHST; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ) indx = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (i) + (ny_g - nGHST + j)*nx_g + (k)*nx_g*ny_g;
          indx_buff = i + k*nx_g + j*nx_g*nz_g ;
          buffer[indx_buff] = Particles.G.density[indx];
        }
      }
    }
    buffer_length = nGHST * nx_g * nz_g;
  }

  //Load X boundaries
  if (direction == 0){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nGHST; i++ ){
          if ( side == 0 ) indx = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
          indx_buff = j + k*ny_g + i*ny_g*nz_g ;
          buffer[indx_buff] = Particles.G.density[indx];
        }
      }
    }
    buffer_length = nGHST * ny_g * nz_g;
  }


  return buffer_length;
}

void Grid3D::Unload_Particles_Density_Boundary_From_Buffer( int direction, int side, Real *buffer  ){

  int i, j, k, indx, indx_buff, buffer_length;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = Particles.G.n_ghost_particles_grid;
  nx_g = Particles.G.nx_local + 2*nGHST;
  ny_g = Particles.G.ny_local + 2*nGHST;
  nz_g = Particles.G.nz_local + 2*nGHST;

  // //Unload Z boundaries
  if (direction == 2){
    for ( k=0; k<nGHST; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ) indx = (i) + (j)*nx_g + (k + nGHST )*nx_g*ny_g;
          if ( side == 1 ) indx = (i) + (j)*nx_g + (nz_g - 2*nGHST + k)*nx_g*ny_g;
          indx_buff = i + j*nx_g + k*nx_g*ny_g ;
          Particles.G.density[indx] += buffer[indx_buff];
        }
      }
    }
  }

  //Unload Y boundaries
  if (direction == 1){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<nGHST; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ) indx = (i) + (j + nGHST)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (i) + (ny_g - 2*nGHST + j)*nx_g + (k)*nx_g*ny_g;
          indx_buff = i + k*nx_g + j*nx_g*nz_g ;
          Particles.G.density[indx] += buffer[indx_buff];
        }
      }
    }
  }

  //Unload X boundaries
  if (direction == 0){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nGHST; i++ ){
          if ( side == 0 ) indx = (i+nGHST) + (j)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (nx_g - 2*nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
          indx_buff = j + k*ny_g + i*ny_g*nz_g ;
          Particles.G.density[indx] += buffer[indx_buff];
          // Particles.G.density[indx] += 1;
        }
      }
    }
  }
}




#endif
#ifdef GRAVITY
#ifdef MPI_CHOLLA


#include "../io.h"
#include "../grid3D.h"
#include "grav3D.h"


void Grid3D::Copy_Potential_Boundaries( int direction, int side ){
  
  int i, j, k, indx_src, indx_dst;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;
  nx_g = Grav.nx_local + 2*nGHST;
  ny_g = Grav.ny_local + 2*nGHST;
  nz_g = Grav.nz_local + 2*nGHST;
  
  //Copy X boundaries
  if (direction == 0){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nGHST; i++ ){
          if ( side == 0 ){
            indx_src = (i+nGHST) + (j)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            indx_src = (nx_g - 2*nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          }
          Grav.F.potential_h[indx_dst] = Grav.F.potential_h[indx_src] ;
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
            indx_src = (i) + (j+nGHST)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (i) + (ny_g - nGHST + j)*nx_g + (k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            indx_src = (i) + (ny_g - 2*nGHST + j)*nx_g + (k)*nx_g*ny_g;
            indx_dst = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          }
          Grav.F.potential_h[indx_dst] = Grav.F.potential_h[indx_src] ;
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
            indx_src = (i) + (j)*nx_g + (k+nGHST)*nx_g*ny_g;
            indx_dst = (i) + (j)*nx_g + (nz_g - nGHST + k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            indx_src = (i) + (j)*nx_g + (nz_g - 2*nGHST + k)*nx_g*ny_g;
            indx_dst = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          }
        Grav.F.potential_h[indx_dst] = Grav.F.potential_h[indx_src] ;
        }
      }
    }  
  }
  
}

int Grid3D::Load_Gravity_Potential_To_Buffer( int direction, int side, Real *buffer, int buffer_start  ){


  int i, j, k, indx, indx_buff, length;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;
  nx_g = Grav.nx_local + 2*nGHST;
  ny_g = Grav.ny_local + 2*nGHST;
  nz_g = Grav.nz_local + 2*nGHST;
  
  //Load X boundaries
  if (direction == 0){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nGHST; i++ ){
          if ( side == 0 ) indx = (i+nGHST) + (j)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (nx_g - 2*nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
          indx_buff = (j) + (k)*ny_g + i*ny_g*nz_g ;
          buffer[buffer_start+indx_buff] = Grav.F.potential_h[indx];
          length = nGHST * nz_g * ny_g;
        }
      }
    }
  }

  //Load Y boundaries
  if (direction == 1){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<nGHST; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ) indx = (i) + (j+nGHST)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (i) + (ny_g - 2*nGHST + j)*nx_g + (k)*nx_g*ny_g;
          indx_buff = (i) + (k)*nx_g + j*nx_g*nz_g ;
          buffer[buffer_start+indx_buff] = Grav.F.potential_h[indx];
          length = nGHST * nz_g * nx_g;
        }
      }
    }
  }

  //Load Z boundaries
  if (direction == 2){
    for ( k=0; k<nGHST; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ) indx = (i) + (j)*nx_g + (k+nGHST)*nx_g*ny_g;
          if ( side == 1 ) indx = (i) + (j)*nx_g + (nz_g - 2*nGHST + k)*nx_g*ny_g;
          indx_buff = (i) + (j)*nx_g + k*nx_g*ny_g ;
          buffer[buffer_start+indx_buff] = Grav.F.potential_h[indx];
          length = nGHST * nx_g * ny_g;
        }
      }
    }
  }
  return length;
}


void Grid3D::Unload_Gravity_Potential_from_Buffer( int direction, int side, Real *buffer, int buffer_start  ){


  int i, j, k, indx, indx_buff;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;
  nx_g = Grav.nx_local + 2*nGHST;
  ny_g = Grav.ny_local + 2*nGHST;
  nz_g = Grav.nz_local + 2*nGHST;

  //Load X boundaries
  if (direction == 0){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nGHST; i++ ){
          if ( side == 0 ) indx = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
          indx_buff = (j) + (k)*ny_g + i*ny_g*nz_g ;
          Grav.F.potential_h[indx] = buffer[buffer_start+indx_buff];
        }
      }
    }
  }

  //Load Y boundaries
  if (direction == 1){
    for ( k=0; k<nz_g; k++ ){
      for ( j=0; j<nGHST; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ) indx = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (i) + (ny_g - nGHST + j)*nx_g + (k)*nx_g*ny_g;
          indx_buff = (i) + (k)*nx_g + j*nx_g*nz_g ;
          Grav.F.potential_h[indx] = buffer[buffer_start+indx_buff];
        }
      }
    }
  }

  //Load Z boundaries
  if (direction == 2){
    for ( k=0; k<nGHST; k++ ){
      for ( j=0; j<ny_g; j++ ){
        for ( i=0; i<nx_g; i++ ){
          if ( side == 0 ) indx = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) indx = (i) + (j)*nx_g + (nz_g - nGHST + k)*nx_g*ny_g;
          indx_buff = (i) + (j)*nx_g + k*nx_g*ny_g ;
          Grav.F.potential_h[indx] = buffer[buffer_start+indx_buff];
        }
      }
    }
  }
}

#endif //GRAVITY
#endif //MPI_CHOLLA

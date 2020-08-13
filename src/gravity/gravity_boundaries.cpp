#ifdef GRAVITY


#include <cmath>
#include "../io.h"
#include "../grid3D.h"
#include "grav3D.h"

#if defined (GRAV_ISOLATED_BOUNDARY_X) || defined (GRAV_ISOLATED_BOUNDARY_Y) || defined(GRAV_ISOLATED_BOUNDARY_Z)

void Grid3D::Compute_Potential_Boundaries_Isolated( int dir ){

  // Set Isolated Boundaries for the ghost cells.
  int bc_potential_id = 0; //Point mass potential GM/r
  if ( dir == 0 ) Compute_Potential_Isolated_Boundary( 0, 0, bc_potential_id );
  if ( dir == 1 ) Compute_Potential_Isolated_Boundary( 0, 1, bc_potential_id );
  if ( dir == 2 ) Compute_Potential_Isolated_Boundary( 1, 0, bc_potential_id );
  if ( dir == 3 ) Compute_Potential_Isolated_Boundary( 1, 1, bc_potential_id );
  if ( dir == 4 ) Compute_Potential_Isolated_Boundary( 2, 0, bc_potential_id );
  if ( dir == 5 ) Compute_Potential_Isolated_Boundary( 2, 1, bc_potential_id );

}

void Grid3D::Set_Potential_Boundaries_Isolated( int direction, int side, int *flags ){
  
  Real *pot_boundary;
  int n_i, n_j, nGHST;
  int nx_g, ny_g, nz_g;
  int nx_local, ny_local, nz_local;
  nGHST = N_GHOST_POTENTIAL;
  nx_g = Grav.nx_local + 2*nGHST;
  ny_g = Grav.ny_local + 2*nGHST;
  nz_g = Grav.nz_local + 2*nGHST;  
  nx_local = Grav.nx_local;
  ny_local = Grav.ny_local;
  nz_local = Grav.nz_local;
  
  #ifdef GRAV_ISOLATED_BOUNDARY_X
  if ( direction == 0 ){
    n_i = Grav.ny_local;
    n_j = Grav.nz_local;
    if ( side == 0 ) pot_boundary = Grav.F.pot_boundary_x0;
    if ( side == 1 ) pot_boundary = Grav.F.pot_boundary_x1;
  }
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  if ( direction == 1 ){
    n_i = Grav.nx_local;
    n_j = Grav.nz_local;
    if ( side == 0 ) pot_boundary = Grav.F.pot_boundary_y0;
    if ( side == 1 ) pot_boundary = Grav.F.pot_boundary_y1;
  }
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  if ( direction == 2 ){
    n_i = Grav.nx_local;
    n_j = Grav.ny_local;
    if ( side == 0 ) pot_boundary = Grav.F.pot_boundary_z0;
    if ( side == 1 ) pot_boundary = Grav.F.pot_boundary_z1;
  }
  #endif  
  
  int i, j, k, id_buffer, id_grid;
  
  for ( k=0; k<nGHST; k++ ){
    for ( i=0; i<n_i; i++ ){
      for ( j=0; j<n_j; j++ ){
        
        id_buffer = i + j*n_i + k*n_i*n_j; 
        
        if ( direction == 0 ){
          if ( side == 0 ) id_grid = (k)                + (i+nGHST)*nx_g + (j+nGHST)*nx_g*ny_g;
          if ( side == 1 ) id_grid = (k+nx_local+nGHST) + (i+nGHST)*nx_g + (j+nGHST)*nx_g*ny_g; 
        }
        if ( direction == 1 ){
          if ( side == 0 ) id_grid = (i+nGHST) + (k)*nx_g                + (j+nGHST)*nx_g*ny_g;
          if ( side == 1 ) id_grid = (i+nGHST) + (k+ny_local+nGHST)*nx_g + (j+nGHST)*nx_g*ny_g; 
        }
        if ( direction == 1 ){
          if ( side == 0 ) id_grid = (i+nGHST) + (j+nGHST)*nx_g + (k)*nx_g*ny_g;
          if ( side == 1 ) id_grid = (i+nGHST) + (j+nGHST)*nx_g + (k+nz_local+nGHST)*nx_g*ny_g; 
        }
        
        Grav.F.potential_h[id_grid] = pot_boundary[id_buffer];
      }
    }
  } 
  
}


void Grid3D::Compute_Potential_Isolated_Boundary( int direction, int side,  int bc_potential_id ){
  
  Real domain_l, Lx_local, Ly_local, Lz_local;
  Real *pot_boundary;
  int n_i, n_j, nGHST;
  nGHST = N_GHOST_POTENTIAL;
  
  Lx_local = Grav.nx_local * Grav.dx;
  Ly_local = Grav.ny_local * Grav.dy;
  Lz_local = Grav.nz_local * Grav.dz;
  
  
  
  #ifdef GRAV_ISOLATED_BOUNDARY_X
  if ( direction == 0 ){
    domain_l = Grav.xMin;
    n_i = Grav.ny_local;
    n_j = Grav.nz_local;
    if ( side == 0 ) pot_boundary = Grav.F.pot_boundary_x0;
    if ( side == 1 ) pot_boundary = Grav.F.pot_boundary_x1;
  }
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  if ( direction == 1 ){
    domain_l = Grav.yMin;
    n_i = Grav.nx_local;
    n_j = Grav.nz_local;
    if ( side == 0 ) pot_boundary = Grav.F.pot_boundary_y0;
    if ( side == 1 ) pot_boundary = Grav.F.pot_boundary_y1;
  }
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  if ( direction == 2 ){
    domain_l = Grav.zMin;
    n_i = Grav.nx_local;
    n_j = Grav.ny_local;
    if ( side == 0 ) pot_boundary = Grav.F.pot_boundary_z0;
    if ( side == 1 ) pot_boundary = Grav.F.pot_boundary_z1;
  }
  #endif  
  
  Real M, cm_pos_x, cm_pos_y, cm_pos_z, pos_x, pos_y, pos_z, r, delta_x, delta_y, delta_z;
  
  if ( bc_potential_id == 0 ){
    M = 0.1005;
    cm_pos_x = 0.5;
    cm_pos_y = 0.5;
    cm_pos_z = 0.5; 
  }
  
  
  Real pot_val;
  int i, j, k, id;
  for ( k=0; k<nGHST; k++ ){
    for ( i=0; i<n_i; i++ ){
      for ( j=0; j<n_j; j++ ){
        
        id = i + j*n_i + k*n_i*n_j; 
        
        if ( direction == 0 ){
          // pos_x = Grav.xMin - ( nGHST + k + 0.5 ) * Grav.dx;
          pos_x = Grav.xMin + ( k + 0.5 - nGHST ) * Grav.dx;
          if ( side == 1 ) pos_x += Lx_local + nGHST*Grav.dx;
          pos_y = Grav.yMin + ( i + 0.5 )* Grav.dy;
          pos_z = Grav.zMin + ( j + 0.5 )* Grav.dz;
        }
        
        if ( direction == 1 ){
          // pos_y = Grav.yMin - ( nGHST + k + 0.5 ) * Grav.dy;
          pos_y = Grav.yMin + ( k + 0.5 - nGHST ) * Grav.dy;
          if ( side == 1 ) pos_y += Ly_local + nGHST*Grav.dy;
          pos_x = Grav.xMin + ( i + 0.5 )* Grav.dx;
          pos_z = Grav.zMin + ( j + 0.5 )* Grav.dz;
        } 
          
        if ( direction == 2 ){
          // pos_z = Grav.zMin - ( nGHST + k + 0.5 ) * Grav.dz;
          pos_z = Grav.zMin + ( k + 0.5 - nGHST ) * Grav.dz;
          if ( side == 1 ) pos_z += Lz_local + nGHST*Grav.dz;
          pos_x = Grav.xMin + ( i + 0.5 )* Grav.dx;
          pos_y = Grav.yMin + ( j + 0.5 )* Grav.dy;
        }
         
        if ( bc_potential_id == 0){ 
          //Point mass potential GM/r
          delta_x = pos_x - cm_pos_x;
          delta_y = pos_y - cm_pos_y;
          delta_z = pos_z - cm_pos_z;
          r = sqrt( ( delta_x * delta_x ) + ( delta_y * delta_y ) + ( delta_z * delta_z ) );
          pot_val = - Grav.Gconst * M / r;
        }
        else{
          chprintf("ERROR: Boundaty Potential not set, need to set appropriate bc_potential_id \n"); 
        }
        
        pot_boundary[id] = pot_val;
                        
      }
    }
  }    
    
}


#endif //GRAV_ISOLATED_BOUNDARY_X

void Grid3D::Copy_Potential_Boundaries( int direction, int side, int *flags ){
  // Flags: 1 (periodic), 2 (reflective), 3 (transmissive), 4 (custom), 5 (mpi)
  
  int boundary_flag;
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
            boundary_flag = flags[0];
            if ( boundary_flag == 1 ) indx_src = (nx_g - 2*nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g; //Periodic
            // if ( boundary_flag == 3 ) indx_src = (2*nGHST-i) + (j)*nx_g + (k)*nx_g*ny_g; // Zero Gradient
            indx_dst = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            boundary_flag = flags[1];
            if ( boundary_flag == 1 ) indx_src = (i+nGHST) + (j)*nx_g + (k)*nx_g*ny_g;   //Periodic
            // if ( boundary_flag == 3 ) indx_src = (nx_g - nGHST - 2 -i ) + (j)*nx_g + (k)*nx_g*ny_g; //Zero Gradient
            indx_dst = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g*ny_g;
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
            boundary_flag = flags[2];
            if ( boundary_flag == 1 ) indx_src = (i) + (ny_g - 2*nGHST + j)*nx_g + (k)*nx_g*ny_g; //Periodic
            // if ( boundary_flag == 3 ) indx_src = (i) + (2*nGHST - j)*nx_g + (k)*nx_g*ny_g;  //Zero Gradient
            indx_dst = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            boundary_flag = flags[3];
            if ( boundary_flag == 1 ) indx_src = (i) + (j+nGHST)*nx_g + (k)*nx_g*ny_g; //Periodic
            // if ( boundary_flag == 3 ) indx_src = (i) + (ny_g - nGHST - 2 - j)*nx_g + (k)*nx_g*ny_g; //Zero Gradient
            indx_dst = (i) + (ny_g - nGHST + j)*nx_g + (k)*nx_g*ny_g;
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
            boundary_flag = flags[4]; 
            if ( boundary_flag == 1 ) indx_src = (i) + (j)*nx_g + (nz_g - 2*nGHST + k)*nx_g*ny_g;  //Periodic
            // if ( boundary_flag == 3 ) indx_src = (i) + (j)*nx_g + (2*nGHST - k)*nx_g*ny_g; //Zero Gradient
            indx_dst = (i) + (j)*nx_g + (k)*nx_g*ny_g;
          }
          if ( side == 1 ){
            boundary_flag = flags[5];
            if ( boundary_flag == 1 ) indx_src = (i) + (j)*nx_g + (k+nGHST)*nx_g*ny_g; //Periodic
            // if ( boundary_flag == 3 ) indx_src = (i) + (j)*nx_g + (nz_g - nGHST - 2 - k)*nx_g*ny_g; //Zero Gradient
            indx_dst = (i) + (j)*nx_g + (nz_g - nGHST + k)*nx_g*ny_g;
          }
        Grav.F.potential_h[indx_dst] = Grav.F.potential_h[indx_src] ;
        }
      }
    }  
  }
  
}

#ifdef MPI_CHOLLA
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

#if defined(GRAVITY) && defined(SOR)

#include "potential_SOR_3D.h"
#include "../io.h"
#include<iostream>
#include <cmath>
#include "../grid3D.h"

#ifdef MPI_CHOLLA
#include "../mpi_routines.h"
#endif


Potential_SOR_3D::Potential_SOR_3D( void ){}

void Potential_SOR_3D::Initialize( Real Lx, Real Ly, Real Lz, Real x_min, Real y_min, Real z_min, int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, Real dx_real, Real dy_real, Real dz_real){
 
  Lbox_x = Lx;
  Lbox_y = Ly;
  Lbox_z = Lz;

  nx_total = nx;
  ny_total = ny;
  nz_total = nz;

  nx_local = nx_real;
  ny_local = ny_real;
  nz_local = nz_real;

  dx = dx_real;
  dy = dy_real;
  dz = dz_real;

  n_ghost = N_GHOST_POTENTIAL;

  nx_pot = nx_local + 2*n_ghost;
  ny_pot = ny_local + 2*n_ghost;
  nz_pot = nz_local + 2*n_ghost;
  
  n_cells_local = nx_local*ny_local*nz_local;
  n_cells_potential = nx_pot*ny_pot*nz_pot;
  n_cells_total = nx_total*ny_total*nz_total;
  
  n_ghost_transfer = 1;
  size_buffer_x = n_ghost_transfer * ny_pot * nz_pot;
  size_buffer_y = n_ghost_transfer * nx_pot * nz_pot;
  size_buffer_z = n_ghost_transfer * nx_pot * ny_pot;
  
  TRANSFER_POISSON_BOUNDARIES = false;
  
  chprintf( " Using Poisson Solver: SOR\n");
  chprintf( "  SOR: L[ %f %f %f ] N[ %d %d %d ] dx[ %f %f %f ]\n", Lbox_x, Lbox_y, Lbox_z, nx_local, ny_local, nz_local, dx, dy, dz );

  chprintf( "  SOR: Allocating memory...\n");
  AllocateMemory_CPU();
  AllocateMemory_GPU();
  
  potential_initialized = false;
  
}


void Potential_SOR_3D::AllocateMemory_CPU( void ){
  F.output_h = (Real *) malloc(n_cells_local*sizeof(Real));
  F.converged_h = (bool *) malloc(sizeof(bool));
}


void Potential_SOR_3D::AllocateMemory_GPU( void ){
  
  Allocate_Array_GPU_Real( &F.input_d, n_cells_local );
  Allocate_Array_GPU_Real( &F.density_d, n_cells_local );
  Allocate_Array_GPU_Real( &F.potential_d, n_cells_potential );
  Allocate_Array_GPU_bool( &F.converged_d, 1 );
  Allocate_Array_GPU_Real( &F.boundaries_buffer_x0_d, size_buffer_x);
  Allocate_Array_GPU_Real( &F.boundaries_buffer_x1_d, size_buffer_x);
  Allocate_Array_GPU_Real( &F.boundaries_buffer_y0_d, size_buffer_y);
  Allocate_Array_GPU_Real( &F.boundaries_buffer_y1_d, size_buffer_y);
  Allocate_Array_GPU_Real( &F.boundaries_buffer_z0_d, size_buffer_z);
  Allocate_Array_GPU_Real( &F.boundaries_buffer_z1_d, size_buffer_z);
  
  #ifdef MPI_CHOLLA
  Allocate_Array_GPU_Real( &F.recv_boundaries_buffer_x0_d, size_buffer_x);
  Allocate_Array_GPU_Real( &F.recv_boundaries_buffer_x1_d, size_buffer_x);
  Allocate_Array_GPU_Real( &F.recv_boundaries_buffer_y0_d, size_buffer_y);
  Allocate_Array_GPU_Real( &F.recv_boundaries_buffer_y1_d, size_buffer_y);
  Allocate_Array_GPU_Real( &F.recv_boundaries_buffer_z0_d, size_buffer_z);
  Allocate_Array_GPU_Real( &F.recv_boundaries_buffer_z1_d, size_buffer_z);
  #endif
  
  #ifdef GRAV_ISOLATED_BOUNDARY_X
  Allocate_Array_GPU_Real( &F.boundary_isolated_x0_d, n_ghost*ny_local*nz_local );
  Allocate_Array_GPU_Real( &F.boundary_isolated_x1_d, n_ghost*ny_local*nz_local );
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_X
  Allocate_Array_GPU_Real( &F.boundary_isolated_y0_d, n_ghost*nx_local*nz_local );
  Allocate_Array_GPU_Real( &F.boundary_isolated_y1_d, n_ghost*nx_local*nz_local );
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  Allocate_Array_GPU_Real( &F.boundary_isolated_z0_d, n_ghost*nx_local*ny_local );
  Allocate_Array_GPU_Real( &F.boundary_isolated_z1_d, n_ghost*nx_local*ny_local );
  #endif
  
}

void Potential_SOR_3D::Copy_Input_And_Initialize( Real *input_density, Real Grav_Constant, Real dens_avrg, Real current_a ){
  Copy_Input( n_cells_local, F.input_d, input_density, Grav_Constant, dens_avrg, current_a );

  if ( !potential_initialized ){
    chprintf( "SOR: Initializing  Potential \n");
    Initialize_Potential( nx_local, ny_local, nz_local, n_ghost, F.potential_d, F.density_d );
    potential_initialized = true;
  }  
}


void Potential_SOR_3D::Poisson_Partial_Iteration( int n_step, Real omega, Real epsilon ){
  if (n_step == 0 ) Poisson_iteration_Patial_1( n_cells_local, nx_local, ny_local, nz_local, n_ghost, dx, dy, dz, omega, epsilon, F.density_d, F.potential_d, F.converged_h, F.converged_d );
  if (n_step == 1 ) Poisson_iteration_Patial_2( n_cells_local, nx_local, ny_local, nz_local, n_ghost, dx, dy, dz, omega, epsilon, F.density_d, F.potential_d, F.converged_h, F.converged_d );
}


void Grid3D::Get_Potential_SOR( Real Grav_Constant, Real dens_avrg, Real current_a, struct parameters *P ){
  
  
  
  Grav.Poisson_solver.Copy_Input_And_Initialize( Grav.F.density_h, Grav_Constant, dens_avrg, current_a );

  //Set Isolated Boundary Conditions
  Grav.Copy_Isolated_Boundaries_To_GPU( P );
  Grav.Poisson_solver.Set_Isolated_Boundary_Conditions( P );


  Real epsilon = 1e-4;
  int max_iter = 10000000;
  int n_iter = 0;

  Grav.Poisson_solver.F.converged_h[0] = 0;

  // For Diriclet Boudaries
  Real omega = 2. / ( 1 + M_PI / Grav.Poisson_solver.nx_total  );

  // For Periodic Boudaries
  // Real omega = 2. / ( 1 + 2*M_PI / nx_total  );
  // chprintf("Omega: %f \n", omega);
  
  bool set_boundaries;
  int n_iter_per_boundaries_transfer = 1;
  
  
  // Iterate to solve Poisson equation
  while ( Grav.Poisson_solver.F.converged_h[0] == 0 ) {

    set_boundaries = false;
    
    if ( n_iter % n_iter_per_boundaries_transfer == 0 ) set_boundaries = true;
     
    if ( set_boundaries ){
      // Grav.Poisson_solver.Load_Transfer_Buffer_GPU_All();
      // Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_All();   
      // 
      Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES = true;
      Set_Boundary_Conditions( *P );
      Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES = false;
    }
    
    Grav.Poisson_solver.Poisson_Partial_Iteration( 0, omega, epsilon ); 

    if ( set_boundaries ){
      // Grav.Poisson_solver.Load_Transfer_Buffer_GPU_All();
      // Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_All();    
    
      Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES = true;
      Set_Boundary_Conditions( *P );
      Grav.Poisson_solver.TRANSFER_POISSON_BOUNDARIES = false;
    }
    
    Grav.Poisson_solver.Poisson_Partial_Iteration( 1, omega, epsilon );

    n_iter += 1;

    #ifdef MPI_CHOLLA
    Grav.Poisson_solver.F.converged_h[0] = Grav.Poisson_solver.Get_Global_Converged( Grav.Poisson_solver.F.converged_h[0] );
    #endif

    if ( n_iter == max_iter ) break;
  }

  if ( n_iter == max_iter ) chprintf(" SOR: No convergence in %d iterations \n", n_iter);
  else chprintf(" SOR: Converged in %d iterations \n", n_iter);

  Grav.Poisson_solver.Copy_Output( Grav.F.potential_h );

    
}

void Grav3D::Copy_Isolated_Boundaries_To_GPU( struct parameters *P ){
  
  if ( P->xl_bcnd != 3 && P->xu_bcnd != 3 && P->yl_bcnd != 3 && P->yu_bcnd != 3 && P->zl_bcnd != 3 && P->zu_bcnd != 3 ) return;
  
  chprintf( " Copying Isolated Boundaries \n");
  if ( P->xl_bcnd == 3 ) Copy_Isolated_Boundary_To_GPU_buffer( F.pot_boundary_x0, Poisson_solver.F.boundary_isolated_x0_d,  Poisson_solver.n_ghost*ny_local*nz_local );
  if ( P->xu_bcnd == 3 ) Copy_Isolated_Boundary_To_GPU_buffer( F.pot_boundary_x1, Poisson_solver.F.boundary_isolated_x1_d,  Poisson_solver.n_ghost*ny_local*nz_local ); 
  if ( P->yl_bcnd == 3 ) Copy_Isolated_Boundary_To_GPU_buffer( F.pot_boundary_y0, Poisson_solver.F.boundary_isolated_y0_d,  Poisson_solver.n_ghost*nx_local*nz_local );
  if ( P->yu_bcnd == 3 ) Copy_Isolated_Boundary_To_GPU_buffer( F.pot_boundary_y1, Poisson_solver.F.boundary_isolated_y1_d,  Poisson_solver.n_ghost*nx_local*nz_local );
  if ( P->zl_bcnd == 3 ) Copy_Isolated_Boundary_To_GPU_buffer( F.pot_boundary_z0, Poisson_solver.F.boundary_isolated_z0_d,  Poisson_solver.n_ghost*nx_local*ny_local );
  if ( P->zu_bcnd == 3 ) Copy_Isolated_Boundary_To_GPU_buffer( F.pot_boundary_z1, Poisson_solver.F.boundary_isolated_z1_d,  Poisson_solver.n_ghost*nx_local*ny_local ); 
  
  
}

void Potential_SOR_3D::Set_Isolated_Boundary_Conditions( struct parameters *P ){
  
  
  if ( P->xl_bcnd != 3 && P->xu_bcnd != 3 && P->yl_bcnd != 3 && P->yu_bcnd != 3 && P->zl_bcnd != 3 && P->zu_bcnd != 3 ) return;
  
  chprintf( " Setting Isolated Boundaries \n");
  if ( P->xl_bcnd == 3 ) Set_Isolated_Boundary_GPU( 0, 0,  F.boundary_isolated_x0_d );
  if ( P->xu_bcnd == 3 ) Set_Isolated_Boundary_GPU( 0, 1,  F.boundary_isolated_x1_d );
  if ( P->yl_bcnd == 3 ) Set_Isolated_Boundary_GPU( 1, 0,  F.boundary_isolated_y0_d );
  if ( P->yu_bcnd == 3 ) Set_Isolated_Boundary_GPU( 1, 1,  F.boundary_isolated_y1_d );
  if ( P->zl_bcnd == 3 ) Set_Isolated_Boundary_GPU( 2, 0,  F.boundary_isolated_z0_d );
  if ( P->zu_bcnd == 3 ) Set_Isolated_Boundary_GPU( 2, 1,  F.boundary_isolated_z1_d );
  
}




void Potential_SOR_3D::Copy_Poisson_Boundary_Periodic( int direction, int side ){
  
  Real *boundaries_buffer;
  
  if( direction == 0 ){
    if ( side == 0 ) boundaries_buffer = F.boundaries_buffer_x0_d;
    if ( side == 1 ) boundaries_buffer = F.boundaries_buffer_x1_d;
  }
  if( direction == 1 ){
    if ( side == 0 ) boundaries_buffer = F.boundaries_buffer_y0_d;
    if ( side == 1 ) boundaries_buffer = F.boundaries_buffer_y1_d;
  }
  if( direction == 2 ){
    if ( side == 0 ) boundaries_buffer = F.boundaries_buffer_z0_d;
    if ( side == 1 ) boundaries_buffer = F.boundaries_buffer_z1_d;
  }
  
  int side_load, side_unload;
  side_load = side;
  side_unload = ( side_load + 1 ) % 2;
  
  Load_Transfer_Buffer_GPU( direction, side_load, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, boundaries_buffer  );
  Unload_Transfer_Buffer_GPU( direction, side_unload, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, boundaries_buffer  );
  
}


void Potential_SOR_3D::FreeMemory_GPU( void ){
  
  Free_Array_GPU_Real( F.input_d );
  Free_Array_GPU_Real( F.density_d );
  Free_Array_GPU_Real( F.potential_d );
  Free_Array_GPU_Real( F.boundaries_buffer_x0_d );
  Free_Array_GPU_Real( F.boundaries_buffer_x1_d );
  Free_Array_GPU_Real( F.boundaries_buffer_y0_d );
  Free_Array_GPU_Real( F.boundaries_buffer_y1_d );
  Free_Array_GPU_Real( F.boundaries_buffer_z0_d );
  Free_Array_GPU_Real( F.boundaries_buffer_z1_d );
  
  #ifdef MPI_CHOLLA
  Free_Array_GPU_Real( F.recv_boundaries_buffer_x0_d );
  Free_Array_GPU_Real( F.recv_boundaries_buffer_x1_d );
  Free_Array_GPU_Real( F.recv_boundaries_buffer_y0_d );
  Free_Array_GPU_Real( F.recv_boundaries_buffer_y1_d );
  Free_Array_GPU_Real( F.recv_boundaries_buffer_z0_d );
  Free_Array_GPU_Real( F.recv_boundaries_buffer_z1_d );
  #endif
  
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  Free_Array_GPU_Real( F.boundary_isolated_x0_d );
  Free_Array_GPU_Real( F.boundary_isolated_x1_d );
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
  Free_Array_GPU_Real( F.boundary_isolated_y0_d );
  Free_Array_GPU_Real( F.boundary_isolated_y1_d );
  #endif
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
  Free_Array_GPU_Real( F.boundary_isolated_z0_d );
  Free_Array_GPU_Real( F.boundary_isolated_z1_d );
  #endif

}


void Potential_SOR_3D::Reset( void ){
  free( F.output_h );
  FreeMemory_GPU();
}

#ifdef MPI_CHOLLA

int Grid3D::Load_Poisson_Boundary_To_Buffer( int direction, int side, Real *buffer_host  ){

  int size_buffer;

  if ( direction == 0 ) size_buffer = Grav.Poisson_solver.size_buffer_x;
  if ( direction == 1 ) size_buffer = Grav.Poisson_solver.size_buffer_y;
  if ( direction == 2 ) size_buffer = Grav.Poisson_solver.size_buffer_z;
  
  
  //Load the transfer buffer in the GPU
  if ( direction == 0 ){
    if ( side == 0 ) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_x0();
    if ( side == 1 ) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_x1();
  }
  if ( direction == 1 ){
    if ( side == 0 ) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_y0();
    if ( side == 1 ) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_y1();
  }
  if ( direction == 2 ){
    if ( side == 0 ) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_z0();
    if ( side == 1 ) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_z1();
  }
  
  // Copy the device_buffer to the host_buffer
  Real *buffer_dev;
  if ( direction == 0 ){
    if ( side == 0 ) buffer_dev = Grav.Poisson_solver.F.boundaries_buffer_x0_d;
    if ( side == 1 ) buffer_dev = Grav.Poisson_solver.F.boundaries_buffer_x1_d; 
  }
  if ( direction == 1 ){
    if ( side == 0 ) buffer_dev = Grav.Poisson_solver.F.boundaries_buffer_y0_d;
    if ( side == 1 ) buffer_dev = Grav.Poisson_solver.F.boundaries_buffer_y1_d; 
  }
  if ( direction == 2 ){
    if ( side == 0 ) buffer_dev = Grav.Poisson_solver.F.boundaries_buffer_z0_d;
    if ( side == 1 ) buffer_dev = Grav.Poisson_solver.F.boundaries_buffer_z1_d; 
  }
  
  Grav.Poisson_solver.Copy_Transfer_Buffer_To_Host( size_buffer, buffer_host, buffer_dev );
  
  
  return size_buffer;
}


void Grid3D::Unload_Poisson_Boundary_From_Buffer( int direction, int side, Real *buffer_host  ){
  
  int size_buffer;
  
  if ( direction == 0 ) size_buffer = Grav.Poisson_solver.size_buffer_x;
  if ( direction == 1 ) size_buffer = Grav.Poisson_solver.size_buffer_y;
  if ( direction == 2 ) size_buffer = Grav.Poisson_solver.size_buffer_z;
  
  
  // Copy the host_buffer to the device_buffer
  Real *buffer_dev;
  if ( direction == 0 ){
    if ( side == 0 ) buffer_dev = Grav.Poisson_solver.F.recv_boundaries_buffer_x0_d;
    if ( side == 1 ) buffer_dev = Grav.Poisson_solver.F.recv_boundaries_buffer_x1_d; 
  }
  if ( direction == 1 ){
    if ( side == 0 ) buffer_dev = Grav.Poisson_solver.F.recv_boundaries_buffer_y0_d;
    if ( side == 1 ) buffer_dev = Grav.Poisson_solver.F.recv_boundaries_buffer_y1_d; 
  }
  if ( direction == 2 ){
    if ( side == 0 ) buffer_dev = Grav.Poisson_solver.F.recv_boundaries_buffer_z0_d;
    if ( side == 1 ) buffer_dev = Grav.Poisson_solver.F.recv_boundaries_buffer_z1_d; 
  }
  
  Grav.Poisson_solver.Copy_Transfer_Buffer_To_Device( size_buffer, buffer_host, buffer_dev );
  
  //Unload the transfer buffer in the GPU
  if ( direction == 0 ){
    if ( side == 0 ) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_x0();
    if ( side == 1 ) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_x1();
  }
  if ( direction == 1 ){
    if ( side == 0 ) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_y0();
    if ( side == 1 ) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_y1();
  }
  if ( direction == 2 ){
    if ( side == 0 ) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_z0();
    if ( side == 1 ) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_z1();
  }
  
  
  
}



void Potential_SOR_3D::Load_Transfer_Buffer_GPU_x0(){
  Load_Transfer_Buffer_GPU( 0, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.boundaries_buffer_x0_d  );
}
void Potential_SOR_3D::Load_Transfer_Buffer_GPU_x1(){
  Load_Transfer_Buffer_GPU( 0, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.boundaries_buffer_x1_d  );
}

void Potential_SOR_3D::Load_Transfer_Buffer_GPU_y0(){
  Load_Transfer_Buffer_GPU( 1, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.boundaries_buffer_y0_d  );
}
void Potential_SOR_3D::Load_Transfer_Buffer_GPU_y1(){
  Load_Transfer_Buffer_GPU( 1, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.boundaries_buffer_y1_d  );
}

void Potential_SOR_3D::Load_Transfer_Buffer_GPU_z0(){
  Load_Transfer_Buffer_GPU( 2, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.boundaries_buffer_z0_d  );
}
void Potential_SOR_3D::Load_Transfer_Buffer_GPU_z1(){
  Load_Transfer_Buffer_GPU( 2, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.boundaries_buffer_z1_d  );
}


void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_x0(){
  Unload_Transfer_Buffer_GPU( 0, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.recv_boundaries_buffer_x0_d  );
}
void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_x1(){
  Unload_Transfer_Buffer_GPU( 0, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.recv_boundaries_buffer_x1_d  );
}

void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_y0(){
  Unload_Transfer_Buffer_GPU( 1, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.recv_boundaries_buffer_y0_d  );
}
void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_y1(){
  Unload_Transfer_Buffer_GPU( 1, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.recv_boundaries_buffer_y1_d  );
}

void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_z0(){
  Unload_Transfer_Buffer_GPU( 2, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.recv_boundaries_buffer_z0_d  );
}
void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_z1(){
  Unload_Transfer_Buffer_GPU( 2, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d, F.recv_boundaries_buffer_z1_d  );
}



bool Potential_SOR_3D::Get_Global_Converged( bool converged_local ){
  
  int in = (int) converged_local;
  int out;
  bool y;

  MPI_Allreduce( &in, &out, 1, MPI_INT, MPI_MIN, world);
  y = (bool) out;
  return y;

}

#endif

// void Potential_SOR_3D::Load_Transfer_Buffer_GPU_All(){
  //   Load_Transfer_Buffer_GPU_x0();
  //   Load_Transfer_Buffer_GPU_x1();
  //   Load_Transfer_Buffer_GPU_y0();
  //   Load_Transfer_Buffer_GPU_y1();
  //   Load_Transfer_Buffer_GPU_z0();
  //   Load_Transfer_Buffer_GPU_z1();
  // }
  // 
  // void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_All(){
    //   Unload_Transfer_Buffer_GPU_x0();
    //   Unload_Transfer_Buffer_GPU_x1();
    //   Unload_Transfer_Buffer_GPU_y0();
    //   Unload_Transfer_Buffer_GPU_y1();
    //   Unload_Transfer_Buffer_GPU_z0();
    //   Unload_Transfer_Buffer_GPU_z1();
    // }

// Real Potential_SOR_3D::Get_Potential( Real *input_density,  Real *output_potential, Real Grav_Constant, Real dens_avrg, Real current_a ){
// 
//   Copy_Input_And_Initialize( input_density, Grav_Constant, dens_avrg, current_a );
// 
// 
//   Real epsilon = 1e-3;
//   int max_iter = 10000000;
//   int n_iter = 0;
// 
//   F.converged_h[0] = 0;
// 
//   // For Diriclet Boudaries
//   Real omega = 2. / ( 1 + M_PI / nx_total  );
// 
//   // For Periodic Boudaries
//   // Real omega = 2. / ( 1 + 2*M_PI / nx_total  );
//   // chprintf("Omega: %f \n", omega);
// 
//   // Iterate to solve Poisson equation
//   while (F.converged_h[0] == 0 ) {
// 
//     Load_Transfer_Buffer_GPU_All();
//     Unload_Transfer_Buffer_GPU_All();   
// 
//     Poisson_Partial_Iteration( 0, omega, epsilon ); 
// 
//     Load_Transfer_Buffer_GPU_All();
//     Unload_Transfer_Buffer_GPU_All();    
// 
//     Poisson_Partial_Iteration( 1, omega, epsilon );
// 
//     n_iter += 1;
// 
//     // #ifdef MPI_CHOLLA
//     // F.converged_h[0] = Get_Global_Converged( F.converged_h[0] );
//     // #endif
// 
//     if ( n_iter == max_iter ) break;
//   }
// 
//   if ( n_iter == max_iter ) chprintf(" SOR: No convergence in %d iterations \n", n_iter);
//   else printf(" SOR: Converged in %d iterations \n", n_iter);
// 
//   Copy_Output( output_potential );
// 
//   return 0;
// 
// }



#endif //GRAVITY
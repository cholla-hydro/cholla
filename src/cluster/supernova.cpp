#ifdef SUPERNOVA
#include <stdio.h>
#include "../io.h"
#include <iostream> //cout
//#include "../global.h" //defines Real
//#include "../grid3D.h" // defines Header
//#include "cluster_list.h" //loads cluster list
#include "supernova.h" // defines interfaces
//#include "supernova_gpu.h"
#include <math.h> // defines ceil
#ifdef CUDA
#include "../global_cuda.h" //defines CudaSafeCall
#include "gpu.hpp"
#endif //CUDA


// defines kernel-callingfunctions in supernova_gpu.cu 


// Define Cluster object
// Initialize by loading cluster_list.h and making cuda array
// Define Cluster Rotate
namespace Supernova {
  //Real cluster_data[];
  
  Real *d_cluster_array;
  Real *d_omega_array;
  bool *d_flags_array;
  Real *d_hydro_array;

  Real R_cl;
  Real SFR;

  Real xMin;
  Real yMin;
  Real zMin;

  Real xMax;
  Real yMax;
  Real zMax;

  Real dx;
  Real dy;
  Real dz;


  int nx;
  int ny;
  int nz;

  int pnx;
  int pny;
  int pnz;

  int n_cluster;
  int n_cells;
  int n_fields;
  void Test(Header H);
  void Initialize(Grid3D G);
  Real Update_Grid(Grid3D G);
}




//H.dx = 1.0;
//H.dy = 1.0;
//H.dz = 1.0;

void Supernova::Test(Header H){
  printf("nx %d ny %d nz %d dx %f dy %f dz %f\n",nx,ny,nz,dx,dy,dz);
  printf("pnx %d pny %d pnz %d n_cell %d n_fields %d\n",pnx,pny,pnz,n_cells,n_fields);
  printf("Min %f %f %f Max %f %f %f",xMin,yMin,zMin,xMax,yMax,zMax);
}


void Supernova::Initialize(Grid3D G){

  #include "cluster_list.data"
  // Defines cluster_data in local scope so it is deleted
  n_cluster = sizeof(cluster_data)/sizeof(cluster_data[0])/5;

  Header H = G.H;

  
  R_cl = 0.03;
  SFR = 20000.0;
  
  dx = H.dx;
  dy = H.dy;
  dz = H.dz;
  
  nx = H.nx;
  ny = H.ny;
  nz = H.nz;

#ifndef   MPI_CHOLLA

  xMin = H.xbound - H.dx*(H.n_ghost);
  xMax = H.xbound + H.dx*(nx-H.n_ghost);

  yMin = H.ybound - H.dy*(H.n_ghost);
  yMax = H.ybound + H.dy*(ny-H.n_ghost);

  zMin = H.zbound - H.dz*(H.n_ghost);
  zMax = H.zbound + H.dz*(nz-H.n_ghost);  

#else   /*MPI_CHOLLA*/

  xMin = H.xblocal - H.dx*(H.n_ghost);
  xMax = H.xblocal + H.dx*(nx-H.n_ghost);

  yMin = H.yblocal - H.dy*(H.n_ghost);
  yMax = H.yblocal + H.dy*(ny-H.n_ghost);

  zMin = H.zblocal - H.dz*(H.n_ghost);
  zMax = H.zblocal + H.dz*(nz-H.n_ghost);  

#endif  /*MPI_CHOLLA*/ 
  pnx = (int)ceil(R_cl/dx);
  pny = (int)ceil(R_cl/dy);
  pnz = (int)ceil(R_cl/dz); 
  
  //pnx,pny,pnz
  n_cells = H.n_cells;
  n_fields = H.n_fields;
  
#ifdef CUDA
  chprintf("Initializing Supernova CUDA arrays\n");
  d_hydro_array = G.C.device;
  CudaSafeCall( cudaMalloc (&d_cluster_array,5*n_cluster*sizeof(Real)));
  cudaMemcpy(d_cluster_array, cluster_data,
	     5*n_cluster*sizeof(Real),
	     cudaMemcpyHostToDevice);
  CudaSafeCall( cudaMalloc (&d_omega_array, n_cluster*sizeof(Real)));
  CudaSafeCall( cudaMalloc (&d_flags_array, n_cluster*sizeof(bool)));
  Calc_Omega();
  InitializeS99();
#endif //CUDA

  printf("\n n_cluster: %d\n",n_cluster);
  printf("nx %d ny %d nz %d dx %f dy %f dz %f\n",nx,ny,nz,dx,dy,dz);
  printf("pnx %d pny %d pnz %d n_cell %d n_fields %d\n",pnx,pny,pnz,n_cells,n_fields);
  printf("Min %f %f %f Max %f %f %f\n",xMin,yMin,zMin,xMax,yMax,zMax);



}

Real Supernova::Update_Grid(Grid3D G,Real old_dti){
  // Return dti
  Real old_dt = G.H.dt;

  Calc_Flags(G.H.t);
  Real new_dti = Feedback(0.1,0.0,G.H.t,old_dt);
#ifdef MPI_CHOLLA                                                             
  new_dti = ReduceRealMax(new_dti);                                             
#endif /*MPI_CHOLLA*/                                                         
                                                                                
                                                                                
  // sync newdti across GPUs 
  if (old_dti < new_dti){
    G.set_dt(new_dti);
    Feedback(0.1,0.0,G.H.t,G.H.dt-old_dt);
    return new_dti;
  }
  return old_dti;
}


#endif //SUPERNOVA

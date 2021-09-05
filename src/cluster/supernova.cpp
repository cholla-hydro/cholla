#include <stdio.h>
#include <iostream> //cout
#include "../global.h" //defines Real
#include "../global_cuda.h" //defines CudaSafeCall
#include "../grid3D.h" // defines Header
#include "cluster_list.h" //loads cluster list
#include "supernova.h" // defines interfaces
#include "supernova_gpu.h"
#include <math.h> // defines ceiol
#ifdef CUDA
#include "gpu.hpp"
#endif


// defines kernel-callingfunctions in supernova_gpu.cu 


// Define Cluster object
// Initialize by loading cluster_list.h and making cuda array
// Define Cluster Rotate
namespace Supernova {
  int length;

  Real *d_cluster_array;
  Real *d_omega_array;
  bool *d_flags_array;

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

  int n_cells;
  int n_fields;
  void Test(Header H);

}




//H.dx = 1.0;
//H.dy = 1.0;
//H.dz = 1.0;

void Supernova::Test(Header H){
  printf("nx %d ny %d nz %d dx %f dy %f dz %f\n",nx,ny,nz,dx,dy,dz);
  printf("pnx %d pny %d pnz %d n_cell %d n_fields %d\n",pnx,pny,pnz,n_cells,n_fields);
  printf("%f %f %f %f %f %f",xMin,yMin,zMin,xMax,yMax,zMax);
}

int main(void){
  std::cout << cluster_data[0] << std::endl;
  std::cout << cluster_data[1] << std::endl;
  std::cout << cluster_data[2] << std::endl;
  std::cout << cluster_data[3] << std::endl;
  //  std::cout << &cluster_data[0] << std::endl;
  std::cout << cluster_data+0 << std::endl;
  std::cout << cluster_data+1 << std::endl;
  std::cout << cluster_data+2 << std::endl;
  std::cout << cluster_num_particles << std::endl;
  std::cout << cos(100000.0) << std::endl;
  Supernova::length = cluster_num_particles;
  struct Header H;
  H.nx = 1;
  H.ny = 2;
  H.nz = 3;
  H.dx = 3.5;
  H.dy = 2.5;
  H.dz = 1.5;
  H.n_ghost = 1;

  std::cout << "H xyz: " << H.dx << H.dy << H.dz << std::endl;
  std::cout << "H xyz: " << H.nx << H.ny << H.nz << std::endl;
    
  Supernova::Initialize(H);
  Supernova::Test(H);
    

  std::cout << Supernova::length << std::endl;
  std::cout << Supernova::nx << std::endl;
  std::cout << Supernova::ny << std::endl;
  std::cout << Supernova::nz << std::endl;
  std::cout << Supernova::dx << std::endl;
  std::cout << Supernova::dy << std::endl;
  std::cout << Supernova::dz << std::endl;
  
  return 0;
}





void Supernova::Initialize(Header H){
  length = cluster_num_particles;
  
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
  cudaMemcpy(d_cluster_array, cluster_data,
	     5*cluster_num_particles*sizeof(Real),
	     cudaMemcpyHostToDevice);
  CudaSafeCall( cudaMalloc (&d_omega_array, cluster_num_particles*sizeof(Real)));
  CudaSafeCall( cudaMalloc (&d_flags_array, cluster_num_particles*sizeof(bool)));
  Calc_Omega();
#endif



}




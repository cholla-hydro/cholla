#ifdef SUPERNOVA
#include<math.h>
#include"io.h"
#include"gpu.hpp"
#include"../global.h"
#include"../global_cuda.h"
#include"../grid3D.h"
#include"supernova.h"


//texture<float, 1, cudaReadModeElementType> mdotTexObj;
//texture<float, 1, cudaReadModeElementType> edotTexObj;
namespace Supernova {
  Real* d_mdot;//table data
  Real* d_edot;//table data
  Real* d_mdot_array;//holds m_dot(cluster)[time]
  Real* d_edot_array;//holds e_dot(cluster)[time]
}


void Supernova::InitializeS99(void){
#include "S99_table.data"
  int n_entries = sizeof(s99_data)/sizeof(s99_data[0])/3;
  Real M_dot[n_entries];
  Real E_dot[n_entries];
  for (int i=0;i<n_entries;i++){
    M_dot[i] = s99_data[3*i+1];
    E_dot[i] = s99_data[3*i+2];
  }
  // Allocate M_dot and E_dot arrays on cuda
  CudaSafeCall( cudaMalloc (&d_mdot,n_entries*sizeof(Real)));
  CudaSafeCall( cudaMalloc (&d_edot,n_entries*sizeof(Real)));
  cudaMemcpy(d_mdot, M_dot, n_entries*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edot, E_dot, n_entries*sizeof(Real), cudaMemcpyHostToDevice);
  CudaSafeCall( cudaMalloc (&d_mdot_array,n_cluster*sizeof(Real)));
  CudaSafeCall( cudaMalloc (&d_edot_array,n_cluster*sizeof(Real)));
}



__device__ Real distance(double x, double y, double z){
  return x*x + y*y + z*z;
}

__device__ void Supernova_Helper(Real *hydro_dev,
				 Real pos_x, Real pos_y, Real pos_z,
				 Real dx, Real dy, Real dz,
				 int local_i, int local_j, int local_k,
				 int n_cells, int n_fields,
				 Real R_cl, Real density, Real energy, int gidx){
  // Compute the effect on hydro_dev fields of a supernova at pos_x,y,z on grid index local_i,j,k for spacing dx,y,z
  
  // pos_x, pos_y, pos_z is supernova position relative to local grid
  // dx,dy,dz grid spacing
  // local_i, local_j, local_k indices relative to local grid


  // cell center distances to supernova
  Real xc = fabs((local_i+0.5)*dx - pos_x);
  Real yc = fabs((local_j+0.5)*dy - pos_y);
  Real zc = fabs((local_k+0.5)*dz - pos_z);
  // cell corner distances to supernova
  Real rl = distance(xc - 0.5*dx, yc - 0.5*dy, zc - 0.5*dz);
  Real rr = distance(xc + 0.5*dx, yc + 0.5*dy, zc + 0.5*dz);
  Real R_cl2 = R_cl*R_cl;
  // Check if local cell overlaps with R_cl radius
  if (rr <= R_cl2) {
    // Add energy simple
    atomicAdd(&hydro_dev[gidx],density);
    atomicAdd(&hydro_dev[gidx+4*n_cells],energy);
    #ifdef SCALAR
    atomicAdd(&hydro_dev[gidx+5*n_cells],density);    
    #endif    
    #ifdef DE
    atomicAdd(&hydro_dev[gidx+(n_fields-1)*n_cells],energy);
    #endif
    return;
  }

  if (rl < R_cl2) {
    /*
    int count = 0;
    // Add energy fractional
    // TODO: implement weight
    for (int i=0;i<10;i++){
      for (int j=0;j<10;j++){
	for (int k=0;k<10;k++){
	  if (distance(xc + (0.1*i - 0.95)*dx, yc + (0.1*j - 0.95)*dy, zc + (0.1*k - 0.95)*dz) < R_cl2){
	    count++;
	  }
	}
      }
    }

    Real weight = count/1000.0;
    */
    Real weight = 0.5;
    atomicAdd(&hydro_dev[gidx],weight*density);
    atomicAdd(&hydro_dev[gidx+4*n_cells],weight*energy);
    #ifdef SCALAR
    atomicAdd(&hydro_dev[gidx+5*n_cells],weight*density);    
    #endif    
    #ifdef DE
    atomicAdd(&hydro_dev[gidx+(n_fields-1)*n_cells],weight*energy);
    #endif
    return;
  }
  return;  
}

__global__ void Particle_Feedback_Kernel(Real *hydro_dev, Real *pos_x_dev, Real *pos_y_dev, Real *pos_z_dev, Real xMin, Real yMin, Real zMin, Real dx, Real dy, Real dz, int nx, int ny, int nz, int n_cells, int n_fields, Real R_cl, Real density, Real energy){
  // Assume x,y,z Min and Max are edges of grid[i][j][k]
  // nx,ny,nz are grid sizes
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;

  // Compute sizes based on R_cl
  int pnx = (int)ceil(R_cl/dx);
  int pny = (int)ceil(R_cl/dy);
  int pnz = (int)ceil(R_cl/dz);
  int isize = 1+2*pnx;
  int jsize = 1+2*pny;
  int ksize = 1+2*pnz;
  int ijsize = isize*jsize;
  int ijksize = ijsize*ksize;

  // Determine Particle
  int pre_pid = tid/ijksize;
  // TODO: calculate which particle by looping through flags until tid/ncells is satisfied
  int pid = pre_pid;
  Real pos_x = pos_x_dev[pid]-xMin;
  Real pos_y = pos_y_dev[pid]-yMin;
  Real pos_z = pos_z_dev[pid]-zMin;

  // i,j,k of the block
  int rel_k = (tid - pre_pid*ijksize)/ijsize;
  int rel_j = (tid - pre_pid*ijksize - rel_k*ijsize)/isize;
  int rel_i =  tid - pre_pid*ijksize - rel_k*ijsize - rel_j*isize;

  // particle cell location in grid - pni is the left corner of our kernel block
  // local_i is index of cell in grid index coordinates
  int local_i = (int) floor(pos_x/dx) - pnx + rel_i;
  int local_j = (int) floor(pos_y/dy) - pny + rel_j;
  int local_k = (int) floor(pos_z/dz) - pnz + rel_k;

  // Check if local cell is inside Grid 
  if (local_i < 0 || local_j < 0 || local_k < 0){
    return;
  }

  if (local_i >= nx || local_j >= ny || local_k >= nz){
    return;
  }
  int gidx = local_i + (local_j + local_k * ny) * nx;
  
  Supernova_Helper(hydro_dev, pos_x, pos_y, pos_z, dx, dy, dz, local_i, local_j, local_k, n_cells, n_fields, R_cl, density, energy, gidx);
}

// TODO Make version of Kernel which launches per-particle kernels 



__global__ void Calc_Omega_Kernel(Real *cluster_array, Real *omega_array, int n_cluster){
  // cluster_array and omega_array should be on device
  // n_cluster is the total number
  Real r_pos;
  Real z_pos;
  Real r_sph, a, v;
  
  // properties of halo and disk
  Real a_disk_r, a_halo, a_halo_r;
  Real M_vir, M_d, R_vir, R_d, z_d, R_h, M_h, c_vir, phi_0_h, x;

  // GN is defined by global.h
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if (tid >= n_cluster){
    return;
  }
  // Get r_pos and z_pos from cluster array
  //
  r_pos = cluster_array[5*tid+2];
  z_pos = cluster_array[5*tid+4];
  
  
  // for halo component, calculate spherical r
  r_sph = sqrt(r_pos * r_pos + z_pos*z_pos);

  // MW model
  /*
  M_vir = 1.0e12; // virial mass in M_sun
  M_d = 6.5e10; // virial mass in M_sun
  R_vir = 261.; // virial radius in kpc
  R_d = 3.5; // disk scale length in kpc
  z_d = 3.5/5.0; // disk scale height in kpc
  c_vir = 20.0; // halo concentration
  */
  
  // M82 model
  M_d = 1.0e10; // mass of disk in M_sun	
  M_vir = 5.0e10; // virial mass in M_sun
  R_d = 0.8; // disk scale length in kpc
  R_vir = R_d/0.015; // virial radius in kpc

  z_d = 0.15; // disk scale height in kpc

  c_vir = 10.0; // halo concentration
  M_h = M_vir - M_d; // halo mass in M_sun
  R_h = R_vir / c_vir; // halo scale length in kpc
  phi_0_h = GN * M_h / (log(1.0+c_vir) - c_vir / (1.0+c_vir));
  x = r_sph / R_h;
  
  // calculate acceleration due to NFW halo & Miyamoto-Nagai disk
  a_halo = - phi_0_h * (log(1+x) - x/(1+x)) / (r_sph*r_sph);
  a_halo_r = a_halo*(r_pos/r_sph);
  a_disk_r = - GN * M_d * r_pos * pow(r_pos*r_pos+ pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
  // total acceleration is the sum of the halo + disk components
  a = fabs(a_halo_r) + fabs(a_disk_r);
  // radial velocity
  v = sqrt(r_pos*a);
  // how far has the cluster gone?
  // omega = v/r_pos;
  omega_array[tid] = v/r_pos;
}

void Supernova::Calc_Omega(void){
  dim3 dim1dGrid((n_cluster+TPB-1)/TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Calc_Omega_Kernel,dim1dGrid,dim1dBlock,0,0,d_cluster_array,d_omega_array,n_cluster);
}


__global__ void Calc_Flag_Kernel(Real *cluster_array, Real *omega_array, bool *flag_array,
				 Real *d_mdot, Real *d_edot, Real *d_mdot_array, Real *d_edot_array,
				 int n_cluster, 
				 Real time, Real xMin, Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, 
				 Real R_cl, Real SFR){
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if (tid >= n_cluster){
    return;
  }
  // Check if it is time for this cluster to be active
  // SF_cl/20000 < t < SF_cl/20000 + 40000
  Real total_SF = cluster_array[5*tid+1];
  Real convert_time = ((time - total_SF/SFR)*1e3-1e4)*1e-5;
  int table_index = (int)floor(convert_time);

  if (time < total_SF/SFR) {
    flag_array[tid] = false;
    return;
  }
  // SB99 table goes up to 9e7 yr = 9e4 kyr (code time)
  if (time > total_SF/SFR + 4e4){
    flag_array[tid] = false;
    return;
  }
  // Check if this cluster can affect the domain
  // Z position Check
  Real pos_z = cluster_array[5*tid+4];
  if (pos_z > zMax + R_cl){
    flag_array[tid] = false;
    return;    
  }
  if (pos_z < zMin - R_cl){
    flag_array[tid] = false;
    return;    
  }
  // XY position checks
  Real pos_phi = cluster_array[5*tid+3] + omega_array[tid]*time;
  Real pos_r = cluster_array[5*tid+2];
  Real pos_y = pos_r * sin(pos_phi);
  Real pos_x = pos_r * cos(pos_phi);
  
  if (pos_y > yMax + R_cl){
    flag_array[tid] = false;
    return;    
  }
  if (pos_y < yMin - R_cl){
    flag_array[tid] = false;
    return;    
  }
  if (pos_x > xMax + R_cl){
    flag_array[tid] = false;
    return;    
  }
  if (pos_x < xMin - R_cl){
    flag_array[tid] = false;
    return;    
  }
  
  /*
  if (pos_y > yMax + R_cl || pos_y < yMin - R_cl || pos_x > xMax + R_cl || pos_x < xMin - R_cl){
    flag_array[tid] = false;
    return;
  }
  */



  flag_array[tid] = true;

  // Use table to set arrays 
  // 1e3 is KYR conversion
  // SB99 table starts at 1e4
  // 1e5 is SB99 table spacing 
  // Real convert_time = ((time - total_SF/SFR)*1e3-1e4)*1e-5;
  // int table_index = (int)floor(convert_time);

  // If we got this far, then table_index will be a valid index for this array

  if (table_index >= 0){
  int table_fraction = convert_time - table_index;
  Real f = cluster_array[5*tid]*1e-6;
  Real M_slope = d_mdot[table_index+1] - d_mdot[table_index];
  Real E_slope = d_edot[table_index+1] - d_edot[table_index];
  d_mdot_array[tid] = f*(d_mdot[table_index]+table_fraction*M_slope);
  d_edot_array[tid] = f*pow(10, (d_edot[table_index] + table_fraction*E_slope)) * TIME_UNIT/(MASS_UNIT*VELOCITY_UNIT*VELOCITY_UNIT);
  }
  return;
}


void Supernova::Calc_Flags(Real time){
  CHECK(cudaDeviceSynchronize());
  double start_time = get_time(); 			    
  dim3 dim1dGrid((n_cluster+TPB-1)/TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Calc_Flag_Kernel,dim1dGrid,dim1dBlock,0,0,
		     d_cluster_array,d_omega_array,d_flags_array,
		     d_mdot, d_edot, d_mdot_array, d_edot_array,
		     n_cluster,
		     time, xMin, yMin, zMin, xMax, yMax, zMax, R_cl, SFR);
  CHECK(cudaDeviceSynchronize());
  double end_time = get_time();
  chprintf("Supernova Calc Flags: %9.4f \n",1000*(end_time-start_time));

}

// Then make a kernel based on this flag kernel thing

// Lastly start doing some cuda timing tests on the flag + supernova step

__global__ void Supernova_Feedback_Kernel(Real *hydro_dev, Real *cluster_array, Real *omega_array, bool *flags_array, Real xMin, Real yMin, Real zMin, Real dx, Real dy, Real dz, int nx, int ny, int nz, int pnx, int pny, int pnz, int n_cells, int n_fields, Real R_cl, Real density, Real energy, Real time, Real dt, int max_pid){
  // Assume x,y,z Min and Max are edges of grid[i][j][k]
  // nx,ny,nz are grid sizes
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;

  // Compute sizes based on R_cl
  /*
  int pnx = (int)ceil(R_cl/dx);
  int pny = (int)ceil(R_cl/dy);
  int pnz = (int)ceil(R_cl/dz);
  */
  int isize = 1+2*pnx;
  int jsize = 1+2*pny;
  int ksize = 1+2*pnz;
  int ijsize = isize*jsize;
  int ijksize = ijsize*ksize;

  // Determine Particle
  int pre_pid = tid/ijksize;
  if (pre_pid >= max_pid){
    return;
  }
  // TODO?: calculate which particle by looping through flags until tid/ncells is satisfied Not necessary, since launching a kernel that gets to this point has almost no cost. 
  int pid = pre_pid;
  if (!flags_array[pid]){
    return;
  }
  Real pos_r = cluster_array[5*pid+2];
  Real pos_phi = cluster_array[5*pid+3] + omega_array[pid]*time;
  Real pos_x = pos_r * cos(pos_phi) - xMin;
  Real pos_y = pos_r * sin(pos_phi) - yMin;
  Real pos_z = cluster_array[5*pid+4] - zMin;

  // i,j,k of the block
  int rel_k = (tid - pre_pid*ijksize)/ijsize;
  int rel_j = (tid - pre_pid*ijksize - rel_k*ijsize)/isize;
  int rel_i =  tid - pre_pid*ijksize - rel_k*ijsize - rel_j*isize;

  // particle cell location in grid - pni is the left corner of our kernel block
  // local_i is index of cell in grid index coordinates
  int local_i = (int) floor(pos_x/dx) - pnx + rel_i;
  int local_j = (int) floor(pos_y/dy) - pny + rel_j;
  int local_k = (int) floor(pos_z/dz) - pnz + rel_k;

  // Check if local cell is inside Grid 
  if (local_i < 0 || local_j < 0 || local_k < 0){
    return;
  }

  if (local_i >= nx || local_j >= ny || local_k >= nz){
    return;
  }
  int gidx = local_i + (local_j + local_k * ny) * nx;
  
  Supernova_Helper(hydro_dev, pos_x, pos_y, pos_z, dx, dy, dz, local_i, local_j, local_k, n_cells, n_fields, R_cl, density, energy, gidx);
}

void Supernova::Feedback(Real density, Real energy, Real time, Real dt){
  CHECK(cudaDeviceSynchronize());
  double start_time = get_time(); 			    
  int isize = 1+2*pnx;
  int jsize = 1+2*pny;
  int ksize = 1+2*pnz;
  dim3 dim1dGrid((n_cluster*isize*jsize*ksize+TPB-1)/TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Supernova_Feedback_Kernel,dim1dGrid,dim1dBlock,0,0,
		     d_hydro_array, d_cluster_array, d_omega_array, d_flags_array,
		     xMin, yMin, zMin, dx, dy, dz, nx, ny, nz, pnx, pny, pnz,
		     n_cells, n_fields, R_cl, density, energy, time, dt, n_cluster);
  CHECK(cudaDeviceSynchronize());
  double end_time = get_time();
  chprintf("Supernova Feedback Time: %9.4f \n",1000*(end_time-start_time));

}

#endif //SUPERNOVA



#include "feedback_analysis.h"
#include "../io/io.h"
#include <cstdio>
#ifdef PARTICLES_GPU

#define MU 0.6
// in cgs, this is 0.01 cm^{-3}
#define MIN_DENSITY 0.01 * MP * MU * LENGTH_UNIT * LENGTH_UNIT * LENGTH_UNIT / MASS_UNIT  // 148279.7
#define TPB_ANALYSIS 1024


__device__ void warpReduce(volatile Real *buff, size_t tid)
{
    if (TPB_ANALYSIS >= 64) buff[tid] += buff[tid + 32];
    if (TPB_ANALYSIS >= 32) buff[tid] += buff[tid + 16];
    if (TPB_ANALYSIS >= 16) buff[tid] += buff[tid +  8];
    if (TPB_ANALYSIS >=  8) buff[tid] += buff[tid +  4];
    if (TPB_ANALYSIS >=  4) buff[tid] += buff[tid +  2];
    if (TPB_ANALYSIS >=  2) buff[tid] += buff[tid +  1];
}


void __global__ Reduce_Tubulence_kernel(int nx, int ny, int nz, int n_ghost, Real *density, Real *momentum_x, Real *momentum_y, 
  Real *momentum_z, Real *circ_vel_x, Real *circ_vel_y, Real *partial_mass, Real *partial_vel) {
  __shared__ Real s_mass[TPB_ANALYSIS];
  __shared__ Real s_vel[TPB_ANALYSIS];
  int id, zid, yid, xid, tid;

  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;
  tid = threadIdx.x;

  s_mass[tid] = 0;
  s_vel[tid] = 0;
  Real vx, vy, vz;
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost && density[id] > MIN_DENSITY) {
    s_mass[tid] = density[id];
    vx =  momentum_x[id]/ density[id];
    vy =  momentum_y[id]/ density[id];
    vz =  momentum_z[id]/ density[id];
    s_vel[tid] = ( (vx - circ_vel_x[id])*(vx - circ_vel_x[id]) +
                   (vy - circ_vel_y[id])*(vy - circ_vel_y[id]) +
                   (vz*vz)
                 )*density[id];
  }
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      s_mass[tid] += s_mass[tid + s];
      s_vel[tid] += s_vel[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    //printf("ReduceKernel 1: blockIdx.x = %d -> s_mass[0] = %.5e, s_vel[0] = %.5e\n", blockIdx.x, s_mass[0], s_vel[0]);
    partial_mass[blockIdx.x] = s_mass[0];
    partial_vel[blockIdx.x] = s_vel[0];
  }
}


void __global__ Reduce_Tubulence_kernel_2(Real *input_m, Real *input_v, Real *output_m, Real *output_v, int n) {
  __shared__ Real s_mass[TPB_ANALYSIS];
  __shared__ Real s_vel[TPB_ANALYSIS];

  size_t tid = threadIdx.x;
  size_t i = blockIdx.x*(TPB_ANALYSIS) + tid;
  size_t gridSize = TPB_ANALYSIS*gridDim.x;
  s_mass[tid] = 0;
  s_vel[tid] = 0;

  while (i < n) { 
    s_mass[tid] += input_m[i];
    s_vel[tid] += input_v[i]; 
    i += gridSize;
  }
  __syncthreads();

  if (TPB_ANALYSIS >= 1024) { if (tid < 512) { s_mass[tid] += s_mass[tid + 512]; s_vel[tid] += s_vel[tid + 512]; } __syncthreads(); }
  if (TPB_ANALYSIS >=  512) { if (tid < 256) { s_mass[tid] += s_mass[tid + 256]; s_vel[tid] += s_vel[tid + 256]; } __syncthreads(); }
  if (TPB_ANALYSIS >=  256) { if (tid < 128) { s_mass[tid] += s_mass[tid + 128]; s_vel[tid] += s_vel[tid + 128]; } __syncthreads(); }
  if (TPB_ANALYSIS >=  128) { if (tid <  64) { s_mass[tid] += s_mass[tid +  64]; s_vel[tid] += s_vel[tid +  64]; } __syncthreads(); }

  if (tid < 32) { warpReduce(s_mass, tid); warpReduce(s_vel, tid); }
  __syncthreads();

  if (tid == 0) {
    //printf("Reduce_Tubulence_kernel 2: n = %d/%d, blockIdx.x = %d -> s_mass[0] = %.5e, s_vel[0] = %.5e\n", 
    //       n, gridDim.x, blockIdx.x, s_mass[0], s_vel[0]);
    output_m[blockIdx.x] = s_mass[0];
    output_v[blockIdx.x] = s_vel[0];
  }
}


void FeedbackAnalysis::Compute_Gas_Velocity_Dispersion_GPU(Grid3D& G) {
  size_t ngrid = std::ceil((1.*G.H.nx*G.H.ny*G.H.nz)/TPB_ANALYSIS);
  
  Real* d_partial_mass;
  Real* d_partial_vel;
  Real* h_partial_mass = (Real *) malloc(ngrid*sizeof(Real));
  Real* h_partial_vel = (Real *) malloc(ngrid*sizeof(Real));
  CHECK(cudaMalloc((void**)&d_partial_mass, ngrid*sizeof(Real)));
  CHECK(cudaMalloc((void**)&d_partial_vel, ngrid*sizeof(Real)));
  
  Real total_mass = 0;
  Real total_vel = 0;

  hipLaunchKernelGGL(Reduce_Tubulence_kernel, ngrid, TPB_ANALYSIS, 0, 0, G.H.nx, G.H.ny, G.H.nz, G.H.n_ghost, 
                     G.C.d_density, G.C.d_momentum_x, G.C.d_momentum_y, G.C.d_momentum_z, 
                     d_circ_vel_x, d_circ_vel_y, d_partial_mass, d_partial_vel);
  
  size_t n = ngrid;
  Real *mass_input = d_partial_mass;
  Real *vel_input = d_partial_vel;
  while (n > TPB_ANALYSIS) {
    ngrid = std::ceil( (n*1.)/TPB_ANALYSIS );
    //printf("Reduce_Tubulence: Next kernel call grid size is %d\n", ngrid);
    hipLaunchKernelGGL(Reduce_Tubulence_kernel_2, ngrid, TPB_ANALYSIS, 0, 0, mass_input, vel_input, d_partial_mass, d_partial_vel, n);
    mass_input = d_partial_mass;
    vel_input = d_partial_vel;
    n = ngrid;
  }

  if (n > 1) {
    hipLaunchKernelGGL(Reduce_Tubulence_kernel_2, 1, TPB_ANALYSIS, 0, 0, d_partial_mass, d_partial_vel, d_partial_mass, d_partial_vel, n);
  }
  
  //cudaDeviceSynchronize();

  CHECK(cudaMemcpy(h_partial_mass, d_partial_mass, ngrid*sizeof(Real), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_partial_vel, d_partial_vel, ngrid*sizeof(Real), cudaMemcpyDeviceToHost));  

  #ifdef MPI_CHOLLA
  MPI_Allreduce(h_partial_mass, &total_mass, 1, MPI_CHREAL, MPI_SUM, world);
  MPI_Allreduce(h_partial_vel, &total_vel, 1, MPI_CHREAL, MPI_SUM, world);
  #else
  total_mass = h_partial_mass[0];
  total_vel = h_partial_vel[0];
  #endif

  if (total_vel < 0 || total_mass < 0) {
    chprintf("feedback trouble.  total_vel = %.3e, total_mass = %.3e\n", total_vel, total_mass);
  }

  chprintf("feedback: time %f, dt=%f, vrms = %f km/s\n",  G.H.t,  G.H.dt, sqrt(total_vel/total_mass)*VELOCITY_UNIT/1e5);

  CHECK(cudaFree(d_partial_vel));
  CHECK(cudaFree(d_partial_mass));  

  free(h_partial_mass);
  free(h_partial_vel);
}

  #endif // PARTICLES_GPU

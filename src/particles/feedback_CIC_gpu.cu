#if defined(FEEDBACK) && defined(PARTICLES_GPU)

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../grid/grid3D.h"
#include "../global/global_cuda.h"
#include "../global/global.h"
#include "supernova.h"


namespace Supernova {
  curandStateMRG32k3a_t*  curandStates;
  part_int_t n_states;
}


__device__ double atomicMax(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(fmax(val, __longlong_as_double(assumed)))
		    );
  } while (assumed != old);
  return __longlong_as_double(old);
}


__global__ void initState_kernel(unsigned int seed, curandStateMRG32k3a_t* states) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}


/**
 * @brief Initialize the cuRAND state, which is analogous to the concept of generators in CPU code.
 * The state object maintains configuration and status the cuRAND context for each thread on the GPU.
 * Initialize more than the number of local particles since the latter will change through MPI transfers.
 * 
 * @param n_local 
 * @param allocation_factor 
 */
void Supernova::initState(struct parameters *P, part_int_t n_local, Real allocation_factor) {
  printf("Supernova::initState start\n");
  n_states = n_local*allocation_factor;
  cudaMalloc((void**) &curandStates, n_states*sizeof(curandStateMRG32k3a_t));

  //int ngrid =  (n_states + TPB_PARTICLES - 1) / TPB_PARTICLES;
  int ngrid =  (n_states + 64- 1) / 64;

  dim3 grid(ngrid);
  //dim3 block(TPB_PARTICLES);
  dim3 block(64);

  printf("Supernova::initState: n_states=%d, ngrid=%d, threads=%d\n", n_states, ngrid, 64);
  hipLaunchKernelGGL(initState_kernel, grid, block, 0, 0, P->prng_seed, curandStates);
  CHECK(cudaDeviceSynchronize());
  printf("Supernova::initState end\n");
}


/*
__device__ void Single_Cluster_Feedback(Real t, Real dt, Real age, Real density, Real* feedback, curandStateMRG32k3a_t* state) {
  int N = 0;
  if (t + age <= Supernova::SN_ERA) { 
     N = curand_poisson (state, Supernova::SNR * dt);
  }
  Real n_0 = density * DENSITY_UNIT / (Supernova::MU*MP);  // in cm^{-3}

  feedback[Supernova::NUMBER]       = N * 1.0;                                                    // number of SN 
  feedback[Supernova::ENERGY]       = N * Supernova::ENERGY_PER_SN;                               // total energy
  feedback[Supernova::MASS]         = N * Supernova::MASS_PER_SN;                                 // total mass
  feedback[Supernova::MOMENTUM]     = Supernova::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(N, 0.93); // final momentum
  feedback[Supernova::SHELL_RADIUS] = Supernova::R_SH * pow(n_0, -0.46) * pow(N, 0.29);           // shell formation radius
}
*/

__device__ Real Calc_Timestep(Real gamma, Real *density, Real *momentum_x, Real *momentum_y, Real *momentum_z, Real *energy, int index, Real dx, Real dy, Real dz){
  Real dens = fmax(density[index], DENS_FLOOR);
  Real d_inv = 1.0 / dens;
  Real vx = momentum_x[index] * d_inv;
  Real vy = momentum_y[index] * d_inv; 
  Real vz = momentum_z[index] * d_inv;
  Real P  = fmax((energy[index]- 0.5*dens*(vx*vx + vy*vy + vz*vz))*(gamma-1.0), TINY_NUMBER);
  Real cs = sqrt(gamma * P * d_inv);
  return fmax( fmax((fabs(vx) + cs)/dx, (fabs(vy) + cs)/dy), (fabs(vz) + cs)/dz );
}


__global__ void Cluster_Feedback_Kernel(part_int_t n_local, Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev, 
    Real mass, Real* age_dev, Real xMin, Real yMin, Real zMin, Real xLen, Real yLen, Real zLen, 
    Real dx, Real dy, Real dz, int nx_g, int ny_g, int nz_g, int n_ghost, Real t, Real dt, Real* dti,
    Real* density, Real* gasEnergy, Real* energy, Real* momentum_x, Real* momentum_y, Real* momentum_z, Real gamma, curandStateMRG32k3a_t* states){

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if ( tid >= n_local) return;
    
    Real xMax, yMax, zMax;
    xMax = xMin + xLen;
    yMax = yMin + yLen;
    zMax = zMin + zLen;

    Real pos_x, pos_y, pos_z;
    Real cell_center_x, cell_center_y, cell_center_z;
    Real delta_x, delta_y, delta_z;
    Real feedback_energy, feedback_density, feedback_momentum, n_0, shell_radius;
    bool is_resolved;
    int pcell_x, pcell_y, pcell_z, pcell_index;
    Real dV = dx*dy*dz;
    Real local_dti = 0.0;

    pos_x = pos_x_dev[tid];
    pos_y = pos_y_dev[tid];
    pos_z = pos_z_dev[tid];

    bool in_local = (pos_x >= xMin && pos_x < zMax) &&
                    (pos_y >= yMin && pos_y < yMax) &&
                    (pos_z >= zMin && pos_z < zMax);
    if (!in_local) {
        printf(" Feedback GPU: Particle outside local domain [%f  %f  %f]  [%f %f] [%f %f] [%f %f]\n ", pos_x, pos_y, pos_z, xMin, xMax, yMin, yMax, zMin, zMax);
        return;
    }

    int indx_x = (int) floor( ( pos_x - xMin - 0.5*dx ) / dx );
    int indx_y = (int) floor( ( pos_y - yMin - 0.5*dy ) / dy );
    int indx_z = (int) floor( ( pos_z - zMin - 0.5*dz ) / dz );

    bool ignore = indx_x < -1 || indx_y < -1 || indx_z < -1 || indx_x > nx_g-3 || indx_y > ny_g-3 || indx_y > nz_g-3;
    if (ignore) {
        printf(" Feedback GPU: Particle CIC index err [%f  %f  %f]  [%d %d %d] [%d %d %d] \n ", pos_x, pos_y, pos_z, indx_x, indx_y, indx_z, nx_g, ny_g, nz_g);
    }

    pcell_x = (int) floor( ( pos_x - xMin ) / dx ) + n_ghost;
    pcell_y = (int) floor( ( pos_y - yMin ) / dy ) + n_ghost;
    pcell_z = (int) floor( ( pos_z - zMin ) / dz ) + n_ghost;
    pcell_index = pcell_x + pcell_y*nx_g + pcell_z*nx_g*ny_g;

    if (t + age_dev[tid] > Supernova::SN_ERA) return;

    curandStateMRG32k3a_t state = states[tid]; // <- more efficient?
    unsigned int N = curand_poisson (&state, Supernova::SNR * dt);
    states[tid] = state;

    if (N == 0) return;
 
    feedback_energy   = N * Supernova::ENERGY_PER_SN / dV;
    feedback_density  = N * Supernova::MASS_PER_SN / dV;
    n_0 = density[pcell_index] * DENSITY_UNIT / (Supernova::MU*MP);
    feedback_momentum = Supernova::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(N, 0.93) / sqrt(3.0) / dV;
    shell_radius  = Supernova::R_SH * pow(n_0, -0.46) * pow(N, 0.29);
    is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius; 

    /*printf(" [%d]: got %d SN\n", tid, N);
    if (is_resolved) printf(" [%d] resolved\n", tid);
    else printf(" [%d] NOT resolved\n", tid);
    printf("      [%d] E=%.3e, D=%.3e, P=%.3e, S_r=%.3e\n", tid,
        feedback_energy*dV*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT,
        feedback_density*DENSITY_UNIT / (Supernova::MU*MP),
        feedback_momentum*dV*VELOCITY_UNIT/1e5, shell_radius);
    */
    cell_center_x = xMin + indx_x*dx + 0.5*dx;
    cell_center_y = yMin + indx_y*dy + 0.5*dy;
    cell_center_z = zMin + indx_z*dz + 0.5*dz;
    delta_x = 1 - ( pos_x - cell_center_x ) / dx;
    delta_y = 1 - ( pos_y - cell_center_y ) / dy;
    delta_z = 1 - ( pos_z - cell_center_z ) / dz;
    indx_x += n_ghost;
    indx_y += n_ghost;
    indx_z += n_ghost;

    int indx = indx_x + indx_y*nx_g + indx_z*nx_g*ny_g;
    if (is_resolved) {
      atomicAdd(&density[indx], feedback_density  * delta_x * delta_y * delta_z);
      atomicAdd(&gasEnergy[indx], feedback_energy  * delta_x * delta_y * delta_z);
      atomicAdd(&energy[indx], feedback_energy  * delta_x * delta_y * delta_z);
      //info[threadId*N_INFO + 3] += feedback_energy  * fabs(delta_x * delta_y * delta_z) * dV;
    } else {
      atomicAdd(&momentum_x[indx], -delta_x * feedback_momentum);
      atomicAdd(&momentum_y[indx], -delta_y * feedback_momentum);
      atomicAdd(&momentum_z[indx], -delta_z * feedback_momentum);
      //info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

    indx = (indx_x+1) + indx_y*nx_g + indx_z*nx_g*ny_g;
    if (is_resolved) {
      atomicAdd(&density[indx], feedback_density  * (1-delta_x) * delta_y * delta_z);
      atomicAdd(&gasEnergy[indx], feedback_energy  * (1-delta_x) * delta_y * delta_z);
      atomicAdd(&energy[indx], feedback_energy  * (1-delta_x) * delta_y * delta_z);
      //info[threadId*N_INFO + 3] += feedback_energy  * fabs((1-delta_x) * delta_y * delta_z) * dV;
    } else {
      atomicAdd(&momentum_x[indx],  delta_x * feedback_momentum);
      atomicAdd(&momentum_y[indx], -delta_y * feedback_momentum);
      atomicAdd(&momentum_z[indx], -delta_z * feedback_momentum);
      //info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

    indx = indx_x + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
    if (is_resolved) {
      atomicAdd(&density[indx], feedback_density  * delta_x * (1-delta_y) * delta_z);
      atomicAdd(&gasEnergy[indx], feedback_energy  * delta_x * (1-delta_y) * delta_z);
      atomicAdd(&energy[indx], feedback_energy  * delta_x * (1-delta_y) * delta_z);
      //info[threadId*N_INFO + 3] += feedback_energy  * fabs(delta_x * (1-delta_y )* delta_z) * dV;
    } else {
      atomicAdd(&momentum_x[indx], -delta_x * feedback_momentum);
      atomicAdd(&momentum_y[indx],  delta_y * feedback_momentum);
      atomicAdd(&momentum_z[indx],  -delta_z * feedback_momentum);
      //info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

    indx = indx_x + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
    if (is_resolved) {
      atomicAdd(&density[indx], feedback_density  * delta_x * delta_y * (1-delta_z));
      atomicAdd(&gasEnergy[indx], feedback_energy  * delta_x * delta_y * (1-delta_z));
      atomicAdd(&energy[indx], feedback_energy  * delta_x * delta_y * (1-delta_z));
      //info[threadId*N_INFO + 3] += feedback_energy  * fabs(delta_x * delta_y * (1 - delta_z)) * dV;
    } else {
      atomicAdd(&momentum_x[indx], -delta_x * feedback_momentum);
      atomicAdd(&momentum_y[indx], -delta_y * feedback_momentum);
      atomicAdd(&momentum_z[indx],  delta_z * feedback_momentum); 
      //info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

    indx = (indx_x+1) + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
    if (is_resolved) {
      atomicAdd(&density[indx], feedback_density  * (1-delta_x) * (1-delta_y) * delta_z);
      atomicAdd(&gasEnergy[indx], feedback_energy  * (1-delta_x) * (1-delta_y) * delta_z);
      atomicAdd(&energy[indx], feedback_energy  * (1-delta_x) * (1-delta_y) * delta_z);
      //info[threadId*N_INFO + 3] += feedback_energy  * fabs((1-delta_x) * (1-delta_y) * delta_z) * dV;
    } else {
      atomicAdd(&momentum_x[indx], delta_x * feedback_momentum);
      atomicAdd(&momentum_y[indx],  delta_y * feedback_momentum);
      atomicAdd(&momentum_z[indx], -delta_z * feedback_momentum);
      //info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

    indx = (indx_x+1) + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
    if (is_resolved) {
      atomicAdd(&density[indx], feedback_density  * (1-delta_x) * delta_y * (1-delta_z));
      atomicAdd(&gasEnergy[indx], feedback_energy  * (1-delta_x) * delta_y * (1-delta_z));
      atomicAdd(&energy[indx], feedback_energy  * (1-delta_x) * delta_y * (1-delta_z));
      //info[threadId*N_INFO + 3] += feedback_energy  * fabs((1-delta_x) * delta_y * (1-delta_z)) * dV;
    } else {
      atomicAdd(&momentum_x[indx],  delta_x * feedback_momentum);
      atomicAdd(&momentum_y[indx], -delta_y * feedback_momentum);
      atomicAdd(&momentum_z[indx],  delta_z * feedback_momentum);
      //info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

    indx = indx_x + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    if (is_resolved) {
      atomicAdd(&density[indx], feedback_density  * delta_x * (1-delta_y) * (1-delta_z));
      atomicAdd(&gasEnergy[indx], feedback_energy  * delta_x * (1-delta_y) * (1-delta_z));
      atomicAdd(&energy[indx], feedback_energy  * delta_x * (1-delta_y) * (1-delta_z));
      //info[threadId*N_INFO + 3], feedback_energy * fabs(delta_x * (1-delta_y) * (1-delta_z)) * dV;
    } else {
      atomicAdd(&momentum_x[indx], -delta_x * feedback_momentum);
      atomicAdd(&momentum_y[indx],  delta_y * feedback_momentum);
      atomicAdd(&momentum_z[indx],  delta_z * feedback_momentum);
      //info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

    indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    if (is_resolved) {
      atomicAdd(&density[indx], feedback_density * (1-delta_x) * (1-delta_y) * (1-delta_z));
      atomicAdd(&gasEnergy[indx], feedback_energy * (1-delta_x) * (1-delta_y) * (1-delta_z));
      atomicAdd(&energy[indx], feedback_energy * (1-delta_x) * (1-delta_y) * (1-delta_z));
      //info[threadId*N_INFO + 3] += feedback_energy * fabs((1-delta_x) * (1-delta_y) * (1-delta_z)) * dV;
    } else {
      atomicAdd(&momentum_x[indx],  delta_x * feedback_momentum);
      atomicAdd(&momentum_y[indx],  delta_y * feedback_momentum);
      atomicAdd(&momentum_z[indx],  delta_z * feedback_momentum);
      //info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));
    atomicMax(dti, local_dti);
}



Real Grid3D::Cluster_Feedback_GPU() {
  if (H.dt == 0) return 0.0;

  if (Particles.n_local > Supernova::n_states) {
    printf("ERROR: not enough cuRAND states (%d) for %f local particles\n", Supernova::n_states, Particles.n_local );
    exit(-1);
  }

  printf("Cluster_Feedback_GPU: start.  dt=%.4e\n", H.dt);
  Real h_dti = 0.0;
  Real* d_dti;
  cudaMalloc(&d_dti, sizeof(Real));
  cudaMemcpy(d_dti, &h_dti, sizeof(Real), cudaMemcpyHostToDevice);

  int ngrid =  (Particles.n_local + 64 - 1) / 64;
  dim3 grid(ngrid);
  dim3 block(64);

  hipLaunchKernelGGL(Cluster_Feedback_Kernel, grid, block, 0, 0,  Particles.n_local, Particles.pos_x_dev, Particles.pos_y_dev, Particles.pos_z_dev, 
       Particles.particle_mass, Particles.age_dev, H.xblocal, H.yblocal, H.zblocal, H.domlen_x, H.domlen_y, H.domlen_z, 
       H.dx, H.dy, H.dz, H.nx, H.ny, H.nz, H.n_ghost, H.t, H.dt, d_dti,
       C.d_density, C.d_GasEnergy, C.d_Energy, C.d_momentum_x, C.d_momentum_y, C.d_momentum_z, gama, Supernova::curandStates);

  cudaMemcpy(&h_dti, d_dti, sizeof(Real), cudaMemcpyDeviceToHost); 
  cudaFree(d_dti);
  printf("Cluster_Feedback_GPU: end.  calc dti=%.4e\n", h_dti);

  return h_dti;
}


#endif //FEEDBACK & PARTICLES_GPU

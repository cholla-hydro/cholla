#if defined(FEEDBACK) && defined(PARTICLES_GPU)

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../grid/grid3D.h"
#include "../global/global_cuda.h"
#include "../global/global.h"
#include "../io/io.h"
#include "supernova.h"

#define TPB_FEEDBACK 64
#define FEED_INFO_N 5

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
    Real* mass_dev, Real* age_dev, Real xMin, Real yMin, Real zMin, Real xLen, Real yLen, Real zLen, 
    Real dx, Real dy, Real dz, int nx_g, int ny_g, int nz_g, int n_ghost, Real t, Real dt, Real* dti, Real* info,
    Real* density, Real* gasEnergy, Real* energy, Real* momentum_x, Real* momentum_y, Real* momentum_z, Real gamma, curandStateMRG32k3a_t* states){

    __shared__ Real s_info[FEED_INFO_N*TPB_FEEDBACK]; // for collecting SN feedback information, like # of SNe or # resolved.
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid ;

    s_info[FEED_INFO_N*tid]     = 0;
    s_info[FEED_INFO_N*tid + 1] = 0;
    s_info[FEED_INFO_N*tid + 2] = 0;
    s_info[FEED_INFO_N*tid + 3] = 0;
    s_info[FEED_INFO_N*tid + 4] = 0;

   if ( gtid < n_local) {
      Real xMax, yMax, zMax;
      xMax = xMin + xLen;
      yMax = yMin + yLen;
      zMax = zMin + zLen;

      Real pos_x, pos_y, pos_z;
      Real cell_center_x, cell_center_y, cell_center_z;
      Real delta_x, delta_y, delta_z;
      Real feedback_energy = 0, feedback_density=0, feedback_momentum=0, n_0, shell_radius;
      bool is_resolved = false;
      int pcell_x, pcell_y, pcell_z, pcell_index;
      Real dV = dx*dy*dz;
      Real local_dti = 0.0;

      pos_x = pos_x_dev[gtid];
      pos_y = pos_y_dev[gtid];
      pos_z = pos_z_dev[gtid];

      bool in_local = (pos_x >= xMin && pos_x < zMax) &&
                      (pos_y >= yMin && pos_y < yMax) &&
                      (pos_z >= zMin && pos_z < zMax);
      if (!in_local) {
          printf(" Feedback GPU: Particle outside local domain [%f  %f  %f]  [%f %f] [%f %f] [%f %f]\n ", 
                pos_x, pos_y, pos_z, xMin, xMax, yMin, yMax, zMin, zMax);
      }

      int indx_x = (int) floor( ( pos_x - xMin - 0.5*dx ) / dx );
      int indx_y = (int) floor( ( pos_y - yMin - 0.5*dy ) / dy );
      int indx_z = (int) floor( ( pos_z - zMin - 0.5*dz ) / dz );

      bool ignore = indx_x < -1 || indx_y < -1 || indx_z < -1 || indx_x > nx_g-3 || indx_y > ny_g-3 || indx_y > nz_g-3;
      if (ignore) {
          printf(" Feedback GPU: Particle CIC index err [%f  %f  %f]  [%d %d %d] [%d %d %d] \n ", 
                pos_x, pos_y, pos_z, indx_x, indx_y, indx_z, nx_g, ny_g, nz_g);
      }

      if (!ignore && in_local) {
        pcell_x = (int) floor( ( pos_x - xMin ) / dx ) + n_ghost;
        pcell_y = (int) floor( ( pos_y - yMin ) / dy ) + n_ghost;
        pcell_z = (int) floor( ( pos_z - zMin ) / dz ) + n_ghost;
        pcell_index = pcell_x + pcell_y*nx_g + pcell_z*nx_g*ny_g;

        unsigned int N = 0;
        if ((t - age_dev[gtid]) <= Supernova::SN_ERA) {
          curandStateMRG32k3a_t state = states[gtid];
          N = curand_poisson (&state, Supernova::SNR * mass_dev[gtid] * dt);
          states[gtid] = state;

          if (N > 0) {
            // first subtract ejected mass from particle
            mass_dev[gtid]   -= N * Supernova::MASS_PER_SN; 
            feedback_energy   = N * Supernova::ENERGY_PER_SN / dV;
            feedback_density  = N * Supernova::MASS_PER_SN / dV;
            n_0 = density[pcell_index] * DENSITY_UNIT / (Supernova::MU*MP);
            feedback_momentum = Supernova::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(N, 0.93) / sqrt(3.0) / dV;
            shell_radius  = Supernova::R_SH * pow(n_0, -0.46) * pow(N, 0.29);
            is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius; 
      
            s_info[FEED_INFO_N*tid] = 1.*N;
            if (is_resolved) s_info[FEED_INFO_N*tid + 1] =  1.0;
            else             s_info[FEED_INFO_N*tid + 2] =  1.0;  

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

            if (!is_resolved) s_info[FEED_INFO_N*tid + 4] = feedback_momentum * dV;
            if (is_resolved) {
              atomicAdd(&density[indx], feedback_density  * delta_x * delta_y * delta_z);
              atomicAdd(&gasEnergy[indx], feedback_energy  * delta_x * delta_y * delta_z);
              atomicAdd(&energy[indx], feedback_energy  * delta_x * delta_y * delta_z);
              s_info[FEED_INFO_N*tid + 3] = feedback_energy  * fabs(delta_x * delta_y * delta_z) * dV;
            } else {
              atomicAdd(&momentum_x[indx], -delta_x * feedback_momentum);
              atomicAdd(&momentum_y[indx], -delta_y * feedback_momentum);
              atomicAdd(&momentum_z[indx], -delta_z * feedback_momentum);
              //s_info[FEED_INFO_N*tid + 4] = (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
            }
            local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

            indx = (indx_x+1) + indx_y*nx_g + indx_z*nx_g*ny_g;
            if (is_resolved) {
              atomicAdd(&density[indx], feedback_density  * (1-delta_x) * delta_y * delta_z);
              atomicAdd(&gasEnergy[indx], feedback_energy  * (1-delta_x) * delta_y * delta_z);
              atomicAdd(&energy[indx], feedback_energy  * (1-delta_x) * delta_y * delta_z);
              s_info[FEED_INFO_N*tid + 3] += feedback_energy  * fabs((1-delta_x) * delta_y * delta_z) * dV;
            } else {
              atomicAdd(&momentum_x[indx],  delta_x * feedback_momentum);
              atomicAdd(&momentum_y[indx], -delta_y * feedback_momentum);
              atomicAdd(&momentum_z[indx], -delta_z * feedback_momentum);
             // s_info[FEED_INFO_N*tid + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
            }
            local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

            indx = indx_x + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
            if (is_resolved) {
              atomicAdd(&density[indx], feedback_density  * delta_x * (1-delta_y) * delta_z);
              atomicAdd(&gasEnergy[indx], feedback_energy  * delta_x * (1-delta_y) * delta_z);
              atomicAdd(&energy[indx], feedback_energy  * delta_x * (1-delta_y) * delta_z);
              s_info[FEED_INFO_N*tid + 3] += feedback_energy  * fabs(delta_x * (1-delta_y )* delta_z) * dV;
            } else {
              atomicAdd(&momentum_x[indx], -delta_x * feedback_momentum);
              atomicAdd(&momentum_y[indx],  delta_y * feedback_momentum);
              atomicAdd(&momentum_z[indx],  -delta_z * feedback_momentum);
              //s_info[FEED_INFO_N*tid + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
            }
            local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

            indx = indx_x + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
            if (is_resolved) {
              atomicAdd(&density[indx], feedback_density  * delta_x * delta_y * (1-delta_z));
              atomicAdd(&gasEnergy[indx], feedback_energy  * delta_x * delta_y * (1-delta_z));
              atomicAdd(&energy[indx], feedback_energy  * delta_x * delta_y * (1-delta_z));
              s_info[FEED_INFO_N*tid + 3] += feedback_energy  * fabs(delta_x * delta_y * (1 - delta_z)) * dV;
            } else {
              atomicAdd(&momentum_x[indx], -delta_x * feedback_momentum);
              atomicAdd(&momentum_y[indx], -delta_y * feedback_momentum);
              atomicAdd(&momentum_z[indx],  delta_z * feedback_momentum); 
              //s_info[FEED_INFO_N*tid + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
            }
            local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

            indx = (indx_x+1) + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
            if (is_resolved) {
              atomicAdd(&density[indx], feedback_density  * (1-delta_x) * (1-delta_y) * delta_z);
              atomicAdd(&gasEnergy[indx], feedback_energy  * (1-delta_x) * (1-delta_y) * delta_z);
              atomicAdd(&energy[indx], feedback_energy  * (1-delta_x) * (1-delta_y) * delta_z);
              s_info[FEED_INFO_N*tid + 3] += feedback_energy  * fabs((1-delta_x) * (1-delta_y) * delta_z) * dV;
            } else {
              atomicAdd(&momentum_x[indx], delta_x * feedback_momentum);
              atomicAdd(&momentum_y[indx],  delta_y * feedback_momentum);
              atomicAdd(&momentum_z[indx], -delta_z * feedback_momentum);
              //s_info[FEED_INFO_N*tid + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
            }
            local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

            indx = (indx_x+1) + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
            if (is_resolved) {
              atomicAdd(&density[indx], feedback_density  * (1-delta_x) * delta_y * (1-delta_z));
              atomicAdd(&gasEnergy[indx], feedback_energy  * (1-delta_x) * delta_y * (1-delta_z));
              atomicAdd(&energy[indx], feedback_energy  * (1-delta_x) * delta_y * (1-delta_z));
              s_info[FEED_INFO_N*tid + 3] += feedback_energy  * fabs((1-delta_x) * delta_y * (1-delta_z)) * dV;
            } else {
              atomicAdd(&momentum_x[indx],  delta_x * feedback_momentum);
              atomicAdd(&momentum_y[indx], -delta_y * feedback_momentum);
              atomicAdd(&momentum_z[indx],  delta_z * feedback_momentum);
              //s_info[FEED_INFO_N*tid + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
            }
            local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

            indx = indx_x + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
            if (is_resolved) {
              atomicAdd(&density[indx], feedback_density  * delta_x * (1-delta_y) * (1-delta_z));
              atomicAdd(&gasEnergy[indx], feedback_energy  * delta_x * (1-delta_y) * (1-delta_z));
              atomicAdd(&energy[indx], feedback_energy  * delta_x * (1-delta_y) * (1-delta_z));
              s_info[FEED_INFO_N*tid + 3] += feedback_energy * fabs(delta_x * (1-delta_y) * (1-delta_z)) * dV;
            } else {
              atomicAdd(&momentum_x[indx], -delta_x * feedback_momentum);
              atomicAdd(&momentum_y[indx],  delta_y * feedback_momentum);
              atomicAdd(&momentum_z[indx],  delta_z * feedback_momentum);
              //s_info[FEED_INFO_N*tid + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
            }
            local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));

            indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
            if (is_resolved) {
              atomicAdd(&density[indx], feedback_density * (1-delta_x) * (1-delta_y) * (1-delta_z));
              atomicAdd(&gasEnergy[indx], feedback_energy * (1-delta_x) * (1-delta_y) * (1-delta_z));
              atomicAdd(&energy[indx], feedback_energy * (1-delta_x) * (1-delta_y) * (1-delta_z));
              s_info[FEED_INFO_N*tid + 3] += feedback_energy * fabs((1-delta_x) * (1-delta_y) * (1-delta_z)) * dV;
            } else {
              atomicAdd(&momentum_x[indx],  delta_x * feedback_momentum);
              atomicAdd(&momentum_y[indx],  delta_y * feedback_momentum);
              atomicAdd(&momentum_z[indx],  delta_z * feedback_momentum);
              //s_info[FEED_INFO_N*tid + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
            }
            local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));
            atomicMax(dti, local_dti);
          } 
        }
      }
    }

    __syncthreads();

    //reduce the info from all the threads in the block
    for (unsigned int s = blockDim.x/2; s > 0; s>>=1) {
      if(tid < s)  {
        s_info[FEED_INFO_N*tid]     += s_info[FEED_INFO_N*(tid + s)];
        s_info[FEED_INFO_N*tid + 1] += s_info[FEED_INFO_N*(tid + s) + 1];
        s_info[FEED_INFO_N*tid + 2] += s_info[FEED_INFO_N*(tid + s) + 2];
        s_info[FEED_INFO_N*tid + 3] += s_info[FEED_INFO_N*(tid + s) + 3];
        s_info[FEED_INFO_N*tid + 4] += s_info[FEED_INFO_N*(tid + s) + 4];
      }
      __syncthreads();
    }

    if (tid == 0) {
      info[FEED_INFO_N*blockIdx.x]     = s_info[0];
      info[FEED_INFO_N*blockIdx.x + 1] = s_info[1];
      info[FEED_INFO_N*blockIdx.x + 2] = s_info[2];
      info[FEED_INFO_N*blockIdx.x + 3] = s_info[3];
      info[FEED_INFO_N*blockIdx.x + 4] = s_info[4];
    }
}



Real Grid3D::Cluster_Feedback_GPU() {
  if (H.dt == 0) return 0.0;

  if (Particles.n_local > Supernova::n_states) {
    printf("ERROR: not enough cuRAND states (%d) for %f local particles\n", Supernova::n_states, Particles.n_local );
    exit(-1);
  }

  Real h_dti = 0.0;
  Real* d_dti;
  CHECK(cudaMalloc(&d_dti, sizeof(Real)));
  CHECK(cudaMemcpy(d_dti, &h_dti, sizeof(Real), cudaMemcpyHostToDevice));
  
  int ngrid = std::ceil((1.*Particles.n_local)/TPB_FEEDBACK);
  Real h_info[5] = {0, 0, 0, 0, 0};
  Real info[5];
  Real* d_info;
  CHECK(cudaMalloc((void**)&d_info,  FEED_INFO_N*ngrid*sizeof(Real)));
  //FIXME info collection only works if ngrid is 1.  The reason being that reduction of 
  // d_info is currently done on each block.  Only the first block reduction 
  // is used 

  hipLaunchKernelGGL(Cluster_Feedback_Kernel, ngrid, TPB_FEEDBACK, 0, 0,  Particles.n_local, Particles.pos_x_dev, Particles.pos_y_dev, Particles.pos_z_dev, 
       Particles.mass_dev, Particles.age_dev, H.xblocal, H.yblocal, H.zblocal, H.domlen_x, H.domlen_y, H.domlen_z, 
       H.dx, H.dy, H.dz, H.nx, H.ny, H.nz, H.n_ghost, H.t, H.dt, d_dti, d_info,
       C.d_density, C.d_GasEnergy, C.d_Energy, C.d_momentum_x, C.d_momentum_y, C.d_momentum_z, gama, Supernova::curandStates);

  CHECK(cudaMemcpy(&h_dti, d_dti, sizeof(Real), cudaMemcpyDeviceToHost)); 
  CHECK(cudaMemcpy(&h_info, d_info, FEED_INFO_N*sizeof(Real), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_dti));
  CHECK(cudaFree(d_info));

  #ifdef MPI_CHOLLA
  MPI_Reduce(&h_info, &info, 5, MPI_CHREAL, MPI_SUM, root, world);
  #else
  info = h_info;
  #endif

  countSN += (int)info[Supernova::SN];
  countResolved += (int)info[Supernova::RESOLVED];
  countUnresolved += (int)info[Supernova::NOT_RESOLVED];
  totalEnergy += info[Supernova::ENERGY];
  totalMomentum += info[Supernova::MOMENTUM];

  Real resolved_ratio = 0.0;
  if (info[Supernova::RESOLVED] > 0 || info[Supernova::NOT_RESOLVED] > 0) {
    resolved_ratio = info[Supernova::RESOLVED]/(info[Supernova::RESOLVED] + info[Supernova::NOT_RESOLVED]);
  }
  Real global_resolved_ratio = 0.0;
  if (countResolved > 0 || countUnresolved > 0) {
    global_resolved_ratio = countResolved / (countResolved + countUnresolved);
  }

  chprintf("iteration %d: number of SN: %d, ratio of resolved %.3e\n", H.n_step, (long)info[Supernova::SN], resolved_ratio);
  chprintf("    this iteration: energy: %.5e erg.  x-momentum: %.5e S.M. km/s\n", 
                info[Supernova::ENERGY]*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT, info[Supernova::MOMENTUM]*VELOCITY_UNIT/1e5);
  chprintf("    cummulative: #SN: %d, ratio of resolved (R: %d, UR: %d) = %.3e\n", (long)countSN, (long)countResolved, (long)countUnresolved, global_resolved_ratio);
  chprintf("    energy: %.5e erg.  Total x-momentum: %.5e S.M. km/s\n", totalEnergy*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT, totalMomentum*VELOCITY_UNIT/1e5);
  

  return h_dti;
}


#endif //FEEDBACK & PARTICLES_GPU

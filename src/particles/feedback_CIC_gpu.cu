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
    curand_init(seed + id, id, 0, &states[id]);
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
  //n_states = 10;
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



__device__ Real frac(int i, Real dx) {
  return (-0.5*i*i -0.5*i + 1 + i*dx)*0.5;
}

__device__ Real d_fr(int i, Real dx) {
  return (dx > 0.5)*i*(1-2*dx) + ((i+1)*dx + 0.5*(i - 1)) -3*(i-1)*(i+1)*(0.5 - dx);
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

    if (gtid < n_local) {
      Real xMax, yMax, zMax;
      xMax = xMin + xLen;
      yMax = yMin + yLen;
      zMax = zMin + zLen;

      Real pos_x, pos_y, pos_z;
      Real cell_center_x, cell_center_y, cell_center_z;
      Real delta_x, delta_y, delta_z;
      Real x_frac, y_frac, z_frac;
      Real px, py, pz, ek, d;
      Real feedback_energy=0, feedback_density=0, feedback_momentum=0, n_0, shell_radius;
      bool is_resolved = false;
      int pcell_x, pcell_y, pcell_z, pcell_index;
      Real dV = dx*dy*dz;
      Real local_dti = 0.0;

      pos_x = pos_x_dev[gtid];
      pos_y = pos_y_dev[gtid];
      pos_z = pos_z_dev[gtid];

      bool in_local = (pos_x >= xMin && pos_x < xMax) &&
                      (pos_y >= yMin && pos_y < yMax) &&
                      (pos_z >= zMin && pos_z < zMax);
      if (!in_local) {
          printf(" Feedback GPU: Particle outside local domain [%f  %f  %f]  [%f %f] [%f %f] [%f %f]\n ", 
                pos_x, pos_y, pos_z, xMin, xMax, yMin, yMax, zMin, zMax);
      }

      int indx_x = (int) floor( ( pos_x - xMin ) / dx );
      int indx_y = (int) floor( ( pos_y - yMin ) / dy );
      int indx_z = (int) floor( ( pos_z - zMin ) / dz );

      bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x > nx_g-2 || indx_y > ny_g-2 || indx_z > nz_g-2;
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
            mass_dev[gtid]   -= N * Supernova::MASS_PER_SN; 
            feedback_energy   = N * Supernova::ENERGY_PER_SN / dV;
            feedback_density  = N * Supernova::MASS_PER_SN / dV;
            n_0 = density[pcell_index] * DENSITY_UNIT / (Supernova::MU*MP);
            feedback_momentum = Supernova::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(N, 0.93);
            shell_radius  = Supernova::R_SH * pow(n_0, -0.46) * pow(N, 0.29);
            //printf("  N=%d, shell_rad=%0.4e, n_0=%0.4e\n", N, shell_radius, n_0);
            is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius; 
      
            s_info[FEED_INFO_N*tid] = 1.*N;
            if (is_resolved) s_info[FEED_INFO_N*tid + 1] =  1.0;
            else             s_info[FEED_INFO_N*tid + 2] =  1.0;  

            cell_center_x = xMin + indx_x*dx + 0.5*dx;
            cell_center_y = yMin + indx_y*dy + 0.5*dy;
            cell_center_z = zMin + indx_z*dz + 0.5*dz;


            int indx;

            if (is_resolved) { //if resolved inject energy and density
              s_info[FEED_INFO_N*tid + 3] = feedback_energy *dV;

              indx_x = (int) floor( ( pos_x - xMin - 0.5*dx ) / dx );
              indx_y = (int) floor( ( pos_y - yMin - 0.5*dy ) / dy );
              indx_z = (int) floor( ( pos_z - zMin - 0.5*dz ) / dz );

              cell_center_x = xMin + indx_x*dx + 0.5*dx;
              cell_center_y = yMin + indx_y*dy + 0.5*dy;
              cell_center_z = zMin + indx_z*dz + 0.5*dz;

              delta_x = 1 - ( pos_x - cell_center_x ) / dx;
              delta_y = 1 - ( pos_y - cell_center_y ) / dy;
              delta_z = 1 - ( pos_z - cell_center_z ) / dz;
              indx_x += n_ghost;
              indx_y += n_ghost;
              indx_z += n_ghost;

              for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                  for (int k = 0; k < 2; k++) {
                    indx = (indx_x+i) + (indx_y+j)*nx_g + (indx_z+k)*nx_g*ny_g;

                    // i_frac are the fractions of energy/density to be allocated
                    // to each of the 8 cells.
                    x_frac = i*(1-delta_x) + (1-i)*delta_x;
                    y_frac = j*(1-delta_y) + (1-j)*delta_y;
                    z_frac = k*(1-delta_z) + (1-k)*delta_z;

                    atomicAdd(&density[indx],   x_frac * y_frac * z_frac * feedback_density);
                    atomicAdd(&gasEnergy[indx], x_frac * y_frac * z_frac * feedback_energy );
                    atomicAdd(&energy[indx],    x_frac * y_frac * z_frac * feedback_energy );

                    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));
                  }
                }
              }
            } else {  //if not resolved, inject momentum and density
              s_info[FEED_INFO_N*tid + 4] = feedback_momentum;
              feedback_momentum /= sqrt(3.0);

              delta_x =  ( pos_x - indx_x*dx ) / dx;
              delta_y =  ( pos_y - indx_y*dy ) / dy;
              delta_z =  ( pos_z - indx_z*dz ) / dz;
              indx_x += n_ghost;
              indx_y += n_ghost;
              indx_z += n_ghost;

              for (int i = -1; i < 2; i++) {
                for (int j = -1; j < 2; j++) {
                  for (int k = -1; k < 2; k++) {
                    // index in array of conserved quantities
                    indx = (indx_x+i) + (indx_y+j)*nx_g + (indx_z+k)*nx_g*ny_g;

                    px = d_fr(i, delta_x) * frac(j, delta_y) * frac(k, delta_z) * feedback_momentum;
                    py = frac(i, delta_x) * d_fr(j, delta_y) * frac(k, delta_z) * feedback_momentum;
                    pz = frac(i, delta_x) * frac(j, delta_y) * d_fr(k, delta_z) * feedback_momentum;
                    d  = frac(i, delta_x) * frac(j, delta_y) * frac(k, delta_z) * feedback_density;
                    ek = (px*px + py+py + pz*pz)/2/d;

                    atomicAdd(&momentum_x[indx], px);
                    atomicAdd(&momentum_y[indx], py);
                    atomicAdd(&momentum_z[indx], pz);
                    atomicAdd(   &density[indx], d );
                    atomicAdd(    &energy[indx], ek);

                    local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));
                  }
                }
              }
            }
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

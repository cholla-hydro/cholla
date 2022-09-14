#if defined(SUPERNOVA) && defined(PARTICLES_GPU)

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../grid/grid3D.h"
#include "../global/global_cuda.h"
#include "../global/global.h"
#include "../io/io.h"
#include "supernova.h"

#define TPB_FEEDBACK 256
#define FEED_INFO_N 6
#define i_RES 1
#define i_UNRES 2
#define i_ENERGY 3
#define i_MOMENTUM 4
#define i_UNRES_ENERGY 5

namespace Supernova {
  curandStateMRG32k3a_t*  curandStates;
  part_int_t n_states;
  Real t_buff, dt_buff;
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
  t_buff = 0;
  dt_buff = 0;
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


/** the prescription for dividing a scalar quantity between 3x3x3 cells is done by imagining a
    2x2x2 cell volume around the SN.  These fractions, then, represent the linear extent of this
    volume into the cell in question.
    For i=0 this should be 1*1/2.
    For i=-1 this should be (1-dx)*1/2.
    For i=+1 this should be dx*1/2.
    In the above the 1/2 factor is normalize over 2 cells/direction.
  */
__device__ Real frac(int i, Real dx) {
  return (-0.5*i*i - 0.5*i + 1 + i*dx)*0.5;
}


__device__ Real d_fr(int i, Real dx) {
  return (dx > 0.5)*i*(1-2*dx) + ((i+1)*dx + 0.5*(i-1)) - 3*(i-1)*(i+1)*(0.5 - dx);
}


__device__ Real GetAverageDensity(Real *density, int xi, int yi, int zi, int nxg, int nyg, int ng) {
  Real d_average = 0.0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        d_average +=  density[(xi + ng + i) + (yi + ng + j)*nxg + (zi + ng + k)*nxg*nyg];
      }
    }
  }
  return d_average / 27;
}


__device__ Real GetAverageNumberDensity_CGS(Real *density, int xi, int yi, int zi, int nxg, int nyg, int ng) {
  return GetAverageDensity(density, xi, yi, zi, nxg, nyg, ng) * DENSITY_UNIT / (Supernova::MU*MP);
}


__global__ void Cluster_Feedback_Kernel(part_int_t n_local, part_int_t *id, Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev, 
  Real* mass_dev, Real* age_dev, Real xMin, Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, 
  Real dx, Real dy, Real dz, int nx_g, int ny_g, int nz_g, int n_ghost, Real t, Real dt, Real* dti, Real* info,
  Real* density, Real* gasEnergy, Real* energy, Real* momentum_x, Real* momentum_y, Real* momentum_z, Real gamma, curandStateMRG32k3a_t* states,
  Real* prev_dens, int* prev_N, short direction){

    __shared__ Real s_info[FEED_INFO_N*TPB_FEEDBACK]; // for collecting SN feedback information, like # of SNe or # resolved.
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid ;

    s_info[FEED_INFO_N*tid]     = 0; // number of supernovae
    s_info[FEED_INFO_N*tid + 1] = 0; // number of resolved events
    s_info[FEED_INFO_N*tid + 2] = 0; // number of unresolved events
    s_info[FEED_INFO_N*tid + 3] = 0; // resolved energy
    s_info[FEED_INFO_N*tid + 4] = 0; // unresolved momentum
    s_info[FEED_INFO_N*tid + 5] = 0; // unresolved KE added via momentum injection

    if (gtid < n_local) {
      Real pos_x, pos_y, pos_z;
      Real cell_center_x, cell_center_y, cell_center_z;
      Real delta_x, delta_y, delta_z;
      Real x_frac, y_frac, z_frac;
      Real px, py, pz, d; 
      //Real t_b, t_a, v_1, v_2, d_b, d_a, p_b, p_a, e;
      Real feedback_energy=0, feedback_density=0, feedback_momentum=0, n_0, shell_radius;
      bool is_resolved = false;
      Real dV = dx*dy*dz;
      Real local_dti = 0.0;

      pos_x = pos_x_dev[gtid];
      pos_y = pos_y_dev[gtid];
      pos_z = pos_z_dev[gtid];
      //printf("(%d): pos:(%.4e, %.4e, %.4e)\n", gtid, pos_x, pos_y, pos_z);
      //printf("(%d): MIN:(%.4e, %.4e, %.4e)\n", gtid, xMin, yMin, xMin);


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
      //printf("(%d): indx:(%d, %d, %d)\n", gtid, indx_x, indx_y, indx_z);


      bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x >= nx_g-2*n_ghost || indx_y >= ny_g-2*n_ghost || indx_z >= nz_g-2*n_ghost;
      if (ignore) {
          printf(" Feedback GPU: Particle CIC index err [%f  %f  %f]  [%d %d %d] [%d %d %d] \n ", 
                pos_x, pos_y, pos_z, indx_x, indx_y, indx_z, nx_g, ny_g, nz_g);
      }

      if (!ignore && in_local) {

        int N = 0;
        if ((t - age_dev[gtid]) <= Supernova::SN_ERA) {
          if (direction == -1) N = -prev_N[gtid]; 
          else {
            curandStateMRG32k3a_t state = states[gtid];
            N = curand_poisson (&state, Supernova::SNR * mass_dev[gtid] * dt);
            states[gtid] = state;
            prev_N[gtid] = N;
          }
          if (N != 0) {
            mass_dev[gtid]   -= N * Supernova::MASS_PER_SN; 
            feedback_energy   = N * Supernova::ENERGY_PER_SN / dV;
            feedback_density  = N * Supernova::MASS_PER_SN / dV;
            if (direction == -1) n_0 = prev_dens[gtid];
            else {
              n_0 = GetAverageNumberDensity_CGS(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
              prev_dens[gtid] = n_0;
            }
            //int devcount;
            //cudaGetDeviceCount(&devcount);
            //int devId;
            //cudaGetDevice(&devId);
            //printf("[%d: %d] N: %d, time: %.4e, dt: %.4e, e: %.4e, n_0: %.4e\n", devId, gtid, N, t, dt, feedback_energy, n_0);

            feedback_momentum = direction*Supernova::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(fabsf(N), 0.93) / dV;
            shell_radius  = Supernova::R_SH * pow(n_0, -0.46) * pow(fabsf(N), 0.29);
            is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius; 
            if (!is_resolved) printf("UR[%f] at (%d, %d, %d)  id=%d, N=%d, shell_rad=%0.4e, n_0=%0.4e\n",
                                        t, indx_x + n_ghost, indx_y + n_ghost, indx_z + n_ghost, (int)id[gtid], N, shell_radius, n_0);
      
            s_info[FEED_INFO_N*tid] = 1.*N;
            if (is_resolved) s_info[FEED_INFO_N*tid + 1] =  direction * 1.0;
            else             s_info[FEED_INFO_N*tid + 2] =  direction * 1.0;  

            int indx;

            if (is_resolved) { //if resolved inject energy and density
              s_info[FEED_INFO_N*tid + 3] = feedback_energy * dV;

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

                    if (abs(momentum_x[indx]/density[indx]) >= C_L) {
                      printf("%d, Rb: (%d, %d, %d) vx = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_x[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    if (abs(momentum_y[indx]/density[indx]) >= C_L) {
                      printf("%d, Rb: (%d, %d, %d) vy = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_y[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    if (abs(momentum_z[indx]/density[indx]) >= C_L) {
                      printf("%d, Rb: (%d, %d, %d) vz = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_z[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }

                    // i_frac are the fractions of energy/density to be allocated
                    // to each of the 8 cells.
                    x_frac = i*(1-delta_x) + (1-i)*delta_x;
                    y_frac = j*(1-delta_y) + (1-j)*delta_y;
                    z_frac = k*(1-delta_z) + (1-k)*delta_z;

                    atomicAdd(&density[indx],   x_frac * y_frac * z_frac * feedback_density);
                    atomicAdd(&gasEnergy[indx], x_frac * y_frac * z_frac * feedback_energy );
                    atomicAdd(&energy[indx],    x_frac * y_frac * z_frac * feedback_energy );

                    if (abs(momentum_x[indx]/density[indx]) >= C_L) {
                      printf("%d, Ra: (%d, %d, %d) vx = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x  + i, indx_y + j, indx_z + k,
                             momentum_x[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    if (abs(momentum_y[indx]/density[indx]) >= C_L) {
                      printf("%d, Ra: (%d, %d, %d) vy = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z  + k,
                             momentum_y[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    if (abs(momentum_z[indx]/density[indx]) >= C_L) {
                      printf("%d, Ra: (%d, %d, %d) vz = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_z[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }

                    if (direction > 0) local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));
                  }
                }
              }
            } else {  //if not resolved, inject momentum and density
              s_info[FEED_INFO_N*tid + 4] = feedback_momentum * dV;

              delta_x =  ( pos_x - xMin - indx_x*dx ) / dx;
              delta_y =  ( pos_y - yMin - indx_y*dy ) / dy;
              delta_z =  ( pos_z - zMin - indx_z*dz ) / dz;
              //printf("(%d):indx:(%d, %d, %d)\n", gtid, indx_x, indx_y, indx_z);
              //printf("(%d): pos:(%.4e, %.4e, %.4e), delta_x (%.2e, %.2e, %.2e)\n", gtid, pos_x, pos_y, pos_z, delta_x, delta_y, delta_z);

              indx_x += n_ghost;
              indx_y += n_ghost;
              indx_z += n_ghost;

              if (abs(feedback_momentum/feedback_density*VELOCITY_UNIT*1e-5) > 40000) {  // injected speeds are greater than 4e4 km/s
                printf("**** (%d, %d, %d) injected speeds are %.3e km/s\n", indx_x, indx_y, indx_z, feedback_momentum/feedback_density*VELOCITY_UNIT*1e-5);
              }
              feedback_momentum /= sqrt(3.0);

              for (int i = -1; i < 2; i++) {
                for (int j = -1; j < 2; j++) {
                  for (int k = -1; k < 2; k++) {
                    // index in array of conserved quantities
                    indx = (indx_x+i) + (indx_y+j)*nx_g + (indx_z+k)*nx_g*ny_g;

                    x_frac = d_fr(i, delta_x) * frac(j, delta_y) * frac(k, delta_z);
                    y_frac = frac(i, delta_x) * d_fr(j, delta_y) * frac(k, delta_z);
                    z_frac = frac(i, delta_x) * frac(j, delta_y) * d_fr(k, delta_z);

                    px =  x_frac * feedback_momentum;
                    py =  y_frac * feedback_momentum;
                    pz =  z_frac * feedback_momentum;
                    //d = (abs(x_frac) + abs(y_frac) + abs(z_frac)) / 6 * (feedback_density + n_0*Supernova::MU*MP/DENSITY_UNIT);
                    d = (abs(x_frac) + abs(y_frac) + abs(z_frac)) / 6 * feedback_density + n_0*Supernova::MU*MP/DENSITY_UNIT;

                    //d  = frac(i, delta_x) * frac(j, delta_y) * frac(k, delta_z) * feedback_density;
                    //e  = frac(i, delta_x) * frac(j, delta_y) * frac(k, delta_z) * feedback_energy;
                    //printf("(%d, %d, %d): delta:(%.4e, %.4e, %.4e), frac: %.4e\n", indx_x, indx_y, indx_z, delta_x, delta_y, delta_z, frac(i, delta_x)*frac(j, delta_y)*frac(k, delta_z));
                    //printf("(%d, %d, %d):(%d SN) (i:%d, j:%d, k:%d) before: %.4e\n", indx_x, indx_y, indx_z, N, i, j, k, density[indx]*DENSITY_UNIT/0.6/MP);


                    //v_1 = sqrt((momentum_x[indx]*momentum_x[indx] + momentum_y[indx]*momentum_y[indx] + momentum_z[indx]*momentum_z[indx])/density[indx]/density[indx])*VELOCITY_UNIT/1e5;
                    //t_b = gasEnergy[indx]*ENERGY_UNIT*(gamma - 1)/(density[indx]*DENSITY_UNIT/0.6/MP*KB);
                    //p_b = sqrt(momentum_x[indx]*momentum_x[indx] + momentum_y[indx]*momentum_y[indx] + momentum_z[indx]*momentum_z[indx])*VELOCITY_UNIT/1e5;
                    //d_b = density[indx]*DENSITY_UNIT/0.6/MP;

                    if (abs(momentum_x[indx]/density[indx]) >= C_L) {
                      printf("%d, Ub: (%d, %d, %d) vx = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_x[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    if (abs(momentum_y[indx]/density[indx]) >= C_L) {
                      printf("%d, Ub: (%d, %d, %d) vy = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_y[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    if (abs(momentum_z[indx]/density[indx]) >= C_L) {
                      printf("%d, Ub: (%d, %d, %d) vz = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_z[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }

                    atomicAdd(&momentum_x[indx], px);
                    atomicAdd(&momentum_y[indx], py);
                    atomicAdd(&momentum_z[indx], pz);
                    density[indx] = d;
                    energy[indx] = (momentum_x[indx]*momentum_x[indx] + momentum_y[indx]*momentum_y[indx] + momentum_z[indx]*momentum_z[indx])/2/density[indx] + gasEnergy[indx];


                    // atomicAdd(    &energy[indx], e );
                    //atomicAdd(   &density[indx], d );

                    s_info[FEED_INFO_N*tid + i_UNRES_ENERGY] += direction*(px*px + py*py + pz*pz)/2/density[indx]*dV;

                    if (abs(momentum_x[indx]/density[indx]) >= C_L) {
                      printf("%d, Ua: (%d, %d, %d) vx = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_x[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    if (abs(momentum_y[indx]/density[indx]) >= C_L) {
                      printf("%d, Ua: (%d, %d, %d) vy = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_y[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    if (abs(momentum_z[indx]/density[indx]) >= C_L) {
                      printf("%d, Ua: (%d, %d, %d) vz = %.3e, d = %.3e, n_0 = %.3e\n", direction, indx_x + i, indx_y + j, indx_z + k,
                             momentum_z[indx]/density[indx]*VELOCITY_UNIT*1e-5, density[indx]*DENSITY_UNIT/0.6/MP, n_0);
                    }
                    //gasEnergy[indx] = energy[indx] - (momentum_x[indx]*momentum_x[indx] + momentum_y[indx]*momentum_y[indx] + momentum_z[indx]*momentum_z[indx])/2/density[indx];
                    //v_2 = sqrt((momentum_x[indx]*momentum_x[indx] + momentum_y[indx]*momentum_y[indx] + momentum_z[indx]*momentum_z[indx])/density[indx]/density[indx]) * VELOCITY_UNIT/1e5;
                    //t_a = gasEnergy[indx]*ENERGY_UNIT*(gamma - 1)/(density[indx]*DENSITY_UNIT/0.6/MP*KB);
                    //d_a = density[indx]*DENSITY_UNIT/0.6/MP;
                    //p_a = sqrt(momentum_x[indx]*momentum_x[indx] + momentum_y[indx]*momentum_y[indx] + momentum_z[indx]*momentum_z[indx])*VELOCITY_UNIT/1e5;


                    //printf("(%d, %d, %d):(CM: %.2e, SN: %d) (i:%d, j:%d, k:%d) v_1: %.5e v_2: %.5e   V_DIFF-> %.4f %%\n", indx_x, indx_y, indx_z, mass_dev[gtid], N, i, j, k, v_1, v_2, (v_2-v_1)/v_1*100);
                    //printf("   (%d, %d, %d):(%d SN) (i:%d, j:%d, k:%d) T_b: %.5e T_a: %.5e   T_DIFF-> %.4f %%\n", indx_x, indx_y, indx_z, N, i, j, k, t_b, t_a, (t_a-t_b)/t_b*100);
                    //printf("      (%d, %d, %d):(%d SN) (i:%d, j:%d, k:%d) d_b: %.5e d_a: %.5e   D_DIFF-> %.1f %%\n", indx_x, indx_y, indx_z, N, i, j, k, d_b, d_a, (d_a-d_b)/d_b*100);
                    //printf("         (%d, %d, %d):(%d SN) (i:%d, j:%d, k:%d) p_b: %.5e p_a: %.5e   P_DIFF-> %.4f %%\n", indx_x, indx_y, indx_z, N, i, j, k, p_b, p_a, (p_a-p_b)/p_b*100);

                    if (direction > 0) {
                      //printf("urs time:%.3e id:%d N:%d d:%.5e\n", t, id[gtid], N, n_0);
                      local_dti = fmax(local_dti, Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz));
                    }
                  }
                }
              }
            }
            if (direction > 0) atomicMax(dti, local_dti);
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
        s_info[FEED_INFO_N*tid + 5] += s_info[FEED_INFO_N*(tid + s) + 5];
      }
      __syncthreads();
    }

    if (tid == 0) {
      info[FEED_INFO_N*blockIdx.x]     = s_info[0];
      info[FEED_INFO_N*blockIdx.x + 1] = s_info[1];
      info[FEED_INFO_N*blockIdx.x + 2] = s_info[2];
      info[FEED_INFO_N*blockIdx.x + 3] = s_info[3];
      info[FEED_INFO_N*blockIdx.x + 4] = s_info[4];
      info[FEED_INFO_N*blockIdx.x + 5] = s_info[5];
    }
}



Real Grid3D::Cluster_Feedback_GPU() {
  if (H.dt == 0) return 0.0;

  if (Particles.n_local > Supernova::n_states) {
    printf("ERROR: not enough cuRAND states (%d) for %f local particles\n", Supernova::n_states, Particles.n_local );
    exit(-1);
  }

  Real h_dti = 0.0;
  int direction, ngrid;
  Real h_info[6] = {0, 0, 0, 0, 0, 0};
  Real info[6];
  Real *d_dti, *d_info;
  // require d_prev_dens & d_prev_N in case we have to undo feedback if the time step is too large.
  Real* d_prev_dens;
  int*  d_prev_N;


  if (Particles.n_local > 0) {
    CHECK(cudaMalloc(&d_dti, sizeof(Real)));
    CHECK(cudaMemcpy(d_dti, &h_dti, sizeof(Real), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_prev_dens, Particles.n_local*sizeof(Real)));
    CHECK(cudaMalloc(&d_prev_N, Particles.n_local*sizeof(int)));
    CHECK(cudaMemset(d_prev_dens, 0, Particles.n_local*sizeof(Real)));
    CHECK(cudaMemset(d_prev_N, 0, Particles.n_local*sizeof(int)));  
  
    ngrid = std::ceil((1.*Particles.n_local)/TPB_FEEDBACK);
    CHECK(cudaMalloc((void**)&d_info,  FEED_INFO_N*ngrid*sizeof(Real)));
  }
  //FIXME info collection and max dti calculation 
  // only works if ngrid is 1.  The reason being that reduction of 
  // d_info is currently done on each block.  Only the first block reduction 
  // is used 

  do {
    direction = 1; 
    if (Particles.n_local > 0) {
      hipLaunchKernelGGL(Cluster_Feedback_Kernel, ngrid, TPB_FEEDBACK, 0, 0,  Particles.n_local, Particles.partIDs_dev, Particles.pos_x_dev, Particles.pos_y_dev, Particles.pos_z_dev,
         Particles.mass_dev, Particles.age_dev, H.xblocal, H.yblocal, H.zblocal, H.xblocal_max, H.yblocal_max, H.zblocal_max, 
         H.dx, H.dy, H.dz, H.nx, H.ny, H.nz, H.n_ghost, H.t, H.dt, d_dti, d_info,
         C.d_density, C.d_GasEnergy, C.d_Energy, C.d_momentum_x, C.d_momentum_y, C.d_momentum_z, gama, Supernova::curandStates, d_prev_dens, d_prev_N, direction);

      CHECK(cudaMemcpy(&h_dti, d_dti, sizeof(Real), cudaMemcpyDeviceToHost));
    }

    #ifdef MPI_CHOLLA
    h_dti = ReduceRealMax(h_dti);
    MPI_Barrier(world);
    #endif // MPI_CHOLLA

    if (h_dti != 0 && (C_cfl/h_dti < H.dt)) {  // timestep too big: need to undo the last operation
      direction = -1;
      if (Particles.n_local > 0) {
        hipLaunchKernelGGL(Cluster_Feedback_Kernel, ngrid, TPB_FEEDBACK, 0, 0,  Particles.n_local, Particles.partIDs_dev, Particles.pos_x_dev, Particles.pos_y_dev, Particles.pos_z_dev,
          Particles.mass_dev, Particles.age_dev, H.xblocal, H.yblocal, H.zblocal, H.xblocal_max, H.yblocal_max, H.zblocal_max, 
          H.dx, H.dy, H.dz, H.nx, H.ny, H.nz, H.n_ghost, H.t, H.dt, d_dti, d_info,
          C.d_density, C.d_GasEnergy, C.d_Energy, C.d_momentum_x, C.d_momentum_y, C.d_momentum_z, gama, Supernova::curandStates, d_prev_dens, d_prev_N, direction);
     
        CHECK(cudaDeviceSynchronize()); 
      }
      H.dt = C_cfl/h_dti;
    }

  } while (direction == -1);

  if (Particles.n_local > 0) {
    CHECK(cudaMemcpy(&h_info, d_info, FEED_INFO_N*sizeof(Real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_dti));
    CHECK(cudaFree(d_info));
    CHECK(cudaFree(d_prev_dens));
    CHECK(cudaFree(d_prev_N));
  }

  #ifdef MPI_CHOLLA
  MPI_Reduce(&h_info, &info, FEED_INFO_N, MPI_CHREAL, MPI_SUM, root, world);
  #else
  info = h_info;
  #endif

  countSN += (int)info[Supernova::SN];
  countResolved += (int)info[Supernova::RESOLVED];
  countUnresolved += (int)info[Supernova::NOT_RESOLVED];
  totalEnergy += info[Supernova::ENERGY];
  totalMomentum += info[Supernova::MOMENTUM];
  totalUnresEnergy += info[Supernova::UNRES_ENERGY];

  Real resolved_ratio = 0.0;
  if (info[Supernova::RESOLVED] > 0 || info[Supernova::NOT_RESOLVED] > 0) {
    resolved_ratio = info[Supernova::RESOLVED]/(info[Supernova::RESOLVED] + info[Supernova::NOT_RESOLVED]);
  }
  Real global_resolved_ratio = 0.0;
  if (countResolved > 0 || countUnresolved > 0) {
    global_resolved_ratio = countResolved / (countResolved + countUnresolved);
  }

  chprintf("iteration %d: number of SN: %d, ratio of resolved %.3e\n", H.n_step, (long)info[Supernova::SN], resolved_ratio);
  chprintf("    this iteration: energy: %.5e erg.  momentum: %.5e S.M. km/s  unres_energy: %.5e erg\n", 
                info[Supernova::ENERGY]*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT, info[Supernova::MOMENTUM]*VELOCITY_UNIT/1e5,
                info[Supernova::UNRES_ENERGY]*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT);
  chprintf("    cummulative: #SN: %d, ratio of resolved (R: %d, UR: %d) = %.3e\n", (long)countSN, (long)countResolved, (long)countUnresolved, global_resolved_ratio);
  chprintf("    energy: %.5e erg.  Total momentum: %.5e S.M. km/s, Total unres energy: %.5e\n", totalEnergy*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT, 
                    totalMomentum*VELOCITY_UNIT/1e5, totalUnresEnergy*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT);
  

  return h_dti;
}


#endif //SUPERNOVA & PARTICLES_GPU

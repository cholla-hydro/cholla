#ifdef PARTICLES
#ifdef DE
#ifdef PARTICLE_AGE
#ifdef FEEDBACK

#include <iostream>
#include <cstring>
#include "feedback_CIC.h"
#include "particles_3D.h"
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../global/global.h"
#include "supernova.h"
#include <random>
#include <tuple>
#include <math.h>
#include <algorithm>

#ifdef MPI_CHOLLA
#include "../mpi/mpi_routines.h"
#endif

#ifdef PARALLEL_OMP
#include"../utils/parallel_omp.h"
#endif


std::random_device rd;
//std::mt19937_64 gen(rd());
std::mt19937_64 generator(42);   //FIXME read this in from init params or ChollaPrngGenerator
/*
void Supernova::initState(struct parameters *P) {
    generator.seed(P->prng_seed);
}*/


std::tuple<int, Real, Real, Real, Real> getClusterFeedback(Real t, Real dt, Real mass, Real age, Real density) {
    int N = 0;
    if (t - age <= 1.0e4) {
       std::poisson_distribution<int> distribution(Supernova::SNR * mass * dt);
       N = distribution(generator);
    }
    Real n_0 = density * DENSITY_UNIT / (Supernova::MU*MP);  // in cm^{-3}
    //std::cout << "n_0 is " << n_0 << std::endl;
    //if (N > 0)  std::cout << "MOMENTUM: " << FINAL_MOMENTUM * pow(n_0, -0.17) * pow(N, 0.93) * VELOCITY_UNIT/1e10 << std::endl;

    return { /* number of SN */           N,
             /* total energy given off */ N * Supernova::ENERGY_PER_SN,
             /* total mass */             N * Supernova::MASS_PER_SN,
             /* final momentum */         Supernova::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(N, 0.93),
             /* shell formation radius */ Supernova::R_SH * pow(n_0, -0.46) * pow(N, 0.29)
           };
}


Real Grid3D::Cluster_Feedback() {
  #ifdef CPU_TIME
  Timer.Feedback.Start();
  #endif

  Real max_sn_dti = 0;
  #ifdef PARTICLES_GPU
  max_sn_dti = Cluster_Feedback_GPU();
  #ifdef MPI_CHOLLA
  max_sn_dti = ReduceRealMax(max_sn_dti);
  #endif // MPI_CHOLLA
  #else
  Real* feedbackInfo;
  Real* thread_dti;
  int totalThreads = 1;
  Real partiallyReducedInfo[N_INFO] = {0, 0, 0, 0, 0};
  Real reducedInfo[N_INFO] = {0, 0, 0, 0, 0};
  const int SN = 0, RESOLVED = 1, NOT_RESOLVED = 2, ENERGY = 3, MOMENTUM = 4;

  #ifndef PARALLEL_OMP

  feedbackInfo = (Real*)calloc(N_INFO, sizeof(Real));
  sn_thread_dti = (Real*)calloc(1, sizeof(Real));
  Cluster_Feedback_Function( 0, Particles.n_local, feedbackInfo, 0, thread_dti);
 
  #else

  totalThreads = N_OMP_THREADS;
  feedbackInfo = (Real*)calloc(N_INFO*totalThreads, sizeof(Real));
  thread_dti = (Real*)calloc(totalThreads, sizeof(Real));
  // malloc array of size N_OMP_THREADS to take the feedback info
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();

    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );
    Cluster_Feedback_Function( p_start, p_end, feedbackInfo, omp_id, thread_dti);
  }  
  #endif //PARALLEL_OMP

  for (int i = 0; i < totalThreads; i++) {
    partiallyReducedInfo[SN] += feedbackInfo[i*N_INFO + SN];
    partiallyReducedInfo[RESOLVED] += feedbackInfo[i*N_INFO + RESOLVED];
    partiallyReducedInfo[NOT_RESOLVED] += feedbackInfo[i*N_INFO + NOT_RESOLVED];
    partiallyReducedInfo[ENERGY] += feedbackInfo[i*N_INFO + ENERGY];
    partiallyReducedInfo[MOMENTUM] += feedbackInfo[i*N_INFO + MOMENTUM];
    max_sn_dti = fmax(max_sn_dti, thread_dti[i]);
  }

  #ifdef MPI_CHOLLA
  max_sn_dti = ReduceRealMax(max_sn_dti);
  MPI_Reduce(&partiallyReducedInfo, &reducedInfo, N_INFO, MPI_CHREAL, MPI_SUM, root, world);
  if (procID==root) {
  #else 
    memcpy(reducedInfo, partiallyReducedInfo, sizeof(partiallyReducedInfo));
  #endif //MPI_CHOLLA

    countSN += reducedInfo[SN];
    countResolved += reducedInfo[RESOLVED];
    countUnresolved += reducedInfo[NOT_RESOLVED];
    totalEnergy += reducedInfo[ENERGY];
    totalMomentum += reducedInfo[MOMENTUM];

    Real resolved_ratio = 0.0;
    if (reducedInfo[RESOLVED] > 0 || reducedInfo[NOT_RESOLVED] > 0) {
      resolved_ratio = reducedInfo[RESOLVED]*1.0/(reducedInfo[RESOLVED] + reducedInfo[NOT_RESOLVED]);
    }
    Real global_resolved_ratio = 0.0;
    if (countResolved > 0 || countUnresolved > 0) {
      global_resolved_ratio = countResolved / (countResolved + countUnresolved);
    }
    /*chprintf("iteration %d: number of SN: %d, ratio of resolved %f\n", H.n_step, (long)reducedInfo[SN], resolved_ratio);
    chprintf("    this iteration: energy: %e erg.  x-momentum: %e S.M. km/s\n", 
             reducedInfo[ENERGY]*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT, reducedInfo[MOMENTUM]*VELOCITY_UNIT/1e5);
    chprintf("    cummulative: #SN: %d, ratio of resolved (R: %d, UR: %d) = %f\n", (long)countSN, (long)countResolved, (long)countUnresolved, global_resolved_ratio);
    chprintf("    energy: %e erg.  Total x-momentum: %e S.M. km/s\n", totalEnergy*MASS_UNIT*LENGTH_UNIT*LENGTH_UNIT/TIME_UNIT/TIME_UNIT, totalMomentum*VELOCITY_UNIT/1e5);
    */

  #ifdef MPI_CHOLLA
  }
  #endif /*MPI_CHOLLA*/

  free(feedbackInfo);
  free(thread_dti);

  #endif //PARTICLES_GPU

  #ifdef CPU_TIME
  Timer.Feedback.End();
  #endif

  return max_sn_dti;
}


// returns the largest 1/dt for the cell with the given index
Real Grid3D::Calc_Timestep(int index) {
  Real density = fmax(C.density[index], DENS_FLOOR);
  Real vx = C.momentum_x[index] / density;
  Real vy = C.momentum_y[index] / density;
  Real vz = C.momentum_z[index] / density;
  Real cs = sqrt(gama * fmax( (C.Energy[index]- 0.5*density*(vx*vx + vy*vy + vz*vz))*(gama-1.0), TINY_NUMBER ) / density);
  return fmax( fmax((fabs(vx) + cs)/H.dx, (fabs(vy) + cs)/H.dy), (fabs(vz) + cs)/H.dz ) ;
}


//Compute the CIC feedback
void Grid3D::Cluster_Feedback_Function(part_int_t p_start, part_int_t p_end, Real* info, int threadId, Real* max_dti) {
  #ifdef PARTICLES_CPU
  int nx_g, ny_g, nz_g;
  nx_g = H.nx;
  ny_g = H.ny;
  nz_g = H.nz;

  Real xMin, yMin, zMin;
  xMin = H.xblocal;  //TODO: make sure this is correct (and not H.xbound) (local min vs. global min)
  yMin = H.yblocal;
  zMin = H.zblocal;
 
  part_int_t pIndx;
  int indx_x, indx_y, indx_z, indx;
  int pcell_x, pcell_y, pcell_z, pcell_index;
  Real pos_x, pos_y, pos_z;
  Real cell_center_x, cell_center_y, cell_center_z;
  Real delta_x, delta_y, delta_z;
  Real dV = H.dx*H.dy*H.dz;
  Real feedback_energy, feedback_density, feedback_momentum;
  bool ignore, in_local, is_resolved;

  for ( pIndx=p_start; pIndx < p_end; pIndx++ ){
    pos_x = Particles.pos_x[pIndx];
    pos_y = Particles.pos_y[pIndx];
    pos_z = Particles.pos_z[pIndx];

    pcell_x = (int) floor( ( pos_x - xMin ) / H.dx ) + H.n_ghost;
    pcell_y = (int) floor( ( pos_y - yMin ) / H.dy ) + H.n_ghost;
    pcell_z = (int) floor( ( pos_z - zMin ) / H.dz ) + H.n_ghost;
    pcell_index = pcell_x + pcell_y*nx_g + pcell_z*nx_g*ny_g;

    auto [N, energy, mass, momentum, r_sf] = getClusterFeedback(H.t, H.dt, Particles.mass[pIndx], Particles.age[pIndx], C.density[pcell_index]);
    if (N == 0) continue;

    Particles.mass[pIndx] -= mass;
    feedback_energy = energy / dV;
    feedback_density = mass / dV;
    feedback_momentum = momentum / sqrt(3) / dV;
    is_resolved = 3 * std::max({H.dx, H.dy, H.dz}) <= r_sf; 
    // now fill in 'info' for logging
    info[threadId*N_INFO] += N*1.0;
    if (is_resolved) info[threadId*N_INFO + 1] += 1.0;
    else info[threadId*N_INFO + 2] += 1.0;

    indx_x = (int) floor( ( pos_x - xMin - 0.5*H.dx ) / H.dx );
    indx_y = (int) floor( ( pos_y - yMin - 0.5*H.dy ) / H.dy );
    indx_z = (int) floor( ( pos_z - zMin - 0.5*H.dz ) / H.dz );

    in_local = (pos_x >= H.xblocal && pos_x < H.xblocal + H.domlen_x) &&
               (pos_y >= H.yblocal && pos_y < H.yblocal + H.domlen_y) &&
               (pos_z >= H.zblocal && pos_z < H.zblocal + H.domlen_z);
    if (!in_local) {
      std::cout << " Cluster_FeedbackError:" << std::endl;
      #ifdef PARTICLE_IDS
      std::cout << " Particle outside local  domain    pID: " << Particles.partIDs[pIndx] << std::endl;
      #else
      std::cout << " Particle outside local  domain " << std::endl;
      #endif
      std::cout << "  Domain X: " << xMin <<  "  " << H.xblocal + H.domlen_x << std::endl;
      std::cout << "  Domain Y: " << yMin <<  "  " << H.yblocal + H.domlen_y << std::endl;
      std::cout << "  Domain Z: " << zMin <<  "  " << H.zblocal + H.domlen_z << std::endl;
      std::cout << "  Particle X: " << pos_x << std::endl;
      std::cout << "  Particle Y: " << pos_y << std::endl;
      std::cout << "  Particle Z: " << pos_z << std::endl;
      continue;
    }

    ignore = indx_x < -1 || indx_y < -1 || indx_z < -1 || indx_x > nx_g-3 || indx_y > ny_g-3 || indx_y > nz_g-3;
    if (ignore){
      #ifdef PARTICLE_IDS
      std::cout << "ERROR Cluster_Feedback Index    pID: " << Particles.partIDs[pIndx] << std::endl;
      #else
      std::cout << "ERROR Cluster_Feedback Index " << std::endl;
      #endif
      std::cout << "xIndx: " << pos_x << "  " << indx_x << std::endl;
      std::cout << "zIndx: " << pos_y << "  " << indx_z << std::endl;
      std::cout << "yIndx: " << pos_z << "  " << indx_y << std::endl;
      std::cout << std::endl;
      continue;
    }
    
    cell_center_x = xMin + indx_x*H.dx + 0.5*H.dx;
    cell_center_y = yMin + indx_y*H.dy + 0.5*H.dy;
    cell_center_z = zMin + indx_z*H.dz + 0.5*H.dz;
    delta_x = 1 - ( pos_x - cell_center_x ) / H.dx;
    delta_y = 1 - ( pos_y - cell_center_y ) / H.dy;
    delta_z = 1 - ( pos_z - cell_center_z ) / H.dz;
    indx_x += H.n_ghost;
    indx_y += H.n_ghost;
    indx_z += H.n_ghost;

    //std::cout << "delta (x, y, z): (" << delta_x << ", " << delta_y << ", " << delta_z << ")" << std::endl;
    //std::cout << "cell center (x, y, z): (" << cell_center_x << ", " << cell_center_y << ", " << cell_center_z << ")" << std::endl;

    indx = indx_x + indx_y*nx_g + indx_z*nx_g*ny_g;
    if (is_resolved) {
      C.density[indx] += feedback_density  * delta_x * delta_y * delta_z;
      C.GasEnergy[indx] += feedback_energy  * delta_x * delta_y * delta_z;
      C.Energy[indx] += feedback_energy  * delta_x * delta_y * delta_z;
      info[threadId*N_INFO + 3] += feedback_energy  * fabs(delta_x * delta_y * delta_z) * dV;
    } else {
      C.momentum_x[indx] += -delta_x * feedback_momentum;
      C.momentum_y[indx] += -delta_y * feedback_momentum;
      C.momentum_z[indx] += -delta_z * feedback_momentum;
      info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    max_dti[threadId] = fmax(max_dti[threadId], Calc_Timestep(indx));

    indx = (indx_x+1) + indx_y*nx_g + indx_z*nx_g*ny_g;
    if (is_resolved) {
      C.density[indx] += feedback_density  * (1-delta_x) * delta_y * delta_z;
      C.GasEnergy[indx] += feedback_energy  * (1-delta_x) * delta_y * delta_z;
      C.Energy[indx] += feedback_energy  * (1-delta_x) * delta_y * delta_z;
      info[threadId*N_INFO + 3] += feedback_energy  * fabs((1-delta_x) * delta_y * delta_z) * dV;
    } else {
      C.momentum_x[indx] +=  delta_x * feedback_momentum;
      C.momentum_y[indx] += -delta_y * feedback_momentum;
      C.momentum_z[indx] += -delta_z * feedback_momentum;
      info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    max_dti[threadId] = fmax(max_dti[threadId], Calc_Timestep(indx));

    indx = indx_x + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
    if (is_resolved) {
      C.density[indx] += feedback_density  * delta_x * (1-delta_y) * delta_z;
      C.GasEnergy[indx] += feedback_energy  * delta_x * (1-delta_y) * delta_z;
      C.Energy[indx] += feedback_energy  * delta_x * (1-delta_y) * delta_z;
      info[threadId*N_INFO + 3] += feedback_energy  * fabs(delta_x * (1-delta_y )* delta_z) * dV;
    } else {
      C.momentum_x[indx] += -delta_x * feedback_momentum;
      C.momentum_y[indx] +=  delta_y * feedback_momentum;
      C.momentum_z[indx] += -delta_z * feedback_momentum;
      info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    max_dti[threadId] = fmax(max_dti[threadId], Calc_Timestep(indx));

    indx = indx_x + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
    if (is_resolved) {
      C.density[indx] += feedback_density  * delta_x * delta_y * (1-delta_z);
      C.GasEnergy[indx] += feedback_energy  * delta_x * delta_y * (1-delta_z);
      C.Energy[indx] += feedback_energy  * delta_x * delta_y * (1-delta_z);
      info[threadId*N_INFO + 3] += feedback_energy  * fabs(delta_x * delta_y * (1 - delta_z)) * dV;
    } else {
      C.momentum_x[indx] += -delta_x * feedback_momentum;
      C.momentum_y[indx] += -delta_y * feedback_momentum;
      C.momentum_z[indx] +=  delta_z * feedback_momentum; 
      info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    max_dti[threadId] = fmax(max_dti[threadId], Calc_Timestep(indx));

    indx = (indx_x+1) + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
    if (is_resolved) {
      C.density[indx] += feedback_density  * (1-delta_x) * (1-delta_y) * delta_z;
      C.GasEnergy[indx] += feedback_energy  * (1-delta_x) * (1-delta_y) * delta_z;
      C.Energy[indx] += feedback_energy  * (1-delta_x) * (1-delta_y) * delta_z;
      info[threadId*N_INFO + 3] += feedback_energy  * fabs((1-delta_x) * (1-delta_y) * delta_z) * dV;
    } else {
      C.momentum_x[indx] += delta_x * feedback_momentum;
      C.momentum_y[indx] += delta_y * feedback_momentum;
      C.momentum_z[indx] += -delta_z * feedback_momentum;
      info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    max_dti[threadId] = fmax(max_dti[threadId], Calc_Timestep(indx));

    indx = (indx_x+1) + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
    if (is_resolved) {
      C.density[indx] += feedback_density  * (1-delta_x) * delta_y * (1-delta_z);
      C.GasEnergy[indx] += feedback_energy  * (1-delta_x) * delta_y * (1-delta_z);
      C.Energy[indx] += feedback_energy  * (1-delta_x) * delta_y * (1-delta_z);
      info[threadId*N_INFO + 3] += feedback_energy  * fabs((1-delta_x) * delta_y * (1-delta_z)) * dV;
    } else {
      C.momentum_x[indx] +=  delta_x * feedback_momentum;
      C.momentum_y[indx] += -delta_y * feedback_momentum;
      C.momentum_z[indx] +=  delta_z * feedback_momentum;
      info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    max_dti[threadId] = fmax(max_dti[threadId], Calc_Timestep(indx));

    indx = indx_x + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    if (is_resolved) {
      C.density[indx] += feedback_density  * delta_x * (1-delta_y) * (1-delta_z);
      C.GasEnergy[indx] += feedback_energy  * delta_x * (1-delta_y) * (1-delta_z);
      C.Energy[indx] += feedback_energy  * delta_x * (1-delta_y) * (1-delta_z);
      info[threadId*N_INFO + 3] += feedback_energy * fabs(delta_x * (1-delta_y) * (1-delta_z)) * dV;
    } else {
      C.momentum_x[indx] += -delta_x * feedback_momentum;
      C.momentum_y[indx] +=  delta_y * feedback_momentum;
      C.momentum_z[indx] +=  delta_z * feedback_momentum;
      info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    max_dti[threadId] = fmax(max_dti[threadId], Calc_Timestep(indx));

    indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    if (is_resolved) {
      C.density[indx] += feedback_density * (1-delta_x) * (1-delta_y) * (1-delta_z);
      C.GasEnergy[indx] += feedback_energy * (1-delta_x) * (1-delta_y) * (1-delta_z);
      C.Energy[indx] += feedback_energy * (1-delta_x) * (1-delta_y) * (1-delta_z);
      info[threadId*N_INFO + 3] += feedback_energy * fabs((1-delta_x) * (1-delta_y) * (1-delta_z)) * dV;
    } else {
      C.momentum_x[indx] +=  delta_x * feedback_momentum;
      C.momentum_y[indx] +=  delta_y * feedback_momentum;
      C.momentum_z[indx] +=  delta_z * feedback_momentum;
      info[threadId*N_INFO + 4] += (fabs(delta_x) /*+ fabs(delta_y) + fabs(delta_z)*/)*feedback_momentum * dV;
    }
    max_dti[threadId] = fmax(max_dti[threadId], Calc_Timestep(indx));
  }
  #endif //PARTICLES_CPU
}
#endif //PARTICLE_AGE
#endif //DE
#endif //PARTICLES
#endif //FEEDBACK

#ifdef PARTICLES
#ifdef DE
#ifdef PARTICLE_AGE

#include <iostream>
#include "../particles/feedback_CIC.h"
#include "../particles/particles_3D.h"
#include "../grid/grid3D.h"
#include "../particles/density_CIC.h"


#ifdef PARALLEL_OMP
#include "../utils/parallel_omp.h"
#endif


// simple energy feedback prescription
Real getClusterEnergyFeedback(Real t, Real dt, Real age) {
    if (t + age <= 1.0e4) return ENERGY_FEEDBACK_RATE * dt;
    else return 0;
}

// simple feedback prescription
Real getClusterMassFeedback(Real t, Real dt, Real age) {
    //if (t + age <= 1.0e4) return 0.1 * dt; // 0.01 SN/ky/cluster * 10 solar mass ejected/SN
    //if (t + age <= 1.0e4) return 10 * dt; // 1 SN/ky/cluster * 10 solar mass ejected/SN
    //else return 0;
    return 0;
}


void Grid3D::Cluster_Feedback(){
  #ifdef PARTICLES_CPU
  #ifndef PARALLEL_OMP
  Cluster_Feedback_Function( 0, Particles.n_local );
  #else
  #pragma omp parallel num_threads( N_OMP_THREADS )
  {
    int omp_id, n_omp_procs;
    part_int_t p_start, p_end;

    omp_id = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();

    Get_OMP_Particles_Indxs( Particles.n_local, N_OMP_THREADS, omp_id,  &p_start, &p_end );

    Cluster_Feedback_Function( p_start, p_end );
  }
  #endif //PARALLEL_OMP
  #endif //PARTICLES_CPU
}


//Compute the CIC feedback
void Grid3D::Cluster_Feedback_Function(part_int_t p_start, part_int_t p_end) {
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
  Real x_pos, y_pos, z_pos;
  Real cell_center_x, cell_center_y, cell_center_z;
  Real delta_x, delta_y, delta_z;
  Real dV_inv = 1./(H.dx*H.dy*H.dz);
  Real feedback_energy, feedback_density;

  bool ignore, in_local;
  for ( pIndx=p_start; pIndx < p_end; pIndx++ ){
    ignore = false;
    in_local = true;
    // pMass = Particles.mass[pIndx] * dV_inv;
    x_pos = Particles.pos_x[pIndx];
    y_pos = Particles.pos_y[pIndx];
    z_pos = Particles.pos_z[pIndx];
    Get_Indexes_CIC( xMin, yMin, zMin, H.dx, H.dy, H.dz, x_pos, y_pos, z_pos, indx_x, indx_y, indx_z );
    if ( indx_x < -1 ) ignore = true;
    if ( indx_y < -1 ) ignore = true;
    if ( indx_z < -1 ) ignore = true;
    if ( indx_x > nx_g-3  ) ignore = true;
    if ( indx_y > ny_g-3  ) ignore = true;
    if ( indx_y > nz_g-3  ) ignore = true;
    if ( x_pos < H.xblocal || x_pos >= H.xblocal + H.domlen_x ) in_local = false;
    if ( y_pos < H.yblocal || y_pos >= H.yblocal + H.domlen_y ) in_local = false;
    if ( z_pos < H.zblocal || z_pos >= H.zblocal + H.domlen_z ) in_local = false;
    if ( ! in_local  ) {
      std::cout << " Cluster_FeedbackError:" << std::endl;
      #ifdef PARTICLE_IDS
      std::cout << " Particle outside Loacal  domain    pID: " << Particles.partIDs[pIndx] << std::endl;
      #else
      std::cout << " Particle outside Loacal  domain " << std::endl;
      #endif
      std::cout << "  Domain X: " << xMin <<  "  " << H.xblocal + H.domlen_x << std::endl;
      std::cout << "  Domain Y: " << yMin <<  "  " << H.yblocal + H.domlen_y << std::endl;
      std::cout << "  Domain Z: " << zMin <<  "  " << H.zblocal + H.domlen_z << std::endl;
      std::cout << "  Particle X: " << x_pos << std::endl;
      std::cout << "  Particle Y: " << y_pos << std::endl;
      std::cout << "  Particle Z: " << z_pos << std::endl;
      continue;
    }
    if ( ignore ){
      #ifdef PARTICLE_IDS
      std::cout << "ERROR Cluster_Feedback Index    pID: " << Particles.partIDs[pIndx] << std::endl;
      #else
      std::cout << "ERROR Cluster_Feedback Index " << std::endl;
      #endif
      std::cout << "Negative xIndx: " << x_pos << "  " << indx_x << std::endl;
      std::cout << "Negative zIndx: " << z_pos << "  " << indx_z << std::endl;
      std::cout << "Negative yIndx: " << y_pos << "  " << indx_y << std::endl;
      std::cout << "Excess xIndx: " << x_pos << "  " << indx_x << std::endl;
      std::cout << "Excess yIndx: " << y_pos << "  " << indx_y << std::endl;
      std::cout << "Excess zIndx: " << z_pos << "  " << indx_z << std::endl;
      std::cout << std::endl;
      continue;
    }

    cell_center_x = xMin + indx_x*H.dx + 0.5*H.dx;
    cell_center_y = yMin + indx_y*H.dy + 0.5*H.dy;
    cell_center_z = zMin + indx_z*H.dz + 0.5*H.dz;
    delta_x = 1 - ( x_pos - cell_center_x ) / H.dx;
    delta_y = 1 - ( y_pos - cell_center_y ) / H.dy;
    delta_z = 1 - ( z_pos - cell_center_z ) / H.dz;
    indx_x += H.n_ghost;
    indx_y += H.n_ghost;
    indx_z += H.n_ghost;

    feedback_energy = getClusterEnergyFeedback(H.t, H.dt, Particles.age[pIndx]) * dV_inv;
    feedback_density = getClusterMassFeedback(H.t, H.dt, Particles.age[pIndx]) * dV_inv;

    indx = indx_x + indx_y*nx_g + indx_z*nx_g*ny_g;
    C.density[indx] += feedback_density  * delta_x * delta_y * delta_z;
    C.GasEnergy[indx] += feedback_energy  * delta_x * delta_y * delta_z;

    indx = (indx_x+1) + indx_y*nx_g + indx_z*nx_g*ny_g;
    C.density[indx] += feedback_density  * (1-delta_x) * delta_y * delta_z;
    C.GasEnergy[indx] += feedback_energy  * (1-delta_x) * delta_y * delta_z;

    indx = indx_x + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
    C.density[indx] += feedback_density  * delta_x * (1-delta_y) * delta_z;
    C.GasEnergy[indx] += feedback_energy  * delta_x * (1-delta_y) * delta_z;

    indx = indx_x + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
    C.density[indx] += feedback_density  * delta_x * delta_y * (1-delta_z);
    C.GasEnergy[indx] += feedback_energy  * delta_x * delta_y * (1-delta_z);

    indx = (indx_x+1) + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
    C.density[indx] += feedback_density  * (1-delta_x) * (1-delta_y) * delta_z;
    C.GasEnergy[indx] += feedback_energy  * (1-delta_x) * (1-delta_y) * delta_z;

    indx = (indx_x+1) + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
    C.density[indx] += feedback_density  * (1-delta_x) * delta_y * (1-delta_z);
    C.GasEnergy[indx] += feedback_energy  * (1-delta_x) * delta_y * (1-delta_z);

    indx = indx_x + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    C.density[indx] += feedback_density  * delta_x * (1-delta_y) * (1-delta_z);
    C.GasEnergy[indx] += feedback_energy  * delta_x * (1-delta_y) * (1-delta_z);

    indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
    C.density[indx] += feedback_density * (1-delta_x) * (1-delta_y) * (1-delta_z);
    C.GasEnergy[indx] += feedback_energy * (1-delta_x) * (1-delta_y) * (1-delta_z);
  }
}

#endif //PARTICLE_AGE
#endif //DE
#endif //PARTICLES

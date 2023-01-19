#ifdef PARTICLES

  #include <stdio.h>
  #include <stdlib.h>

  #include <iostream>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../particles/particles_3D.h"
  #include "math.h"

  #ifdef PARALLEL_OMP
    #include "../utils/parallel_omp.h"
  #endif

// Get the particles Cloud-In-Cell interpolated density
void Particles_3D::Get_Density_CIC()
{
  #ifdef PARTICLES_CPU
    #ifdef PARALLEL_OMP
  Get_Density_CIC_OMP();
    #else
  Get_Density_CIC_Serial();
    #endif  // PARALLEL_OMP
  #endif

  #ifdef PARTICLES_GPU
  Get_Density_CIC_GPU();
  #endif
}

// Compute the particles density and copy it to the array in Grav to compute the
// potential
void Grid3D::Copy_Particles_Density_to_Gravity(struct parameters P)
{
  #ifdef CPU_TIME
  Timer.Part_Density.Start();
  #endif

  // Step 1: Get Particles CIC Density
  Particles.Clear_Density();
  Particles.Get_Density_CIC();

  #ifdef CPU_TIME
  Timer.Part_Density.End();
  #endif

  #ifdef CPU_TIME
  Timer.Part_Dens_Transf.Start();
  #endif
  // Step 2: Transfer Particles CIC density Boundaries
  Transfer_Particles_Density_Boundaries(P);

  // Step 3: Copy Particles density to Gravity array
  Copy_Particles_Density();

  #ifdef CPU_TIME
  Timer.Part_Dens_Transf.End();
  #endif
}

// Copy the particles density to the density array in Grav to compute the
// potential
void Grid3D::Copy_Particles_Density()
{
  #ifdef GRAVITY_GPU
    #ifdef PARTICLES_CPU
  Copy_Particles_Density_to_GPU();
    #endif
  Copy_Particles_Density_GPU();
  #else

    #ifndef PARALLEL_OMP
  Copy_Particles_Density_function(0, Grav.nz_local);
    #else

      #pragma omp parallel num_threads(N_OMP_THREADS)
  {
    int omp_id, n_omp_procs;
    int g_start, g_end;

    omp_id      = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();

    Get_OMP_Grid_Indxs(Grav.nz_local, n_omp_procs, omp_id, &g_start, &g_end);

    Copy_Particles_Density_function(g_start, g_end);
  }
    #endif  // PARALLEL_OMP

  #endif  // GRAVITY_GPU
}

void Grid3D::Copy_Particles_Density_function(int g_start, int g_end)
{
  int nx_part, ny_part, nz_part, nGHST;
  nGHST   = Particles.G.n_ghost_particles_grid;
  nx_part = Particles.G.nx_local + 2 * nGHST;
  ny_part = Particles.G.ny_local + 2 * nGHST;
  nz_part = Particles.G.nz_local + 2 * nGHST;

  int nx_dens, ny_dens, nz_dens;
  nx_dens = Grav.nx_local;
  ny_dens = Grav.ny_local;
  nz_dens = Grav.nz_local;

  int i, j, k, id_CIC, id_grid;
  for (k = g_start; k < g_end; k++) {
    for (j = 0; j < ny_dens; j++) {
      for (i = 0; i < nx_dens; i++) {
        id_CIC = (i + nGHST) + (j + nGHST) * nx_part +
                 (k + nGHST) * nx_part * ny_part;
        id_grid                   = i + j * nx_dens + k * nx_dens * ny_dens;
        Grav.F.density_h[id_grid] = Particles.G.density[id_CIC];
      }
    }
  }
}

// Clear the density array: density=0
void ::Particles_3D::Clear_Density()
{
  #ifdef PARTICLES_CPU
  for (int i = 0; i < G.n_cells; i++) G.density[i] = 0;
  #endif

  #ifdef PARTICLES_GPU
  Clear_Density_GPU();
  #endif
}

  #ifdef PARTICLES_GPU

void Particles_3D::Clear_Density_GPU()
{
  Clear_Density_GPU_function(G.density_dev, G.n_cells);
}

void Particles_3D::Get_Density_CIC_GPU()
{
  Get_Density_CIC_GPU_function(
      n_local, particle_mass, G.xMin, G.xMax, G.yMin, G.yMax, G.zMin, G.zMax,
      G.dx, G.dy, G.dz, G.nx_local, G.ny_local, G.nz_local,
      G.n_ghost_particles_grid, G.n_cells, G.density, G.density_dev, pos_x_dev,
      pos_y_dev, pos_z_dev, mass_dev);
}

  #endif  // PARTICLES_GPU

  #ifdef PARTICLES_CPU
// Get the CIC index from the particle position
void Get_Indexes_CIC(Real xMin, Real yMin, Real zMin, Real dx, Real dy, Real dz,
                     Real pos_x, Real pos_y, Real pos_z, int &indx_x,
                     int &indx_y, int &indx_z)
{
  indx_x = (int)floor((pos_x - xMin - 0.5 * dx) / dx);
  indx_y = (int)floor((pos_y - yMin - 0.5 * dy) / dy);
  indx_z = (int)floor((pos_z - zMin - 0.5 * dz) / dz);
}

// Comute the CIC density (NO OpenMP)
void Particles_3D::Get_Density_CIC_Serial()
{
  int nGHST = G.n_ghost_particles_grid;
  int nx_g  = G.nx_local + 2 * nGHST;
  int ny_g  = G.ny_local + 2 * nGHST;
  int nz_g  = G.nz_local + 2 * nGHST;

  Real xMin, yMin, zMin, dx, dy, dz;
  xMin = G.xMin;
  yMin = G.yMin;
  zMin = G.zMin;
  dx   = G.dx;
  dy   = G.dy;
  dz   = G.dz;

  part_int_t pIndx;
  int indx_x, indx_y, indx_z, indx;
  Real pMass, x_pos, y_pos, z_pos;

  Real cell_center_x, cell_center_y, cell_center_z;
  Real delta_x, delta_y, delta_z;
  Real dV_inv = 1. / (G.dx * G.dy * G.dz);
  bool ignore, in_local;

  for (pIndx = 0; pIndx < n_local; pIndx++) {
    ignore   = false;
    in_local = true;

    #ifdef SINGLE_PARTICLE_MASS
    pMass = particle_mass * dV_inv;
    #else
    pMass = mass[pIndx] * dV_inv;
    #endif
    x_pos = pos_x[pIndx];
    y_pos = pos_y[pIndx];
    z_pos = pos_z[pIndx];
    Get_Indexes_CIC(xMin, yMin, zMin, dx, dy, dz, x_pos, y_pos, z_pos, indx_x,
                    indx_y, indx_z);
    if (indx_x < -1) ignore = true;
    if (indx_y < -1) ignore = true;
    if (indx_z < -1) ignore = true;
    if (indx_x > nx_g - 3) ignore = true;
    if (indx_y > ny_g - 3) ignore = true;
    if (indx_y > nz_g - 3) ignore = true;
    if (x_pos < G.xMin || x_pos >= G.xMax) in_local = false;
    if (y_pos < G.yMin || y_pos >= G.yMax) in_local = false;
    if (z_pos < G.zMin || z_pos >= G.zMax) in_local = false;
    if (!in_local) {
      std::cout << " Density CIC Error:" << std::endl;
    #ifdef PARTICLE_IDS
      std::cout << " Particle outside Local  domain    pID: " << partIDs[pIndx]
                << std::endl;
    #else
      std::cout << " Particle outside Local  domain " << std::endl;
    #endif
      std::cout << "  Domain X: " << G.xMin << "  " << G.xMax << std::endl;
      std::cout << "  Domain Y: " << G.yMin << "  " << G.yMax << std::endl;
      std::cout << "  Domain Z: " << G.zMin << "  " << G.zMax << std::endl;
      std::cout << "  Particle X: " << x_pos << std::endl;
      std::cout << "  Particle Y: " << y_pos << std::endl;
      std::cout << "  Particle Z: " << z_pos << std::endl;
      continue;
    }
    if (ignore) {
    #ifdef PARTICLE_IDS
      std::cout << "ERROR Density CIC Index    pID: " << partIDs[pIndx]
                << std::endl;
    #else
      std::cout << "ERROR Density CIC Index " << std::endl;
    #endif
      std::cout << "Negative xIndx: " << x_pos << "  " << indx_x << std::endl;
      std::cout << "Negative zIndx: " << z_pos << "  " << indx_z << std::endl;
      std::cout << "Negative yIndx: " << y_pos << "  " << indx_y << std::endl;
      std::cout << "Excess xIndx: " << x_pos << "  " << indx_x << std::endl;
      std::cout << "Excess yIndx: " << y_pos << "  " << indx_y << std::endl;
      std::cout << "Excess zIndx: " << z_pos << "  " << indx_z << std::endl;
      std::cout << std::endl;
      // exit(-1);
      continue;
    }
    cell_center_x = xMin + indx_x * dx + 0.5 * dx;
    cell_center_y = yMin + indx_y * dy + 0.5 * dy;
    cell_center_z = zMin + indx_z * dz + 0.5 * dz;
    delta_x       = 1 - (x_pos - cell_center_x) / dx;
    delta_y       = 1 - (y_pos - cell_center_y) / dy;
    delta_z       = 1 - (z_pos - cell_center_z) / dz;
    indx_x += nGHST;
    indx_y += nGHST;
    indx_z += nGHST;

    indx = indx_x + indx_y * nx_g + indx_z * nx_g * ny_g;
    G.density[indx] += pMass * delta_x * delta_y * delta_z;

    indx = (indx_x + 1) + indx_y * nx_g + indx_z * nx_g * ny_g;
    G.density[indx] += pMass * (1 - delta_x) * delta_y * delta_z;

    indx = indx_x + (indx_y + 1) * nx_g + indx_z * nx_g * ny_g;
    G.density[indx] += pMass * delta_x * (1 - delta_y) * delta_z;

    indx = indx_x + indx_y * nx_g + (indx_z + 1) * nx_g * ny_g;
    G.density[indx] += pMass * delta_x * delta_y * (1 - delta_z);

    indx = (indx_x + 1) + (indx_y + 1) * nx_g + indx_z * nx_g * ny_g;
    G.density[indx] += pMass * (1 - delta_x) * (1 - delta_y) * delta_z;

    indx = (indx_x + 1) + indx_y * nx_g + (indx_z + 1) * nx_g * ny_g;
    G.density[indx] += pMass * (1 - delta_x) * delta_y * (1 - delta_z);

    indx = indx_x + (indx_y + 1) * nx_g + (indx_z + 1) * nx_g * ny_g;
    G.density[indx] += pMass * delta_x * (1 - delta_y) * (1 - delta_z);

    indx = (indx_x + 1) + (indx_y + 1) * nx_g + (indx_z + 1) * nx_g * ny_g;
    G.density[indx] += pMass * (1 - delta_x) * (1 - delta_y) * (1 - delta_z);
  }
}

    #ifdef PARALLEL_OMP
// Compute the CIC density when PARALLEL_OMP
void Particles_3D::Get_Density_CIC_OMP()
{
      // Span OpenMP threads
      #pragma omp parallel num_threads(N_OMP_THREADS)
  {
    int omp_id;
    int g_start, g_end;
    int n_omp_procs;

    omp_id      = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();

    int nGHST = G.n_ghost_particles_grid;
    int nx_g  = G.nx_local + 2 * nGHST;
    int ny_g  = G.ny_local + 2 * nGHST;
    int nz_g  = G.nz_local + 2 * nGHST;

    Real xMin, yMin, zMin, dx, dy, dz;
    xMin        = G.xMin;
    yMin        = G.yMin;
    zMin        = G.zMin;
    dx          = G.dx;
    dy          = G.dy;
    dz          = G.dz;
    Real dV_inv = 1. / (G.dx * G.dy * G.dz);

    Get_OMP_Grid_Indxs(nz_g, n_omp_procs, omp_id, &g_start, &g_end);

    part_int_t pIndx;
    int indx_x, indx_y, indx_z, indx;
    Real pMass, x_pos, y_pos, z_pos;

    Real cell_center_x, cell_center_y, cell_center_z;
    Real delta_x, delta_y, delta_z;
    bool ignore, in_local;
    bool add_1, add_2;

    for (pIndx = 0; pIndx < n_local; pIndx++) {
      add_1 = false;
      add_2 = false;

      z_pos  = pos_z[pIndx];
      indx_z = (int)floor((z_pos - zMin - 0.5 * dz) / dz);
      indx_z += nGHST;
      if ((indx_z >= g_start) && (indx_z < g_end)) add_1 = true;
      if (((indx_z + 1) >= g_start) && ((indx_z + 1) < g_end)) add_2 = true;
      if (!(add_1 || add_2)) continue;

      ignore = false;
      x_pos  = pos_x[pIndx];
      y_pos  = pos_y[pIndx];

      indx_x = (int)floor((x_pos - xMin - 0.5 * dx) / dx);
      indx_y = (int)floor((y_pos - yMin - 0.5 * dy) / dy);
      indx_z -= nGHST;

      if (indx_x < -1) ignore = true;
      if (indx_y < -1) ignore = true;
      if (indx_z < -1) ignore = true;
      if (indx_x > nx_g - 3) ignore = true;
      if (indx_y > ny_g - 3) ignore = true;
      if (indx_y > nz_g - 3) ignore = true;
      if (ignore) {
      #ifdef PARTICLE_IDS
        std::cout << "ERROR CIC Index    pID: " << partIDs[pIndx] << std::endl;
      #else
        std::cout << "ERROR CIC Index " << std::endl;
      #endif
        std::cout << "Negative xIndx: " << x_pos << "  " << indx_x << std::endl;
        std::cout << "Negative zIndx: " << z_pos << "  " << indx_z << std::endl;
        std::cout << "Negative yIndx: " << y_pos << "  " << indx_y << std::endl;
        std::cout << "Excess xIndx: " << x_pos << "  " << indx_x << std::endl;
        std::cout << "Excess yIndx: " << y_pos << "  " << indx_y << std::endl;
        std::cout << "Excess zIndx: " << z_pos << "  " << indx_z << std::endl;
        std::cout << std::endl;
        // exit(-1);
        continue;
      }
      in_local = true;
      if (x_pos < G.xMin || x_pos >= G.xMax) in_local = false;
      if (y_pos < G.yMin || y_pos >= G.yMax) in_local = false;
      if (z_pos < G.zMin || z_pos >= G.zMax) in_local = false;
      if (!in_local) {
        std::cout << " Density CIC Error:" << std::endl;
      #ifdef PARTICLE_IDS
        std::cout << " Particle outside Local  domain    pID: "
                  << partIDs[pIndx] << std::endl;
      #else
        std::cout << " Particle outside Local  domain " << std::endl;
      #endif
        std::cout << "  Domain X: " << G.xMin << "  " << G.xMax << std::endl;
        std::cout << "  Domain Y: " << G.yMin << "  " << G.yMax << std::endl;
        std::cout << "  Domain Z: " << G.zMin << "  " << G.zMax << std::endl;
        std::cout << "  Particle X: " << x_pos << std::endl;
        std::cout << "  Particle Y: " << y_pos << std::endl;
        std::cout << "  Particle Z: " << z_pos << std::endl;
        continue;
      }

      #ifdef SINGLE_PARTICLE_MASS
      pMass = particle_mass * dV_inv;
      #else
      pMass = mass[pIndx] * dV_inv;
      #endif

      cell_center_x = xMin + indx_x * dx + 0.5 * dx;
      cell_center_y = yMin + indx_y * dy + 0.5 * dy;
      cell_center_z = zMin + indx_z * dz + 0.5 * dz;
      delta_x       = 1 - (x_pos - cell_center_x) / dx;
      delta_y       = 1 - (y_pos - cell_center_y) / dy;
      delta_z       = 1 - (z_pos - cell_center_z) / dz;
      indx_x += nGHST;
      indx_y += nGHST;
      indx_z += nGHST;

      if (add_1) {
        indx = indx_x + indx_y * nx_g + indx_z * nx_g * ny_g;
        G.density[indx] += pMass * delta_x * delta_y * delta_z;

        indx = (indx_x + 1) + indx_y * nx_g + indx_z * nx_g * ny_g;
        G.density[indx] += pMass * (1 - delta_x) * delta_y * delta_z;

        indx = indx_x + (indx_y + 1) * nx_g + indx_z * nx_g * ny_g;
        G.density[indx] += pMass * delta_x * (1 - delta_y) * delta_z;

        indx = (indx_x + 1) + (indx_y + 1) * nx_g + indx_z * nx_g * ny_g;
        G.density[indx] += pMass * (1 - delta_x) * (1 - delta_y) * delta_z;
      }

      if (add_2) {
        indx = indx_x + indx_y * nx_g + (indx_z + 1) * nx_g * ny_g;
        G.density[indx] += pMass * delta_x * delta_y * (1 - delta_z);

        indx = (indx_x + 1) + indx_y * nx_g + (indx_z + 1) * nx_g * ny_g;
        G.density[indx] += pMass * (1 - delta_x) * delta_y * (1 - delta_z);

        indx = indx_x + (indx_y + 1) * nx_g + (indx_z + 1) * nx_g * ny_g;
        G.density[indx] += pMass * delta_x * (1 - delta_y) * (1 - delta_z);

        indx = (indx_x + 1) + (indx_y + 1) * nx_g + (indx_z + 1) * nx_g * ny_g;
        G.density[indx] +=
            pMass * (1 - delta_x) * (1 - delta_y) * (1 - delta_z);
      }
    }
  }
}
    #endif  // PARALLEL_OMP

  #endif  // PARTICLES_CPU

#endif

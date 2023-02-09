#ifdef PARTICLES

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../utils/gpu.hpp"
  #include "particles_3D.h"

  #ifdef GRAVITY_GPU
    #include "../grid/grid3D.h"
  #endif

  #ifdef PARTICLES_GPU

// Copy the potential from host to device
void Particles_3D::Copy_Potential_To_GPU(Real *potential_host, Real *potential_dev, int n_cells_potential)
{
  CudaSafeCall(cudaMemcpy(potential_dev, potential_host, n_cells_potential * sizeof(Real), cudaMemcpyHostToDevice));
}

// Kernel to compute the gradient of the potential
__global__ void Get_Gravity_Field_Particles_Kernel(Real *potential_dev, Real *gravity_x_dev, Real *gravity_y_dev,
                                                   Real *gravity_z_dev, int nx, int ny, int nz,
                                                   int n_ghost_particles_grid, int n_ghost_potential, Real dx, Real dy,
                                                   Real dz)
{
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  int tid_z = blockIdx.z * blockDim.z + threadIdx.z;

  int nx_grav, ny_grav, nz_grav;
  nx_grav = nx + 2 * n_ghost_particles_grid;
  ny_grav = ny + 2 * n_ghost_particles_grid;
  nz_grav = nz + 2 * n_ghost_particles_grid;

  if (tid_x >= nx_grav || tid_y >= ny_grav || tid_z >= nz_grav) return;
  int tid = tid_x + tid_y * nx_grav + tid_z * nx_grav * ny_grav;

  int nx_pot, ny_pot;
  nx_pot = nx + 2 * n_ghost_potential;
  ny_pot = ny + 2 * n_ghost_potential;

  int nGHST = n_ghost_potential - n_ghost_particles_grid;

  Real phi_l, phi_r;
  int id_l, id_r;
    #ifdef GRAVITY_5_POINTS_GRADIENT
  Real phi_ll, phi_rr;
  int id_ll, id_rr;
    #endif

  // Get Potential Gradient X
  id_l  = (tid_x - 1 + nGHST) + (tid_y + nGHST) * nx_pot + (tid_z + nGHST) * ny_pot * nx_pot;
  id_r  = (tid_x + 1 + nGHST) + (tid_y + nGHST) * nx_pot + (tid_z + nGHST) * ny_pot * nx_pot;
  phi_l = potential_dev[id_l];
  phi_r = potential_dev[id_r];
    #ifdef GRAVITY_5_POINTS_GRADIENT
  id_ll              = (tid_x - 2 + nGHST) + (tid_y + nGHST) * nx_pot + (tid_z + nGHST) * ny_pot * nx_pot;
  id_rr              = (tid_x + 2 + nGHST) + (tid_y + nGHST) * nx_pot + (tid_z + nGHST) * ny_pot * nx_pot;
  phi_ll             = potential_dev[id_ll];
  phi_rr             = potential_dev[id_rr];
  gravity_x_dev[tid] = -1 * (-phi_rr + 8 * phi_r - 8 * phi_l + phi_ll) / (12 * dx);
    #else
  gravity_x_dev[tid] = -0.5 * (phi_r - phi_l) / dx;
    #endif

  // Get Potential Gradient Y
  id_l  = (tid_x + nGHST) + (tid_y - 1 + nGHST) * nx_pot + (tid_z + nGHST) * ny_pot * nx_pot;
  id_r  = (tid_x + nGHST) + (tid_y + 1 + nGHST) * nx_pot + (tid_z + nGHST) * ny_pot * nx_pot;
  phi_l = potential_dev[id_l];
  phi_r = potential_dev[id_r];
    #ifdef GRAVITY_5_POINTS_GRADIENT
  id_ll              = (tid_x + nGHST) + (tid_y - 2 + nGHST) * nx_pot + (tid_z + nGHST) * ny_pot * nx_pot;
  id_rr              = (tid_x + nGHST) + (tid_y + 2 + nGHST) * nx_pot + (tid_z + nGHST) * ny_pot * nx_pot;
  phi_ll             = potential_dev[id_ll];
  phi_rr             = potential_dev[id_rr];
  gravity_y_dev[tid] = -1 * (-phi_rr + 8 * phi_r - 8 * phi_l + phi_ll) / (12 * dy);
    #else
  gravity_y_dev[tid] = -0.5 * (phi_r - phi_l) / dy;
    #endif

  // Get Potential Gradient Z
  id_l  = (tid_x + nGHST) + (tid_y + nGHST) * nx_pot + (tid_z - 1 + nGHST) * ny_pot * nx_pot;
  id_r  = (tid_x + nGHST) + (tid_y + nGHST) * nx_pot + (tid_z + 1 + nGHST) * ny_pot * nx_pot;
  phi_l = potential_dev[id_l];
  phi_r = potential_dev[id_r];
    #ifdef GRAVITY_5_POINTS_GRADIENT
  id_ll              = (tid_x + nGHST) + (tid_y + nGHST) * nx_pot + (tid_z - 2 + nGHST) * ny_pot * nx_pot;
  id_rr              = (tid_x + nGHST) + (tid_y + nGHST) * nx_pot + (tid_z + 2 + nGHST) * ny_pot * nx_pot;
  phi_ll             = potential_dev[id_ll];
  phi_rr             = potential_dev[id_rr];
  gravity_z_dev[tid] = -1 * (-phi_rr + 8 * phi_r - 8 * phi_l + phi_ll) / (12 * dz);
    #else
  gravity_z_dev[tid] = -0.5 * (phi_r - phi_l) / dz;
    #endif
}

// Call the kernel to compute the gradient of the potential
void Particles_3D::Get_Gravity_Field_Particles_GPU_function(int nx_local, int ny_local, int nz_local,
                                                            int n_ghost_particles_grid, int n_cells_potential, Real dx,
                                                            Real dy, Real dz, Real *potential_host, Real *potential_dev,
                                                            Real *gravity_x_dev, Real *gravity_y_dev,
                                                            Real *gravity_z_dev)
{
    #ifndef GRAVITY_GPU
  Copy_Potential_To_GPU(potential_host, potential_dev, n_cells_potential);
    #endif

  int nx_g, ny_g, nz_g;
  nx_g = nx_local + 2 * N_GHOST_POTENTIAL;
  ny_g = ny_local + 2 * N_GHOST_POTENTIAL;
  nz_g = nz_local + 2 * N_GHOST_POTENTIAL;

  // set values for GPU kernels
  int tpb_x   = 8;
  int tpb_y   = 8;
  int tpb_z   = 8;
  int ngrid_x = (nx_g + tpb_x - 1) / tpb_x;
  int ngrid_y = (ny_g + tpb_y - 1) / tpb_y;
  int ngrid_z = (nz_g + tpb_z - 1) / tpb_z;
  // number of blocks per 1D grid
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);

  hipLaunchKernelGGL(Get_Gravity_Field_Particles_Kernel, dim3dGrid, dim3dBlock, 0, 0, potential_dev, gravity_x_dev,
                     gravity_y_dev, gravity_z_dev, nx_local, ny_local, nz_local, n_ghost_particles_grid,
                     N_GHOST_POTENTIAL, dx, dy, dz);
  CudaCheckError();
}

// Get CIC indexes from the particles positions
__device__ void Get_Indexes_CIC_Gravity(Real xMin, Real yMin, Real zMin, Real dx, Real dy, Real dz, Real pos_x,
                                        Real pos_y, Real pos_z, int &indx_x, int &indx_y, int &indx_z)
{
  indx_x = (int)floor((pos_x - xMin - 0.5 * dx) / dx);
  indx_y = (int)floor((pos_y - yMin - 0.5 * dy) / dy);
  indx_z = (int)floor((pos_z - zMin - 0.5 * dz) / dz);
}

// Kernel to compute the gravitational field at the particles positions via
// Cloud-In-Cell
__global__ void Get_Gravity_CIC_Kernel(part_int_t n_local, Real *gravity_x_dev, Real *gravity_y_dev,
                                       Real *gravity_z_dev, Real *pos_x_dev, Real *pos_y_dev, Real *pos_z_dev,
                                       Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev, Real xMin, Real yMin,
                                       Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx,
                                       int ny, int nz, int n_ghost)
{
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n_local) return;

  int nx_g, ny_g;
  nx_g = nx + 2 * n_ghost;
  ny_g = ny + 2 * n_ghost;

  Real pos_x, pos_y, pos_z;
  Real cell_center_x, cell_center_y, cell_center_z;
  Real delta_x, delta_y, delta_z;
  Real g_x_bl, g_x_br, g_x_bu, g_x_bru, g_x_tl, g_x_tr, g_x_tu, g_x_tru;
  Real g_y_bl, g_y_br, g_y_bu, g_y_bru, g_y_tl, g_y_tr, g_y_tu, g_y_tru;
  Real g_z_bl, g_z_br, g_z_bu, g_z_bru, g_z_tl, g_z_tr, g_z_tu, g_z_tru;
  Real g_x, g_y, g_z;

  pos_x = pos_x_dev[tid];
  pos_y = pos_y_dev[tid];
  pos_z = pos_z_dev[tid];

  int indx_x, indx_y, indx_z, indx;
  Get_Indexes_CIC_Gravity(xMin, yMin, zMin, dx, dy, dz, pos_x, pos_y, pos_z, indx_x, indx_y, indx_z);

  bool in_local = true;

  if (pos_x < xMin || pos_x >= xMax) in_local = false;
  if (pos_y < yMin || pos_y >= yMax) in_local = false;
  if (pos_z < zMin || pos_z >= zMax) in_local = false;
  if (!in_local) {
    printf(" Gravity CIC Error: Particle outside local domain");
    return;
  }

  cell_center_x = xMin + indx_x * dx + 0.5 * dx;
  cell_center_y = yMin + indx_y * dy + 0.5 * dy;
  cell_center_z = zMin + indx_z * dz + 0.5 * dz;
  delta_x       = 1 - (pos_x - cell_center_x) / dx;
  delta_y       = 1 - (pos_y - cell_center_y) / dy;
  delta_z       = 1 - (pos_z - cell_center_z) / dz;
  indx_x += n_ghost;
  indx_y += n_ghost;
  indx_z += n_ghost;

  indx   = indx_x + indx_y * nx_g + indx_z * nx_g * ny_g;
  g_x_bl = gravity_x_dev[indx];
  g_y_bl = gravity_y_dev[indx];
  g_z_bl = gravity_z_dev[indx];

  indx   = (indx_x + 1) + (indx_y)*nx_g + (indx_z)*nx_g * ny_g;
  g_x_br = gravity_x_dev[indx];
  g_y_br = gravity_y_dev[indx];
  g_z_br = gravity_z_dev[indx];

  indx   = (indx_x) + (indx_y + 1) * nx_g + (indx_z)*nx_g * ny_g;
  g_x_bu = gravity_x_dev[indx];
  g_y_bu = gravity_y_dev[indx];
  g_z_bu = gravity_z_dev[indx];

  indx    = (indx_x + 1) + (indx_y + 1) * nx_g + (indx_z)*nx_g * ny_g;
  g_x_bru = gravity_x_dev[indx];
  g_y_bru = gravity_y_dev[indx];
  g_z_bru = gravity_z_dev[indx];

  indx   = (indx_x) + (indx_y)*nx_g + (indx_z + 1) * nx_g * ny_g;
  g_x_tl = gravity_x_dev[indx];
  g_y_tl = gravity_y_dev[indx];
  g_z_tl = gravity_z_dev[indx];

  indx   = (indx_x + 1) + (indx_y)*nx_g + (indx_z + 1) * nx_g * ny_g;
  g_x_tr = gravity_x_dev[indx];
  g_y_tr = gravity_y_dev[indx];
  g_z_tr = gravity_z_dev[indx];

  indx   = (indx_x) + (indx_y + 1) * nx_g + (indx_z + 1) * nx_g * ny_g;
  g_x_tu = gravity_x_dev[indx];
  g_y_tu = gravity_y_dev[indx];
  g_z_tu = gravity_z_dev[indx];

  indx    = (indx_x + 1) + (indx_y + 1) * nx_g + (indx_z + 1) * nx_g * ny_g;
  g_x_tru = gravity_x_dev[indx];
  g_y_tru = gravity_y_dev[indx];
  g_z_tru = gravity_z_dev[indx];

  g_x = g_x_bl * (delta_x) * (delta_y) * (delta_z) + g_x_br * (1 - delta_x) * (delta_y) * (delta_z) +
        g_x_bu * (delta_x) * (1 - delta_y) * (delta_z) + g_x_bru * (1 - delta_x) * (1 - delta_y) * (delta_z) +
        g_x_tl * (delta_x) * (delta_y) * (1 - delta_z) + g_x_tr * (1 - delta_x) * (delta_y) * (1 - delta_z) +
        g_x_tu * (delta_x) * (1 - delta_y) * (1 - delta_z) + g_x_tru * (1 - delta_x) * (1 - delta_y) * (1 - delta_z);

  g_y = g_y_bl * (delta_x) * (delta_y) * (delta_z) + g_y_br * (1 - delta_x) * (delta_y) * (delta_z) +
        g_y_bu * (delta_x) * (1 - delta_y) * (delta_z) + g_y_bru * (1 - delta_x) * (1 - delta_y) * (delta_z) +
        g_y_tl * (delta_x) * (delta_y) * (1 - delta_z) + g_y_tr * (1 - delta_x) * (delta_y) * (1 - delta_z) +
        g_y_tu * (delta_x) * (1 - delta_y) * (1 - delta_z) + g_y_tru * (1 - delta_x) * (1 - delta_y) * (1 - delta_z);

  g_z = g_z_bl * (delta_x) * (delta_y) * (delta_z) + g_z_br * (1 - delta_x) * (delta_y) * (delta_z) +
        g_z_bu * (delta_x) * (1 - delta_y) * (delta_z) + g_z_bru * (1 - delta_x) * (1 - delta_y) * (delta_z) +
        g_z_tl * (delta_x) * (delta_y) * (1 - delta_z) + g_z_tr * (1 - delta_x) * (delta_y) * (1 - delta_z) +
        g_z_tu * (delta_x) * (1 - delta_y) * (1 - delta_z) + g_z_tru * (1 - delta_x) * (1 - delta_y) * (1 - delta_z);

  grav_x_dev[tid] = g_x;
  grav_y_dev[tid] = g_y;
  grav_z_dev[tid] = g_z;
}

// Call the kernel to compote the gravitational field at the particles positions
// ( CIC )
void Particles_3D::Get_Gravity_CIC_GPU_function(part_int_t n_local, int nx_local, int ny_local, int nz_local,
                                                int n_ghost_particles_grid, Real xMin, Real xMax, Real yMin, Real yMax,
                                                Real zMin, Real zMax, Real dx, Real dy, Real dz, Real *pos_x_dev,
                                                Real *pos_y_dev, Real *pos_z_dev, Real *grav_x_dev, Real *grav_y_dev,
                                                Real *grav_z_dev, Real *gravity_x_dev, Real *gravity_y_dev,
                                                Real *gravity_z_dev)
{
  // set values for GPU kernels
  int ngrid = (n_local + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  // Only runs if there are local particles
  if (n_local > 0) {
    hipLaunchKernelGGL(Get_Gravity_CIC_Kernel, dim1dGrid, dim1dBlock, 0, 0, n_local, gravity_x_dev, gravity_y_dev,
                       gravity_z_dev, pos_x_dev, pos_y_dev, pos_z_dev, grav_x_dev, grav_y_dev, grav_z_dev, xMin, yMin,
                       zMin, xMax, yMax, zMax, dx, dy, dz, nx_local, ny_local, nz_local, n_ghost_particles_grid);
    CudaCheckError();
  }
}

  #endif  // PARTICLES_GPU

  #ifdef GRAVITY_GPU

void __global__ Copy_Particles_Density_Kernel(Real *dst_density, Real *src_density, int nx_local, int ny_local,
                                              int nz_local, int n_ghost)
{
  int tid_x, tid_y, tid_z, tid_CIC, tid_dens;
  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  tid_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (tid_x >= nx_local || tid_y >= ny_local || tid_z >= nz_local) return;

  tid_dens = tid_x + tid_y * nx_local + tid_z * nx_local * ny_local;

  tid_x += n_ghost;
  tid_y += n_ghost;
  tid_z += n_ghost;

  int nx_CIC, ny_CIC;
  nx_CIC  = nx_local + 2 * n_ghost;
  ny_CIC  = ny_local + 2 * n_ghost;
  tid_CIC = tid_x + tid_y * nx_CIC + tid_z * nx_CIC * ny_CIC;

  dst_density[tid_dens] = src_density[tid_CIC];
}

// Copy the particles density to the density array in Grav to compute the
// potential
void Grid3D::Copy_Particles_Density_GPU()
{
  int nx_local, ny_local, nz_local, n_ghost;
  n_ghost  = Particles.G.n_ghost_particles_grid;
  nx_local = Grav.nx_local;
  ny_local = Grav.ny_local;
  nz_local = Grav.nz_local;

  // set values for GPU kernels
  int tpb_x   = 16;
  int tpb_y   = 8;
  int tpb_z   = 8;
  int ngrid_x = (nx_local - 1) / tpb_x + 1;
  int ngrid_y = (ny_local - 1) / tpb_y + 1;
  int ngrid_z = (nz_local - 1) / tpb_z + 1;
  // number of blocks per 1D grid
  dim3 dim3dGrid(ngrid_x, ngrid_y, ngrid_z);
  //  number of threads per 1D block
  dim3 dim3dBlock(tpb_x, tpb_y, tpb_z);

  hipLaunchKernelGGL(Copy_Particles_Density_Kernel, dim3dGrid, dim3dBlock, 0, 0, Grav.F.density_d,
                     Particles.G.density_dev, nx_local, ny_local, nz_local, n_ghost);
}

  #endif  // GRAVITY_GPU

#endif  // PARTICLES

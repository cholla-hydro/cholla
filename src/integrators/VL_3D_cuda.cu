/*! \file VL_3D_cuda.cu
 *  \brief Definitions of the cuda 3 D VL algorithm functions. MHD algorithm
 *  from Stone & Gardiner 2009 "A simple unsplit Godunov method for
 *  multidimensional MHD"
 */

#if defined(CUDA) && defined(VL)

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../hydro/hydro_cuda.h"
  #include "../integrators/VL_3D_cuda.h"
  #include "../io/io.h"
  #include "../mhd/ct_electric_fields.h"
  #include "../mhd/magnetic_update.h"
  #include "../reconstruction/pcm_cuda.h"
  #include "../reconstruction/plmc_cuda.h"
  #include "../reconstruction/plmp_cuda.h"
  #include "../reconstruction/ppmc_cuda.h"
  #include "../reconstruction/ppmp_cuda.h"
  #include "../riemann_solvers/exact_cuda.h"
  #include "../riemann_solvers/hll_cuda.h"
  #include "../riemann_solvers/hllc_cuda.h"
  #include "../riemann_solvers/hlld_cuda.h"
  #include "../riemann_solvers/roe_cuda.h"
  #include "../utils/gpu.hpp"
  #include "../utils/hydro_utilities.h"

__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x,
                                                   Real *dev_F_y, Real *dev_F_z, int nx, int ny, int nz, int n_ghost,
                                                   Real dx, Real dy, Real dz, Real dt, Real gamma, int n_fields,
                                                   Real density_floor);

void VL_Algorithm_3D_CUDA(Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
                          int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound,
                          Real dt, int n_fields, Real density_floor, Real U_floor, Real *host_grav_potential)
{
  // Here, *dev_conserved contains the entire
  // set of conserved variables on the grid
  // concatenated into a 1-d array

  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;

  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  // host_grav_potential is NULL if not using GRAVITY
  temp_potential = host_grav_potential;

  if (!memory_allocated) {
    // allocate memory on the GPU
    dev_conserved = d_conserved;

  // Set the size of the interface and flux arrays
  #ifdef MHD
    // In MHD/Constrained Transport the interface arrays have one fewer fields
    // since the magnetic field that is stored on the face does not require
    // reconstructions. Similarly the fluxes have one fewer fields since the
    // magnetic field on that face doesn't have an associated flux. Each
    // interface array store the magnetic fields on that interface that are
    // not perpendicular to the interface and arranged cyclically. I.e. the
    // `Q_Lx` interface store the reconstructed Y and Z magnetic fields in
    // that order, the `Q_Ly` interface stores the Z and X mangetic fields in
    // that order, and the `Q_Lz` interface stores the X and Y magnetic fields
    // in that order. These fields can be indexed with the Q_?_dir grid_enums.
    // The interface state arrays store in the interface on the "right" side of
    // the cell, so the flux arrays store the fluxes through the right interface
    //
    // According to Stone et al. 2008 section 5.3 and the source code of
    // Athena, the following equation relate the magnetic flux to the face
    // centered electric fields/EMF. -cross(V,B)x is the negative of the
    // x-component of V cross B. Note that "X" is the direction the solver is
    // running in this case, not necessarily the true "X".
    //  F_x[(grid_enum::fluxX_magnetic_z)*n_cells] = VxBy - BxVy =
    //  -(-cross(V,B))z = -EMF_Z F_x[(grid_enum::fluxX_magnetic_y)*n_cells] =
    //  VxBz - BxVz =  (-cross(V,B))y =  EMF_Y
    //  F_y[(grid_enum::fluxY_magnetic_x)*n_cells] = VxBy - BxVy =
    //  -(-cross(V,B))z = -EMF_X F_y[(grid_enum::fluxY_magnetic_z)*n_cells] =
    //  VxBz - BxVz =  (-cross(V,B))y =  EMF_Z
    //  F_z[(grid_enum::fluxZ_magnetic_y)*n_cells] = VxBy - BxVy =
    //  -(-cross(V,B))z = -EMF_Y F_z[(grid_enum::fluxZ_magnetic_x)*n_cells] =
    //  VxBz - BxVz =  (-cross(V,B))y =  EMF_X
    size_t const arraySize   = (n_fields - 1) * n_cells * sizeof(Real);
    size_t const ctArraySize = 3 * n_cells * sizeof(Real);
  #else   // not MHD
    size_t const arraySize = n_fields * n_cells * sizeof(Real);
  #endif  // MHD
    CudaSafeCall(cudaMalloc((void **)&dev_conserved_half, n_fields * n_cells * sizeof(Real)));
    CudaSafeCall(cudaMalloc((void **)&Q_Lx, arraySize));
    CudaSafeCall(cudaMalloc((void **)&Q_Rx, arraySize));
    CudaSafeCall(cudaMalloc((void **)&Q_Ly, arraySize));
    CudaSafeCall(cudaMalloc((void **)&Q_Ry, arraySize));
    CudaSafeCall(cudaMalloc((void **)&Q_Lz, arraySize));
    CudaSafeCall(cudaMalloc((void **)&Q_Rz, arraySize));
    CudaSafeCall(cudaMalloc((void **)&F_x, arraySize));
    CudaSafeCall(cudaMalloc((void **)&F_y, arraySize));
    CudaSafeCall(cudaMalloc((void **)&F_z, arraySize));

    cuda_utilities::initGpuMemory(dev_conserved_half, n_fields * n_cells * sizeof(Real));
    cuda_utilities::initGpuMemory(Q_Lx, arraySize);
    cuda_utilities::initGpuMemory(Q_Rx, arraySize);
    cuda_utilities::initGpuMemory(Q_Ly, arraySize);
    cuda_utilities::initGpuMemory(Q_Ry, arraySize);
    cuda_utilities::initGpuMemory(Q_Lz, arraySize);
    cuda_utilities::initGpuMemory(Q_Rz, arraySize);
    cuda_utilities::initGpuMemory(F_x, arraySize);
    cuda_utilities::initGpuMemory(F_y, arraySize);
    cuda_utilities::initGpuMemory(F_z, arraySize);

  #ifdef MHD
    CudaSafeCall(cudaMalloc((void **)&ctElectricFields, ctArraySize));
  #endif  // MHD

  #if defined(GRAVITY)
    dev_grav_potential = d_grav_potential;
  #else   // not GRAVITY
    dev_grav_potential     = NULL;
  #endif  // GRAVITY

    // If memory is single allocated: memory_allocated becomes true and
    // successive timesteps won't allocate memory. If the memory is not single
    // allocated: memory_allocated remains Null and memory is allocated every
    // timestep.
    memory_allocated = true;
  }

  #if defined(GRAVITY) && !defined(GRAVITY_GPU)
  CudaSafeCall(cudaMemcpy(dev_grav_potential, temp_potential, n_cells * sizeof(Real), cudaMemcpyHostToDevice));
  #endif  // GRAVITY and GRAVITY_GPU

  // Step 1: Use PCM reconstruction to put primitive variables into interface
  // arrays
  hipLaunchKernelGGL(PCM_Reconstruction_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz,
                     Q_Rz, nx, ny, nz, n_ghost, gama, n_fields);
  CudaCheckError();

  // Step 2: Calculate first-order upwind fluxes
  #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
                     gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost,
                     gama, 2, n_fields);
  #endif  // EXACT
  #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
                     1, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama,
                     2, n_fields);
  #endif  // ROE
  #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
                     gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost,
                     gama, 2, n_fields);
  #endif  // HLLC
  #ifdef HLL
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
                     1, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama,
                     2, n_fields);
  #endif  // HLL
  #ifdef HLLD
  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx,
                     &(dev_conserved[(grid_enum::magnetic_x)*n_cells]), F_x, n_cells, gama, 0, n_fields);
  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry,
                     &(dev_conserved[(grid_enum::magnetic_y)*n_cells]), F_y, n_cells, gama, 1, n_fields);
  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz,
                     &(dev_conserved[(grid_enum::magnetic_z)*n_cells]), F_z, n_cells, gama, 2, n_fields);
  #endif  // HLLD
  CudaCheckError();

  #ifdef MHD
  // Step 2.5: Compute the Constrained transport electric fields
  hipLaunchKernelGGL(mhd::Calculate_CT_Electric_Fields, dim1dGrid, dim1dBlock, 0, 0, F_x, F_y, F_z, dev_conserved,
                     ctElectricFields, nx, ny, nz, n_cells);
  CudaCheckError();
  #endif  // MHD

  // Step 3: Update the conserved variables half a timestep
  hipLaunchKernelGGL(Update_Conserved_Variables_3D_half, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, dev_conserved_half,
                     F_x, F_y, F_z, nx, ny, nz, n_ghost, dx, dy, dz, 0.5 * dt, gama, n_fields, density_floor);
  CudaCheckError();
  #ifdef MHD
  // Update the magnetic fields
  hipLaunchKernelGGL(mhd::Update_Magnetic_Field_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, dev_conserved_half,
                     ctElectricFields, nx, ny, nz, n_cells, 0.5 * dt, dx, dy, dz);
  CudaCheckError();
  #endif  // MHD

  // Step 4: Construct left and right interface values using updated conserved
  // variables
  #ifdef PCM
  hipLaunchKernelGGL(PCM_Reconstruction_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, Q_Ly, Q_Ry,
                     Q_Lz, Q_Rz, nx, ny, nz, n_ghost, gama, n_fields);
  #endif  // PCM
  #ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx,
                     dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy,
                     dt, gama, 1, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz,
                     dt, gama, 2, n_fields);
  #endif  // PLMP
  #ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama,
                     0, n_fields);
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, dy, dt, gama,
                     1, n_fields);
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lz, Q_Rz, nx, ny, nz, dz, dt, gama,
                     2, n_fields);
  #endif  // PLMC
  #ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx,
                     dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy,
                     dt, gama, 1, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz,
                     dt, gama, 2, n_fields);
  #endif  // PPMP
  #ifdef PPMC
  hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx,
                     dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy,
                     dt, gama, 1, n_fields);
  hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz,
                     dt, gama, 2, n_fields);
  #endif  // PPMC
  CudaCheckError();

  // Step 5: Calculate the fluxes again
  #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
                     gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost,
                     gama, 2, n_fields);
  #endif  // EXACT
  #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
                     1, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama,
                     2, n_fields);
  #endif  // ROE
  #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
                     gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost,
                     gama, 2, n_fields);
  #endif  // HLLC
  #ifdef HLL
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
                     1, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama,
                     2, n_fields);
  #endif  // HLLC
  #ifdef HLLD
  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx,
                     &(dev_conserved_half[(grid_enum::magnetic_x)*n_cells]), F_x, n_cells, gama, 0, n_fields);
  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry,
                     &(dev_conserved_half[(grid_enum::magnetic_y)*n_cells]), F_y, n_cells, gama, 1, n_fields);
  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz,
                     &(dev_conserved_half[(grid_enum::magnetic_z)*n_cells]), F_z, n_cells, gama, 2, n_fields);
  #endif  // HLLD
  CudaCheckError();

  #ifdef DE
  // Compute the divergence of Vel before updating the conserved array, this
  // solves synchronization issues when adding this term on
  // Update_Conserved_Variables_3D
  hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx,
                     Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dx, dy, dz, dt, gama, n_fields);
  CudaCheckError();
  #endif  // DE

  #ifdef MHD
  // Step 5.5: Compute the Constrained transport electric fields
  hipLaunchKernelGGL(mhd::Calculate_CT_Electric_Fields, dim1dGrid, dim1dBlock, 0, 0, F_x, F_y, F_z, dev_conserved_half,
                     ctElectricFields, nx, ny, nz, n_cells);
  CudaCheckError();
  #endif  // MHD

  // Step 6: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry,
                     Q_Lz, Q_Rz, F_x, F_y, F_z, nx, ny, nz, x_off, y_off, z_off, n_ghost, dx, dy, dz, xbound, ybound,
                     zbound, dt, gama, n_fields, density_floor, dev_grav_potential);
  CudaCheckError();

  #ifdef MHD
  // Update the magnetic fields
  hipLaunchKernelGGL(mhd::Update_Magnetic_Field_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, dev_conserved,
                     ctElectricFields, nx, ny, nz, n_cells, dt, dx, dy, dz);
  CudaCheckError();
  #endif  // MHD

  #ifdef DE
  hipLaunchKernelGGL(Select_Internal_Energy_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost,
                     n_fields);
  hipLaunchKernelGGL(Sync_Energies_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, gama, n_fields);
  CudaCheckError();
  #endif  // DE

  #ifdef TEMPERATURE_FLOOR
  hipLaunchKernelGGL(Apply_Temperature_Floor, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields,
                     U_floor);
  CudaCheckError();
  #endif  // TEMPERATURE_FLOOR

  return;
}

void Free_Memory_VL_3D()
{
  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(dev_conserved_half);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(Q_Lz);
  cudaFree(Q_Rz);
  cudaFree(F_x);
  cudaFree(F_y);
  cudaFree(F_z);
  cudaFree(ctElectricFields);
}

__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x,
                                                   Real *dev_F_y, Real *dev_F_z, int nx, int ny, int nz, int n_ghost,
                                                   Real dx, Real dy, Real dz, Real dt, Real gamma, int n_fields,
                                                   Real density_floor)
{
  Real dtodx  = dt / dx;
  Real dtody  = dt / dy;
  Real dtodz  = dt / dz;
  int n_cells = nx * ny * nz;

  // get a global thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = tid / (nx * ny);
  int yid = (tid - zid * nx * ny) / nx;
  int xid = tid - zid * nx * ny - yid * nx;
  int id  = xid + yid * nx + zid * nx * ny;

  int imo = xid - 1 + yid * nx + zid * nx * ny;
  int jmo = xid + (yid - 1) * nx + zid * nx * ny;
  int kmo = xid + yid * nx + (zid - 1) * nx * ny;

  #ifdef DE
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo, P, E, E_kin, GE;
  int ipo, jpo, kpo;
  #endif  // DE

  #ifdef DENSITY_FLOOR
  Real dens_0;
  #endif  // DENSITY_FLOOR

  // threads corresponding to all cells except outer ring of ghost cells do the
  // calculation
  if (xid > 0 && xid < nx - 1 && yid > 0 && yid < ny - 1 && zid > 0 && zid < nz - 1) {
  #ifdef DE
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    // PRESSURE_DE
    E     = dev_conserved[4 * n_cells + id];
    GE    = dev_conserved[(n_fields - 1) * n_cells + id];
    E_kin = hydro_utilities::Calc_Kinetic_Energy_From_Velocity(d, vx, vy, vz);
    #ifdef MHD
    // Add the magnetic energy
    auto const [centeredBx, centeredBy, centeredBz] =
        mhd::utils::cellCenteredMagneticFields(dev_conserved, id, xid, yid, zid, n_cells, nx, ny) E_kin +=
        mhd::utils::computeMagneticEnergy(centeredBx, centeredBy, centeredBz);
    #endif  // MHD
    P = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, GE, gamma);
    P = fmax(P, (Real)TINY_NUMBER);
    // P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) *
    // (gamma - 1.0);
    // if (d < 0.0 || d != d) printf("Negative density before half step
    // update.\n"); if (P < 0.0) printf("%d Negative pressure before half step
    // update.\n", id);
    ipo    = xid + 1 + yid * nx + zid * nx * ny;
    jpo    = xid + (yid + 1) * nx + zid * nx * ny;
    kpo    = xid + yid * nx + (zid + 1) * nx * ny;
    vx_imo = dev_conserved[1 * n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1 * n_cells + ipo] / dev_conserved[ipo];
    vy_jmo = dev_conserved[2 * n_cells + jmo] / dev_conserved[jmo];
    vy_jpo = dev_conserved[2 * n_cells + jpo] / dev_conserved[jpo];
    vz_kmo = dev_conserved[3 * n_cells + kmo] / dev_conserved[kmo];
    vz_kpo = dev_conserved[3 * n_cells + kpo] / dev_conserved[kpo];
  #endif  // DE

    // update the conserved variable array
    dev_conserved_half[id] = dev_conserved[id] + dtodx * (dev_F_x[imo] - dev_F_x[id]) +
                             dtody * (dev_F_y[jmo] - dev_F_y[id]) + dtodz * (dev_F_z[kmo] - dev_F_z[id]);
    dev_conserved_half[n_cells + id] = dev_conserved[n_cells + id] +
                                       dtodx * (dev_F_x[n_cells + imo] - dev_F_x[n_cells + id]) +
                                       dtody * (dev_F_y[n_cells + jmo] - dev_F_y[n_cells + id]) +
                                       dtodz * (dev_F_z[n_cells + kmo] - dev_F_z[n_cells + id]);
    dev_conserved_half[2 * n_cells + id] = dev_conserved[2 * n_cells + id] +
                                           dtodx * (dev_F_x[2 * n_cells + imo] - dev_F_x[2 * n_cells + id]) +
                                           dtody * (dev_F_y[2 * n_cells + jmo] - dev_F_y[2 * n_cells + id]) +
                                           dtodz * (dev_F_z[2 * n_cells + kmo] - dev_F_z[2 * n_cells + id]);
    dev_conserved_half[3 * n_cells + id] = dev_conserved[3 * n_cells + id] +
                                           dtodx * (dev_F_x[3 * n_cells + imo] - dev_F_x[3 * n_cells + id]) +
                                           dtody * (dev_F_y[3 * n_cells + jmo] - dev_F_y[3 * n_cells + id]) +
                                           dtodz * (dev_F_z[3 * n_cells + kmo] - dev_F_z[3 * n_cells + id]);
    dev_conserved_half[4 * n_cells + id] = dev_conserved[4 * n_cells + id] +
                                           dtodx * (dev_F_x[4 * n_cells + imo] - dev_F_x[4 * n_cells + id]) +
                                           dtody * (dev_F_y[4 * n_cells + jmo] - dev_F_y[4 * n_cells + id]) +
                                           dtodz * (dev_F_z[4 * n_cells + kmo] - dev_F_z[4 * n_cells + id]);
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      dev_conserved_half[(5 + i) * n_cells + id] =
          dev_conserved[(5 + i) * n_cells + id] +
          dtodx * (dev_F_x[(5 + i) * n_cells + imo] - dev_F_x[(5 + i) * n_cells + id]) +
          dtody * (dev_F_y[(5 + i) * n_cells + jmo] - dev_F_y[(5 + i) * n_cells + id]) +
          dtodz * (dev_F_z[(5 + i) * n_cells + kmo] - dev_F_z[(5 + i) * n_cells + id]);
    }
  #endif  // SCALAR
  #ifdef DE
    dev_conserved_half[(n_fields - 1) * n_cells + id] =
        dev_conserved[(n_fields - 1) * n_cells + id] +
        dtodx * (dev_F_x[(n_fields - 1) * n_cells + imo] - dev_F_x[(n_fields - 1) * n_cells + id]) +
        dtody * (dev_F_y[(n_fields - 1) * n_cells + jmo] - dev_F_y[(n_fields - 1) * n_cells + id]) +
        dtodz * (dev_F_z[(n_fields - 1) * n_cells + kmo] - dev_F_z[(n_fields - 1) * n_cells + id]) +
        0.5 * P * (dtodx * (vx_imo - vx_ipo) + dtody * (vy_jmo - vy_jpo) + dtodz * (vz_kmo - vz_kpo));
  #endif  // DE

  #ifdef DENSITY_FLOOR
    if (dev_conserved_half[id] < density_floor) {
      dens_0 = dev_conserved_half[id];
      printf("###Thread density change  %f -> %f \n", dens_0, density_floor);
      dev_conserved_half[id] = density_floor;
      // Scale the conserved values to the new density
      dev_conserved_half[1 * n_cells + id] *= (density_floor / dens_0);
      dev_conserved_half[2 * n_cells + id] *= (density_floor / dens_0);
      dev_conserved_half[3 * n_cells + id] *= (density_floor / dens_0);
      dev_conserved_half[4 * n_cells + id] *= (density_floor / dens_0);
    #ifdef DE
      dev_conserved_half[(n_fields - 1) * n_cells + id] *= (density_floor / dens_0);
    #endif  // DE
    }
  #endif  // DENSITY_FLOOR
  }
}

#endif  // CUDA and VL

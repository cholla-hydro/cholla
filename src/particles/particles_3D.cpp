#ifdef PARTICLES

  #include "particles_3D.h"

  #include <unistd.h>

  #include <cmath>
  #include <cstdint>
  #include <functional>
  #include <random>

  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../model/disk_galaxy.h"
  #include "../utils/error_handling.h"
  #include "../utils/prng_utilities.h"

  #ifdef MPI_CHOLLA
    #include "../mpi/mpi_routines.h"
  #endif

  #ifdef PARALLEL_OMP
    #include "../utils/parallel_omp.h"
  #endif

Particles_3D::Particles_3D(void) : TRANSFER_DENSITY_BOUNDARIES(false), TRANSFER_PARTICLES_BOUNDARIES(false) {}

void Grid3D::Initialize_Particles(struct parameters *P)
{
  chprintf("\nInitializing Particles...\n");

  Particles.Initialize(P, Grav, H.xbound, H.ybound, H.zbound, H.xdglobal, H.ydglobal, H.zdglobal);

  #if defined(PARTICLES_GPU) && defined(GRAVITY_GPU)
  // Set the GPU array for the particles potential equal to the Gravity GPU
  // array for the potential
  Particles.G.potential_dev = Grav.F.potential_d;
  #endif

  if (strcmp(P->init, "Uniform") == 0) {
    Initialize_Uniform_Particles();
  }

  #ifdef MPI_CHOLLA
  MPI_Barrier(world);
  #endif
  chprintf("Particles Initialized Successfully. \n\n");
}

void Particles_3D::Initialize(struct parameters *P, Grav3D &Grav, Real xbound, Real ybound, Real zbound, Real xdglobal,
                              Real ydglobal, Real zdglobal)
{
  // Initialize local and total number of particles to 0
  n_local         = 0;
  n_total         = 0;
  n_total_initial = 0;

  // Initialize the simulation time and delta_t to 0
  dt = 0.0;
  t  = 0.0;
  // Set the maximum delta_t for particles, this can be changed depending on the
  // problem.
  max_dt = 10000;

  // Courant CFL condition factor for particles
  C_cfl = 0.3;

  #ifndef SINGLE_PARTICLE_MASS
  particle_mass = 0;  // The particle masses are stored in a separate array
  #endif

  #ifdef PARTICLES_CPU
  // Vectors for positions, velocities and accelerations
  real_vector_t pos_x;
  real_vector_t pos_y;
  real_vector_t pos_z;
  real_vector_t vel_x;
  real_vector_t vel_y;
  real_vector_t vel_z;
  real_vector_t grav_x;
  real_vector_t grav_y;
  real_vector_t grav_z;

    #ifndef SINGLE_PARTICLE_MASS
  // Vector for masses
  real_vector_t mass;
    #endif
    #ifdef PARTICLE_IDS
  // Vector for particle IDs
  int_vector_t partIDs;
    #endif
    #ifdef PARTICLE_AGE
  real_vector_t age;
    #endif

    #ifdef MPI_CHOLLA
  // Vectors for the indices of the particles that need to be transferred via
  // MPI
  int_vector_t out_indxs_vec_x0;
  int_vector_t out_indxs_vec_x1;
  int_vector_t out_indxs_vec_y0;
  int_vector_t out_indxs_vec_y1;
  int_vector_t out_indxs_vec_z0;
  int_vector_t out_indxs_vec_z1;
    #endif

  #endif  // PARTICLES_CPU

  // Initialize Grid Values
  // Local and total number of cells
  G.nx_local = Grav.nx_local;
  G.ny_local = Grav.ny_local;
  G.nz_local = Grav.nz_local;
  G.nx_total = Grav.nx_total;
  G.ny_total = Grav.ny_total;
  G.nz_total = Grav.nz_total;

  // Uniform (dx, dy, dz)
  G.dx = Grav.dx;
  G.dy = Grav.dy;
  G.dz = Grav.dz;

  // Left boundaries of the local domain
  G.xMin = Grav.xMin;
  G.yMin = Grav.yMin;
  G.zMin = Grav.zMin;

  // Right boundaries of the local domain
  G.xMax = Grav.xMax;
  G.yMax = Grav.yMax;
  G.zMax = Grav.zMax;

  // Left boundaries of the global domain
  G.domainMin_x = xbound;
  G.domainMin_y = ybound;
  G.domainMin_z = zbound;

  // Right boundaries of the global domain
  G.domainMax_x = xbound + xdglobal;
  G.domainMax_y = ybound + ydglobal;
  G.domainMax_z = zbound + zdglobal;

  // Number of ghost cells for the particles grid. For CIC one ghost cell is
  // needed
  G.n_ghost_particles_grid = 1;

  // Number of cells for the particles grid including ghost cells
  G.n_cells = (G.nx_local + 2 * G.n_ghost_particles_grid) * (G.ny_local + 2 * G.n_ghost_particles_grid) *
              (G.nz_local + 2 * G.n_ghost_particles_grid);

  // Set the boundary types
  #ifdef MPI_CHOLLA
  G.boundary_type_x0 = P->xlg_bcnd;
  G.boundary_type_x1 = P->xug_bcnd;
  G.boundary_type_y0 = P->ylg_bcnd;
  G.boundary_type_y1 = P->yug_bcnd;
  G.boundary_type_z0 = P->zlg_bcnd;
  G.boundary_type_z1 = P->zug_bcnd;
  #else
  G.boundary_type_x0        = P->xl_bcnd;
  G.boundary_type_x1        = P->xu_bcnd;
  G.boundary_type_y0        = P->yl_bcnd;
  G.boundary_type_y1        = P->yu_bcnd;
  G.boundary_type_z0        = P->zl_bcnd;
  G.boundary_type_z1        = P->zu_bcnd;
  #endif

  #ifdef PARTICLES_GPU
    // Factor to allocate the particles data arrays on the GPU.
    // When using MPI particles will be transferred to other GPU, for that
    // reason we need extra memory allocated
    #ifdef MPI_CHOLLA
  G.gpu_allocation_factor = 1.25;
    #else
  G.gpu_allocation_factor = 1.0;
    #endif

  G.size_blocks_array = 1024 * 128;
  G.n_cells_potential = (G.nx_local + 2 * N_GHOST_POTENTIAL) * (G.ny_local + 2 * N_GHOST_POTENTIAL) *
                        (G.nz_local + 2 * N_GHOST_POTENTIAL);

    #ifdef SINGLE_PARTICLE_MASS
  mass_dev = NULL;  // This array won't be used
    #endif

  #endif  // PARTICLES_GPU

  // Flags for Initial and tranfer the particles and density
  INITIAL                       = true;
  TRANSFER_DENSITY_BOUNDARIES   = false;
  TRANSFER_PARTICLES_BOUNDARIES = false;

  Allocate_Memory();

  // Initialize the particles density and gravitational field to 0.
  Initialize_Grid_Values();

  // Initialize Particles
  if (strcmp(P->init, "Spherical_Overdensity_3D") == 0) {
    Initialize_Sphere(P);
  } else if (strcmp(P->init, "Zeldovich_Pancake") == 0) {
    Initialize_Zeldovich_Pancake(P);
  } else if (strcmp(P->init, "Read_Grid") == 0) {
    Load_Particles_Data(P);
  #if defined(PARTICLE_AGE) && !defined(SINGLE_PARTICLE_MASS) && defined(PARTICLE_IDS)
  } else if (strcmp(P->init, "Disk_3D_particles") == 0) {
    Initialize_Disk_Stellar_Clusters(P);
  #endif
  }

  #ifdef MPI_CHOLLA
  n_total_initial = ReducePartIntSum(n_local);
  #else
  n_total_initial           = n_local;
  #endif

  chprintf("Particles Initialized: \n n_local: %lu \n", n_local);
  chprintf(" n_total: %lu \n", n_total_initial);
  chprintf(" xDomain_local:  [%.4f %.4f ] [%.4f %.4f ] [%.4f %.4f ]\n", G.xMin, G.xMax, G.yMin, G.yMax, G.zMin, G.zMax);
  chprintf(" xDomain_global: [%.4f %.4f ] [%.4f %.4f ] [%.4f %.4f ]\n", G.domainMin_x, G.domainMax_x, G.domainMin_y,
           G.domainMax_y, G.domainMin_z, G.domainMax_z);
  chprintf(" dx: %f  %f  %f\n", G.dx, G.dy, G.dz);

  #ifdef PARTICLE_IDS
  chprintf(" Tracking particle IDs\n");
  #endif

  #if defined(MPI_CHOLLA) && defined(PRINT_DOMAIN)
  for (int n = 0; n < nproc; n++) {
    if (procID == n)
      std::cout << procID << " x[" << G.xMin << "," << G.xMax << "] "
                << " y[" << G.yMin << "," << G.yMax << "] "
                << " z[" << G.zMin << "," << G.zMax << "] " << std::endl;
    usleep(100);
  }
  #endif

  #if defined(PARALLEL_OMP) && defined(PARTICLES_CPU)
  chprintf(" Using OMP for particles calculations\n");
  int n_omp_max = omp_get_max_threads();
  chprintf("  MAX OMP Threads: %d\n", n_omp_max);
  chprintf("  N OMP Threads per MPI process: %d\n", N_OMP_THREADS);
    #ifdef PRINT_OMP_DOMAIN
      // Print omp domain for each omp thread
      #pragma omp parallel num_threads(N_OMP_THREADS)
  {
    int omp_id, n_omp_procs;
    part_int_t omp_pIndx_start, omp_pIndx_end;
    omp_id      = omp_get_thread_num();
    n_omp_procs = omp_get_num_threads();
      #pragma omp barrier
    Get_OMP_Particles_Indxs(n_local, n_omp_procs, omp_id, &omp_pIndx_start, &omp_pIndx_end);

    for (int omp_indx = 0; omp_indx < n_omp_procs; omp_indx++) {
      if (omp_id == omp_indx)
        chprintf("  omp_id:%d  p_start:%ld  p_end:%ld  \n", omp_id, omp_pIndx_start, omp_pIndx_end);
    }
  }
    #endif  // PRINT_OMP_DOMAIN
  #endif    // PARALLEL_OMP

  #ifdef MPI_CHOLLA
  chprintf(" N_Particles Boundaries Buffer Size: %d\n", N_PARTICLES_TRANSFER);
  chprintf(" N_Data per Particle Transfer: %d\n", N_DATA_PER_PARTICLE_TRANSFER);

    #ifdef PARTICLES_GPU
  Allocate_Memory_GPU_MPI();
    #endif  // PARTICLES_GPU
  #endif    // MPI_CHOLLA
}

void Particles_3D::Allocate_Memory(void)
{
  // Allocate arrays for density and gravitational field

  G.density = (Real *)malloc(G.n_cells * sizeof(Real));
  #ifdef PARTICLES_CPU
  G.gravity_x = (Real *)malloc(G.n_cells * sizeof(Real));
  G.gravity_y = (Real *)malloc(G.n_cells * sizeof(Real));
  G.gravity_z = (Real *)malloc(G.n_cells * sizeof(Real));
    #ifdef GRAVITY_GPU
  // Array to copy the particles density to the device for computing the
  // potential in the device
  Allocate_Particles_Grid_Field_Real(&G.density_dev, G.n_cells);
    #endif
  #endif

  #ifdef PARTICLES_GPU
  Allocate_Memory_GPU();
  G.dti_array_host = (Real *)malloc(G.size_blocks_array * sizeof(Real));
  #endif
}

  #ifdef PARTICLES_GPU
void Particles_3D::Allocate_Memory_GPU()
{
  // Allocate arrays for density and gravitational field on the GPU

  Allocate_Particles_Grid_Field_Real(&G.density_dev, G.n_cells);
  Allocate_Particles_Grid_Field_Real(&G.gravity_x_dev, G.n_cells);
  Allocate_Particles_Grid_Field_Real(&G.gravity_y_dev, G.n_cells);
  Allocate_Particles_Grid_Field_Real(&G.gravity_z_dev, G.n_cells);
  Allocate_Particles_Grid_Field_Real(&G.dti_array_dev, G.size_blocks_array);
    #ifndef GRAVITY_GPU
  Allocate_Particles_Grid_Field_Real(&G.potential_dev, G.n_cells_potential);
    #endif
  chprintf(" Allocated GPU memory.\n");
}

part_int_t Particles_3D::Compute_Particles_GPU_Array_Size(part_int_t n)
{
  part_int_t buffer_size = n * G.gpu_allocation_factor;
  return buffer_size;
}

    #ifdef MPI_CHOLLA

void Particles_3D::ReAllocate_Memory_GPU_MPI()
{
  // Free the previous arrays
  Free_GPU_Array_bool(G.transfer_particles_flags_d);
  Free_GPU_Array_int(G.transfer_particles_indices_d);
  Free_GPU_Array_int(G.replace_particles_indices_d);
  Free_GPU_Array_int(G.transfer_particles_prefix_sum_d);
  Free_GPU_Array_int(G.transfer_particles_prefix_sum_blocks_d);

  // Allocate new resized arrays for the particles MPI transfers
  part_int_t buffer_size, half_blocks_size;
  buffer_size      = particles_array_size;
  half_blocks_size = ((buffer_size - 1) / 2) / TPB_PARTICLES + 1;
  Allocate_Particles_GPU_Array_bool(&G.transfer_particles_flags_d, buffer_size);
  Allocate_Particles_GPU_Array_int(&G.transfer_particles_indices_d, buffer_size);
  Allocate_Particles_GPU_Array_int(&G.replace_particles_indices_d, buffer_size);
  Allocate_Particles_GPU_Array_int(&G.transfer_particles_prefix_sum_d, buffer_size);
  Allocate_Particles_GPU_Array_int(&G.transfer_particles_prefix_sum_blocks_d, half_blocks_size);
  printf(" New allocation of arrays for particles transfers   new_size: %d \n", (int)buffer_size);
}

void Particles_3D::Allocate_Memory_GPU_MPI()
{
  // Allocate memory for the the particles MPI transfers
  part_int_t buffer_size, half_blocks_size;

  buffer_size      = Compute_Particles_GPU_Array_Size(n_local);
  half_blocks_size = ((buffer_size - 1) / 2) / TPB_PARTICLES + 1;

  Allocate_Particles_GPU_Array_bool(&G.transfer_particles_flags_d, buffer_size);
  Allocate_Particles_GPU_Array_int(&G.transfer_particles_indices_d, buffer_size);
  Allocate_Particles_GPU_Array_int(&G.replace_particles_indices_d, buffer_size);
  Allocate_Particles_GPU_Array_int(&G.transfer_particles_prefix_sum_d, buffer_size);
  Allocate_Particles_GPU_Array_int(&G.transfer_particles_prefix_sum_blocks_d, half_blocks_size);
  Allocate_Particles_GPU_Array_int(&G.n_transfer_d, 1);

  G.n_transfer_h = (int *)malloc(sizeof(int));

  // Used the global particles send/recv buffers that already have been
  // alloctaed in Allocate_MPI_DeviceBuffers_BLOCK
  G.send_buffer_size_x0 = buffer_length_particles_x0_send;
  G.send_buffer_size_x1 = buffer_length_particles_x1_send;
  G.send_buffer_size_y0 = buffer_length_particles_y0_send;
  G.send_buffer_size_y1 = buffer_length_particles_y1_send;
  G.send_buffer_size_z0 = buffer_length_particles_z0_send;
  G.send_buffer_size_z1 = buffer_length_particles_z1_send;

  G.send_buffer_x0_d = d_send_buffer_x0_particles;
  G.send_buffer_x1_d = d_send_buffer_x1_particles;
  G.send_buffer_y0_d = d_send_buffer_y0_particles;
  G.send_buffer_y1_d = d_send_buffer_y1_particles;
  G.send_buffer_z0_d = d_send_buffer_z0_particles;
  G.send_buffer_z1_d = d_send_buffer_z1_particles;

  G.recv_buffer_size_x0 = buffer_length_particles_x0_recv;
  G.recv_buffer_size_x1 = buffer_length_particles_x1_recv;
  G.recv_buffer_size_y0 = buffer_length_particles_y0_recv;
  G.recv_buffer_size_y1 = buffer_length_particles_y1_recv;
  G.recv_buffer_size_z0 = buffer_length_particles_z0_recv;
  G.recv_buffer_size_z1 = buffer_length_particles_z1_recv;

  G.recv_buffer_x0_d = d_recv_buffer_x0_particles;
  G.recv_buffer_x1_d = d_recv_buffer_x1_particles;
  G.recv_buffer_y0_d = d_recv_buffer_y0_particles;
  G.recv_buffer_y1_d = d_recv_buffer_y1_particles;
  G.recv_buffer_z0_d = d_recv_buffer_z0_particles;
  G.recv_buffer_z1_d = d_recv_buffer_z1_particles;
}
    #endif  // MPI_CHOLLA

void Particles_3D::Free_Memory_GPU()
{
  Free_GPU_Array_Real(G.density_dev);
  Free_GPU_Array_Real(G.gravity_x_dev);
  Free_GPU_Array_Real(G.gravity_y_dev);
  Free_GPU_Array_Real(G.gravity_z_dev);
  Free_GPU_Array_Real(G.dti_array_dev);

    #ifndef GRAVITY_GPU
  Free_GPU_Array_Real(G.potential_dev);
    #endif

  Free_GPU_Array_Real(pos_x_dev);
  Free_GPU_Array_Real(pos_y_dev);
  Free_GPU_Array_Real(pos_z_dev);
  Free_GPU_Array_Real(vel_x_dev);
  Free_GPU_Array_Real(vel_y_dev);
  Free_GPU_Array_Real(vel_z_dev);
  Free_GPU_Array_Real(grav_x_dev);
  Free_GPU_Array_Real(grav_y_dev);
  Free_GPU_Array_Real(grav_z_dev);
    #ifdef PARTICLE_IDS
  Free_GPU_Array(partIDs_dev);
    #endif
    #ifdef PARTICLE_AGE
  Free_GPU_Array_Real(age_dev);
    #endif
    #ifndef SINGLE_PARTICLE_MASS
  Free_GPU_Array_Real(mass_dev);
    #endif

    #ifdef MPI_CHOLLA
  Free_GPU_Array_bool(G.transfer_particles_flags_d);
  Free_GPU_Array_int(G.transfer_particles_prefix_sum_d);
  Free_GPU_Array_int(G.transfer_particles_prefix_sum_blocks_d);
  Free_GPU_Array_int(G.transfer_particles_indices_d);
  Free_GPU_Array_int(G.replace_particles_indices_d);
  Free_GPU_Array_int(G.n_transfer_d);
  free(G.n_transfer_h);

  Free_GPU_Array_Real(G.send_buffer_x0_d);
  Free_GPU_Array_Real(G.send_buffer_x1_d);
  Free_GPU_Array_Real(G.send_buffer_y0_d);
  Free_GPU_Array_Real(G.send_buffer_y1_d);
  Free_GPU_Array_Real(G.send_buffer_z0_d);
  Free_GPU_Array_Real(G.send_buffer_z1_d);

  Free_GPU_Array_Real(G.recv_buffer_x0_d);
  Free_GPU_Array_Real(G.recv_buffer_x1_d);
  Free_GPU_Array_Real(G.recv_buffer_y0_d);
  Free_GPU_Array_Real(G.recv_buffer_y1_d);
  Free_GPU_Array_Real(G.recv_buffer_z0_d);
  Free_GPU_Array_Real(G.recv_buffer_z1_d);

    #endif  // MPI_CHOLLA
}

  #endif  // PARTICLES_GPU

void Particles_3D::Initialize_Grid_Values(void)
{
  // Initialize density and gravitational field to 0.

  int id;
  for (id = 0; id < G.n_cells; id++) {
    G.density[id] = 0;
  #ifdef PARTICLES_CPU
    G.gravity_x[id] = 0;
    G.gravity_y[id] = 0;
    G.gravity_z[id] = 0;
  #endif
  }
}

void Particles_3D::Initialize_Sphere(struct parameters *P)
{
  // Initialize Random positions for sphere of quasi-uniform density
  chprintf(" Initializing Particles Uniform Sphere\n");

  int i, j, k, id;
  Real center_x, center_y, center_z, radius, sphereR;
  center_x = 0.5;
  center_y = 0.5;
  center_z = 0.5;
  ;
  sphereR = 0.2;

  // Set the number of particles equal to the number of grid cells
  part_int_t n_particles_local = G.nx_local * G.ny_local * G.nz_local;
  part_int_t n_particles_total = G.nx_total * G.ny_total * G.nz_total;

  // Set the initial density for the particles
  Real rho_start = 1;
  Real M_sphere  = 4. / 3 * M_PI * rho_start * sphereR * sphereR * sphereR;
  Real Mparticle = M_sphere / n_particles_total;

  #ifdef SINGLE_PARTICLE_MASS
  particle_mass = Mparticle;
  #endif

  #ifdef PARTICLES_GPU
  // Alocate memory in GPU for particle data
  particles_array_size = Compute_Particles_GPU_Array_Size(n_particles_local);
  Allocate_Particles_GPU_Array_Real(&pos_x_dev, particles_array_size);
  Allocate_Particles_GPU_Array_Real(&pos_y_dev, particles_array_size);
  Allocate_Particles_GPU_Array_Real(&pos_z_dev, particles_array_size);
  Allocate_Particles_GPU_Array_Real(&vel_x_dev, particles_array_size);
  Allocate_Particles_GPU_Array_Real(&vel_y_dev, particles_array_size);
  Allocate_Particles_GPU_Array_Real(&vel_z_dev, particles_array_size);
  Allocate_Particles_GPU_Array_Real(&grav_x_dev, particles_array_size);
  Allocate_Particles_GPU_Array_Real(&grav_y_dev, particles_array_size);
  Allocate_Particles_GPU_Array_Real(&grav_z_dev, particles_array_size);
    #ifndef SINGLE_PARTICLE_MASS
  Allocate_Particles_GPU_Array_Real(&mass_dev, particles_array_size);
    #endif
    #ifdef PARTICLE_IDS
  Allocate_Particles_GPU_Array_Part_Int(&partIDs_dev, particles_array_size);
    #endif
  n_local = n_particles_local;

  // Allocate temporal Host arrays for the particles data
  Real *temp_pos_x = (Real *)malloc(particles_array_size * sizeof(Real));
  Real *temp_pos_y = (Real *)malloc(particles_array_size * sizeof(Real));
  Real *temp_pos_z = (Real *)malloc(particles_array_size * sizeof(Real));
  Real *temp_vel_x = (Real *)malloc(particles_array_size * sizeof(Real));
  Real *temp_vel_y = (Real *)malloc(particles_array_size * sizeof(Real));
  Real *temp_vel_z = (Real *)malloc(particles_array_size * sizeof(Real));
    #ifndef SINGLE_PARTICLE_MASS
  Real *temp_mass = (Real *)malloc(particles_array_size * sizeof(Real));
    #endif
    #ifdef PARTICLE_IDS
  auto *temp_id = (part_int_t *)malloc(particles_array_size * sizeof(part_int_t));
    #endif

  chprintf(" Allocated GPU memory for particle data\n");
  #endif  // PARTICLES_GPU

  chprintf(" Initializing Random Positions\n");

  part_int_t pID = 0;
  Real pPos_x, pPos_y, pPos_z, r;
  std::mt19937_64 generator(P->prng_seed);
  std::uniform_real_distribution<Real> xPositionPrng(G.xMin, G.xMax);
  std::uniform_real_distribution<Real> yPositionPrng(G.yMin, G.yMax);
  std::uniform_real_distribution<Real> zPositionPrng(G.zMin, G.zMax);
  while (pID < n_particles_local) {
    pPos_x = xPositionPrng(generator);
    pPos_y = yPositionPrng(generator);
    pPos_z = zPositionPrng(generator);

    r = sqrt((pPos_x - center_x) * (pPos_x - center_x) + (pPos_y - center_y) * (pPos_y - center_y) +
             (pPos_z - center_z) * (pPos_z - center_z));
    if (r > sphereR) {
      continue;
    }

  #ifdef PARTICLES_CPU
    // Copy the particle data to the particles vectors
    pos_x.push_back(pPos_x);
    pos_y.push_back(pPos_y);
    pos_z.push_back(pPos_z);
    vel_x.push_back(0.0);
    vel_y.push_back(0.0);
    vel_z.push_back(0.0);
    grav_x.push_back(0.0);
    grav_y.push_back(0.0);
    grav_z.push_back(0.0);
    #ifdef PARTICLE_IDS
    partIDs.push_back(pID);
    #endif
    #ifndef SINGLE_PARTICLE_MASS
    mass.push_back(Mparticle);
    #endif
  #endif  // PARTICLES_CPU

  #ifdef PARTICLES_GPU
    // Copy the particle data to the temporal Host Buffers
    temp_pos_x[pID] = pPos_x;
    temp_pos_y[pID] = pPos_y;
    temp_pos_z[pID] = pPos_z;
    temp_vel_x[pID] = 0.0;
    temp_vel_y[pID] = 0.0;
    temp_vel_z[pID] = 0.0;
    #ifndef SINGLE_PARTICLE_MASS
    temp_mass[pID] = Mparticle;
    #endif
    #ifdef PARTICLE_IDS
    temp_id[pID] = pID;
    #endif
  #endif  // PARTICLES_GPU

    pID += 1;
  }

  #ifdef PARTICLES_CPU
  n_local = pos_x.size();
  #endif  // PARTICLES_CPU

  #if defined(PARTICLE_IDS) && defined(MPI_CHOLLA)
  // Get global IDs: Offset the local IDs to get unique global IDs across the
  // MPI ranks
  chprintf(" Computing Global Particles IDs offset \n");
  part_int_t global_id_offset;
  global_id_offset = Get_Particles_IDs_Global_MPI_Offset(n_local);
    #ifdef PARTICLES_CPU
  for (int p_indx = 0; p_indx < n_local; p_indx++) {
    partIDs[p_indx] += global_id_offset;
  }
    #endif  // PARTICLES_CPU
    #ifdef PARTICLES_GPU
  for (int p_indx = 0; p_indx < n_local; p_indx++) {
    temp_id[p_indx] += global_id_offset;
  }
    #endif  // PARTICLES_GPU
  #endif    // PARTICLE_IDS and MPI_CHOLLA

  #ifdef PARTICLES_GPU
  // Copyt the particle data from tepmpotal Host buffer to GPU memory
  Copy_Particles_Array_Real_Host_to_Device(temp_pos_x, pos_x_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device(temp_pos_y, pos_y_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device(temp_pos_z, pos_z_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device(temp_vel_x, vel_x_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device(temp_vel_y, vel_y_dev, n_local);
  Copy_Particles_Array_Real_Host_to_Device(temp_vel_z, vel_z_dev, n_local);
    #ifndef SINGLE_PARTICLE_MASS
  Copy_Particles_Array_Real_Host_to_Device(temp_mass, mass_dev, n_local);
    #endif
    #ifdef PARTICLE_IDS
  Copy_Particles_Array_Int_Host_to_Device(temp_id, partIDs_dev, n_local);
    #endif

  // Free the temporal host buffers
  free(temp_pos_x);
  free(temp_pos_y);
  free(temp_pos_z);
  free(temp_vel_x);
  free(temp_vel_y);
  free(temp_vel_z);
    #ifndef SINGLE_PARTICLE_MASS
  free(temp_mass);
    #endif
    #ifdef PARTICLE_IDS
  free(temp_id);
    #endif
  #endif  // PARTICLES_GPU

  chprintf(" Particles Uniform Sphere Initialized, n_local: %lu\n", n_local);
}

  #if defined(PARTICLE_AGE) && !defined(SINGLE_PARTICLE_MASS) && defined(PARTICLE_IDS)
/**
 *   Initializes a disk population of uniform mass stellar clusters
 */
void Particles_3D::Initialize_Disk_Stellar_Clusters(struct parameters *P)
{
  chprintf(" Initializing Particles Stellar Disk\n");

  // Set up the PRNG
  std::mt19937_64 generator(P->prng_seed);

  std::gamma_distribution<Real> radialDist(2, 1);  // for generating cyclindrical radii
  std::uniform_real_distribution<Real> zDist(-0.005, 0.005);
  std::uniform_real_distribution<Real> vzDist(-1e-8, 1e-8);
  std::uniform_real_distribution<Real> phiDist(0, 2 * M_PI);  // for generating phi
  std::normal_distribution<Real> speedDist(0, 1);             // for generating random speeds.

  Real M_d   = Galaxies::MW.getM_d();  // MW disk mass in M_sun (assumed to be all in stars)
  Real R_d   = Galaxies::MW.getR_d();  // MW stellar disk scale length in kpc
  Real Z_d   = Galaxies::MW.getZ_d();  // MW stellar height scale length in kpc
  Real R_max = P->xlen / 2.0 - 0.2;    // keep this consistent with Sigma_disk_D3D in disk_ICs.cpp

  real_vector_t temp_pos_x;
  real_vector_t temp_pos_y;
  real_vector_t temp_pos_z;
  real_vector_t temp_vel_x;
  real_vector_t temp_vel_y;
  real_vector_t temp_vel_z;
  real_vector_t temp_grav_x;
  real_vector_t temp_grav_y;
  real_vector_t temp_grav_z;
  real_vector_t temp_mass;
  int_vector_t temp_ids;
  real_vector_t temp_age;

  Real x, y, z, R, phi;
  Real vx, vy, vz, vel, ac;
  Real expFactor, vR_rms, vR, vPhi_str, vPhi, v_c2, vPhi_rand_rms, kappa2;
  // unsigned long int N = (long int)(6.5e6 * 0.11258580827352116);  //2kpc
  // radius unsigned long int N = 13; //(long int)(6.5e6 * 0.9272485558395908);
  // // 15kpc radius
  Real cumulative_mass           = 0;
  Real upper_limit_cluster_mass = 8e6;
  Real SFR                      = 1e2;
  Real t_cluster_creation       = -4e4;
  long lost_particles           = 0;
  part_int_t id                 = -1;
  while (cumulative_mass < upper_limit_cluster_mass) {
    Real cluster_mass = Galaxies::MW.singleClusterMass(generator);
    cumulative_mass += cluster_mass;
    id += 1;  // do this here before we check whether the particle is in the MPI
              // domain, otherwise could end up with duplicated IDs
    do {
      R = R_d * radialDist(generator);
    } while (R > R_max);

    phi = phiDist(generator);
    x   = R * cos(phi);
    y   = R * sin(phi);
    z   = zDist(generator);

    // set creation time of cluster on how long
    // it would take star formation to add that
    // much mass
    t_cluster_creation += cluster_mass / SFR;
    chprintf("cluster %d, age %.4e, mass %.4e\n", id, t_cluster_creation, cluster_mass);

    if (x < G.xMin || x >= G.xMax) continue;
    if (y < G.yMin || y >= G.yMax) continue;
    if (z < G.zMin || z >= G.zMax) continue;

    ac   = fabs(Galaxies::MW.gr_disk_D3D(R, 0) + Galaxies::MW.gr_halo_D3D(R, 0));
    vPhi = sqrt(R * ac);

    vx = -vPhi * sin(phi);
    vy = vPhi * cos(phi);
    vz = 0.0;  // vzDist(generator);

    // add particle data to the particles vectors
    temp_pos_x.push_back(x);
    temp_pos_y.push_back(y);
    temp_pos_z.push_back(z);
    temp_vel_x.push_back(vx);
    temp_vel_y.push_back(vy);
    temp_vel_z.push_back(vz);
    temp_grav_x.push_back(0.0);
    temp_grav_y.push_back(0.0);
    temp_grav_z.push_back(0.0);
    temp_mass.push_back(cluster_mass);
    temp_age.push_back(t_cluster_creation);
    temp_ids.push_back(id);
  }

  n_local = temp_pos_x.size();

    /*
      part_int_t global_id_offset = 0;
      #ifdef MPI_CHOLLA
      // Get global IDs: Offset the local IDs to get unique global IDs across
      the MPI ranks chprintf( " Computing Global Particles IDs offset \n" );
      global_id_offset = Get_Particles_IDs_Global_MPI_Offset( n_local );
      #endif //MPI_CHOLLA
      for ( int i=0; i<n_local; i++ ){
        temp_ids.push_back( i + global_id_offset);
      }
      */

    #ifdef PARTICLES_CPU
  pos_x   = temp_pos_x;
  pos_y   = temp_pos_y;
  pos_z   = temp_pos_z;
  vel_x   = temp_vel_x;
  vel_y   = temp_vel_y;
  vel_z   = temp_vel_z;
  grav_x  = temp_grav_x;
  grav_y  = temp_grav_y;
  grav_z  = temp_grav_z;
  mass    = temp_mass;
  partIDs = temp_ids;
  age     = temp_age;
    #endif  // PARTICLES_CPU

    #ifdef PARTICLES_GPU
  particles_array_size = Compute_Particles_GPU_Array_Size(n_local);
  Allocate_Particles_GPU_Array_Real(&pos_x_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_pos_x.data(), pos_x_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&pos_y_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_pos_y.data(), pos_y_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&pos_z_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_pos_z.data(), pos_z_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&vel_x_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_vel_x.data(), vel_x_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&vel_y_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_vel_y.data(), vel_y_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&vel_z_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_vel_z.data(), vel_z_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&grav_x_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_grav_x.data(), grav_x_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&grav_y_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_grav_y.data(), grav_y_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&grav_z_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_grav_z.data(), grav_z_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&mass_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_mass.data(), mass_dev, n_local);
  Allocate_Particles_GPU_Array_Part_Int(&partIDs_dev, particles_array_size);
  Copy_Particles_Array_Int_Host_to_Device(temp_ids.data(), partIDs_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&age_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(temp_age.data(), age_dev, n_local);
    #endif  // PARTICLES_GPU

  if (lost_particles > 0) {
    chprintf("  lost %lu particles\n", lost_particles);
  }
  chprintf(
      "Stellar Disk Particles Initialized, n_total: %lu, n_local: %lu, "
      "total_mass: %.3e s.m.\n",
      id + 1, n_local, cumulative_mass);
}
  #endif

void Particles_3D::Initialize_Zeldovich_Pancake(struct parameters *P)
{
  // No particles for the Zeldovich Pancake problem. n_local=0

  chprintf("Setting Zeldovich Pancake initial conditions...\n");

  // n_local = pos_x.size();
  n_local = 0;

  chprintf(" Particles Zeldovich Pancake Initialized, n_local: %lu\n", n_local);
}

void Grid3D::Initialize_Uniform_Particles()
{
  // Initialize positions assigning one particle at each cell in a uniform grid

  int i, j, k, id;
  Real x_pos, y_pos, z_pos;

  Real dVol, Mparticle;
  dVol      = H.dx * H.dy * H.dz;
  Mparticle = dVol;

  #ifdef SINGLE_PARTICLE_MASS
  Particles.particle_mass = Mparticle;
  #endif

  part_int_t pID = 0;
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        // // get the centered cell positions at (i,j,k)
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

  #ifdef PARTICLES_CPU
        Particles.pos_x.push_back(x_pos - 0.25 * H.dx);
        Particles.pos_y.push_back(y_pos - 0.25 * H.dy);
        Particles.pos_z.push_back(z_pos - 0.25 * H.dz);
        Particles.vel_x.push_back(0.0);
        Particles.vel_y.push_back(0.0);
        Particles.vel_z.push_back(0.0);
        Particles.grav_x.push_back(0.0);
        Particles.grav_y.push_back(0.0);
        Particles.grav_z.push_back(0.0);
    #ifdef PARTICLE_IDS
        Particles.partIDs.push_back(pID);
    #endif
    #ifndef SINGLE_PARTICLE_MASS
        Particles.mass.push_back(Mparticle);
    #endif
  #endif  // PARTICLES_CPU

        pID += 1;
      }
    }
  }

  #ifdef PARTICLES_CPU
  Particles.n_local = Particles.pos_x.size();
  #endif

  #ifdef MPI_CHOLLA
  Particles.n_total_initial = ReducePartIntSum(Particles.n_local);
  #else
  Particles.n_total_initial = Particles.n_local;
  #endif

  chprintf(" Particles Uniform Grid Initialized, n_local: %lu, n_total: %lu\n", Particles.n_local,
           Particles.n_total_initial);
}

void Particles_3D::Free_Memory(void)
{
  // Free the particles arrays
  free(G.density);

  #ifdef PARTICLES_CPU
  free(G.gravity_x);
  free(G.gravity_y);
  free(G.gravity_z);

  // Free the particles vectors
  pos_x.clear();
  pos_y.clear();
  pos_z.clear();
  vel_x.clear();
  vel_y.clear();
  vel_z.clear();
  grav_x.clear();
  grav_y.clear();
  grav_z.clear();

    #ifdef PARTICLE_IDS
  partIDs.clear();
    #endif

    #ifndef SINGLE_PARTICLE_MASS
  mass.clear();
    #endif

  #endif  // PARTICLES_CPU
}

void Particles_3D::Reset(void)
{
  Free_Memory();

  #ifdef PARTICLES_GPU
  free(G.dti_array_host);
  Free_Memory_GPU();
  #endif
}

#endif  // PARTICLES

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
  } else if (strcmp(P->init, "Isolated_Stellar_Cluster") == 0) {
    Initialize_Isolated_Stellar_Cluster(P);
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
  part_int_t *temp_id = (part_int_t *)malloc(particles_array_size * sizeof(part_int_t));
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

/* Actualy handles the initialization of stellar cluster particles based on their properties
 *
 * \note
 * Depending on how Cholla is compiled, it may mutate the real_props and int_props arguments.
 */
void Particles_3D::Initialize_Stellar_Clusters_Helper_(std::map<std::string, real_vector_t> &real_props,
                                                       std::map<std::string, int_vector_t> &int_props)
{
  // come up with a list of expected particle-property names. prop_name_l[i].first indicates the
  // name of a property and prop_name_l[i].second denotes whether it is Real
  const std::vector<std::pair<std::string, bool>> expected_prop_l = {
  #ifdef PARTICLE_AGE
      {"age", true},
  #endif
  #ifndef SINGLE_PARTICLE_MASS
      {"mass", true},
  #endif
  #ifdef PARTICLE_IDS
      {"id", false},
  #endif
      {"pos_x", true}, {"pos_y", true},  {"pos_z", true},  {"vel_x", true}, {"vel_y", true},
      {"vel_z", true}, {"grav_x", true}, {"grav_y", true}, {"grav_z", true}};

  // Here, we do some self-consistency checks!
  auto query_prop_len = [](const auto &prop_map, const std::string &name) -> part_int_t {
    auto rslt = prop_map.find(name);
    if (rslt == prop_map.end()) return -1;
    return part_int_t((rslt->second).size());
  };

  n_local = real_props.at("pos_x").size();

  std::size_t expected_real_prop_count = 0;
  std::size_t expected_int_prop_count  = 0;

  for (const auto &[name, is_real_prop] : expected_prop_l) {
    part_int_t cur_size = 0;
    if (is_real_prop) {
      expected_real_prop_count++;
      cur_size = query_prop_len(real_props, name);
    } else {
      expected_int_prop_count++;
      cur_size = query_prop_len(int_props, name);
    }

    // TODO: start using CHOLLA_ERROR in the following branches!
    if (cur_size < 0) {
      std::string type_string = (is_real_prop) ? "Real" : "int";
      chprintf("Expected the %s property to be specified as a %s-type property", name.c_str(), type_string.c_str());
      exit(1);
    } else if (cur_size != n_local) {
      chprintf("the %s property has a length of %lld. Other properties have a length of %lld", name.c_str(),
               (long long)(cur_size), (long long)(n_local));
      exit(1);
    }
  }

  if ((expected_int_prop_count != int_props.size()) or (expected_real_prop_count != real_props.size())) {
    // TODO: start using CHOLLA_ERROR
    chprintf(
        "%zu Real particle properties were provided -- %zu were expected. "
        "%zu integer particle properties were provided -- %zu were expected.",
        real_props.size(), expected_real_prop_count, int_props.size(), expected_int_prop_count);
    exit(1);
  }

  #ifdef PARTICLES_CPU
  // maybe use std::move in this cases to avoid the heap-allocation
    #ifdef(PARTICLE_AGE)
  this->age = real_props.at("age");
    #endif
    #ifndef SINGLE_PARTICLE_MASS
  this->mass = real_props.at("mass");
    #endif
    #ifdef PARTICLE_IDS
  this->partIDs = int_props.at("id");
    #endif
  this->pos_x  = real_props.at("pos_x");
  this->pos_y  = real_props.at("pos_y");
  this->pos_z  = real_props.at("pos_z");
  this->vel_x  = real_props.at("vel_x");
  this->vel_y  = real_props.at("vel_y");
  this->vel_z  = real_props.at("vel_z");
  this->grav_x = real_props.at("grav_x");
  this->grav_y = real_props.at("grav_y");
  this->grav_z = real_props.at("grav_z");
  #endif  // PARTICLES_CPU

  #ifdef PARTICLES_GPU
  particles_array_size = Compute_Particles_GPU_Array_Size(n_local);
    #ifdef PARTICLE_AGE
  Allocate_Particles_GPU_Array_Real(&age_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("age").data(), age_dev, n_local);
    #endif  // PARTICLE_AGE
    #ifndef SINGLE_PARTICLE_MASS
  Allocate_Particles_GPU_Array_Real(&mass_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("mass").data(), mass_dev, n_local);
    #endif  // SINGLE_PARTICLE_MASS
    #ifdef PARTICLE_IDS
  Allocate_Particles_GPU_Array_Part_Int(&partIDs_dev, particles_array_size);
  Copy_Particles_Array_Int_Host_to_Device(int_props.at("id").data(), partIDs_dev, n_local);
    #endif  // PARTICLE_IDS
  Allocate_Particles_GPU_Array_Real(&pos_x_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("pos_x").data(), pos_x_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&pos_y_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("pos_y").data(), pos_y_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&pos_z_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("pos_z").data(), pos_z_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&vel_x_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("vel_x").data(), vel_x_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&vel_y_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("vel_y").data(), vel_y_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&vel_z_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("vel_z").data(), vel_z_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&grav_x_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("grav_x").data(), grav_x_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&grav_y_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("grav_y").data(), grav_y_dev, n_local);
  Allocate_Particles_GPU_Array_Real(&grav_z_dev, particles_array_size);
  Copy_Particles_Array_Real_Host_to_Device(real_props.at("grav_z").data(), grav_z_dev, n_local);
  #endif  // PARTICLES_GPU
}

/* Initializes an isolated stellar cluster
 */
void Particles_3D::Initialize_Isolated_Stellar_Cluster(struct parameters *P)
{

  std::map<std::string, int_vector_t> int_props = {{"id", {}}};
  std::map<std::string, real_vector_t> real_props = {
    {"age", {}}, {"mass", {}}, {"pos_x", {}}, {"pos_y", {}}, {"pos_z", {}},
    {"vel_x", {}}, {"vel_y", {}}, {"vel_z", {}},
    {"grav_x", {}}, {"grav_y", {}}, {"grav_z", {}},
  };

  const double nominal_x = 0.0;
  const double nominal_y = 0.0;
  const double nominal_z = 0.0;

  if (((G.xMin <= nominal_x) and (nominal_x < G.xMax)) and
      ((G.yMin <= nominal_y) and (nominal_y < G.yMax)) and
      ((G.zMin <= nominal_z) and (nominal_z < G.zMax))) {
    int_props.at("id").push_back(0);
    real_props.at("age").push_back(-1e4); // 10 Myr (when positive, stars haven't formed yet)
    real_props.at("mass").push_back(2e4); // in solar masses I think this is reasonable?
    real_props.at("pos_x").push_back(nominal_x);
    real_props.at("pos_y").push_back(nominal_y);
    real_props.at("pos_z").push_back(nominal_z);
    const std::vector<std::string> axis_l = {"x", "y", "z"};
    for (const std::string& axis : axis_l) {
      real_props.at("vel_" + axis).push_back(0.0);
      real_props.at("grav_" + axis).push_back(0.0);
    }
  }

  this->Initialize_Stellar_Clusters_Helper_(real_props, int_props);
}

// An anonymous namespace to group together a bunch of local-only code used to help
// with initializing disk particles
namespace { // stuff inside the anonymous namespace is local-only

/* This class encapsulates the logic for determining the mass and formation times
 * of stellar cluster particles
 *
 * In short, we initialize an instance of this class. Then we call the next_cluster
 * method to determine the formation time and cluster-mass of the next cluster that
 * will be formed (it also updates the internal state of this object).
 *
 * \tparam UsePoissonPointProcess Specifies the strategy determining cluster formation times.
 *
 * Currently, there are 2 strategies for determing the properties of a newly formed
 * cluster. While both approaches randomly draw the cluster mass in the same way,
 * there are some differences in exactly how the formation times are determined:
 *   1. The legacy approach will have precomputed the formation time of a given cluster
 *      ahead of time. After determining the mass of the current cluster, it
 *      will set the formation time of the next cluster to the sum of the current
 *      cluster's formation time and `cur_cluster_mass/SFR`, where `SFR` specifies
 *      the desired global star formation rate
 *   2. The alternate apporach models the cluster formation rate as a Poisson process,
 *      with a cluster-formation-rate of `SFR/avg_cluster_mass`, where `avg_cluster_mass`
 *      is computed from the initial-cluster-mass PDF.
 *        - The delay in formation-times between different clusters is drawn from a
 *          separate distribution from the one used to determine the cluster mass.
 *        - One can show that this is strategy is equivalent to modelling N independent
 *          Poisson point-processes that each model that the rate of an individual cluster
 *          mass. Of course N would be a really large number (it approaches infinity).
 *
 * Both strategies should acheive the desired star-formation-rate. However, the second
 * strategy is arguably more robust (since the formation times of a cluster in one part
 * of the galaxy shouldn't really care about the size of the last cluster formed in a
 * different part of the galaxy)
 */
template<bool UsePoissonPointProcess>
class ClusterCreator{
public: // interface

  ClusterCreator() = delete;

  /* Primary constructor */
  ClusterCreator(const ClusterMassDistribution& cluster_mass_distribution,
                 Real SFR, Real earliest_t_formation)
    : cluster_mass_distribution_(cluster_mass_distribution),
      SFR_(SFR),
      cluster_formation_rate_(SFR / cluster_mass_distribution.meanClusterMass()),
      cached_formation_time_(earliest_t_formation)
  { }

  /* fetch the next cluster (and prepare internally for the following cluster)
   *
   * \param[in] generator Reference to the PRNG used for generating cluster masses
   * \param[out] t_formation_time The formation time of the next cluster
   * \param[out] cluster_mass The mass of the next cluster
   */
  void next_cluster(std::mt19937_64& generator,
                    Real& t_formation_time, Real& cluster_mass)
  {
    // first, determine the mass of the next cluster that will be formed
    cluster_mass = cluster_mass_distribution_.singleClusterMass(generator);

    // next, determine the formation time of that cluster & make the appropriate
    // updates to the internal state of this.
    if (UsePoissonPointProcess){
      // In a PoissonPoint Process with rate lambda = cluster_formation_rate_,
      // the holding times between points follows an exponential distribution
      // with lambda = cluster_formation_rate_
      std::exponential_distribution<Real> holding_times_(cluster_formation_rate_);

      // in this case, cached_formation_time_ stores the age of the last cluster
      t_formation_time = holding_times_(generator) + cached_formation_time_;
      cached_formation_time_ = t_formation_time;
    } else {
      // in this case, cached_formation_time_ already stores the age of this next
      // cluster
      t_formation_time = cached_formation_time_;
      cached_formation_time_ += cluster_mass / SFR_;
    }
  }

  /* returns time represents the earliest possible formation time of the next cluster */
  Real peek_next_min_formation_time() const { return cached_formation_time_; }

private: // attributes
  // we declare certain attributes as const to make it clear which ones are fixed
  // after initialization. The other ones represent mutable state.

  const ClusterMassDistribution cluster_mass_distribution_;
  /* represents the desired global star formation rate */
  const Real SFR_;
  /* represents the rate of forming clusters. This is computed from SFR_
   * and is only used when modelling cluster formation as a point process */
  const Real cluster_formation_rate_;

  /* This essentially encodes the state of the ClusterCreator
   * - when `UsePoissonPointProcess` is `true`, this represents the time at which
   *   the previous cluster formed
   * - otherwise, this directly stores the formation time of the next cluster
   */
  Real cached_formation_time_;
};

/* A simple struct used to hold the results of disk_star_cluster_init_ */
struct StarClusterInitRsltPack {
  std::map<std::string, int_vector_t> int_props;
  std::map<std::string, real_vector_t> real_props;
};

/* Helper function used to initialize the properties of stellar-clusters in a disk
 *
 * This initializes all necessary clusters in a simulation. The "age" property
 * effectively specifies the formation time. If the simulation time is smaller
 * than the "age" property, then the particle has not "formed yet".
 *
 * \param generator Reference to the PRNG used for initialization
 * \param R_max specifies the maximum radius (in code-units). Clusters initialized
 *     outside this radius are omitted from the simulation.
 * \param t_max specifies the final time at which we want to form a cluster. This
 *     typically coincides with the final simulation time.
 * \param G specifies the domain properties
 */
template<bool UsePoissonPointProcess = false>
StarClusterInitRsltPack disk_stellar_cluster_init_(std::mt19937_64& generator,
                                                   const Real R_max,
                                                   const Real t_max,
                                                   const Particles_3D::Grid& G)
{
  // todo: move away from using the distribution functions defined in the standard library
  //  -> apparently, the results of distribution functions are not portable across different
  //     implementions (the C++ standard was not precise enough to guarantee this)
  //  -> with that said, it's fine to use the generator classes (e.g. std::mt19937_64)
  const bool provide_summary = true;

  // fetch governing physical parameters:
  // todo: store the following directly within the Galaxy object
  const Real SFR               = 2e3;                            // global MW SFR: 2 SM / yr
  const Real Rgas_scale_length = Galaxies::MW.getGasDisk().R_d;  // gas-disk scale length
  // the following are theoretically tunable
  const Real k_s_power            = 1.4;     // the power in the Kennicut-Schmidt law
                                             // (at the moment, this isn't tunable)
  const Real earliest_t_formation = -4e4;    // earliest cluster time

  if (provide_summary) {
    chprintf(
      "Stellar Disk Particle Initialization:\n"
      "  SFR: %.3e Msun/kyr\n"
      "  Gas disk scale-length: %.3e kpc\n"
      "  Kennicutt–Schmidt power: %.3f\n"
      "  earliest-cluster-formation time: %.3e kyr\n",
      SFR, Rgas_scale_length, k_s_power, earliest_t_formation);
  }

  // define distribution for generating cyclindrical radii
  std::gamma_distribution<Real> radialDistHelper(2, 1);
  auto radialDist = [&radialDistHelper, Rgas_scale_length, k_s_power](std::mt19937_64& generator) -> Real {
    // we use the Kennicutt–Schmidt law to determine the distribution of particles with respect to r_cyl
    //   Sigma_SFR(r_cyl) = a * Sigma_gas^k_s_power,
    // where `a` is some arbitrary normalization constant and k_s_power is usually 1.4. We can combine
    // this with the formula for the gas surface density,
    //   Sigma_gas(r_cyl) = Sigma_gas0 * exp(-r_cyl / Rgas_scale_length),
    // to get a more detailed formula for star-formation surfacte denstiy:
    //   Sigma_SFR(r_cyl) = a * Sigma_gas^k_s_power * exp(-k_s_power * r_cyl / Rgas_scale_length).
    //
    // The incremental rate of star-formation dSFR in the disk between r_cyl and (r_cyl + dr_cyl) is
    // given by
    //   dSFR = a * Sigma_gas^k_s_power * exp(-k_s_power * r_cyl / Rgas_scale_length) * (2*pi*r_cyl*dr_cyl)
    // In principle, we could compute `a` by choosing a value that gives the desired global SFR rate
    // when we integrate dSFR from r_cyl = 0 to infinity. But that's not really necessary. Instead,
    // we consolidate all constants into a variable b:
    //   dSFR = b * r_cyl * exp(-k_s_power * r_cyl / Rgas_scale_length) * dr_cyl
    // If we multiply by some duration of time tau, the number of stars formed between r_cyl and
    // (r_cyl + dr_cyl) over that duration is:
    //   dN = tau*b * r_cyl * exp(-k_s_power * r_cyl / Rgas_scale_length) * dr_cyl
    // If you were to randomly pick a star formed over this duration, the probability that it would
    // lie between cylindrical radii R1 and R2 is
    //   prob = CONST * int_{R1}^{R2} r_cyl * exp(-k_s_power * r_cyl / Rgas_scale_length) * dr_cyl.
    // If u = (k_s_power * r_cyl / Rgas_scale_length), then this just the gamma distribution with
    // alpha = 2 and beta = 1.

    Real u = radialDistHelper(generator);      // draw u from gamma distribution
    return Rgas_scale_length * u / k_s_power;  // compute r_cyl from u
  };

  // define the other distributions
  std::uniform_real_distribution<Real> zDist(-0.005, 0.005);
  std::uniform_real_distribution<Real> vzDist(-1e-8, 1e-8);
  std::uniform_real_distribution<Real> phiDist(0, 2 * M_PI);  // for generating phi
  std::normal_distribution<Real> speedDist(0, 1);             // for generating random speeds.

  ClusterCreator<UsePoissonPointProcess> cluster_creator(Galaxies::MW.getClusterMassDistribution(),
                                                         SFR, earliest_t_formation);

  // initialize the std::map instances of vectors used to hold the output properties
  // part a: Initialize the maps in the output struct
  StarClusterInitRsltPack pack = {
    /* initialize int_props: */
    {{"id", {}}},
    /* initialize real_props */
    {{"age", {}}, {"mass", {}}, {"pos_x", {}}, {"pos_y", {}}, {"pos_z", {}},
     {"vel_x", {}}, {"vel_y", {}}, {"vel_z", {}},
     {"grav_x", {}}, {"grav_y", {}}, {"grav_z", {}}}
  };

  // part b: create references to each of vectors stored in the dictionare
  int_vector_t& temp_ids     = pack.int_props.at("id");
  real_vector_t& temp_age    = pack.real_props.at("age");
  real_vector_t& temp_mass   = pack.real_props.at("mass");
  real_vector_t& temp_pos_x  = pack.real_props.at("pos_x");
  real_vector_t& temp_pos_y  = pack.real_props.at("pos_y");
  real_vector_t& temp_pos_z  = pack.real_props.at("pos_z");
  real_vector_t& temp_vel_x  = pack.real_props.at("vel_x");
  real_vector_t& temp_vel_y  = pack.real_props.at("vel_y");
  real_vector_t& temp_vel_z  = pack.real_props.at("vel_z");
  real_vector_t& temp_grav_x = pack.real_props.at("grav_x");
  real_vector_t& temp_grav_y = pack.real_props.at("grav_y");
  real_vector_t& temp_grav_z = pack.real_props.at("grav_z");

  // initialize accumulator variables updated throughout the loop
  Real cumulative_mass = 0;
  long lost_particles  = 0;
  part_int_t id        = -1;

  // now actually initialize the clusters
  if (SFR > 0.0) {
    while (cluster_creator.peek_next_min_formation_time() < t_max) {
      Real t_cluster, cluster_mass;
      cluster_creator.next_cluster(generator, t_cluster, cluster_mass);

      Real R = radialDist(generator);
      if (R > R_max) continue;

      id += 1;  // do this here before we check whether the particle is in the MPI
                // domain, otherwise could end up with duplicated IDs
      Real phi = phiDist(generator);
      Real x   = R * cos(phi);
      Real y   = R * sin(phi);
      Real z   = zDist(generator);

      cumulative_mass += cluster_mass;
      if ((x < G.xMin || x >= G.xMax) || (y < G.yMin || y >= G.yMax) || (z < G.zMin || z >= G.zMax)) {
        continue;
      }

      Real vPhi;
      if (true) {
        vPhi = std::sqrt(Galaxies::MW.circular_vel2_with_selfgrav_estimates(R, z));
      } else {
        vPhi = std::sqrt(Galaxies::MW.circular_vel2(R, z));
      }

      Real vx = -vPhi * sin(phi);
      Real vy = vPhi * cos(phi);
      Real vz = 0.0;  // vzDist(generator);

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
      temp_age.push_back(t_cluster);
      temp_ids.push_back(id);
    } // end of while-loop

  }

  // print out summary:
  if (provide_summary) {
    chprintf(
      "Disk Particle Statistics: \n"
      "  lost %lu particles\n"
      "  n_total = %lu, n_local = %zu, total_mass = %.3e s.m.\n",
      lost_particles, id + 1, temp_ids.size(), cumulative_mass
    );
  }

  return pack;
}

} // anonymous namespace

  #if defined(PARTICLE_AGE) && !defined(SINGLE_PARTICLE_MASS) && defined(PARTICLE_IDS)

/**
 *   Initializes a disk population of stellar clusters
 */
void Particles_3D::Initialize_Disk_Stellar_Clusters(struct parameters *P)
{
  chprintf(" Initializing Particles Stellar Disk\n");

  // Set up the PRNG
  std::mt19937_64 generator(P->prng_seed);

  Real R_max = Get_StarCluster_Truncation_Radius(*P);
  Real t_max = P->tout;

  // in the future, we may want to let users adjust this parameter OR we may want
  // to remove the older strategy of determining cluster formation times
  bool poisson_process_formation_strat = true;

  StarClusterInitRsltPack pack = (poisson_process_formation_strat)
    ? disk_stellar_cluster_init_<true>(generator, R_max, t_max, this->G)
    : disk_stellar_cluster_init_<false>(generator, R_max, t_max, this->G);

  this->n_local = pack.int_props.at("id").size();  // may be a little redundant

  this->Initialize_Stellar_Clusters_Helper_(pack.real_props, pack.int_props);

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

#ifdef MPI_CHOLLA
  #include "../mpi/mpi_routines.h"

  #include <math.h>
  #include <mpi.h>

  #include <iostream>
  #include <tuple>

  #include "../global/global.h"
  #include "../io/io.h"
  #include "../mpi/cuda_mpi_routines.h"
  #include "../utils/error_handling.h"

/*Global MPI Variables*/
int procID; /*process rank*/
int nproc;  /*number of processes in global comm*/
int root;   /*rank of root process*/

int procID_node; /*process rank on node*/
int nproc_node;  /*number of MPI processes on node*/

MPI_Comm world; /*global communicator*/

MPI_Datatype MPI_CHREAL; /*set equal to MPI_FLOAT or MPI_DOUBLE*/

  #ifdef PARTICLES
MPI_Datatype MPI_PART_INT; /*set equal to MPI_INT or MPI_LONG*/
  #endif

// MPI_Requests for nonblocking comm
MPI_Request *send_request;
MPI_Request *recv_request;

// MPI destinations and sources
int dest[6];
int source[6];

// Communication buffers

// For BLOCK
Real *d_send_buffer_x0;
Real *d_send_buffer_x1;
Real *d_send_buffer_y0;
Real *d_send_buffer_y1;
Real *d_send_buffer_z0;
Real *d_send_buffer_z1;
Real *d_recv_buffer_x0;
Real *d_recv_buffer_x1;
Real *d_recv_buffer_y0;
Real *d_recv_buffer_y1;
Real *d_recv_buffer_z0;
Real *d_recv_buffer_z1;

Real *h_send_buffer_x0;
Real *h_send_buffer_x1;
Real *h_send_buffer_y0;
Real *h_send_buffer_y1;
Real *h_send_buffer_z0;
Real *h_send_buffer_z1;
Real *h_recv_buffer_x0;
Real *h_recv_buffer_x1;
Real *h_recv_buffer_y0;
Real *h_recv_buffer_y1;
Real *h_recv_buffer_z0;
Real *h_recv_buffer_z1;

int send_buffer_length;
int recv_buffer_length;
int x_buffer_length;
int y_buffer_length;
int z_buffer_length;

  #ifdef PARTICLES
// Buffers for particles transfers
Real *d_send_buffer_x0_particles;
Real *d_send_buffer_x1_particles;
Real *d_send_buffer_y0_particles;
Real *d_send_buffer_y1_particles;
Real *d_send_buffer_z0_particles;
Real *d_send_buffer_z1_particles;
Real *d_recv_buffer_x0_particles;
Real *d_recv_buffer_x1_particles;
Real *d_recv_buffer_y0_particles;
Real *d_recv_buffer_y1_particles;
Real *d_recv_buffer_z0_particles;
Real *d_recv_buffer_z1_particles;

// Buffers for particles transfers
Real *h_send_buffer_x0_particles;
Real *h_send_buffer_x1_particles;
Real *h_send_buffer_y0_particles;
Real *h_send_buffer_y1_particles;
Real *h_send_buffer_z0_particles;
Real *h_send_buffer_z1_particles;
Real *h_recv_buffer_x0_particles;
Real *h_recv_buffer_x1_particles;
Real *h_recv_buffer_y0_particles;
Real *h_recv_buffer_y1_particles;
Real *h_recv_buffer_z0_particles;
Real *h_recv_buffer_z1_particles;

// Size of the buffers for particles transfers
int buffer_length_particles_x0_send;
int buffer_length_particles_x0_recv;
int buffer_length_particles_x1_send;
int buffer_length_particles_x1_recv;
int buffer_length_particles_y0_send;
int buffer_length_particles_y0_recv;
int buffer_length_particles_y1_send;
int buffer_length_particles_y1_recv;
int buffer_length_particles_z0_send;
int buffer_length_particles_z0_recv;
int buffer_length_particles_z1_send;
int buffer_length_particles_z1_recv;

// Request for Number Of Particles to be transferred
MPI_Request *send_request_n_particles;
MPI_Request *recv_request_n_particles;
// Request for Particles Transfer
MPI_Request *send_request_particles_transfer;
MPI_Request *recv_request_particles_transfer;
  #endif  // PARTICLES

/*local domain sizes*/
/*none of these include ghost cells!*/
ptrdiff_t nx_global;
ptrdiff_t ny_global;
ptrdiff_t nz_global;
ptrdiff_t nx_local;
ptrdiff_t ny_local;
ptrdiff_t nz_local;
ptrdiff_t nx_local_start;
ptrdiff_t ny_local_start;
ptrdiff_t nz_local_start;

/*number of MPI procs in each dimension*/
int nproc_x;
int nproc_y;
int nproc_z;

  #ifdef FFTW
ptrdiff_t n_local_complex;
  #endif /*FFTW*/

/*\fn void InitializeChollaMPI(void) */
/* Routine to initialize MPI */
void InitializeChollaMPI(int *pargc, char **pargv[])
{
  /*initialize MPI*/
  MPI_Init(pargc, pargv);

  /*set process ids in comm world*/
  MPI_Comm_rank(MPI_COMM_WORLD, &procID);

  /*find number of processes in comm world*/
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  /*print a cute message*/
  // printf("Processor %d of %d: Hello!\n", procID, nproc);

  /* set the root process rank */
  root = 0;

  /* set the global communicator */
  world = MPI_COMM_WORLD;

  /* set the precision of MPI floating point numbers */
  #if PRECISION == 1
  MPI_CHREAL = MPI_FLOAT;
  #endif /*PRECISION*/
  #if PRECISION == 2
  MPI_CHREAL = MPI_DOUBLE;
  #endif /*PRECISION*/

  #ifdef PARTICLES
    #ifdef PARTICLES_LONG_INTS
  MPI_PART_INT = MPI_LONG;
    #else
  MPI_PART_INT = MPI_INT;
    #endif
  #endif

  /*create the MPI_Request arrays for non-blocking sends. If the malloc fails then print an error and exit*/
  send_request = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
  if (!send_request) {
    chprintf("Error allocating send_request.\n");
    chexit(-2);
  }
  recv_request = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
  if (!recv_request) {
    chprintf("Error allocating recv_request.\n");
    chexit(-2);
  }

  #ifdef PARTICLES
  send_request_n_particles = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
  if (!send_request_n_particles) {
    chprintf(
        "Error allocating send_request for number of particles for "
        "transfer.\n");
    chexit(-2);
  }
  recv_request_n_particles = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
  if (!recv_request_n_particles) {
    chprintf(
        "Error allocating recv_request for number of particles for "
        "transfer.\n");
    chexit(-2);
  }
  send_request_particles_transfer = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
  if (!send_request_particles_transfer) {
    chprintf("Error allocating send_request for particles transfer.\n");
    chexit(-2);
  }
  recv_request_particles_transfer = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
  if (!recv_request_particles_transfer) {
    chprintf("Error allocating recv_request for particles transfer.\n");
    chexit(-2);
  }
  #endif

  /*set up node communicator*/
  std::tie(procID_node, nproc_node) = MPI_Comm_node();

  // #ifdef ONLY_PARTICLES
  // chprintf("ONLY_PARTICLES: Initializing without CUDA support.\n");
  // #else
  // #ifndef GRAVITY
  // // Needed to initialize cuda after gravity in order to work on Summit
  // //initialize cuda for use with mpi
  #ifdef CUDA
  if (initialize_cuda_mpi(procID_node, nproc_node)) {
    chprintf("Error initializing cuda with mpi.\n");
    chexit(-10);
  }
  #endif /*CUDA*/
  // #endif//ONLY_PARTICLES
}

/* Perform domain decomposition */
void DomainDecomposition(struct parameters *P, struct Header *H, int nx_gin, int ny_gin, int nz_gin)
{
  DomainDecompositionBLOCK(P, H, nx_gin, ny_gin, nz_gin);

  // set grid dimensions
  H->nx      = nx_local + 2 * H->n_ghost;
  H->nx_real = nx_local;
  if (ny_local == 1) {
    H->ny = 1;
  } else {
    H->ny = ny_local + 2 * H->n_ghost;
  }
  H->ny_real = ny_local;
  if (nz_local == 1) {
    H->nz = 1;
  } else {
    H->nz = nz_local + 2 * H->n_ghost;
  }
  H->nz_real = nz_local;

  // set total number of cells
  H->n_cells = H->nx * H->ny * H->nz;

  // printf("In DomainDecomposition: nx %d ny %d nz %d nc
  // %d\n",H->nx,H->ny,H->nz,H->n_cells);

  // Allocate communication buffers
  Allocate_MPI_DeviceBuffers(H);
}

/* Perform domain decomposition */
void DomainDecompositionBLOCK(struct parameters *P, struct Header *H, int nx_gin, int ny_gin, int nz_gin)
{
  int n;
  int i, j, k;
  int *ix;
  int *iy;
  int *iz;

  // enforce an even number of processes
  if (nproc % 2 && nproc > 1) {
    chprintf("WARNING: Odd number of processors > 1 is not officially supported\n");
  }

  /* record global size */
  nx_global = nx_gin;
  ny_global = ny_gin;
  nz_global = nz_gin;

  /*allocate subdomain indices*/
  ix = (int *)malloc(nproc * sizeof(int));
  iy = (int *)malloc(nproc * sizeof(int));
  iz = (int *)malloc(nproc * sizeof(int));

  /*tile the MPI processes in blocks*/
  /*this sets nproc_x, nproc_y, nproc_z */
  // chprintf("About to enter tiling block decomp\n");
  MPI_Barrier(world);
  TileBlockDecomposition();

  if (nz_global > nx_global) {
    int tmp;
    tmp     = nproc_x;
    nproc_x = nproc_z;
    nproc_z = tmp;
  }

  #ifdef SET_MPI_GRID
  // Set the MPI Processes grid [n_proc_x, n_proc_y, n_proc_z]
  nproc_x = P->n_proc_x;
  nproc_y = P->n_proc_y;
  nproc_z = P->n_proc_z;
  chprintf("Setting MPI grid: nx=%d  ny=%d  nz=%d\n", nproc_x, nproc_y, nproc_z);
  // chprintf("Setting MPI grid: nx=%d  ny=%d  nz=%d\n", P->n_proc_x,
  // P->n_proc_y, P->n_proc_z);
  #endif

  // chprintf("Allocating tiling.\n");
  MPI_Barrier(world);
  int ***tiling = three_dimensional_int_array(nproc_x, nproc_y, nproc_z);

  // find indices
  // chprintf("Setting indices.\n");
  MPI_Barrier(world);
  n = 0;
  // Gravity: Change the order of MPI processes assignment to match the
  // assignment done by PFFT Original:
  //  for(i=0;i<nproc_x;i++)
  //    for(j=0;j<nproc_y;j++)
  //      for(k=0;k<nproc_z;k++)
  //

  for (k = 0; k < nproc_z; k++) {
    for (j = 0; j < nproc_y; j++) {
      for (i = 0; i < nproc_x; i++) {
        ix[n] = i;
        iy[n] = j;
        iz[n] = k;

        tiling[i][j][k] = n;

        if (n == procID) {
          dest[0] = i - 1;
          if (dest[0] < 0) {
            dest[0] += nproc_x;
          }
          dest[1] = i + 1;
          if (dest[1] >= nproc_x) {
            dest[1] -= nproc_x;
          }

          dest[2] = j - 1;
          if (dest[2] < 0) {
            dest[2] += nproc_y;
          }
          dest[3] = j + 1;
          if (dest[3] >= nproc_y) {
            dest[3] -= nproc_y;
          }

          dest[4] = k - 1;
          if (dest[4] < 0) {
            dest[4] += nproc_z;
          }
          dest[5] = k + 1;
          if (dest[5] >= nproc_z) {
            dest[5] -= nproc_z;
          }
        }
        n++;
      }
    }
  }

  /* set local x, y, z subdomain sizes */
  n = nx_global % nproc_x;
  if (!n) {
    // nx_global splits evenly along x procs*/
    nx_local       = nx_global / nproc_x;
    nx_local_start = ix[procID] * nx_local;
  } else {
    nx_local = nx_global / nproc_x;
    if (ix[procID] < n) {
      nx_local++;
      nx_local_start = ix[procID] * nx_local;
    } else {
      // check nx_local_start offsets -- should n be (n-1) below?
      nx_local_start = n * (nx_local + 1) + (ix[procID] - n) * nx_local;
    }
  }
  n = ny_global % nproc_y;
  if (!n) {
    // ny_global splits evenly along y procs*/
    ny_local       = ny_global / nproc_y;
    ny_local_start = iy[procID] * ny_local;
  } else {
    ny_local = ny_global / nproc_y;
    if (iy[procID] < n) {
      ny_local++;
      ny_local_start = iy[procID] * ny_local;
    } else {
      ny_local_start = n * (ny_local + 1) + (iy[procID] - n) * ny_local;
    }
  }
  n = nz_global % nproc_z;
  if (!n) {
    // nz_global splits evenly along z procs*/
    nz_local       = nz_global / nproc_z;
    nz_local_start = iz[procID] * nz_local;
  } else {
    nz_local = nz_global / nproc_z;
    if (iz[procID] < n) {
      nz_local++;
      nz_local_start = iz[procID] * nz_local;
    } else {
      nz_local_start = n * (nz_local + 1) + (iz[procID] - n) * nz_local;
    }
  }

  // find MPI sources
  for (i = 0; i < 6; i++) {
    source[i] = dest[i];
  }

  // find MPI destinations
  dest[0] = tiling[dest[0]][iy[procID]][iz[procID]];
  dest[1] = tiling[dest[1]][iy[procID]][iz[procID]];
  dest[2] = tiling[ix[procID]][dest[2]][iz[procID]];
  dest[3] = tiling[ix[procID]][dest[3]][iz[procID]];
  dest[4] = tiling[ix[procID]][iy[procID]][dest[4]];
  dest[5] = tiling[ix[procID]][iy[procID]][dest[5]];

  source[0] = tiling[source[0]][iy[procID]][iz[procID]];
  source[1] = tiling[source[1]][iy[procID]][iz[procID]];
  source[2] = tiling[ix[procID]][source[2]][iz[procID]];
  source[3] = tiling[ix[procID]][source[3]][iz[procID]];
  source[4] = tiling[ix[procID]][iy[procID]][source[4]];
  source[5] = tiling[ix[procID]][iy[procID]][source[5]];

  chprintf("nproc_x %d nproc_y %d nproc_z %d\n", nproc_x, nproc_y, nproc_z);

  // free the tiling
  deallocate_three_dimensional_int_array(tiling, nproc_x, nproc_y, nproc_z);

  /*adjust boundary condition flags*/

  /*remember global boundary conditions*/
  P->xlg_bcnd = P->xl_bcnd;
  P->xug_bcnd = P->xu_bcnd;
  P->ylg_bcnd = P->yl_bcnd;
  P->yug_bcnd = P->yu_bcnd;
  P->zlg_bcnd = P->zl_bcnd;
  P->zug_bcnd = P->zu_bcnd;

  /*do x bcnds first*/
  /*exterior faces have to be treated separately*/
  /*as long as there is more than one cell in the x direction*/
  if (nproc_x != 1) {
    if ((ix[procID] == 0) || (ix[procID] == nproc_x - 1)) {
      if (ix[procID] == 0) {
        P->xu_bcnd = 5;
        // if the global bcnd is periodic, use MPI bcnds at ends
        if (P->xl_bcnd == 1) {
          P->xl_bcnd = 5;
        }
      } else {
        P->xl_bcnd = 5;
        // if the global bcnd is periodic, use MPI bcnds at ends
        if (P->xu_bcnd == 1) {
          P->xu_bcnd = 5;
        }
      }
    } else {
      // this is completely an interior cell
      // along the x direction, so
      // set both x bcnds to MPI bcnds
      P->xl_bcnd = 5;
      P->xu_bcnd = 5;
    }
  }

  /*do y bcnds next*/
  /*exterior faces have to be treated separately*/
  /*as long as there is more than one cell in the x direction*/
  if (nproc_y != 1) {
    if ((iy[procID] == 0) || (iy[procID] == nproc_y - 1)) {
      if (iy[procID] == 0) {
        P->yu_bcnd = 5;
        // if the global bcnd is periodic, use MPI bcnds at ends
        if (P->yl_bcnd == 1) {
          P->yl_bcnd = 5;
        }
      } else {
        P->yl_bcnd = 5;
        // if the global bcnd is periodic, use MPI bcnds at ends
        if (P->yu_bcnd == 1) {
          P->yu_bcnd = 5;
        }
      }
    } else {
      // this is completely an interior cell
      // along the y direction, so
      // set both y bcnds to MPI bcnds
      P->yl_bcnd = 5;
      P->yu_bcnd = 5;
    }
  }

  /*do z bcnds next*/
  /*exterior faces have to be treated separately*/
  /*as long as there is more than one cell in the x direction*/
  if (nproc_z != 1) {
    if ((iz[procID] == 0) || (iz[procID] == nproc_z - 1)) {
      if (iz[procID] == 0) {
        P->zu_bcnd = 5;
        // if the global bcnd is periodic, use MPI bcnds at ends
        if (P->zl_bcnd == 1) {
          P->zl_bcnd = 5;
        }
      } else {
        P->zl_bcnd = 5;
        // if the global bcnd is periodic, use MPI bcnds at ends
        if (P->zu_bcnd == 1) {
          P->zu_bcnd = 5;
        }
      }
    } else {
      // this is completely an interior cell
      // along the z direction, so
      // set both z bcnds to MPI bcnds
      P->zl_bcnd = 5;
      P->zu_bcnd = 5;
    }
  }

  // free indices
  free(ix);
  free(iy);
  free(iz);
}

void Allocate_MPI_DeviceBuffers(struct Header *H)
{
  int xbsize = 0, ybsize = 0, zbsize = 0;
  if (H->ny == 1 && H->nz == 1) {
    xbsize = H->n_fields * H->n_ghost;
    ybsize = 1;
    zbsize = 1;
  }
  // 2D
  if (H->ny > 1 && H->nz == 1) {
    xbsize = H->n_fields * H->n_ghost * (H->ny - 2 * H->n_ghost);
    ybsize = H->n_fields * H->n_ghost * (H->nx);
    zbsize = 1;
  }
  // 3D
  if (H->ny > 1 && H->nz > 1) {
    xbsize = H->n_fields * H->n_ghost * (H->ny - 2 * H->n_ghost) * (H->nz - 2 * H->n_ghost);
    ybsize = H->n_fields * H->n_ghost * (H->nx) * (H->nz - 2 * H->n_ghost);
    zbsize = H->n_fields * H->n_ghost * (H->nx) * (H->ny);
  }

  x_buffer_length = xbsize;
  y_buffer_length = ybsize;
  z_buffer_length = zbsize;

  #ifdef PARTICLES
  // Set Initial sizes for particles buffers
  int n_max            = std::max(H->nx, H->ny);
  n_max                = std::max(H->nz, n_max);
  int factor           = 2;
  N_PARTICLES_TRANSFER = n_max * n_max * factor;

  // Set the number of values that will be transferred for each particle
  N_DATA_PER_PARTICLE_TRANSFER = 6;  // 3 positions and 3 velocities
    #ifndef SINGLE_PARTICLE_MASS
  N_DATA_PER_PARTICLE_TRANSFER += 1;  // one more for the particle mass
    #endif
    #ifdef PARTICLE_IDS
  N_DATA_PER_PARTICLE_TRANSFER += 1;  // one more for the particle ID
    #endif
    #ifdef PARTICLE_AGE
  N_DATA_PER_PARTICLE_TRANSFER += 1;  // one more for the particle age
    #endif

  buffer_length_particles_x0_send = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_x0_recv = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_x1_send = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_x1_recv = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_y0_send = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_y0_recv = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_y1_send = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_y1_recv = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_z0_send = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_z0_recv = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_z1_send = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  buffer_length_particles_z1_recv = N_PARTICLES_TRANSFER * N_DATA_PER_PARTICLE_TRANSFER;
  #endif  // PARTICLES

  chprintf("Allocating MPI communication buffers on GPU ");
  chprintf("(nx = %ld, ny = %ld, nz = %ld).\n", xbsize, ybsize, zbsize);

  CudaSafeCall(cudaMalloc(&d_send_buffer_x0, xbsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_x1, xbsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_x0, xbsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_x1, xbsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_y0, ybsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_y1, ybsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_y0, ybsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_y1, ybsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_z0, zbsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_z1, zbsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_z0, zbsize * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_z1, zbsize * sizeof(Real)));

  #if !defined(MPI_GPU)
  h_send_buffer_x0 = (Real *)malloc(xbsize * sizeof(Real));
  h_send_buffer_x1 = (Real *)malloc(xbsize * sizeof(Real));
  h_recv_buffer_x0 = (Real *)malloc(xbsize * sizeof(Real));
  h_recv_buffer_x1 = (Real *)malloc(xbsize * sizeof(Real));
  h_send_buffer_y0 = (Real *)malloc(ybsize * sizeof(Real));
  h_send_buffer_y1 = (Real *)malloc(ybsize * sizeof(Real));
  h_recv_buffer_y0 = (Real *)malloc(ybsize * sizeof(Real));
  h_recv_buffer_y1 = (Real *)malloc(ybsize * sizeof(Real));
  h_send_buffer_z0 = (Real *)malloc(zbsize * sizeof(Real));
  h_send_buffer_z1 = (Real *)malloc(zbsize * sizeof(Real));
  h_recv_buffer_z0 = (Real *)malloc(zbsize * sizeof(Real));
  h_recv_buffer_z1 = (Real *)malloc(zbsize * sizeof(Real));
  #endif

  // NOTE: When changing this ifdef check for compatibility with
  // Grid3D::Load_NTtransfer_and_Request_Receive_Particles_Transfer
  // in particles/particles_boundaries.cpp

  // Whether or not MPI_GPU is on, the device has transfer buffers for
  // PARTICLES_GPU

  #if defined(PARTICLES) && defined(PARTICLES_GPU)
  chprintf(
      "Allocating MPI communication buffers on GPU for particle transfers ( "
      "N_Particles: %d ).\n",
      N_PARTICLES_TRANSFER);
  CudaSafeCall(cudaMalloc(&d_send_buffer_x0_particles, buffer_length_particles_x0_send * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_x1_particles, buffer_length_particles_x1_send * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_y0_particles, buffer_length_particles_y0_send * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_y1_particles, buffer_length_particles_y1_send * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_z0_particles, buffer_length_particles_z0_send * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_send_buffer_z1_particles, buffer_length_particles_z1_send * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_x0_particles, buffer_length_particles_x0_recv * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_x1_particles, buffer_length_particles_x1_recv * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_y0_particles, buffer_length_particles_y0_recv * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_y1_particles, buffer_length_particles_y1_recv * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_z0_particles, buffer_length_particles_z0_recv * sizeof(Real)));
  CudaSafeCall(cudaMalloc(&d_recv_buffer_z1_particles, buffer_length_particles_z1_recv * sizeof(Real)));
  #endif  // PARTICLES && PARTICLES_GPU

  // CPU relies on host buffers, GPU without MPI_GPU relies on host buffers

  #ifdef PARTICLES
    #if (defined(PARTICLES_GPU) && !defined(MPI_GPU)) || defined(PARTICLES_CPU)
  chprintf(
      "Allocating MPI communication buffers on Host for particle transfers ( "
      "N_Particles: %d ).\n",
      N_PARTICLES_TRANSFER);
  h_send_buffer_x0_particles = (Real *)malloc(buffer_length_particles_x0_send * sizeof(Real));
  h_send_buffer_x1_particles = (Real *)malloc(buffer_length_particles_x1_send * sizeof(Real));
  h_send_buffer_y0_particles = (Real *)malloc(buffer_length_particles_y0_send * sizeof(Real));
  h_send_buffer_y1_particles = (Real *)malloc(buffer_length_particles_y1_send * sizeof(Real));
  h_send_buffer_z0_particles = (Real *)malloc(buffer_length_particles_z0_send * sizeof(Real));
  h_send_buffer_z1_particles = (Real *)malloc(buffer_length_particles_z1_send * sizeof(Real));
  h_recv_buffer_x0_particles = (Real *)malloc(buffer_length_particles_x0_recv * sizeof(Real));
  h_recv_buffer_x1_particles = (Real *)malloc(buffer_length_particles_x1_recv * sizeof(Real));
  h_recv_buffer_y0_particles = (Real *)malloc(buffer_length_particles_y0_recv * sizeof(Real));
  h_recv_buffer_y1_particles = (Real *)malloc(buffer_length_particles_y1_recv * sizeof(Real));
  h_recv_buffer_z0_particles = (Real *)malloc(buffer_length_particles_z0_recv * sizeof(Real));
  h_recv_buffer_z1_particles = (Real *)malloc(buffer_length_particles_z1_recv * sizeof(Real));
    #endif  // (defined(PARTICLES_GPU) && !defined(MPI_GPU)) ||
            // defined(PARTICLES_CPU)
  #endif    // PARTICLES
}

/* MPI reduction wrapper for max(Real)*/
Real ReduceRealMax(Real x)
{
  Real in = x;
  Real out;
  Real y;

  MPI_Allreduce(&in, &out, 1, MPI_CHREAL, MPI_MAX, world);
  y = (Real)out;
  return y;
}

/* MPI reduction wrapper for min(Real)*/
Real ReduceRealMin(Real x)
{
  Real in = x;
  Real out;
  Real y;

  MPI_Allreduce(&in, &out, 1, MPI_CHREAL, MPI_MIN, world);
  y = (Real)out;
  return y;
}

/* MPI reduction wrapper for avg(Real)*/
Real ReduceRealAvg(Real x)
{
  Real in = x;
  Real out;
  Real y;

  MPI_Allreduce(&in, &out, 1, MPI_CHREAL, MPI_SUM, world);
  y = (Real)out / nproc;
  return y;
}

size_t Reduce_size_t_Max(size_t in)
{
  // Get the right MPI type
  #if SIZE_MAX == UCHAR_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
  #elif SIZE_MAX == USHRT_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
  #elif SIZE_MAX == UINT_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED
  #elif SIZE_MAX == ULONG_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
  #elif SIZE_MAX == ULLONG_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
  #else
    #error "Error: Type of size_t not supported by Reduce_size_t_Max"
  #endif

  // Perform the reduction
  size_t out;
  MPI_Allreduce(&in, &out, 1, my_MPI_SIZE_T, MPI_MAX, world);
  return out;
}

  #ifdef PARTICLES
/* MPI reduction wrapper for sum(part_int)*/
Real ReducePartIntSum(part_int_t x)
{
  part_int_t in = x;
  part_int_t out;
  part_int_t y;

    #ifdef PARTICLES_LONG_INTS
  MPI_Allreduce(&in, &out, 1, MPI_LONG, MPI_SUM, world);
    #else
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_SUM, world);
    #endif
  y = (part_int_t)out;
  return y;
}

// Count the particles in the MPI ranks lower than this rank (procID) to get a
// global offset for the local IDs.
part_int_t Get_Particles_IDs_Global_MPI_Offset(part_int_t n_local)
{
  part_int_t global_offset;
  part_int_t *n_local_all, *n_local_send;
  n_local_send    = (part_int_t *)malloc(1 * sizeof(part_int_t));
  n_local_all     = (part_int_t *)malloc(nproc * sizeof(part_int_t));
  n_local_send[0] = n_local;

  MPI_Allgather(n_local_send, 1, MPI_PART_INT, n_local_all, 1, MPI_PART_INT, world);
  global_offset = 0;
  for (int other_rank = 0; other_rank < nproc; other_rank++) {
    if (other_rank < procID) {
      global_offset += n_local_all[other_rank];
    }
  }
  // printf("global_offset = %ld \n", global_offset );
  free(n_local_send);
  free(n_local_all);
  return global_offset;
}

  #endif

/* Print information about the domain properties */
void Print_Domain_Properties(struct Header H)
{
  int i;
  fflush(stdout);
  MPI_Barrier(world);
  for (i = 0; i < nproc; i++) {
    if (i == procID) {
      printf("procID %d nxl %ld nxls %ld\n", procID, nx_local, nx_local_start);
      printf("xb %e yb %e zb %e xbl %e ybl %e zbl %e\n", H.xbound, H.ybound, H.zbound, H.xblocal, H.yblocal, H.zblocal);
      printf("dx %e\n", H.dx);
      printf("dy %e\n", H.dy);
      printf("dz %e\n", H.dz);
      printf("*********\n");
    }
    fflush(stdout);
    MPI_Barrier(world);
  }
}

  #ifdef PARTICLES
// Funtion that checks if the buffer size For the particles transfer is large
// enough, and grows the buffer if needed.
void Check_and_Grow_Particles_Buffer(Real **part_buffer, int *current_size_ptr, int new_size)
{
  int current_size = *current_size_ptr;
  if (new_size <= current_size) {
    return;
  }

  new_size = (int)2 * new_size;
  std::cout << " #######  Growing Particles Transfer Buffer, size: " << current_size << "  new_size: " << new_size
            << std::endl;

  Real *new_buffer;
  new_buffer = (Real *)realloc(*part_buffer, new_size * sizeof(Real));
  if (new_buffer == NULL) {
    std::cout << " Error When Allocating New Particles Transfer Buffer" << std::endl;
    chexit(-1);
  }
  *part_buffer      = new_buffer;
  *current_size_ptr = new_size;
}
  #endif  // PARTICLES

/* find the greatest prime factor of an integer */
int greatest_prime_factor(int n)
{
  int ns = n;
  int np = 2;

  if (n == 1 || n == 2) {
    return n;
  }

  while (true) {
    while (!(ns % np)) {
      ns = ns / np;
    }

    if (ns == 1) {
      break;
    }

    np++;
  }
  return np;
}

/*
 tile MPI processes in a block arrangement for the 3D case
 */
void TileBlockDecomposition3D(int number, int &np_x, int &np_y, int &np_z)
{
  int dims[3]  = {1, 1, 1};
  size_t index = 0;
  while (number > 1) {
    int gpf = greatest_prime_factor(number);
    number /= gpf;
    dims[index % 3] *= gpf;
    index += 1;
  }
  np_x = dims[0];
  np_y = dims[1];
  np_z = dims[2];
}

/*tile MPI processes in a block arrangement*/
void TileBlockDecomposition(void)
{
  int n_gpf;

  // initialize np_x, np_y, np_z
  int np_x = 1;
  int np_y = 1;
  int np_z = 1;
  // printf("nproc %d n_gpf %d\n",nproc,n_gpf);

  /* 1-D case is trivial */
  if (nz_global == 1 && ny_global == 1) {
    nproc_x = nproc;
    nproc_y = 1;
    nproc_z = 1;
    return;
  }

  /*find the greatest prime factor of the number of MPI processes*/
  n_gpf = greatest_prime_factor(nproc);

  /* 2-D case
     we can just assign domain*/

  if (nz_global == 1) {
    np_x = n_gpf;
    np_y = nproc / np_x;
    // ensure nproc_x > nproc_y
    if (np_x < np_y) {
      nproc_x = np_y;
      nproc_y = np_x;
    } else {
      nproc_x = np_x;
      nproc_y = np_y;
    }
    nproc_z = 1;
    return;
  }

  TileBlockDecomposition3D(nproc, np_x, np_y, np_z);
  // reorder x, y, z

  int n_tmp;
  if (np_z > np_y) {
    n_tmp = np_y;
    np_y  = np_z;
    np_z  = n_tmp;
  }
  if (np_y > np_x) {
    n_tmp = np_x;
    np_x  = np_y;
    np_y  = n_tmp;
  }
  if (np_z > np_y) {
    n_tmp = np_y;
    np_y  = np_z;
    np_z  = n_tmp;
  }

  // save result
  nproc_x = np_x;
  nproc_y = np_y;
  nproc_z = np_z;
}

/*! \fn int ***three_dimensional_int_array(int n, int l, int m)
 *  *  \brief Allocate a three dimensional (n x l x m) int array
 *   */
int ***three_dimensional_int_array(int n, int l, int m)
{
  int ***x;

  x = new int **[n];
  for (int i = 0; i < n; i++) {
    x[i] = new int *[l];
    for (int j = 0; j < l; j++) {
      x[i][j] = new int[m];
    }
  }

  return x;
}
/*! \fn void deallocate_three_int_dimensional_array(int ***x, int n, int l, int
 * m)
 *  *  \brief De-allocate a three dimensional (n x l x m) int array.
 *   */
void deallocate_three_dimensional_int_array(int ***x, int n, int l, int m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < l; j++) {
      delete[] x[i][j];
    }
    delete[] x[i];
  }
  delete x;
}

void copyHostToDeviceReceiveBuffer(int direction)
{
  int xbsize = x_buffer_length, ybsize = y_buffer_length, zbsize = z_buffer_length;

  switch (direction) {
    case (0):
      cudaMemcpy(d_recv_buffer_x0, h_recv_buffer_x0, xbsize * sizeof(Real), cudaMemcpyHostToDevice);
      break;
    case (1):
      cudaMemcpy(d_recv_buffer_x1, h_recv_buffer_x1, xbsize * sizeof(Real), cudaMemcpyHostToDevice);
      break;
    case (2):
      cudaMemcpy(d_recv_buffer_y0, h_recv_buffer_y0, ybsize * sizeof(Real), cudaMemcpyHostToDevice);
      break;
    case (3):
      cudaMemcpy(d_recv_buffer_y1, h_recv_buffer_y1, ybsize * sizeof(Real), cudaMemcpyHostToDevice);
      break;
    case (4):
      cudaMemcpy(d_recv_buffer_z0, h_recv_buffer_z0, zbsize * sizeof(Real), cudaMemcpyHostToDevice);
      break;
    case (5):
      cudaMemcpy(d_recv_buffer_z1, h_recv_buffer_z1, zbsize * sizeof(Real), cudaMemcpyHostToDevice);
      break;
  }
}

std::pair<int, int> MPI_Comm_node()
{
  // get the global process rank
  int myid, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // if there is the only one process, then just return the global rank and size
  if (nproc == 1) {
    return {myid, nproc};
  }

  // get the hostname of the node
  std::string pname;  // node hostname
  pname.resize(MPI_MAX_PROCESSOR_NAME);
  int pname_length;  // length of node hostname

  MPI_Get_processor_name(pname.data(), &pname_length);

  // hash the name of the node. MPI_Comm_split doesn't like negative numbers and accepts ints not unsigned ints so we
  // need to take the absolute value
  int const hash = std::abs(static_cast<int>(std::hash<std::string>{}(pname)));

  // split the communicator
  MPI_Comm node_comm;  // communicator for the procs on each node
  MPI_Comm_split(MPI_COMM_WORLD, hash, myid, &node_comm);

  // get size and rank
  MPI_Comm_rank(node_comm, &myid);
  MPI_Comm_size(node_comm, &nproc);

  return {myid, nproc};
}

#endif /*MPI_CHOLLA*/

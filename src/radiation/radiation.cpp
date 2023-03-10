/*! \file radiation.cpp
 *  \brief Definitions for the radiative transfer wrapper */

#ifdef RT

  #include "radiation.h"

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "RT_functions.h"
  #include "alt/atomic_data.h"
  #include "alt/photo_rates_csi_gpu.h"

Rad3D::Rad3D(const Header& grid_) : grid(grid_)
{
  Physics::AtomicData::Create();
  /// photoRates = new PhotoRatesCSI::TableWrapperGPU(2,6);
  photoRates = new PhotoRatesCSI::TableWrapperGPU(1, 6);
}

Rad3D::~Rad3D()
{
  delete photoRates;
  Physics::AtomicData::Delete();
}

void Rad3D::Initialize_Start(const parameters& params)
{
  num_iterations = params.num_iterations;

  // allocate memory on the host
  rtFields.rf = (Real*)malloc((1 + 2 * n_freq) * grid.n_cells * sizeof(Real));

  // set boundary flags
  flags[0] = params.xl_bcnd;
  flags[1] = params.xu_bcnd;
  flags[2] = params.yl_bcnd;
  flags[3] = params.yu_bcnd;
  flags[4] = params.zl_bcnd;
  flags[5] = params.zu_bcnd;
  printf("%d %d %d %d %d %d\n", flags[0], flags[1], flags[2], flags[3], flags[4], flags[5]);
}

// function to do various initialization tasks, i.e. allocating memory, etc.
void Rad3D::Initialize_Finish()
{
  chprintf("Initializing Radiative Transfer...\n");

  // Allocate memory for abundances (passive scalars added to the hydro grid)
  // This is done in grid3D::Allocate_Memory
  chprintf(" N scalar fields: %d \n", NSCALARS);

  // Allocate memory for radiation fields (non-advecting, 2 per frequency plus 1 optically thin field)
  chprintf("Allocating memory for radiation fields. \n");
  // allocate memory on the device
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_rf, (1 + 2 * n_freq) * grid.n_cells * sizeof(Real)));

  // Allocate memory for Eddington tensor only on device
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_et, 6 * grid.n_cells * sizeof(Real)));

  // Allocate memory for radiation source field only on device
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_rs, grid.n_cells * sizeof(Real)));

  // Allocate temporary fields on device
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_abc, n_freq * grid.n_cells * sizeof(Real)));
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_rfNew, 2 * grid.n_cells * sizeof(Real)));

  // Initialize Field values (for now)
  this->Initialize_GPU();
}

// function to call the radiation solver from main
void Grid3D::Update_RT()
{
  // passes d_scalar as that is the pointer to the first abundance array, rho_HI
  Rad.rtSolve(C.d_scalar);
}

// Set boundary cells for radiation fields
// Current implementation only supports periodic boundaries
void Rad3D::rtBoundaries(void)
{
  int buffer_length;
  int xbsize = x_buffer_length, ybsize = y_buffer_length, zbsize = z_buffer_length;
  int wait_max = 0;
  int index    = 0;
  int ireq     = 0;
  MPI_Status status;

  // Send MPI x-boundaries
  if (flags[0] == 5) {
    buffer_length =
        Load_RT_Fields_To_Buffer(0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_send_buffer_x0);
  #ifndef MPI_GPU
    cudaMemcpy(h_send_buffer_x0, d_send_buffer_x0, xbsize * sizeof(Real), cudaMemcpyDeviceToHost);
  #endif
    wait_max++;

  #if defined(MPI_GPU)
    // post non-blocking receive left x communication buffer
    MPI_Irecv(d_recv_buffer_x0, buffer_length, MPI_CHREAL, source[0], 0, world, &recv_request[ireq]);
    // non-blocking send left x communication buffer
    MPI_Isend(d_send_buffer_x0, buffer_length, MPI_CHREAL, dest[0], 1, world, &send_request[0]);
  #else
    // post non-blocking receive left x communication buffer
    MPI_Irecv(h_recv_buffer_x0, buffer_length, MPI_CHREAL, source[0], 0, world, &recv_request[ireq]);
    // non-blocking send left x communication buffer
    MPI_Isend(h_send_buffer_x0, buffer_length, MPI_CHREAL, dest[0], 1, world, &send_request[0]);
  #endif  // MPI_GPU

    MPI_Request_free(send_request);
    // keep track of how many sends and receives are expected
    ireq++;
  }

  if (flags[1] == 5) {
    buffer_length =
        Load_RT_Fields_To_Buffer(0, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_send_buffer_x1);
  #ifndef MPI_GPU
    cudaMemcpy(h_send_buffer_x1, d_send_buffer_x1, xbsize * sizeof(Real), cudaMemcpyDeviceToHost);
  #endif
    wait_max++;

  #if defined(MPI_GPU)
    // post non-blocking receive right x communication buffer
    MPI_Irecv(d_recv_buffer_x1, buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request[ireq]);
    // non-blocking send right x communication buffer
    MPI_Isend(d_send_buffer_x1, buffer_length, MPI_CHREAL, dest[1], 0, world, &send_request[1]);
  #else
    // post non-blocking receive right x communication buffer
    MPI_Irecv(h_recv_buffer_x1, buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request[ireq]);
    // non-blocking send right x communication buffer
    MPI_Isend(h_send_buffer_x1, buffer_length, MPI_CHREAL, dest[1], 0, world, &send_request[1]);
  #endif  // MPI_GPU

    MPI_Request_free(send_request + 1);
    // keep track of how many sends and receives are expected
    ireq++;
  }
  // printf("ireq: %d\n", ireq);
  // printf("wait_max: %d\n", wait_max);

  /* Set non-MPI x-boundaries */
  if (flags[0] == 1) {
    Set_RT_Boundaries_Periodic(0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  }
  if (flags[1] == 1) {
    Set_RT_Boundaries_Periodic(0, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  }

  /* Receive MPI x-boundaries */
  // wait for any receives to complete
  for (int iwait = 0; iwait < wait_max; iwait++) {
    // printf("ProcID %d waiting for request %d.\n", procID, iwait);
    //  wait for recv completion
    MPI_Waitany(wait_max, recv_request, &index, &status);
    // printf("ProcID %d with index %d, status %d.\n", procID, index, status.MPI_TAG);
  //  depending on which face arrived, unload the buffer into the ghost grid
  #ifndef MPI_GPU
    copyHostToDeviceReceiveBuffer(status.MPI_TAG);
  #endif  // MPI_GPU
    if (status.MPI_TAG == 0)
      Unload_RT_Fields_From_Buffer(0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_recv_buffer_x0);
    if (status.MPI_TAG == 1)
      Unload_RT_Fields_From_Buffer(0, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_recv_buffer_x1);
  }

  // Barrier between directions
  MPI_Barrier(world);
  fflush(stdout);
  ireq     = 0;
  wait_max = 0;

  // Send MPI y-boundaries
  if (flags[2] == 5) {
    buffer_length =
        Load_RT_Fields_To_Buffer(1, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_send_buffer_y0);
  #ifndef MPI_GPU
    cudaMemcpy(h_send_buffer_y0, d_send_buffer_y0, ybsize * sizeof(Real), cudaMemcpyDeviceToHost);
  #endif
    wait_max++;

  #if defined(MPI_GPU)
    // post non-blocking receive left y communication buffer
    MPI_Irecv(d_recv_buffer_y0, buffer_length, MPI_CHREAL, source[2], 2, world, &recv_request[ireq]);
    // non-blocking send left y communication buffer
    MPI_Isend(d_send_buffer_y0, buffer_length, MPI_CHREAL, dest[2], 3, world, &send_request[0]);
  #else
    // post non-blocking receive left y communication buffer
    MPI_Irecv(h_recv_buffer_y0, buffer_length, MPI_CHREAL, source[2], 2, world, &recv_request[ireq]);
    // non-blocking send left y communication buffer
    MPI_Isend(h_send_buffer_y0, buffer_length, MPI_CHREAL, dest[2], 3, world, &send_request[0]);
  #endif  // MPI_GPU

    MPI_Request_free(send_request);
    // keep track of how many sends and receives are expected
    ireq++;
  }

  if (flags[3] == 5) {
    buffer_length =
        Load_RT_Fields_To_Buffer(1, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_send_buffer_y1);
  #ifndef MPI_GPU
    cudaMemcpy(h_send_buffer_y1, d_send_buffer_y1, ybsize * sizeof(Real), cudaMemcpyDeviceToHost);
  #endif
    wait_max++;

  #if defined(MPI_GPU)
    // post non-blocking receive right y communication buffer
    MPI_Irecv(d_recv_buffer_y1, buffer_length, MPI_CHREAL, source[3], 3, world, &recv_request[ireq]);
    // non-blocking send right y communication buffer
    MPI_Isend(d_send_buffer_y1, buffer_length, MPI_CHREAL, dest[3], 2, world, &send_request[1]);
  #else
    // post non-blocking receive right y communication buffer
    MPI_Irecv(h_recv_buffer_y1, buffer_length, MPI_CHREAL, source[3], 3, world, &recv_request[ireq]);
    // non-blocking send right y communication buffer
    MPI_Isend(h_send_buffer_y1, buffer_length, MPI_CHREAL, dest[3], 2, world, &send_request[1]);
  #endif  // MPI_GPU

    MPI_Request_free(send_request + 1);
    // keep track of how many sends and receives are expected
    ireq++;
  }
  // printf("ireq: %d\n", ireq);
  // printf("wait_max: %d\n", wait_max);

  /* Set non-MPI y-boundaries */
  if (flags[2] == 1) {
    Set_RT_Boundaries_Periodic(1, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  }
  if (flags[3] == 1) {
    Set_RT_Boundaries_Periodic(1, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  }

  /* Receive MPI y-boundaries */
  // wait for any receives to complete
  for (int iwait = 0; iwait < wait_max; iwait++) {
    // printf("ProcID %d waiting for request %d.\n", procID, iwait);
    //  wait for recv completion
    MPI_Waitany(wait_max, recv_request, &index, &status);
    // printf("ProcID %d with index %d, status %d.\n", procID, index, status.MPI_TAG);
  //  depending on which face arrived, unload the buffer into the ghost grid
  #ifndef MPI_GPU
    copyHostToDeviceReceiveBuffer(status.MPI_TAG);
  #endif  // MPI_GPU
    if (status.MPI_TAG == 2)
      Unload_RT_Fields_From_Buffer(1, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_recv_buffer_y0);
    if (status.MPI_TAG == 3)
      Unload_RT_Fields_From_Buffer(1, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_recv_buffer_y1);
  }

  // Barrier between directions
  MPI_Barrier(world);
  fflush(stdout);
  ireq     = 0;
  wait_max = 0;

  // Send MPI z-boundaries
  if (flags[4] == 5) {
    buffer_length =
        Load_RT_Fields_To_Buffer(2, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_send_buffer_z0);
    // printf("%d %d\n", buffer_length, zbsize);
  #ifndef MPI_GPU
    cudaMemcpy(h_send_buffer_z0, d_send_buffer_z0, zbsize * sizeof(Real), cudaMemcpyDeviceToHost);
  #endif
    wait_max++;

  #if defined(MPI_GPU)
    // post non-blocking receive left z communication buffer
    MPI_Irecv(d_recv_buffer_z0, buffer_length, MPI_CHREAL, source[4], 4, world, &recv_request[ireq]);
    // non-blocking send left z communication buffer
    MPI_Isend(d_send_buffer_z0, buffer_length, MPI_CHREAL, dest[4], 5, world, &send_request[0]);
  #else
    // post non-blocking receive left z communication buffer
    MPI_Irecv(h_recv_buffer_z0, buffer_length, MPI_CHREAL, source[4], 5, world, &recv_request[ireq]);
    // non-blocking send left z communication buffer
    MPI_Isend(h_send_buffer_z0, buffer_length, MPI_CHREAL, dest[4], 5, world, &send_request[0]);
  #endif  // MPI_GPU

    MPI_Request_free(send_request);
    // keep track of how many sends and receives are expected
    ireq++;
  }

  if (flags[5] == 5) {
    buffer_length =
        Load_RT_Fields_To_Buffer(2, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_send_buffer_z1);
    // printf("%d %d\n", buffer_length, zbsize);
  #ifndef MPI_GPU
    cudaMemcpy(h_send_buffer_z1, d_send_buffer_z1, zbsize * sizeof(Real), cudaMemcpyDeviceToHost);
  #endif
    wait_max++;

  #if defined(MPI_GPU)
    // post non-blocking receive right z communication buffer
    MPI_Irecv(d_recv_buffer_z1, buffer_length, MPI_CHREAL, source[5], 5, world, &recv_request[ireq]);
    // non-blocking send right z communication buffer
    MPI_Isend(d_send_buffer_z1, buffer_length, MPI_CHREAL, dest[5], 4, world, &send_request[1]);
  #else
    // post non-blocking receive right z communication buffer
    MPI_Irecv(h_recv_buffer_z1, buffer_length, MPI_CHREAL, source[5], 5, world, &recv_request[ireq]);
    // non-blocking send right z communication buffer
    MPI_Isend(h_send_buffer_z1, buffer_length, MPI_CHREAL, dest[5], 4, world, &send_request[1]);
  #endif  // MPI_GPU

    MPI_Request_free(send_request + 1);
    // keep track of how many sends and receives are expected
    ireq++;
  }
  // printf("ireq: %d\n", ireq);
  // printf("wait_max: %d\n", wait_max);

  /* Set non-MPI z-boundaries */
  if (flags[4] == 1) {
    Set_RT_Boundaries_Periodic(2, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  }
  if (flags[5] == 1) {
    Set_RT_Boundaries_Periodic(2, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  }

  /* Receive MPI z-boundaries */
  // wait for any receives to complete
  for (int iwait = 0; iwait < wait_max; iwait++) {
    // printf("ProcID %d waiting for request %d.\n", procID, iwait);
    //  wait for recv completion
    MPI_Waitany(wait_max, recv_request, &index, &status);
    // printf("ProcID %d with index %d, status %d.\n", procID, index, status.MPI_TAG);
  //  depending on which face arrived, unload the buffer into the ghost grid
  #ifndef MPI_GPU
    copyHostToDeviceReceiveBuffer(status.MPI_TAG);
  #endif  // MPI_GPU
    if (status.MPI_TAG == 4)
      Unload_RT_Fields_From_Buffer(2, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_recv_buffer_z0);
    if (status.MPI_TAG == 5)
      Unload_RT_Fields_From_Buffer(2, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields, d_recv_buffer_z1);
  }
}

void Rad3D::Free_Memory(void)
{
  cudaFree(rtFields.dev_rf);
  cudaFree(rtFields.dev_et);
  cudaFree(rtFields.dev_rs);
  cudaFree(rtFields.dev_abc);
  cudaFree(rtFields.dev_rfNew);

  if (rtFields.et != nullptr) free(rtFields.et);
  if (rtFields.rs != nullptr) free(rtFields.rs);
  free(rtFields.rf);
}

#endif  // RT

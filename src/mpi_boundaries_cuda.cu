#ifdef MPI_CHOLLA
#include"grid3D.h"
#include"mpi_routines.h"
#include"io.h"
#include"error_handling.h"
#include <iostream>

#include "nvtx.h"
#include "global_cuda.h"

#include <cassert>

#ifdef DEVICE_COMM

template <bool load, bool forward>
__global__ void Load_Hydro_Buffer_X_Cuda_kernel(Header H, Real *dev_send_recv_buffer, Real *dev_conserved) {

  int offset = H.n_ghost*(H.ny-2*H.n_ghost)*(H.nz-2*H.n_ghost);

  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_idx >= (H.ny-2*H.n_ghost) * (H.nz-2*H.n_ghost)) { return; }
  int j = thread_idx % (H.ny-2*H.n_ghost);
  int k = thread_idx / (H.ny-2*H.n_ghost);

  int i_offset = load ? (forward ? H.nx-2*H.n_ghost : H.n_ghost) : (forward ? H.nx-H.n_ghost : 0);

  for(int i=0;i<H.n_ghost;i++)
  {
    int idx  = (i+i_offset) + (j+H.n_ghost)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
    int gidx = i + j*H.n_ghost + k*H.n_ghost*(H.ny-2*H.n_ghost);
    for (int ii=0; ii<H.n_fields; ii++) {
      if (load) {
        *(dev_send_recv_buffer + gidx + ii*offset) = dev_conserved[idx + ii*H.n_cells];
      } else { // unload
        dev_conserved[idx + ii*H.n_cells] = *(dev_send_recv_buffer + gidx + ii*offset);
      }
    }
  }
}

template <bool load, bool forward>
__global__ void Load_Hydro_Buffer_Y_Cuda_kernel(Header H, Real *dev_send_recv_buffer, Real *dev_conserved) {

  int offset = H.n_ghost*H.nx*(H.nz-2*H.n_ghost);

  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_idx >= H.nx*(H.nz-2*H.n_ghost)) { return; }
  int i = thread_idx % (H.nx);
  int k = thread_idx / (H.nx);

  int j_offset = load ? (forward ? H.ny-2*H.n_ghost : H.n_ghost) : (forward ? H.ny-H.n_ghost : 0);

  for(int j=0;j<H.n_ghost;j++)
  {
    int idx  = i + (j+j_offset)*H.nx + (k+H.n_ghost)*H.nx*H.ny;
    int gidx = i + j*H.nx + k*H.nx*H.n_ghost;
    for (int ii=0; ii<H.n_fields; ii++) {
      if (load) {
        *(dev_send_recv_buffer + gidx + ii*offset) = dev_conserved[idx + ii*H.n_cells];
      } else { // unload
        dev_conserved[idx + ii*H.n_cells] = *(dev_send_recv_buffer + gidx + ii*offset);
      }
    }
  }
}

template <bool load, bool forward>
__global__ void Load_Hydro_Buffer_Z_Cuda_kernel(Header H, Real *dev_send_recv_buffer, Real *dev_conserved) {

  int offset = H.n_ghost*H.nx*H.ny;

  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_idx >= H.nx*H.ny) { return; }
  int i = thread_idx % (H.nx);
  int j = thread_idx / (H.nx);

  int k_offset = load ? (forward ? H.nz-2*H.n_ghost : H.n_ghost) : (forward ? H.nz-H.n_ghost : 0);

  for(int k=0;k<H.n_ghost;k++)
  {
    int idx  = i + j*H.nx + (k+k_offset)*H.nx*H.ny;
    int gidx = i + j*H.nx + k*H.nx*H.ny;
    for (int ii=0; ii<H.n_fields; ii++) {
      if (load) {
        *(dev_send_recv_buffer + gidx + ii*offset) = dev_conserved[idx + ii*H.n_cells];
      } else { // unload
        dev_conserved[idx + ii*H.n_cells] = *(dev_send_recv_buffer + gidx + ii*offset);
      }
    }
  }
}

// load left x communication buffer
int Grid3D::Load_Hydro_Buffer_X0_Cuda(){
  nvtx_raii _nvtx(__FUNCTION__, __LINE__);

  // 1D
  if (H.ny == 1 && H.nz == 1) {
    assert(false);
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    assert(false);
  }
  // 3D
  if (H.ny > 1 && H.nz > 1) {
    int block = 256;
    int grid = ((H.ny-2*H.n_ghost) * (H.nz-2*H.n_ghost) + block - 1) / block;
    // forward = false
    Load_Hydro_Buffer_X_Cuda_kernel<true, false><<<grid, block>>>(H, dev_send_buffer_x0, dev_conserved);
    CudaSafeCall(cudaStreamSynchronize(0));
  }
  return x_buffer_length;
}

// load right x communication buffer
int Grid3D::Load_Hydro_Buffer_X1_Cuda(){
  nvtx_raii _nvtx(__FUNCTION__, __LINE__);

  // 1D
  if (H.ny == 1 && H.nz == 1) {
    assert(false);
  }
  // 2D
  if (H.ny > 1 && H.nz == 1) {
    assert(false);
  }
  // 3D
  if (H.ny > 1 && H.nz > 1) {
    int block = 256;
    int grid = ((H.ny-2*H.n_ghost) * (H.nz-2*H.n_ghost) + block - 1) / block;
    // forward = true
    Load_Hydro_Buffer_X_Cuda_kernel<true, true><<<grid, block>>>(H, dev_send_buffer_x1, dev_conserved);
    CudaSafeCall(cudaStreamSynchronize(0));
  }
  return x_buffer_length;
}

// load left y communication buffer
int Grid3D::Load_Hydro_Buffer_Y0_Cuda(){
  nvtx_raii _nvtx(__FUNCTION__, __LINE__);
  // 2D
  if (H.nz == 1) {
    assert(false);
  }
  // 3D
  if (H.nz > 1) {
    int block = 256;
    int grid = (H.nx*(H.nz-2*H.n_ghost) + block - 1) / block;
    // forward = false
    Load_Hydro_Buffer_Y_Cuda_kernel<true, false><<<grid, block>>>(H, dev_send_buffer_y0, dev_conserved);
    CudaSafeCall(cudaStreamSynchronize(0));
  }
  return y_buffer_length;
}

// load right y communication buffer
int Grid3D::Load_Hydro_Buffer_Y1_Cuda(){
  nvtx_raii _nvtx(__FUNCTION__, __LINE__);

  // 2D
  if (H.nz == 1) {
    assert(false);
  }
  // 3D
  if (H.nz > 1) {
    int block = 256;
    int grid = (H.nx*(H.nz-2*H.n_ghost) + block - 1) / block;
    // forward = true
    Load_Hydro_Buffer_Y_Cuda_kernel<true, true><<<grid, block>>>(H, dev_send_buffer_y1, dev_conserved);
    CudaSafeCall(cudaStreamSynchronize(0));
  }
  return y_buffer_length;
}

// load left z communication buffer
int Grid3D::Load_Hydro_Buffer_Z0_Cuda(){
  nvtx_raii _nvtx(__FUNCTION__, __LINE__);

  int block = 256;
  int grid = (H.nx*H.ny + block - 1) / block;
  // forward = false
  Load_Hydro_Buffer_Z_Cuda_kernel<true, false><<<grid, block>>>(H, dev_send_buffer_z0, dev_conserved);
  CudaSafeCall(cudaStreamSynchronize(0));

  return z_buffer_length;
}

// load right z communication buffer
int Grid3D::Load_Hydro_Buffer_Z1_Cuda(){
  nvtx_raii _nvtx(__FUNCTION__, __LINE__);

  int block = 256;
  int grid = (H.nx*H.ny + block - 1) / block;
  // forward = true
  Load_Hydro_Buffer_Z_Cuda_kernel<true, true><<<grid, block>>>(H, dev_send_buffer_z1, dev_conserved);
  CudaSafeCall(cudaStreamSynchronize(0));

  return z_buffer_length;
}

void Grid3D::Unload_MPI_Comm_Buffers_BLOCK_Cuda(int index)
{
  nvtx_raii _nvtx(__FUNCTION__, __LINE__);

  if ( H.TRANSFER_HYDRO_BOUNDARIES ){
    //unload left x communication buffer
    if(index==0)
    {
      // 1D
      if (H.ny == 1 && H.nz == 1) {
        assert(false);
      }
      // 2D
      if (H.ny > 1 && H.nz == 1) {
        assert(false);
      }
      // 3D
      if (H.nz > 1) {
        int block = 256;
        int grid = ((H.ny-2*H.n_ghost)*(H.nz-2*H.n_ghost) + block - 1) / block;
        // forward = false
        Load_Hydro_Buffer_X_Cuda_kernel<false, false><<<grid, block>>>(H, dev_recv_buffer_x0, dev_conserved);
        CudaSafeCall(cudaStreamSynchronize(0));
      }
    }

    //unload right x communication buffer
    if(index==1)
    {
      // 1D
      if (H.ny == 1 && H.nz == 1) {
        assert(false);
      }
      // 2D
      if (H.ny > 1 && H.nz == 1) {
        assert(false);
      }
      // 3D
      if (H.nz > 1) {
        int block = 256;
        int grid = ((H.ny-2*H.n_ghost)*(H.nz-2*H.n_ghost) + block - 1) / block;
        // forward = true
        Load_Hydro_Buffer_X_Cuda_kernel<false, true><<<grid, block>>>(H, dev_recv_buffer_x1, dev_conserved);
        CudaSafeCall(cudaStreamSynchronize(0));
      }
    }


    //unload left y communication buffer
    if(index==2)
    {
      // 2D
      if (H.nz == 1) {
        assert(false);
      }
      // 3D
      if (H.nz > 1) {
        int block = 256;
        int grid = (H.nx*(H.nz-2*H.n_ghost) + block - 1) / block;
        // forward = false
        Load_Hydro_Buffer_Y_Cuda_kernel<false, false><<<grid, block>>>(H, dev_recv_buffer_y0, dev_conserved);
        CudaSafeCall(cudaStreamSynchronize(0));
      }
    }

    //unload right y communication buffer
    if(index==3)
    {
      // 2D
      if (H.nz == 1) {
        assert(false);
      }
      // 3D
      if (H.nz > 1) {
        int block = 256;
        int grid = (H.nx*(H.nz-2*H.n_ghost) + block - 1) / block;
        // forward = true
        Load_Hydro_Buffer_Y_Cuda_kernel<false, true><<<grid, block>>>(H, dev_recv_buffer_y1, dev_conserved);
        CudaSafeCall(cudaStreamSynchronize(0));
      }
    }

    //unload left z communication buffer
    if(index==4)
    {
      int block = 256;
      int grid = (H.nx*H.ny + block - 1) / block;
      // forward = false
      Load_Hydro_Buffer_Z_Cuda_kernel<false, false><<<grid, block>>>(H, dev_recv_buffer_z0, dev_conserved);
      CudaSafeCall(cudaStreamSynchronize(0));
    }

    //unload right z communication buffer
    if(index==5)
    {
      int block = 256;
      int grid = (H.nx*H.ny + block - 1) / block;
      // forward = false
      Load_Hydro_Buffer_Z_Cuda_kernel<false, true><<<grid, block>>>(H, dev_recv_buffer_z1, dev_conserved);
      CudaSafeCall(cudaStreamSynchronize(0));
    }
  }

  #if( defined(GRAVITY)  )
  if ( Grav.TRANSFER_POTENTIAL_BOUNDARIES ){
    if ( index == 0 ) Unload_Gravity_Potential_from_Buffer( 0, 0, recv_buffer_x0, 0  );
    if ( index == 1 ) Unload_Gravity_Potential_from_Buffer( 0, 1, recv_buffer_x1, 0  );
    if ( index == 2 ) Unload_Gravity_Potential_from_Buffer( 1, 0, recv_buffer_y0, 0  );
    if ( index == 3 ) Unload_Gravity_Potential_from_Buffer( 1, 1, recv_buffer_y1, 0  );
    if ( index == 4 ) Unload_Gravity_Potential_from_Buffer( 2, 0, recv_buffer_z0, 0  );
    if ( index == 5 ) Unload_Gravity_Potential_from_Buffer( 2, 1, recv_buffer_z1, 0  );
  }
  #endif

  #ifdef PARTICLES
  if (  Particles.TRANSFER_DENSITY_BOUNDARIES ){
    if ( index == 0 ) Unload_Particles_Density_Boundary_From_Buffer( 0, 0, recv_buffer_x0 );
    if ( index == 1 ) Unload_Particles_Density_Boundary_From_Buffer( 0, 1, recv_buffer_x1 );
    if ( index == 2 ) Unload_Particles_Density_Boundary_From_Buffer( 1, 0, recv_buffer_y0 );
    if ( index == 3 ) Unload_Particles_Density_Boundary_From_Buffer( 1, 1, recv_buffer_y1 );
    if ( index == 4 ) Unload_Particles_Density_Boundary_From_Buffer( 2, 0, recv_buffer_z0 );
    if ( index == 5 ) Unload_Particles_Density_Boundary_From_Buffer( 2, 1, recv_buffer_z1 );
  }
  #endif

}

#endif

#endif /*MPI_CHOLLA*/

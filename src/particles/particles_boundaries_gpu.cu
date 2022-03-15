#if defined(PARTICLES) && defined(PARTICLES_GPU)

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../utils/gpu.hpp"
#include <iostream>
#include "../io/io.h"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../grid/grid3D.h"
#include "particles_boundaries_gpu.h"
#include "particles_3D.h"

#define SCAN_SHARED_SIZE 2*TPB_PARTICLES


__global__ void Set_Particles_Boundary_Kernel( int side, part_int_t n_local,  Real *pos_dev, Real d_min, Real d_max, Real d_length ){

  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_local) return;

  Real pos;
  pos = pos_dev[tid];

  if ( side == 0 ){
    if ( pos < d_min ) pos += d_length;
  }

  if ( side == 1 ){
    if ( pos >= d_max ) pos -= d_length;
  }

  pos_dev[tid] = pos;

}


void Grid3D::Set_Particles_Boundary_GPU( int dir, int side ){

  Real d_min, d_max, L;
  Real *pos_dev;
  if ( dir == 0 ){
    d_min = Particles.G.zMin;
    d_max = Particles.G.zMax;
    pos_dev = Particles.pos_x_dev;
  }
  if ( dir == 1 ){
    d_min = Particles.G.yMin;
    d_max = Particles.G.yMax;
    pos_dev = Particles.pos_y_dev;
  }
  if ( dir == 2 ){
    d_min = Particles.G.zMin;
    d_max = Particles.G.zMax;
    pos_dev = Particles.pos_z_dev;
  }

  L = d_max - d_min;

  // set values for GPU kernels
  int grid_size =  (Particles.n_local - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(grid_size, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  hipLaunchKernelGGL(Set_Particles_Boundary_Kernel, dim1dGrid, dim1dBlock, 0, 0,  side, Particles.n_local, pos_dev, d_min, d_max, L  );
  CudaCheckError();

}


// #ifdef MPI_CHOLLA

__global__ void Get_Transfer_Flags_Kernel( part_int_t n_total, int side,  Real d_min, Real d_max, Real *pos_d, bool *transfer_flags_d ){

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;

  bool transfer = 0;

  Real pos = pos_d[tid];

  if ( side == 0 && pos < d_min) transfer = 1;
  if ( side == 1 && pos >= d_max) transfer = 1;

  transfer_flags_d[tid] = transfer;
}


__global__ void Scan_Kernel( part_int_t n_total, bool *transfer_flags_d, int *prefix_sum_d, int *prefix_sum_block_d ){

  __shared__ int data_sh[SCAN_SHARED_SIZE];

  int tid_block, block_start;
  // tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_block = threadIdx.x;

  block_start = 2*blockIdx.x*blockDim.x;

  data_sh[2*tid_block] = block_start + 2*tid_block < n_total ? (int) transfer_flags_d[block_start + 2*tid_block]  :  0;
  data_sh[2*tid_block+1] = block_start + 2*tid_block+1 < n_total ?  (int) transfer_flags_d[block_start + 2*tid_block+1]  :  0;
  __syncthreads();

  int offset = 1;
  int n = blockDim.x*2;

  int ai, bi;
  int t;

  for (int d = n/2; d>0; d/=2){

    __syncthreads();
    if ( tid_block < d ){
      ai = offset*(2*tid_block+1)-1;
      bi = offset*(2*tid_block+2)-1;
      data_sh[bi] += data_sh[ai];
    }
    offset *= 2;
  }

  // Clear the last element
  if (tid_block == 0) data_sh[n - 1] = 0;

  // Traverse down tree & build scan
  for (int d = 1; d < n; d *= 2){

    __syncthreads();
    offset /=2;
    if (tid_block < d){

      ai = offset*(2*tid_block+1)-1;
      bi = offset*(2*tid_block+2)-1;

      t = data_sh[ai];
      data_sh[ai] = data_sh[bi];
      data_sh[bi] += t;
    }
  }
  __syncthreads();

  // Write results to device memory
  if ( block_start + 2*tid_block < n_total )  prefix_sum_d[block_start + 2*tid_block] = data_sh[2*tid_block];
  if ( block_start + 2*tid_block+1 < n_total) prefix_sum_d[block_start + 2*tid_block+1] = data_sh[2*tid_block+1];

  // Write the block sum
  int last_flag_block = (int) transfer_flags_d[block_start + 2*(blockDim.x-1)+1];
  if (tid_block == 0) prefix_sum_block_d[blockIdx.x] = data_sh[2*(blockDim.x-1)+1] + last_flag_block;
}


__global__ void Prefix_Sum_Blocks_Kernel( int n_partial, int *prefix_sum_block_d ){

  int tid_block, val,  start_index, n_threads;
  tid_block = threadIdx.x;
  n_threads = blockDim.x;

  __shared__ int data_sh[TPB_PARTICLES];


  int sum = 0;
  int n = 0;
  start_index = n * n_threads;
  while( start_index < n_partial ){
    data_sh[tid_block] = start_index+tid_block < n_partial  ?  prefix_sum_block_d[start_index+tid_block] :  0;
    __syncthreads();


    if (tid_block == 0){
      for ( int i=0; i<n_threads; i++ ){
        val = data_sh[i];
        data_sh[i] = sum;
        sum += val;
      }
    }
    __syncthreads();

    if (start_index + tid_block < n_partial) prefix_sum_block_d[start_index+tid_block] = data_sh[tid_block];
    n += 1;
    start_index = n * n_threads;

  }
}


__global__ void Sum_Blocks_Kernel( part_int_t n_total,  int *prefix_sum_d, int *prefix_sum_block_d ){

  int tid, tid_block, block_id, data_id;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_block = threadIdx.x;
  block_id = blockIdx.x;
  data_id = block_id/2;

  __shared__ int block_sum_sh[1];

  if ( tid_block == 0 ){
    block_sum_sh[0] = prefix_sum_block_d[data_id];
    // printf( "%d   %d\n",  block_id/2, prefix_sum_block[data_id] );
  }
  __syncthreads();

  if (tid < n_total) prefix_sum_d[tid] += block_sum_sh[0];
}


__global__ void Get_N_Transfer_Particles_Kernel( part_int_t n_total, int *n_transfer_d, bool *transfer_flags_d, int *prefix_sum_d  ){
  n_transfer_d[0] = prefix_sum_d[n_total-1] + (int)transfer_flags_d[n_total-1];
  // if ( n_transfer_d[0] > 0 ) printf( "##Thread transfer: %d\n", n_transfer_d[0]);
}

__global__ void Get_Transfer_Indices_Kernel( part_int_t n_total, bool *transfer_flags_d, int *prefix_sum_d, int *transfer_indices_d ){

  int tid, transfer_index;
  tid =  threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  transfer_index = prefix_sum_d[tid];
  
  if ( transfer_index < 0 || transfer_index >= n_total ){
    printf( "#### PARTICLE TRANSFER ERROR:  transfer index outside domain: %d \n", transfer_index  ); 
    return;
  }
  
  if ( transfer_flags_d[tid] ) transfer_indices_d[transfer_index] = tid;

}


__global__ void Select_Indices_to_Replace_Transfered_Kernel( part_int_t n_total, int n_transfer, bool *transfer_flags_d, int *prefix_sum_d, int *replace_indices_d ){

  int tid, tid_inv;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  tid_inv = n_total - tid - 1;

  bool transfer_flag = transfer_flags_d[tid];
  if ( transfer_flag ) return;

  int prefix_sum_inv, replace_id;

  prefix_sum_inv = n_transfer - prefix_sum_d[tid];
  replace_id = tid_inv - prefix_sum_inv;
  
  
  if ( replace_id < 0 || replace_id >= n_total ){
    printf( "#### PARTICLE TRANSFER ERROR:  replace index outside domain: %d \n", replace_id  );
    return;
  } 
  replace_indices_d[replace_id] = tid;

}


template< typename T>
__global__ void Replace_Transfered_Particles_Kernel( int n_transfer, T *field_d, int *transfer_indices_d, int *replace_indices_d, bool print_replace ){
  int tid;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_transfer ) return;

  int dst_id, src_id;
  dst_id = transfer_indices_d[tid];
  src_id = replace_indices_d[tid];

  if ( dst_id < src_id ){
    if (print_replace) printf("Replacing: %f \n", field_d[dst_id]*1.0 );
    field_d[dst_id] = field_d[src_id];
  }

}


void Replace_Transfered_Particles_GPU_function(  int n_transfer, Real *field_d, int *transfer_indices_d, int *replace_indices_d, bool print_replace ){
  int grid_size;
  grid_size =  (n_transfer - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(grid_size, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  hipLaunchKernelGGL( Replace_Transfered_Particles_Kernel, dim1dGrid, dim1dBlock, 0, 0,  n_transfer,  field_d, transfer_indices_d, replace_indices_d, print_replace );
  CudaCheckError();

}


void Replace_Transfered_Particles_Int_GPU_function(  int n_transfer, part_int_t *field_d, int *transfer_indices_d, int *replace_indices_d, bool print_replace ){
  int grid_size;
  grid_size =  (n_transfer - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(grid_size, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  hipLaunchKernelGGL( Replace_Transfered_Particles_Kernel, dim1dGrid, dim1dBlock, 0, 0,  n_transfer,  field_d, transfer_indices_d, replace_indices_d, print_replace );
  CudaCheckError();
}


part_int_t Select_Particles_to_Transfer_GPU_function( part_int_t n_local, int side, Real domainMin, Real domainMax, Real *pos_d, int *n_transfer_d, int *n_transfer_h, bool *transfer_flags_d, int *transfer_indices_d, int *replace_indices_d, int *transfer_prefix_sum_d, int *transfer_prefix_sum_blocks_d  ){
  // set values for GPU kernels
  int grid_size, grid_size_half;
  grid_size =  (n_local - 1) / TPB_PARTICLES + 1;
  grid_size_half = ( (n_local-1)/2 ) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(grid_size, 1, 1);
  dim3 dim1dGrid_half(grid_size_half, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  // Initialize the number of tranfer particles
  n_transfer_h[0] = 0;

  hipLaunchKernelGGL( Get_Transfer_Flags_Kernel, dim1dGrid, dim1dBlock, 0, 0,  n_local, side, domainMin, domainMax, pos_d, transfer_flags_d);
  CudaCheckError();

  hipLaunchKernelGGL( Scan_Kernel, dim1dGrid_half, dim1dBlock, 0, 0,  n_local, transfer_flags_d, transfer_prefix_sum_d, transfer_prefix_sum_blocks_d );
  CudaCheckError();

  hipLaunchKernelGGL( Prefix_Sum_Blocks_Kernel, 1, dim1dBlock , 0, 0,  grid_size_half, transfer_prefix_sum_blocks_d );
  CudaCheckError();
  
  hipLaunchKernelGGL( Sum_Blocks_Kernel, dim1dGrid,   dim1dBlock, 0, 0,  n_local, transfer_prefix_sum_d, transfer_prefix_sum_blocks_d );
  CudaCheckError();
  
  hipLaunchKernelGGL( Get_N_Transfer_Particles_Kernel, 1, 1, 0, 0,  n_local,  n_transfer_d, transfer_flags_d, transfer_prefix_sum_d );
  CudaCheckError();
  
  CudaSafeCall( cudaMemcpy( n_transfer_h, n_transfer_d, sizeof(int), cudaMemcpyDeviceToHost) );
  CudaCheckError();
  
  hipLaunchKernelGGL( Get_Transfer_Indices_Kernel, dim1dGrid, dim1dBlock, 0, 0,  n_local , transfer_flags_d, transfer_prefix_sum_d, transfer_indices_d );
  CudaCheckError();

  hipLaunchKernelGGL( Select_Indices_to_Replace_Transfered_Kernel, dim1dGrid, dim1dBlock , 0, 0,  n_local, n_transfer_h[0], transfer_flags_d, transfer_prefix_sum_d, replace_indices_d );
  CudaCheckError();

  // if ( n_transfer_h[0] > 0 )printf( "N transfer: %d\n", n_transfer_h[0]);
  return n_transfer_h[0];

}



__global__ void Load_Transfered_Particles_to_Buffer_Kernel( int n_transfer, int field_id, int n_fields_to_transfer, Real *field_d, int *transfer_indices_d, Real *send_buffer_d, Real domainMin, Real domainMax, int boundary_type  ){

  int tid;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_transfer ) return;

  int src_id, dst_id;
  Real field_val;
  src_id = transfer_indices_d[tid];
  dst_id = tid * n_fields_to_transfer + field_id;
  field_val = field_d[src_id];

  // Set global periodic boundary conditions
  if ( boundary_type == 1 && field_val < domainMin )  field_val += ( domainMax - domainMin );
  if ( boundary_type == 1 && field_val >= domainMax ) field_val -= ( domainMax - domainMin );
  send_buffer_d[dst_id] = field_val;

}

void Load_Particles_to_Transfer_GPU_function(  int n_transfer, int field_id, int n_fields_to_transfer,  Real *field_d, int *transfer_indices_d, Real *send_buffer_d, Real domainMin, Real domainMax, int boundary_type ){

  // set values for GPU kernels
  int grid_size;
  grid_size =  (n_transfer - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(grid_size, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  hipLaunchKernelGGL( Load_Transfered_Particles_to_Buffer_Kernel, dim1dGrid, dim1dBlock , 0, 0,  n_transfer, field_id, n_fields_to_transfer, field_d, transfer_indices_d, send_buffer_d, domainMin, domainMax, boundary_type );
  CudaCheckError();

}

__global__ void Load_Transfered_Particles_Ints_to_Buffer_Kernel( int n_transfer, int field_id, int n_fields_to_transfer, part_int_t *field_d, int *transfer_indices_d, Real *send_buffer_d, Real domainMin, Real domainMax, int boundary_type  ){

  int tid;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_transfer ) return;

  int src_id, dst_id;
  part_int_t field_val;
  src_id = transfer_indices_d[tid];
  dst_id = tid * n_fields_to_transfer + field_id;
  field_val = field_d[src_id];

  // Set global periodic boundary conditions
  if ( boundary_type == 1 && field_val < domainMin )  field_val += ( domainMax - domainMin );
  if ( boundary_type == 1 && field_val >= domainMax ) field_val -= ( domainMax - domainMin );
  send_buffer_d[dst_id] = __longlong_as_double(field_val);

}


void Load_Particles_to_Transfer_Int_GPU_function(  int n_transfer, int field_id, int n_fields_to_transfer,  part_int_t *field_d, int *transfer_indices_d, Real *send_buffer_d, Real domainMin, Real domainMax, int boundary_type ){
  // set values for GPU kernels
  int grid_size;
  grid_size =  (n_transfer - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(grid_size, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  hipLaunchKernelGGL( Load_Transfered_Particles_Ints_to_Buffer_Kernel, dim1dGrid, dim1dBlock , 0, 0,  n_transfer, field_id, n_fields_to_transfer, field_d, transfer_indices_d, send_buffer_d, domainMin, domainMax, boundary_type );
  CudaCheckError();

}

#ifdef MPI_CHOLLA
void Copy_Particles_GPU_Buffer_to_Host_Buffer( int n_transfer, Real *buffer_h, Real *buffer_d ){

  int transfer_size;
  transfer_size = n_transfer * N_DATA_PER_PARTICLE_TRANSFER;
  CudaSafeCall( cudaMemcpy( buffer_h, buffer_d, transfer_size*sizeof(Real), cudaMemcpyDeviceToHost) );
  CudaCheckError();

}



void Copy_Particles_Host_Buffer_to_GPU_Buffer( int n_transfer, Real *buffer_h, Real *buffer_d ){

  int transfer_size;
  transfer_size = n_transfer * N_DATA_PER_PARTICLE_TRANSFER;
  CudaSafeCall( cudaMemcpy( buffer_d, buffer_h, transfer_size*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaCheckError();

}
#endif //MPI_CHOLLA

__global__ void Unload_Transfered_Particles_from_Buffer_Kernel( int n_local, int n_transfer, int field_id, int n_fields_to_transfer, Real *field_d,  Real *recv_buffer_d  ){

  int tid;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_transfer ) return;

  int src_id, dst_id;
  src_id = tid * n_fields_to_transfer + field_id;
  dst_id = n_local + tid;
  field_d[dst_id] = recv_buffer_d[src_id];

}

void Unload_Particles_to_Transfer_GPU_function( int n_local, int n_transfer, int field_id, int n_fields_to_transfer,  Real *field_d,  Real *recv_buffer_d  ){

  // set values for GPU kernels
  int grid_size;
  grid_size =  (n_transfer - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(grid_size, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  hipLaunchKernelGGL( Unload_Transfered_Particles_from_Buffer_Kernel, dim1dGrid, dim1dBlock , 0, 0, n_local, n_transfer, field_id, n_fields_to_transfer, field_d, recv_buffer_d );
  CudaCheckError();

}

__global__ void Unload_Transfered_Particles_Int_from_Buffer_Kernel( int n_local, int n_transfer, int field_id, int n_fields_to_transfer, part_int_t *field_d,  Real *recv_buffer_d  ){

  int tid;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_transfer ) return;

  int src_id, dst_id;
  src_id = tid * n_fields_to_transfer + field_id;
  dst_id = n_local + tid;
  field_d[dst_id] = __double_as_longlong(recv_buffer_d[src_id]);

}

void Unload_Particles_Int_to_Transfer_GPU_function( int n_local, int n_transfer, int field_id, int n_fields_to_transfer,  part_int_t *field_d,  Real *recv_buffer_d  ){

  // set values for GPU kernels
  int grid_size;
  grid_size =  (n_transfer - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(grid_size, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  hipLaunchKernelGGL( Unload_Transfered_Particles_Int_from_Buffer_Kernel, dim1dGrid, dim1dBlock , 0, 0, n_local, n_transfer, field_id, n_fields_to_transfer, field_d, recv_buffer_d );
  CudaCheckError();

}

// #endif//MPI_CHOLLA


#endif //PARTICLES

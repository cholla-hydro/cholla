#if defined(PARTICLES) && defined(PARTICLES_GPU)

#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include <iostream>
#include"../io.h"
#include"../global.h"
#include"../global_cuda.h"
#include "particles_3D.h"
#include "../grid3D.h"
#include "particles_boundaries_gpu.h"

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
  int ngrid =  (Particles.n_local + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  Set_Particles_Boundary_Kernel<<<dim1dGrid,dim1dBlock>>>( side, Particles.n_local, pos_dev, d_min, d_max, L  );
  CudaCheckError();

}


// #ifdef MPI_CHOLLA


__global__ void Get_Transfer_Flags_Kernel( part_int_t n_total, int side,  Real d_min, Real d_max, Real *pos_d, bool *transfer_flags_d ){
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  
  bool transfer = 0;
  
  Real pos = pos_d[tid];
  // if ( tid < 1 ) printf( "%f\n", pos);
  
  if ( side == 0 ){
    if ( pos < d_min ) transfer = 1;
  }
  
  if ( side == 1 ){
    if ( pos >= d_max ) transfer = 1;
  }
  
  // if ( transfer ) printf( "##Thread particles transfer\n");
  
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


__global__ void Sum_Blocks_Kernel( part_int_t n_total,  int *prefix_sum_d, int *prefix_sum_block_d){
  
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


__global__ void Get_Transfer_Indexs_Kernel( part_int_t n_total, bool *transfer_flags_d, int *prefix_sum_d, int *transfer_indxs_d){
  
  int tid, transfer_index;
  tid =  threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  transfer_index = prefix_sum_d[tid];
    
  if ( transfer_flags_d[tid] ) transfer_indxs_d[transfer_index] = tid;  
  
}

__global__ void Get_N_Transfer_Particles_Kernel( part_int_t n_total, int *n_transfer_d, bool *transfer_flags_d, int *prefix_sum_d ){
  n_transfer_d[0] = prefix_sum_d[n_total-1] + (int)transfer_flags_d[n_total-1];
  // if (n_transfer_d[0] != 0 ) printf( "##Thread transfer: %d\n", n_transfer_d[0]); 
}



__global__ void Remove_Transfred_Particles_Kernel( part_int_t n_total, int *n_transfered_d, bool *transfer_flags_d, int *prefix_sum_d, int *transfer_indxs_d, double *data_d ){

int tid_block, start_index, n, n_threads ;
tid_block =  threadIdx.x;
n_threads = blockDim.x;
int dst_indx;

int n_replace, n_transfered;
n_transfered = n_transfered_d[0];
// n_replace = n_transfered > ( n_total - n_transfered)  ?  n_total - n_transfered  : n_transfered;
n_replace = n_transfered;

__shared__ int N_replaced_sh[1];
__shared__ bool transfer_flags_sh[TPB_PARTICLES];

if ( tid_block == 0 ) N_replaced_sh[0] = 0;
__syncthreads();
// 
// if (tid_block == 0 ) printf( " N replace: %d \n", n_replace );
// 
n = 1;
while( N_replaced_sh[0] < n_replace ){
  // if (n==100) break;
  // if (tid_block == 0 ) printf(" Iteration: %d\n", n );
  start_index =  n_total - n*n_threads;
  if ( start_index + n_threads < 0 ) break;
  if ( start_index + tid_block >= 0 && start_index + tid_block < n_total) transfer_flags_sh[tid_block] = transfer_flags_d[start_index + tid_block];
  // else  transfer_flags_sh[tid_block] = 1; 
  __syncthreads();
  // printf( "%d \n", n*n_threads );
// 
  // if (tid_block == 0 ){
  //   for ( int i=n_threads-1; i>=0; i--){
  //     if ( start_index + i >= n_total || start_index + i < 0 ) continue;
  //     printf("%d    %d     %d \n", start_index + i, (int)transfer_flags_d[start_index + i], (int)transfer_flags_sh[i]  );
  //     // if (transfer_flags_d[start_index + i] != transfer_flags_sh[i] && start_index + i < n_total) printf("ERROR\n");
  //   }
  // }
  // 
  if ( tid_block == 0 ){
    for ( int i=n_threads-1; i>=0; i--){
      if ( start_index + i >= n_total || start_index + i < 0 ){
        // printf("Error in deleting tranfered particle data \n" );
        continue;
      }
      // printf( "%d  %d\n", start_index + i, (int)transfer_flags_sh[i]  );
      // if ( !transfer_flags_sh[i] ){
      if ( !transfer_flags_d[start_index+i] ){
        if ( N_replaced_sh[0] == n_replace ) continue;
        dst_indx = transfer_indxs_d[N_replaced_sh[0]];
        // printf("moving  %d   to   %d   %f  ->  %f  %d   %d  n_replaced: %d\n", start_index + i, dst_indx, data_d[start_index + i], data_d[dst_indx], (int) transfer_flags_d[start_index + i], (int) transfer_flags_sh[i], N_replaced_sh[0] +1 );
        data_d[dst_indx] = data_d[start_index + i];
        N_replaced_sh[0] += 1;
        __syncthreads();
      }
    }
    // printf("%d\n",n );
  }
  n += 1;
  __syncthreads();
}


// if ( tid_block == 0 ) printf(" N iterations: %d\n", n );


}


int Select_Particles_to_Transfer_GPU_function( part_int_t n_local, int side, Real domainMin, Real domainMax, Real *pos_d, int *n_transfer_d, int *n_transfer_h, bool *transfer_flags_d, int *transfer_indxs_d, int *transfer_partial_sum_d, int *transfer_sum_d  ){ 
  // set values for GPU kernels
  int ngrid, ngrid_half;
  ngrid =  (n_local - 1) / TPB_PARTICLES + 1;
  ngrid_half = ( n_local/2 ) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dGrid_half(ngrid_half, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  
  // chprintf("Getting transfer flags \n");
  Get_Transfer_Flags_Kernel<<<dim1dGrid,dim1dBlock>>>( n_local, side, domainMin, domainMax, pos_d, transfer_flags_d);
  CudaCheckError();
  
  Scan_Kernel<<<dim1dGrid_half,dim1dBlock>>>( n_local, transfer_flags_d, transfer_sum_d, transfer_partial_sum_d );
  CudaCheckError();
  
  Prefix_Sum_Blocks_Kernel<<<1,dim1dBlock >>>( ngrid_half, transfer_partial_sum_d );
  CudaCheckError();
  
  Sum_Blocks_Kernel<<<dim1dGrid,dim1dBlock>>>( n_local, transfer_sum_d, transfer_partial_sum_d );
  CudaCheckError();
  
  Get_N_Transfer_Particles_Kernel<<<1,1>>>( n_local,  n_transfer_d, transfer_flags_d, transfer_sum_d );
  CudaCheckError();
  
  CudaSafeCall( cudaMemcpy(n_transfer_h, n_transfer_d, sizeof(int), cudaMemcpyDeviceToHost) );
  CudaCheckError();
  
  Get_Transfer_Indexs_Kernel<<<dim1dGrid,dim1dBlock>>>( n_local , transfer_flags_d, transfer_sum_d, transfer_indxs_d );
  CudaCheckError();
  
  
  Remove_Transfred_Particles_Kernel<<<1,dim1dBlock >>>( n_local, n_transfer_d, transfer_flags_d, transfer_sum_d, transfer_indxs_d, pos_d );
  CudaCheckError();
  
  // chprintf( "N transfer: %d\n", n_transfer_h[0]);

  
  return n_transfer_h[0];


}


// #endif//MPI_CHOLLA


#endif //PARTICLES
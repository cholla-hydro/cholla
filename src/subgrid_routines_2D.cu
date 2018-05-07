/*! \file subgrid_routines_2D.cu
 *  \brief Definitions of the routines for subgrid gpu staging for 2D CTU. */

#ifdef CUDA

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include"global.h"
#include"subgrid_routines_2D.h"



void sub_dimensions_2D(int nx, int ny, int n_ghost, int *nx_s, int *ny_s, int *block1_tot, int *block2_tot, int *remainder1, int *remainder2, int n_fields) {

  int sx = 2;
  int sy = 2;
  size_t free;
  size_t total;
  int cell_mem, max_vol;

  *nx_s = nx;
  *ny_s = ny;

  // determine the amount of free memory available on the device
  cudaMemGetInfo(&free, &total);

  // use that to determine the maximum subgrid block volume
  // memory used per cell (arrays allocated on GPU)
  cell_mem = 8*n_fields*sizeof(Real);
  cell_mem += 4*sizeof(Real);
  max_vol = free / cell_mem; 
  // plus a buffer for dti array
  max_vol = max_vol - 400;

  //max_vol = 1000;


  // split if necessary
  while ((*nx_s)*(*ny_s) > max_vol) {

    // if x dimension is bigger, split in x
    if (*nx_s > *ny_s) {
      *nx_s = ceil(Real (nx-2*n_ghost) / Real (sx)) + 2*n_ghost;
      sx++;
    }
    // otherwise split in y
    else {
      *ny_s = ceil(Real (ny-2*n_ghost) / Real (sy)) + 2*n_ghost;
      sy++;
    }
  }

  // determine the number of blocks needed
  if (*nx_s == nx && *ny_s == ny) {
    *block1_tot = 1;
    *block2_tot = 1;
    *remainder1 = 0;
    *remainder2 = 0;
    return;
  }
  else if (*nx_s < nx && *ny_s == ny) {
    *block1_tot = ceil(Real (nx-2*n_ghost) / Real (*nx_s-2*n_ghost) );
    *block2_tot = 1;
    // calculate the remainder
    *remainder1 = (nx-2*n_ghost)%(*nx_s-2*n_ghost);
    *remainder2 = 0;
  }  
  else if (*nx_s == nx && *ny_s < ny) {
    *block1_tot = 1;
    *block2_tot = ceil(Real (ny-2*n_ghost) / Real (*ny_s-2*n_ghost) );
    // calculate the remainder
    *remainder1 = 0;
    *remainder2 = (ny-2*n_ghost)%(*ny_s-2*n_ghost);
  }
  else if (*nx_s < nx && *ny_s < ny) {
    *block1_tot = ceil(Real (nx-2*n_ghost) / Real (*nx_s-2*n_ghost) );
    *block2_tot = ceil(Real (ny-2*n_ghost) / Real (*ny_s-2*n_ghost) );
    // calculate the remainder
    *remainder1 = (nx-2*n_ghost)%(*nx_s-2*n_ghost);
    *remainder2 = (ny-2*n_ghost)%(*ny_s-2*n_ghost);
  }  
  else {
    printf("Error determining number of subgrid blocks.\n");
    exit(0);
  }


}


void get_offsets_2D(int nx_s, int ny_s, int n_ghost, int x_off, int y_off, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int *x_off_s, int *y_off_s) {

  int block1;
  int block2;

  // determine which row of subgrid blocks we're on for each dimension
  block2 = block / block1_tot; // yid of current block
  block1 = block - block2*block1_tot; // xid of current block
  // calculate global offsets
  *x_off_s = x_off + (nx_s-2*n_ghost)*block1;
  *y_off_s = y_off + (ny_s-2*n_ghost)*block2;
  // need to be careful on the last block due to remainder offsets
  if (remainder1 != 0 && block1 == block1_tot-1) *x_off_s = x_off + (nx_s-2*n_ghost)*(block1-1) + remainder1;
  if (remainder2 != 0 && block2 == block2_tot-1) *y_off_s = y_off + (ny_s-2*n_ghost)*(block2-1) + remainder2;
  
}




// copy the conserved variable block into the buffer
void host_copy_block_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real *buffer, int n_fields) {

  int n_cells = nx*ny;
  int block1, block2;
  int x_offset, y_offset;
  int x_host, y_host;  

  // if no subgrid blocks, do nothing
  if (nx_s == nx && ny_s == ny) return;

  // splitting only in x
  else if (nx_s < nx && ny_s == ny) {

    block1 = block; // xid of block

    // if we are on the last block, make sure it doesn't go past
    // the bounds of the host array
    x_offset = 0;
    if (block1 == block1_tot-1 && remainder1 != 0) {
      x_offset = nx_s - 2*n_ghost - remainder1;
    }
    // calculate the x location in the host array to copy from
    x_host = block1*(nx_s-2*n_ghost) - x_offset;

    // copy data from host conserved array into buffer
    for (int j=0; j<ny_s; j++) {
      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&buffer[ii*BLOCK_VOL + j*nx_s], &host_conserved[x_host + ii*n_cells + j*nx], nx_s*sizeof(Real)); 
      }
    }

    return;
  }

  // splitting only in y 
  else if (nx_s == nx && ny_s < ny) {

    block2 = block; // yid of block

    // if we are on the last block, make sure it doesn't go past 
    // the bounds of the host array
    y_offset = 0;
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }
    // calculate the y location in the host array to copy from
    y_host = block2*nx*(ny_s-2*n_ghost) - nx*y_offset;

    // copy data from host conserved array into buffer
    for (int ii=0; ii<n_fields; ii++) {
      memcpy(&buffer[ii*BLOCK_VOL], &host_conserved[y_host + ii*n_cells], BLOCK_VOL*sizeof(Real));     
    }

    return;
  }

  // splitting in x & y
  else if (nx_s < nx && ny_s < ny) {

    block2 = block / block1_tot; // yid of current block
    block1 = block - block2*block1_tot; // xid of current block

    // if we are on the last y block, make sure it doesn't go past 
    // the bounds of the host array
    y_offset = 0;
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }
    // calculate the y location in the host array to copy from
    y_host = block2*nx*(ny_s-2*n_ghost) - nx*y_offset;

    // if we are on the last x block, make sure it doesn't go past 
    // the bounds of the host array
    x_offset = 0;
    if (block1 == block1_tot-1 && remainder1 != 0) {
        x_offset = nx_s - 2*n_ghost - remainder1;
    }
    // calculate the x location in the host array to copy from
    x_host = block1*(nx_s-2*n_ghost) - x_offset;

    // copy data from the host conserved array into buffer
    for (int j=0; j<ny_s; j++) {
      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&buffer[ii*BLOCK_VOL + j*nx_s], &host_conserved[x_host + y_host + ii*n_cells + j*nx], nx_s*sizeof(Real)); 
      }
    }

    return;

  }

  else {
    printf("Error copying conserved variable block into CPU buffer. Illegal grid dimensions.\n");
    exit(0);
  }

}



// return the values from buffer to the host_conserved array
void host_return_block_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real *buffer, int n_fields) {

  int n_cells = nx*ny;
  int block1, block2;
  int x_offset, y_offset;
  int x_host, x_gpu, y_host, y_gpu;
  int length;

  // if no subgrid blocks, do nothing
  if (nx_s == nx && ny_s == ny) return;

  // splitting only in x
  else if (nx_s < nx && ny_s == ny) {

    // return values based on current block id
    block1 = block; // xid of current block

    // if we just did the last x block, make sure to copy the cells to the right place
    x_offset = 0;
    if (block1 == block1_tot-1 && remainder1 != 0) {
      x_offset = nx_s - 2*n_ghost - remainder1;
    }

    y_host = nx*n_ghost;
    x_host = block1*(nx_s-2*n_ghost) + (n_ghost-x_offset);
    y_gpu  = n_ghost*nx_s;
    x_gpu  = n_ghost;
    length = (nx_s-2*n_ghost); // number of cells to copy back
    
    for (int j=0; j<ny_s-2*n_ghost; j++) {
      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&host_conserved[x_host + y_host + j*nx + ii*n_cells], &buffer[x_gpu + y_gpu + j*nx_s + ii*BLOCK_VOL], length*sizeof(Real));
      }
    }

    return;
  }

  // splitting only in y
  else if (nx_s == nx && ny_s < ny) {

    // return values based on current block id
    block2 = block;

    // if we just did the last y block, make sure to copy the cells to the right place
    y_offset = 0;
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }

    y_host = block2*nx*(ny_s-2*n_ghost) + nx*(n_ghost-y_offset);
    y_gpu  = n_ghost*nx_s;
    length = nx_s*(ny_s-2*n_ghost); // number of cells to copy back

    for (int ii=0; ii<n_fields; ii++) {
      memcpy(&host_conserved[y_host + ii*n_cells], &buffer[y_gpu + ii*BLOCK_VOL], length*sizeof(Real));
    }

    return;
  }

  // splitting in x and y
  else if (nx_s < nx && ny_s < ny) {

    // return values based on current block id
    block2 = block / block1_tot; // yid of current block
    block1 = block - block2*block1_tot; // xid of current block

    // if we just did the last y block, make sure to copy the cells to the right place
    y_offset = 0;
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }

    // if we just did the last x block, make sure to copy the cells to the right place
    x_offset = 0;
    if (block1 == block1_tot-1 && remainder1 != 0) {
      x_offset = nx_s - 2*n_ghost - remainder1;
    }

    y_host = block2*nx*(ny_s-2*n_ghost) + nx*(n_ghost-y_offset);
    x_host = block1*(nx_s-2*n_ghost) + (n_ghost-x_offset);
    y_gpu  = n_ghost*nx_s;
    x_gpu  = n_ghost;
    length = (nx_s-2*n_ghost); // number of cells to copy back
    
    for (int j=0; j<ny_s-2*n_ghost; j++) {
      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&host_conserved[x_host + y_host + j*nx + ii*n_cells], &buffer[x_gpu + y_gpu + j*nx_s + ii*BLOCK_VOL], length*sizeof(Real));
      }
    }

    return;
  }


  else {
    printf("Error copying values back into host array. Illegal grid dimensions.\n");
    exit(0);
  }


}


#endif //CUDA

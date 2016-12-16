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
    // caclulate the remainder
    *remainder1 = (nx-2*n_ghost)%(*nx_s-2*n_ghost);
    *remainder2 = 0;
  }  
  else if (*nx_s == nx && *ny_s < ny) {
    *block1_tot = 1;
    *block2_tot = ceil(Real (ny-2*n_ghost) / Real (*ny_s-2*n_ghost) );
    // caclulate the remainder
    *remainder1 = 0;
    *remainder2 = (ny-2*n_ghost)%(*ny_s-2*n_ghost);
  }
  else if (*nx_s < nx && *ny_s < ny) {
    *block1_tot = ceil(Real (nx-2*n_ghost) / Real (*nx_s-2*n_ghost) );
    *block2_tot = ceil(Real (ny-2*n_ghost) / Real (*ny_s-2*n_ghost) );
    // caclulate the remainder
    *remainder1 = (nx-2*n_ghost)%(*nx_s-2*n_ghost);
    *remainder2 = (ny-2*n_ghost)%(*ny_s-2*n_ghost);
  }  
  else {
    printf("Error determining number of subgrid blocks.\n");
    exit(0);
  }


}




// allocate memory for the CPU buffers
void allocate_buffers_2D(int block1_tot, int block2_tot, int BLOCK_VOL, Real **&buffer, int n_fields) {

  int n;

  // if we don't need any buffers, don't allocate any 
  if (block1_tot == 1 && block2_tot == 1) {
    return;
  }

  // splitting only in x, need two buffers
  else if (block1_tot > 1 && block2_tot == 1) {
    n = 2;
  }
 
  // splitting only in y, need two buffers
  else if (block1_tot == 1 && block2_tot > 1) {
    n = 2; 
  }
  
  // splitting in x & y, need 2*block1_tot + 1 buffers
  else if (block1_tot > 1 && block2_tot > 1) {
    n = 2*block1_tot + 1;
  }

  // else throw an error
  else {
    printf("Error allocating subgrid buffers for GPU transfer. Unsupported grid dimensions.\n");
    exit(0);
  }
  
  // allocate n buffers within buffer array
  buffer = (Real **) malloc(n*sizeof(Real *));

  // allocate memory for each of those buffers
  for (int i=0; i<n; i++)
  {
    buffer[i] = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real));
  }

}






// copy the first conserved variable block(s) into buffer
void host_copy_init_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int remainder1, int BLOCK_VOL, Real *host_conserved, Real **buffer, Real **tmp1, Real **tmp2, int n_fields) {

  int n_cells = nx*ny;
  int block1;
  int x_offset, x_host;  

  // if no need for subgrid blocks, simply point tmp1
  // and tmp2 to the host array
  if (nx_s == nx && ny_s == ny) {
    *tmp1 = host_conserved;
    *tmp2 = host_conserved;
    return;
  }

  // splitting only in x
  else if (nx_s < nx && ny_s == ny) {

    // copy the first block into a buffer
    for (int j=0; j<ny_s; j++) {
      for (int ii=0; ii<n_fields; ii++) {
      memcpy(&buffer[0][ii*BLOCK_VOL + j*nx_s], &host_conserved[ii*n_cells + j*nx], nx_s*sizeof(Real)); 
      }
    }

    // point tmp1 to the buffer we will
    // copy from, and tmp2 to the buffer we will
    // return calculated blocks to
    *tmp1 = buffer[0];    
    *tmp2 = buffer[1];

    return;
  }

  // splitting only in y 
  else if (nx_s == nx && ny_s < ny) {
    // copy the first block into a cpu buffer
    for (int ii=0; ii<n_fields; ii++) {
      memcpy(&buffer[0][n_fields*BLOCK_VOL], &host_conserved[n_fields*n_cells], BLOCK_VOL*sizeof(Real)); // Energy
    }

    // point tmp1 to the buffer we will
    // copy from, and tmp2 to the buffer we will
    // return calculated blocks to
    *tmp1 = buffer[0];
    *tmp2 = buffer[1];

    return;
  }

  // splitting in x & y
  else if (nx_s < nx && ny_s < ny) {

    // copy the first x row into buffers
    for (int n=0; n<block1_tot; n++) {

      block1 = block+n; // xid of block

      // if we are about to copy the last x block, make sure it doesn't go past 
      // the bounds of the host array
      x_offset = 0;
      if (block1 == block1_tot-1 && remainder1 != 0) {
          x_offset = nx_s - 2*n_ghost - remainder1;
      }

      // calculate the x locations in the host array to copy from
      x_host = block1*(nx_s-2*n_ghost) - x_offset;

      for (int j=0; j<ny_s; j++) {
        for (int ii=0; ii<n_fields; ii++) {
          memcpy(&buffer[n][n_fields*BLOCK_VOL + j*nx_s], &host_conserved[x_host + n_fields*n_cells + j*nx], nx_s*sizeof(Real)); 
        }
      }

    }

    // point tmp1 to the first buffer we will
    // copy from, and tmp2 to the buffer we will
    // return calculated blocks to
    *tmp1 = buffer[0];    
    *tmp2 = buffer[2*block1_tot];

    return;

  }

  else {
    printf("Error copying first conserved variable block into CPU buffer. Illegal grid dimensions.\n");
    exit(0);
  }

}






// copy the next conserved variable blocks into the remaining buffers
void host_copy_next_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real **buffer, Real **tmp1, int n_fields) {
  
  int n_cells = nx*ny;
  int block1, block2;
  int x_offset, y_offset;
  int x_host, y_host;
  int buf_offset;

  // if no subgrid blocks, do nothing
  if (nx_s == nx && ny_s == ny) return;


  // splitting only in x
  else if (nx_s < nx && ny_s == ny) {

    block1 = block+1; // xid of next block
  
    // don't copy if we are currently on the last block
    if (block1 < block1_tot) {

      // if we are on the second-to-last block, make sure the last block doesn't go past 
      // the bounds of the host array
      x_offset = 0;
      if (block1 == block1_tot-1 && remainder1 != 0) {
        x_offset = nx_s - 2*n_ghost - remainder1;
      }

      // calculate the x location in the host array to copy from
      x_host = block1*(nx_s-2*n_ghost) - x_offset;

      for (int j=0; j<ny_s; j++) {
        for (int ii=0; ii<n_fields; ii++) {
          memcpy(&buffer[0][n_fields*BLOCK_VOL + j*nx_s], &host_conserved[x_host + n_fields*n_cells + j*nx], nx_s*sizeof(Real)); // Energy
        }
      }

      // point to the next buffer
      //*tmp1 = buffer[0];

    }

    return;
  }


  // splitting only in y
  else if (nx_s == nx && ny_s < ny) {

    block2 = block+1; // yid of next block
  
    // don't copy if we are currently on the last block
    if (block2 < block2_tot) {

      // if we are on the second-to-last block, make sure the last block doesn't go past 
      // the bounds of the host array
      y_offset = 0;
      if (block2 == block2_tot-1 && remainder2 != 0) {
        y_offset = ny_s - 2*n_ghost - remainder2;
      }

      // calculate the y location in the host array to copy from
      y_host = block2*nx*(ny_s-2*n_ghost) - nx*y_offset;

      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&buffer[0][n_fields*BLOCK_VOL], &host_conserved[y_host + ii*n_cells], BLOCK_VOL*sizeof(Real));
      }

      // point to the next buffer
      //*tmp1 = buffer[0];

    }

    return;
  }

  // splitting in x & y
  else if (nx_s < nx && ny_s < ny) {

    block2 = block / block1_tot; // yid of current block
    block1 = block - block2*block1_tot; // xid of current block

    // if we are about to copy the last y block, make sure it doesn't go past 
    // the bounds of the host array
    y_offset = 0;
    if (block2 == block2_tot-2 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }

    // if we are at the start of an x row, and not on the last y column, 
    // copy the next x row into buffers
    // so we don't overwrite anything when we return the data to the CPU
    if (block1 == 0 && block2 < block2_tot - 1) {

      // even y column, copy x row into buffers n:2(n-1)
      if (block2%2 == 0) buf_offset = block1_tot;
      // odd  y column, copy x row into buffers 0:n-1
      if (block2%2 == 1) buf_offset = 0;

      for (int n=0; n<block1_tot; n++) {

        // if we are about to copy the last x block, make sure it doesn't go past 
        // the bounds of the host array
        x_offset = 0;
        if (n == block1_tot-1 && remainder1 != 0) {
          x_offset = nx_s - 2*n_ghost - remainder1;
        }

        // calculate the x & y locations in the host array to copy from
        y_host = (block2+1)*nx*(ny_s-2*n_ghost) - nx*y_offset;
        x_host = (block1+n)*(nx_s-2*n_ghost) - x_offset;

        for (int j=0; j<ny_s; j++) {
          for (int ii=0; ii<n_fields; ii++) {
            memcpy(&buffer[n+buf_offset][n_fields*BLOCK_VOL + j*nx_s], &host_conserved[x_host + y_host + n_fields*n_cells + j*nx], nx_s*sizeof(Real)); 
          }
        }
      }
    }

    // unless we are currently on the last block,
    // set tmp1 to the address of the buffer containing the next block to be calculated
    if (block+1 < block1_tot*block2_tot) {

      // even y column, use pointers from buffers 0:n-1
      if (block2%2 == 0) buf_offset = 0;
      // odd  y column, use pointers from buffers n:2(n-1)
      if (block2%2 == 1) buf_offset = block1_tot;

      // point to the next buffer
      *tmp1 = buffer[(block1+buf_offset+1)%(2*block1_tot)];
      }

      return;
  }

  else {
    printf("Error copying next blocks into buffers. Illegal grid dimensions.\n");
    exit(0);
  }


}





// return the values from buffer to the host_conserved array
void host_return_values_2D(int nx, int ny, int nx_s, int ny_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real **buffer, int n_fields) {

  int n_cells = nx*ny;
  int block1, block2;
  int x_offset, y_offset;
  int x_host, x_gpu, y_host, y_gpu;
  int length;
  int n;

  // if no subgrid blocks, do nothing
  if (nx_s == nx && ny_s == ny) return;

  // splitting only in x
  else if (nx_s < nx && ny_s == ny) {

    // return values based on current block id
    block1 = block; // xid of current block

    x_offset = 0;
    // if we just did the last x slice, make sure to copy the cells to the right place
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
        memcpy(&host_conserved[x_host + y_host + j*nx + n_fields*n_cells], &buffer[1][x_gpu + y_gpu + j*nx_s + n_fields*BLOCK_VOL], length*sizeof(Real));
      }
    }

    return;
  }

  // splitting only in y
  else if (nx_s == nx && ny_s < ny) {

    // return values based on current block id
    block2 = block;
    y_offset = 0;

    // if we just did the last slice, make sure to copy the cells to the right place
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }

    y_host = block2*nx*(ny_s-2*n_ghost) + nx*(n_ghost-y_offset);
    y_gpu  = n_ghost*nx_s;
    length = nx_s*(ny_s-2*n_ghost); // number of cells to copy back

    for (int ii=0; ii<n_fields; ii++) {
      memcpy(&host_conserved[y_host + n_fields*n_cells], &buffer[1][y_gpu + n_fields*BLOCK_VOL], length*sizeof(Real));
    }

    return;
  }

  // splitting in x and y
  else if (nx_s < nx && ny_s < ny) {

    // buffer holding the returned values (always the last buffer)
    n = 2*block1_tot;

    // return values based on current block id
    block2 = block / block1_tot; // yid of current block
    block1 = block - block2*block1_tot; // xid of current block

    y_offset = 0;
    // if we just did the y last slice, make sure to copy the cells to the right place
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }

    x_offset = 0;
    // if we just did the last x slice, make sure to copy the cells to the right place
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
        memcpy(&host_conserved[x_host + y_host + j*nx + n_fields*n_cells], &buffer[n][x_gpu + y_gpu + j*nx_s + n_fields*BLOCK_VOL], length*sizeof(Real));
      }
    }

    return;
  }


  else {
    printf("Error copying values back into host array. Illegal grid dimensions.\n");
    exit(0);
  }


}




void free_buffers_2D(int nx, int ny, int nx_s, int ny_s, int block1_tot, int block2_tot, Real **buffer) {

  int n;

  if (nx_s == nx && ny_s == ny) return;

  else if (nx_s < nx && ny_s == ny) n = 2; 

  else if (nx_s == nx && ny_s < ny) n = 2; 

  else if (nx_s < nx && ny_s < ny) n = 2*block1_tot + 1;

  else {
    printf("Illegal grid dimensions.\n");
    exit(0);
  }

  for (int i=0; i<n; i++) {
    free(buffer[i]);
  }

  free(buffer);

}



#endif //CUDA

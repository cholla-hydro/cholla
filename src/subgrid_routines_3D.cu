/*! \file subgrid_routines_3D.cu
 *  \brief Definitions of the routines for subgrid gpu staging for 3D CTU. */

#ifdef CUDA

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include"global.h"
#include"mpi_routines.h"
#include"subgrid_routines_3D.h"





void sub_dimensions_3D(int nx, int ny, int nz, int n_ghost, int *nx_s, int *ny_s, int *nz_s, int *block1_tot, int *block2_tot, int *block3_tot, int *remainder1, int *remainder2, int *remainder3, int n_fields) {

  int sx = 2;
  int sy = 2;
  int sz = 2;
  size_t free;
  size_t total;
  int cell_mem, max_vol;

  *nx_s = nx;
  *ny_s = ny;
  *nz_s = nz;

  // determine the amount of free memory available on the device
  cudaMemGetInfo(&free, &total);

  // use that to determine the maximum subgrid block volume
  // memory used per cell (arrays allocated on GPU)
  cell_mem = 11*n_fields*sizeof(Real);
  cell_mem += 6*sizeof(Real);
  max_vol = free / cell_mem; 
  // plus a buffer for dti array
  max_vol = max_vol - 400;
  max_vol = 262144;

  // split if necessary - subgrid block volume cannot exceed MAX_VOL_3D
  // try to keep the ratio of the y & z dimensions close to 1
  // do not let the ratio of the geometric mean of the y&z dimensions to the
  // x dimension exceed 5 - we don't want cubes, but we don't want
  // REALLY NOT CUBES
  while ((*nx_s)*(*ny_s)*(*nz_s) > max_vol) {

    // if the aspect ratio has gotten too large, split in x 
    if ((*nx_s) / sqrt((*ny_s)*(*nz_s)) > 5) {
        *nx_s = ceil(Real (nx-2*n_ghost) / Real (sx)) + 2*n_ghost;
        sx++;
    }
    else {
      if (*ny_s > *nz_s) {
        *ny_s = ceil(Real (ny-2*n_ghost) / Real (sy)) + 2*n_ghost;
        sy++;
      }
      else {
        *nz_s = ceil(Real (nz-2*n_ghost) / Real (sz)) + 2*n_ghost;
        sz++;
      }
    }

  }

  // determine the number of blocks needed
  // not splitting
  if (*nx_s == nx && *ny_s == ny && *nz_s == nz) {
    *block1_tot = 1;
    *block2_tot = 1;
    *block3_tot = 1;
    *remainder1 = 0;
    *remainder2 = 0;
    *remainder3 = 0;
    return;
  }
  // splitting in x
  else if (*nx_s < nx && *ny_s == ny && *nz_s == nz) {
    *block1_tot = ceil(Real (nx-2*n_ghost) / Real (*nx_s-2*n_ghost) );
    *block2_tot = 1;
    *block3_tot = 1;
    // caclulate the remainder
    *remainder1 = (nx-2*n_ghost)%(*nx_s-2*n_ghost);
    *remainder2 = 0;
    *remainder3 = 0;
  }  
  // splitting in y
  else if (*nx_s == nx && *ny_s < ny && *nz_s == nz) {
    *block1_tot = 1;
    *block2_tot = ceil(Real (ny-2*n_ghost) / Real (*ny_s-2*n_ghost) );
    *block3_tot = 1;
    // caclulate the remainder
    *remainder1 = 0;
    *remainder2 = (ny-2*n_ghost)%(*ny_s-2*n_ghost);
    *remainder3 = 0;
  }  
  // splitting in z
  else if (*nx_s == nx && *ny_s == ny && *nz_s < nz) {
    *block1_tot = 1;
    *block2_tot = 1;
    *block3_tot = ceil(Real (nz-2*n_ghost) / Real (*nz_s-2*n_ghost) );
    // caclulate the remainder
    *remainder1 = 0;
    *remainder2 = 0;
    *remainder3 = (nz-2*n_ghost)%(*nz_s-2*n_ghost);
  }
  // splitting in x & y
  else if (*nx_s < nx && *ny_s < ny && *nz_s == nz) {
    *block1_tot = ceil(Real (nx-2*n_ghost) / Real (*nx_s-2*n_ghost) );
    *block2_tot = ceil(Real (ny-2*n_ghost) / Real (*ny_s-2*n_ghost) );
    *block3_tot = 1;
    // caclulate the remainder
    *remainder1 = (nx-2*n_ghost)%(*nx_s-2*n_ghost);
    *remainder2 = (ny-2*n_ghost)%(*ny_s-2*n_ghost);
    *remainder3 = 0;
  }  
  // splitting in y & z
  else if (*nx_s == nx && *ny_s < ny && *nz_s < nz) {
    *block1_tot = 1;
    *block2_tot = ceil(Real (ny-2*n_ghost) / Real (*ny_s-2*n_ghost) );
    *block3_tot = ceil(Real (nz-2*n_ghost) / Real (*nz_s-2*n_ghost) );
    // caclulate the remainders
    *remainder1 = 0;
    *remainder2 = (ny-2*n_ghost)%(*ny_s-2*n_ghost);
    *remainder3 = (nz-2*n_ghost)%(*nz_s-2*n_ghost);
  }
  // splitting in x & z
  else if (*nx_s < nx && *ny_s == ny && *nz_s < nz) {
    *block1_tot = ceil(Real (nx-2*n_ghost) / Real (*nx_s-2*n_ghost) );
    *block2_tot = 1;
    *block3_tot = ceil(Real (nz-2*n_ghost) / Real (*nz_s-2*n_ghost) );
    // caclulate the remainder
    *remainder1 = (nx-2*n_ghost)%(*nx_s-2*n_ghost);
    *remainder2 = 0;
    *remainder3 = (nz-2*n_ghost)%(*nz_s-2*n_ghost);
  }  
  // splitting in x, y & z
  else if (*nx_s < nx && *ny_s < ny && *nz_s < nz) {
    *block1_tot = ceil(Real (nx-2*n_ghost) / Real (*nx_s-2*n_ghost) );
    *block2_tot = ceil(Real (ny-2*n_ghost) / Real (*ny_s-2*n_ghost) );
    *block3_tot = ceil(Real (nz-2*n_ghost) / Real (*nz_s-2*n_ghost) );
    // caclulate the remainders
    *remainder1 = (nx-2*n_ghost)%(*nx_s-2*n_ghost);
    *remainder2 = (ny-2*n_ghost)%(*ny_s-2*n_ghost);
    *remainder3 = (nz-2*n_ghost)%(*nz_s-2*n_ghost);
  }  
  else {
    printf("Error determining number and size of subgrid blocks.\n");
    exit(0);
  }


}


void get_offsets_3D(int nx_s, int ny_s, int nz_s, int n_ghost, int x_off, int y_off, int z_off, int block, int block1_tot, int block2_tot, int block3_tot, int remainder1, int remainder2, int remainder3, int *x_off_s, int *y_off_s, int *z_off_s) {

  int block1;
  int block2;
  int block3;

  // determine which row of subgrid blocks we're on for each dimension
  block3 = block / (block2_tot*block1_tot); // zid of current block
  block2 = (block - block3*block2_tot*block1_tot) / block1_tot; // yid of current block
  block1 = block - block3*block2_tot*block1_tot - block2*block1_tot; // xid of current block
  // calculate global offsets
  *x_off_s = x_off + (nx_s-2*n_ghost)*block1;
  *y_off_s = y_off + (ny_s-2*n_ghost)*block2;
  *z_off_s = z_off + (nz_s-2*n_ghost)*block3;
  // need to be careful on the last block due to remainder offsets
  if (remainder1 != 0 && block1 == block1_tot-1) *x_off_s = x_off + (nx_s-2*n_ghost)*(block1-1) + remainder1;
  if (remainder2 != 0 && block2 == block2_tot-1) *y_off_s = y_off + (ny_s-2*n_ghost)*(block2-1) + remainder2;
  if (remainder3 != 0 && block3 == block3_tot-1) *z_off_s = z_off + (nz_s-2*n_ghost)*(block3-1) + remainder3;
  
}



// allocate memory for the CPU buffers
void allocate_buffers_3D(int block1_tot, int block2_tot, int block3_tot, int BLOCK_VOL, Real **&buffer, int n_fields) {

  int n;

  // if we don't need any buffers, don't allocate any 
  if (block1_tot == 1 && block2_tot == 1 && block3_tot == 1) {
    return;
  }
  // splitting only in x, need two buffers
  else if (block1_tot > 1 && block2_tot == 1 && block3_tot == 1) {
    n = 2; 
  }
  // splitting only in y, need two buffers
  else if (block1_tot == 1 && block2_tot > 1 && block3_tot == 1) {
    n = 2; 
  }
  // splitting only in z, need two buffers
  else if (block1_tot == 1 && block2_tot == 1 && block3_tot > 1) {
    n = 2; 
  }
  // splitting in y and z, need 2*block2_tot + 1 buffers 
  else if (block1_tot == 1 && block2_tot > 1 && block3_tot > 1) {
    n = 2*block2_tot + 1;
  }
  // splitting in x, y, and z, need 2*block1_tot*block2_tot + 1 buffers
  else if (block1_tot > 1 && block2_tot > 1 && block3_tot > 1) {
    n = 2*block1_tot*block2_tot + 1;
  }

  else {
    printf("Error allocating subgrid buffers for GPU transfer. Unsupported subgrid grid dimensions.\n");
    printf("x blocks: %d  y blocks: %d  z blocks: %d.\n", block1_tot, block2_tot, block3_tot);
    exit(0);
  }
  
  // allocate n buffers within buffer array
  buffer = (Real **) malloc(n*sizeof(Real *));
  
  //printf("xblocks: %d  yblocks: %d  zblocks: %d\n", block1_tot, block2_tot, block3_tot);
  //printf("BLOCK_VOL: %d\n", BLOCK_VOL);

  // allocate memory for each of those buffers
  for (int i=0; i<n; i++)
  {
    //printf("Buffer %d\n", i);
    if ( NULL == ( buffer[i] = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real)) ) ) {
      printf("Failed to allocate CPU buffers.\n");
    }
  }

}




// copy the first conserved variable block(s) into buffer(s)
void host_copy_init_3D(int nx, int ny, int nz, int nx_s, int ny_s, int nz_s, int n_ghost, int block, int block1_tot, int block2_tot, int remainder1, int remainder2, int BLOCK_VOL, Real *host_conserved, Real **buffer, Real **tmp1, Real **tmp2, int n_fields) {

  int n_cells = nx*ny*nz;
  int block1, block2;
  int x_offset, x_host, y_offset, y_host;  

  // if no need for subgrid blocks, simply point tmp1
  // and tmp2 to the host array
  if (nx_s == nx && ny_s == ny && nz_s == nz) {
    *tmp1 = host_conserved;
    *tmp2 = host_conserved;
    return;
  }

  // splitting only in x
  else if (nx_s < nx && ny_s == ny && nz_s == nz) {

    // copy the first block into a buffer
    for (int k=0; k<nz_s; k++) {
      for (int j=0; j<ny_s; j++) {
        for (int ii=0; ii<n_fields; ii++) {
          memcpy(&buffer[0][ii*BLOCK_VOL + j*nx_s + k*nx_s*ny_s], &host_conserved[ii*n_cells + j*nx + k*nx*ny], nx_s*sizeof(Real)); 
        }
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
  else if (nx_s == nx && ny_s < ny && nz_s == nz) {

    // copy the first block into a buffer
    for (int k=0; k<nz_s; k++) {
      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&buffer[0][ii*BLOCK_VOL + k*nx_s*ny_s], &host_conserved[ii*n_cells + k*nx*ny], nx_s*ny_s*sizeof(Real)); 
      }
    }

    // point tmp1 to the buffer we will
    // copy from, and tmp2 to the buffer we will
    // return calculated blocks to
    *tmp1 = buffer[0];    
    *tmp2 = buffer[1];

    return;
  }

  // splitting only in z 
  else if (nx_s == nx && ny_s == ny && nz_s < nz) {

    // copy the first block into a cpu buffer
    for (int ii=0; ii<n_fields; ii++) {
      memcpy(&buffer[0][ii*BLOCK_VOL], &host_conserved[ii*n_cells], BLOCK_VOL*sizeof(Real)); 
    }

    // point tmp1 to the first buffer we will
    // copy from, and tmp2 to the buffer we will
    // return calculated blocks to
    *tmp1 = buffer[0];
    *tmp2 = buffer[1];

    return;
  }


  // splitting in y and z
  else if (nx_s == nx && ny_s < ny && nz_s < nz) {

    // copy the first y row into buffers
    for (int n=0; n<block2_tot; n++) {

      block2 = n; // yid of block

      // if we are about to copy the last y block, make sure it doesn't go past 
      // the bounds of the host array
      y_offset = 0;
      if (block2 == block2_tot-1 && remainder2 != 0) {
          y_offset = ny_s - 2*n_ghost - remainder2;
      }

      // calculate the y locations in the host array to copy from
      y_host = block2*nx*(ny_s-2*n_ghost) - nx*y_offset;

      for (int k=0; k<nz_s; k++) {
        for (int ii=0; ii<n_fields; ii++) {
          memcpy(&buffer[n][ii*BLOCK_VOL + k*nx_s*ny_s], &host_conserved[y_host + ii*n_cells + k*nx*ny], nx_s*ny_s*sizeof(Real)); 
        }
      }

    }

    // point tmp1 to the first buffer we will
    // copy from, and tmp2 to the buffer we will
    // return calculated blocks to
    *tmp1 = buffer[0];    
    *tmp2 = buffer[2*block2_tot];

    return;
  }

  // splitting in x, y, and z
  else if (nx_s < nx && ny_s < ny && nz_s < nz) {


    // copy the first y row into buffers
    for (int n=0; n<block2_tot; n++) {
      // copy the first x row into buffers
      for (int m=0; m<block1_tot; m++) {

        block1 = m; // xid of block
        block2 = n; // yid of block

        // if we are about to copy the last x block, make sure it doesn't go past 
        // the bounds of the host array
        x_offset = 0;
        if (block1 == block1_tot-1 && remainder1 != 0) {
            x_offset = nx_s - 2*n_ghost - remainder1;
        }

        // if we are about to copy the last y block, make sure it doesn't go past 
        // the bounds of the host array
        y_offset = 0;
        if (block2 == block2_tot-1 && remainder2 != 0) {
            y_offset = ny_s - 2*n_ghost - remainder2;
        }

        // calculate the x & y locations in the host array to copy from
        x_host = block1*(nx_s-2*n_ghost) - x_offset;
        y_host = block2*nx*(ny_s-2*n_ghost) - nx*y_offset;

        for (int k=0; k<nz_s; k++) {
          for (int j=0; j<ny_s; j++) {
            for (int ii=0; ii<n_fields; ii++) {
              memcpy(&buffer[m + n*block1_tot][ii*BLOCK_VOL + j*nx_s + k*nx_s*ny_s], &host_conserved[x_host + y_host + ii*n_cells + j*nx + k*nx*ny], nx_s*sizeof(Real)); 
            }
          }
        }

      }

    }

    // point tmp1 to the first buffer we will
    // copy from, and tmp2 to the buffer we will
    // return calculated blocks to
    *tmp1 = buffer[0];    
    *tmp2 = buffer[2*block1_tot*block2_tot];

    return;
  }


  else {
    printf("Error copying first conserved variable block into CPU buffer. Unsupported grid dimensions.\n");
    printf("nx: %d  nx_s: %d  ny: %d  ny_s: %d  nz: %d  nz_s: %d.\n", nx, nx_s, ny, ny_s, nz, nz_s);
    exit(0);
  }

}





// copy the next conserved variable blocks into the remaining buffers
void host_copy_next_3D(int nx, int ny, int nz, int nx_s, int ny_s, int nz_s, int n_ghost, int block, int block1_tot, int block2_tot, int block3_tot, int remainder1, int remainder2, int remainder3, int BLOCK_VOL, Real *host_conserved, Real **buffer, Real **tmp1, int n_fields) {
  
  int n_cells = nx*ny*nz;
  int block1, block2, block3;
  int x_offset, y_offset, z_offset;
  int x_host, y_host, z_host;
  int buf_offset;

  // if no subgrid blocks, do nothing
  if (nx_s == nx && ny_s == ny && nz_s == nz) return;


  // splitting only in x
  else if (nx_s < nx && ny_s == ny && nz_s == nz) {

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

      // copy the next block into a buffer
      for (int k=0; k<nz_s; k++) {
        for (int j=0; j<ny_s; j++) {
          for (int ii=0; ii<n_fields; ii++) {
            memcpy(&buffer[0][ii*BLOCK_VOL + j*nx_s + k*nx_s*ny_s], &host_conserved[x_host + ii*n_cells + j*nx + k*nx*ny], nx_s*sizeof(Real)); 
          }
        }
      }
    }

    return;

  }

  // splitting only in y
  else if (nx_s == nx && ny_s < ny && nz_s == nz) {

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
      y_host = block2*nx_s*(ny_s-2*n_ghost) - nx*y_offset;

      // copy the next block into a buffer
      for (int k=0; k<nz_s; k++) {
        for (int ii=0; ii<n_fields; ii++) {
          memcpy(&buffer[0][ii*BLOCK_VOL + k*nx_s*ny_s], &host_conserved[y_host + ii*n_cells + k*nx*ny], nx_s*sizeof(Real)); 
        }
      }
    }

    return;

  }


  // splitting only in z
  else if (nx_s == nx && ny_s == ny && nz_s < nz) {

    block3 = block+1; // zid of next block
  
    // don't copy if we are currently on the last block
    if (block3 < block3_tot) {

      // if we are on the second-to-last block, make sure the last block doesn't go past 
      // the bounds of the host array
      z_offset = 0;
      if (block3 == block3_tot-1 && remainder3 != 0) {
        z_offset = nz_s - 2*n_ghost - remainder3;
      }

      // calculate the z location in the host array to copy from
      z_host = block3*nx*ny*(nz_s-2*n_ghost) - nx*ny*(z_offset);

      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&buffer[0][ii*BLOCK_VOL], &host_conserved[z_host + ii*n_cells], BLOCK_VOL*sizeof(Real));
      }

    }

    return;
  }


  // splitting in y and z
  else if (nx_s == nx && ny_s < ny && nz_s < nz) {

    block3 = block / block2_tot; // zid of current block
    block2 = block - block3*block2_tot; // yid of current block

    // if we are about to copy the last z block, make sure it doesn't go past 
    // the bounds of the host array
    z_offset = 0;
    if (block3 == block3_tot-2 && remainder3 != 0) {
      z_offset = nz_s - 2*n_ghost - remainder3;
    }


    // if we are at the start of a y row, and not on the last z column, 
    // copy the next y row into buffers
    // so we don't overwrite anything when we return the data to the CPU
    if (block2 == 0 && block3 < block3_tot - 1) {

      // even z column, copy y row into buffers n:2(n-1)
      if (block3%2 == 0) buf_offset = block2_tot;
      // odd  z column, copy y row into buffers 0:n-1
      if (block3%2 == 1) buf_offset = 0;

      for (int n=0; n<block2_tot; n++) {

        // if we are about to copy the last y block, make sure it doesn't go past 
        // the bounds of the host array
        y_offset = 0;
        if (n == block2_tot-1 && remainder2 != 0) {
          y_offset = ny_s - 2*n_ghost - remainder2;
        }

        // calculate the y & z locations in the host array to copy from
        z_host = (block3+1)*nx*ny*(nz_s-2*n_ghost) - nx*ny*(z_offset);
        y_host = (block2+n)*nx*(ny_s-2*n_ghost) - nx*y_offset;

        for (int k=0; k<nz_s; k++) {
          for (int ii=0; ii<n_fields; ii++) {
            memcpy(&buffer[n+buf_offset][ii*BLOCK_VOL + k*nx_s*ny_s], &host_conserved[z_host + y_host + ii*n_cells + k*nx*ny], nx_s*ny_s*sizeof(Real)); 
          }
        }
      }
    }

    // unless we are currently on the last block,
    // set tmp1 to the address of the buffer containing the next block to be calculated
    if (block+1 < block2_tot*block3_tot) {

      block3 = (block+1) / block2_tot; // zid of next block

      // even z column, use pointers from buffers 0:n-1
      if (block3%2 == 0) buf_offset = 0;
      // odd  z column, use pointers from buffers n:2(n-1)
      if (block3%2 == 1) buf_offset = block2_tot;

      // point to the next buffer
      *tmp1 = buffer[(block+1)%(block2_tot) + buf_offset];
    }

      return;
  }

  // splitting in x, y, and z
  else if (nx_s < nx && ny_s < ny && nz_s < nz) {

    block3 = block / (block2_tot*block1_tot); // zid of current block
    block2 = (block - block3*block2_tot*block1_tot) / block1_tot; // yid of current block
    block1 = block - block3*block2_tot*block1_tot - block2*block1_tot; // xid of current block

    // if we are about to copy the last z slab, make sure it doesn't go past 
    // the bounds of the host array
    z_offset = 0;
    if (block3 == block3_tot-2 && remainder3 != 0) {
      z_offset = nz_s - 2*n_ghost - remainder3;
    }


    // if we are at the start of an xy slice, and not on the last z slab, 
    // copy the next xy slice into buffers
    // so we don't overwrite anything when we return the data to the CPU
    if (block1 == 0 && block2 == 0 && block3 < block3_tot - 1) {

      // even z slab, copy xy slice into buffers m*n:2*m*n-1
      if (block3%2 == 0) buf_offset = block1_tot*block2_tot;
      // odd  z slab, copy xy slice into buffers 0:m*n-1
      if (block3%2 == 1) buf_offset = 0;

      for (int n=0; n<block2_tot; n++) {
        for (int m=0; m<block1_tot; m++) {

          // if we are about to copy the last x block, make sure it doesn't go past 
          // the bounds of the host array
          x_offset = 0;
          if (m == block1_tot-1 && remainder1 != 0) {
            x_offset = nx_s - 2*n_ghost - remainder1;
          }

          // if we are about to copy the last y block, make sure it doesn't go past 
          // the bounds of the host array
          y_offset = 0;
          if (n == block2_tot-1 && remainder2 != 0) {
            y_offset = ny_s - 2*n_ghost - remainder2;
          }

          // calculate the x, y & z locations in the host array to copy from
          x_host = (block1+m)*(nx_s-2*n_ghost) - x_offset;
          y_host = (block2+n)*nx*(ny_s-2*n_ghost) - nx*y_offset;
          z_host = (block3+1)*nx*ny*(nz_s-2*n_ghost) - nx*ny*(z_offset);

          for (int k=0; k<nz_s; k++) {
            for (int j=0; j<ny_s; j++) {
              for (int ii=0; ii<n_fields; ii++) {
              memcpy(&buffer[m+n*block1_tot+buf_offset][ii*BLOCK_VOL + j*nx_s + k*nx_s*ny_s], &host_conserved[x_host + y_host + z_host + ii*n_cells + j*nx + k*nx*ny], nx_s*sizeof(Real)); 
              }
            }
          }
        }
      }
    }

    // unless we are currently on the last block,
    // set tmp1 to the address of the buffer containing the next block to be calculated
    if (block+1 < block1_tot*block2_tot*block3_tot) {

      block3 = (block+1) / (block2_tot*block1_tot); // zid of next block

      // even z column, use pointers from buffers 0 : m*n-1
      if (block3%2 == 0) buf_offset = 0;
      // odd  z column, use pointers from buffers m*n : 2*m*n-1
      if (block3%2 == 1) buf_offset = block1_tot*block2_tot;

      // point to the next buffer
      *tmp1 = buffer[(block+1)%(block1_tot*block2_tot) + buf_offset];
      }

      return;
  }

  else {
    printf("Error copying next blocks. Unsupported grid dimensions.\n");
    printf("nx: %d  nx_s: %d  ny: %d  ny_s: %d  nz: %d  nz_s: %d.\n", nx, nx_s, ny, ny_s, nz, nz_s);
    exit(0);
  }


}






// return the values from buffer to the host_conserved array
void host_return_values_3D(int nx, int ny, int nz, int nx_s, int ny_s, int nz_s, int n_ghost, int block, int block1_tot, int block2_tot, int block3_tot, int remainder1, int remainder2, int remainder3, int BLOCK_VOL, Real *host_conserved, Real **buffer, int n_fields) {

  int n_cells = nx*ny*nz;
  int block1, block2, block3;
  int x_offset, y_offset, z_offset;
  int x_host, y_host, z_host, x_gpu, y_gpu, z_gpu, host_loc, gpu_loc;
  int length, hid, gid, n;

  // if no subgrid blocks, do nothing
  if (nx_s == nx && ny_s == ny && nz_s == nz) return;

  // splitting only in x
  else if (nx_s < nx && ny_s == ny && nz_s == nz) {

    // return values based on current block id
    block1 = block;

    // if we just did the last slice, make sure to copy the cells to the right place
    x_offset = 0;
    if (block1 == block1_tot-1 && remainder1 != 0) {
      x_offset = nx_s - 2*n_ghost - remainder1;
    }

    x_host = block1*(nx_s-2*n_ghost) + (n_ghost-x_offset);
    y_host = n_ghost*nx;
    z_host = n_ghost*nx*ny;
    host_loc = x_host + y_host + z_host;
    x_gpu = n_ghost;
    y_gpu = n_ghost*nx_s;
    z_gpu = n_ghost*nx_s*ny_s;
    gpu_loc = x_gpu + y_gpu + z_gpu;
    length = (nx_s-2*n_ghost); // number of cells to copy back

    for (int k=0; k<nz_s-2*n_ghost; k++) {
      for (int j=0; j<ny_s-2*n_ghost; j++) {
        hid = j*nx + k*nx*ny;
        gid = j*nx_s + k*nx_s*ny_s;
        for (int ii=0; ii<n_fields; ii++) {
          memcpy(&host_conserved[host_loc + hid + ii*n_cells], &buffer[1][gpu_loc + gid + ii*BLOCK_VOL], length*sizeof(Real));
        }
      }
    }

    return;
  }

  // splitting only in y
  else if (nx_s == nx && ny_s < ny && nz_s == nz) {

    // return values based on current block id
    block2 = block;

    // if we just did the last slice, make sure to copy the cells to the right place
    y_offset = 0;
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }

    y_host = block2*nx*(ny_s-2*n_ghost) + nx*(n_ghost-y_offset);
    z_host = n_ghost*nx*ny;
    host_loc = y_host + z_host;
    y_gpu = n_ghost*nx_s;
    z_gpu = n_ghost*nx_s*ny_s;
    gpu_loc = y_gpu + z_gpu;
    length = nx_s*(ny_s-2*n_ghost); // number of cells to copy back

    for (int k=0; k<nz_s-2*n_ghost; k++) {
      hid = k*nx*ny;
      gid = k*nx_s*ny_s;
      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&host_conserved[host_loc + hid + ii*n_cells], &buffer[1][gpu_loc + gid + ii*BLOCK_VOL], length*sizeof(Real));
      }
    }

    return;
  }

  // splitting only in z
  else if (nx_s == nx && ny_s == ny && nz_s < nz) {

    // return values based on current block id
    block3 = block;

    z_offset = 0;
    // if we just did the last slice, make sure to copy the cells to the right place
    if (block3 == block3_tot-1 && remainder3 != 0) {
      z_offset = nz_s - 2*n_ghost - remainder3;
    }

    z_host = block3*nx*ny*(nz_s-2*n_ghost) + nx*ny*(n_ghost-z_offset);
    z_gpu = n_ghost*nx_s*ny_s;
    length = nx_s*ny_s*(nz_s-2*n_ghost); // number of cells to copy back

    for (int ii=0; ii<n_fields; ii++) {
      memcpy(&host_conserved[z_host + ii*n_cells], &buffer[1][z_gpu + ii*BLOCK_VOL], length*sizeof(Real));
    }

    return;
  }


  // splitting in y and z
  else if (nx_s == nx && ny_s < ny && nz_s < nz) {

    // buffer holding the returned values (always the last buffer)
    n = 2*block2_tot;

    // return values based on current block id
    block3 = block / block2_tot; // zid of current block
    block2 = block - block3*block2_tot; // yid of current block

    z_offset = 0;
    // if we just did the z last slice, make sure to copy the cells to the right place
    if (block3 == block3_tot-1 && remainder3 != 0) {
      z_offset = nz_s - 2*n_ghost - remainder3;
    }

    y_offset = 0;
    // if we just did the y last slice, make sure to copy the cells to the right place
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }

    z_host = block3*nx*ny*(nz_s-2*n_ghost) + nx*ny*(n_ghost-z_offset);
    y_host = block2*nx*(ny_s-2*n_ghost) + nx*(n_ghost-y_offset);
    host_loc = y_host + z_host;
    z_gpu  = n_ghost*nx_s*ny_s;
    y_gpu  = n_ghost*nx_s;
    gpu_loc = y_gpu + z_gpu;
    length = nx_s*(ny_s-2*n_ghost); // number of cells to copy back
    
    for (int k=0; k<nz_s-2*n_ghost; k++) {
      hid = k*nx*ny;
      gid = k*nx_s*ny_s;
      for (int ii=0; ii<n_fields; ii++) {
        memcpy(&host_conserved[host_loc + hid + ii*n_cells], &buffer[n][gpu_loc + gid + ii*BLOCK_VOL], length*sizeof(Real));
      }
    }

    return;
  }

  // splitting in x, y, and z
  else if (nx_s < nx && ny_s < ny && nz_s < nz) {

    // buffer holding the returned values (always the last buffer)
    n = 2*block1_tot*block2_tot;

    // return values based on current block id
    block3 = block / (block2_tot*block1_tot); // zid of current block
    block2 = (block - block3*block2_tot*block1_tot) / block1_tot; // yid of current block
    block1 = block - block3*block2_tot*block1_tot - block2*block1_tot; // xid of current block

    z_offset = 0;
    // if we just did the z last slice, make sure to copy the cells to the right place
    if (block3 == block3_tot-1 && remainder3 != 0) {
      z_offset = nz_s - 2*n_ghost - remainder3;
    }

    y_offset = 0;
    // if we just did the y last slice, make sure to copy the cells to the right place
    if (block2 == block2_tot-1 && remainder2 != 0) {
      y_offset = ny_s - 2*n_ghost - remainder2;
    }

    x_offset = 0;
    // if we just did the x last slice, make sure to copy the cells to the right place
    if (block1 == block1_tot-1 && remainder1 != 0) {
      x_offset = nx_s - 2*n_ghost - remainder1;
    }

    z_host = block3*nx*ny*(nz_s-2*n_ghost) + nx*ny*(n_ghost-z_offset);
    y_host = block2*nx*(ny_s-2*n_ghost) + nx*(n_ghost-y_offset);
    x_host = block1*(nx_s-2*n_ghost) + (n_ghost-x_offset);
    host_loc = x_host + y_host + z_host;
    z_gpu = n_ghost*nx_s*ny_s;
    y_gpu = n_ghost*nx_s;
    x_gpu = n_ghost;
    gpu_loc = x_gpu + y_gpu + z_gpu;
    length = (nx_s-2*n_ghost); // number of cells to copy back
    
    for (int k=0; k<nz_s-2*n_ghost; k++) {
      for (int j=0; j<ny_s-2*n_ghost; j++) {
        hid = j*nx + k*nx*ny;
        gid = j*nx_s + k*nx_s*ny_s;
        for (int ii=0; ii<n_fields; ii++) {
          memcpy(&host_conserved[host_loc + hid + ii*n_cells], &buffer[n][gpu_loc + gid + ii*BLOCK_VOL], length*sizeof(Real));
        }
      }
    }

    return;
  }


  else {
    printf("Error returning values to host. Unsupported grid dimensions.\n");
    printf("nx: %d  nx_s: %d  ny: %d  ny_s: %d  nz: %d  nz_s: %d.\n", nx, nx_s, ny, ny_s, nz, nz_s);
    exit(0);
  }


}






void free_buffers_3D(int nx, int ny, int nz, int nx_s, int ny_s, int nz_s, int block1_tot, int block2_tot, int block3_tot, Real **buffer) {

  int n;

  if (nx_s == nx && ny_s == ny && nz_s == nz) return;

  else if (nx_s < nx && ny_s == ny && nz_s == nz) n = 2; 

  else if (nx_s == nx && ny_s < ny && nz_s == nz) n = 2; 

  else if (nx_s == nx && ny_s == ny && nz_s < nz) n = 2; 

  else if (nx_s == nx && ny_s < ny && nz_s < nz) n = 2*block2_tot + 1;

  else if (nx_s < nx && ny_s < ny && nz_s < nz) n = 2*block1_tot*block2_tot + 1;

  else {
    printf("Unsupported grid dimensions.\n");
    exit(0);
  }

  for (int i=0; i<n; i++) {
    free(buffer[i]);
  }

  free(buffer);

}

#endif //CUDA

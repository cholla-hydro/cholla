/*! \file hydro_cuda.cu
 *  \brief Definitions of functions used in all cuda integration algorithms. */
#ifdef CUDA

#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"


__global__ void Update_Conserved_Variables_2D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, Real gamma)
{
  int id, xid, yid, n_cells;
  int imo, jmo;

  #ifdef DE
  Real d, d_inv, vx, vy, vz, P;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo;
  int ipo, jpo;
  #endif

  Real dtodx = dt/dx;
  Real dtody = dt/dy;

  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;
  imo = xid-1 + yid*nx;
  jmo = xid + (yid-1)*nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost)
  {
    #ifdef DE
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    ipo = xid+1 + yid*nx;
    jpo = xid + (yid+1)*nx;
    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo]; 
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo]; 
    vy_jmo = dev_conserved[2*n_cells + jmo] / dev_conserved[jmo]; 
    vy_jpo = dev_conserved[2*n_cells + jpo] / dev_conserved[jpo]; 
    #endif
    
    // update the conserved variable array
    dev_conserved[            id] += dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                  +  dtody * (dev_F_y[            jmo] - dev_F_y[            id]);
    dev_conserved[  n_cells + id] += dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id]) 
                                  +  dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id]);
    dev_conserved[2*n_cells + id] += dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id]) 
                                  +  dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id]); 
    dev_conserved[3*n_cells + id] += dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                  +  dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id]);
    dev_conserved[4*n_cells + id] += dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                  +  dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id]);
    #ifdef DE
    dev_conserved[5*n_cells + id] += dtodx * (dev_F_x[5*n_cells + imo] - dev_F_x[5*n_cells + id])
                                  +  dtody * (dev_F_y[5*n_cells + jmo] - dev_F_y[5*n_cells + id])
                                  +  0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo));
    #endif
    if (dev_conserved[id] < 0.0 || dev_conserved[id] != dev_conserved[id]) {
      printf("%3d %3d Thread crashed in final update. %f %f %f %f %f\n", xid, yid, d, dtodx*(dev_F_x[imo]-dev_F_x[id]), dev_F_y[jmo],dev_F_y[id], dev_conserved[id]);
    }   
    /*
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    if (P < 0.0)
      printf("%3d %3d Negative pressure after final update. %f %f %f %f\n", xid, yid, dev_conserved[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, P);    
    */
  }

}


__global__ void Sync_Energies_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real gamma)
{
  int id, xid, yid, n_cells;
  Real d, d_inv, vx, vy, vz, P, E;
  Real ge1, ge2, Emax;
  int imo, ipo, jmo, jpo;
  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;

  imo = max(xid-1, n_ghost);
  imo = imo + yid*nx;
  ipo = min(xid+1, nx-n_ghost-1);
  ipo = ipo + yid*nx;
  jmo = max(yid-1, n_ghost);
  jmo = xid + jmo*nx;
  jpo = min(yid+1, ny-n_ghost-1);
  jpo = xid + jpo*nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    E  =  dev_conserved[4*n_cells + id];
    P  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    // separately tracked internal energy 
    ge1 =  dev_conserved[5*n_cells + id];
    // internal energy calculated from total energy
    ge2 = dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    // if the ratio of conservatively calculated internal energy to total energy
    // is greater than 1/1000, use the conservatively calculated internal energy
    // to do the internal energy update
    if (ge2/E > 0.001) {
      dev_conserved[5*n_cells + id] = ge2;
      ge1 = ge2;
    }     
    //find the max nearby total energy 
    Emax = fmax(dev_conserved[4*n_cells + imo], E);
    Emax = fmax(Emax, dev_conserved[4*n_cells + ipo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + jmo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + jpo]);
    // if the ratio of conservatively calculated internal energy to max nearby total energy
    // is greater than 1/10, continue to use the conservatively calculated internal energy 
    if (ge2/Emax > 0.1) {
      dev_conserved[5*n_cells + id] = ge2;
    }
    // sync the total energy with the internal energy 
    else {
      dev_conserved[4*n_cells + id] += ge1 - ge2;
    }
    // recalculate the pressure 
    P = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);    
    if (P < 0.0) printf("%d Negative pressure after internal energy sync. %f %f \n", id, ge1, ge2);    
  }
}


__global__ void Calc_dt_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real dx, Real dy, Real *dti_array, Real gamma)
{
  __shared__ Real max_dti[TPB];

  Real d, d_inv, vx, vy, vz, P, cs;
  int id, tid, xid, yid, n_cells;
  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;
  // and a thread id within the block
  tid = threadIdx.x;

  // set shared memory to 0
  max_dti[tid] = 0;
  __syncthreads();

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    P  = fmax(P, (Real) 1.0e-20);
    // find the max wavespeed in that cell, use it to calculate the inverse timestep
    cs = sqrt(d_inv * gamma * P);
    max_dti[tid] = fmax((fabs(vx)+cs)/dx, (fabs(vy)+cs)/dy);
  }
  __syncthreads();
  
  // do the reduction in shared memory (find the max inverse timestep in the block)
  for (unsigned int s=1; s<blockDim.x; s*=2) {
    if (tid % (2*s) == 0) {
      max_dti[tid] = fmax(max_dti[tid], max_dti[tid + s]);
    }
    __syncthreads();
  }

  // write the result for this block to global memory
  if (tid == 0) dti_array[blockId] = max_dti[0];

}



#endif //CUDA

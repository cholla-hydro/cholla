/*! \file hydro_cuda.cu
 *  \brief Definitions of functions used in all cuda integration algorithms. */
#ifdef CUDA

#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"


__global__ void Update_Conserved_Variables_1D(Real *dev_conserved, Real *dev_F, int n_cells, int x_off, int n_ghost, Real dx, Real dt, Real gamma)
{
  int id;
  #ifdef DE
  Real d, d_inv, vx, vy, vz, P;  
  Real vx_imo, vx_ipo;
  #endif

  #ifdef GRAVITY
  Real gx, d_n, d_inv_n, vx_n;
  gx = 0.0;
  #endif
  

  Real dtodx = dt/dx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;


  // threads corresponding to real cells do the calculation
  if (id > n_ghost - 1 && id < n_cells-n_ghost)
  {
    #ifdef DE
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    vx_imo = dev_conserved[1*n_cells + id-1]/dev_conserved[id-1];
    vx_ipo = dev_conserved[1*n_cells + id+1]/dev_conserved[id+1];
    #endif
  
    // update the conserved variable array
    dev_conserved[            id] += dtodx * (dev_F[            id-1] - dev_F[            id]);
    dev_conserved[  n_cells + id] += dtodx * (dev_F[  n_cells + id-1] - dev_F[  n_cells + id]);
    dev_conserved[2*n_cells + id] += dtodx * (dev_F[2*n_cells + id-1] - dev_F[2*n_cells + id]);
    dev_conserved[3*n_cells + id] += dtodx * (dev_F[3*n_cells + id-1] - dev_F[3*n_cells + id]);
    dev_conserved[4*n_cells + id] += dtodx * (dev_F[4*n_cells + id-1] - dev_F[4*n_cells + id]);
    #ifdef DE
    dev_conserved[5*n_cells + id] += dtodx * (dev_F[5*n_cells + id-1] - dev_F[5*n_cells + id])
                                  +  dtodx * P * 0.5 * (vx_imo - vx_ipo);
    #endif
    if (dev_conserved[id] != dev_conserved[id]) printf("%3d Thread crashed in final update.\n", id);
    /*
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    if (P < 0.0) printf("%d Negative pressure after final update.\n", id);
    */
  }


}


__global__ void Update_Conserved_Variables_1D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F, int n_cells, int n_ghost, Real dx, Real dt, Real gamma)
{
  int id;
  Real dtodx = dt/dx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;

  // threads corresponding all cells except outer ring of ghost cells do the calculation
  if (id > 0 && id < n_cells-1)
  {
    // update the conserved variable array
    dev_conserved_half[            id] = dev_conserved[            id] + dtodx * (dev_F[            id-1] - dev_F[            id]);
    dev_conserved_half[  n_cells + id] = dev_conserved[  n_cells + id] + dtodx * (dev_F[  n_cells + id-1] - dev_F[  n_cells + id]);
    dev_conserved_half[2*n_cells + id] = dev_conserved[2*n_cells + id] + dtodx * (dev_F[2*n_cells + id-1] - dev_F[2*n_cells + id]);
    dev_conserved_half[3*n_cells + id] = dev_conserved[3*n_cells + id] + dtodx * (dev_F[3*n_cells + id-1] - dev_F[3*n_cells + id]);
    dev_conserved_half[4*n_cells + id] = dev_conserved[4*n_cells + id] + dtodx * (dev_F[4*n_cells + id-1] - dev_F[4*n_cells + id]);
  }


}


__global__ void Update_Conserved_Variables_2D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real dt, Real gamma)
{
  int id, xid, yid, n_cells;
  int imo, jmo;

  #if defined (DE) || defined(GRAVITY)
  Real d, d_inv, vx, vy;
  #endif
  #ifdef DE
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz, P;
  int ipo, jpo;
  #endif

  #ifdef GRAVITY
  Real gx, gy, d_n, d_inv_n, vx_n, vy_n;
  gx = 0.0;
  gy = 0.0;
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
    #if defined (DE) || defined (GRAVITY)
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    #endif //GRAVITY
    #ifdef DE
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
    #ifdef GRAVITY // add gravitational source terms, time averaged from n to n+1
    d_n  =  dev_conserved[            id];
    d_inv_n = 1.0 / d_n;
    vx_n =  dev_conserved[1*n_cells + id] * d_inv_n;
    vy_n =  dev_conserved[2*n_cells + id] * d_inv_n;
    dev_conserved[  n_cells + id] += 0.5*dt*gx*(d + d_n);
    dev_conserved[2*n_cells + id] += 0.5*dt*gy*(d + d_n);
    dev_conserved[4*n_cells + id] += 0.25*dt*gx*(d + d_n)*(vx + vx_n)
                                  +  0.25*dt*gy*(d + d_n)*(vy + vy_n);
    #endif
    #ifdef DE
    dev_conserved[5*n_cells + id] += dtodx * (dev_F_x[5*n_cells + imo] - dev_F_x[5*n_cells + id])
                                  +  dtody * (dev_F_y[5*n_cells + jmo] - dev_F_y[5*n_cells + id])
                                  +  0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo));
    //if (dev_conserved[5*n_cells + id] < 0.0) printf("%3d %3d Negative internal energy after final update.\n", xid, yid);
    #endif
    if (dev_conserved[id] < 0.0 || dev_conserved[id] != dev_conserved[id]) {
      printf("%3d %3d Thread crashed in final update. %f %f %f\n", xid, yid, dtodx*(dev_F_x[imo]-dev_F_x[id]), dtody*(dev_F_y[jmo]-dev_F_y[id]), dev_conserved[id]);
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


__global__ void Update_Conserved_Variables_2D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, Real gamma)
{
  int id, xid, yid, n_cells;
  int imo, jmo;

  Real dtodx = dt/dx;
  Real dtody = dt/dy;

  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;


  // all threads but one outer ring of ghost cells 
  if (xid > 0 && xid < nx-1 && yid > 0 && yid < ny-1)
  {
    // update the conserved variable array
    imo = xid-1 + yid*nx;
    jmo = xid + (yid-1)*nx;
    dev_conserved_half[            id] = dev_conserved[            id] 
                                       + dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                       + dtody * (dev_F_y[            jmo] - dev_F_y[            id]);
    dev_conserved_half[  n_cells + id] = dev_conserved[  n_cells + id] 
                                       + dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id]) 
                                       + dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id]);
    dev_conserved_half[2*n_cells + id] = dev_conserved[2*n_cells + id] 
                                       + dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id]) 
                                       + dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id]); 
    dev_conserved_half[3*n_cells + id] = dev_conserved[3*n_cells + id] 
                                       + dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                       + dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id]);
    dev_conserved_half[4*n_cells + id] = dev_conserved[4*n_cells + id] 
                                       + dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                       + dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id]);
  } 
}



__global__ void Update_Conserved_Variables_3D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real dt,
                                              Real gamma)
{
  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;
  #ifdef DE
  Real d, d_inv, vx, vy, vz, P;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo;
  int ipo, jpo, kpo;
  #endif

  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  Real dtodz = dt/dz;
  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;
  imo = xid-1 + yid*nx + zid*nx*ny;
  jmo = xid + (yid-1)*nx + zid*nx*ny;
  kmo = xid + yid*nx + (zid-1)*nx*ny;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    #ifdef DE
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    //if (d < 0.0 || d != d) printf("Negative density before final update.\n");
    //if (P < 0.0) printf("%d Negative pressure before final update.\n", id);
    ipo = xid+1 + yid*nx + zid*nx*ny;
    jpo = xid + (yid+1)*nx + zid*nx*ny;
    kpo = xid + yid*nx + (zid+1)*nx*ny;
    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo]; 
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo]; 
    vy_jmo = dev_conserved[2*n_cells + jmo] / dev_conserved[jmo]; 
    vy_jpo = dev_conserved[2*n_cells + jpo] / dev_conserved[jpo]; 
    vz_kmo = dev_conserved[3*n_cells + kmo] / dev_conserved[kmo]; 
    vz_kpo = dev_conserved[3*n_cells + kpo] / dev_conserved[kpo]; 
    #endif

    // update the conserved variable array
    dev_conserved[            id] += dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                  +  dtody * (dev_F_y[            jmo] - dev_F_y[            id])
                                  +  dtodz * (dev_F_z[            kmo] - dev_F_z[            id]);
    dev_conserved[  n_cells + id] += dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id])
                                  +  dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id])
                                  +  dtodz * (dev_F_z[  n_cells + kmo] - dev_F_z[  n_cells + id]);
    dev_conserved[2*n_cells + id] += dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id])
                                  +  dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id])
                                  +  dtodz * (dev_F_z[2*n_cells + kmo] - dev_F_z[2*n_cells + id]);
    dev_conserved[3*n_cells + id] += dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                  +  dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id])
                                  +  dtodz * (dev_F_z[3*n_cells + kmo] - dev_F_z[3*n_cells + id]);
    dev_conserved[4*n_cells + id] += dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                  +  dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id])
                                  +  dtodz * (dev_F_z[4*n_cells + kmo] - dev_F_z[4*n_cells + id]);
    #ifdef DE
    dev_conserved[5*n_cells + id] += dtodx * (dev_F_x[5*n_cells + imo] - dev_F_x[5*n_cells + id])
                                  +  dtody * (dev_F_y[5*n_cells + jmo] - dev_F_y[5*n_cells + id])
                                  +  dtodz * (dev_F_z[5*n_cells + kmo] - dev_F_z[5*n_cells + id])
                                  +  0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));
    #endif
    if (dev_conserved[id] < 0.0 || dev_conserved[id] != dev_conserved[id]) {
      printf("%3d %3d %3d Thread crashed in final update. %f %f %f %f\n", xid, yid, zid, dtodx*(dev_F_x[imo]-dev_F_x[id]), dtody*(dev_F_y[jmo]-dev_F_y[id]), dtodz*(dev_F_z[kmo]-dev_F_z[id]), dev_conserved[id]);
    }
    // every thread collects the conserved variables it needs from global memory
    /*
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    if (P < 0.0) printf("%3d %3d %3d Negative pressure after final update. %f %f %f %f %f\n", xid, yid, zid, dev_conserved[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz, P);
    */
  }

}



__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma)
{

  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;

  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  Real dtodz = dt/dz;
  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;


  // threads corresponding to all cells except outer ring of ghost cells do the calculation
  if (xid > 0 && xid < nx-1 && yid > 0 && yid < ny-1 && zid > 0 && zid < nz-1)
  {
    // update the conserved variable array
    imo = xid-1 + yid*nx + zid*nx*ny;
    jmo = xid + (yid-1)*nx + zid*nx*ny;
    kmo = xid + yid*nx + (zid-1)*nx*ny;
    dev_conserved_half[            id] = dev_conserved[            id]
                                       + dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                       + dtody * (dev_F_y[            jmo] - dev_F_y[            id])
                                       + dtodz * (dev_F_z[            kmo] - dev_F_z[            id]);
    dev_conserved_half[  n_cells + id] = dev_conserved[  n_cells + id] 
                                       + dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id])
                                       + dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id])
                                       + dtodz * (dev_F_z[  n_cells + kmo] - dev_F_z[  n_cells + id]);
    dev_conserved_half[2*n_cells + id] = dev_conserved[2*n_cells + id] 
                                       + dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id])
                                       + dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id])
                                       + dtodz * (dev_F_z[2*n_cells + kmo] - dev_F_z[2*n_cells + id]);
    dev_conserved_half[3*n_cells + id] = dev_conserved[3*n_cells + id] 
                                       + dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                       + dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id])
                                       + dtodz * (dev_F_z[3*n_cells + kmo] - dev_F_z[3*n_cells + id]);
    dev_conserved_half[4*n_cells + id] = dev_conserved[4*n_cells + id] 
                                       + dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                       + dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id])
                                       + dtodz * (dev_F_z[4*n_cells + kmo] - dev_F_z[4*n_cells + id]);
    if (dev_conserved_half[id] < 0.0 || dev_conserved_half[id] != dev_conserved_half[id]) {
      printf("%3d %3d %3d Thread crashed in half step update.\n", xid, yid, zid);
    }    


  }

}



__global__ void Sync_Energies_1D(Real *dev_conserved, int n_cells, int n_ghost, Real gamma)
{
  int id;
  Real d, d_inv, vx, vy, vz, P, E;
  Real ge1, ge2, Emax;
  int im1, ip1;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  
  im1 = max(id-1, n_ghost);
  ip1 = min(id+1, n_cells-n_ghost-1);

  // threads corresponding to real cells do the calculation
  if (id > n_ghost - 1 && id < n_cells-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    E  =  dev_conserved[4*n_cells + id];
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    // separately tracked internal energy 
    ge1 = dev_conserved[5*n_cells + id];
    // internal energy calculated from total energy
    ge2 = dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    // if the ratio of conservatively calculated internal energy to total energy
    // is greater than 1/1000, use the conservatively calculated internal energy
    // to do the internal energy update
    if (ge2/E > 0.001) {
      dev_conserved[5*n_cells + id] = ge2;
      ge1 = ge2;
    }     
    // find the max nearby total energy 
    Emax = fmax(dev_conserved[4*n_cells + im1], E);
    Emax = fmax(dev_conserved[4*n_cells + ip1], Emax);
    // if the ratio of conservatively calculated internal energy to max nearby total energy
    // is greater than 1/10, continue to use the conservatively calculated internal energy 
    if (ge2/Emax > 0.1) {
      dev_conserved[5*n_cells + id] = ge2;
    }
    // sync the total energy with the internal energy 
    else {
      dev_conserved[4*n_cells + id] += ge1 - ge2;
    }
    /*
    // if the conservatively calculated internal energy is greater than the estimate of the truncation error,
    // use the internal energy computed from the total energy to do the update
    //find the max nearby velocity difference (estimate of truncation error) 
    vmax = fmax(fabs(vx-dev_conserved[1*n_cells + im1]/dev_conserved[im1]), fabs(dev_conserved[1*n_cells + ip1]/dev_conserved[ip1]-vx));
    //printf("%3d %f %f %f %f\n", id, ge1, ge2, vmax, 0.25*d*vmax*vmax);
    if (ge2 > 0.25*d*vmax*vmax) {
      dev_conserved[5*n_cells + id] = ge2;
      ge1 = ge2;
    }
    //else printf("%d Using ge1 %f %f %f %f\n", id, ge1, ge2, vmax, 0.25*d*vmax*vmax);
    */
    // update the total energy
     
    // recalculate the pressure 
    P = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);    
    if (P < 0.0) printf("%d Negative pressure after internal energy sync. %f %f \n", id, ge1, ge2);    
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




__global__ void Sync_Energies_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real gamma)
{
  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, P, E;
  Real ge1, ge2, Emax;
  int imo, ipo, jmo, jpo, kmo, kpo;
  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;

  imo = max(xid-1, n_ghost);
  imo = imo + yid*nx + zid*nx*ny;
  ipo = min(xid+1, nx-n_ghost-1);
  ipo = ipo + yid*nx + zid*nx*ny;
  jmo = max(yid-1, n_ghost);
  jmo = xid + jmo*nx + zid*nx*ny;
  jpo = min(yid+1, ny-n_ghost-1);
  jpo = xid + jpo*nx + zid*nx*ny;
  kmo = max(zid-1, n_ghost);
  kmo = xid + yid*nx + kmo*nx*ny;
  kpo = min(zid+1, nz-n_ghost-1);
  kpo = xid + yid*nx + kpo*nx*ny;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
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
    Emax = fmax(Emax, dev_conserved[4*n_cells + kmo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + kpo]);
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
    if (P < 0.0) printf("%3d %3d %3d Negative pressure after internal energy sync. %f %f %f\n", xid, yid, zid, P/(gamma-1.0), ge1, ge2);    
  }
}



__global__ void Calc_dt_1D(Real *dev_conserved, int n_cells, int n_ghost, Real dx, Real *dti_array, Real gamma)
{
  __shared__ Real max_dti[TPB];

  Real d, d_inv, vx, vy, vz, P, cs;
  int id, tid;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  // and a thread id within the block
  tid = threadIdx.x;

  // set shared memory to 0
  max_dti[tid] = 0;
  __syncthreads();


  // threads corresponding to real cells do the calculation
  if (id > n_ghost - 1 && id < n_cells-n_ghost)
  {
    // start timestep calculation here
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    P  = fmax(P, (Real) TINY_NUMBER);
    // find the max wavespeed in that cell, use it to calculate the inverse timestep
    cs = sqrt(d_inv * gamma * P);
    max_dti[tid] = (fabs(vx)+cs)/dx;
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
  if (tid == 0) dti_array[blockIdx.x] = max_dti[0];


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


__global__ void Calc_dt_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real *dti_array, Real gamma)
{
  __shared__ Real max_dti[TPB];

  Real d, d_inv, vx, vy, vz, P, cs;
  int id, xid, yid, zid, n_cells;
  int tid;

  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;
  // and a thread id within the block  
  tid = threadIdx.x;

  // set shared memory to 0
  max_dti[tid] = 0;
  __syncthreads();

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    cs = sqrt(d_inv * gamma * P);
    max_dti[tid] = fmax((fabs(vx)+cs)/dx, (fabs(vy)+cs)/dy);
    max_dti[tid] = fmax(max_dti[tid], (fabs(vz)+cs)/dz);
    //max_dti[tid] = (fabs(vx)+cs)/dx + (fabs(vy)+cs)/dy + (fabs(vz)+cs)/dz;
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
  if (tid == 0) dti_array[blockIdx.x] = max_dti[0];

}




#endif //CUDA

/*! \file hydro_cuda.cu
 *  \brief Definitions of functions used in all cuda integration algorithms. */
#ifdef CUDA

#include <stdio.h>
#include <math.h>
#include <float.h>

#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../hydro/hydro_cuda.h"
#include "../gravity/gravity_cuda.h"
#include "../utils/hydro_utilities.h"
#include "../utils/cuda_utilities.h"
#include "../utils/reduction_utilities.h"


__global__ void Update_Conserved_Variables_1D(Real *dev_conserved, Real *dev_F, int n_cells, int x_off, int n_ghost, Real dx, Real xbound, Real dt, Real gamma, int n_fields)
{
  int id;
  #ifdef STATIC_GRAV
  Real d, d_inv, vx;
  Real gx, d_n, d_inv_n, vx_n;
  gx = 0.0;
  #endif

  Real dtodx = dt/dx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;


  // threads corresponding to real cells do the calculation
  if (id > n_ghost - 1 && id < n_cells-n_ghost)
  {
    #ifdef STATIC_GRAV
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    #endif

    // update the conserved variable array
    dev_conserved[            id] += dtodx * (dev_F[            id-1] - dev_F[            id]);
    dev_conserved[  n_cells + id] += dtodx * (dev_F[  n_cells + id-1] - dev_F[  n_cells + id]);
    dev_conserved[2*n_cells + id] += dtodx * (dev_F[2*n_cells + id-1] - dev_F[2*n_cells + id]);
    dev_conserved[3*n_cells + id] += dtodx * (dev_F[3*n_cells + id-1] - dev_F[3*n_cells + id]);
    dev_conserved[4*n_cells + id] += dtodx * (dev_F[4*n_cells + id-1] - dev_F[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved[(5+i)*n_cells + id] += dtodx * (dev_F[(5+i)*n_cells + id-1] - dev_F[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_conserved[(n_fields-1)*n_cells + id] += dtodx * (dev_F[(n_fields-1)*n_cells + id-1] - dev_F[(n_fields-1)*n_cells + id]);
    #endif
    #ifdef STATIC_GRAV // add gravitational source terms, time averaged from n to n+1
    calc_g_1D(id, x_off, n_ghost, dx, xbound, &gx);
    d_n  =  dev_conserved[            id];
    d_inv_n = 1.0 / d_n;
    vx_n =  dev_conserved[1*n_cells + id] * d_inv_n;
    dev_conserved[  n_cells + id] += 0.5*dt*gx*(d + d_n);
    dev_conserved[4*n_cells + id] += 0.25*dt*gx*(d + d_n)*(vx + vx_n);
    #endif
    if (dev_conserved[id] != dev_conserved[id]) printf("%3d Thread crashed in final update. %f\n", id, dev_conserved[id]);
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


__global__ void Update_Conserved_Variables_2D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real dt, Real gamma, int n_fields)
{
  int id, xid, yid, n_cells;
  int imo, jmo;

  #ifdef STATIC_GRAV
  Real d, d_inv, vx, vy;
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
    #ifdef STATIC_GRAV
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
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
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved[(5+i)*n_cells + id] += dtodx * (dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                        +  dtody * (dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_conserved[(n_fields-1)*n_cells + id] += dtodx * (dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id])
                                  +  dtody * (dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id]);
    #endif
    #ifdef STATIC_GRAV
    // calculate the gravitational acceleration as a function of x & y position
    calc_g_2D(xid, yid, x_off, y_off, n_ghost, dx, dy, xbound, ybound, &gx, &gy);
    // add gravitational source terms, time averaged from n to n+1
    d_n  =  dev_conserved[            id];
    d_inv_n = 1.0 / d_n;
    vx_n =  dev_conserved[1*n_cells + id] * d_inv_n;
    vy_n =  dev_conserved[2*n_cells + id] * d_inv_n;
    dev_conserved[  n_cells + id] += 0.5*dt*gx*(d + d_n);
    dev_conserved[2*n_cells + id] += 0.5*dt*gy*(d + d_n);
    dev_conserved[4*n_cells + id] += 0.25*dt*gx*(d + d_n)*(vx + vx_n)
                                  +  0.25*dt*gy*(d + d_n)*(vy + vy_n);
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



__global__ void Update_Conserved_Variables_3D(Real *dev_conserved,
                                              Real *Q_Lx, Real *Q_Rx, Real *Q_Ly, Real *Q_Ry, Real *Q_Lz, Real *Q_Rz,
                                              Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost,
                                              Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt,
                                              Real gamma, int n_fields, Real density_floor, Real *dev_potential )
{
  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;

  #ifdef STATIC_GRAV
  Real d, d_inv, vx, vy, vz;
  Real gx, gy, gz, d_n, d_inv_n, vx_n, vy_n, vz_n;
  gx = 0.0;
  gy = 0.0;
  gz = 0.0;
  #endif

  #ifdef DENSITY_FLOOR
  Real dens_0;
  #endif

  #ifdef GRAVITY
  Real d, d_inv, vx, vy, vz;
  Real gx, gy, gz, d_n, d_inv_n, vx_n, vy_n, vz_n;
  Real pot_l, pot_r;
  int id_l, id_r;
  gx = 0.0;
  gy = 0.0;
  gz = 0.0;

  #ifdef GRAVITY_5_POINTS_GRADIENT
  int id_ll, id_rr;
  Real pot_ll, pot_rr;
  #endif

  #endif //GRAVITY

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
    #if defined(STATIC_GRAV) ||  defined(GRAVITY)
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
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
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved[(5+i)*n_cells + id] += dtodx * (dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                    +  dtody * (dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id])
                                    +  dtodz * (dev_F_z[(5+i)*n_cells + kmo] - dev_F_z[(5+i)*n_cells + id]);
      #ifdef COOLING_GRACKLE
      // If the updated value is negative, then revert to the value before the update
      if ( dev_conserved[(5+i)*n_cells + id] < 0 ){
        dev_conserved[(5+i)*n_cells + id] -= dtodx * (dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                      +  dtody * (dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id])
                                      +  dtodz * (dev_F_z[(5+i)*n_cells + kmo] - dev_F_z[(5+i)*n_cells + id]);
      }
      #endif
    }
    #endif
    #ifdef DE
    dev_conserved[(n_fields-1)*n_cells + id] += dtodx * (dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id])
                                  +  dtody * (dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id])
                                  +  dtodz * (dev_F_z[(n_fields-1)*n_cells + kmo] - dev_F_z[(n_fields-1)*n_cells + id]);
                                  // +  0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));
                                  //Note: this term is added in a separate kernel to avoid synchronization issues
    #endif

    #ifdef DENSITY_FLOOR
    if ( dev_conserved[            id] < density_floor ){
      if (dev_conserved[            id] > 0){
        dens_0 = dev_conserved[            id];
        // Set the density to the density floor
        dev_conserved[            id] = density_floor;
        // Scale the conserved values to the new density
        dev_conserved[1*n_cells + id] *= (density_floor / dens_0);
        dev_conserved[2*n_cells + id] *= (density_floor / dens_0);
        dev_conserved[3*n_cells + id] *= (density_floor / dens_0);
        dev_conserved[4*n_cells + id] *= (density_floor / dens_0);
        #ifdef DE
        dev_conserved[(n_fields-1)*n_cells + id] *= (density_floor / dens_0);
        #endif
      }
      else{
        // If the density is negative: average the density on that cell
        dens_0 = dev_conserved[            id];
        Average_Cell_Single_Field( 0, xid, yid, zid, nx, ny, nz, n_cells, dev_conserved );
      }
    }
    #endif//DENSITY_FLOOR

    #ifdef STATIC_GRAV
    calc_g_3D(xid, yid, zid, x_off, y_off, z_off, n_ghost, dx, dy, dz, xbound, ybound, zbound, &gx, &gy, &gz);
    d_n  =  dev_conserved[            id];
    d_inv_n = 1.0 / d_n;
    vx_n =  dev_conserved[1*n_cells + id] * d_inv_n;
    vy_n =  dev_conserved[2*n_cells + id] * d_inv_n;
    vz_n =  dev_conserved[3*n_cells + id] * d_inv_n;
    dev_conserved[  n_cells + id] += 0.5*dt*gx*(d + d_n);
    dev_conserved[2*n_cells + id] += 0.5*dt*gy*(d + d_n);
    dev_conserved[3*n_cells + id] += 0.5*dt*gz*(d + d_n);
    dev_conserved[4*n_cells + id] += 0.25*dt*gx*(d + d_n)*(vx + vx_n)
                                  +  0.25*dt*gy*(d + d_n)*(vy + vy_n)
                                  +  0.25*dt*gz*(d + d_n)*(vz + vz_n);
    #endif

    #ifdef GRAVITY
    d_n  =  dev_conserved[            id];
    d_inv_n = 1.0 / d_n;
    vx_n =  dev_conserved[1*n_cells + id] * d_inv_n;
    vy_n =  dev_conserved[2*n_cells + id] * d_inv_n;
    vz_n =  dev_conserved[3*n_cells + id] * d_inv_n;

    // Calculate the -gradient of potential
    // Get X componet of gravity field
    id_l = (xid-1) + (yid)*nx + (zid)*nx*ny;
    id_r = (xid+1) + (yid)*nx + (zid)*nx*ny;
    pot_l = dev_potential[id_l];
    pot_r = dev_potential[id_r];
    #ifdef GRAVITY_5_POINTS_GRADIENT
    id_ll = (xid-2) + (yid)*nx + (zid)*nx*ny;
    id_rr = (xid+2) + (yid)*nx + (zid)*nx*ny;
    pot_ll = dev_potential[id_ll];
    pot_rr = dev_potential[id_rr];
    gx = -1 * ( -pot_rr + 8*pot_r - 8*pot_l + pot_ll) / (12*dx);
    #else
    gx = -0.5*( pot_r - pot_l ) / dx;
    #endif

    //Get Y componet of gravity field
    id_l = (xid) + (yid-1)*nx + (zid)*nx*ny;
    id_r = (xid) + (yid+1)*nx + (zid)*nx*ny;
    pot_l = dev_potential[id_l];
    pot_r = dev_potential[id_r];
    #ifdef GRAVITY_5_POINTS_GRADIENT
    id_ll = (xid) + (yid-2)*nx + (zid)*nx*ny;
    id_rr = (xid) + (yid+2)*nx + (zid)*nx*ny;
    pot_ll = dev_potential[id_ll];
    pot_rr = dev_potential[id_rr];
    gy = -1 * ( -pot_rr + 8*pot_r - 8*pot_l + pot_ll) / (12*dx);
    #else
    gy = -0.5*( pot_r - pot_l ) / dy;
    #endif
    //Get Z componet of gravity field
    id_l = (xid) + (yid)*nx + (zid-1)*nx*ny;
    id_r = (xid) + (yid)*nx + (zid+1)*nx*ny;
    pot_l = dev_potential[id_l];
    pot_r = dev_potential[id_r];
    #ifdef GRAVITY_5_POINTS_GRADIENT
    id_ll = (xid) + (yid)*nx + (zid-2)*nx*ny;
    id_rr = (xid) + (yid)*nx + (zid+2)*nx*ny;
    pot_ll = dev_potential[id_ll];
    pot_rr = dev_potential[id_rr];
    gz = -1 * ( -pot_rr + 8*pot_r - 8*pot_l + pot_ll) / (12*dx);
    #else
    gz = -0.5*( pot_r - pot_l ) / dz;
    #endif

    //Add gravity term to Momentum
    dev_conserved[  n_cells + id] += 0.5*dt*gx*(d + d_n);
    dev_conserved[2*n_cells + id] += 0.5*dt*gy*(d + d_n);
    dev_conserved[3*n_cells + id] += 0.5*dt*gz*(d + d_n);

    //Add gravity term to Total Energy
    //Add the work done by the gravitational force
    dev_conserved[4*n_cells + id] += 0.5* dt * ( gx*(d*vx + d_n*vx_n) +  gy*(d*vy + d_n*vy_n) +  gz*(d*vz + d_n*vz_n) );

    #endif


    #if !( defined(DENSITY_FLOOR) && defined(TEMPERATURE_FLOOR) )
    if (dev_conserved[id] < 0.0 || dev_conserved[id] != dev_conserved[id] || dev_conserved[4*n_cells + id] < 0.0 || dev_conserved[4*n_cells+id] != dev_conserved[4*n_cells+id]) {
      printf("%3d %3d %3d Thread crashed in final update. %e %e %e %e %e\n", xid+x_off, yid+y_off, zid+z_off, dev_conserved[id], dtodx*(dev_F_x[imo]-dev_F_x[id]), dtody*(dev_F_y[jmo]-dev_F_y[id]), dtodz*(dev_F_z[kmo]-dev_F_z[id]), dev_conserved[4*n_cells+id]);
    }
    #endif//DENSITY_FLOOR
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

 __device__ __host__ Real hydroInverseCrossingTime(Real const &E,
                                                   Real const &d,
                                                   Real const &d_inv,
                                                   Real const &vx,
                                                   Real const &vy,
                                                   Real const &vz,
                                                   Real const &dx,
                                                   Real const &dy,
                                                   Real const &dz,
                                                   Real const &gamma)
{
  // Compute pressure and sound speed
  Real P  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
  Real cs = sqrt(d_inv * gamma * P);

  // Find maximum inverse crossing time in the cell (i.e. minimum crossing time)
  Real cellMaxInverseDt = fmax((fabs(vx)+cs)/dx, (fabs(vy)+cs)/dy);
  cellMaxInverseDt      = fmax(cellMaxInverseDt, (fabs(vz)+cs)/dz);
  cellMaxInverseDt      = fmax(cellMaxInverseDt, 0.0);

  return cellMaxInverseDt;
}

__device__ __host__ Real mhdInverseCrossingTime(Real const &E,
                                                Real const &d,
                                                Real const &d_inv,
                                                Real const &vx,
                                                Real const &vy,
                                                Real const &vz,
                                                Real const &avgBx,
                                                Real const &avgBy,
                                                Real const &avgBz,
                                                Real const &dx,
                                                Real const &dy,
                                                Real const &dz,
                                                Real const &gamma)
{
  // Compute the gas pressure and fast magnetosonic speed
  Real gasP = mhdUtils::computeGasPressure(E, d, vx*d, vy*d, vz*d, avgBx, avgBy, avgBz, gamma);
  Real cf   = mhdUtils::fastMagnetosonicSpeed(d, gasP, avgBx, avgBy, avgBz, gamma);

  // Find maximum inverse crossing time in the cell (i.e. minimum crossing time)
  Real cellMaxInverseDt = fmax((fabs(vx)+cf)/dx, (fabs(vy)+cf)/dy);
  cellMaxInverseDt      = fmax(cellMaxInverseDt, (fabs(vz)+cf)/dz);
  cellMaxInverseDt      = fmax(cellMaxInverseDt, 0.0);

  return cellMaxInverseDt;
}



__global__ void Calc_dt_1D(Real *dev_conserved, Real *dev_dti, Real gamma, int n_ghost, int nx, Real dx)
{
  Real max_dti = -DBL_MAX;

  Real d, d_inv, vx, vy, vz, P, cs;
  int n_cells = nx;

  // Grid stride loop to perform as much of the reduction as possible. The
  // fact that `id` has type `size_t` is important. I'm not totally sure why
  // but setting it to int results in some kind of silent over/underflow issue
  // even though we're not hitting those kinds of numbers. Setting it to type
  // uint or size_t fixes them
  for(size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x)
  {
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
      max_dti = fmax(max_dti,(fabs(vx)+cs)/dx);
    }
  }

  // do the block wide reduction (find the max inverse timestep in the block)
  // then write it to that block's location in the dev_dti array
  max_dti = reduction_utilities::blockReduceMax(max_dti);
  if (threadIdx.x == 0) dev_dti[blockIdx.x] = max_dti;
}



__global__ void Calc_dt_2D(Real *dev_conserved, Real *dev_dti, Real gamma, int n_ghost, int nx, int ny, Real dx, Real dy)
{
  Real max_dti = -DBL_MAX;

  Real d, d_inv, vx, vy, vz, P, cs;
  int xid, yid, n_cells;
  n_cells = nx*ny;

  // Grid stride loop to perform as much of the reduction as possible. The
  // fact that `id` has type `size_t` is important. I'm not totally sure why
  // but setting it to int results in some kind of silent over/underflow issue
  // even though we're not hitting those kinds of numbers. Setting it to type
  // uint or size_t fixes them
  for(size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x)
  {
    // get a global thread ID
    yid = id / nx;
    xid = id - yid*nx;

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
      max_dti = fmax(max_dti,fmax((fabs(vx)+cs)/dx, (fabs(vy)+cs)/dy));
    }
  }

  // do the block wide reduction (find the max inverse timestep in the block)
  // then write it to that block's location in the dev_dti array
  max_dti = reduction_utilities::blockReduceMax(max_dti);
  if (threadIdx.x == 0) dev_dti[blockIdx.x] = max_dti;
}


__global__ void Calc_dt_3D(Real *dev_conserved, Real *dev_dti, Real gamma, int n_ghost, int n_fields, int nx, int ny, int nz, Real dx, Real dy, Real dz)
{
  Real max_dti = -DBL_MAX;

  Real d, d_inv, vx, vy, vz, E;
  #ifdef  MHD
    Real avgBx, avgBy, avgBz;
  #endif  //MHD
  int xid, yid, zid, n_cells;

  n_cells = nx*ny*nz;

  // Grid stride loop to perform as much of the reduction as possible. The
  // fact that `id` has type `size_t` is important. I'm not totally sure why
  // but setting it to int results in some kind of silent over/underflow issue
  // even though we're not hitting those kinds of numbers. Setting it to type
  // uint or size_t fixes them
  for(size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x)
  {
    // get a global thread ID
    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

    // threads corresponding to real cells do the calculation
    if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
    {
      // every thread collects the conserved variables it needs from global memory
      d     = dev_conserved[            id];
      d_inv = 1.0 / d;
      vx    = dev_conserved[1*n_cells + id] * d_inv;
      vy    = dev_conserved[2*n_cells + id] * d_inv;
      vz    = dev_conserved[3*n_cells + id] * d_inv;
      E     = dev_conserved[4*n_cells + id];
      #ifdef  MHD
        // Compute the cell centered magnetic field using a straight average of
        // the faces
        mhdUtils::cellCenteredMagneticFields(dev_conserved, id, xid, yid, zid, n_cells, nx, ny, avgBx, avgBy, avgBz);
      #endif  //MHD

      // Compute the maximum inverse crossing time in the cell
      #ifdef  MHD
        max_dti = fmax(max_dti,mhdInverseCrossingTime(E, d, d_inv, vx, vy, vz, avgBx, avgBy, avgBz, dx, dy, dz, gamma));
      #else  // not MHD
        max_dti = fmax(max_dti,hydroInverseCrossingTime(E, d, d_inv, vx, vy, vz, dx, dy, dz, gamma));
      #endif  //MHD

      Real P  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
      Real cs = sqrt(d_inv * gamma * P);
      Real n = d*DENSITY_UNIT/(0.6*MP);
      Real T = hydro_utilities::Calc_Temp(P, n);

      if (max_dti > 1) {
        printf("\nmax_dti: %e\n", max_dti);
        printf("E: %e g/(cm^2⋅s^2)\n", E*ENERGY_UNIT);
        printf("P: %e g/(cm⋅s^2)\n", P*PRESSURE_UNIT);
        printf("T: %e K\n", T);
        printf("cs: %e km/s\n", cs*1e-5*VELOCITY_UNIT);
        printf("d: %e g/cm^3\n", d*DENSITY_UNIT);
        printf("vx: %e km/s\n", vx*1e-5*VELOCITY_UNIT);
      }

    }
  }

  // do the block wide reduction (find the max inverse timestep in the block)
  // then write it to that block's location in the dev_dti array
  max_dti = reduction_utilities::blockReduceMax(max_dti);
  if (threadIdx.x == 0) dev_dti[blockIdx.x] = max_dti;
}

Real Calc_dt_GPU(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy, Real dz, Real gamma )
{
  // set values for GPU kernels
  uint threadsPerBlock, numBlocks;
  int ngrid = (nx*ny*nz + TPB - 1 )/TPB;
  // reduction_utilities::reductionLaunchParams(numBlocks, threadsPerBlock); // Uncomment this if we fix the AtomicDouble bug - Alwin
  threadsPerBlock = TPB;
  numBlocks = ngrid;

  Real* dev_dti = dev_dti_array;


  // compute dt and store in dev_dti
  if (nx > 1 && ny == 1 && nz == 1) //1D
  {
    hipLaunchKernelGGL(Calc_dt_1D, numBlocks, threadsPerBlock, 0, 0, dev_conserved, dev_dti, gamma, n_ghost, nx, dx);
  }
  else if (nx > 1 && ny > 1 && nz == 1) //2D
  {
    hipLaunchKernelGGL(Calc_dt_2D, numBlocks, threadsPerBlock, 0, 0, dev_conserved, dev_dti, gamma, n_ghost, nx, ny, dx, dy);
  }
  else if (nx > 1 && ny > 1 && nz > 1) //3D
  {
    hipLaunchKernelGGL(Calc_dt_3D, numBlocks, threadsPerBlock, 0, 0, dev_conserved, dev_dti, gamma, n_ghost, n_fields, nx, ny, nz, dx, dy, dz);
  }
  CudaCheckError();

  Real max_dti=0;

  /* Uncomment the below if we fix the AtomicDouble bug - Alwin
  // copy device side max_dti to host side max_dti


  CudaSafeCall( cudaMemcpy(&max_dti, dev_dti, sizeof(Real), cudaMemcpyDeviceToHost) );
  cudaDeviceSynchronize();

  return max_dti;
  */

  int dev_dti_length = numBlocks;
  CudaSafeCall(cudaMemcpy(host_dti_array,dev_dti, dev_dti_length*sizeof(Real), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  for (int i=0;i<dev_dti_length;i++){
    max_dti = fmax(max_dti,host_dti_array[i]);
  }

  return max_dti;
}


#ifdef AVERAGE_SLOW_CELLS

void Average_Slow_Cells( Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy, Real dz, Real gamma, Real max_dti_slow ){

  // set values for GPU kernels
  int n_cells = nx*ny*nz;
  int ngrid = (n_cells + TPB - 1) / TPB;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  if (nx > 1 && ny > 1 && nz > 1){ //3D
    hipLaunchKernelGGL(Average_Slow_Cells_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dx, dy, dz, gamma, max_dti_slow );
  }
}

__global__ void Average_Slow_Cells_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy, Real dz, Real gamma, Real max_dti_slow ){

  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, E, max_dti;
  #ifdef  MHD
    Real avgBx, avgBy, avgBz;
  #endif  //MHD

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  n_cells = nx*ny*nz;

  cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);


  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    E  =  dev_conserved[4*n_cells + id];

    #ifdef  MHD
      // Compute the cell centered magnetic field using a straight average of the faces
      mhdUtils::cellCenteredMagneticFields(dev_conserved, id, xid, yid, zid, n_cells, nx, ny, avgBx, avgBy, avgBz);
    #endif  //MHD

    // Compute the maximum inverse crossing time in the cell
    #ifdef  MHD
      max_dti = mhdInverseCrossingTime(E, d, d_inv, vx, vy, vz, avgBx, avgBy, avgBz, dx, dy, dz, gamma);
    #else  // not MHD
      max_dti = hydroInverseCrossingTime(E, d, d_inv, vx, vy, vz, dx, dy, dz, gamma);
    #endif  //MHD

    if (max_dti > max_dti_slow){
      // Average this cell
      printf(" Average Slow Cell [ %d %d %d ] -> dt_cell=%f    dt_min=%f\n", xid, yid, zid, 1./max_dti,  1./max_dti_slow );
      Average_Cell_All_Fields( xid, yid, zid, nx, ny, nz, n_cells, n_fields, dev_conserved );
    }
  }
}
#endif //AVERAGE_SLOW_CELLS


#ifdef DE
__global__ void Partial_Update_Advected_Internal_Energy_1D( Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, int nx, int n_ghost, Real dx, Real dt, Real gamma, int n_fields ){

  int id, xid, n_cells;
  int imo, ipo;
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo;
  Real  P, E, E_kin, GE;


  Real dtodx = dt/dx;
  n_cells = nx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  xid = id;


  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost)
  {
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    GE = dev_conserved[(n_fields-1)*n_cells + id];
    E_kin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );
    P = hydro_utilities::Get_Pressure_From_DE( E, E - E_kin, GE, gamma );
    P  = fmax(P, (Real) TINY_NUMBER);

    imo = xid-1;
    ipo = xid+1;

    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo];

    // Use center values of neighbor cells for the divergence of velocity
    dev_conserved[(n_fields-1)*n_cells + id] += 0.5*P*(dtodx*(vx_imo-vx_ipo));

  }
}


__global__ void Partial_Update_Advected_Internal_Energy_2D( Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly, Real *Q_Ry, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, Real gamma, int n_fields ){

  int id, xid, yid, n_cells;
  int imo, jmo;
  int ipo, jpo;
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo;
  Real  P, E, E_kin, GE;


  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;


  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost)
  {
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    GE = dev_conserved[(n_fields-1)*n_cells + id];
    E_kin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );
    P = hydro_utilities::Get_Pressure_From_DE( E, E - E_kin, GE, gamma );
    P  = fmax(P, (Real) TINY_NUMBER);

    imo = xid-1 + yid*nx;
    ipo = xid+1 + yid*nx;
    jmo = xid + (yid-1)*nx;
    jpo = xid + (yid+1)*nx;

    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo];
    vy_jmo = dev_conserved[2*n_cells + jmo] / dev_conserved[jmo];
    vy_jpo = dev_conserved[2*n_cells + jpo] / dev_conserved[jpo];

    // Use center values of neighbor cells for the divergence of velocity
    dev_conserved[(n_fields-1)*n_cells + id] += 0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo));

  }
}

__global__ void Partial_Update_Advected_Internal_Energy_3D( Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly, Real *Q_Ry, Real *Q_Lz, Real *Q_Rz, int nx, int ny, int nz,  int n_ghost, Real dx, Real dy, Real dz,  Real dt, Real gamma, int n_fields ){

  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;
  int ipo, jpo, kpo;
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo;
  Real  P, E, E_kin, GE;
  // Real vx_L, vx_R, vy_L, vy_R, vz_L, vz_R;


  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  Real dtodz = dt/dz;
  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    GE = dev_conserved[(n_fields-1)*n_cells + id];
    E_kin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );
    P = hydro_utilities::Get_Pressure_From_DE( E, E - E_kin, GE, gamma );
    P  = fmax(P, (Real) TINY_NUMBER);

    imo = xid-1 + yid*nx + zid*nx*ny;
    jmo = xid + (yid-1)*nx + zid*nx*ny;
    kmo = xid + yid*nx + (zid-1)*nx*ny;

    ipo = xid+1 + yid*nx + zid*nx*ny;
    jpo = xid + (yid+1)*nx + zid*nx*ny;
    kpo = xid + yid*nx + (zid+1)*nx*ny;

    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo];
    vy_jmo = dev_conserved[2*n_cells + jmo] / dev_conserved[jmo];
    vy_jpo = dev_conserved[2*n_cells + jpo] / dev_conserved[jpo];
    vz_kmo = dev_conserved[3*n_cells + kmo] / dev_conserved[kmo];
    vz_kpo = dev_conserved[3*n_cells + kpo] / dev_conserved[kpo];

    // Use center values of neighbor cells for the divergence of velocity
    dev_conserved[(n_fields-1)*n_cells + id] += 0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));

    // OPTION 2: Use the reconstructed velocities to compute the velocity gradient
    //Use the reconstructed Velocities instead of neighbor cells centered values
    // vx_R = Q_Lx[1*n_cells + id]  / Q_Lx[id];
    // vx_L = Q_Rx[1*n_cells + imo] / Q_Rx[imo];
    // vy_R = Q_Ly[2*n_cells + id]  / Q_Ly[id];
    // vy_L = Q_Ry[2*n_cells + jmo] / Q_Ry[jmo];
    // vz_R = Q_Lz[3*n_cells + id]  / Q_Lz[id];
    // vz_L = Q_Rz[3*n_cells + kmo] / Q_Rz[kmo];

    //Use the reconstructed Velocities instead of neighbor cells centered values
    // dev_conserved[(n_fields-1)*n_cells + id] +=  P * ( dtodx * ( vx_L - vx_R ) + dtody * ( vy_L - vy_R ) + dtodz * ( vz_L - vz_R ) );


  }
}


__global__ void Select_Internal_Energy_1D( Real *dev_conserved, int nx, int n_ghost, int n_fields ){

  int id, xid, n_cells;
  Real d, d_inv, vx, vy, vz, E, U_total, U_advected, U, Emax;
  int imo, ipo;
  n_cells = nx;

  Real eta_2 = DE_ETA_2;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  xid = id;

  imo = max(xid-1, n_ghost);
  ipo = min(xid+1, nx-n_ghost-1);


  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    E  =  dev_conserved[4*n_cells + id];
    U_advected = dev_conserved[(n_fields-1)*n_cells + id];
    U_total = E - 0.5*d*( vx*vx + vy*vy + vz*vz );

    //find the max nearby total energy
    Emax = fmax(dev_conserved[4*n_cells + imo], E);
    Emax = fmax(Emax, dev_conserved[4*n_cells + ipo]);

    if (U_total/Emax > eta_2 ) U = U_total;
    else U = U_advected;

    //Optional: Avoid Negative Internal  Energies
    U = fmax(U, (Real) TINY_NUMBER);

    //Write Selected internal energy to the GasEnergy array ONLY
    //to avoid mixing updated and non-updated values of E
    //since the Dual Energy condition depends on the neighbor cells
    dev_conserved[(n_fields-1)*n_cells + id] = U;

  }
}


__global__ void Select_Internal_Energy_2D( Real *dev_conserved, int nx, int ny, int n_ghost, int n_fields ){

  int id, xid, yid, n_cells;
  Real d, d_inv, vx, vy, vz, E, U_total, U_advected, U, Emax;
  int imo, ipo, jmo, jpo;
  n_cells = nx*ny;

  Real eta_2 = DE_ETA_2;

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
    U_advected = dev_conserved[(n_fields-1)*n_cells + id];
    U_total = E - 0.5*d*( vx*vx + vy*vy + vz*vz );

    //find the max nearby total energy
    Emax = fmax(dev_conserved[4*n_cells + imo], E);
    Emax = fmax(Emax, dev_conserved[4*n_cells + ipo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + jmo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + jpo]);

    if (U_total/Emax > eta_2 ) U = U_total;
    else U = U_advected;

    //Optional: Avoid Negative Internal  Energies
    U = fmax(U, (Real) TINY_NUMBER);

    //Write Selected internal energy to the GasEnergy array ONLY
    //to avoid mixing updated and non-updated values of E
    //since the Dual Energy condition depends on the neighbour cells
    dev_conserved[(n_fields-1)*n_cells + id] = U;

  }
}


__global__ void Select_Internal_Energy_3D( Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields ){

  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, E, U_total, U_advected, U, Emax;
  int imo, ipo, jmo, jpo, kmo, kpo;
  n_cells = nx*ny*nz;

  Real eta_2 = DE_ETA_2;

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
    U_advected = dev_conserved[(n_fields-1)*n_cells + id];
    U_total = E - 0.5*d*( vx*vx + vy*vy + vz*vz );

    //find the max nearby total energy
    Emax = fmax(dev_conserved[4*n_cells + imo], E);
    Emax = fmax(Emax, dev_conserved[4*n_cells + ipo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + jmo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + jpo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + kmo]);
    Emax = fmax(Emax, dev_conserved[4*n_cells + kpo]);

    if (U_total/Emax > eta_2 ) U = U_total;
    else U = U_advected;

    //Optional: Avoid Negative Internal  Energies
    U = fmax(U, (Real) TINY_NUMBER);

    //Write Selected internal energy to the GasEnergy array ONLY
    //to avoid mixing updated and non-updated values of E
    //since the Dual Energy condition depends on the neighbour cells
    dev_conserved[(n_fields-1)*n_cells + id] = U;

  }
}

__global__ void Sync_Energies_1D(Real *dev_conserved, int nx, int n_ghost, Real gamma, int n_fields)
{
  int id, xid, n_cells;
  Real d, d_inv, vx, vy, vz, U;
  n_cells = nx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  xid = id;


  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    U = dev_conserved[(n_fields-1)*n_cells + id];

    //Use the previously selected Internal Energy to update the total energy
    dev_conserved[4*n_cells + id] = 0.5*d*( vx*vx + vy*vy + vz*vz ) + U;
  }

}


__global__ void Sync_Energies_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real gamma, int n_fields)
{
  int id, xid, yid, n_cells;
  Real d, d_inv, vx, vy, vz, U;
  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;


  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    U = dev_conserved[(n_fields-1)*n_cells + id];

    //Use the previously selected Internal Energy to update the total energy
    dev_conserved[4*n_cells + id] = 0.5*d*( vx*vx + vy*vy + vz*vz ) + U;
  }

}


__global__ void Sync_Energies_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real gamma, int n_fields)
{
  //Called in a separate kernel to avoid interfering with energy selection in Select_Internal_Energy

  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, U;
  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    U = dev_conserved[(n_fields-1)*n_cells + id];

    //Use the previously selected Internal Energy to update the total energy
    dev_conserved[4*n_cells + id] = 0.5*d*( vx*vx + vy*vy + vz*vz ) + U;
  }
}


#endif //DE

#ifdef TEMPERATURE_FLOOR
__global__ void Apply_Temperature_Floor(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,  Real U_floor )
{
  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, E, Ekin, U;
  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;


  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    E  =  dev_conserved[4*n_cells + id];
    Ekin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );

    U = ( E - Ekin ) / d;
    if ( U < U_floor ) dev_conserved[4*n_cells + id] = Ekin + d*U_floor;

    #ifdef DE
    U = dev_conserved[(n_fields-1)*n_cells + id] / d ;
    if ( U < U_floor ) dev_conserved[(n_fields-1)*n_cells + id] = d*U_floor ;
    #endif
  }
}
#endif //TEMPERATURE_FLOOR


__device__ Real Average_Cell_Single_Field( int field_indx, int i, int j, int k, int nx, int ny, int nz, int ncells, Real *conserved ){
  Real v_l, v_r, v_d, v_u, v_b, v_t, v_avrg;
  int id;

  id = (i-1) + (j)*nx + (k)*nx*ny;
  v_l = conserved[ field_indx*ncells + id ];
  id = (i+1) + (j)*nx + (k)*nx*ny;
  v_r = conserved[ field_indx*ncells + id ];
  id = (i) + (j-1)*nx + (k)*nx*ny;
  v_d = conserved[ field_indx*ncells + id ];
  id = (i) + (j+1)*nx + (k)*nx*ny;
  v_u = conserved[ field_indx*ncells + id ];
  id = (i) + (j)*nx + (k-1)*nx*ny;
  v_b = conserved[ field_indx*ncells + id ];
  id = (i) + (j)*nx + (k+1)*nx*ny;
  v_t = conserved[ field_indx*ncells + id ];
  v_avrg = ( v_l + v_r + v_d + v_u + v_b + v_t ) / 6;
  id = (i) + (j)*nx + (k)*nx*ny;
  conserved[ field_indx*ncells + id ] = v_avrg;
  return v_avrg;

}

__device__ void Average_Cell_All_Fields( int i, int j, int k, int nx, int ny, int nz, int ncells, int n_fields, Real *conserved ){

  // Average Density
  Average_Cell_Single_Field( 0, i, j, k, nx, ny, nz, ncells, conserved );
  // Average Momentum_x
  Average_Cell_Single_Field( 1, i, j, k, nx, ny, nz, ncells, conserved );
  // Average Momentum_y
  Average_Cell_Single_Field( 2, i, j, k, nx, ny, nz, ncells, conserved );
  // Average Momentum_z
  Average_Cell_Single_Field( 3, i, j, k, nx, ny, nz, ncells, conserved );
  // Average Energy
  Average_Cell_Single_Field( 4, i, j, k, nx, ny, nz, ncells, conserved );
  #ifdef  MHD
    // Average MHD
    Average_Cell_Single_Field( 5+NSCALARS, i,   j,   k,   nx, ny, nz, ncells, conserved );
    Average_Cell_Single_Field( 6+NSCALARS, i,   j,   k,   nx, ny, nz, ncells, conserved );
    Average_Cell_Single_Field( 7+NSCALARS, i,   j,   k,   nx, ny, nz, ncells, conserved );
    Average_Cell_Single_Field( 5+NSCALARS, i-1, j,   k,   nx, ny, nz, ncells, conserved );
    Average_Cell_Single_Field( 6+NSCALARS, i,   j-1, k,   nx, ny, nz, ncells, conserved );
    Average_Cell_Single_Field( 7+NSCALARS, i,   j,   k-1, nx, ny, nz, ncells, conserved );
  #endif  //MHD
  #ifdef DE
  // Average GasEnergy
  Average_Cell_Single_Field( n_fields-1, i, j, k, nx, ny, nz, ncells, conserved );
  #endif  //DE
}


#endif //CUDA

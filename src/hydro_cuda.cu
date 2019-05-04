/*! \file hydro_cuda.cu
 *  \brief Definitions of functions used in all cuda integration algorithms. */
#ifdef CUDA

#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"
#include"gravity_cuda.h"


__global__ void Update_Conserved_Variables_1D(Real *dev_conserved, Real *dev_F, int n_cells, int x_off, int n_ghost, Real dx, Real xbound, Real dt, Real gamma, int n_fields)
{
  int id;
  #if defined(DE) || defined(STATIC_GRAV)
  Real d, d_inv, vx;  
  #endif
  #ifdef DE
  Real vx_imo, vx_ipo, vy, vz, P;
  #endif
  #ifdef STATIC_GRAV
  Real gx, d_n, d_inv_n, vx_n;
  gx = 0.0;
  #endif
  
  Real dtodx = dt/dx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;


  // threads corresponding to real cells do the calculation
  if (id > n_ghost - 1 && id < n_cells-n_ghost)
  {
    #if defined(DE) || defined(STATIC_GRAV)
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    #endif
    #ifdef DE
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
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved[(5+i)*n_cells + id] += dtodx * (dev_F[(5+i)*n_cells + id-1] - dev_F[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_conserved[(n_fields-1)*n_cells + id] += dtodx * (dev_F[(n_fields-1)*n_cells + id-1] - dev_F[(n_fields-1)*n_cells + id])
                                  +  dtodx * P * 0.5 * (vx_imo - vx_ipo);
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

  #if defined (DE) || defined(STATIC_GRAV)
  Real d, d_inv, vx, vy;
  #endif
  #ifdef DE
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz, P;
  int ipo, jpo;
  #endif

  #ifdef STATIC_GRAV
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
    #if defined (DE) || defined (STATIC_GRAV)
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    #endif
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
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved[(5+i)*n_cells + id] += dtodx * (dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                        +  dtody * (dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_conserved[(n_fields-1)*n_cells + id] += dtodx * (dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id])
                                  +  dtody * (dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id])
                                  +  0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo));
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



__global__ void Update_Conserved_Variables_3D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, 
                                              Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt,
                                              Real gamma, int n_fields, Real density_floor )
{
  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;
  #if defined (DE) || defined(STATIC_GRAV) || ( defined(GRAVITY) && defined(GRAVITY_COUPLE_GPU) )
  Real d, d_inv, vx, vy, vz;
  #endif
  #ifdef DE
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo, P, E, E_kin, GE;
  int ipo, jpo, kpo;
  #endif

  #ifdef STATIC_GRAV
  Real gx, gy, gz, d_n, d_inv_n, vx_n, vy_n, vz_n;
  gx = 0.0;
  gy = 0.0;
  gz = 0.0;
  #endif
  
  #ifdef DENSITY_FLOOR
  Real dens_0;
  #endif
  
  #if ( defined(GRAVITY) && defined(GRAVITY_COUPLE_GPU) )
  Real gx, gy, gz, d_n, d_inv_n, vx_n, vy_n, vz_n;
  Real pot_l, pot_r;
  int id_l, id_r;
  gx = 0.0;
  gy = 0.0;
  gz = 0.0;
  int field_pot;

  #ifdef DE
  field_pot = n_fields - 2;
  #else
  field_pot = n_fields - 1;
  #endif //DE
  
  #ifdef COUPLE_DELTA_E_KINETIC
  Real Ekin_0, Ekin_1;
  #endif//COUPLE_DELTA_E_KINETIC
  #endif //GRAVTY

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
    #if defined (DE) || defined(STATIC_GRAV) || ( defined(GRAVITY) && defined(GRAVITY_COUPLE_GPU) )
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    #endif
    #ifdef DE
    //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    GE = dev_conserved[(n_fields-1)*n_cells + id];
    E_kin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );
    P = Get_Pressure_From_DE( E, E - E_kin, GE, gamma );  
    P  = fmax(P, (Real) TINY_NUMBER); 
    // P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
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
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved[(5+i)*n_cells + id] += dtodx * (dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                    +  dtody * (dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id])
                                    +  dtodz * (dev_F_z[(5+i)*n_cells + kmo] - dev_F_z[(5+i)*n_cells + id]);
    }                              
    #endif
    #ifdef DE
    dev_conserved[(n_fields-1)*n_cells + id] += dtodx * (dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id])
                                  +  dtody * (dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id])
                                  +  dtodz * (dev_F_z[(n_fields-1)*n_cells + kmo] - dev_F_z[(n_fields-1)*n_cells + id])
                                  +  0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));
    #endif
    
    #ifdef DENSITY_FLOOR
    if ( dev_conserved[            id] < density_floor ){
      dens_0 = dev_conserved[            id];
      printf("###Thread density change  %f -> %f \n", dens_0, density_floor );
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
    #endif

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
    
    #if ( defined(GRAVITY) && defined(GRAVITY_COUPLE_GPU) )
    d_n  =  dev_conserved[            id];
    d_inv_n = 1.0 / d_n;
    vx_n =  dev_conserved[1*n_cells + id] * d_inv_n;
    vy_n =  dev_conserved[2*n_cells + id] * d_inv_n;
    vz_n =  dev_conserved[3*n_cells + id] * d_inv_n;
    
    #ifdef COUPLE_DELTA_E_KINETIC
    Ekin_0 = 0.5 * d_n * ( vx_n*vx_n + vy_n*vy_n + vz_n*vz_n );
    #endif
    
    // Calculate the -gradient of potential
    // Get X componet of gravity field
    id_l = (xid-1) + (yid)*nx + (zid)*nx*ny;
    id_r = (xid+1) + (yid)*nx + (zid)*nx*ny;
    pot_l = dev_conserved[field_pot*n_cells + id_l];
    pot_r = dev_conserved[field_pot*n_cells + id_r];
    gx = -0.5*( pot_r - pot_l ) / dx;
    
    //Get Y componet of gravity field
    id_l = (xid) + (yid-1)*nx + (zid)*nx*ny;
    id_r = (xid) + (yid+1)*nx + (zid)*nx*ny;
    pot_l = dev_conserved[field_pot*n_cells + id_l];
    pot_r = dev_conserved[field_pot*n_cells + id_r];
    gy = -0.5*( pot_r - pot_l ) / dy;
    
    //Get Z componet of gravity field
    id_l = (xid) + (yid)*nx + (zid-1)*nx*ny;
    id_r = (xid) + (yid)*nx + (zid+1)*nx*ny;
    pot_l = dev_conserved[field_pot*n_cells + id_l];
    pot_r = dev_conserved[field_pot*n_cells + id_r];
    gz = -0.5*( pot_r - pot_l ) / dz;
    
    
    dev_conserved[  n_cells + id] += 0.5*dt*gx*(d + d_n);
    dev_conserved[2*n_cells + id] += 0.5*dt*gy*(d + d_n);
    dev_conserved[3*n_cells + id] += 0.5*dt*gz*(d + d_n);
    
    #ifdef COUPLE_GRAVITATIONAL_WORK
    dev_conserved[4*n_cells + id] += 0.5* dt * ( gx*(d*vx + d_n*vx_n) +  gy*(d*vy + d_n*vy_n) +  gz*(d*vz + d_n*vz_n) );
    #endif
    
    #ifdef COUPLE_DELTA_E_KINETIC
    vx_n =  dev_conserved[1*n_cells + id] * d_inv_n;
    vy_n =  dev_conserved[2*n_cells + id] * d_inv_n;
    vz_n =  dev_conserved[3*n_cells + id] * d_inv_n;
    Ekin_1 = 0.5 * d_n * ( vx_n*vx_n + vy_n*vy_n + vz_n*vz_n );
    dev_conserved[4*n_cells + id] += Ekin_1 - Ekin_0;
    #endif
    
    
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


__global__ void Sync_Energies_1D(Real *dev_conserved, int n_cells, int n_ghost, Real gamma, int n_fields)
{
  int id;
  Real d, d_inv, vx, vy, vz, E;
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
    // separately tracked internal energy 
    ge1 = dev_conserved[(n_fields-1)*n_cells + id];
    // internal energy calculated from total energy
    ge2 = dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    // if the ratio of conservatively calculated internal energy to total energy
    // is greater than 1/1000, use the conservatively calculated internal energy
    // to do the internal energy update
    if (ge2/E > 0.001) {
      dev_conserved[(n_fields-1)*n_cells + id] = ge2;
      ge1 = ge2;
    }     
    // find the max nearby total energy 
    Emax = fmax(dev_conserved[4*n_cells + im1], E);
    Emax = fmax(dev_conserved[4*n_cells + ip1], Emax);
    // if the ratio of conservatively calculated internal energy to max nearby total energy
    // is greater than 1/10, continue to use the conservatively calculated internal energy 
    if (ge2/Emax > 0.1) {
      dev_conserved[(n_fields-1)*n_cells + id] = ge2;
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
    // calculate the pressure 
    //P = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);    
    //if (P < 0.0) printf("%d Negative pressure after internal energy sync. %f %f \n", id, ge1, ge2);    
  }

}


__global__ void Sync_Energies_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real gamma, int n_fields)
{
  int id, xid, yid, n_cells;
  Real d, d_inv, vx, vy, vz, E;
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
    // separately tracked internal energy 
    ge1 =  dev_conserved[(n_fields-1)*n_cells + id];
    // internal energy calculated from total energy
    ge2 = dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    // if the ratio of conservatively calculated internal energy to total energy
    // is greater than 1/1000, use the conservatively calculated internal energy
    // to do the internal energy update
    if (ge2/E > 0.001) {
      dev_conserved[(n_fields-1)*n_cells + id] = ge2;
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
      dev_conserved[(n_fields-1)*n_cells + id] = ge2;
    }
    // sync the total energy with the internal energy 
    else {
      dev_conserved[4*n_cells + id] += ge1 - ge2;
    }
    // calculate the pressure 
    //Real P = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);    
    //if (P < 0.0) printf("%d Negative pressure after internal energy sync. %f %f \n", id, ge1, ge2);    
  }
}




__global__ void Sync_Energies_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real gamma, int n_fields)
{
  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, E;
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
    // don't do the energy sync if this thread has crashed
    if (E < 0.0 || E != E) return;
    // separately tracked internal energy 
    ge1 =  dev_conserved[(n_fields-1)*n_cells + id];
    // internal energy calculated from total energy
    ge2 = dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    // if the ratio of conservatively calculated internal energy to total energy
    // is greater than 1/1000, use the conservatively calculated internal energy
    // to do the internal energy update
    if (ge2 > 0.0 && E > 0.0 && ge2/E > 0.001) {
      dev_conserved[(n_fields-1)*n_cells + id] = ge2;
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
    if (ge2/Emax > 0.1 && ge2 > 0.0 && Emax > 0.0) {
      dev_conserved[(n_fields-1)*n_cells + id] = ge2;
    }
    // sync the total energy with the internal energy 
    else {
      if (ge1 > 0.0) dev_conserved[4*n_cells + id] += ge1 - ge2;
      else dev_conserved[(n_fields-1)*n_cells+id] = ge2;
    }
    // calculate the pressure 
    //Real P = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    //if (P < 0.0) printf("%3d %3d %3d Negative pressure after internal energy sync. %f %f %f\n", xid, yid, zid, P/(gamma-1.0), ge1, ge2);    
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

  Real d, d_inv, vx, vy, vz, E, P, cs;
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
    E  = dev_conserved[4*n_cells + id];
    P  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    cs = sqrt(d_inv * gamma * P);
    max_dti[tid] = fmax((fabs(vx)+cs)/dx, (fabs(vy)+cs)/dy);
    max_dti[tid] = fmax(max_dti[tid], (fabs(vz)+cs)/dz);
    max_dti[tid] = fmax(max_dti[tid], 0.0);
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

#ifdef DE //PRESSURE_DE
__host__ __device__ Real Get_Pressure_From_DE( Real E, Real U_total, Real U_advected, Real gamma ){
  
  Real U, P;
  Real eta = DE_LIMIT;
  
  if( U_total / E > eta ) U = U_total;
  else U = U_advected;

  P = U * (gamma - 1.0);
  return P;
}

#endif //DE

#endif //CUDA

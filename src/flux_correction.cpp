/*! \file flux_correction.cpp
 *  \brief First-order flux correction */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"global.h"
#include"flux_correction.h"
#include"exact.h"
#include"roe.h"
#include"hllc.h"
#ifdef MPI_CHOLLA
#include"mpi_routines.h"
#endif



void Flux_Correction_3D(Real *C1, Real *C2, int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt)
{

  int n_cells = nx*ny*nz;
  int id, imo, ipo, jmo, jpo, kmo, kpo;
  int istart, istop, jstart, jstop, kstart, kstop;
  int nfields = 5;
  #ifdef DE
  nfields = 6;
  #endif

  Real d_old, vx_old, vy_old, vz_old, P_old;
  Real d_new, vx_new, vy_new, vz_new, P_new;
  Real etah = 0.0;

  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  Real dtodz = dt/dz;
  
  // sweep through real cells and look for negative densities or pressures in new data
  istart = n_ghost; istop = nx-n_ghost;
  jstart = n_ghost; jstop = ny-n_ghost;
  kstart = n_ghost; kstop = nz-n_ghost;
  for (int k=kstart; k<kstop; k++) {
    for (int j=jstart; j<jstop; j++) {
      for (int i=istart; i<istop; i++) {

        id = i + j*nx + k*nx*ny;
        
        d_new = C2[id];
        vx_new = C2[n_cells+id]/d_new;
        vy_new = C2[2*n_cells+id]/d_new;
        vz_new = C2[3*n_cells+id]/d_new;
        P_new = (C2[4*n_cells+id] - 0.5*d_new*(vx_new*vx_new + vy_new*vy_new + vz_new*vz_new))*(gama-1.0);
  
        // if there is a problem, redo the update for that cell using first-order fluxes
        if (d_new < 0.0 || d_new != d_new || P_new < 0.0 || P_new != P_new) {
          //printf("Flux correction: (%d, %d %d) d: %e  p:%e\n", i, j, k, d_new, P_new);
          //printf("%3d %3d %3d Old density / pressure: d: %e p: %e\n", i+nx_local_start, j+ny_local_start, k+nz_local_start, d_new, P_new);
          //printf("Previous timestep data: d: %e mx: %e my: %e mz: %e E: %e\n", C1[id], C1[n_cells+id], C1[2*n_cells+id], C1[3*n_cells+id], C1[4*n_cells+id]);
          //printf("Uncorrected data: d: %e E: %e\n", C2[id], C2[4*n_cells+id]);
          P_old = P_new;

          Real C_half[nfields];
          Real C_half_imo[nfields];
          Real C_half_ipo[nfields];
          Real C_half_jmo[nfields];
          Real C_half_jpo[nfields];
          Real C_half_kmo[nfields];
          Real C_half_kpo[nfields];

          // calculate the first order half step update for the cell in question
          half_step_update(C_half, C1, i, j, k, dtodx, dtody, dtodz, nfields, nx, ny, nz, n_cells);
          //printf("Half step data: d: %e E: %e\n", C_half[0], C_half[4]);
          // need C_half for all the surrounding cells, as well
          half_step_update(C_half_imo, C1, i-1, j, k, dtodx, dtody, dtodz, nfields, nx, ny, nx, n_cells);
          //printf("Half step data: d: %e E: %e\n", C_half_imo[0], C_half_imo[4]);
          half_step_update(C_half_ipo, C1, i+1, j, k, dtodx, dtody, dtodz, nfields, nx, ny, nx, n_cells);
          //printf("Half step data: d: %e E: %e\n", C_half_ipo[0], C_half_ipo[4]);
          half_step_update(C_half_jmo, C1, i, j-1, k, dtodx, dtody, dtodz, nfields, nx, ny, nx, n_cells);
          //printf("Half step data: d: %e E: %e\n", C_half_jmo[0], C_half_jmo[4]);
          half_step_update(C_half_jpo, C1, i, j+1, k, dtodx, dtody, dtodz, nfields, nx, ny, nx, n_cells);
          //printf("Half step data: d: %e E: %e\n", C_half_jpo[0], C_half_jpo[4]);
          half_step_update(C_half_kmo, C1, i, j, k-1, dtodx, dtody, dtodz, nfields, nx, ny, nx, n_cells);
          //printf("Half step data: d: %e E: %e\n", C_half_kmo[0], C_half_kmo[4]);
          half_step_update(C_half_kpo, C1, i, j, k+1, dtodx, dtody, dtodz, nfields, nx, ny, nx, n_cells);
          //printf("Half step data: d: %e E: %e\n", C_half_kpo[0], C_half_kpo[4]);

          // Recalculate the fluxes, again using piecewise constant reconstruction
          // and update the conserved variables using the new first-order fluxes
          full_step_update(C1, C2, i, j, k, dtodx, dtody, dtodz, nfields, nx, ny, nz, n_cells, C_half, C_half_imo, C_half_ipo, C_half_jmo, C_half_jpo, C_half_kmo, C_half_kpo);
          //printf("Flux corrected data: d: %e E: %e\n", C2[id], C2[4*n_cells+id]);

          // Reset with the new values of the conserved variables
          d_new = C2[id];
          vx_new = C2[n_cells+id]/d_new;
          vy_new = C2[2*n_cells+id]/d_new;
          vz_new = C2[3*n_cells+id]/d_new;
          P_new = (C2[4*n_cells+id] - 0.5*d_new*(vx_new*vx_new + vy_new*vy_new + vz_new*vz_new))*(gama-1.0);
          // And apply gravity
          #ifdef STATIC_GRAV
          Real gx, gy, gz;
          gx = gy = gz = 0.0;
          calc_g_3D(i, j, k, x_off, y_off, z_off, n_ghost, dx, dy, dz, xbound, ybound, zbound, &gx, &gy, &gz);
          d_old = C1[id];
          vx_old = C1[n_cells+id]/d_old;
          vy_old = C1[2*n_cells+id]/d_old;
          vz_old = C1[3*n_cells+id]/d_old;
          C2[  n_cells + id] += 0.5*dt*gx*(d_old + d_new);
          C2[2*n_cells + id] += 0.5*dt*gy*(d_old + d_new);
          C2[3*n_cells + id] += 0.5*dt*gz*(d_old + d_new);
          C2[4*n_cells + id] += 0.25*dt*gx*(d_old + d_new)*(vx_old + vx_new) 
                              + 0.25*dt*gy*(d_old + d_new)*(vy_old + vy_new)
                              + 0.25*dt*gz*(d_old + d_new)*(vz_old + vz_new);
          #endif
          //printf("Before internal energy sync. d: %e vx: %e vy: %e vz: %e P: %e\n", d_new, vx_new, vy_new, vz_new, P_new);
          // sync the internal and total energy
          #ifdef DE
          Real ge1, ge2, E, Emax;
          int ipo, imo, jpo, jmo, kpo, kmo;
          E = C2[4*n_cells+id];
          // separately tracked internal energy
          ge1 = C2[5*n_cells+id];
          // internal energy calculated from total energy
          ge2 = P_new / (gama-1.0);
          // if the ratio of conservatively calculated internal energy to total energy
          // is greater than 1/1000, use the conservatively calculated internal energy
          // to do the internal energy update
          if (ge2/E > 0.001) {
            C2[5*n_cells + id] = ge2;
            ge1 = ge2;
          }     
          //find the max nearby total energy 
          imo = fmax(i-1, n_ghost);
          imo = imo + j*nx + k*nx*ny;
          ipo = fmin(i+1, nx-n_ghost-1);
          ipo = ipo + j*nx + k*nx*ny;
          jmo = fmax(j-1, n_ghost);
          jmo = i + jmo*nx + k*nx*ny;
          jpo = fmin(j+1, ny-n_ghost-1);
          jpo = i + jpo*nx + k*nx*ny;
          kmo = fmax(k-1, n_ghost);
          kmo = i + j*nx + kmo*nx*ny;
          kpo = fmin(k+1, nz-n_ghost-1);
          kpo = i + j*nx + kpo*nx*ny;
          Emax = fmax(C2[4*n_cells + imo+j*nx+k*nx*ny], E);
          Emax = fmax(Emax, C2[4*n_cells + ipo]);
          Emax = fmax(Emax, C2[4*n_cells + jmo]);
          Emax = fmax(Emax, C2[4*n_cells + jpo]);
          Emax = fmax(Emax, C2[4*n_cells + kmo]);
          Emax = fmax(Emax, C2[4*n_cells + kpo]);
          // if the ratio of conservatively calculated internal energy to max nearby total energy
          // is greater than 1/10, continue to use the conservatively calculated internal energy 
          if (ge2/Emax > 0.1) {
            C2[5*n_cells + id] = ge2;
          }
          // sync the total energy with the internal energy 
          else {
            C2[4*n_cells + id] += ge1 - ge2;
          }
          // recalculate the pressure
          P_new = (C2[4*n_cells+id] - 0.5*d_new*(vx_new*vx_new + vy_new*vy_new + vz_new*vz_new))*(gama-1.0);
          #endif          
          //printf("%3d %3d %3d New density / pressure: d: %e p: %e\n", i+nx_local_start, j+ny_local_start, k+nz_local_start, d_new, P_new);
          //if (d_new < 0.0 || d_new != d_new || P_new < 0.0 || P_new != P_new) printf("FLUX CORRECTION FAILED: %d %d %d %e %e\n", i+nx_local_start, j+ny_local_start, k+nz_local_start, d_new, P_new);

          // apply cooling
          #ifdef COOLING_GPU
          cooling_CPU(C2, id, n_cells, dt);
          #endif

        }

      }
    }
  }



}


void fill_flux_array(Real *C1, int idl, int idr, Real cW[], int n_cells)
{

  cW[0] = C1[idl];
  cW[1] = C1[idr];
  cW[2] = C1[n_cells+idl];
  cW[3] = C1[n_cells+idr];
  cW[4] = C1[2*n_cells+idl];
  cW[5] = C1[2*n_cells+idr];
  cW[6] = C1[3*n_cells+idl];
  cW[7] = C1[3*n_cells+idr];
  cW[8] = C1[4*n_cells+idl];
  cW[9] = C1[4*n_cells+idr];
  #ifdef DE
  cW[10] = C1[5*n_cells+idl];
  cW[11] = C1[5*n_cells+idr];
  #endif

}


void fill_flux_array_2(Real C_half_l[], Real C_half_r[], Real cW[], int n_cells)
{
  cW[0] = C_half_l[0];
  cW[1] = C_half_r[0];
  cW[2] = C_half_l[1];
  cW[3] = C_half_r[1];
  cW[4] = C_half_l[2];
  cW[5] = C_half_r[2];
  cW[6] = C_half_l[3];
  cW[7] = C_half_r[3];
  cW[8] = C_half_l[4];
  cW[9] = C_half_r[4];
  #ifdef DE
  cW[10] = C_half_l[5];
  cW[11] = C_half_r[5];
  #endif
}


void half_step_update(Real C_half[], Real *C1, int i, int j, int k, Real dtodx, Real dtody, Real dtodz, int nfields, int nx, int ny, int nz, int n_cells)
{
  int id = i + j*nx + k*nx*ny;
  int imo = i-1 + j*nx + k*nx*ny;
  int ipo = i+1 + j*nx + k*nx*ny;
  int jmo = i + (j-1)*nx + k*nx*ny;
  int jpo = i + (j+1)*nx + k*nx*ny;
  int kmo = i + j*nx + (k-1)*nx*ny;
  int kpo = i + j*nx + (k+1)*nx*ny;
  Real etah = 0.0;

  #ifdef DE
  Real d, d_inv, vx, vy, vz, P, vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo;
  d = C1[id];
  d_inv = 1.0 / d;
  vx = C1[1*n_cells+id]*d_inv;
  vy = C1[2*n_cells+id]*d_inv;
  vz = C1[3*n_cells+id]*d_inv;
  P  = (C1[4*n_cells+id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gama - 1.0);
  vx_imo = C1[1*n_cells + imo] / C1[imo]; 
  vx_ipo = C1[1*n_cells + ipo] / C1[ipo]; 
  vy_jmo = C1[2*n_cells + jmo] / C1[jmo]; 
  vy_jpo = C1[2*n_cells + jpo] / C1[jpo]; 
  vz_kmo = C1[3*n_cells + kmo] / C1[kmo]; 
  vz_kpo = C1[3*n_cells + kpo] / C1[kpo]; 
  #endif

  Real cW[2*nfields];
  Real F_Lx[nfields];
  Real F_Rx[nfields];
  Real F_Ly[nfields];
  Real F_Ry[nfields];
  Real F_Lz[nfields];
  Real F_Rz[nfields];

  // using piecewise constant reconstruction,
  // calculate the first set of fluxes

  // Lx
  fill_flux_array(C1, imo, id, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Lx, gama, etah, 0);
  #endif
  
  // Rx
  fill_flux_array(C1, id, ipo, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Rx, gama, etah, 0);
  #endif

  // Ly
  fill_flux_array(C1, jmo, id, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ly, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ly, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ly, gama, etah, 1);
  #endif

  // Ry
  fill_flux_array(C1, id, jpo, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ry, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ry, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ry, gama, etah, 1);
  #endif

  // Lz
  fill_flux_array(C1, kmo, id, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lz, gama, etah);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes(cW, F_Lz, gama, etah, 2);
  #endif

  // Rz
  fill_flux_array(C1, id, kpo, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rz, gama, etah);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes(cW, F_Rz, gama, etah, 2);
  #endif
  for (int ii=0; ii<nfields; ii++) {
    if (F_Lx[ii] != F_Lx[ii]) printf("Failure in Riemann solve F_Lx[%d]\n", ii);
    if (F_Rx[ii] != F_Rx[ii]) printf("Failure in Riemann solve F_Rx[%d]\n", ii);
    if (F_Ly[ii] != F_Ly[ii]) printf("Failure in Riemann solve F_Ly[%d]\n", ii);
    if (F_Ry[ii] != F_Ry[ii]) printf("Failure in Riemann solve F_Ry[%d]\n", ii);
    if (F_Lz[ii] != F_Lz[ii]) printf("Failure in Riemann solve F_Lz[%d]\n", ii);
    if (F_Rz[ii] != F_Rz[ii]) printf("Failure in Riemann solve F_Rz[%d]\n", ii);
  }

  // Update the conserved variables for the cell by a half step
  C_half[0] = C1[id+0*n_cells] + 0.5*(dtodx*(F_Lx[0] - F_Rx[0]) + dtody*(F_Ly[0] - F_Ry[0]) + dtodz*(F_Lz[0] - F_Rz[0]));
  C_half[1] = C1[id+1*n_cells] + 0.5*(dtodx*(F_Lx[1] - F_Rx[1]) + dtody*(F_Ly[1] - F_Ry[1]) + dtodz*(F_Lz[1] - F_Rz[1]));
  C_half[2] = C1[id+2*n_cells] + 0.5*(dtodx*(F_Lx[2] - F_Rx[2]) + dtody*(F_Ly[2] - F_Ry[2]) + dtodz*(F_Lz[2] - F_Rz[2]));
  C_half[3] = C1[id+3*n_cells] + 0.5*(dtodx*(F_Lx[3] - F_Rx[3]) + dtody*(F_Ly[3] - F_Ry[3]) + dtodz*(F_Lz[3] - F_Rz[3]));
  C_half[4] = C1[id+4*n_cells] + 0.5*(dtodx*(F_Lx[4] - F_Rx[4]) + dtody*(F_Ly[4] - F_Ry[4]) + dtodz*(F_Lz[4] - F_Rz[4]));
  #ifdef DE
  C_half[5] = C1[id+5*n_cells] + 0.5*(dtodx*(F_Lx[5] - F_Rx[5]) + dtody*(F_Ly[5] - F_Ry[5]) + dtodz*(F_Lz[5] - F_Rz[5])
            + P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo)));
  #endif


}



void full_step_update(Real *C1, Real *C2, int i, int j, int k, Real dtodx, Real dtody, Real dtodz, int nfields, int nx, int ny, int nz, int n_cells, Real C_half[], Real C_half_imo[], Real C_half_ipo[], Real C_half_jmo[], Real C_half_jpo[], Real C_half_kmo[], Real C_half_kpo[])
{
  int id = i + j*nx + k*nx*ny;
  int imo = i-1 + j*nx + k*nx*ny;
  int ipo = i+1 + j*nx + k*nx*ny;
  int jmo = i + (j-1)*nx + k*nx*ny;
  int jpo = i + (j+1)*nx + k*nx*ny;
  int kmo = i + j*nx + (k-1)*nx*ny;
  int kpo = i + j*nx + (k+1)*nx*ny;
  Real etah = 0.0;

  Real cW[2*nfields];
  Real F_Lx[nfields];
  Real F_Rx[nfields];
  Real F_Ly[nfields];
  Real F_Ry[nfields];
  Real F_Lz[nfields];
  Real F_Rz[nfields];

  #ifdef DE
  Real d, d_inv, vx, vy, vz, P, vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo;
  d = C1[id];
  d_inv = 1.0 / d;
  vx = C1[1*n_cells+id]*d_inv;
  vy = C1[2*n_cells+id]*d_inv;
  vz = C1[3*n_cells+id]*d_inv;
  P  = (C1[4*n_cells+id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gama - 1.0);
  vx_imo = C1[1*n_cells + imo] / C1[imo]; 
  vx_ipo = C1[1*n_cells + ipo] / C1[ipo]; 
  vy_jmo = C1[2*n_cells + jmo] / C1[jmo]; 
  vy_jpo = C1[2*n_cells + jpo] / C1[jpo]; 
  vz_kmo = C1[3*n_cells + kmo] / C1[kmo]; 
  vz_kpo = C1[3*n_cells + kpo] / C1[kpo]; 
  #endif


  // using piecewise constant reconstruction,
  // calculate the second set of fluxes

  // Lx
  fill_flux_array_2(C_half_imo, C_half, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Lx, gama, etah, 0);
  #endif
  
  // Rx
  fill_flux_array_2(C_half, C_half_ipo, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Rx, gama, etah, 0);
  #endif

  // Ly
  fill_flux_array_2(C_half_jmo, C_half, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ly, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ly, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ly, gama, etah, 1);
  #endif

  // Ry
  fill_flux_array_2(C_half, C_half_jpo, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ry, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ry, gama, etah);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes(cW, F_Ry, gama, etah, 1);
  #endif

  // Lz
  fill_flux_array_2(C_half_kmo, C_half, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lz, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Lz, gama, etah, 2);
  #endif

  // Rz
  fill_flux_array_2(C_half, C_half_kpo, cW, n_cells);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rz, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Rz, gama, etah, 2);
  #endif


  // Update the conserved variables for the cell by a full step
  C2[id+0*n_cells] = C1[id+0*n_cells] + dtodx*(F_Lx[0] - F_Rx[0]) + dtody*(F_Ly[0] - F_Ry[0]) + dtodz*(F_Lz[0] - F_Rz[0]);
  C2[id+1*n_cells] = C1[id+1*n_cells] + dtodx*(F_Lx[1] - F_Rx[1]) + dtody*(F_Ly[1] - F_Ry[1]) + dtodz*(F_Lz[1] - F_Rz[1]);
  C2[id+2*n_cells] = C1[id+2*n_cells] + dtodx*(F_Lx[2] - F_Rx[2]) + dtody*(F_Ly[2] - F_Ry[2]) + dtodz*(F_Lz[2] - F_Rz[2]);
  C2[id+3*n_cells] = C1[id+3*n_cells] + dtodx*(F_Lx[3] - F_Rx[3]) + dtody*(F_Ly[3] - F_Ry[3]) + dtodz*(F_Lz[3] - F_Rz[3]);
  C2[id+4*n_cells] = C1[id+4*n_cells] + dtodx*(F_Lx[4] - F_Rx[4]) + dtody*(F_Ly[4] - F_Ry[4]) + dtodz*(F_Lz[4] - F_Rz[4]);
  #ifdef DE
  C2[id+5*n_cells] = C1[id+5*n_cells] + dtodx*(F_Lx[5] - F_Rx[5]) + dtody*(F_Ly[5] - F_Ry[5]) + dtodz*(F_Lz[5] - F_Rz[5])
                   + 0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));
  #endif


}


void calc_g_3D(int xid, int yid, int zid, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real *gx, Real *gy, Real *gz)
{
  Real x_pos, y_pos, z_pos, r_disk, r_halo;
  // use the offsets and global boundaries to calculate absolute positions on the grid
  x_pos = (x_off + xid - n_ghost + 0.5)*dx + xbound;
  y_pos = (y_off + yid - n_ghost + 0.5)*dy + ybound;
  z_pos = (z_off + zid - n_ghost + 0.5)*dz + zbound;

  // for disk components, calculate polar r
  r_disk = sqrt(x_pos*x_pos + y_pos*y_pos);
  // for halo, calculate spherical r
  r_halo = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);

  // set properties of halo and disk (these must match initial conditions)
  Real a_disk_r, a_disk_z, a_halo, a_halo_r, a_halo_z;
  Real M_vir, M_d, R_vir, R_d, z_d, R_h, M_h, c_vir, phi_0_h, x;
  M_vir = 5.0e10; // viral mass of in M_sun
  M_d = 1.0e10; // mass of disk in M_sun
  M_h = M_vir - M_d; // halo mass in M_sun
  R_d = 0.8; // disk scale length in kpc
  z_d = 0.15; // disk scale height in kpc
  R_vir = R_d/0.015; // viral radius in kpc
  c_vir = 10.0; // halo concentration
  R_h = R_vir / c_vir; // halo scale length in kpc
  phi_0_h = GN * M_h / (log(1.0+c_vir) - c_vir / (1.0+c_vir));
  x = r_halo / R_h;
  
  // calculate acceleration due to NFW halo & Miyamoto-Nagai disk
  a_halo = - phi_0_h * (log(1+x) - x/(1+x)) / (r_halo*r_halo);
  a_halo_r = a_halo*(r_disk/r_halo);
  a_halo_z = a_halo*(z_pos/r_halo);
  a_disk_r = - GN * M_d * r_disk * pow(r_disk*r_disk+ pow(R_d + sqrt(z_pos*z_pos + z_d*z_d),2), -1.5);
  a_disk_z = - GN * M_d * z_pos * (R_d + sqrt(z_pos*z_pos + z_d*z_d)) / ( pow(r_disk*r_disk + pow(R_d + sqrt(z_pos*z_pos + z_d*z_d), 2), 1.5) * sqrt(z_pos*z_pos + z_d*z_d) );

  // total acceleration is the sum of the halo + disk components
  *gx = (x_pos/r_disk)*(a_disk_r+a_halo_r);
  *gy = (y_pos/r_disk)*(a_disk_r+a_halo_r);
  *gz = a_disk_z+a_halo_z;

}


void cooling_CPU(Real *C2, int id, int n_cells, Real dt) {

  Real d, E;
  Real n, T, T_init;
  Real del_T, dt_sub;
  Real mu; // mean molecular weight
  Real cool; //cooling rate per volume, erg/s/cm^3
  #ifndef DE
  Real vx, vy, vz, p;
  #endif
  #ifdef DE
  Real ge;
  #endif

  //Real T_min = 1.0e4;
  Real T_min = 0.0;
  mu = 0.6;

  // load values of density and pressure
  d  =  C2[            id];
  E  =  C2[4*n_cells + id];
  #ifndef DE
  vx =  C2[1*n_cells + id] / d;
  vy =  C2[2*n_cells + id] / d;
  vz =  C2[3*n_cells + id] / d;
  p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gama - 1.0);
  p  = fmax(p, (Real) TINY_NUMBER);
  #endif
  #ifdef DE
  ge = C2[5*n_cells + id] / d;
  ge = fmax(ge, (Real) TINY_NUMBER);
  #endif
    
  // calculate the number density of the gas (in cgs)
  n = d*DENSITY_UNIT / (mu * MP);

  // calculate the temperature of the gas
  #ifndef DE
  T_init = p*PRESSURE_UNIT/ (n*KB);
  #endif
  #ifdef DE
  T_init = ge*(gama-1.0)*SP_ENERGY_UNIT*mu*MP/KB;
  #endif

  // calculate cooling rate per volume
  T = T_init;

  // call the cooling function
  //cool = Schure_cool_CPU(n, T); 
  cool = Wiersma_cool_CPU(n, T); 
    
  // calculate change in temperature given dt
  del_T = cool*dt*TIME_UNIT*(gama-1.0)/(n*KB);

  // limit change in temperature to 5%
  while (del_T/T > 0.05) {
    // what dt gives del_T = 0.1*T?
    dt_sub = 0.05*T*n*KB/(cool*TIME_UNIT*(gama-1.0));
    // apply that dt
    T -= cool*dt_sub*TIME_UNIT*(gama-1.0)/(n*KB);
    // how much time is left from the original timestep?
    dt -= dt_sub;
    // calculate cooling again
    //cool = Schure_cool_CPU(n, T);
    cool = Wiersma_cool_CPU(n, T);
    // calculate new change in temperature
    del_T = cool*dt*TIME_UNIT*(gama-1.0)/(n*KB);
  }

  // calculate final temperature
  T -= del_T;

  if (T < T_min) {
    T = T_min;
  }

  // adjust value of energy based on total change in temperature
  del_T = T_init - T; // total change in T
  E -= n*KB*del_T / ((gama-1.0)*ENERGY_UNIT);
  #ifdef DE
  ge -= KB*del_T / (mu*MP*(gama-1.0)*SP_ENERGY_UNIT);
  #endif

  // and update the energies 
  C2[4*n_cells + id] = E;
  #ifdef DE
  C2[5*n_cells + id] = d*ge;
  #endif


}


Real Schure_cool_CPU(Real n, Real T) {

  Real lambda = 0.0; //cooling rate, erg s^-1 cm^3
  Real cool = 0.0; //cooling per unit volume, erg /s / cm^3
  
  // fit to Schure cooling function 
  if (log10(T) > 5.36) {
    lambda = pow(10.0, (0.38 * (log10(T) -7.5) * (log10(T) - 7.5) - 22.6));
  }
  else if (log10(T) < 4.0) {
    lambda = 0.0;
  }
  else {
    lambda = pow(10.0, (-2.5 * (log10(T) - 5.1) * (log10(T) - 5.1) - 20.7));
  }

  // cooling rate per unit volume
  cool = n*n*lambda;

  return cool;

}

Real Wiersma_cool_CPU(Real n, Real T) {

  Real lambda = 0.0; //cooling rate, erg s^-1 cm^3
  Real cool = 0.0; //cooling per unit volume, erg /s / cm^3
  
  // fit to Wiersma 2009 CIE cooling function 
  if (log10(T) < 4.0) {
    lambda = 0.0;
  }
  else if (log10(T) >= 4.0 && log10(T) < 5.9) {
    lambda = pow(10.0, (-1.3 * (log10(T) - 5.25) * (log10(T) - 5.25) - 21.25));
  }
  else if (log10(T) >= 5.9 && log10(T) < 7.4) {
    lambda = pow(10.0, (0.7 * (log10(T) - 7.1) * (log10(T) - 7.1) - 22.8));
  }
  else {
    lambda = pow(10.0, (0.45*log10(T) - 26.065));
  }

  // cooling rate per unit volume
  cool = n*n*lambda;

  return cool;

}

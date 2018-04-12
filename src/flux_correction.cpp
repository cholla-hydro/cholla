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
#include"plmc.h"
#include"io.h"
#include"error_handling.h"
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
  nfields++;
  #endif

  Real d_old, vx_old, vy_old, vz_old, P_old, E_old;
  Real d_new, vx_new, vy_new, vz_new, P_new, E_new;
  Real n, T;
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
        E_new = C2[4*n_cells+id];
        vx_new = C2[1*n_cells+id]/d_new;
        vy_new = C2[2*n_cells+id]/d_new;
        vz_new = C2[3*n_cells+id]/d_new;
        P_new = (E_new - 0.5*d_new*(vx_new*vx_new + vy_new*vy_new + vz_new*vz_new))*(gama-1.0);
        n = d_new*DENSITY_UNIT/(0.6*MP);
        #ifdef DE
        T = C2[(nfields-1)*n_cells+id]*(gama-1.0)*PRESSURE_UNIT/(n*KB);
        #else
        T = P_new*PRESSURE_UNIT/(n*KB);
        #endif
  
        // if there is a problem, redo the update for that cell using first-order fluxes
        //if (d_new < 0.0 || d_new != d_new || P_new < 0.0 || P_new != P_new || E_new < 0.0 || E_new != E_new || T > 1.0e9) {
        if (d_new < 0.0 || d_new != d_new || P_new < 0.0 || P_new != P_new || E_new < 0.0 || E_new != E_new) {
          printf("%3d %3d %3d BC: d: %e  E:%e  P:%e  T:%e\n", i+nx_local_start, j+ny_local_start, k+nz_local_start, d_new, E_new, P_new, T);
          //printf("%3d %3d %3d BC: d: %e  E:%e  P:%e  T:%e\n", i, j, k, d_new, E_new, P_new, T);

/*
          // Update the conserved variables for the affected cell 
          // using first-order fluxes (nonconservative update)
          first_order_fluxes(C1, C2, i, j, k, dtodx, dtody, dtodz, nfields, nx, ny, nz, n_cells);

          // Reset with the new values of the conserved variables
          d_new = C2[id];
          E_new = C2[4*n_cells+id];
          vx_new = C2[1*n_cells+id]/d_new;
          vy_new = C2[2*n_cells+id]/d_new;
          vz_new = C2[3*n_cells+id]/d_new;
          P_new = (E_new - 0.5*d_new*(vx_new*vx_new + vy_new*vy_new + vz_new*vz_new))*(gama-1.0);
          n = d_new*DENSITY_UNIT/(0.6*MP);
          #ifdef DE
          T = C2[5*n_cells+id]*(gama-1.0)*PRESSURE_UNIT/(n*KB);
          #else
          T = P_new*PRESSURE_UNIT/(n*KB);
          #endif

          // if there is STILL a problem, average over surrounding cells
          if (d_new < 0.0 || d_new != d_new || P_new < 0.0 || P_new != P_new || E_new < 0.0 || E_new != E_new || T > 1.0e9) {
            printf("%3d %3d %3d Averaging: d: %e  E:%e  P:%e  T:%e\n", i+nx_local_start, j+ny_local_start, k+nz_local_start, d_new, E_new, P_new, T);
*/
          average_cell(C2, i, j, k, nx, ny, nz, n_cells);
          d_new = C2[id];
          E_new = C2[4*n_cells+id];
          vx_new = C2[1*n_cells+id]/d_new;
          vy_new = C2[2*n_cells+id]/d_new;
          vz_new = C2[3*n_cells+id]/d_new;
          P_new = (E_new - 0.5*d_new*(vx_new*vx_new + vy_new*vy_new + vz_new*vz_new))*(gama-1.0);
          n = d_new*DENSITY_UNIT/(0.6*MP);
          #ifdef DE
          T = C2[(nfields-1)*n_cells+id]*(gama-1.0)*PRESSURE_UNIT/(n*KB);
          #else
          T = P_new*PRESSURE_UNIT/(n*KB);
          #endif
          //}

          // Apply gravity
          #ifdef STATIC_GRAV
          Real gx, gy, gz;
          gx = gy = gz = 0.0;
          calc_g_3D(i, j, k, x_off, y_off, z_off, n_ghost, dx, dy, dz, xbound, ybound, zbound, &gx, &gy, &gz);
          d_old = C1[id];
          vx_old = C1[1*n_cells+id]/d_old;
          vy_old = C1[2*n_cells+id]/d_old;
          vz_old = C1[3*n_cells+id]/d_old;
          C2[1*n_cells + id] += 0.5*dt*gx*(d_old + d_new);
          C2[2*n_cells + id] += 0.5*dt*gy*(d_old + d_new);
          C2[3*n_cells + id] += 0.5*dt*gz*(d_old + d_new);
          C2[4*n_cells + id] += 0.25*dt*gx*(d_old + d_new)*(vx_old + vx_new) 
                              + 0.25*dt*gy*(d_old + d_new)*(vy_old + vy_new)
                              + 0.25*dt*gz*(d_old + d_new)*(vz_old + vz_new);
          // Reset with the new values of the conserved variables
          vx_new = C2[1*n_cells+id]/d_new;
          vy_new = C2[2*n_cells+id]/d_new;
          vz_new = C2[3*n_cells+id]/d_new;
          E_new  = C2[4*n_cells+id];
          P_new = (E_new - 0.5*d_new*(vx_new*vx_new + vy_new*vy_new + vz_new*vz_new))*(gama-1.0);
          //T_c = P_new*PRESSURE_UNIT / (n*KB);
          //T_ie = C2[5*n_cells+id]*(gama-1.0)*PRESSURE_UNIT / (n*KB);
          //printf("%3d %3d %3d After gravity d: %e  P:%e  T_cons: %e  T_ie: %e\n", i+nx_local_start, j+ny_local_start, k+nz_local_start, d_new, P_new, T_c, T_ie);
          #endif
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
          // is greater than 1/1000,
          // use the conservatively calculated internal energy to do the internal energy update
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
          else {
            // sync the total energy with the internal energy 
            C2[4*n_cells + id] += ge1 - ge2;
          }
          //C2[4*n_cells + id] += ge1 - ge2;
          #endif          

          // apply cooling
          #ifdef COOLING_GPU
          //cooling_CPU(C2, id, n_cells, dt);
          #endif

          // recalculate the pressure
          E_new = C2[4*n_cells + id];
          P_new = (E_new - 0.5*d_new*(vx_new*vx_new + vy_new*vy_new + vz_new*vz_new))*(gama-1.0);
          // recalculate the temperature
          Real n = d_new*DENSITY_UNIT / (1.27 * MP);
          Real T = P_new*PRESSURE_UNIT/(n*KB);
          //printf("%3d %3d %3d FC  d: %e  E:%e  P:%e  T:%e\n", i+nx_local_start, j+ny_local_start, k+nz_local_start, d_new, E_new, P_new, T);
          //printf("%3d %3d %3d FC  d: %e  E:%e  P:%e  T:%e\n", i, j, k, d_new, E_new, P_new, T);
          //if (d_new < 0.0 || d_new != d_new || P_new < 0.0 || P_new != P_new) printf("FLUX CORRECTION FAILED: %d %d %d %e %e\n", i+nx_local_start, j+ny_local_start, k+nz_local_start, d_new, P_new);
          //if (d_new < 0.0 || d_new != d_new || P_new < 0.0 || P_new != P_new) printf("FLUX CORRECTION FAILED: %d %d %d %e %e\n", i, j, k, d_new, P_new);
          if (d_new < 0.0 || d_new != d_new || E_new < 0.0 || E_new != E_new) exit(-1);
          
        }

      }
    }
  }


}


void fill_flux_array_pcm(Real *C1, int idl, int idr, Real cW[], int n_cells, int dir)
{

  cW[0] = C1[idl];
  cW[1] = C1[idr];
  if (dir == 0) {
    cW[2] = C1[1*n_cells+idl];
    cW[3] = C1[1*n_cells+idr];
    cW[4] = C1[2*n_cells+idl];
    cW[5] = C1[2*n_cells+idr];
    cW[6] = C1[3*n_cells+idl];
    cW[7] = C1[3*n_cells+idr];
  }
  if (dir == 1) {
    cW[2] = C1[2*n_cells+idl];
    cW[3] = C1[2*n_cells+idr];
    cW[4] = C1[3*n_cells+idl];
    cW[5] = C1[3*n_cells+idr];
    cW[6] = C1[1*n_cells+idl];
    cW[7] = C1[1*n_cells+idr];
  }
  if (dir == 2) {
    cW[2] = C1[3*n_cells+idl];
    cW[3] = C1[3*n_cells+idr];
    cW[4] = C1[1*n_cells+idl];
    cW[5] = C1[1*n_cells+idr];
    cW[6] = C1[2*n_cells+idl];
    cW[7] = C1[2*n_cells+idr];
  }
  cW[8] = C1[4*n_cells+idl];
  cW[9] = C1[4*n_cells+idr];
  #ifdef DE
  cW[10] = C1[5*n_cells+idl];
  cW[11] = C1[5*n_cells+idr];
  #endif

}



void second_order_fluxes(Real *C1, Real *C2, Real C_i[], Real C_imo[], Real C_imt[], Real C_ipo[], Real C_ipt[], Real C_jmo[], Real C_jmt[], Real C_jpo[], Real C_jpt[], Real C_kmo[], Real C_kmt[], Real C_kpo[], Real C_kpt[], int i, int j, int k, Real dx, Real dy, Real dz, Real dt, int n_fields, int nx, int ny, int nz, int n_cells)
{
  int id = i + j*nx + k*nx*ny;
  int imo = i-1 + j*nx + k*nx*ny;
  int imt = i-2 + j*nx + k*nx*ny;
  int ipo = i+1 + j*nx + k*nx*ny;
  int ipt = i+2 + j*nx + k*nx*ny;
  int jmo = i + (j-1)*nx + k*nx*ny;
  int jmt = i + (j-2)*nx + k*nx*ny;
  int jpo = i + (j+1)*nx + k*nx*ny;
  int jpt = i + (j+2)*nx + k*nx*ny;
  int kmo = i + j*nx + (k-1)*nx*ny;
  int kmt = i + j*nx + (k-2)*nx*ny;
  int kpo = i + j*nx + (k+1)*nx*ny;
  int kpt = i + j*nx + (k+2)*nx*ny;
  Real etah = 0.0;
  Real dtodx = dt / dx;
  Real dtody = dt / dy;
  Real dtodz = dt / dz;

  // Arrays to hold second-order fluxes
  Real F_Lx[n_fields];
  Real F_Rx[n_fields];
  Real F_Ly[n_fields];
  Real F_Ry[n_fields];
  Real F_Lz[n_fields];
  Real F_Rz[n_fields];

  // Array to hold stencil
  Real stencil[3*n_fields];
  // Array to hold temporary boundary values returned from reconstruction
  Real bounds[2*n_fields];
  // Array to hold temporary interface values
  Real cW[2*n_fields];


  /********** LX INTERFACE ***********/
  // fill the stencil for the x-direction, cell i-1
  stencil[0] = C_imo[0]; 
  stencil[1] = C_imo[1];
  stencil[2] = C_imo[2];
  stencil[3] = C_imo[3];
  stencil[4] = C_imo[4];
  stencil[5] = C_imt[0];
  stencil[6] = C_imt[1];
  stencil[7] = C_imt[2];
  stencil[8] = C_imt[3];
  stencil[9] = C_imt[4];
  stencil[10] = C_i[0];
  stencil[11] = C_i[1];
  stencil[12] = C_i[2];
  stencil[13] = C_i[3];
  stencil[14] = C_i[4];
  #ifdef DE
  stencil[15] = C_imo[5];
  stencil[16] = C_imt[5];
  stencil[17] = C_i[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dx, dt, gama);
  #endif

  // put the boundary values in the temporary reconstruction array
  cW[0] = bounds[5];
  cW[2] = bounds[6];
  cW[4] = bounds[7];
  cW[6] = bounds[8];
  cW[8] = bounds[9];
  #ifdef DE
  cW[10] = bounds[11];
  #endif

  // fill the stencil for the x-direction, cell i
  stencil[0] = C_i[0]; 
  stencil[1] = C_i[1];
  stencil[2] = C_i[2];
  stencil[3] = C_i[3];
  stencil[4] = C_i[4];
  stencil[5] = C_imo[0];
  stencil[6] = C_imo[1];
  stencil[7] = C_imo[2];
  stencil[8] = C_imo[3];
  stencil[9] = C_imo[4];
  stencil[10] = C_ipo[0];
  stencil[11] = C_ipo[1];
  stencil[12] = C_ipo[2];
  stencil[13] = C_ipo[3];
  stencil[14] = C_ipo[4];
  #ifdef DE
  stencil[15] = C_i[5];
  stencil[16] = C_imo[5];
  stencil[17] = C_ipo[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dx, dt, gama);
  #endif

  // put the boundary values in the temporary reconstruction array
  cW[1] = bounds[0];
  cW[3] = bounds[1];
  cW[5] = bounds[2];
  cW[7] = bounds[3];
  cW[9] = bounds[4];
  #ifdef DE
  cW[11] = bounds[10];
  #endif

  // calculate the fluxes
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Lx, gama, etah);
  #endif


  /********** RX INTERFACE ***********/
  // put the boundary values in the temporary reconstruction array
  // (re-using values calculated for cell i here)
  cW[0] = bounds[5];
  cW[2] = bounds[6];
  cW[4] = bounds[7];
  cW[6] = bounds[8];
  cW[8] = bounds[9];
  #ifdef DE
  cW[10] = bounds[11];
  #endif

  // fill the stencil for the x-direction, cell i+1
  stencil[0] = C_ipo[0]; 
  stencil[1] = C_ipo[1];
  stencil[2] = C_ipo[2];
  stencil[3] = C_ipo[3];
  stencil[4] = C_ipo[4];
  stencil[5] = C_i[0];
  stencil[6] = C_i[1];
  stencil[7] = C_i[2];
  stencil[8] = C_i[3];
  stencil[9] = C_i[4];
  stencil[10] = C_ipt[0];
  stencil[11] = C_ipt[1];
  stencil[12] = C_ipt[2];
  stencil[13] = C_ipt[3];
  stencil[14] = C_ipt[4];
  #ifdef DE
  stencil[15] = C_ipo[5];
  stencil[16] = C_i[5];
  stencil[17] = C_ipt[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dx, dt, gama);
  #endif
  
  // put the boundary values in the temporary reconstruction array
  cW[1] = bounds[0];
  cW[3] = bounds[1];
  cW[5] = bounds[2];
  cW[7] = bounds[3];
  cW[9] = bounds[4];
  #ifdef DE
  cW[11] = bounds[10];
  #endif

  // calculate the fluxes
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Rx, gama, etah);
  #endif

  /********** LY INTERFACE ***********/
  // fill the stencil for the y-direction, cell j-1
  stencil[0] = C_jmo[0]; 
  stencil[1] = C_jmo[2];
  stencil[2] = C_jmo[3];
  stencil[3] = C_jmo[1];
  stencil[4] = C_jmo[4];
  stencil[5] = C_jmt[0];
  stencil[6] = C_jmt[2];
  stencil[7] = C_jmt[3];
  stencil[8] = C_jmt[1];
  stencil[9] = C_jmt[4];
  stencil[10] = C_i[0];
  stencil[11] = C_i[2];
  stencil[12] = C_i[3];
  stencil[13] = C_i[1];
  stencil[14] = C_i[4];
  #ifdef DE
  stencil[15] = C_jmo[5];
  stencil[16] = C_jmt[5];
  stencil[17] = C_i[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dy, dt, gama);
  #endif
  // put the boundary values in the temporary reconstruction array
  cW[0] = bounds[5];
  cW[2] = bounds[6];
  cW[4] = bounds[7];
  cW[6] = bounds[8];
  cW[8] = bounds[9];
  #ifdef DE
  cW[10] = bounds[11];
  #endif

  // fill the stencil for the y-direction, cell i 
  stencil[0] = C_i[0]; 
  stencil[1] = C_i[2];
  stencil[2] = C_i[3];
  stencil[3] = C_i[1];
  stencil[4] = C_i[4];
  stencil[5] = C_jmo[0];
  stencil[6] = C_jmo[2];
  stencil[7] = C_jmo[3];
  stencil[8] = C_jmo[1];
  stencil[9] = C_jmo[4];
  stencil[10] = C_jpo[0];
  stencil[11] = C_jpo[2];
  stencil[12] = C_jpo[3];
  stencil[13] = C_jpo[1];
  stencil[14] = C_jpo[4];
  #ifdef DE
  stencil[15] = C_i[5];
  stencil[16] = C_jmo[5];
  stencil[17] = C_jpo[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dy, dt, gama);
  #endif

  // put the boundary values in the temporary reconstruction array
  cW[1] = bounds[0];
  cW[3] = bounds[1];
  cW[5] = bounds[2];
  cW[7] = bounds[3];
  cW[9] = bounds[4];
  #ifdef DE
  cW[11] = bounds[10];
  #endif
  
  // calculate the fluxes
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ly, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ly, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ly, gama, etah);
  #endif


  /********** RY INTERFACE ***********/
  // put the boundary values in the temporary reconstruction array
  // (re-using values calculated for cell i here)
  cW[0] = bounds[5];
  cW[2] = bounds[6];
  cW[4] = bounds[7];
  cW[6] = bounds[8];
  cW[8] = bounds[9];
  #ifdef DE
  cW[10] = bounds[11];
  #endif

  // fill the stencil for the y-direction, cell j+1
  stencil[0] = C_jpo[0]; 
  stencil[1] = C_jpo[2];
  stencil[2] = C_jpo[3];
  stencil[3] = C_jpo[1];
  stencil[4] = C_jpo[4];
  stencil[5] = C_i[0];
  stencil[6] = C_i[2];
  stencil[7] = C_i[3];
  stencil[8] = C_i[1];
  stencil[9] = C_i[4];
  stencil[10] = C_jpt[0];
  stencil[11] = C_jpt[2];
  stencil[12] = C_jpt[3];
  stencil[13] = C_jpt[1];
  stencil[14] = C_jpt[4];
  #ifdef DE
  stencil[15] = C_jpo[5];
  stencil[16] = C_i[5];
  stencil[17] = C_jpt[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dy, dt, gama);
  #endif

  // put the boundary values in the temporary reconstruction array
  cW[1] = bounds[0];
  cW[3] = bounds[1];
  cW[5] = bounds[2];
  cW[7] = bounds[3];
  cW[9] = bounds[4];
  #ifdef DE
  cW[11] = bounds[10];
  #endif

  // calculate the fluxes
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ry, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ry, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ry, gama, etah);
  #endif
  

  /********** LZ INTERFACE ***********/
  // fill the stencil for the z-direction, k-1
  stencil[0] = C_kmo[0]; 
  stencil[1] = C_kmo[3];
  stencil[2] = C_kmo[1];
  stencil[3] = C_kmo[2];
  stencil[4] = C_kmo[4];
  stencil[5] = C_kmt[0];
  stencil[6] = C_kmt[3];
  stencil[7] = C_kmt[1];
  stencil[8] = C_kmt[2];
  stencil[9] = C_kmt[4];
  stencil[10] = C_i[0];
  stencil[11] = C_i[3];
  stencil[12] = C_i[1];
  stencil[13] = C_i[2];
  stencil[14] = C_i[4];
  #ifdef DE
  stencil[15] = C_kmo[5];
  stencil[16] = C_kmt[5];
  stencil[17] = C_i[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dz, dt, gama);
  #endif

  // put the boundary values in the temporary reconstruction array
  cW[0] = bounds[5];
  cW[2] = bounds[6];
  cW[4] = bounds[7];
  cW[6] = bounds[8];
  cW[8] = bounds[9];
  #ifdef DE
  cW[10] = bounds[11];
  #endif

  // fill the stencil for the z-direction, cell i 
  stencil[0] = C_i[0]; 
  stencil[1] = C_i[3];
  stencil[2] = C_i[1];
  stencil[3] = C_i[2];
  stencil[4] = C_i[4];
  stencil[5] = C_kmo[0];
  stencil[6] = C_kmo[3];
  stencil[7] = C_kmo[1];
  stencil[8] = C_kmo[2];
  stencil[9] = C_kmo[4];
  stencil[10] = C_kpo[0];
  stencil[11] = C_kpo[3];
  stencil[12] = C_kpo[1];
  stencil[13] = C_kpo[2];
  stencil[14] = C_kpo[4];
  #ifdef DE
  stencil[15] = C_i[5];
  stencil[16] = C_kmo[5];
  stencil[17] = C_kpo[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dz, dt, gama);
  #endif

  // put the boundary values in the temporary reconstruction array
  cW[1] = bounds[0];
  cW[3] = bounds[1];
  cW[5] = bounds[2];
  cW[7] = bounds[3];
  cW[9] = bounds[4];
  #ifdef DE
  cW[11] = bounds[10];
  #endif
  
  // calculate the fluxes
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lz, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Lz, gama, etah);
  #endif


  /********** RZ INTERFACE ***********/
  // put the boundary values in the temporary reconstruction array
  // (re-using values calculated for cell i here)
  cW[0] = bounds[5];
  cW[2] = bounds[6];
  cW[4] = bounds[7];
  cW[6] = bounds[8];
  cW[8] = bounds[9];
  #ifdef DE
  cW[10] = bounds[11];
  #endif

  // fill the stencil for the z-direction, cell k+1
  stencil[0] = C_kpo[0]; 
  stencil[1] = C_kpo[3];
  stencil[2] = C_kpo[1];
  stencil[3] = C_kpo[2];
  stencil[4] = C_kpo[4];
  stencil[5] = C_i[0];
  stencil[6] = C_i[3];
  stencil[7] = C_i[1];
  stencil[8] = C_i[2];
  stencil[9] = C_i[4];
  stencil[10] = C_kpt[0];
  stencil[11] = C_kpt[3];
  stencil[12] = C_kpt[1];
  stencil[13] = C_kpt[2];
  stencil[14] = C_kpt[4];
  #ifdef DE
  stencil[15] = C_kpo[5];
  stencil[16] = C_i[5];
  stencil[17] = C_kpt[5];
  #endif

  // pass the stencil to the linear reconstruction function - returns the reconstructed left
  // and right boundary values for the cell (conserved variables)
  #ifdef PLMC
  plmc(stencil, bounds, dz, dt, gama);
  #endif

  // put the boundary values in the temporary reconstruction array
  cW[1] = bounds[0];
  cW[3] = bounds[1];
  cW[5] = bounds[2];
  cW[7] = bounds[3];
  cW[9] = bounds[4];
  #ifdef DE
  cW[11] = bounds[10];
  #endif

  // calculate the fluxes
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rz, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Rz, gama, etah);
  #endif
  
  // subtract the relevant second-order fluxes from each of the surrounding cells
  // cell i-1
  C2[imo+0*n_cells] -= dtodx*(-F_Rx[0]);
  C2[imo+1*n_cells] -= dtodx*(-F_Rx[1]);
  C2[imo+2*n_cells] -= dtodx*(-F_Rx[2]);
  C2[imo+3*n_cells] -= dtodx*(-F_Rx[3]);
  C2[imo+4*n_cells] -= dtodx*(-F_Rx[4]);
  #ifdef DE
  C2[id+5*n_cells] -= dtodx*(-F_Rx[5]);
  #endif
  // cell i+1
  C2[ipo+0*n_cells] -= dtodx*(F_Lx[0]);
  C2[ipo+1*n_cells] -= dtodx*(F_Lx[1]);
  C2[ipo+2*n_cells] -= dtodx*(F_Lx[2]);
  C2[ipo+3*n_cells] -= dtodx*(F_Lx[3]);
  C2[ipo+4*n_cells] -= dtodx*(F_Lx[4]);
  #ifdef DE
  C2[ipo+5*n_cells] -= dtodx*(F_Lx[5]);
  #endif
  // cell j-1
  C2[jmo+0*n_cells] -= dtody*(- F_Ry[0]);
  C2[jmo+1*n_cells] -= dtody*(- F_Ry[3]);
  C2[jmo+2*n_cells] -= dtody*(- F_Ry[1]);
  C2[jmo+3*n_cells] -= dtody*(- F_Ry[2]);
  C2[jmo+4*n_cells] -= dtody*(- F_Ry[4]);
  #ifdef DE
  C2[jmo+5*n_cells] -= dtody*(- F_Ry[5]);
  #endif
  // cell j+1
  C2[jpo+0*n_cells] -= dtody*(F_Ly[0]);
  C2[jpo+1*n_cells] -= dtody*(F_Ly[3]);
  C2[jpo+2*n_cells] -= dtody*(F_Ly[1]);
  C2[jpo+3*n_cells] -= dtody*(F_Ly[2]);
  C2[jpo+4*n_cells] -= dtody*(F_Ly[4]);
  #ifdef DE
  C2[jpo+5*n_cells] -= dtody*(F_Ly[5]);
  #endif
  // cell k-1
  C2[kmo+0*n_cells] -= dtodz*(- F_Rz[0]);
  C2[kmo+1*n_cells] -= dtodz*(- F_Rz[2]);
  C2[kmo+2*n_cells] -= dtodz*(- F_Rz[3]);
  C2[kmo+3*n_cells] -= dtodz*(- F_Rz[1]);
  C2[kmo+4*n_cells] -= dtodz*(- F_Rz[4]);
  #ifdef DE
  C2[kmo+5*n_cells] -= dtodz*(- F_Rz[5]);
  #endif
  // cell k+1
  C2[kpo+0*n_cells] -= dtodz*(F_Lz[0]);
  C2[kpo+1*n_cells] -= dtodz*(F_Lz[2]);
  C2[kpo+2*n_cells] -= dtodz*(F_Lz[3]);
  C2[kpo+3*n_cells] -= dtodz*(F_Lz[1]);
  C2[kpo+4*n_cells] -= dtodz*(F_Lz[4]);
  #ifdef DE
  C2[kpo+5*n_cells] -= dtodz*(F_Lz[5]);
  #endif  
}


void average_cell(Real *C1, int i, int j, int k, int nx, int ny, int nz, int n_cells)
{
  int id = i + j*nx + k*nx*ny;
  int imo = i-1 + j*nx + k*nx*ny;
  int ipo = i+1 + j*nx + k*nx*ny;
  int jmo = i + (j-1)*nx + k*nx*ny;
  int jpo = i + (j+1)*nx + k*nx*ny;
  int kmo = i + j*nx + (k-1)*nx*ny;
  int kpo = i + j*nx + (k+1)*nx*ny;
  //printf("%3d %3d %3d  d_i: %e d_imo: %e d_ipo: %e d_jmo: %e d_jpo: %e d_kmo: %e d_kpo: %e\n", i, j, k, C1[id], C1[imo], C1[ipo], C1[jmo], C1[jpo], C1[kmo], C1[kpo]);
  //printf("%3d %3d %3d  vx_i: %e vx_imo: %e vx_ipo: %e\n", i, j, k, C1[id+n_cells]/C1[id], C1[imo+n_cells]/C1[imo], C1[ipo+n_cells]/C1[ipo]);
  //printf("%3d %3d %3d  vy_i: %e vy_jmo: %e vy_jpo: %e\n", i, j, k, C1[id+2*n_cells]/C1[id], C1[jmo+2*n_cells]/C1[jmo], C1[jpo+2*n_cells]/C1[jpo]);
  //printf("%3d %3d %3d  vz_i: %e vz_kmo: %e vz_kpo: %e\n", i, j, k, C1[id+3*n_cells]/C1[id], C1[kmo+3*n_cells]/C1[kmo], C1[kpo+3*n_cells]/C1[kpo]);

  int N = 0;
  Real d, d_av, mx, my, mz, vx_av, vy_av, vz_av, P, P_av;
  d_av = vx_av = vy_av = vz_av = P_av = 0.0;
  for (int kk=k-1;kk<=k+1;kk++) {
  for (int jj=j-1;jj<=j+1;jj++) {
  for (int ii=i-1;ii<=i+1;ii++) {
    id = ii+jj*nx+kk*nx*ny;
    d = C1[id];
    mx = C1[id+1*n_cells];
    my = C1[id+2*n_cells];
    mz = C1[id+3*n_cells];
    P = (C1[id+4*n_cells] - (0.5/d)*(mx*mx + my*my + mz*mz))*(gama-1.0);
    if (d > 0.0 && P > 0.0) {
      d_av += d;
      vx_av += mx;
      vy_av += my;
      vz_av += mz;
      P_av += P/(gama-1.0);
      N++;
    }
  }
  }
  }
  P_av = P_av/N;
  vx_av = vx_av/d_av;
  vy_av = vy_av/d_av;
  vz_av = vz_av/d_av;
  d_av = d_av/N;
  id = i+j*nx+k*nx*ny;
  C1[id] = d_av;
  C1[id+1*n_cells] = d_av*vx_av;
  C1[id+2*n_cells] = d_av*vy_av;
  C1[id+3*n_cells] = d_av*vz_av;
  C1[id+4*n_cells] = P_av/(gama-1.0) + 0.5*d_av*(vx_av*vx_av+vy_av*vy_av+vz_av*vz_av);
  #ifdef DE
  C1[id+5*n_cells] = P_av/(gama-1.0);
  #endif

  Real n = d_av*DENSITY_UNIT/(0.6*MP);
  Real T = P_av*PRESSURE_UNIT/(n*KB);
  if (T < 1.0e1) {
    P_av = n*KB*1.0e1/PRESSURE_UNIT;
  }
  C1[id+4*n_cells] = P_av/(gama-1.0) + 0.5*d_av*(vx_av*vx_av+vy_av*vy_av+vz_av*vz_av);
  #ifdef DE
  C1[id+5*n_cells] = P_av/(gama-1.0);
  T = C1[id+5*n_cells]*(gama-1.0)*PRESSURE_UNIT/(n*KB);
  #endif

  //printf("%3d %3d %3d  d_i: %e vx_i: %e vy_i: %e vz_i: %e E: %e n_i: %e T_i: %e\n", i, j, k, d_av, vx_av, vy_av, vz_av, C1[id+4*n_cells], n, T);
  //printf("%3d %3d %3d  d_i: %e d_imo: %e d_ipo: %e d_jmo: %e d_jpo: %e d_kmo: %e d_kpo: %e\n", i, j, k, C1[id], C1[imo], C1[ipo], C1[jmo], C1[jpo], C1[kmo], C1[kpo]);

}


void first_order_fluxes(Real *C1, Real *C2, int i, int j, int k, Real dtodx, Real dtody, Real dtodz, int nfields, int nx, int ny, int nz, int n_cells)
{
  int id = i + j*nx + k*nx*ny;
  int imo = i-1 + j*nx + k*nx*ny;
  int ipo = i+1 + j*nx + k*nx*ny;
  int jmo = i + (j-1)*nx + k*nx*ny;
  int jpo = i + (j+1)*nx + k*nx*ny;
  int kmo = i + j*nx + (k-1)*nx*ny;
  int kpo = i + j*nx + (k+1)*nx*ny;
  Real etah = 0.0;
  printf("%3d %3d %3d  d_i: %5.3e d_imo: %5.3e d_ipo: %5.3e d_jmo: %5.3e d_jpo: %5.3e d_kmo: %5.3e d_kpo: %5.3e\n", i, j, k, C1[id], C1[imo], C1[ipo], C1[jmo], C1[jpo], C1[kmo], C1[kpo]);
  printf("%3d %3d %3d  vx_i: %e vx_imo: %e vx_ipo: %e\n", i, j, k, C1[id+n_cells]/C1[id], C1[imo+n_cells]/C1[imo], C1[ipo+n_cells]/C1[ipo]);
  printf("%3d %3d %3d  vy_i: %e vy_jmo: %e vy_jpo: %e\n", i, j, k, C1[id+2*n_cells]/C1[id], C1[jmo+2*n_cells]/C1[jmo], C1[jpo+2*n_cells]/C1[jpo]);
  printf("%3d %3d %3d  vz_i: %e vz_kmo: %e vz_kpo: %e\n", i, j, k, C1[id+3*n_cells]/C1[id], C1[kmo+3*n_cells]/C1[kmo], C1[kpo+3*n_cells]/C1[kpo]);

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
  // calculate the fluxes
  // Lx
  fill_flux_array_pcm(C1, imo, id, cW, n_cells, 0);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Lx, gama, etah);
  #endif
  
  // Rx
  fill_flux_array_pcm(C1, id, ipo, cW, n_cells, 0);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Rx, gama, etah);
  #endif

  // Ly
  fill_flux_array_pcm(C1, jmo, id, cW, n_cells, 1);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ly, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ly, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ly, gama, etah);
  #endif

  // Ry
  fill_flux_array_pcm(C1, id, jpo, cW, n_cells, 1);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ry, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ry, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ry, gama, etah);
  #endif

  // Lz
  fill_flux_array_pcm(C1, kmo, id, cW, n_cells, 2);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lz, gama, etah);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes(cW, F_Lz, gama, etah);
  #endif

  // Rz
  fill_flux_array_pcm(C1, id, kpo, cW, n_cells, 2);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rz, gama, etah);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes(cW, F_Rz, gama, etah);
  #endif


  // Update the conserved variables for the affected cell using the first-order fluxes 
  C2[id+0*n_cells] = C1[id+0*n_cells] + dtodx*(F_Lx[0] - F_Rx[0]) + dtody*(F_Ly[0] - F_Ry[0]) + dtodz*(F_Lz[0] - F_Rz[0]);
  C2[id+1*n_cells] = C1[id+1*n_cells] + dtodx*(F_Lx[1] - F_Rx[1]) + dtody*(F_Ly[3] - F_Ry[3]) + dtodz*(F_Lz[2] - F_Rz[2]);
  C2[id+2*n_cells] = C1[id+2*n_cells] + dtodx*(F_Lx[2] - F_Rx[2]) + dtody*(F_Ly[1] - F_Ry[1]) + dtodz*(F_Lz[3] - F_Rz[3]);
  C2[id+3*n_cells] = C1[id+3*n_cells] + dtodx*(F_Lx[3] - F_Rx[3]) + dtody*(F_Ly[2] - F_Ry[2]) + dtodz*(F_Lz[1] - F_Rz[1]);
  C2[id+4*n_cells] = C1[id+4*n_cells] + dtodx*(F_Lx[4] - F_Rx[4]) + dtody*(F_Ly[4] - F_Ry[4]) + dtodz*(F_Lz[4] - F_Rz[4]);
  #ifdef DE
  C2[id+5*n_cells] = C1[id+5*n_cells] + dtodx*(F_Lx[5] - F_Rx[5]) + dtody*(F_Ly[5] - F_Ry[5]) + dtodz*(F_Lz[5] - F_Rz[5])
                   + 0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));
  #endif

/*
  // Correct the values of the conserved variables for the surrounding cells 
  // using the relevant first-order fluxes
  // Cell i-1
  C2[imo+0*n_cells] += dtodx*(-F_Lx[0]);
  C2[imo+1*n_cells] += dtodx*(-F_Lx[1]);
  C2[imo+2*n_cells] += dtodx*(-F_Lx[2]);
  C2[imo+3*n_cells] += dtodx*(-F_Lx[3]);
  C2[imo+4*n_cells] += dtodx*(-F_Lx[4]);
  #ifdef DE
  C2[imo+5*n_cells] += dtodx*(-F_Lx[5]);
  #endif
  // Cell i+1
  C2[ipo+0*n_cells] += dtodx*(F_Rx[0]);
  C2[ipo+1*n_cells] += dtodx*(F_Rx[1]);
  C2[ipo+2*n_cells] += dtodx*(F_Rx[2]);
  C2[ipo+3*n_cells] += dtodx*(F_Rx[3]);
  C2[ipo+4*n_cells] += dtodx*(F_Rx[4]);
  #ifdef DE
  C2[ipo+5*n_cells] += dtodx*(F_Rx[5]);
  #endif
  // Cell j-1
  C2[jmo+0*n_cells] += dtody*(-F_Ly[0]);
  C2[jmo+1*n_cells] += dtody*(-F_Ly[3]);
  C2[jmo+2*n_cells] += dtody*(-F_Ly[1]);
  C2[jmo+3*n_cells] += dtody*(-F_Ly[2]);
  C2[jmo+4*n_cells] += dtody*(-F_Ly[4]);
  #ifdef DE
  C2[jmo+5*n_cells] += dtody*(-F_Ly[5]);
  #endif
  // Cell j+1
  C2[jpo+0*n_cells] += dtody*(F_Ry[0]);
  C2[jpo+1*n_cells] += dtody*(F_Ry[3]);
  C2[jpo+2*n_cells] += dtody*(F_Ry[1]);
  C2[jpo+3*n_cells] += dtody*(F_Ry[2]);
  C2[jpo+4*n_cells] += dtody*(F_Ry[4]);
  #ifdef DE
  C2[jpo+5*n_cells] += dtody*(F_Ry[5]);
  #endif
  // Cell k-1
  C2[kmo+0*n_cells] += dtodz*(-F_Lz[0]);
  C2[kmo+1*n_cells] += dtodz*(-F_Lz[2]);
  C2[kmo+2*n_cells] += dtodz*(-F_Lz[3]);
  C2[kmo+3*n_cells] += dtodz*(-F_Lz[1]);
  C2[kmo+4*n_cells] += dtodz*(-F_Lz[4]);
  #ifdef DE
  C2[kmo+5*n_cells] += dtodz*(-F_Ly[5]);
  #endif
  // Cell k+1
  C2[kpo+0*n_cells] += dtodz*(F_Rz[0]);
  C2[kpo+1*n_cells] += dtodz*(F_Rz[2]);
  C2[kpo+2*n_cells] += dtodz*(F_Rz[3]);
  C2[kpo+3*n_cells] += dtodz*(F_Rz[1]);
  C2[kpo+4*n_cells] += dtodz*(F_Rz[4]);
  #ifdef DE
  C2[kpo+5*n_cells] += dtodz*(F_Ry[5]);
  #endif  
*/
  //printf("%3d %3d %3d  d: %e d_imo: %e d_ipo: %e d_jmo: %e d_jpo: %e d_kmo: %e d_kpo: %e\n", i, j, k, C2[id], C2[imo], C2[ipo], C2[jmo], C2[jpo], C2[kmo], C2[kpo]);
}



void first_order_update(Real *C1, Real *C_half, int i, int j, int k, Real dtodx, Real dtody, Real dtodz, int nfields, int nx, int ny, int nz, int n_cells)
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
  // calculate the fluxes
  // Lx
  fill_flux_array_pcm(C1, imo, id, cW, n_cells, 0);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Lx, gama, etah);
  #endif
  
  // Rx
  fill_flux_array_pcm(C1, id, ipo, cW, n_cells, 0);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rx, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rx, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Rx, gama, etah);
  #endif

  // Ly
  fill_flux_array_pcm(C1, jmo, id, cW, n_cells, 1);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ly, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ly, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ly, gama, etah);
  #endif

  // Ry
  fill_flux_array_pcm(C1, id, jpo, cW, n_cells, 1);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Ry, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Ry, gama, etah);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes(cW, F_Ry, gama, etah);
  #endif

  // Lz
  fill_flux_array_pcm(C1, kmo, id, cW, n_cells, 2);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Lz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Lz, gama, etah);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes(cW, F_Lz, gama, etah);
  #endif

  // Rz
  fill_flux_array_pcm(C1, id, kpo, cW, n_cells, 2);
  #ifdef EXACT
  Calculate_Exact_Fluxes(cW, F_Rz, gama);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes(cW, F_Rz, gama, etah);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes(cW, F_Rz, gama, etah);
  #endif


  // Update the conserved variables using the first order fluxes 
  C_half[0] = C1[id+0*n_cells] + dtodx*(F_Lx[0] - F_Rx[0]) + dtody*(F_Ly[0] - F_Ry[0]) + dtodz*(F_Lz[0] - F_Rz[0]);
  C_half[1] = C1[id+1*n_cells] + dtodx*(F_Lx[1] - F_Rx[1]) + dtody*(F_Ly[3] - F_Ry[3]) + dtodz*(F_Lz[2] - F_Rz[2]);
  C_half[2] = C1[id+2*n_cells] + dtodx*(F_Lx[2] - F_Rx[2]) + dtody*(F_Ly[1] - F_Ry[1]) + dtodz*(F_Lz[3] - F_Rz[3]);
  C_half[3] = C1[id+3*n_cells] + dtodx*(F_Lx[3] - F_Rx[3]) + dtody*(F_Ly[2] - F_Ry[2]) + dtodz*(F_Lz[1] - F_Rz[1]);
  C_half[4] = C1[id+4*n_cells] + dtodx*(F_Lx[4] - F_Rx[4]) + dtody*(F_Ly[4] - F_Ry[4]) + dtodz*(F_Lz[4] - F_Rz[4]);
  #ifdef DE
  C_half[5] = C1[id+5*n_cells] + dtodx*(F_Lx[5] - F_Rx[5]) + dtody*(F_Ly[5] - F_Ry[5]) + dtodz*(F_Lz[5] - F_Rz[5])
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

  Real T_min = 1.0e4;
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
  //if (T_init > 1.0e8) printf("Bad cell: %e\n", T_init);

  // calculate cooling rate per volume
  T = T_init;

  // call the cooling function
  //cool = Schure_cool_CPU(n, T); 
  cool = Wiersma_cool_CPU(n, T); 
    
  // calculate change in temperature given dt
  del_T = cool*dt*TIME_UNIT*(gama-1.0)/(n*KB);

  // limit change in temperature to 1%
  while (del_T/T > 0.01 && T > T_min) {
    // what dt gives del_T = 0.1*T?
    dt_sub = 0.01*T*n*KB/(cool*TIME_UNIT*(gama-1.0));
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

  // set a temperature floor
  T = fmax(T, T_min);

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

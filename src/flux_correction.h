/*! \file flux_correction.h
 *  \brief Declarations of functions used in the first-order flux correction method. */

#ifndef FLUX_CORRECTION_H
#define FLUX_CORRECTION_H

void Flux_Correction_3D(Real *C1, Real *C2, int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt);

void fill_flux_array_pcm(Real *C1, int idl, int idr, Real cW[], int n_cells, int dir);

void second_order_fluxes(Real *C1, Real *C2, Real C_i[], Real C_imo[], Real C_imt[], Real C_ipo[], Real C_ipt[], Real C_jmo[], Real C_jmt[], Real C_jpo[], Real C_jpt[], Real C_kmo[], Real C_kmt[], Real C_kpo[], Real C_kpt[], int i, int j, int k, Real dx, Real dy, Real dz, Real dt, int n_fields, int nx, int ny, int nz, int n_cells);

void average_cell(Real *C1, int i, int j, int k, int nx, int ny, int nz, int n_cells, int n_fields);

void first_order_fluxes(Real *C1, Real *C2, int i, int j, int k, Real dtodx, Real dtody, Real dtodz, int nfields, int nx, int ny, int nz, int n_cells);

void first_order_update(Real *C1, Real *C_half, int i, int j, int k, Real dtodx, Real dtody, Real dtodz, int nfields, int nx, int ny, int nz, int n_cells);

void calc_g_3D(int xid, int yid, int zid, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real *gx, Real *gy, Real *gz);

void cooling_CPU(Real *C2, int id, int n_cells, Real dt);

Real Schure_cool_CPU(Real n, Real T);

Real Wiersma_cool_CPU(Real n, Real T);

#endif //FLUX_CORRECTION_H

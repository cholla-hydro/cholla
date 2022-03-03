/*! \file CTU_3D.h
 *  \brief Declarations for the 3D CTU algorithm. */

#ifndef CTU_3D_H
#define CTU_3D_H

#include "../global/global.h"

struct States_3D
{
  /*! \var d_Lx
   *  \brief Array containing the left input state for the density
             at the x+1/2 interface for each cell */
  Real *d_Lx;

  /*! \var d_Rx
   *  \brief Array containing the right input state for the density
             at the x+1/2 interface for each cell */
  Real *d_Rx;

  /*! \var d_Ly
   *  \brief Array containing the left input state for the density
             at the y+1/2 interface for each cell */
  Real *d_Ly;

  /*! \var d_Ry
   *  \brief Array containing the right input state for the density
             at the y+1/2 interface for each cell */
  Real *d_Ry;

  /*! \var d_Lz
   *  \brief Array containing the left input state for the density
             at the z+1/2 interface for each cell */
  Real *d_Lz;

  /*! \var d_Rz
   *  \brief Array containing the right input state for the density
             at the z+1/2 interface for each cell */
  Real *d_Rz;

  /*! \var mx_Lx
   *  \brief Array containing the left input state for the x momentum
             at the x+1/2 interface for each cell */
  Real *mx_Lx;

  /*! \var mx_Rx
   *  \brief Array containing the right input state for the x momentum
             at the x+1/2 interface for each cell */
  Real *mx_Rx;

  /*! \var mx_Ly
   *  \brief Array containing the left input state for the x momentum
             at the y+1/2 interface for each cell */
  Real *mx_Ly;

  /*! \var mx_Ry
   *  \brief Array containing the right input state for the x momentum
             at the y+1/2 interface for each cell */
  Real *mx_Ry;

  /*! \var mx_Lz
   *  \brief Array containing the left input state for the x momentum
             at the z+1/2 interface for each cell */
  Real *mx_Lz;

  /*! \var mx_Rz
   *  \brief Array containing the right input state for the x momentum
             at the z+1/2 interface for each cell */
  Real *mx_Rz;

  /*! \var my_Lx
   *  \brief Array containing the left input state for the y momentum
             at the x+1/2 interface for each cell */
  Real *my_Lx;

  /*! \var my_Rx
   *  \brief Array containing the right input state for the y momentum
             at the x+1/2 interface for each cell */
  Real *my_Rx;

  /*! \var my_Ly
   *  \brief Array containing the left input state for the y momentum
             at the y+1/2 interface for each cell */
  Real *my_Ly;

  /*! \var my_Ry
   *  \brief Array containing the right input state for the y momentum
             at the y+1/2 interface for each cell */
  Real *my_Ry;

  /*! \var my_Lz
   *  \brief Array containing the left input state for the y momentum
             at the z+1/2 interface for each cell */
  Real *my_Lz;

  /*! \var my_Rz
   *  \brief Array containing the right input state for the y momentum
             at the z+1/2 interface for each cell */
  Real *my_Rz;

  /*! \var mz_Lx
   *  \brief Array containing the left input state for the z momentum
             at the x+1/2 interface for each cell */
  Real *mz_Lx;

  /*! \var mz_Rx
   *  \brief Array containing the right input state for the z momentum
             at the x+1/2 interface for each cell */
  Real *mz_Rx;

  /*! \var mz_Ly
   *  \brief Array containing the left input state for the z momentum
             at the y+1/2 interface for each cell */
  Real *mz_Ly;

  /*! \var mz_Ry
   *  \brief Array containing the right input state for the z momentum
             at the y+1/2 interface for each cell */
  Real *mz_Ry;

  /*! \var mz_Lz
   *  \brief Array containing the left input state for the z momentum
             at the z+1/2 interface for each cell */
  Real *mz_Lz;

  /*! \var mz_Rz
   *  \brief Array containing the right input state for the z momentum
             at the z+1/2 interface for each cell */
  Real *mz_Rz;

  /*! \var E_Lx
   *  \brief Array containing the left input state for the Energy
             at the x+1/2 interface for each cell */
  Real *E_Lx;

  /*! \var E_Rx
   *  \brief Array containing the right input state for the Energy
             at the x+1/2 interface for each cell */
  Real *E_Rx;

  /*! \var E_Ly
   *  \brief Array containing the left input state for the Energy
             at the y+1/2 interface for each cell */
  Real *E_Ly;

  /*! \var E_Ry
   *  \brief Array containing the right input state for the Energy
             at the y+1/2 interface for each cell */
  Real *E_Ry;

  /*! \var E_Lz
   *  \brief Array containing the left input state for the Energy
             at the z+1/2 interface for each cell */
  Real *E_Lz;

  /*! \var E_Rz
   *  \brief Array containing the right input state for the Energy
             at the z+1/2 interface for each cell */
  Real *E_Rz;


  // constructor
  States_3D(int n_cells);

};

struct Fluxes_3D
{
  /*! \var dflux_x
   *  \brief Array containing the density flux at the x+1/2 interface for each cell */
  Real *dflux_x;

  /*! \var dflux_y
   *  \brief Array containing the density flux at the y+1/2 interface for each cell */
  Real *dflux_y;

  /*! \var dflux_z
   *  \brief Array containing the density flux at the z+1/2 interface for each cell */
  Real *dflux_z;

  /*! \var xmflux_x
   *  \brief Array containing the x momentum flux at the x+1/2 interface for each cell */
  Real *xmflux_x;

  /*! \var xmflux_y
   *  \brief Array containing the x momentum flux at the y+1/2 interface for each cell */
  Real *xmflux_y;

  /*! \var xmflux_z
   *  \brief Array containing the x momentum flux at the z+1/2 interface for each cell */
  Real *xmflux_z;

  /*! \var ymflux_x
   *  \brief Array containing the y momentum flux at the x+1/2 interface for each cell */
  Real *ymflux_x;

  /*! \var ymflux_y
   *  \brief Array containing the y momentum flux at the y+1/2 interface for each cell */
  Real *ymflux_y;

  /*! \var ymflux_z
   *  \brief Array containing the y momentum flux at the z+1/2 interface for each cell */
  Real *ymflux_z;

  /*! \var zmflux_x
   *  \brief Array containing the z momentum flux at the x+1/2 interface for each cell */
  Real *zmflux_x;

  /*! \var zmflux_y
   *  \brief Array containing the z momentum flux at the y+1/2 interface for each cell */
  Real *zmflux_y;

  /*! \var zmflux_z
   *  \brief Array containing the z momentum flux at the z+1/2 interface for each cell */
  Real *zmflux_z;	

  /*! \var Eflux_x
   *  \brief Array containing the Energy flux at the x+1/2 interface for each cell */
  Real *Eflux_x;

  /*! \var Eflux_y
   *  \brief Array containing the Energy flux at the y+1/2 interface for each cell */
  Real *Eflux_y;

  /*! \var Eflux_z
   *  \brief Array containing the Energy flux at the z+1/2 interface for each cell */
  Real *Eflux_z;


  Fluxes_3D(int n_cells);

};

/*! \fn CTU_Algorithm_3D(Real *C, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt)
 *! \brief The corner transport upwind algorithm of Gardiner & Stone, 2008. */
void CTU_Algorithm_3D(Real *C, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt);



#endif //CTU_3D_H

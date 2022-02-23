/*! \file CTU_1D.h
 *  \brief Declarations for the 1D CTU algorithm. */

#ifndef CTU_1D_H
#define CTU_1D_H

#include "../global/global.h"

struct States_1D
{
  /*! \var d_L
   *  \brief Array containing the left input state for the density
             at the x+1/2 interface for each cell */
  Real *d_L;

  /*! \var d_R
   *  \brief Array containing the right input state for the density
             at the x+1/2 interface for each cell */
  Real *d_R;

  /*! \var mx_L
   *  \brief Array containing the left input state for the x momentum
             at the x+1/2 interface for each cell */
  Real *mx_L;

  /*! \var mx_R
   *  \brief Array containing the right input state for the x momentum
             at the x+1/2 interface for each cell */
  Real *mx_R;

  /*! \var my_L
   *  \brief Array containing the left input state for the y momentum
             at the x+1/2 interface for each cell */
  Real *my_L;

  /*! \var my_R
   *  \brief Array containing the right input state for the y momentum
             at the x+1/2 interface for each cell */
  Real *my_R;

  /*! \var mz_L
   *  \brief Array containing the left input state for the z momentum
             at the x+1/2 interface for each cell */
  Real *mz_L;

  /*! \var mz_R
   *  \brief Array containing the right input state for the z momentum
             at the x+1/2 interface for each cell */
  Real *mz_R;

  /*! \var E_L
   *  \brief Array containing the left input state for the Energy
             at the x+1/2 interface for each cell */
  Real *E_L;

  /*! \var E_R
   *  \brief Array containing the right input state for the Energy
             at the x+1/2 interface for each cell */
  Real *E_R;


  // constructor
  States_1D(int n_cells);

};

struct Fluxes_1D
{
  /*! \var dflux
   *  \brief Array containing the density flux at the x+1/2 interface for each cell */
  Real *dflux;

  /*! \var xmflux
   *  \brief Array containing the x momentum flux at the x+1/2 interface for each cell */
  Real *xmflux;

  /*! \var ymflux
   *  \brief Array containing the y momentum flux at the x+1/2 interface for each cell */
  Real *ymflux;

  /*! \var zmflux
   *  \brief Array containing the z momentum flux at the x+1/2 interface for each cell */
  Real *zmflux;

  /*! \var Eflux
   *  \brief Array containing the Energy flux at the x+1/2 interface for each cell */
  Real *Eflux;


  Fluxes_1D(int n_cells);

};

/*! \fn CTU_Algorithm_1D(Real *C, int nx, int n_ghost, Real dx, Real dt)
 *! \brief The corner transport upwind algorithm of Gardiner & Stone, 2008. */
void CTU_Algorithm_1D(Real *C, int nx, int n_ghost, Real dx, Real dt);



#endif //CTU_1D_H

/*! \file RT.h
 *  \brief Declarations for the radiative transfer functions */

#ifdef RT

#ifndef RT_H
#define RT_H

#include "../global/global.h"


class Rad3D
{
  public:

  // number of ghost cells for RT boundaries
  int n_ghost = 2;

  // cells in radiation fields grid
  int nx;
  int ny;
  int nz;
  int n_cells;

  // number of frequencies
  const static int n_freq = 3;

  struct RT_Fields
  {
    // pointers to near and far radiation fields on the host and device
    // near field
    Real *rfn;
    Real *dev_rfn;
    // far field
    Real *rff;
    Real *dev_rff;
    // optically thin near field
    Real *ot;
    Real *dev_ot;

  } rtFields;


  void Free_Memory_RT(void);

};

#endif
#endif //RT

/*! \file RT.h
 *  \brief Declarations for the radiative transfer functions */

#ifdef RT

#ifndef RT_H
#define RT_H

#include "../global/global.h"

#define TPB_RT 1024

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

  // flag for the last iteration
  bool lastIteration = false;

  // number of frequencies
  const static int n_freq = 3;

  // prefactor for the far field source (q*<kF> in nedin2014) 
  Real rsFarFactor = 0; // the default value is used in tests

  //  cell size (assuming cubic cells)
  Real dx;

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
    // Eddington tensor. By default it is not needed on host, but some tests require it.
    Real *et = nullptr;
    Real *dev_et;
    // radiation source field. By default it is not needed on host, but some tests require it.
    Real *rs = nullptr;
    Real *dev_rs;

  } rtFields;

  struct TMP_Fields
  {
    // additional temporary fields
    // absorption coefficient;
    Real *dev_abc;
    // updated fields on the device
    Real *dev_rfnNew, *dev_rffNew;
  } tmpFields;

  void Initialize_RT_Fields(void);

  void Initialize_RT_Fields_GPU(void);

  void Copy_RT_Fields(void);

  void rtSolve(Real *dev_scalar);

  void Calc_Absorption(Real *dev_scalar);

  void OTVETIteration(void);

  void rtBoundaries();

  void Free_Memory_RT(void);

};

#endif
#endif //RT

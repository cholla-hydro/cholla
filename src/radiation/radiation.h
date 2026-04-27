/*! \file RT.h
 *  \brief Declarations for the radiative transfer functions */

#ifdef RT

  #ifndef RT_H
    #define RT_H

    #include "../global/global.h"

    #define TPB_RT 1024

struct Header;
class Grav3D;

namespace PhotoRatesCSI
{
struct TableWrapperGPU;
};

class Rad3D
{
 public:
  // Number of sub-cycling iterations for the RT solver to take. This, in effect, sets the reduced speed of light used
  // by the code: the reduced speed of light is num_iterations times the maximum speed of light anywhere in the
  // computational domain.
  int num_iterations;

  // flag for the last iteration
  bool lastIteration = false;

  // number of frequencies
  const static int n_freq = 3;

  // array of boundary flags
  int flags[6] = {0, 0, 0, 0, 0, 0};

  // prefactor for the far field source (q*<kF> in nedin2014)
  Real rsFarFactor = 0;  // the default value is used in tests

  struct RT_Fields {
    // pointers to radiation fields on the host and device (including the OT field - packed together for chemistry
    // update)
    Real *rf;
    Real *dev_rf;
    // Eddington tensor. By default it is not needed on host, but some tests require it.
    Real *et = nullptr;
    Real *dev_et;
    // radiation source field. By default it is not needed on host, but some tests require it.
    Real *rs = nullptr;
    Real *dev_rs;

    // additional temporary fields
    // absorption coefficient;
    Real *dev_abc;
    // updated fields on the device
    Real *dev_rfNew;
  } rtFields;

  PhotoRatesCSI::TableWrapperGPU *photoRates;
  const Header &grid;

  Rad3D(const Header &grid_);
  ~Rad3D();

  void Initialize_Start(const parameters &params);
  void Initialize_Finish();
  void Initialize_GPU();

#ifdef GRAVITY
  void ComputeEddingtonTensor(const parameters &params, Grav3D& G);
#endif

  void Copy_RT_Fields();

  void rtSolve(Real *dev_scalar);

  void Calc_Absorption(Real *dev_scalar);

  void OTVETIteration();

  void rtBoundaries();

  void Free_Memory();
};

  #endif
#endif  // RT

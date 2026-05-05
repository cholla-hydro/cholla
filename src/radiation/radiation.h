/*! \file RT.h
 *  \brief Declarations for the radiative transfer functions */

#ifdef RT

  #ifndef RT_H
    #define RT_H

    #include "../global/global.h"
    //#include "../io/io.h"

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

  // Number of radiation fields, depends on the method
  int n_rf;

  // flag for the last iteration
  bool lastIteration = false;

#ifdef OTVET
  // number of frequencies
  const static int n_freq = 3;  // 3 frequencies plus the 0 field is not included
  
  // number of fields per frequency
  const static int n_fpfreq = 2; // original OTVET, near and far
#endif //OTVET

#ifdef M1
  // number of frequencies
  const static int n_freq = 4;  // 3 frequencies plus the 0 field

  // number of fields per frequency
  const static int n_fpfreq = 4; // M1 has four fields per frequencies
#endif



  // array of boundary flags
  int flags[6] = {0, 0, 0, 0, 0, 0};

  // prefactor for the far field source (q*<kF> in nedin2014)
  Real rsFarFactor = 0;  // the default value is used in tests

  struct RT_Fields {
    // pointers to radiation fields on the host and device (including the OT field - packed together for chemistry
    // update)
    Real *rf;
    Real *dev_rf;

#ifdef OTVET
    // Eddington tensor. By default it is not needed on host, but some tests require it.
    Real *et = nullptr;
    Real *dev_et;
#endif //OTVET

    // radiation source field. By default it is not needed on host, but some tests require it.
    Real *rs = nullptr;
    Real *dev_rs;

#ifdef M1
    Real *dev_pij; // Pressure fields on the device
#endif //M1

    // additional temporary fields
    // absorption coefficient;
    Real *dev_abc;

    // updated fields on the device
    // consider making this a temporary allocation
    Real *dev_rfNew;

  } rtFields;

  PhotoRatesCSI::TableWrapperGPU *photoRates;
  const Header &grid;

  Rad3D(const Header &grid_);
  ~Rad3D();

  void Initialize_Start(const Parameters &params);
  void Initialize_Finish();
  void Initialize_GPU();

#ifdef GRAVITY
  void ComputeEddingtonTensor(const Parameters &params, Grav3D& G);
#endif

  void Copy_RT_Fields();

  void rtSolve(Real *dev_scalar);

  void Calc_Absorption(Real *dev_scalar);

  void OTVETIteration();   // original OTVET implementation
#ifdef M1
  void StepRFiIteration(Real cdt2dxRSL, Real gamma_sis); // For M1, ported from OTVET + Altair, step the radiation fields
  void ClipRFiIteration();// For M1, limit the radiation fields
#endif //M1
  

  // io
  void Radiation_Restart_Filename(char* filename, char* dirname, int nfile);
  void Read_Restart_HDF5( Parameters* P, int nfile);
  void Write_Restart_HDF5(Parameters* P, int nfile, const FnameTemplate& fname_template);

  void rtBoundaries();

  void Free_Memory();
};

  #endif
#endif  // RT

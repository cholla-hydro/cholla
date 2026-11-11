/*! \file RT.h
 *  \brief Declarations for the radiative transfer functions */

#ifndef RT_H
#define RT_H

#include "../global/global.h"
#include "../io/FnameTemplate.h"

#ifdef RT

  #define TPB_RT 1024

  #ifdef HDF5
    #include <hdf5.h>
  #endif

  #include "sources/rt_sources.h"

struct Header;
class Grav3D;

namespace rt_photo_rates_csi
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

  // Reduced speed of light: c_eff = reduced_c * c, limited by num_iterations substeps.
  // reduced_c <= 0 selects the legacy scheme (light advances one cell per hydro step).
  Real reduced_c;

  // Number of radiation fields, depends on the method
  int n_rf;

  // flag for the last iteration
  bool lastIteration = false;

  #ifdef RT_OTVET
  // number of frequencies
  const static int n_freq = 3;  // 3 frequencies plus the 0 field is not included

  // number of fields per frequency
  const static int n_fpfreq = 2;  // original RT_OTVET, near and far
  #endif                          // RT_OTVET

  #ifdef RT_M1
  // number of frequencies
  const static int n_freq = 4;  // 3 frequencies plus the 0 field

  // number of fields per frequency
  const static int n_fpfreq = 4;  // RT_M1 has four fields per frequencies
  #endif

  // powers of scale factor for converting
  // to cosmological integration units
  // rf is a photon number density: comoving field Ntilde = a^3 * n (same convention as gas density)
  static constexpr int RT_COSMO_FIELD_ASCALE_POWER = 3;
  // power of a applied when writing to disk: 0 = comoving, like gas (default); 3 = proper
  static constexpr int RT_DISK_FIELD_ASCALE_POWER = 0;

  #ifdef COSMOLOGY
  Real current_a;  // current cosmological scale factor
  Real cosmo_h;    // "little h", H0/100
  Real rho_0_gas;  // comoving density unit
  Real v_0_gas;    // velocity unit -- H_0/h in km/s kpc units
  #endif

  // RT unit layer: the only place RT code units are converted to CGS.
  // Defaults are the non-cosmological values; Set_Units() overwrites them
  // from Cosmo every step under COSMOLOGY.
  Real c_code       = 2.998e10 / VELOCITY_UNIT;  // speed of light in code velocity units
  Real rs_unit      = 1.0;                       // code source shape -> kernel source
  Real rf_to_proper = 1.0;                       // code (comoving) field -> proper field
  void Set_Units(Real a, Real h, Real rho_0, Real v_0);

  // One code length in cm (comoving). Normalizes photo-rate tables and source
  // shapes in the ICs, which run before Cosmo is initialized.
  static Real Length_Unit_CGS(const Parameters &P)
  {
  #ifdef COSMOLOGY
    return LENGTH_UNIT / (P.H0 / 100.0);
  #else
    return LENGTH_UNIT;
  #endif
  }

  // array of boundary flags
  int flags[6] = {0, 0, 0, 0, 0, 0};

  // prefactor for the far field source (q*<kF> in nedin2014)
  Real rsFarFactor = 0;  // the default value is used in tests

  int n_outputs;
  int next_output_indx;
  real_vector_t rt_outputs;
  Real next_output;
  bool exit_now;
  bool is_restart = false;

  struct RT_Fields {
    // pointers to radiation fields on the host and device (including the OT field - packed together for chemistry
    // update)
    // note that for COSMOLOGY, rf/dev_rf are stored in comoving units.
    Real *rf;
    Real *dev_rf;

  #ifdef RT_OTVET
    // Eddington tensor. By default it is not needed on host, but some tests require it.
    Real *et = nullptr;
    Real *dev_et;
  #endif  // RT_OTVET

    // radiation source field(s). By default it is not needed on host, but some tests require it.
    // note that for COSMOLOGY, rs/dev_rs are stored in comoving units. Any code that changes or
    // populates rs/dev_rs needs to respect this or call RS_Physical_to_Comoving[_GPU]() after
    // setting the source properties. In principle, rs/dev_rs can hold multipls source populations
    // if n_pops > 1, but that's for the future

    Real *rs = nullptr;
    Real *dev_rs;

  #ifdef RT_M1
    Real *dev_pij;  // Pressure fields on the device
  #endif            // RT_M1

    // additional temporary fields
    // absorption coefficient;
    Real *dev_abc;

    // updated fields on the device
    // consider making this a temporary allocation
    Real *dev_rfNew;

  } rtFields;

  int n_pops;  // Number of source populations (0 or 1 for now)
  RTSourceModel sources; // RT source model

  rt_photo_rates_csi::TableWrapperGPU *photoRates;
  const Header &grid;

  explicit Rad3D(const Header &grid_);
  ~Rad3D();

  void Initialize_Start(const Parameters &params);
  void Initialize_Finish(const Parameters &params, Real current_time);
  void Initialize_GPU();

  #ifdef GRAVITY
  void ComputeEddingtonTensor(const Parameters &params, Grav3D &G);
  #endif

  void Copy_RT_Fields();

  void rtSolve(Real *dev_scalar);

  void Calc_Absorption(Real *dev_scalar);

  void RT_OTVETIteration();  // original RT_OTVET implementation
  #ifdef RT_M1
  void StepRFiIteration(Real cdt2dxRSL,
                        Real gamma_sis);  // For RT_M1, ported from RT_OTVET + Altair, step the radiation fields
  void ClipRFiIteration();                // For RT_M1, limit the radiation fields
  #endif                                  // RT_M1

  // io
  void Radiation_Restart_Filename(char *filename, char *dirname, int nfile);
  #ifdef HDF5
  void Read_Restart_HDF5(Parameters *P, int nfile);
  void Write_Restart_HDF5(Parameters *P, int nfile, const FnameTemplate &fname_template);
  herr_t Write_HDF5_Attribute(hid_t file_id, hid_t dataspace_id, int *attribute, const char *name);
  herr_t Write_HDF5_Attribute(hid_t file_id, hid_t dataspace_id, double *attribute, const char *name);

  void Load_Outputs(const Parameters &P, Real current_time);
  void Set_Outputs(const Parameters &P, Real current_time);
  void Set_Next_Output();
  #endif

  void rtBoundaries();

  #ifdef COSMOLOGY
  void RF_Physical_to_Comoving_Units(void);
  void RF_Comoving_to_Physical_Units(void);
  void RS_Physical_to_Comoving_Units(void);
  void RS_Comoving_to_Physical_Units(void);
  void RF_Physical_to_Comoving_Units_GPU(void);
  void RF_Comoving_to_Physical_Units_GPU(void);
  void RS_Physical_to_Comoving_Units_GPU(void);
  void RS_Comoving_to_Physical_Units_GPU(void);
  void RS_Scale_for_Kernel_GPU(Real factor);
  #endif
};

#endif
#endif  // RT

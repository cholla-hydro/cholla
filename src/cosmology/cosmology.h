#ifdef COSMOLOGY

  #ifndef COSMOLOGY_H
    #define COSMOLOGY_H

    #include <stdio.h>

    #include <cmath>
    #include <memory>

    #include "../global/global.h"
    #include "../gravity/grav3D.h"
    #include "../particles/particles_3D.h"
    #include "tabulated_dynamicalDE_EoS.h"

class Cosmology
{
 public:
  Real H0;
  Real Omega_M;
  Real Omega_L;
  Real Omega_K;
  Real Omega_b;
  Real Omega_R;
  Real w0;
  Real wa;

  /*! Calculate rho_DE(z) / rho_DE(z=0) at some scale factor */
  Real Get_DE_Density_from_a(Real a);

  /*! Stores pointer to dynamical DE equation of state table class */
  std::unique_ptr<TabulatedDynamicalDarkEnergyEoS> tab_dynamicalDE_EoS;

  /*! Indicate whether the simulation is configured to use a DynamicalDE EOS table */
  bool Using_DynamicalDE_Table() { return tab_dynamicalDE_EoS != nullptr; }

  Real cosmo_G;
  Real cosmo_h;
  Real current_z;
  Real current_a;
  Real max_delta_a;
  Real delta_a;

  Real r_0_dm;
  Real t_0_dm;
  Real v_0_dm;
  Real rho_0_dm;
  Real phi_0_dm;
  Real rho_mean_baryon;

  Real time_conversion;
  Real dt_secs;
  Real t_secs;

  // Real dens_avrg;

  Real r_0_gas;
  Real v_0_gas;
  Real t_0_gas;
  Real phi_0_gas;
  Real rho_0_gas;
  Real p_0_gas;
  Real e_0_gas;

  int n_outputs;
  int next_output_indx;
  real_vector_t scale_outputs;
  Real next_output;
  bool exit_now;

  Cosmology(void);
  void Initialize(struct Parameters *P, Grav3D &Grav, Particles3D &Particles);

  void Load_Scale_Outputs(struct Parameters *P);
  void Set_Scale_Outputs(struct Parameters *P);

  void Set_Next_Scale_Output();

  Real Get_Hubble_Parameter(Real a);

  Real dtda_cosmo(Real da, Real a);
  Real Get_dt_from_da_rk(Real da, Real a);
  Real Get_da_from_dt(Real dt);
  Real Get_dt_from_da(Real da, Real a);

  // growth function calculation
  // and interpolation
  /*! \brief Precompue the cosmological growth function */
  void Compute_Growth_Function(struct Parameters *P);
  /*! \brief Create the file for recording the growth function history */
  void Create_Growth_Function_File(struct Parameters *P);
  /*! \brief Perform linear interpolation on vectors */
  Real LinearInterpolation(const std::vector<Real> &x, const std::vector<Real> &y, Real a);
  /*! \brief Cosmological growth function at scale factor a */
  Real D_Growth(Real a);
  /*! \brief Cosmological growth function time derivative at scale factor a */
  Real dDdt_Growth(Real a);
  /*! \brief Function to precompute the growth function scale factor derivative */
  Real dDda_Growth(Real a);

  std::vector<Real> t_array;
  std::vector<Real> a_array;
  std::vector<Real> D_array;
  std::vector<Real> dDdt_array;
  // Real OmegaDEz(Real z);
  // std::vector<Real> growth_factor_system(Real z, std::vector<Real> y, std::vector<Real> params);

  // write expansion history log file
  void Create_Expansion_History_File(struct Parameters *P);
  void Write_Expansion_History_Entry(void);
};
Real Hubble_Growth_Function(Real a, Real H0, Real Omega_r, Real Omega_m, Real Omega_DE, Real w0, Real wa);
Real dHda_Growth_Function(Real a, Real H0, Real Omega_r, Real Omega_m, Real Omega_DE, Real w0, Real wa);
Real OmegaDEz_Growth_Function(Real z, Real Omega_DE, Real w0, Real wa);
std::vector<Real> growth_factor_system(Real z, const std::vector<Real>& y, const std::vector<Real>& params);

  #endif
#endif

#ifdef COSMOLOGY

  #include "../cosmology/cosmology.h"

  #include "../io/io.h"

Cosmology::Cosmology(void) {}

void Cosmology::Initialize(struct Parameters *P, Grav3D &Grav, Particles3D &Particles)
{
  chprintf("Cosmological Simulation\n");

  H0      = P->H0;
  cosmo_h = H0 / 100;
  H0 /= 1000;  //[km/s / kpc]
  Omega_M = P->Omega_M;
  Omega_L = P->Omega_L;
  Omega_R = P->Omega_R;
  Omega_K = 1 - (Omega_M + Omega_L + Omega_R);
  Omega_b = P->Omega_b;
  w0      = P->w0;
  wa      = P->wa;

  if (strcmp(P->init, "Read_Grid") == 0) {
    // Read scale factor value from Particles
    current_z = Particles.current_z;
    current_a = Particles.current_a;
  } else {
    current_z           = P->Init_redshift;
    current_a           = 1. / (current_z + 1);
    Particles.current_z = current_z;
    Particles.current_a = current_a;
  }

  // Set Scale factor in Gravity
  Grav.current_a = current_a;

  // Gravitational Constant in Cosmological Units
  cosmo_G = G_COSMO;

  // Set gravitational constant to use for potential calculation
  Grav.Gconst = cosmo_G;

  max_delta_a = 0.001;
  delta_a     = max_delta_a;

  // Initialize Time and set the time conversion
  time_conversion = KPC;
  t_secs          = 0;

  // The following code computes the universal time
  // at the scale factor of the ICs or restart file
  // Pick a small scale factor step for integrating the universal time
  Real da_t_sec = 1.0e-2 * current_a;
  // Pick a small but non-zero starting scale factor for the integral
  Real a_t_sec = 1.0e-6;
  // Step for the time integral, corresponding to da_t_sec
  Real dt_physical;

  // Advance a_t_sec until it matches current_a, and integrate time
  while (a_t_sec < current_a) {
    // Limit the scale a_t_sec factor to current_a
    if (a_t_sec + da_t_sec > current_a) {
      da_t_sec = current_a - a_t_sec;
    }

    // Compute the time step
    dt_physical = Get_dt_from_da(da_t_sec, a_t_sec);

    // Advance the time in seconds and the scale factor
    t_secs += dt_physical * time_conversion;
    a_t_sec += da_t_sec;
    // chprintf(" Revised a_t_sec %f da_t_sec %f t_start : %e Myr\n", a_t_sec, da_t_sec, t_secs / MYR);
  }
  chprintf(" Revised t_start : %f Myr\n", t_secs / MYR);

  // Set Normalization factors
  r_0_dm          = P->xlen / P->nx;
  t_0_dm          = 1. / H0;
  v_0_dm          = r_0_dm / t_0_dm / cosmo_h;
  rho_0_dm        = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_M / cosmo_h / cosmo_h;
  rho_mean_baryon = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_b / cosmo_h / cosmo_h;
  // dens_avrg = 0;

  r_0_gas   = 1.0;
  rho_0_gas = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_M / cosmo_h / cosmo_h;
  t_0_gas   = 1 / H0 * cosmo_h;
  v_0_gas   = r_0_gas / t_0_gas;
  phi_0_gas = v_0_gas * v_0_gas;
  p_0_gas   = rho_0_gas * v_0_gas * v_0_gas;
  e_0_gas   = v_0_gas * v_0_gas;

  chprintf(" H0: %f\n", H0 * 1000);
  chprintf(" Omega_L: %f\n", Omega_L);
  chprintf(" Omega_M: %f\n", Omega_M);
  chprintf(" Omega_b: %f\n", Omega_b);
  chprintf(" Current_a: %f\n", current_a);
  chprintf(" Current_z: %f\n", current_z);
  chprintf(" rho_0: %f\n", rho_0_gas);
  chprintf(" v_0: %f \n", v_0_gas);
  chprintf(" Max delta_a: %f \n", MAX_DELTA_A);

  Set_Scale_Outputs(P);
}

#endif

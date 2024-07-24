#ifdef COSMOLOGY
  #include <fstream>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../io/io.h"

void Grid3D::Initialize_Cosmology(struct Parameters *P)
{
  chprintf("Initializing Cosmology... \n");
  Cosmo.Initialize(P, Grav, Particles);

  // Create expansion history log file
  Cosmo.Create_Expansion_History_File(P);

  // Change to comoving Cosmological System
  Change_Cosmological_Frame_System(true);

  if (fabs(Cosmo.current_a - Cosmo.next_output) < 1e-5) {
    H.Output_Now = true;
  }

  chprintf("Cosmology Successfully Initialized. \n\n");
}

/* Computes dt/da * da */
Real Cosmology::dtda_cosmo(Real da, Real a)
{
  Real a2     = a * a;
  Real fac_de = pow(a, -3 * (1 + w0 + wa)) * exp(-3 * wa * (1 - current_a));
  Real a_dot  = sqrt(Omega_R / a2 + Omega_M / a + a2 * Omega_L * fac_de + Omega_K) * H0;
  return da / a_dot;
}

/* Compute dt/da * da. dt/da is computed with a Runge-Kutta integration step */
Real Cosmology::Get_dt_from_da_rk(Real da, Real a)
{
  Real a3 = 0.3;
  Real a4 = 0.6;
  Real a5 = 1.0;
  Real a6 = 0.875;
  Real c1 = 37.0 / 378.0;
  Real c3 = 250.0 / 621.0;
  Real c4 = 125.0 / 594.0;
  Real c6 = 512.0 / 1771.0;

  // compute RK average derivatives
  Real ak1 = dtda_cosmo(da, a);
  Real ak3 = dtda_cosmo(da, a + a3 * da);
  Real ak4 = dtda_cosmo(da, a + a4 * da);
  Real ak6 = dtda_cosmo(da, a + a6 * da);

  // compute timestep
  Real dt = (c1 * ak1 + c3 * ak3 + c4 * ak4 + c6 * ak6);

  // return timestep
  return dt;
}

Real Cosmology::Get_da_from_dt(Real dt)
{
  Real a2     = current_a * current_a;
  Real fac_de = pow(current_a, -3 * (1 + w0 + wa)) * exp(-3 * wa * (1 - current_a));
  Real a_dot  = sqrt(Omega_R / a2 + Omega_M / current_a + a2 * Omega_L * fac_de + Omega_K) * H0;
  return a_dot * dt;
}

Real Cosmology::Get_dt_from_da(Real da, Real a)
{
  return Get_dt_from_da_rk(da, a);

  /* The following commented code was the original Euler
     integrator for computing time from the scale factor.
     This has been left here temporarily to ease comparison
     with the Runge-Kutta integrator, but it can be removed
     eventually. */
  /* Real a2     = a * a;
  Real fac_de = pow(a, -3 * (1 + w0 + wa)) * exp(-3 * wa * (1 - a));
  Real a_dot  = sqrt(Omega_R / a2 + Omega_M / a + a2 * Omega_L * fac_de + Omega_K) * H0;
  return da / a_dot; */
}

Real Cosmology::Get_Hubble_Parameter(Real a)
{
  Real a2     = a * a;
  Real a3     = a2 * a;
  Real a4     = a2 * a2;
  Real fac_de = pow(a, -3 * (1 + w0 + wa)) * exp(-3 * wa * (1 - a));
  Real factor = (Omega_R / a4 + Omega_M / a3 + Omega_K / a2 + Omega_L * fac_de);
  return H0 * sqrt(factor);
}

void Grid3D::Change_Cosmological_Frame_System(bool forward)
{
  if (forward) {
    chprintf(" Converting to Cosmological Comoving System\n");
  } else {
    chprintf(" Converting to Cosmological Physical System\n");
  }

  Change_DM_Frame_System(forward);
  #ifndef ONLY_PARTICLES

  Change_GAS_Frame_System_GPU(forward);

  Change_GAS_Frame_System(forward);
  #endif  // ONLY_PARTICLES
}
void Grid3D::Change_DM_Frame_System(bool forward)
{
  #ifdef PARTICLES_CPU

  part_int_t pIndx;
  Real vel_factor;
  vel_factor = 1;

  for (pIndx = 0; pIndx < Particles.n_local; pIndx++) {
    Particles.vel_x[pIndx] *= vel_factor;
    Particles.vel_y[pIndx] *= vel_factor;
    Particles.vel_z[pIndx] *= vel_factor;
  }

  #endif  // PARTICLES_CPU

  // NOTE:Not implemented for PARTICLES_GPU, doesn't matter as long as
  // vel_factor=1
}

void Grid3D::Change_GAS_Frame_System(bool forward)
{
  Real dens_factor, momentum_factor, energy_factor;
  if (forward) {
    dens_factor     = 1 / Cosmo.rho_0_gas;
    momentum_factor = 1 / Cosmo.rho_0_gas / Cosmo.v_0_gas * Cosmo.current_a;
    energy_factor   = 1 / Cosmo.rho_0_gas / Cosmo.v_0_gas / Cosmo.v_0_gas * Cosmo.current_a * Cosmo.current_a;
  } else {
    dens_factor     = Cosmo.rho_0_gas;
    momentum_factor = Cosmo.rho_0_gas * Cosmo.v_0_gas / Cosmo.current_a;
    energy_factor   = Cosmo.rho_0_gas * Cosmo.v_0_gas * Cosmo.v_0_gas / Cosmo.current_a / Cosmo.current_a;
  }
  int k, j, i, id;
  for (k = 0; k < H.nz; k++) {
    for (j = 0; j < H.ny; j++) {
      for (i = 0; i < H.nx; i++) {
        id               = i + j * H.nx + k * H.nx * H.ny;
        C.density[id]    = C.density[id] * dens_factor;
        C.momentum_x[id] = C.momentum_x[id] * momentum_factor;
        C.momentum_y[id] = C.momentum_y[id] * momentum_factor;
        C.momentum_z[id] = C.momentum_z[id] * momentum_factor;
        C.Energy[id]     = C.Energy[id] * energy_factor;

  #ifdef DE
        C.GasEnergy[id] = C.GasEnergy[id] * energy_factor;
  #endif

  #ifdef COOLING_GRACKLE
        C.HI_density[id] *= dens_factor;
        C.HII_density[id] *= dens_factor;
        C.HeI_density[id] *= dens_factor;
        C.HeII_density[id] *= dens_factor;
        C.HeIII_density[id] *= dens_factor;
        C.e_density[id] *= dens_factor;
    #ifdef GRACKLE_METALS
        C.metal_density[id] *= dens_factor;
    #endif
  #endif  // COOLING_GRACKLE

  #ifdef CHEMISTRY_GPU
        C.HI_density[id] *= dens_factor;
        C.HII_density[id] *= dens_factor;
        C.HeI_density[id] *= dens_factor;
        C.HeII_density[id] *= dens_factor;
        C.HeIII_density[id] *= dens_factor;
        C.e_density[id] *= dens_factor;
  #endif
      }
    }
  }
}

/* create the file for recording the expansion history */
void Cosmology::Create_Expansion_History_File(struct Parameters *P)
{
  if (not Is_Root_Proc()) {
    return;
  }

  std::string file_name(EXPANSION_HISTORY_FILE_NAME);
  chprintf("\nCreating Expansion History File: %s \n\n", file_name.c_str());

  bool file_exists = false;
  if (FILE *file = fopen(file_name.c_str(), "r")) {
    file_exists = true;
    chprintf("  File exists, appending values: %s \n\n", file_name.c_str());
    fclose(file);
  }

  // current date/time based on current system
  time_t now = time(0);
  // convert now to string form
  char *dt = ctime(&now);

  std::string message = "# H0 OmegaM Omega_b OmegaL w0 wa Omega_R Omega_K\n";
  message += "# " + std::to_string(H0 * 1e3) + " " + std::to_string(Omega_M);
  message += " " + std::to_string(Omega_b);
  message += " " + std::to_string(Omega_L) + " " + std::to_string(w0) + " " + std::to_string(wa);
  message += " " + std::to_string(Omega_R) + " " + std::to_string(Omega_K);

  std::ofstream out_file;
  out_file.open(file_name.c_str(), std::ios::app);
  out_file << "# Run date: " << dt;
  out_file << message.c_str() << std::endl;
  out_file.close();
}

/* Write the current entry to the expansion history file */
void Cosmology::Write_Expansion_History_Entry(void)
{
  if (not Is_Root_Proc()) {
    return;
  }

  std::string message = std::to_string(t_secs / MYR) + " " + std::to_string(current_a);
  std::string file_name(EXPANSION_HISTORY_FILE_NAME);
  std::ofstream out_file;
  out_file.open(file_name.c_str(), std::ios::app);
  out_file << message.c_str() << std::endl;
  out_file.close();
}

#endif

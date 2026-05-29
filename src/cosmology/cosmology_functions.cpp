#ifdef COSMOLOGY
  #include <fstream>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../io/io.h"
  #include "../rk/rk4.h"

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

  // Finalize cosmological ICs HERE

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

Real Hubble_Growth_Function(Real a, Real H0, Real Omega_r, Real Omega_m, Real Omega_DE, Real w0, Real wa)
{
	//set redshifts, limit scale factor to 1.0e-6 minimum
	Real aa = a;
	if(aa<1.0e-6)
		aa = 1.0e-6;
	Real z = 1./aa -1.;
  return H0*sqrt( Omega_r*pow(1+z,4) + Omega_m*pow(1+z,3) + OmegaDEz_Growth_Function(z,Omega_DE,w0,wa) );
}

//Real Cosmology::OmegaDEz(Real z)
Real OmegaDEz_Growth_Function(Real z, Real Omega_DE, Real w0, Real wa)
{
  Real A = pow(1+z,3*(1+w0+wa));
  Real B = Omega_DE * exp(-3*wa*z/(1+z));
  return A*B;
}

//std::vector<Real> Cosmology::growth_factor_system(Real z, std::vector<Real> y, std::vector<Real> params)
std::vector<Real> growth_factor_system(Real z, std::vector<Real> y, std::vector<Real> params)
{

  int ny = y.size();
  std::vector<Real> dydz(ny);

  Real aa;
  Real a = y[0];
  Real delta = y[1];
  Real delta_dot = y[2];
  Real da_dt, d2delta_dt2;

  Real H0 = params[0];
  Real Omega_r = params[1];
  Real Omega_m = params[2];
  Real Omega_DE = params[3];
  Real w0 = params[4];
  Real wa = params[5];

  aa=a;
  if(aa<1.0e-7)
    aa = 1.0e-7;

  // get current hubble parameter at this
  // scale factor and time
  Real H = Hubble_Growth_Function(aa, H0, Omega_r, Omega_m, Omega_DE, w0, wa);
  //Real H = Get_Hubble_Parameter(aa);

  // get the current redshift
  z = 1./aa -1.;

  // Get the current fraction of the critical
  // density contributed by matter and DE
  Real Omega_r_z = Omega_r * pow(1+z,4);
  Real Omega_m_z = Omega_m * pow(1+z,3);
//  Real Omega_DE_z = OmegaDEz(z, Omega_DE, w0, wa);
  Real Omega_DE_z = OmegaDEz_Growth_Function(z, Omega_DE, w0, wa);
//  Real Omega_DE_z = OmegaDEz(z);
  Real Omega_tot = Omega_m_z + Omega_DE_z  + Omega_r_z;

  // get the current da/dt = H*a
  da_dt = H * a;

  // get the current d^2 delta/dt^2 = -2 H ddelta/dt + 4\piG\rho_0 \delta
  // \rho_0 = 3 \Omega_m(z)/\Omega_tot H^2 / 8 \pi G 
  // so the second term is 1.5*(Omega_m_z/Omega_tot)*(H**2)*delta
  d2delta_dt2 = -2*H*delta_dot + 1.5*(Omega_m_z/Omega_tot)*(H*H)*delta;

  dydz[0] = da_dt;
  dydz[1] = delta_dot;
  dydz[2] = d2delta_dt2;
  return dydz;
}
// Function to precompute the growth function
void Cosmology::Compute_Cosmo_Growth_Function(struct Parameters *P)
{


  int np = 6;
  int ny = 3;

  std::vector<Real> params(np);

  Real error;
  std::vector<Real> y_n(ny,0);
  std::vector<Real> yp (ny,0);

  RK_Integrator RK;
  RK.InitializeRK(3);

  std::vector<Real> y;

//HERE
  Real H0      = P->H0;
  H0 /= 1000;  //[km/s / kpc]
  Real Omega_M = P->Omega_M;
  Real Omega_L = P->Omega_L;
  Real Omega_R = P->Omega_R;
  Real Omega_K = 1 - (Omega_M + Omega_L + Omega_R);
  Real Omega_b = P->Omega_b;
  Real w0      = P->w0;
  Real wa      = P->wa;


  //parameters
  params[0] = H0;
  params[1] = Omega_R;
  params[2] = Omega_M;
  params[3] = Omega_L;
  params[4] = w0;
  params[5] = wa;

  printf("H0 %e OR %e OM %e OL %e w0 %e wa %e\n",H0,Omega_R,Omega_M,Omega_L,w0,wa);

  // initial scale factor, not important
  y_n[0] = 1.0e-7;
  y_n[1] = 1.0e-8;
  y_n[2] = 1.0e-8;

  Real t = 0;

  t_array.push_back(t);
  a_array.push_back(y_n[0]);
  D_array.push_back(y_n[1]);
  dDdt_array.push_back(y_n[2]);
  Real tmax = 1./H0;

  Real dt = 1.0e-4 * tmax;
  Real dt_new;
  Real dt_max = 1.0e-2 * tmax;

  Real a_max = 1.0;
  while( (t<tmax)&(y_n[0]<a_max) )
  {
    if(t+dt>tmax)
    {
      dt = tmax-t;
    }

    // evolve ODE by one timestep
    RK.rk4_ode( growth_factor_system, t , y_n, &dt, &dt_new, params, yp, &error);
    //RK.rk4_ode( [this]() {this->growth_factor_system();}, t , y_n, &dt, &dt_new, params, yp, &error);

    // iterate time
    t += dt;

    for(int i=0;i<yp.size();i++)
      y_n[i] = yp[i];

    // limit to the largest dz allowable
    if(dt_new<dt_max)
      dt_new = dt_max;

    // update the redshift step
    dt = dt_new;

    t_array.push_back(t);
    a_array.push_back(y_n[0]);
    D_array.push_back(y_n[1]);
    dDdt_array.push_back(y_n[2]);
  }

  //for(int i=0;i<t_array.size();i++)
  //  printf("%e\t%e\t%e\t%e\n",t_array[i],a_array[i],D_array[i],dDdt_array[i]);

  RK.FreeMemory();
}

void Grid3D::Change_Cosmological_Frame_System(bool forward)
{
  if (forward) {
    chprintf(" Converting to Cosmological Comoving System\n");
  } else {
    chprintf(" Converting to Cosmological Physical System\n");
  }

  Change_DM_Frame_System(forward); //does nothing
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
void Cosmology::Create_Growth_Factor_File(struct Parameters *P)
{
  if (not Is_Root_Proc()) {
    return;
  }

  std::string file_name(GROWTH_FACTOR_FILE_NAME);
  chprintf("\nCreating Growth Factor File: %s \n\n", file_name.c_str());

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

  // add columns to header
  out_file << "# t [1/H0] a D dD/dt [H0]" << std::endl;

  for(int i=0;i<t_array.size();i++)
  {
    message  = std::to_string(t_array[i]) + " " + std::to_string(a_array[i]);
    message += std::to_string(D_array[i]) + " " + std::to_string(dDdt_array[i]);
    out_file << message.c_str() << std::endl;
  }
  out_file.close();
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

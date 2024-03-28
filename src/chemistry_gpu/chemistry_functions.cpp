#ifdef CHEMISTRY_GPU

  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../utils/hydro_utilities.h"
  #include "../utils/mhd_utilities.h"
  #include "chemistry_gpu.h"
  #include "rates.cuh"

  #ifdef DE
    #include "../hydro/hydro_cuda.h"
  #endif

  #define TINY 1e-20

void Grid3D::Initialize_Chemistry(struct Parameters *P)
{
  chprintf("Initializing the GPU Chemistry Solver... \n");

  Chem.nx = H.nx;
  Chem.ny = H.ny;
  Chem.nz = H.nz;

  Chem.H.runtime_chemistry_step = 0;

  Chem.use_case_B_recombination = false;

  // Initialize the Chemistry Header
  Chem.H.gamma       = gama;
  Chem.H.N_Temp_bins = 600;
  Chem.H.Temp_start  = 1.0;
  Chem.H.Temp_end    = 1000000000.0;

  Chem.H.H_fraction = INITIAL_FRACTION_HI + INITIAL_FRACTION_HII;

  #ifdef COSMOLOGY
  Chem.H.H0      = P->H0;
  Chem.H.Omega_M = P->Omega_M;
  Chem.H.Omega_L = P->Omega_L;
  #endif  // COSMOLOGY

  // Set up the units system.
  Real Msun, kpc_cgs, kpc_km, dens_to_CGS;
  Msun        = MSUN_CGS;
  kpc_cgs     = KPC_CGS;
  kpc_km      = KPC;
  dens_to_CGS = Msun / kpc_cgs / kpc_cgs / kpc_cgs;
  #ifdef COSMOLOGY
  dens_to_CGS = dens_to_CGS * Cosmo.rho_0_gas * Cosmo.cosmo_h * Cosmo.cosmo_h;
  #endif  // COSMOLOGY

  // These are conversions from code units to cgs. Following Grackle
  Chem.H.density_units    = dens_to_CGS;
  Chem.H.length_units     = kpc_cgs;
  Chem.H.time_units       = kpc_km;
  Chem.H.dens_number_conv = Chem.H.density_units / MH;
  #ifdef COSMOLOGY
  Chem.H.a_value          = Cosmo.current_a;
  Chem.H.density_units    = Chem.H.density_units / Chem.H.a_value / Chem.H.a_value / Chem.H.a_value;
  Chem.H.length_units     = Chem.H.length_units / Cosmo.cosmo_h * Chem.H.a_value;
  Chem.H.time_units       = Chem.H.time_units / Cosmo.cosmo_h;
  Chem.H.dens_number_conv = Chem.H.dens_number_conv * pow(Chem.H.a_value, 3);
  #endif  // COSMOLOGY
  Chem.H.velocity_units = Chem.H.length_units / Chem.H.time_units;

  Real dens_base, length_base, time_base;
  dens_base   = Chem.H.density_units;
  length_base = Chem.H.length_units;
  #ifdef COSMOLOGY
  dens_base   = dens_base * Chem.H.a_value * Chem.H.a_value * Chem.H.a_value;
  length_base = length_base / Chem.H.a_value;
  #endif  // COSMOLOGY

  time_base             = Chem.H.time_units;
  Chem.H.cooling_units  = (pow(length_base, 2) * pow(MH, 2)) / (dens_base * pow(time_base, 3));
  Chem.H.reaction_units = MH / (dens_base * time_base);
  Chem.H.max_iter       = 10000;

  // The chemistry GPU functions need access to the temperature floor
  // but they don't have access to P. Use Chem.H as a carrier.
  // The P->temperature_floor is always defined, so safe to set here.
  Chem.H.temperature_floor = P->temperature_floor;

  // Initialize all the rates
  Chem.Initialize(P);

  #ifdef COSMOLOGY
  // Real kpc_cgs = KPC_CGS;
  Chem.H.density_conversion = Cosmo.rho_0_gas * Cosmo.cosmo_h * Cosmo.cosmo_h / pow(kpc_cgs, 3) * MSUN_CGS;
  Chem.H.energy_conversion  = Cosmo.v_0_gas * Cosmo.v_0_gas * 1e10;  // km^2 -> cm^2 ;
  #else                                                              // Not COSMOLOGY
  Chem.H.density_conversion = 1.0;
  Chem.H.energy_conversion  = 1.0;
  #endif
  Chem.H.n_uvb_rates_samples    = Chem.n_uvb_rates_samples;
  Chem.H.uvb_rates_redshift_d   = Chem.rates_z_d;
  Chem.H.photo_ion_HI_rate_d    = Chem.Ion_rates_HI_d;
  Chem.H.photo_ion_HeI_rate_d   = Chem.Ion_rates_HeI_d;
  Chem.H.photo_ion_HeII_rate_d  = Chem.Ion_rates_HeII_d;
  Chem.H.photo_heat_HI_rate_d   = Chem.Heat_rates_HI_d;
  Chem.H.photo_heat_HeI_rate_d  = Chem.Heat_rates_HeI_d;
  Chem.H.photo_heat_HeII_rate_d = Chem.Heat_rates_HeII_d;

  chprintf("Allocating Memory. \n\n");
  int n_cells               = H.nx * H.ny * H.nz;
  Chem.Fields.temperature_h = (Real *)malloc(n_cells * sizeof(Real));

  chprintf("Chemistry Solver Successfully Initialized. \n\n");
}

void Chem_GPU::Generate_Reaction_Rate_Table(Real **rate_table_array_d, Rate_Function_T rate_function, Real units)
{
  // Host array for storing the rates
  Real *rate_table_array_h = (Real *)malloc(H.N_Temp_bins * sizeof(Real));

  // Get the temperature spacing.
  Real T, logT, logT_start, d_logT;
  logT_start = log(H.Temp_start);
  d_logT     = (log(H.Temp_end) - logT_start) / (H.N_Temp_bins - 1);

  // Evaluate the rate at each temperature.
  for (int i = 0; i < H.N_Temp_bins; i++) {
    rate_table_array_h[i] = CHEM_TINY;
    logT                  = logT_start + i * d_logT;
    T                     = exp(logT);
    rate_table_array_h[i] = rate_function(T, units);
  }

  // Allocate the device array for the rate and copy from host
  Allocate_Array_GPU_Real(rate_table_array_d, H.N_Temp_bins);
  Copy_Real_Array_to_Device(H.N_Temp_bins, rate_table_array_h, *rate_table_array_d);

  // Free the host array
  free(rate_table_array_h);
}

void Chem_GPU::Initialize(struct Parameters *P)
{
  Initialize_Cooling_Rates();

  Initialize_Reaction_Rates();

  Initialize_UVB_Ionization_and_Heating_Rates(P);
}

void Chem_GPU::Initialize_Cooling_Rates()
{
  chprintf(" Initializing Cooling Rates... \n");
  Real units = H.cooling_units;
  Generate_Reaction_Rate_Table(&H.cool_ceHI_d, cool_ceHI_rate, units);
  Generate_Reaction_Rate_Table(&H.cool_ceHeI_d, cool_ceHeI_rate, units);
  Generate_Reaction_Rate_Table(&H.cool_ceHeII_d, cool_ceHeII_rate, units);

  Generate_Reaction_Rate_Table(&H.cool_ciHI_d, cool_ciHI_rate, units);
  Generate_Reaction_Rate_Table(&H.cool_ciHeI_d, cool_ciHeI_rate, units);
  Generate_Reaction_Rate_Table(&H.cool_ciHeII_d, cool_ciHeII_rate, units);
  Generate_Reaction_Rate_Table(&H.cool_ciHeIS_d, cool_ciHeIS_rate, units);

  if (!use_case_B_recombination) {
    Generate_Reaction_Rate_Table(&H.cool_reHII_d, cool_reHII_rate_case_A, units);
    Generate_Reaction_Rate_Table(&H.cool_reHeII_1_d, cool_reHeII1_rate_case_A, units);
    Generate_Reaction_Rate_Table(&H.cool_reHeIII_d, cool_reHeIII_rate_case_A, units);
  } else {
    Generate_Reaction_Rate_Table(&H.cool_reHII_d, cool_reHII_rate_case_B, units);
    Generate_Reaction_Rate_Table(&H.cool_reHeII_1_d, cool_reHeII1_rate_case_B, units);
    Generate_Reaction_Rate_Table(&H.cool_reHeIII_d, cool_reHeIII_rate_case_B, units);
  }
  Generate_Reaction_Rate_Table(&H.cool_reHeII_2_d, cool_reHeII2_rate, units);

  Generate_Reaction_Rate_Table(&H.cool_brem_d, cool_brem_rate, units);

  H.cool_compton = 5.65e-36 / units;
}

void Chem_GPU::Initialize_Reaction_Rates()
{
  chprintf(" Initializing Reaction Rates... \n");
  Real units = H.reaction_units;
  Generate_Reaction_Rate_Table(&H.k_coll_i_HI_d, coll_i_HI_rate, units);
  Generate_Reaction_Rate_Table(&H.k_coll_i_HeI_d, coll_i_HeI_rate, units);
  Generate_Reaction_Rate_Table(&H.k_coll_i_HeII_d, coll_i_HeII_rate, units);
  Generate_Reaction_Rate_Table(&H.k_coll_i_HI_HI_d, coll_i_HI_HI_rate, units);
  Generate_Reaction_Rate_Table(&H.k_coll_i_HI_HeI_d, coll_i_HI_HeI_rate, units);

  if (!use_case_B_recombination) {
    Generate_Reaction_Rate_Table(&H.k_recomb_HII_d, recomb_HII_rate_case_A, units);
    Generate_Reaction_Rate_Table(&H.k_recomb_HeII_d, recomb_HeII_rate_case_A, units);
    Generate_Reaction_Rate_Table(&H.k_recomb_HeIII_d, recomb_HeIII_rate_case_A, units);
  } else {
    Generate_Reaction_Rate_Table(&H.k_recomb_HII_d, recomb_HII_rate_case_B, units);
    Generate_Reaction_Rate_Table(&H.k_recomb_HeII_d, recomb_HeII_rate_case_B, units);
    Generate_Reaction_Rate_Table(&H.k_recomb_HeIII_d, recomb_HeIII_rate_case_B, units);
  }
}

void Chem_GPU::Initialize_UVB_Ionization_and_Heating_Rates(struct Parameters *P)
{
  chprintf(" Initializing UVB Rates... \n");
  Load_UVB_Ionization_and_Heating_Rates(P);

  Copy_UVB_Rates_to_GPU();

  #ifdef TEXTURES_UVB_INTERPOLATION
  Bind_GPU_Textures(n_uvb_rates_samples, Heat_rates_HI_h, Heat_rates_HeI_h, Heat_rates_HeII_h, Ion_rates_HI_h,
                    Ion_rates_HeI_h, Ion_rates_HeII_h);
  #endif
}

void Chem_GPU::Copy_UVB_Rates_to_GPU()
{
  Allocate_Array_GPU_float(&rates_z_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Heat_rates_HI_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Heat_rates_HeI_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Heat_rates_HeII_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Ion_rates_HI_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Ion_rates_HeI_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Ion_rates_HeII_d, n_uvb_rates_samples);

  Copy_Float_Array_to_Device(n_uvb_rates_samples, rates_z_h, rates_z_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Heat_rates_HI_h, Heat_rates_HI_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Heat_rates_HeI_h, Heat_rates_HeI_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Heat_rates_HeII_h, Heat_rates_HeII_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Ion_rates_HI_h, Ion_rates_HI_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Ion_rates_HeI_h, Ion_rates_HeI_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Ion_rates_HeII_h, Ion_rates_HeII_d);
}

void Grid3D::Update_Chemistry()
{
  #ifdef COSMOLOGY
  Chem.H.current_z = Cosmo.current_z;
  #else
  Chem.H.current_z = 0;
  #endif

  Do_Chemistry_Update(C.device, H.nx, H.ny, H.nz, H.n_ghost, H.n_fields, H.dt, Chem.H);
}

void Grid3D::Compute_Gas_Temperature(Real *temperature, bool convert_cosmo_units)
{
  int k, j, i, id;
  Real dens_HI, dens_HII, dens_HeI, dens_HeII, dens_HeIII, dens_e, gamma;
  Real d, vx, vy, vz, E, GE, mu, temp, cell_dens, cell_n;
  Real current_a, a2;
  gamma = gama;

  for (k = 0; k < H.nz; k++) {
    for (j = 0; j < H.ny; j++) {
      for (i = 0; i < H.nx; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        d  = C.density[id];
        vx = C.momentum_x[id] / d;
        vy = C.momentum_y[id] / d;
        vz = C.momentum_z[id] / d;
        E  = C.Energy[id];

  #ifdef DE
        GE = C.GasEnergy[id];
  #else
        GE = E - hydro_utilities::Calc_Kinetic_Energy_From_Velocity(d, vx, vy, vz);
    #ifdef MHD
        GE -= mhd::utils::computeMagneticEnergy(C.magnetic_x[id], C.magnetic_y[id], C.magnetic_z[id]);
    #endif  // MHD
  #endif

        dens_HI    = C.HI_density[id];
        dens_HII   = C.HII_density[id];
        dens_HeI   = C.HeI_density[id];
        dens_HeII  = C.HeII_density[id];
        dens_HeIII = C.HeIII_density[id];
        dens_e     = C.e_density[id];

        cell_dens = dens_HI + dens_HII + dens_HeI + dens_HeII + dens_HeIII;
        cell_n    = dens_HI + dens_HII + (dens_HeI + dens_HeII + dens_HeIII) / 4 + dens_e;
        mu        = cell_dens / cell_n;

  #ifdef COSMOLOGY
        if (convert_cosmo_units) {
          current_a = Cosmo.current_a;
          a2        = current_a * current_a;
          GE *= Chem.H.energy_conversion / a2;
        } else {
          GE *= 1e10;  // convert from (km/s)^2 to (cm/s)^2
        }
  #endif

        temp = GE * MP * mu / d / KB * (gamma - 1.0);
        ;
        temperature[id] = temp;
        // chprintf( "mu: %e \n", mu );
        // if ( temp > 1e7 ) chprintf( "Temperature: %e   mu: %e \n", temp, mu
        // );
      }
    }
  }
}

void Chem_GPU::Reset()
{
  free(rates_z_h);
  free(Heat_rates_HI_h);
  free(Heat_rates_HeI_h);
  free(Heat_rates_HeII_h);
  free(Ion_rates_HI_h);
  free(Ion_rates_HeI_h);
  free(Ion_rates_HeII_h);

  Free_Array_GPU_float(rates_z_d);
  Free_Array_GPU_float(Heat_rates_HI_d);
  Free_Array_GPU_float(Heat_rates_HeI_d);
  Free_Array_GPU_float(Heat_rates_HeII_d);
  Free_Array_GPU_float(Ion_rates_HI_d);
  Free_Array_GPU_float(Ion_rates_HeI_d);
  Free_Array_GPU_float(Ion_rates_HeII_d);

  free(Fields.temperature_h);
}

#endif

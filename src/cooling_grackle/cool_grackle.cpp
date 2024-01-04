#ifdef COOLING_GRACKLE

  #include "../cooling_grackle/cool_grackle.h"

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>

  #include "../grid/grid_enum.h"
  #include "../io/io.h"

Cool_GK::Cool_GK(void) {}

void Grid3D::Initialize_Grackle(struct Parameters *P)
{
  chprintf("Initializing Grackle... \n");

  Cool.Initialize(P, Cosmo);

  Allocate_Memory_Grackle();

  Initialize_Fields_Grackle();

  chprintf("Grackle Initialized Successfully. \n\n");
}

void Cool_GK::Initialize(struct Parameters *P, Cosmology &Cosmo)
{
  chprintf(" Using Grackle for chemistry and cooling \n");
  chprintf(" N scalar fields: %d \n", NSCALARS);

  grackle_verbose = 1;
  #ifdef MPI_CHOLLA
  // Enable output
  if (procID != 0) grackle_verbose = 0;
  #endif

  tiny_number = 1.e-20;
  gamma       = P->gamma;

  dens_conv   = Cosmo.rho_0_gas;
  energy_conv = Cosmo.v_0_gas * Cosmo.v_0_gas;

  Real Msun = MSUN_CGS;
  Real kpc  = KPC_CGS;
  Real km   = KM_CGS

      dens_to_CGS = dens_conv * Msun / kpc / kpc / kpc * Cosmo.cosmo_h * Cosmo.cosmo_h;
  vel_to_CGS      = km;
  energy_to_CGS   = km * km;

  // First, set up the units system.
  // These are conversions from code units to cgs.
  units.comoving_coordinates = 1;    // 1 if cosmological sim, 0 if not
  units.a_units              = 1.0;  // units for the expansion factor
  units.a_value              = Cosmo.current_a / units.a_units;
  units.density_units        = dens_to_CGS / Cosmo.current_a / Cosmo.current_a / Cosmo.current_a;
  units.length_units         = kpc / Cosmo.cosmo_h * Cosmo.current_a;
  units.time_units           = KPC / Cosmo.cosmo_h;
  units.velocity_units       = units.length_units / Cosmo.current_a / units.time_units;  // since u = a * dx/dt

  // Second, create a chemistry object for parameters.  This needs to be a
  // pointer.
  data = new chemistry_data;
  if (set_default_chemistry_parameters(data) == 0) {
    chprintf("GRACKLE: Error in set_default_chemistry_parameters.\n");
    exit(-1);
  }
  // Set parameter values for chemistry.
  // Access the parameter storage with the struct you've created
  // or with the grackle_data pointer declared in grackle.h (see further below).
  data->use_grackle            = 1;  // chemistry on
  data->with_radiative_cooling = 1;  // Cooling on
  data->primordial_chemistry   = 1;  // molecular network with H, He
  data->UVbackground           = 1;  // UV background on
  // data->grackle_data_file = "src/cooling/CloudyData_UVB=HM2012.h5"; // data
  // file data->grackle_data_file =
  // "src/cooling/CloudyData_UVB=HM2012_cloudy.h5"; // data file
  // data->grackle_data_file =
  // "src/cooling_grackle/CloudyData_UVB=Puchwein2018_cloudy.h5"; // data file
  data->grackle_data_file = P->UVB_rates_file;  // data file
  // data->grackle_data_file = "src/cooling/CloudyData_UVB=FG2011.h5"; // data
  // file
  data->use_specific_heating_rate   = 0;
  data->use_volumetric_heating_rate = 0;
  data->cmb_temperature_floor       = 1;

  #ifdef GRACKLE_METALS
  data->metal_cooling = 1;  // metal cooling off
  #else
  chprintf("WARNING: Metal Cooling is Off. \n");
  data->metal_cooling = 0;  // metal cooling off
  #endif

  #ifdef PARALLEL_OMP
  data->omp_nthreads = N_OMP_THREADS_GRACKLE;
  #endif

  if (data->UVbackground == 1) chprintf("GRACKLE: Loading UV Background File: %s\n", data->grackle_data_file);

  // Finally, initialize the chemistry object.
  if (initialize_chemistry_data(&units) == 0) {
    chprintf("GRACKLE: Error in initialize_chemistry_data.\n");
    exit(-1);
  }

  if (data->UVbackground == 1) {
    scale_factor_UVB_on = 1 / (data->UVbackground_redshift_on + 1);
    chprintf("GRACKLE: UVB on: %f \n", scale_factor_UVB_on);
  }
}

void Grid3D::Allocate_Memory_Grackle()
{
  int n_cells = H.nx * H.ny * H.nz;
  int nx      = Grav.nx_local;
  int ny      = Grav.ny_local;
  int nz      = Grav.nz_local;
  // Set grid dimension and size.
  Cool.field_size               = n_cells;
  Cool.fields.grid_rank         = 3;
  Cool.fields.grid_dimension    = new int[3];
  Cool.fields.grid_start        = new int[3];
  Cool.fields.grid_end          = new int[3];
  Cool.fields.grid_dimension[0] = H.nx;  // the active dimension
  Cool.fields.grid_dimension[1] = H.ny;  // the active dimension
  Cool.fields.grid_dimension[2] = H.nz;  // the active dimension
  // grid_start and grid_end are used to ignore ghost zones.
  Cool.fields.grid_start[0] = H.n_ghost;
  Cool.fields.grid_start[1] = H.n_ghost;
  Cool.fields.grid_start[2] = H.n_ghost;
  Cool.fields.grid_end[0]   = H.nx - H.n_ghost - 1;
  Cool.fields.grid_end[1]   = H.ny - H.n_ghost - 1;
  Cool.fields.grid_end[2]   = H.nz - H.n_ghost - 1;

  Cool.fields.grid_dx = 0.0;  // used only for H2 self-shielding approximation

  Cool.fields.density         = C.density;
  Cool.fields.internal_energy = (Real *)malloc(Cool.field_size * sizeof(Real));
  // Cool.fields.x_velocity      = (Real *) malloc(Cool.field_size *
  // sizeof(Real)); Cool.fields.y_velocity      = (Real *)
  // malloc(Cool.field_size * sizeof(Real)); Cool.fields.z_velocity      = (Real
  // *) malloc(Cool.field_size * sizeof(Real));
  Cool.fields.x_velocity = NULL;
  Cool.fields.y_velocity = NULL;
  Cool.fields.z_velocity = NULL;

  chprintf(" Allocating memory for: HI, HII, HeI, HeII, HeIII, e   densities\n");
  Cool.fields.HI_density    = &C.host[H.n_cells * grid_enum::HI_density];
  Cool.fields.HII_density   = &C.host[H.n_cells * grid_enum::HII_density];
  Cool.fields.HeI_density   = &C.host[H.n_cells * grid_enum::HeI_density];
  Cool.fields.HeII_density  = &C.host[H.n_cells * grid_enum::HeII_density];
  Cool.fields.HeIII_density = &C.host[H.n_cells * grid_enum::HeIII_density];
  Cool.fields.e_density     = &C.host[H.n_cells * grid_enum::e_density];

  #ifdef GRACKLE_METALS
  chprintf(" Allocating memory for: metal density\n");
  Cool.fields.metal_density = &C.host[H.n_cells * grid_enum::metal_density];
  #else
  Cool.fields.metal_density = NULL;
  #endif

  #ifdef OUTPUT_TEMPERATURE
  Cool.temperature = (Real *)malloc(Cool.field_size * sizeof(Real));
  #endif
}

void Cool_GK::Free_Memory()
{
  // free( fields.x_velocity );
  // free( fields.y_velocity );
  // free( fields.z_velocity );
  free(fields.internal_energy);

  #ifdef OUTPUT_TEMPERATURE
  free(temperature);
  #endif
}

#endif

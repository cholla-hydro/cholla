#if defined(FEEDBACK) && defined(PARTICLES_GPU) && defined(PARTICLE_AGE) && defined(PARTICLE_IDS)

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>

  #include <cstring>
  #include <fstream>
  #include <sstream>
  #include <vector>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../feedback/s99table.h"
  #include "feedback.h"

  #define FEED_INFO_N     8
  #define i_RES           1
  #define i_UNRES         2
  #define i_ENERGY        3
  #define i_MOMENTUM      4
  #define i_UNRES_ENERGY  5
  #define i_WIND_MOMENTUM 6
  #define i_WIND_ENERGY   7

  // the starburst 99 total stellar mass input
  // stellar wind momentum fluxes and SN rates
  // must be divided by this to get per solar
  // mass values.
  #define S_99_TOTAL_MASS 1e6

  #define TPB_FEEDBACK 128
  // seed for poisson random number generator
  #define FEEDBACK_SEED 42

namespace feedback
{
Real *dev_snr, snr_dt, time_sn_start, time_sn_end;
Real *dev_sw_p, *dev_sw_e, sw_dt, time_sw_start, time_sw_end;
int snr_n;
}  // namespace feedback

  #ifndef O_HIP
inline __device__ double atomicMax(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}
  #endif  // O_HIP

inline __device__ Real Calc_Timestep(Real gamma, Real* density, Real* momentum_x, Real* momentum_y, Real* momentum_z,
                                     Real* energy, int index, Real dx, Real dy, Real dz)
{
  Real dens  = fmax(density[index], DENS_FLOOR);
  Real d_inv = 1.0 / dens;
  Real vx    = momentum_x[index] * d_inv;
  Real vy    = momentum_y[index] * d_inv;
  Real vz    = momentum_z[index] * d_inv;
  Real P     = fmax((energy[index] - 0.5 * dens * (vx * vx + vy * vy + vz * vz)) * (gamma - 1.0), TINY_NUMBER);
  Real cs    = sqrt(gamma * P * d_inv);
  return fmax(fmax((fabs(vx) + cs) / dx, (fabs(vy) + cs) / dy), (fabs(vz) + cs) / dz);
}

/** the prescription for dividing a scalar quantity between 3x3x3 cells is done
   by imagining a 2x2x2 cell volume around the SN.  These fractions, then,
   represent the linear extent of this volume into the cell in question. For i=0
   this should be 1*1/2. For i=-1 this should be (1-dx)*1/2. For i=+1 this
   should be dx*1/2. In the above the 1/2 factor is normalize over 2
   cells/direction.
  */
inline __device__ Real Frac(int i, Real dx) { return (-0.5 * i * i - 0.5 * i + 1 + i * dx) * 0.5; }

inline __device__ Real D_Frac(int i, Real dx)
{
  return (dx > 0.5) * i * (1 - 2 * dx) + ((i + 1) * dx + 0.5 * (i - 1)) - 3 * (i - 1) * (i + 1) * (0.5 - dx);
}

/** This function used for debugging potential race conditions.  Feedback from neighboring
    particles could simultaneously alter one hydro cell's conserved quantities.
 */
inline __device__ bool Particle_Is_Alone(Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev, part_int_t n_local,
                                         int gtid, Real dx)
{
  Real x0 = pos_x_dev[gtid];
  Real y0 = pos_y_dev[gtid];
  Real z0 = pos_z_dev[gtid];
  // Brute force loop to see if particle is alone
  for (int i = 0; i < n_local; i++) {
    if (i == gtid) continue;
    if (abs(x0 - pos_x_dev[i]) > dx) continue;
    if (abs(y0 - pos_y_dev[i]) > dx) continue;
    if (abs(z0 - pos_z_dev[i]) > dx) continue;
    // If we made it here, something is too close.
    return false;
  }
  return true;
}

inline __device__ Real Get_Average_Density(Real* density, int xi, int yi, int zi, int nx_grid, int ny_grid, int n_ghost)
{
  Real d_average = 0.0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        d_average +=
            density[(xi + n_ghost + i) + (yi + n_ghost + j) * nx_grid + (zi + n_ghost + k) * nx_grid * ny_grid];
      }
    }
  }
  return d_average / 27;
}

inline __device__ Real Get_Average_Number_Density_CGS(Real* density, int xi, int yi, int zi, int nx_grid, int ny_grid,
                                                      int n_ghost)
{
  return Get_Average_Density(density, xi, yi, zi, nx_grid, ny_grid, n_ghost) * DENSITY_UNIT / (MU * MP);
}

  #ifndef NO_SN_FEEDBACK
/**
 * @brief
 * -# Read in SN rate data from Starburst 99. If no file exists, assume a
 * constant rate.
 *
 * @param P pointer to parameters struct. Passes in starburst 99 filename and
 * random number gen seed.
 */
void feedback::Init_State(struct parameters* P)
{
  chprintf("feedback::Init_State start\n");
  std::string snr_filename(P->snr_filename);
  if (not snr_filename.empty()) {
    chprintf("Specified a SNR filename %s.\n", snr_filename.data());

    feedback::S99Table tab = parse_s99_table(snr_filename, feedback::S99TabKind::supernova);
    const std::size_t time_col = tab.col_index("TIME");
    const std::size_t rate_col = tab.col_index("ALL SUPERNOVAE: TOTAL RATE");
    const std::size_t nrows = tab.nrows();

    // read in array of supernova rate values.
    std::vector<Real> snr_time(nrows);
    std::vector<Real> snr(nrows);

    for (std::size_t i = 0; i < nrows; i++){
      // in the following divide by # years per kyr (1000)
      snr_time[i] = tab(time_col, i) / 1000;
      snr[i] = pow(10,tab(rate_col, i)) * 1000 / S_99_TOTAL_MASS;
    }

    time_sn_end   = snr_time[snr_time.size() - 1];
    time_sn_start = snr_time[0];
    // the following is the time interval between data points
    // (i.e. assumes regular temporal spacing)
    snr_dt = (time_sn_end - time_sn_start) / (snr.size() - 1);

    CHECK(cudaMalloc((void**)&dev_snr, snr.size() * sizeof(Real)));
    CHECK(cudaMemcpy(dev_snr, snr.data(), snr.size() * sizeof(Real), cudaMemcpyHostToDevice));

  } else {
    chprintf("No SN rate file specified.  Using constant rate\n");
    time_sn_start = DEFAULT_SN_START;
    time_sn_end   = DEFAULT_SN_END;
  }
}
  #endif  // NO_SN_FEEDBACK

  #ifndef NO_WIND_FEEDBACK
/**
 * @brief
 * Read in Stellar wind data from Starburst 99. If no file exists, assume a
 * constant rate.
 *
 *
 * @param P pointer to parameters struct. Passes in starburst 99 filepath
 */
void feedback::Init_Wind_State(struct parameters* P)
{
  chprintf("Init_Wind_State start\n");
  std::string sw_filename(P->sw_filename);
  if (sw_filename.empty()) {
    CHOLLA_ERROR("must specify a stellar wind file.\n");
  }

  feedback::S99Table tab = parse_s99_table(sw_filename, feedback::S99TabKind::stellar_wind);

  const std::size_t COL_TIME       = tab.col_index("TIME");
  const std::size_t COL_POWER      = tab.col_index("POWER: ALL");
  const std::size_t COL_ALL_P_FLUX = tab.col_index("MOMENTUM FLUX: ALL");
  const std::size_t nrows          = tab.nrows();

  std::vector<Real> sw_time(nrows);
  std::vector<Real> sw_p(nrows);
  std::vector<Real> sw_e(nrows);

  for (std::size_t i = 0; i < nrows; i++){
    sw_time[i] = tab(COL_TIME, i) / 1000;  // divide by # years per kyr (1000)
    sw_e[i]    = tab(COL_POWER, i);
    sw_p[i]    = tab(COL_ALL_P_FLUX, i);
  }

  time_sw_end   = sw_time[sw_time.size() - 1];
  time_sw_start = sw_time[0];
  // the following is the time interval between data points
  // (i.e. assumes regular temporal spacing)
  sw_dt = (time_sw_end - time_sw_start) / (sw_p.size() - 1);
  chprintf("wind t_s %.5e, t_e %.5e, delta T %0.5e\n", time_sw_start, time_sw_end, sw_dt);

  CHECK(cudaMalloc((void**)&dev_sw_p, sw_p.size() * sizeof(Real)));
  CHECK(cudaMemcpy(dev_sw_p, sw_p.data(), sw_p.size() * sizeof(Real), cudaMemcpyHostToDevice));

  CHECK(cudaMalloc((void**)&dev_sw_e, sw_e.size() * sizeof(Real)));
  CHECK(cudaMemcpy(dev_sw_e, sw_e.data(), sw_e.size() * sizeof(Real), cudaMemcpyHostToDevice));

  chprintf("first 40 stellar wind momentum values:\n");
  for (int i = 0; i < 40; i++) {
    chprintf("%0.5e  %5f %5f \n", sw_time.at(i), sw_e.at(i), sw_p.at(i));
  }
}

  #endif  // NO_WIND_FEEDBACK

/**
 * @brief Get the Starburst 99 stellar wind momentum flux per solar mass.
 *
 * @param t cluster age in kyr
 * @param dev_sw_p device array of log base 10 momentum flux values in dynes.
 * @param sw_dt time interval between table data points in kyr.
 * @param t_start cluster age when flux becomes non-negligible (kyr).
 * @param t_end  cluster age when stellar winds turn off (kyr).
 * @return flux (in Cholla force units) per solar mass.
 */
__device__ Real Get_Wind_Flux(Real t, Real* dev_sw_p, Real sw_dt, Real t_start, Real t_end)
{
  if (t < t_start || t >= t_end) return 0;

  int index        = (int)((t - t_start) / sw_dt);
  Real log_p_dynes = dev_sw_p[index] + (t - index * sw_dt) * (dev_sw_p[index + 1] - dev_sw_p[index]) / sw_dt;
  return pow(10, log_p_dynes) / FORCE_UNIT / S_99_TOTAL_MASS;
}

/**
 * @brief Get the Starburst 99 stellar wind emitted power per solar mass.
 *
 * @param t cluster age in kyr
 * @param dev_sw_e device array of log base 10 power (erg/s).
 * @param sw_dt time interval between table data points in kyr.
 * @param t_start cluster age when power becomes non-negligible (kyr).
 * @param t_end  cluster age when stellar winds turn off (kyr).
 * @return power (in Cholla units) per solar mass.
 */
__device__ Real Get_Wind_Power(Real t, Real* dev_sw_e, Real sw_dt, Real t_start, Real t_end)
{
  if (t < t_start || t >= t_end) return 0;

  int index  = (int)((t - t_start) / sw_dt);
  Real log_e = dev_sw_e[index] + (t - index * sw_dt) * (dev_sw_e[index + 1] - dev_sw_e[index]) / sw_dt;
  Real e     = pow(10, log_e) / (MASS_UNIT * VELOCITY_UNIT * VELOCITY_UNIT) * TIME_UNIT / S_99_TOTAL_MASS;
  return e;
}

/**
 * @brief Get the mass flux associated with stellar wind momentum flux
 *        and stellar wind power scaled per cluster mass.
 *
 * @param flux
 * @return mass flux in g/s per solar mass
 */
__device__ Real Get_Wind_Mass(Real flux, Real power)
{
  if (flux <= 0 || power <= 0) return 0;
  return flux * flux / power / 2;
}

/**
 * @brief returns SNR from starburst 99 (or default analytical rate).
 *        Time is in kyr.  Does a basic interpolation of S'99 table
 *        values.
 *
 * @param t   The cluster age.
 * @param dev_snr  device array with rate info
 * @param snr_dt  time interval between table data.  Constant value.
 * @param t_start cluster age when SNR is greater than zero.
 * @param t_end   cluster age when SNR drops to zero.
 * @return double number of SNe per kyr per solar mass
 */
__device__ Real Get_SN_Rate(Real t, Real* dev_snr, Real snr_dt, Real t_start, Real t_end)
{
  if (t < t_start || t >= t_end) return 0;
  if (dev_snr == nullptr) return feedback::DEFAULT_SNR;

  int index = (int)((t - t_start) / snr_dt);
  return dev_snr[index] + (t - index * snr_dt) * (dev_snr[index + 1] - dev_snr[index]) / snr_dt;
}

/**
 * @brief Get an actual number of SNe given the expected number.
 * Both the simulation step number and cluster ID is used to
 * set the state of the random number generator in a unique and
 * deterministic way.
 *
 * @param ave_num_sn expected number of SN, based on cluster
 * age, mass and time step.
 * @param n_step sim step number
 * @param cluster_id
 * @return number of supernovae
 */
inline __device__ int Get_Number_Of_SNe_In_Cluster(Real ave_num_sn, int n_step, part_int_t cluster_id)
{
  feedback_prng_t state;
  curand_init(FEEDBACK_SEED, 0, 0, &state);
  unsigned long long skip = n_step * 10000 + cluster_id;
  skipahead(skip, &state);  // provided by curand
  return (int)curand_poisson(&state, ave_num_sn);
}

__device__ Real Apply_Resolved_SN(Real pos_x, Real pos_y, Real pos_z, Real xMin, Real yMin, Real zMin, Real dx, Real dy,
                                  Real dz, int nx_g, int ny_g, int n_ghost, int n_cells, Real gamma,
                                  Real* conserved_device, short time_direction, Real feedback_density,
                                  Real feedback_energy)
{
  // For 2x2x2, a particle between 0-0.5 injects onto cell - 1
  int indx_x = (int)floor((pos_x - xMin - 0.5 * dx) / dx);
  int indx_y = (int)floor((pos_y - yMin - 0.5 * dy) / dy);
  int indx_z = (int)floor((pos_z - zMin - 0.5 * dz) / dz);

  Real cell_center_x = xMin + indx_x * dx + 0.5 * dx;
  Real cell_center_y = yMin + indx_y * dy + 0.5 * dy;
  Real cell_center_z = zMin + indx_z * dz + 0.5 * dz;

  Real delta_x = 1 - (pos_x - cell_center_x) / dx;
  Real delta_y = 1 - (pos_y - cell_center_y) / dy;
  Real delta_z = 1 - (pos_z - cell_center_z) / dz;

  Real* density    = conserved_device;
  Real* momentum_x = &conserved_device[n_cells * grid_enum::momentum_x];
  Real* momentum_y = &conserved_device[n_cells * grid_enum::momentum_y];
  Real* momentum_z = &conserved_device[n_cells * grid_enum::momentum_z];
  Real* energy     = &conserved_device[n_cells * grid_enum::Energy];
#ifdef DE
  Real* gasEnergy  = &conserved_device[n_cells * grid_enum::GasEnergy];
#endif

  Real local_dti = 0;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        int indx    = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;
        Real x_frac = i * (1 - delta_x) + (1 - i) * delta_x;
        Real y_frac = j * (1 - delta_y) + (1 - j) * delta_y;
        Real z_frac = k * (1 - delta_z) + (1 - k) * delta_z;

        atomicAdd(&density[indx], x_frac * y_frac * z_frac * feedback_density);
#ifdef DE
        atomicAdd(&gasEnergy[indx], x_frac * y_frac * z_frac * feedback_energy);
#endif
        atomicAdd(&energy[indx], x_frac * y_frac * z_frac * feedback_energy);

        if (time_direction > 0) {
          Real cell_dti = Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz);

          local_dti = fmax(local_dti, cell_dti);
        }
      }  // k loop
    }    // j loop
  }      // i loop

  return local_dti;
}

/* \brief Function used for depositing energy or momentum from an unresolved
 * supernova or from a stellar wind
 *
 * \note
 * Previously there were 2 separate functions defined to perform this operation.
 * They were functionally the same. They only differences were the names of
 * variables.
 *
 * \par
 * There are currently issues with the internals of this function:
 * - this requires the codebase to be compiled with the dual energy formalism
 * - momentum and total energy are not updated self-consistently
 */
__device__ Real Apply_Energy_Momentum_Deposition(Real pos_x, Real pos_y, Real pos_z, Real xMin, Real yMin, Real zMin,
                                                 Real dx, Real dy, Real dz, int nx_g, int ny_g, int n_ghost,
                                                 int n_cells, Real gamma, Real* conserved_device, short time_direction,
                                                 Real feedback_density, Real feedback_momentum, Real feedback_energy,
                                                 int indx_x, int indx_y, int indx_z)
{
  Real delta_x = (pos_x - xMin - indx_x * dx) / dx;
  Real delta_y = (pos_y - yMin - indx_y * dy) / dy;
  Real delta_z = (pos_z - zMin - indx_z * dz) / dz;

  Real local_dti = 0;

  Real* density    = conserved_device;
  Real* momentum_x = &conserved_device[n_cells * grid_enum::momentum_x];
  Real* momentum_y = &conserved_device[n_cells * grid_enum::momentum_y];
  Real* momentum_z = &conserved_device[n_cells * grid_enum::momentum_z];
  Real* energy     = &conserved_device[n_cells * grid_enum::Energy];
#ifdef DE
  Real* gas_energy = &conserved_device[n_cells * grid_enum::GasEnergy];
#endif

  // loop over the 27 cells to add up all the allocated feedback
  // momentum magnitudes.  For each cell allocate density and
  // energy based on the ratio of allocated momentum to this overall sum.
  Real mag = 0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        Real x_frac = D_Frac(i, delta_x) * Frac(j, delta_y) * Frac(k, delta_z);
        Real y_frac = Frac(i, delta_x) * D_Frac(j, delta_y) * Frac(k, delta_z);
        Real z_frac = Frac(i, delta_x) * Frac(j, delta_y) * D_Frac(k, delta_z);

        mag += sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac);
      }
    }
  }

  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        // index in array of conserved quantities
        int indx = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;

        Real x_frac = D_Frac(i, delta_x) * Frac(j, delta_y) * Frac(k, delta_z);
        Real y_frac = Frac(i, delta_x) * D_Frac(j, delta_y) * Frac(k, delta_z);
        Real z_frac = Frac(i, delta_x) * Frac(j, delta_y) * D_Frac(k, delta_z);

        Real px       = x_frac * feedback_momentum;
        Real py       = y_frac * feedback_momentum;
        Real pz       = z_frac * feedback_momentum;
        Real f_dens   = sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac) / mag * feedback_density;
        Real f_energy = sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac) / mag * feedback_energy;

        atomicAdd(&density[indx], f_dens);
        atomicAdd(&momentum_x[indx], px);
        atomicAdd(&momentum_y[indx], py);
        atomicAdd(&momentum_z[indx], pz);
        atomicAdd(&energy[indx], f_energy);

#ifdef DE
        gas_energy[indx] = energy[indx] - (momentum_x[indx] * momentum_x[indx] + momentum_y[indx] * momentum_y[indx] +
                                           momentum_z[indx] * momentum_z[indx]) /
                                              (2 * density[indx]);
#endif
        /*
        energy[indx] = ( momentum_x[indx] * momentum_x[indx] +
                         momentum_y[indx] * momentum_y[indx] +
                         momentum_z[indx] * momentum_z[indx] ) /
                       2 / density[indx] + gasEnergy[indx];
        */
        if (time_direction > 0) {
          Real cell_dti = Calc_Timestep(gamma, density, momentum_x, momentum_y, momentum_z, energy, indx, dx, dy, dz);
          local_dti     = fmax(local_dti, cell_dti);
        }
      }  // k loop
    }    // j loop
  }      // i loop

  return local_dti;
}

__device__ void SN_Feedback(Real pos_x, Real pos_y, Real pos_z, Real age, Real* mass_dev, part_int_t* id_dev, Real xMin,
                            Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g,
                            int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt, Real* dti, Real* dev_snr,
                            Real snr_dt, Real time_sn_start, Real time_sn_end, Real* prev_dens, short time_direction,
                            Real* s_info, Real* conserved_dev, Real gamma, int loop, int indx_x, int indx_y, int indx_z)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  Real dV = dx * dy * dz;
  Real feedback_density, feedback_momentum, feedback_energy;
  Real local_dti = 0.0;
  int n_cells    = nx_g * ny_g * nz_g;

  Real average_num_sn = Get_SN_Rate(age, dev_snr, snr_dt, time_sn_start, time_sn_end) * mass_dev[gtid] * dt;
  int N               = Get_Number_Of_SNe_In_Cluster(average_num_sn, n_step, id_dev[gtid]) * time_direction;
  /*
  if (gtid == 0) {
    kernel_printf("SNUMBER n_step: %d, id: %lld, N: %d\n", n_step, id_dev[gtid], N);
  }
  */

  // no sense doing anything if there was no SN
  if (N != 0) {
    Real n_0;
    if (time_direction == -1) {
      n_0 = prev_dens[gtid];
    } else {
      Real* density             = conserved_dev;
      n_0                       = Get_Average_Number_Density_CGS(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
      prev_dens[gtid]           = n_0;
      s_info[FEED_INFO_N * tid] = 1. * N;
    }

    feedback_energy  = N * feedback::ENERGY_PER_SN / dV;
    feedback_density = N * feedback::MASS_PER_SN / dV;

    Real shell_radius = feedback::R_SH * pow(n_0, -0.46) * pow(fabsf(N), 0.29);
  #ifdef ONLY_RESOLVED
    bool is_resolved = true;
  #else
    bool is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius;
  #endif

    if (is_resolved) {
      // inject energy and density
      if (time_direction > 0) {
        s_info[FEED_INFO_N * tid + i_RES]    = 1. * N;
        s_info[FEED_INFO_N * tid + i_ENERGY] = feedback_energy * dV;
      }
      local_dti = Apply_Resolved_SN(pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost, n_cells,
                                    gamma, conserved_dev, time_direction, feedback_density, feedback_energy);
    } else {
      // inject momentum and density
      feedback_momentum =
          time_direction * feedback::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(fabsf(N), 0.93) / dV / sqrt(3.0);
      if (time_direction > 0) {
        s_info[FEED_INFO_N * tid + i_UNRES]        = 1. * N;
        s_info[FEED_INFO_N * tid + i_MOMENTUM]     = feedback_momentum * dV * sqrt(3.0);
        s_info[FEED_INFO_N * tid + i_UNRES_ENERGY] = feedback_energy * dV;
      }
      local_dti = Apply_Energy_Momentum_Deposition(
          pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost, n_cells, gamma, conserved_dev,
          time_direction, feedback_density, feedback_momentum, feedback_energy, indx_x, indx_y, indx_z);
    }
  }

  if (time_direction > 0) atomicMax(dti, local_dti);
}

__device__ void Wind_Feedback(Real pos_x, Real pos_y, Real pos_z, Real age, Real* mass_dev, part_int_t* id_dev,
                              Real xMin, Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy,
                              Real dz, int nx_g, int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt,
                              Real* dti, Real* dev_sw_p, Real* dev_sw_e, Real sw_dt, Real time_sw_start,
                              Real time_sw_end, short time_direction, Real* s_info, Real* conserved_dev, Real gamma,
                              int loop, int indx_x, int indx_y, int indx_z)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  Real dV = dx * dy * dz;
  Real feedback_density, feedback_momentum, feedback_energy;
  Real local_dti = 0.0;
  int n_cells    = nx_g * ny_g * nz_g;

  if (age < 0 || age > time_sw_end) return;
  feedback_momentum = Get_Wind_Flux(age, dev_sw_p, sw_dt, time_sw_start, time_sw_end);
  // no sense in proceeding if there is no feedback.
  if (feedback_momentum == 0) return;
  feedback_energy  = Get_Wind_Power(age, dev_sw_e, sw_dt, time_sw_start, time_sw_end);
  feedback_density = Get_Wind_Mass(feedback_momentum, feedback_energy);

  // feedback_momentum now becomes momentum component along one direction.
  feedback_momentum *= mass_dev[gtid] * dt / dV / sqrt(3.0) * time_direction;
  feedback_density *= mass_dev[gtid] * dt / dV * time_direction;
  feedback_energy *= mass_dev[gtid] * dt / dV * time_direction;

  /* TODO refactor into separate kernel call
  if (time_direction > 0) {
    mass_dev[gtid]   -= feedback_density * dV;
  }*/

  if (time_direction > 0) {
    // we log net momentum, not momentum density, and magnitude (not the
    // component along a direction)
    s_info[FEED_INFO_N * tid + i_WIND_MOMENTUM] = feedback_momentum * dV * sqrt(3.0);
    s_info[FEED_INFO_N * tid + i_WIND_ENERGY]   = feedback_energy * dV;
  }

  local_dti = Apply_Energy_Momentum_Deposition(pos_x, pos_y, pos_z, xMin, yMin, zMin, dx, dy, dz, nx_g, ny_g, n_ghost,
                                               n_cells, gamma, conserved_dev, time_direction, feedback_density,
                                               feedback_momentum, feedback_energy, indx_x, indx_y, indx_z);

  if (time_direction > 0) atomicMax(dti, local_dti);
}

__device__ void Cluster_Feedback_Helper(part_int_t n_local, Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev,
                                        Real* age_dev, Real* mass_dev, part_int_t* id_dev, Real xMin, Real yMin,
                                        Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g,
                                        int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt, Real* dti,
                                        Real* dev_snr, Real snr_dt, Real time_sn_start, Real time_sn_end,
                                        Real* prev_dens, Real* dev_sw_p, Real* dev_sw_e, Real sw_dt, Real time_sw_start,
                                        Real time_sw_end, short time_direction, Real* s_info, Real* conserved_dev,
                                        Real gamma, int loop)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;
  // Bounds check on particle arrays
  if (gtid >= n_local) return;

  Real pos_x    = pos_x_dev[gtid];
  Real pos_y    = pos_y_dev[gtid];
  Real pos_z    = pos_z_dev[gtid];
  bool in_local = (pos_x >= xMin && pos_x < xMax) && (pos_y >= yMin && pos_y < yMax) && (pos_z >= zMin && pos_z < zMax);
  // Particle is outside bounds, exit
  if (!in_local) return;

  int indx_x  = (int)floor((pos_x - xMin) / dx);
  int indx_y  = (int)floor((pos_y - yMin) / dy);
  int indx_z  = (int)floor((pos_z - zMin) / dz);
  bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x >= nx_g - 2 * n_ghost ||
                indx_y >= ny_g - 2 * n_ghost || indx_z >= nz_g - 2 * n_ghost;
  // Ignore this particle, exit
  if (ignore) return;

  // bool is_alone = Particle_Is_Alone(pos_x_dev, pos_y_dev, pos_z_dev, n_local, gtid, 6*dx);
  // if (is_alone) kernel_printf(" particle not alone: step %d, id %ld\n", n_step, id_dev[gtid]);
  // if (!is_alone) return;

  // note age_dev is actually the time of birth
  Real age = t - age_dev[gtid];

  bool is_sn_feedback = false;
  bool is_wd_feedback = false;
  #ifndef NO_SN_FEEDBACK
  is_sn_feedback = true;
  #endif
  #ifndef NO_WIND_FEEDBACK
  is_wd_feedback = true;
  #endif

  // when applying different types of feedback, undoing the step requires
  // reverising the order
  if (time_direction > 0) {
    if (is_sn_feedback)
      SN_Feedback(pos_x, pos_y, pos_z, age, mass_dev, id_dev, xMin, yMin, zMin, xMax, yMax, zMax, dx, dy, dz, nx_g,
                  ny_g, nz_g, n_ghost, n_step, t, dt, dti, dev_snr, snr_dt, time_sn_start, time_sn_end, prev_dens,
                  time_direction, s_info, conserved_dev, gamma, loop, indx_x, indx_y, indx_z);
    if (is_wd_feedback)
      Wind_Feedback(pos_x, pos_y, pos_z, age, mass_dev, id_dev, xMin, yMin, zMin, xMax, yMax, zMax, dx, dy, dz, nx_g,
                    ny_g, nz_g, n_ghost, n_step, t, dt, dti, dev_sw_p, dev_sw_e, sw_dt, time_sw_start, time_sw_end,
                    time_direction, s_info, conserved_dev, gamma, loop, indx_x, indx_y, indx_z);
  } else {
    if (is_wd_feedback)
      Wind_Feedback(pos_x, pos_y, pos_z, age, mass_dev, id_dev, xMin, yMin, zMin, xMax, yMax, zMax, dx, dy, dz, nx_g,
                    ny_g, nz_g, n_ghost, n_step, t, dt, dti, dev_sw_p, dev_sw_e, sw_dt, time_sw_start, time_sw_end,
                    time_direction, s_info, conserved_dev, gamma, loop, indx_x, indx_y, indx_z);
    if (is_sn_feedback)
      SN_Feedback(pos_x, pos_y, pos_z, age, mass_dev, id_dev, xMin, yMin, zMin, xMax, yMax, zMax, dx, dy, dz, nx_g,
                  ny_g, nz_g, n_ghost, n_step, t, dt, dti, dev_snr, snr_dt, time_sn_start, time_sn_end, prev_dens,
                  time_direction, s_info, conserved_dev, gamma, loop, indx_x, indx_y, indx_z);
  }

  return;
}

__global__ void Cluster_Feedback_Kernel(part_int_t n_local, part_int_t* id_dev, Real* pos_x_dev, Real* pos_y_dev,
                                        Real* pos_z_dev, Real* mass_dev, Real* age_dev, Real xMin, Real yMin, Real zMin,
                                        Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g, int ny_g,
                                        int nz_g, int n_ghost, Real t, Real dt, Real* dti, Real* info, Real* density,
                                        Real gamma, Real* prev_dens, short time_direction, Real* dev_snr, Real snr_dt,
                                        Real time_sn_start, Real time_sn_end, Real* dev_sw_p, Real* dev_sw_e,
                                        Real sw_dt, Real time_sw_start, Real time_sw_end, int n_step, int loop)
{
  int tid = threadIdx.x;

  // for collecting SN feedback information
  __shared__ Real s_info[FEED_INFO_N * TPB_FEEDBACK];
  s_info[FEED_INFO_N * tid]     = 0;  // number of supernovae
  s_info[FEED_INFO_N * tid + 1] = 0;  // number of resolved events
  s_info[FEED_INFO_N * tid + 2] = 0;  // number of unresolved events
  s_info[FEED_INFO_N * tid + 3] = 0;  // resolved energy
  s_info[FEED_INFO_N * tid + 4] = 0;  // unresolved momentum
  s_info[FEED_INFO_N * tid + 5] = 0;  // unresolved KE added via momentum
  s_info[FEED_INFO_N * tid + 6] = 0;  // wind momentum
  s_info[FEED_INFO_N * tid + 7] = 0;  // wind energy added

  Cluster_Feedback_Helper(n_local, pos_x_dev, pos_y_dev, pos_z_dev, age_dev, mass_dev, id_dev, xMin, yMin, zMin, xMax,
                          yMax, zMax, dx, dy, dz, nx_g, ny_g, nz_g, n_ghost, n_step, t, dt, dti, dev_snr, snr_dt,
                          time_sn_start, time_sn_end, prev_dens, dev_sw_p, dev_sw_e, sw_dt, time_sw_start, time_sw_end,
                          time_direction, s_info, density, gamma, loop);

  __syncthreads();

  // reduce the info from all the threads in the block
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_info[FEED_INFO_N * tid] += s_info[FEED_INFO_N * (tid + s)];
      s_info[FEED_INFO_N * tid + 1] += s_info[FEED_INFO_N * (tid + s) + 1];
      s_info[FEED_INFO_N * tid + 2] += s_info[FEED_INFO_N * (tid + s) + 2];
      s_info[FEED_INFO_N * tid + 3] += s_info[FEED_INFO_N * (tid + s) + 3];
      s_info[FEED_INFO_N * tid + 4] += s_info[FEED_INFO_N * (tid + s) + 4];
      s_info[FEED_INFO_N * tid + 5] += s_info[FEED_INFO_N * (tid + s) + 5];
      s_info[FEED_INFO_N * tid + 6] += s_info[FEED_INFO_N * (tid + s) + 6];
      s_info[FEED_INFO_N * tid + 7] += s_info[FEED_INFO_N * (tid + s) + 7];
    }
    __syncthreads();
  }

  // atomicAdd reduces across all blocks
  if (tid == 0) {
    atomicAdd(info, s_info[0]);
    atomicAdd(info + 1, s_info[1]);
    atomicAdd(info + 2, s_info[2]);
    atomicAdd(info + 3, s_info[3]);
    atomicAdd(info + 4, s_info[4]);
    atomicAdd(info + 5, s_info[5]);
    atomicAdd(info + 6, s_info[6]);
    atomicAdd(info + 7, s_info[7]);
  }
}

__global__ void Adjust_Cluster_Mass_Kernel(part_int_t n_local, Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev,
                                           Real* age_dev, Real* mass_dev, part_int_t* id_dev, Real xMin, Real yMin,
                                           Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz,
                                           int nx_g, int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt,
                                           Real* dev_snr, Real snr_dt, Real time_sn_start, Real time_sn_end,
                                           Real* dev_sw_p, Real* dev_sw_e, Real sw_dt, Real time_sw_start,
                                           Real time_sw_end)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;
  // Bounds check on particle arrays
  if (gtid >= n_local) return;

  Real pos_x    = pos_x_dev[gtid];
  Real pos_y    = pos_y_dev[gtid];
  Real pos_z    = pos_z_dev[gtid];
  bool in_local = (pos_x >= xMin && pos_x < xMax) && (pos_y >= yMin && pos_y < yMax) && (pos_z >= zMin && pos_z < zMax);
  // Particle is outside bounds, exit
  if (!in_local) return;

  int indx_x  = (int)floor((pos_x - xMin) / dx);
  int indx_y  = (int)floor((pos_y - yMin) / dy);
  int indx_z  = (int)floor((pos_z - zMin) / dz);
  bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x >= nx_g - 2 * n_ghost ||
                indx_y >= ny_g - 2 * n_ghost || indx_z >= nz_g - 2 * n_ghost;
  // Ignore this particle, exit
  if (ignore) return;

  // bool is_alone = Particle_Is_Alone(pos_x_dev, pos_y_dev, pos_z_dev, n_local, gtid, 6*dx);
  // if (is_alone) kernel_printf(" particle not alone: step %d, id %ld\n", n_step, id_dev[gtid]);
  // if (!is_alone) return;

  Real age = t - age_dev[gtid];

  #ifndef NO_SN_FEEDBACK
  Real average_num_sn = Get_SN_Rate(age, dev_snr, snr_dt, time_sn_start, time_sn_end) * mass_dev[gtid] * dt;
  int N               = Get_Number_Of_SNe_In_Cluster(average_num_sn, n_step, id_dev[gtid]);
  mass_dev[gtid] -= N * feedback::MASS_PER_SN;
  #endif

  #ifndef NO_WIND_FEEDBACK
  Real feedback_momentum  = Get_Wind_Flux(age, dev_sw_p, sw_dt, time_sw_start, time_sw_end);
  Real feedback_energy    = Get_Wind_Power(age, dev_sw_e, sw_dt, time_sw_start, time_sw_end);
  Real feedback_mass_rate = Get_Wind_Mass(feedback_momentum, feedback_energy);

  mass_dev[gtid] -= feedback_mass_rate * dt;
  #endif
}

__device__ void Set_Average_Density(int indx_x, int indx_y, int indx_z, int nx_g, int ny_g, int n_ghost, Real* density,
                                    Real ave_dens)
{
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        int indx = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;

        density[indx] = ave_dens;
      }
    }
  }
}

__global__ void Set_Ave_Density_Kernel(part_int_t n_local, Real* pos_x_dev, Real* pos_y_dev, Real* pos_z_dev,
                                       Real* mass_dev, Real* age_dev, part_int_t* id_dev, Real xMin, Real yMin,
                                       Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx_g,
                                       int ny_g, int nz_g, int n_ghost, Real t, Real dt, Real* density, Real* dev_snr,
                                       Real snr_dt, Real time_sn_start, Real time_sn_end, Real time_sw_start,
                                       Real time_sw_end, int n_step)
{
  int tid  = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;
  // Bounds check on particle arrays
  if (gtid >= n_local) return;

  Real pos_x    = pos_x_dev[gtid];
  Real pos_y    = pos_y_dev[gtid];
  Real pos_z    = pos_z_dev[gtid];
  bool in_local = (pos_x >= xMin && pos_x < xMax) && (pos_y >= yMin && pos_y < yMax) && (pos_z >= zMin && pos_z < zMax);
  // Particle is outside bounds, exit
  if (!in_local) return;

  int indx_x  = (int)floor((pos_x - xMin) / dx);
  int indx_y  = (int)floor((pos_y - yMin) / dy);
  int indx_z  = (int)floor((pos_z - zMin) / dz);
  bool ignore = indx_x < 0 || indx_y < 0 || indx_z < 0 || indx_x >= nx_g - 2 * n_ghost ||
                indx_y >= ny_g - 2 * n_ghost || indx_z >= nz_g - 2 * n_ghost;
  // Ignore this particle, exit
  if (ignore) return;

  // bool is_alone = Particle_Is_Alone(pos_x_dev, pos_y_dev, pos_z_dev, n_local, gtid, 6*dx);
  // if (is_alone) kernel_printf(" particle not alone: step %d, id %ld\n", n_step, id_dev[gtid]);
  // if (!is_alone) return;

  bool is_sn_feedback   = false;
  bool is_wind_feedback = false;
  #ifndef NO_SN_FEEDBACK
  is_sn_feedback = true;
  #endif
  #ifndef NO_WIND_FEEDBACK
  is_wind_feedback = true;
  #endif

  Real ave_dens;
  Real age = t - age_dev[gtid];
  if (is_wind_feedback) {
    if (time_sw_start <= age && age <= time_sw_end) {
      ave_dens = Get_Average_Density(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
      Set_Average_Density(indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost, density, ave_dens);
      // since we've set the average density, no need to keep
      // checking whether we should do so.
      return;
    }
  }
  if (is_sn_feedback) {
    if (time_sn_start <= age && age <= time_sn_end) {
      Real average_num_sn = Get_SN_Rate(age, dev_snr, snr_dt, time_sn_start, time_sn_end) * mass_dev[gtid] * dt;
      int N               = Get_Number_Of_SNe_In_Cluster(average_num_sn, n_step, id_dev[gtid]);
      /*
      if (gtid == 0) {
        kernel_printf("AVEDENS n_step: %d, id: %lld, N: %d\n", n_step, id_dev[gtid], N);
      }*/
      Real n_0          = Get_Average_Number_Density_CGS(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
      Real shell_radius = feedback::R_SH * pow(n_0, -0.46) * pow(N, 0.29);
  #ifdef ONLY_RESOLVED
      bool is_resolved = true;
  #else
      bool is_resolved = 3 * max(dx, max(dy, dz)) <= shell_radius;
  #endif

      // resolved SN feedback does not average densities.
      if (!is_resolved && N > 0) {
        ave_dens = n_0 * MU * MP / DENSITY_UNIT;
        Set_Average_Density(indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost, density, ave_dens);
      }
    }
  }
}

/**
 * @brief Stellar feedback function (SNe and stellar winds)
 *
 * @param G
 * @param analysis
 * @return Real
 */
Real feedback::Cluster_Feedback(Grid3D& G, FeedbackAnalysis& analysis,
                                const bool disable_rewind)
{
  #ifdef CPU_TIME
  G.Timer.Feedback.Start();
  #endif

  if (G.H.dt == 0) return 0.0;

  Real h_dti = 0.0;
  // h_info is used to store feedback summary info on the host. The following
  // syntax sets all entries to 0 -- important if a process has no particles
  // (this is valid C++ syntax, but historically wasn't valid C syntax)
  Real h_info[FEED_INFO_N] = {};
  int time_direction, ngrid;
  Real *d_dti, *d_info;
  // require d_prev_dens in case we have to undo feedback if the time
  // step is too large.
  Real* d_prev_dens;

  // only apply feedback if we have clusters
  if (G.Particles.n_local > 0) {
    CHECK(cudaMalloc(&d_dti, sizeof(Real)));
    CHECK(cudaMemcpy(d_dti, &h_dti, sizeof(Real), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_prev_dens, G.Particles.n_local * sizeof(Real)));
    CHECK(cudaMemset(d_prev_dens, 0, G.Particles.n_local * sizeof(Real)));

    // I have no idea what ngrid is used for...
    ngrid = (G.Particles.n_local - 1) / TPB_FEEDBACK + 1;
    CHECK(cudaMalloc((void**)&d_info, FEED_INFO_N * sizeof(Real)));

    // before applying feedback, set gas density around clusters to the
    // average value from the 27 neighboring cells.  We don't want to
    // do this during application of feedback since "undoing it" in the
    // event that the time step is too large becomes difficult.
    hipLaunchKernelGGL(Set_Ave_Density_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local, G.Particles.pos_x_dev,
                       G.Particles.pos_y_dev, G.Particles.pos_z_dev, G.Particles.mass_dev, G.Particles.age_dev,
                       G.Particles.partIDs_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal, G.H.xblocal_max, G.H.yblocal_max,
                       G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny, G.H.nz, G.H.n_ghost, G.H.t, G.H.dt,
                       G.C.d_density, dev_snr, snr_dt, time_sn_start, time_sn_end, time_sw_start, time_sw_end,
                       G.H.n_step);
  }

  int loop_counter = 0;

  do {
    time_direction = 1;
    loop_counter++;

    if (G.Particles.n_local > 0) {
      // always reset d_info to 0 since otherwise do/while looping could add
      // values that should have been reverted.
      cudaMemset(d_info, 0, FEED_INFO_N * sizeof(Real));
      cudaMemset(d_dti, 0, sizeof(Real));
      hipLaunchKernelGGL(Cluster_Feedback_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local,
                         G.Particles.partIDs_dev, G.Particles.pos_x_dev, G.Particles.pos_y_dev, G.Particles.pos_z_dev,
                         G.Particles.mass_dev, G.Particles.age_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal,
                         G.H.xblocal_max, G.H.yblocal_max, G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny,
                         G.H.nz, G.H.n_ghost, G.H.t, G.H.dt, d_dti, d_info, G.C.d_density, gama, d_prev_dens,
                         time_direction, dev_snr, snr_dt, time_sn_start, time_sn_end, dev_sw_p, dev_sw_e, sw_dt,
                         time_sw_start, time_sw_end, G.H.n_step, loop_counter);

      CHECK(cudaMemcpy(&h_dti, d_dti, sizeof(Real), cudaMemcpyDeviceToHost));
    }

  #ifdef MPI_CHOLLA
    h_dti = ReduceRealMax(h_dti);
    MPI_Barrier(world);
  #endif  // MPI_CHOLLA
    if (h_dti != 0) {
      chprintf("+++++++  feed dt = %.12e, H.dt = %.12e\n", C_cfl / h_dti, G.H.dt);
    }

    if ((not disable_rewind) and (h_dti != 0) and (C_cfl / h_dti < G.H.dt)) {
      // timestep too big: need to undo the last operation
      time_direction = -1;
      if (G.Particles.n_local > 0) {
        hipLaunchKernelGGL(Cluster_Feedback_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local,
                           G.Particles.partIDs_dev, G.Particles.pos_x_dev, G.Particles.pos_y_dev, G.Particles.pos_z_dev,
                           G.Particles.mass_dev, G.Particles.age_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal,
                           G.H.xblocal_max, G.H.yblocal_max, G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny,
                           G.H.nz, G.H.n_ghost, G.H.t, G.H.dt, d_dti, d_info, G.C.d_density, gama, d_prev_dens,
                           time_direction, dev_snr, snr_dt, time_sn_start, time_sn_end, dev_sw_p, dev_sw_e, sw_dt,
                           time_sw_start, time_sw_end, G.H.n_step, loop_counter);

        CHECK(cudaDeviceSynchronize());
      }

      G.H.dt = C_cfl / h_dti;
      if (loop_counter > 2) {  // avoid excessive looping
        G.H.dt = 0.9 * C_cfl / h_dti;
      }
    }
  } while (time_direction == -1);

  if (G.Particles.n_local > 0) {
    // There was previously a comment here stating "TODO reduce cluster mass",
    // but we're already  doing this here, right?
    hipLaunchKernelGGL(Adjust_Cluster_Mass_Kernel, ngrid, TPB_FEEDBACK, 0, 0, G.Particles.n_local,
                       G.Particles.pos_x_dev, G.Particles.pos_y_dev, G.Particles.pos_z_dev, G.Particles.age_dev,
                       G.Particles.mass_dev, G.Particles.partIDs_dev, G.H.xblocal, G.H.yblocal, G.H.zblocal,
                       G.H.xblocal_max, G.H.yblocal_max, G.H.zblocal_max, G.H.dx, G.H.dy, G.H.dz, G.H.nx, G.H.ny,
                       G.H.nz, G.H.n_ghost, G.H.n_step, G.H.t, G.H.dt, dev_snr, snr_dt, time_sn_start, time_sn_end,
                       dev_sw_p, dev_sw_e, sw_dt, time_sw_start, time_sw_end);
  }

  chprintf("*******  looped %d time(s)\n", loop_counter);

  if (G.Particles.n_local > 0) {
    // copy summary data back to the host
    CHECK(cudaMemcpy(&h_info, d_info, FEED_INFO_N * sizeof(Real), cudaMemcpyDeviceToHost));

    // CLEAN UP ALLOCATIONS:
    CHECK(cudaFree(d_dti));
    CHECK(cudaFree(d_info));
    CHECK(cudaFree(d_prev_dens));
  }

  // now gather the feedback summary info into an array called info.
  #ifdef MPI_CHOLLA
  Real info[FEED_INFO_N];
  MPI_Reduce(&h_info, &info, FEED_INFO_N, MPI_CHREAL, MPI_SUM, root, world);
  #else
  Real* info = h_info;
  #endif

  #ifdef MPI_CHOLLA  // only do stats gathering on root rank
  if (procID == 0) {
  #endif

    analysis.countSN += (long)info[feedback::SN];
    analysis.countResolved += (long)info[feedback::RESOLVED];
    analysis.countUnresolved += (long)info[feedback::NOT_RESOLVED];
    analysis.totalEnergy += info[feedback::ENERGY];
    analysis.totalMomentum += info[feedback::MOMENTUM];
    analysis.totalUnresEnergy += info[feedback::UNRES_ENERGY];
    analysis.totalWindMomentum += info[i_WIND_MOMENTUM];
    analysis.totalWindEnergy += info[i_WIND_ENERGY];

    chprintf("iteration %d, t %.4e, dt %.4e", G.H.n_step, G.H.t, G.H.dt);

  #ifndef NO_SN_FEEDBACK
    Real global_resolved_ratio = 0.0;
    if (analysis.countResolved > 0 || analysis.countUnresolved > 0) {
      global_resolved_ratio = analysis.countResolved / (analysis.countResolved + analysis.countUnresolved);
    }
    chprintf(": number of SN: %d,(R: %d, UR: %d)\n", (int)info[feedback::SN], (long)info[feedback::RESOLVED],
             (long)info[feedback::NOT_RESOLVED]);
    chprintf("    cummulative: #SN: %d, ratio of resolved (R: %d, UR: %d) = %.3e\n", (long)analysis.countSN,
             (long)analysis.countResolved, (long)analysis.countUnresolved, global_resolved_ratio);
    chprintf("    sn  r energy  : %.5e erg, cumulative: %.5e erg\n", info[feedback::ENERGY] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalEnergy * FORCE_UNIT * LENGTH_UNIT);
    chprintf("    sn ur energy  : %.5e erg, cumulative: %.5e erg\n",
             info[feedback::UNRES_ENERGY] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalUnresEnergy * FORCE_UNIT * LENGTH_UNIT);
    chprintf("    sn momentum  : %.5e SM km/s, cumulative: %.5e SM km/s\n",
             info[feedback::MOMENTUM] * VELOCITY_UNIT / 1e5, analysis.totalMomentum * VELOCITY_UNIT / 1e5);
  #endif  // NO_SN_FEEDBACK

  #ifndef NO_WIND_FEEDBACK
    chprintf("    wind momentum: %.5e S.M. km/s,  cumulative: %.5e S.M. km/s\n",
             info[i_WIND_MOMENTUM] * VELOCITY_UNIT / 1e5, analysis.totalWindMomentum * VELOCITY_UNIT / 1e5);
    chprintf("    wind energy  : %.5e erg,  cumulative: %.5e erg\n", info[i_WIND_ENERGY] * FORCE_UNIT * LENGTH_UNIT,
             analysis.totalWindEnergy * FORCE_UNIT * LENGTH_UNIT);
  #endif  // NO_WIND_FEEDBACK

  #ifdef MPI_CHOLLA
  }  //   end if procID == 0
  #endif

  #ifdef CPU_TIME
  G.Timer.Feedback.End();
  #endif

  return h_dti;
}

#endif  // FEEDBACK & PARTICLES_GPU & PARTICLE_IDS & PARTICLE_AGE

/* This file defines feeback prescriptions that can actually be used within Cholla
 *
 * Some (not all) of these prescriptions are defined in terms of factored out stencils. Those stencils
 * are defined separately in a different header file.
 */

#pragma once

#include "../global/global.h"
#include "../feedback/feedback.h"
#include "../feedback/ratecalc.h"
#include "../feedback/feedback_stencil.h"

namespace feedback_model {

template<typename Stencil>
struct ResolvedSNPrescription{

  template<typename Function>
  static __device__ void for_each_possible_overlap(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                                   int nx_g, int ny_g, Function f)
  {
    Stencil::for_each(pos_x_indU, pos_y_indU, pos_z_indU, nx_g, ny_g, f);
  }

  static __device__ void apply_feedback(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU, Real vel_x, Real vel_y, Real vel_z,
                                        Real age, Real& mass_ref, part_int_t particle_id,
                                        Real dx, Real dy, Real dz, int nx_g, int ny_g, int nz_g,
                                        int n_ghost, int num_SN, Real* s_info, Real* conserved_dev)
  {
    int tid  = threadIdx.x;

    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::countSN]       += num_SN;
    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::countResolved] += num_SN;
    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalEnergy]   += feedback::ENERGY_PER_SN;

    Real dV               = dx * dy * dz;
    Real feedback_energy  = num_SN * feedback::ENERGY_PER_SN / dV;
    Real feedback_density = num_SN * feedback::MASS_PER_SN / dV;

    mass_ref -= num_SN * feedback::MASS_PER_SN; // update the cluster mass

    if (num_SN == 0)  return; // TODO: see if we can remove this!

    ResolvedSNPrescription::apply(pos_x_indU, pos_y_indU, pos_z_indU, vel_x, vel_y, vel_z,
                                  nx_g, ny_g, nx_g * ny_g * nz_g, conserved_dev,
                                  feedback_density, feedback_energy);
  }

  /* apply the resolved feedback prescription */
  static __device__ void apply(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                               Real vel_x, Real vel_y, Real vel_z,
                               int nx_g, int ny_g, int n_cells,
                               Real* conserved_device, Real feedback_density, Real feedback_energy)
  {
    Real* density    = conserved_device;
    Real* momentum_x = &conserved_device[n_cells * grid_enum::momentum_x];
    Real* momentum_y = &conserved_device[n_cells * grid_enum::momentum_y];
    Real* momentum_z = &conserved_device[n_cells * grid_enum::momentum_z];
    Real* energy     = &conserved_device[n_cells * grid_enum::Energy];
#ifdef DE
    Real* gasEnergy  = &conserved_device[n_cells * grid_enum::GasEnergy];
#endif

    Stencil::for_each(
      pos_x_indU, pos_y_indU, pos_z_indU, nx_g, ny_g,
      [=](double stencil_vol_frac, int idx3D) {
        // stencil_vol_frac is the fraction of the total stencil volume enclosed by the given cell
        // indx3D can be used to index the conserved fields (it assumes ghost-zones are present)

        // Step 1: substract off the kinetic-energy-density from total energy density.
        //  - While we aren't going to inject any of the supernova energy directly as kinetic energy,
        //    the kinetic energy density will change to some degree because the gas density and gas
        //    momentum will be changed

        Real intial_ke_density = 0.5 * (momentum_x[idx3D] * momentum_x[idx3D] +
                                        momentum_y[idx3D] * momentum_y[idx3D] +
                                        momentum_z[idx3D] * momentum_z[idx3D]) / density[idx3D];

        energy[idx3D] -= intial_ke_density;

        // Step 2: convert the momentum-density into the star's reference frame, update the density,
        //  and then update the momentum-density back into the initial reference-frame
        //  - since we aren't explicitly injecting the supernova-energy as kinetic energy, this is
        //    equivalent to adding momentum in the original frame as is done below
        double injected_density = stencil_vol_frac * feedback_density;

        momentum_x[idx3D] += vel_x * injected_density;
        momentum_y[idx3D] += vel_y * injected_density;
        momentum_z[idx3D] += vel_z * injected_density;

        // Step 2b: actually update the density
        density[idx3D] += injected_density;

        // Step 3: inject thermal energy
  #ifdef DE
        gasEnergy[idx3D] += stencil_vol_frac * feedback_energy;
  #endif
        energy[idx3D] += stencil_vol_frac * feedback_energy;

        // Step 4: reintroduce the kinetic energy density back to the total energy field
        energy[idx3D] += 0.5 * (momentum_x[idx3D] * momentum_x[idx3D] +
                                momentum_y[idx3D] * momentum_y[idx3D] +
                                momentum_z[idx3D] * momentum_z[idx3D]) / density[idx3D];
      }
    );
  }

};

using CiCResolvedSNPrescription = ResolvedSNPrescription<fb_stencil::CIC>;

using Sphere27ResolvedSNPrescription = ResolvedSNPrescription<fb_stencil::Sphere27<2>>;

using SphereBinaryResolvedSNPrescription = ResolvedSNPrescription<fb_stencil::SphereBinary<3>>;

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

inline __device__ void Set_Average_Density(int indx_x, int indx_y, int indx_z, int nx_g, int ny_g, int n_ghost, Real* density,
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
inline __device__ void Apply_Energy_Momentum_Deposition(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                                        Real vel_x, Real vel_y, Real vel_z,
                                                        int nx_g, int ny_g, int n_ghost,
                                                        int n_cells, Real* conserved_device,
                                                        Real feedback_density, Real feedback_momentum, Real feedback_energy)
{

  Real* density    = conserved_device;
  Real* momentum_x = &conserved_device[n_cells * grid_enum::momentum_x];
  Real* momentum_y = &conserved_device[n_cells * grid_enum::momentum_y];
  Real* momentum_z = &conserved_device[n_cells * grid_enum::momentum_z];
  Real* energy     = &conserved_device[n_cells * grid_enum::Energy];
#ifdef DE
  Real* gas_energy = &conserved_device[n_cells * grid_enum::GasEnergy];
#endif

  fb_stencil::LegacyCIC27 stencil{n_ghost};

  stencil.for_each_vecflavor(
    {pos_x_indU, pos_y_indU, pos_z_indU}, nx_g, ny_g,
    [=](Real scalar_weight, Arr3<Real> momentum_weights, int idx3D) {

      // precompute 1/initial_density (take care to avoid divide by 0)
      const Real inv_initial_density = 1.0 / (density[idx3D] + TINY_NUMBER * (density[idx3D] == 0.0));

      // Step 1: substract off the kinetic-energy-density from total energy density.
      //  - Regardles of whether we inject thermal energy, the kinetic energy density will change to
      //    some degree because the gas density and gas momentum will be changed

      const Real intial_ke_density = 0.5 * inv_initial_density * (momentum_x[idx3D] * momentum_x[idx3D] +
                                                                  momentum_y[idx3D] * momentum_y[idx3D] +
                                                                  momentum_z[idx3D] * momentum_z[idx3D]);
      energy[idx3D] -= intial_ke_density;

      // Step 2: convert the gas's momentum density to its value in the particle's reference frame
      //  - This must be done after subtracting off KE
      //  - This could probably be written more concisely (momentum_x[idx3D] -= density[idx3D] * vel_x),
      //    but before we do that, we should leave the 3 lines of algebra used to derive that in the
      //    comments since the abbreviated form "looks wrong" at a quick glance
      {
        // compute the local velocity
        Real gas_vx = inv_initial_density * momentum_x[idx3D];
        Real gas_vy = inv_initial_density * momentum_y[idx3D];
        Real gas_vz = inv_initial_density * momentum_z[idx3D];

        // adjust the velocity so its in the new frame
        gas_vx -= vel_x;
        gas_vy -= vel_y;
        gas_vz -= vel_z;

        // update the momentum
        momentum_x[idx3D] = density[idx3D] * gas_vx;
        momentum_y[idx3D] = density[idx3D] * gas_vy;
        momentum_z[idx3D] = density[idx3D] * gas_vz;
      }

      // step 3a: inject density, and momentum
      density[idx3D] += scalar_weight * feedback_density;
      momentum_x[idx3D] += momentum_weights[0] * feedback_momentum;
      momentum_y[idx3D] += momentum_weights[1] * feedback_momentum;
      momentum_z[idx3D] += momentum_weights[2] * feedback_momentum;

      // Step 3b: inject any thermal energy
      // - Note: its weird to be inject a fixed amount of thermal energy and momentum. This means we are
      //   injecting a variable amount of total energy...

      energy[idx3D] += scalar_weight * feedback_energy;
#ifdef DE
      gas_energy[idx3D] += scalar_weight * feedback_energy;
#endif

      // precompute 1/final_density (take care to avoid divide by 0)
      const Real inv_final_density = 1.0 / (density[idx3D] + TINY_NUMBER * (density[idx3D] == 0.0));

      // Step 4: convert the momentum back to the starting reference frame.
      //  - again, this could certainly be done more concisely
      {
        // compute the local velocity
        Real gas_vx = inv_final_density * momentum_x[idx3D];
        Real gas_vy = inv_final_density * momentum_y[idx3D];
        Real gas_vz = inv_final_density * momentum_z[idx3D];

        // adjust the velocity that it's in the original frame (it's no longer in the particle's frame)
        gas_vx += vel_x;
        gas_vy += vel_y;
        gas_vz += vel_z;

        // update the momentum
        momentum_x[idx3D] = density[idx3D] * gas_vx;
        momentum_y[idx3D] = density[idx3D] * gas_vy;
        momentum_z[idx3D] = density[idx3D] * gas_vz;
      }

      // Step 5: add the new kinetic energy density to the total_energy density field
      //  - currently the total_energy density field just holds the non-kinetic energy density
      //  - this needs to happen after changing reference frames (since KE is reference frame dependent)
      energy[idx3D] += 0.5 * inv_final_density * (momentum_x[idx3D] * momentum_x[idx3D] +
                                                  momentum_y[idx3D] * momentum_y[idx3D] +
                                                  momentum_z[idx3D] * momentum_z[idx3D]);
    }
  );
}


/*
inline __device__ void Apply_Energy_Momentum_Deposition(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU, int nx_g, int ny_g, int n_ghost,
                                                        int n_cells, Real* conserved_device,
                                                        Real feedback_density, Real feedback_momentum, Real feedback_energy)
{

  int indx_x = (int)floor(pos_x_indU - n_ghost);
  int indx_y = (int)floor(pos_y_indU - n_ghost);
  int indx_z = (int)floor(pos_z_indU - n_ghost);

  Real delta_x = (pos_x_indU - n_ghost) - indx_x;
  Real delta_y = (pos_y_indU - n_ghost) - indx_y;
  Real delta_z = (pos_z_indU - n_ghost) - indx_z;

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
        Real x_frac = fb_stencil::D_Frac(i, delta_x) * fb_stencil::Frac(j, delta_y) * fb_stencil::Frac(k, delta_z);
        Real y_frac = fb_stencil::Frac(i, delta_x) * fb_stencil::D_Frac(j, delta_y) * fb_stencil::Frac(k, delta_z);
        Real z_frac = fb_stencil::Frac(i, delta_x) * fb_stencil::Frac(j, delta_y) * fb_stencil::D_Frac(k, delta_z);

        mag += sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac);
      }
    }
  }

  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        // index in array of conserved quantities
        int indx = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;

        Real x_frac = fb_stencil::D_Frac(i, delta_x) * fb_stencil::Frac(j, delta_y) * fb_stencil::Frac(k, delta_z);
        Real y_frac = fb_stencil::Frac(i, delta_x) * fb_stencil::D_Frac(j, delta_y) * fb_stencil::Frac(k, delta_z);
        Real z_frac = fb_stencil::Frac(i, delta_x) * fb_stencil::Frac(j, delta_y) * fb_stencil::D_Frac(k, delta_z);

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

        //energy[indx] = ( momentum_x[indx] * momentum_x[indx] +
        //                 momentum_y[indx] * momentum_y[indx] +
        //                 momentum_z[indx] * momentum_z[indx] ) /
        //               2 / density[indx] + gasEnergy[indx];
      }  // k loop
    }    // j loop
  }      // i loop
}
*/

/* Legacy SNe prescription that combines resolved and unresolved */
template<typename ResolvedPrescriptionT>
struct LegacySNe {

  template<typename Function>
  static __device__ void for_each_possible_overlap(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                                   int nx_g, int ny_g, Function f)
  {
    // for right now, we are assuming that the stencil of the unresolved feedback is the same size or
    // bigger than the stencil used for the resolved feedback
    int indx_x = (int)floor(pos_x_indU);
    int indx_y = (int)floor(pos_y_indU);
    int indx_z = (int)floor(pos_z_indU);

    for (int i = -1; i < 2; i++) {
      for (int j = -1; j < 2; j++) {
        for (int k = -1; k < 2; k++) {
          const Real dummy = 0.0;
          f(dummy, (indx_x + i) + ((indx_y + j) + (indx_z + k) * ny_g)* nx_g);
        }
      }
    }

  }

  static __device__ void apply_feedback(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU, Real vel_x, Real vel_y, Real vel_z,
                                        Real age, Real& mass_ref, part_int_t particle_id,
                                        Real dx, Real dy, Real dz, int nx_g, int ny_g, int nz_g, int n_ghost,
                                        int num_SN, Real* s_info, Real* conserved_dev)
  {
    int tid  = threadIdx.x;

    Real dV = dx * dy * dz;
    int n_cells    = nx_g * ny_g * nz_g;

    // no sense doing anything if there was no SN
    if (num_SN == 0) return; // TODO: see if we can remove this!

    Real* density = conserved_dev;
    int indx_x = (int)floor(pos_x_indU - n_ghost);
    int indx_y = (int)floor(pos_y_indU - n_ghost);
    int indx_z = (int)floor(pos_z_indU - n_ghost);
    Real n_0                  = Get_Average_Number_Density_CGS(density, indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost);
    s_info[feedinfoLUT::LEN * tid + feedinfoLUT::countSN] += num_SN;

    Real feedback_energy  = num_SN * feedback::ENERGY_PER_SN / dV;
    Real feedback_density = num_SN * feedback::MASS_PER_SN / dV;

    Real shell_radius = feedback::R_SH * pow(n_0, -0.46) * pow(fabsf(num_SN), 0.29);

    bool is_resolved =  (3 * max(dx, max(dy, dz)) <= shell_radius);

    // update the cluster mass
    mass_ref -= num_SN * feedback::MASS_PER_SN;

    if (is_resolved) {
      // inject energy and density
      s_info[feedinfoLUT::LEN * tid + feedinfoLUT::countResolved] += num_SN;
      s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalEnergy]   += feedback_energy * dV;

      ResolvedPrescriptionT::apply(pos_x_indU, pos_y_indU, pos_z_indU, vel_x, vel_y, vel_z, nx_g, ny_g, n_cells,
                                   conserved_dev, feedback_density, feedback_energy);
    } else {
      // historically, only unresolved SN feedback involves averaging the densities.
      // - this step is NOT self-consistent... since this only touches the density fields, it implicitly changes the
      //   amount of thermal energy (since the kinetic energy is density dependent)
      // Real ave_dens = n_0 * MU * MP / DENSITY_UNIT;
      // Set_Average_Density(indx_x, indx_y, indx_z, nx_g, ny_g, n_ghost, density, ave_dens);

      // for now, don't inject any energy
      feedback_energy = 0.0;

      // inject momentum and density
      Real feedback_momentum = feedback::FINAL_MOMENTUM * pow(n_0, -0.17) * pow(fabsf(num_SN), 0.93) / dV / sqrt(3.0);
      s_info[feedinfoLUT::LEN * tid + feedinfoLUT::countUnresolved]  += num_SN;
      s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalMomentum]    += feedback_momentum * dV * sqrt(3.0);
      s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalUnresEnergy] += feedback_energy * dV;
      Apply_Energy_Momentum_Deposition(
          pos_x_indU, pos_y_indU, pos_z_indU, vel_x, vel_y, vel_z, nx_g, ny_g, n_ghost, n_cells, conserved_dev,
          feedback_density, feedback_momentum, feedback_energy);
    }
  }
};

/*
inline __device__ void Wind_Feedback(Real pos_x, Real pos_y, Real pos_z, Real age, Real& mass_ref, part_int_t particle_id,
                                     Real xMin, Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy,
                                     Real dz, int nx_g, int ny_g, int nz_g, int n_ghost, int n_step, Real t, Real dt,
                                     const feedback::SWRateCalc sw_calc, Real* s_info,
                                     Real* conserved_dev, Real gamma, int indx_x, int indx_y, int indx_z)
{
  int tid  = threadIdx.x;

  Real dV = dx * dy * dz;
  int n_cells    = nx_g * ny_g * nz_g;

  if ((age < 0) or not sw_calc.is_active(age)) return;
  Real feedback_momentum = sw_calc.Get_Wind_Flux(age);
  // no sense in proceeding if there is no feedback.
  if (feedback_momentum == 0) return;
  Real feedback_energy  = sw_calc.Get_Wind_Power(age);
  Real feedback_density = sw_calc.Get_Wind_Mass(feedback_momentum, feedback_energy);

  // feedback_momentum now becomes momentum component along one direction.
  feedback_momentum *= mass_ref * dt / dV / sqrt(3.0);
  feedback_density *= mass_ref * dt / dV;
  feedback_energy *= mass_ref * dt / dV;

  mass_ref   -= feedback_density * dV;

  // we log net momentum, not momentum density, and magnitude (not the
  // component along a direction)
  s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalWindMomentum] += feedback_momentum * dV * sqrt(3.0);
  s_info[feedinfoLUT::LEN * tid + feedinfoLUT::totalWindEnergy]   += feedback_energy * dV;


  const double pos_x_indU = (pos_x - xMin) / dx + n_ghost;
  const double pos_y_indU = (pos_y - yMin) / dy + n_ghost;
  const double pos_z_indU = (pos_z - zMin) / dz + n_ghost;

  Apply_Energy_Momentum_Deposition(pos_x_indU, pos_y_indU, pos_z_indU, nx_g, ny_g, n_ghost,
                                   n_cells, conserved_dev, feedback_density,
                                   feedback_momentum, feedback_energy);
}
*/

} // feedback_model namespace
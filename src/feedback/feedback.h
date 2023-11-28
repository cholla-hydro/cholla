#pragma once

  #include <functional>

  #include "../analysis/feedback_analysis.h"
  #include "../global/global.h"

namespace feedinfoLUT {
  // this enum acts like a lookup table (LUT). It maps the names of analysis statistics to
  // contiguous indices. LEN specfies the number of named analysis statistics
  enum {
    countSN = 0,
    countResolved,
    countUnresolved,
    totalEnergy,
    totalMomentum,
    totalUnresEnergy,
    totalWindMomentum,
    totalWindEnergy,
    // make sure the following is always the last entry so that it reflects the number of entries
    LEN
  };
}

namespace feedback
{
static const Real ENERGY_PER_SN = 1e51 / MASS_UNIT * TIME_UNIT * TIME_UNIT / LENGTH_UNIT / LENGTH_UNIT;
// 10 solarMasses per SN
static const Real MASS_PER_SN = 10.0;
// 2.8e5 M_s km/s * n_0^{-0.17} -> eq.(34) Kim & Ostriker (2015)
static const Real FINAL_MOMENTUM = 2.8e5 / LENGTH_UNIT * 1e5 * TIME_UNIT;
// 30.2 pc * n_0^{-0.46} -> eq.(31) Kim & Ostriker (2015)
static const Real R_SH = 0.0302;

/* construct the feedback function (or not based on the specified parameters & compilation mode)
 *
 * \note
 * we could probably define the following function regardless of the defined compiler flags */
std::function<void(Grid3D&)> configure_feedback_callback(struct parameters& P,
                                                         FeedbackAnalysis& analysis);

}  // namespace feedback

// The following is only here to simplify testing. In the future it may make sense to move it to a different header
namespace feedback_details{

/* Group together all of the particle-property arguments */
struct ParticleProps {
  part_int_t n_local;
  const part_int_t* id_dev;
  const Real* pos_x_dev;
  const Real* pos_y_dev;
  const Real* pos_z_dev;
  const Real* vel_x_dev;
  const Real* vel_y_dev;
  const Real* vel_z_dev;
  Real* mass_dev;
  const Real* age_dev;
};

/* Group together all of arguments describing the spatial field structure. */
struct FieldSpatialProps{
  Real xMin;
  Real yMin;
  Real zMin;
  Real xMax;
  Real yMax;
  Real zMax;
  Real dx;
  Real dy;
  Real dz;
  int nx_g;
  int ny_g;
  int nz_g;
  int n_ghost;
};

/* Groups properties of the simulation's current (global) iteration cycle */
struct CycleProps{
  Real t;      /*!< The current time */
  Real dt;     /*!< Size of the current timestep */
  int n_step;  /*!< This is the current step of the simulation */
};

} // feedback_details namespace
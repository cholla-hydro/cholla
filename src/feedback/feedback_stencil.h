#pragma once

#include <cstdint>

#include "../global/global.h"
#include "../feedback/feedback.h"

// I'm defining this by hand at the moment.
//
// It's not clear:
// - whether cuda/rocm provide intrinsics to speed this up.
// - whether I should just use std::clamp
template<typename T>
__device__ __host__ T clamp(T val, T lo, T hi) {
  const T tmp = val < lo ? lo : val;
  return tmp > hi ? hi : tmp;
}


/* This is kind of a placeholder right now! It's here to help reduce the number of
 * arguments we pass around. Eseentially, it's used in the case where we have 3 entries 
 * (like a mathematical vector).
 *
 * It's not clear if we should lean into the mathematical vector-concept or model this after
 * std::array, or if we should just use int3, float3, double3
 *
 * No matter what we choose to do, it's important that this is an aggregate type!
 *
 * Because this is an aggregate type, you can initi can construct it as:
 * \code{.cpp}
 *    Arr3<double> my_var{1.0, 2.0, 3.0}
 *    Arr3<float> my_var2 = {1.0, 2.0, 3.0};
 *    Arr3<int> my_var3;  // Using the default constructor. Based on a note from the docs of
 *                        // std::array, the values for a non-class type (like float/double/int)
 *                        // may be indeterminate in this case.
 * \endcode
 */
template <typename T>
struct Arr3{
  /* Tracks the values held by the class. To ensure the class is an aggregate, it needs
   * to be public. With that said, it should be treated as an implementation detail */
   
     T arr_[3];

  // constructors are implicitly defined! To ensure this class is an aggregate, these must
  // all use the default implementation!

  // destructor is implicitly defined

  // move/copy assignment operations are implicitly defined

  __device__ __host__ T& operator[](std::size_t i) {return arr_[i];}
  __device__ __host__ const T& operator[](std::size_t i) const {return arr_[i];}

  __device__ __host__ T* data() {return arr_; }

  // we may want the following:
  /*
  __device__ __host__ T& x() noexcept  {return arr[0];}
  __device__ __host__ const T& x() const noexcept {return arr[0];}
  __device__ __host__ T& y() noexcept  {return arr[1];}
  __device__ __host__ const T& y() const noexcept {return arr[1];}
  __device__ __host__ T& z() noexcept  {return arr[2];}
  __device__ __host__ const T& z() const noexcept {return arr[2];}
  */
};

enum struct StencilEvalKind{
  enclosed_stencil_vol_frac, /*!< compute the fraction of the total stencil volume enclosed by each cell */
  enclosed_cell_vol_frac,    /*!< compute the fraction of each cell's volume that is enclosed by the stencil */
  for_each_overlap_zone      /*!< execute function at each cell with any non-zero amount of overlap (but
                              *!< without specifying the amount of overlap)*/
};

// maybe this should be called feedback_stencil
namespace fb_stencil {

/* helper function used to help implement stencils calculate the nearest location to (pos_x_indU, pos_y_indU, pos_z_indU) that the
 * stencil's center can be shifted to in order to avoid overlapping with the ghost zone.
 *
 * If the specified location already does not overlap with the ghost zone, that is the returned
 *
 * \param min_stencil_offset The minimum distance a stencil must be from a cell-edge such that the stencil does not extend 
 *    past the edge.
 *
 * \note
 * It's okay for this to be a static function and live in a header since this header should only included in a single source file
 * (2 if running unit-tests).
 */
static inline __device__ Arr3<Real> nearest_noGhostOverlap_pos_(Real min_stencil_offset, Arr3<Real> pos_indU, 
                                                                int ng_x, int ng_y, int ng_z, int n_ghost)
{
  const Real edge_offset = n_ghost + min_stencil_offset;
  return { clamp(pos_indU[0], edge_offset, ng_x - edge_offset),
           clamp(pos_indU[1], edge_offset, ng_y - edge_offset),
           clamp(pos_indU[2], edge_offset, ng_z - edge_offset) };
}

/* Represents the stencil for cloud-in-cell deposition */
struct CIC {

  /* along any axis, gives the max number of neighboring cells that may be enclosed by the stencil,
   * that are on one side of the cell containing the stencil's center.
   *
   * \note
   * this primarily exists for testing purposes
   */
  inline static constexpr int max_enclosed_neighbors = 1;

  /* excute f at each location included in the stencil centered at (pos_x_indU, pos_y_indU, pos_z_indU).
   *
   * The function should expect 2 arguments: 
   *   1. ``stencil_enclosed_frac``: the fraction of the stencil enclosed by the cell
   *   2. ``indx3x``: the index used to index a 3D array (that has ghost zones)
   */
  template<typename Function>
  static __device__ void for_each(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f)
  {
    // Step 1: along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    //  - Consider the cell containing the stencil-center. If the stencil-center is at all to the left
    //    of that cell-center, then the stencil overlaps with the current cell and the one to the left
    //  - otherwise, the stencil covers the current cell and the one to the right
    int leftmost_indx_x = int(pos_indU[0] - 0.5);
    int leftmost_indx_y = int(pos_indU[1] - 0.5);
    int leftmost_indx_z = int(pos_indU[2] - 0.5);

    // Step 2: along each axis, compute the distance between the stencil-center of the leftmost cell
    //  - Recall that an integer index, ``indx``, specifies the position of the left edge of a cell.
    //    In other words the reference point of the cell is on the left edge.
    //  - The center of the cell specified by ``indx`` is actually ``indx+0.5``
    Real delta_x = pos_indU[0] - (leftmost_indx_x + 0.5);
    Real delta_y = pos_indU[1] - (leftmost_indx_y + 0.5);
    Real delta_z = pos_indU[2] - (leftmost_indx_z + 0.5);

    // Step 3: Actually invoke f at each cell-location that overlaps with the stencil location, passing both:
    //  1. fraction of the total stencil volume enclosed by the given cell
    //  2. the 1d index specifying cell-location (for a field with ghost zones)
    //
    // note: it's not exactly clear to me how we go from delta_x,delta_y,delta_z to volume-frac, (I just
    //       refactored the code I inherited and get consistent and sensible results)

    #define to_idx3D(i,j,k) ( (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k)) )

    f((1-delta_x) * (1 - delta_y) * (1 - delta_z), to_idx3D(0, 0, 0));  // (i=0, j = 0, k = 0)
    f((1-delta_x) * (1 - delta_y) *      delta_z , to_idx3D(0, 0, 1));  // (i=0, j = 0, k = 1)
    f((1-delta_x) *      delta_y  * (1 - delta_z), to_idx3D(0, 1, 0));  // (i=0, j = 1, k = 0)
    f((1-delta_x) *      delta_y  *      delta_z , to_idx3D(0, 1, 1));  // (i=0, j = 1, k = 1)
    f(   delta_x  * (1 - delta_y) * (1 - delta_z), to_idx3D(1, 0, 0));  // (i=1, j = 0, k = 0)
    f(   delta_x  * (1 - delta_y) *      delta_z , to_idx3D(1, 0, 1));  // (i=1, j = 0, k = 1)
    f(   delta_x  *      delta_y  * (1 - delta_z), to_idx3D(1, 1, 0));  // (i=1, j = 1, k = 0)
    f(   delta_x  *      delta_y  *      delta_z , to_idx3D(1, 1, 1));  // (i=1, j = 1, k = 1)
  }

  /* identical to for_each (provided for compatability with interfaces of other stencils). */
  template<typename Function>
  static __device__ void for_each_enclosedCellVol(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f)
  {
    CIC::for_each(pos_indU, nx_g, ny_g, f);
  }

  ///* calls the unary function f at ever location where there probably is non-zero overlap with
  // * the stencil.
  // *
  // * \note
  // * This is intended to be conservative (it's okay for this to call the function on a cell with
  // * non-zero overlap). The reason this exacts (rather than just calling for_each), is that it
  // * it may be significantly cheaper for some stencils
  // */
  template<typename UnaryFunction>
  static __device__ void for_each_overlap_zone(Arr3<Real> pos_indU, int ng_x, int ng_y, UnaryFunction f)
  {
    // this is a little crude!
    CIC::for_each(pos_indU, ng_x, ng_y,
                  [f](double stencil_enclosed_frac, int idx3D) { if (stencil_enclosed_frac > 0) f(idx3D);});
  }

  /* returns the nearest location to (pos_x_indU, pos_y_indU, pos_z_indU) that the stencil's center
   * can be shifted to in order to avoid overlapping with the ghost zone.
   *
   * If the specified location already does not overlap with the ghost zone, that is the returned
   * value.
   */
  static __device__ Arr3<Real> nearest_noGhostOverlap_pos(Arr3<Real> pos_indU, int ng_x, int ng_y, int ng_z, int n_ghost)
  {
    const Real min_stencil_offset = 0.5;
    return nearest_noGhostOverlap_pos_(min_stencil_offset, pos_indU, ng_x, ng_y, ng_z, n_ghost);
  }

};

namespace { // nested unnamed namespace (everything here has internal linkage)

/* implements a stencil for depositing scalar quantities into a rectangular-prism region
 * (each side-length centered on `pos_indU` that has a length of 2 cell-widths along each
 * direction.
 */
template<typename Function, StencilEvalKind flavor>
__device__ void for_each_cic27_(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f) {
  // this visits a 3x3x3 cells region

  int leftmost_indx_x = int(pos_indU[0]) - 1;
  int leftmost_indx_y = int(pos_indU[1]) - 1;
  int leftmost_indx_z = int(pos_indU[2]) - 1;

  // compute the distance between the left edge of the cell containing pos_indU and pos_indU
  Real offset_x = pos_indU[0] - int(pos_indU[0]);
  Real offset_y = pos_indU[1] - int(pos_indU[1]);
  Real offset_z = pos_indU[2] - int(pos_indU[2]);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        const int ind3D = (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k));

        // compute the length (in units of cellwidths) of the stencil that overlaps with the current
        // cell along each axis. Along the x-axis this is given by
        //       1 - offset_x: when i == 0
        //                  1: when i == 1
        //           offset_x: when i == 2
        Real x_len = (i < 2) + (i - 1) * offset_x;
        Real y_len = (j < 2) + (j - 1) * offset_y;
        Real z_len = (k < 2) + (k - 1) * offset_z;

        // The volume enclosed by the current cell is
        Real volEnclosed = x_len * y_len * z_len;

        if constexpr(flavor == StencilEvalKind::enclosed_stencil_vol_frac) {
          // the fraction of the stencil that is enclosed is volEnclosed/volStencil
          // and volStencil is 4 cell_widths^3
          f(0.25 * volEnclosed,ind3D);
        } else if constexpr (flavor == StencilEvalKind::enclosed_cell_vol_frac) {
          f(volEnclosed,ind3D);
        } else if constexpr (flavor == StencilEvalKind::for_each_overlap_zone) {
          if (volEnclosed > 0) f(ind3D);
        }
      }
    }
  }

}

// Define the legacy stencil previously used for feedback with momentum deposition
//
// I don't totally understand the logic (other than the fact that it uses 27-Cell CIC stencil).
// It has some quirks. Including the fact that the amount of scalar deposition is directly related
// to the magnitude of the vector.

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
  // I believe this is a piecwise function that does the following:
  //    (i == -1, dx <= 0.5): -1.0
  //    (i == -1, dx  > 0.5): -2*dx - 2
  //    (i ==  0,    any dx): 1 - 2*dx
  //    (i ==  1, dx <= 0.5): -1.0
  //    (i ==  1, dx  > 0.5): 2*dx - 2
  //  - elif (i == 1):  2*dx + (1 - 2*dx)*(dx > 0.5)
  return (dx > 0.5) * i * (1 - 2 * dx) + ((i + 1) * dx + 0.5 * (i - 1)) - 3 * (i - 1) * (i + 1) * (0.5 - dx);
}

} // nested unnamed namespace

struct LegacyCIC27 {

  /* along any axis, gives the max number of neighboring cells that may be enclosed by the stencil,
   * that are on one side of the cell containing the stencil's center.
   *
   * \note
   * this primarily exists for testing purposes!
   */
  inline static constexpr int max_enclosed_neighbors = 1;

  /* excute f at each location included in the stencil centered at (pos_x_indU, pos_y_indU, pos_z_indU).
   *
   * The function should expect 2 arguments: 
   *   1. ``stencil_enclosed_frac``: the fraction of the stencil enclosed by the cell
   *   2. ``indx3x``: the index used to index a 3D array (that has ghost zones)
   */
  template<typename Function>
  static __device__ void for_each(Arr3<Real> pos_indU, int nx_g, int ny_g, Function &&f)
  {
    for_each_cic27_<Function, StencilEvalKind::enclosed_stencil_vol_frac>(
      pos_indU, nx_g, ny_g, std::forward<Function>(f));
  }

  /* excute ``f`` at each location included in the stencil centered at (pos_x_indU, pos_y_indU, pos_z_indU).
   *
   * This is just like for_each, except that it passes the fraction of the cell-volume that is enclosed
   * by the stencil to ``f`` (instead of passing fraction of the stencil-volume enclosed by the cell).
   *
   * \note
   * This is primarily intended for testing purposes.
   */
  template<typename Function>
  static __device__ void for_each_enclosedCellVol(Arr3<Real> pos_indU, int nx_g, int ny_g, Function&& f)
  {
    for_each_cic27_<Function, StencilEvalKind::enclosed_cell_vol_frac>(
      pos_indU, nx_g, ny_g, std::forward<Function>(f));
  }

  /* excute f at each location included in the stencil centered at (pos_x_indU, pos_y_indU, pos_z_indU).
   *
   * The function should expect 3 arguments (it's not totally clear to what the first 2 arguments truly "mean",
   * but they are used similarly to the corresponding arguments passed by other kernels' for_each_vecflavor): 
   *   1. ``scalar_weight``: multiply this by the scalar to determine how much scalar to inject
   *   2. ``vec_comp_factor``: multiply each by a momentumvelocity component to get the amount of momentum to inject.
   *   2. ``indx3x``: the index used to index a 3D array (that has ghost zones)
   */
  template<typename Function>
  static __device__ void for_each_vecflavor(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f)
  {
    const Real pos_x_indU = pos_indU[0];
    const Real pos_y_indU = pos_indU[1];
    const Real pos_z_indU = pos_indU[2];

    int indx_x = (int)floor(pos_x_indU);
    int indx_y = (int)floor(pos_y_indU);
    int indx_z = (int)floor(pos_z_indU);

    Real delta_x = pos_x_indU - indx_x;
    Real delta_y = pos_y_indU - indx_y;
    Real delta_z = pos_z_indU - indx_z;

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

    Real inv_mag = 1.0 / mag;

    for (int i = -1; i < 2; i++) {
      for (int j = -1; j < 2; j++) {
        for (int k = -1; k < 2; k++) {
          // index in array of conserved quantities
          int indx = (indx_x + i) + (indx_y + j) * nx_g + (indx_z + k) * nx_g * ny_g;

          Real x_frac = D_Frac(i, delta_x) * Frac(j, delta_y) * Frac(k, delta_z);
          Real y_frac = Frac(i, delta_x) * D_Frac(j, delta_y) * Frac(k, delta_z);
          Real z_frac = Frac(i, delta_x) * Frac(j, delta_y) * D_Frac(k, delta_z);
          Real scalar_weight = sqrt(x_frac * x_frac + y_frac * y_frac + z_frac * z_frac) * inv_mag;
          Arr3<Real> momentum_weights{x_frac * inv_mag, y_frac * inv_mag, z_frac * inv_mag};

          f(scalar_weight, momentum_weights, indx);

        }  // k loop
      }    // j loop
    }      // i loop

  }


  /* calls the unary function f at ever location where there probably is non-zero overlap with
   * the stencil.
   *
   * \note
   * This is intended to be conservative (it's okay for this to call the function on a cell with
   * non-zero overlap). The reason this exacts (rather than just calling for_each), is that it
   * it may be significantly cheaper for some stencils
   */
  template<typename UnaryFunction>
  static __device__ void for_each_overlap_zone(Arr3<Real> pos_indU, int nx_g, int ny_g, UnaryFunction f)
  {
    for_each_cic27_<UnaryFunction, StencilEvalKind::for_each_overlap_zone>(
      pos_indU, nx_g, ny_g, std::forward<UnaryFunction>(f));
  }


  /* returns the nearest location to (pos_x_indU, pos_y_indU, pos_z_indU) that the stencil's center
   * can be shifted to in order to avoid overlapping with the ghost zone.
   *
   * If the specified location already does not overlap with the ghost zone, that is the returned
   * value.
   */
  static __device__ Arr3<Real> nearest_noGhostOverlap_pos(Arr3<Real> pos_indU, int ng_x, int ng_y, int ng_z, int n_ghost)
  {
    const Real min_stencil_offset = 1.0; // I think this is right, I'm a little fuzzy on the precised
    return nearest_noGhostOverlap_pos_(min_stencil_offset, pos_indU, ng_x, ng_y, ng_z, n_ghost);
  }

};










/* Represents a sphere. This is used to help implement stencils. */
struct SphereObj{

public:  // attributes
  double center_indU[3]; /*!< center of the sphere (in index units). An integer value corresponds to a cell-edge.
                          *!< Integer-values plus 0.5 correspond to cell-centers*/
  int raidus2_indU; /*!< squared radius of the sphere (in units of cell-widths)*/

public:  // interface
  /* queries whether the sphere encloses a given point */
  __forceinline__ __device__ bool encloses_point(double pos_x_indU, double pos_y_indU, double pos_z_indU) const {
    double delta_x = pos_x_indU - center_indU[0];
    double delta_y = pos_y_indU - center_indU[1];
    double delta_z = pos_z_indU - center_indU[2];

    return (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) < raidus2_indU; 
  }

  /* queries whether the sphere encloses any super-sampled points within a cell that correspond to integer indices of
   * (cell_idx_x, cell_idx_y, cell_idx_z).
   *
   *  \tparam Log2DivsionsPerAx parameterizes the amount of super-sampling. There are ``2^Log2DivsionsPerAx`` equidistant 
   *       points along each axis of the algorithm. In other words, this
   *       can return a max value of ``2^(Log2DivsionsPerAx_PerCell*3)``.
   */
  template<int Log2DivsionsPerAx>
  __device__ bool Encloses_Any_Supersample(int cell_idx_x, int cell_idx_y, int cell_idx_z) const {

    // compute some basic information for the algorithm
    // - we employ ternary conditionals to avoid functions-calls/floating-point operations for the most
    //   common choice of Log2DivsionsPerAx
    // - since Log2DivsionsPerAx is a template-arg, these branches should be compiled away
    // - we could probably be a little more clever here
    const int num_subdivisions_per_ax    = (Log2DivsionsPerAx == 2) ?     4 : std::pow(2,Log2DivsionsPerAx);
    const double subgrid_width           = (Log2DivsionsPerAx == 2) ?  0.25 : 1.0 / num_subdivisions_per_ax;
    const double leftmost_subgrid_offset = (Log2DivsionsPerAx == 2) ? 0.125 : 0.5 * subgrid_width;

    // the following is mathematically equivalent to 1-leftmost_subgrid_offset, but we do the following to try to
    // have consistent rounding with other supersampling calculations
    const double rightmost_subgrid_offset = leftmost_subgrid_offset + ((num_subdivisions_per_ax-1) * subgrid_width);

    // IMPLICIT ASSUMPTION is that the radius is 1 cell-width or larger

    double dx_left  = center_indU[0] - (cell_idx_x + leftmost_subgrid_offset);
    double dx_right = center_indU[0] - (cell_idx_x + rightmost_subgrid_offset);
    double dy_left  = center_indU[1] - (cell_idx_y + leftmost_subgrid_offset);
    double dy_right = center_indU[1] - (cell_idx_y + rightmost_subgrid_offset);
    double dz_left  = center_indU[2] - (cell_idx_z + leftmost_subgrid_offset);
    double dz_right = center_indU[2] - (cell_idx_z + rightmost_subgrid_offset);

    double min_squared_dist = (fmin(dx_left * dx_left, dx_right * dx_right) +
                               fmin(dy_left * dy_left, dy_right * dy_right) +
                               fmin(dz_left * dz_left, dz_right * dz_right));
    return min_squared_dist < raidus2_indU;
  }

  /* returns the count of the number of super-sampled points within a cell that correspond to integer indices of
   * (cell_idx_x, cell_idx_y, cell_idx_z).
   *
   *  \tparam Log2DivsionsPerAx parameterizes the amount of super-sampling. The super-sampling algorithm checks
   *       ``2^Log2DivsionsPerAx`` equidistant points along each axis of the algorithm. In other words, this
   *       can return a max value of ``2^(Log2DivsionsPerAx_PerCell*3)``.
   *
   * \note
   * In the context of this function, integer indices specify the left edge of a cell. An integer index + 0.5
   * specifies the center of a cell.
   *
   * \note
   * None of the super-samples are placed on the edges of the cells.
   */
  template<int Log2DivsionsPerAx>
  __device__ unsigned int Count_Super_Samples(int cell_idx_x, int cell_idx_y, int cell_idx_z) const
  {
    static_assert((0 <= Log2DivsionsPerAx) and ((Log2DivsionsPerAx*3) <= (8*sizeof(unsigned int))),
                  "Log2DivsionsPerAx must be a non-negative integer AND 2^(Log2DivsionsPerAx*3), the total "
                  "number of super-samples in a given cell, must be representable by an unsigned int");

    // compute some basic information for the algorithm
    // - we employ ternary conditionals to avoid functions-calls/floating-point operations for the most
    //   common choice of Log2DivsionsPerAx
    // - since Log2DivsionsPerAx is a template-arg, these branches should be compiled away
    // - we could probably be a little more clever here
    const int num_subdivisions_per_ax    = (Log2DivsionsPerAx == 2) ?     4 : std::pow(2,Log2DivsionsPerAx);
    const double subgrid_width           = (Log2DivsionsPerAx == 2) ?  0.25 : 1.0 / num_subdivisions_per_ax;
    const double leftmost_subgrid_offset = (Log2DivsionsPerAx == 2) ? 0.125 : 0.5 * subgrid_width;

    unsigned int count = 0;
    for (int ix = 0; ix < num_subdivisions_per_ax; ix++) {
      for (int iy = 0; iy < num_subdivisions_per_ax; iy++) {
        for (int iz = 0; iz < num_subdivisions_per_ax; iz++) {
          // since cell_idx_x, cell_idx_y, cell_idx_z are all integers, they specify
          // the position of the left edge of the cell
          double x = cell_idx_x + (leftmost_subgrid_offset + ix * subgrid_width);
          double y = cell_idx_y + (leftmost_subgrid_offset + iy * subgrid_width);
          double z = cell_idx_z + (leftmost_subgrid_offset + iz * subgrid_width);

          count += encloses_point(x, y, z);
        }
      }
    }

    return count;
  }

  /* Estimates the volume integral over the overlapping region of a cell of the radial unit-vector measured from
   * ``ref_pos_IndU``. Specifically, the cell corresponds to integer indices of (cell_idx_x, cell_idx_y, cell_idx_z)
   * and the integral makes use of super-sampling.
   *
   * The result has units of subcell-volume. To convert it to units of cell-volume multiply by `pow(2,-3*Log2DivsionsPerAx)`
   *
   * In more detail, The evaluated integral looks like:
   * \f[
   *   \hat{x}\int (\hat{x} \cdot\hat{r})\  dV_{\rm cell} + \hat{y}\int (\hat{y} \cdot\hat{r})\  dV_{\rm cell} + 
   *   \hat{z}\int (\hat{z} \cdot\hat{r})\  dV_{\rm cell}
   * \f]
   * Where the bounds of the integral are understood to only include the portion of the region of the specified sphere
   * that overlaps with the sphere. Additionally, \f$ \hat{r} \f$ is the radial unit vector measured after transforming
   * the coordinate-system so that the origin coincides with the ``ref_pos_IndU`` argument.
   *
   * The calculation makes 2 assumptions:
   *   1. We assume that subcells are either entirely enclosed by the sphere or aren't enclosed at all
   *   2. Throughout a given subcell, \f$ \hat{r} \f$ is constant; it's equal to the value at the center of the subcell.
   *      (Note: it would be possible to avoid assumption. There is an exact analytic solution, it's just very involved).
   *
   * Under these assumptions the evaluated integral becomes:
   * \f[
   *   V_{\rm subcell}\sum_{ijk}^{\rm subcells} \frac{W_{ijk} (x_i \hat{x} + y_j \hat{y} + z_k\hat{z})}{r_{ijk}}
   * \f]
   * where the subscripted variables are computed at the center of each subcell. \f$ W_{ijk} \f$ has a 
   * value of 1 for subcells whose centers lie within the sphere and are zero in other cases
   *
   *  \tparam Log2DivsionsPerAx parameterizes the amount of super-sampling. The super-sampling algorithm checks
   *       ``2^Log2DivsionsPerAx`` equidistant points along each axis of the algorithm. In other words, this 
   *       can return a max value of ``2^(Log2DivsionsPerAx_PerCell*3)``.
   *
   * \note
   * In the context of this function, integer indices specify the left edge of a cell. An integer index + 0.5 
   * specifies the center of a cell.
   *
   * \note
   * None of the super-samples are placed on the edges of the cells.
   */
  template<int Log2DivsionsPerAx>
  __device__ Arr3<Real> Super_Sampled_RadialUnitVec_VolIntegral(int cell_idx_x, int cell_idx_y, int cell_idx_z,
                                                                const Arr3<Real> ref_pos_IndU) const
  {
    static_assert((0 <= Log2DivsionsPerAx) and ((Log2DivsionsPerAx*3) <= (8*sizeof(unsigned int))),
                  "Log2DivsionsPerAx must be a non-negative integer AND 2^(Log2DivsionsPerAx*3), the total "
                  "number of super-samples in a given cell, must be representable by an unsigned int");

    // compute some basic information for the algorithm
    // - we employ ternary conditionals to avoid functions-calls/floating-point operations for the most
    //   common choice of Log2DivsionsPerAx
    // - since Log2DivsionsPerAx is a template-arg, these branches should be compiled away
    // - we could probably be a little more clever here
    const int num_subdivisions_per_ax    = (Log2DivsionsPerAx == 2) ?     4 : std::pow(2,Log2DivsionsPerAx);
    const double subgrid_width           = (Log2DivsionsPerAx == 2) ?  0.25 : 1.0 / num_subdivisions_per_ax;
    const double leftmost_subgrid_offset = (Log2DivsionsPerAx == 2) ? 0.125 : 0.5 * subgrid_width;

    Arr3<Real> out{0.0, 0.0, 0.0};

    for (int ix = 0; ix < num_subdivisions_per_ax; ix++) {
      for (int iy = 0; iy < num_subdivisions_per_ax; iy++) {
        for (int iz = 0; iz < num_subdivisions_per_ax; iz++) {
          // since cell_idx_x, cell_idx_y, cell_idx_z are all integers, they specify
          // the position of the left edge of the cell

          // compute the center of the subcell
          const double orig_frame_x = cell_idx_x + (leftmost_subgrid_offset + ix * subgrid_width);
          const double orig_frame_y = cell_idx_y + (leftmost_subgrid_offset + iy * subgrid_width);
          const double orig_frame_z = cell_idx_z + (leftmost_subgrid_offset + iz * subgrid_width);

          const bool subcell_enclosed_by_sphere = encloses_point(orig_frame_x, orig_frame_y, orig_frame_z);

          // compute the x, y, and z components in the coordinate system that has been translated so that
          // ref_pos_IndU coincides with the origin
          const double x = orig_frame_x - ref_pos_IndU[0];
          const double y = orig_frame_y - ref_pos_IndU[1];
          const double z = orig_frame_z - ref_pos_IndU[2];

          // we explicitly use bitwise operators here for speed purposes
          const bool coincides_with_origin = ((x == 0.0) & (y == 0.0) & (z==0.0));

          // for r = sqrt((x*x) + (y*y) + (z*z)), we need to compute x/r, y/r, z/r.
          // - we add coincides_with_origin here to make sure we don't divide by zero if (x,y,z) = (0,0,0)
          const double inv_r_mag = 1.0 / (coincides_with_origin + sqrt((x*x) + (y*y) + (z*z)));

          out[0] += subcell_enclosed_by_sphere * x * inv_r_mag;
          out[1] += subcell_enclosed_by_sphere * y * inv_r_mag;
          out[2] += subcell_enclosed_by_sphere * z * inv_r_mag;
        }
      }
    }
    
    return out;
  }

};

namespace { // nested unnamed namespace (everything here has internal linkage)

/* implements a stencil for depositing scalar quantities into a rectangular-prism region
 * (each side-length centered on `pos_indU` that has a length of 2 cell-widths along each
 * direction.
 */
template<typename Function, StencilEvalKind flavor, int CellsPerDiameter, int Log2DivsionsPerAx_PerCell>
__forceinline__ __device__ void for_each_sphere_(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f) {

  const SphereObj sphere{/* center = */ {pos_indU[0], pos_indU[1], pos_indU[2]},
                         /* squared_radius = */ 1*1};
  const Real l_offset = ((CellsPerDiameter % 2) == 0) ? CellsPerDiameter / 2 : 0.5 * CellsPerDiameter;

  // Step 1: along each axis, identify the integer-index of the leftmost cell covered by the stencil.
  const int leftmost_indx_x = int(pos_indU[0] - l_offset);
  const int leftmost_indx_y = int(pos_indU[1] - l_offset);
  const int leftmost_indx_z = int(pos_indU[2] - l_offset);

  static_assert(CellsPerDiameter == 2); // this is temporary!

  // Step 2: get the number of super-samples within each of the 27 possible cells (This is not
  //         actually necessary for some stencil evaluation-flavors)

  // Step 2a: declare variables used to accumulate the total count and to act as a cache
  //          for tracking the number of super-sample per cell
  // -> we label these with [[maybe_unused]] attribute to suppress warnings for the flavors where
  //    these variables aren't used.
  // -> for applicable "flavors", the compiler should optimize out the unusued variables.
  // -> we want to keep the array-element size small for the cached_counts variable in order to
  //    reduce memory-pressure on the stack (especially since every thread will be allocating this
  //    much stack-space at the same time)
  [[maybe_unused]] unsigned long total_count = 0;
  [[maybe_unused]] uint_least16_t cached_counts[3][3][3];

  // Step 2b: actually get the number of supersamples
  if constexpr (flavor == StencilEvalKind::enclosed_stencil_vol_frac) {

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          unsigned int count = sphere.Count_Super_Samples<Log2DivsionsPerAx_PerCell>(leftmost_indx_x + i,
                                                                                     leftmost_indx_y + j,
                                                                                     leftmost_indx_z + k);
          cached_counts[i][j][k] = std::uint_least16_t(count);
          total_count += count;
        }
      }
    }

  }

  //kernel_printf("ref: %g, %g, %g\n", pos_indU[0], pos_indU[1], pos_indU[2]);

  // Step 3: actually invoke f at each cell-location that overlaps with the stencil location
  //    (for flavors where we specify some kind of enclosed volume as a function argument,
  //    its okay to specify the function at a location without any overlap)
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        const int indx_x = leftmost_indx_x + i;
        const int indx_y = leftmost_indx_y + j;
        const int indx_z = leftmost_indx_z + k;
        const int ind3D = indx_x + nx_g * (indx_y + ny_g * indx_z);

        //kernel_printf("%d, %d, %d: %g\n", leftmost_indx_x + i, (leftmost_indx_y + j), (leftmost_indx_z + k),
        //              double(counts[i][j][k])/total_count);

        if constexpr (flavor == StencilEvalKind::enclosed_stencil_vol_frac) {
          // pass both of the following to the function
          //  1. fraction of the total stencil volume enclosed by the given cell
          //  2. the 1d index specifying cell-location (for a field with ghost zones)
          f(double(cached_counts[i][j][k])/total_count, ind3D);

        } else if constexpr (flavor == StencilEvalKind::enclosed_cell_vol_frac) {
          // pass both of the following to the function:
          // 1. pass the fraction of the cell-volume that is enclosed by the stencil
          // 2. the 1d index specifying cell-location

          // it would nominally make more sense to precompute the following outside of this loop,
          // but that's probably fine (after all, this branch is mostly for testing purposes)
          double inverse_max_counts_per_cell = 1.0 / double(std::pow(2,Log2DivsionsPerAx_PerCell*3));

          // in this case, we have not precomputed the compute
          unsigned int count = sphere.Count_Super_Samples<Log2DivsionsPerAx_PerCell>(indx_x,
                                                                                     indx_y,
                                                                                     indx_z);
          f(count*inverse_max_counts_per_cell, ind3D);

        } else if constexpr (flavor == StencilEvalKind::for_each_overlap_zone) {

          bool is_enclosed = sphere.Encloses_Any_Supersample<Log2DivsionsPerAx_PerCell>(indx_x,
                                                                                        indx_y,
                                                                                        indx_z);
          if (is_enclosed) f(ind3D);
        }

      }
    }
  }

}

} // nested unnamed namespace

/* Represents a 27-cell deposition stencil for a sphere with a radius of 1 cell-width. This stencil computes
 * the fraction of the stencil that is enclosed in each cell. The overlap between the stencil and a given cell
 * is computed via super-sampling.
 *
 * \tparam Log2DivsionsPerAx_PerCell parameterizes the amount of super-sampling. For a given cell, the super-sampling 
 *    algorithm the number of subgrid-points enclosed by the stencil; there are ``2^Log2DivsionsPerAx_PerCell``
 *    sub-grid points along each axis. In other words, there are ``2^(Log2DivsionsPerAx_PerCell*3)`` subgrid-points
 *    per cell.
 */
template<int Log2DivsionsPerAx_PerCell = 2>
struct Sphere27 {

  static_assert((Log2DivsionsPerAx_PerCell >= 0) and (Log2DivsionsPerAx_PerCell <= 5),
                "Log2DivsionsPerAx_PerCell must be a non-negative integer. It also can't exceed 5 "
                "so that 2^(Log2DivsionsPerAx_PerCell*3) can be represented by uint16_t");

  /* along any axis, gives the max number of neighboring cells that may be enclosed by the stencil,
   * that are on one side of the cell containing the stencil's center.
   *
   * \note
   * this primarily exists for testing purposes!
   */
  inline static constexpr int max_enclosed_neighbors = 1;

  /* excute f at each location included in the stencil centered at (pos_x_indU, pos_y_indU, pos_z_indU).
   *
   * The function should expect 2 arguments: 
   *   1. ``stencil_enclosed_frac``: the fraction of the total stencil volume enclosed by the cell
   *   2. ``indx3x``: the index used to index a 3D array (that has ghost zones)
   */
  template<typename Function>
  static __device__ void for_each(Arr3<Real> pos_indU, int nx_g, int ny_g, Function &&f)
  {
    for_each_sphere_<Function, StencilEvalKind::enclosed_stencil_vol_frac, 2, Log2DivsionsPerAx_PerCell>(
      pos_indU, nx_g, ny_g, std::forward<Function>(f));
  }

  /* excute f at each location included in the stencil centered at (pos_x_indU, pos_y_indU, pos_z_indU).
   *
   * The function should expect 3 arguments: 
   *   1. ``stencil_enclosed_frac``: the fraction of the total stencil volume enclosed by the cell. In other
   *      words, its the volume of the cell enclosed by the stencil divided by the total stencil volume.
   *   2. ``vec_comp_factor``: a `Arr3<Real>` where the elements represent math-vector components (x, y, z).
   *      Essentially, this stores the volume integral (of the region enclosed by the stencil) over the 
   *      radial-unit vector (originating from the stencil center) divided by a normalization constant.
   *      - The normalization constant is computed by taking the sum of each volume-integrated radial-unit 
   *        vector computed at each cell enclosed by the stencil
   *      - The alternative would be to just normalize by the total volume. The problem with this alternative
   *        is if you are trying to inject a constant amount of radial momentum per unit-volume, then
   *        cancelation in the most-central cell may cause you to underinject momementum (primarily in the
   *        case where the stencil is near the center of a cell)
   *   2. ``indx3x``: the index used to index a 3D array (that has ghost zones)
   */
  template<typename Function>
  static __device__ void for_each_vecflavor(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f)
  {
    // Step 1: along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    const int leftmost_indx_x = int(pos_indU[0] - 1);
    const int leftmost_indx_y = int(pos_indU[1] - 1);
    const int leftmost_indx_z = int(pos_indU[2] - 1);

    // Step 2: get the number of super-samples within each of the 27 possible cells
    const SphereObj sphere{{pos_indU[0], pos_indU[1], pos_indU[2]}, 1*1};

    // we intentionally keep the array-element size to reduce memory pressure on the stack (especially 
    // since every thread will be allocating this much stack-space at the same time)
    // - If we weren't concerned about memory-pressure (e.g. we used cooperative_groups), we could
    //   save time and consolidate the calculation of integrated vector components and the enclosed
    //   volume into a single operation)
    uint_least16_t cached_counts[3][3][3];

    unsigned long total_count = 0;
    Real vector_norm = 0.0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          unsigned int cur_count = sphere.Count_Super_Samples<Log2DivsionsPerAx_PerCell>(
            leftmost_indx_x + i, leftmost_indx_y + j, leftmost_indx_z + k);
          total_count += cur_count;  // update total_count
          cached_counts[i][j][k] = std::uint_least16_t(cur_count);  // cache the value of 

          const Arr3<Real> integrated_vec = sphere.Super_Sampled_RadialUnitVec_VolIntegral<Log2DivsionsPerAx_PerCell>(
            leftmost_indx_x + i, leftmost_indx_y + j, leftmost_indx_z + k, pos_indU);
          vector_norm += norm3d(integrated_vec[0], integrated_vec[1], integrated_vec[2]);
          // we don't cache the value of integrated_vec... That would put a LOT of strain on registers
        }
      }
    }

    const Real vec_factor = 1.0 / vector_norm;

    // Step 3: actually invoke f at each cell-location that overlaps with the stencil location, passing both:
    //  1. fraction of the total stencil volume enclosed by the given cell
    //  2. the volume integral (of the region enclosed by the stencil) over the  radial-unit vector (originating 
    //     from the stencil center) divided by the total stencil volume
    //  3. the 1d index specifying cell-location (for a field with ghost zones)
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          const int ind3D = (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k));

          //kernel_printf("%d, %d, %d: %g\n", leftmost_indx_x + i, (leftmost_indx_y + j), (leftmost_indx_z + k),
          //              double(counts[i][j][k])/total_count);

          // this has units of subcell volume. We need to divide it by total_count before passing it along
          const Arr3<Real> tmp = sphere.Super_Sampled_RadialUnitVec_VolIntegral<Log2DivsionsPerAx_PerCell>(
            leftmost_indx_x + i, leftmost_indx_y + j, leftmost_indx_z + k, pos_indU);

          f(double(cached_counts[i][j][k])/total_count,
            Arr3<Real>{tmp[0] * vec_factor, tmp[1]*vec_factor, tmp[2]*vec_factor},
            ind3D);
        }
      }
    }

  }

  /* excute ``f`` at each location included in the stencil centered at (pos_x_indU, pos_y_indU, pos_z_indU).
   *
   * This is just like for_each, except that it passes the fraction of the cell-volume that is enclosed
   * by the stencil to ``f`` (instead of passing fraction of the stencil-volume enclosed by the cell).
   *
   * \note
   * This is primarily intended for testing purposes.
   */
  template<typename Function>
  static __device__ void for_each_enclosedCellVol(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f)
  {
    for_each_sphere_<Function, StencilEvalKind::enclosed_cell_vol_frac, 2, Log2DivsionsPerAx_PerCell>(
      pos_indU, nx_g, ny_g, std::forward<Function>(f));
  }

  /* calls the unary function f at ever location where there probably is non-zero overlap with
   * the stencil.
   *
   * \note
   * This is is significantly cheaper than calling for_each.
   */
  template<typename UnaryFunction>
  static __device__ void for_each_overlap_zone(Arr3<Real> pos_indU, int nx_g, int ny_g, UnaryFunction &&f)
  {
    for_each_sphere_<UnaryFunction, StencilEvalKind::for_each_overlap_zone, 2, Log2DivsionsPerAx_PerCell>(
      pos_indU, nx_g, ny_g, std::forward<UnaryFunction>(f));
  }

  /* returns the nearest location to (pos_x_indU, pos_y_indU, pos_z_indU) that the stencil's center
   * can be shifted to in order to avoid overlapping with the ghost zone.
   *
   * If the specified location already does not overlap with the ghost zone, that is the returned
   * value.
   */
  static __device__ Arr3<Real> nearest_noGhostOverlap_pos(Arr3<Real> pos_indU, int ng_x, int ng_y, int ng_z, int n_ghost)
  {
    // we actually provide an alternative more clever. technically, we just can't overlap with nearest
    // super-sampled point inside of ghost zone. Any other amount of overlap is fair game!
    constexpr Real min_stencil_offset = 1.0;
    return nearest_noGhostOverlap_pos_(min_stencil_offset, pos_indU, ng_x, ng_y, ng_z, n_ghost);
  }

};


/* Represents a spherical stencil with a radius of 3 cells, where the inclusion where inclusion of
 * cells in the sphere is a binary choice.
 *
 * Specifically, a cell is included if the cell-center lies within the sphere.
 */
template<int CellsPerRadius = 3>
struct SphereBinary {
  static_assert(CellsPerRadius > 0);

  /* along any axis, gives the max number of neighboring cells that may be enclosed by the stencil,
   * that are on one side of the cell containing the stencil's center.
   *
   * \note
   * this primarily exists for testing purposes
   */
  inline static constexpr int max_enclosed_neighbors = CellsPerRadius;

  template<typename Function>
  static __device__ void for_each(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f)
  {
    // Step 1: along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    int leftmost_indx_x = int(pos_indU[0]) - CellsPerRadius;
    int leftmost_indx_y = int(pos_indU[1]) - CellsPerRadius;
    int leftmost_indx_z = int(pos_indU[2]) - CellsPerRadius;

    // Step 2: get the number of cells enclosed by the sphere
    const SphereObj sphere{{pos_indU[0], pos_indU[1], pos_indU[2]}, CellsPerRadius*CellsPerRadius};
    int total_count = 0;

    const int stop = (2 * CellsPerRadius) + 1;
    for (int i = 0; i < stop; i++) {
      for (int j = 0; j < stop; j++) {
        for (int k = 0; k < stop; k++) {
          total_count += sphere.encloses_point(leftmost_indx_x + i + 0.5,
                                               leftmost_indx_y + j + 0.5,
                                               leftmost_indx_z + k + 0.5);
        }
      }
    }

    double enclosed_stencil_frac = 1.0/total_count;  // each enclosed cell, encloses this fraction of the sphere

    // Step 3: actually invoke f at each cell-location that overlaps with the stencil location, passing both:
    //  1. fraction of the total stencil volume enclosed by the given cell
    //  2. the 1d index specifying cell-location (for a field with ghost zones)
    for (int i = 0; i < stop; i++) {
      for (int j = 0; j < stop; j++) {
        for (int k = 0; k < stop; k++) {
          bool is_enclosed = sphere.encloses_point(leftmost_indx_x + i + 0.5, leftmost_indx_y + j + 0.5, leftmost_indx_z + k + 0.5);
          //kernel_printf("(%d, %d, %d), enclosed: %d\n", i,j,k, is_enclosed);
          if (is_enclosed){
            const int ind3D = (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k));
            f(enclosed_stencil_frac, ind3D);
          }

        }
      }
    }

  }


  /* excute ``f`` at each location included in the stencil centered at (pos_x_indU, pos_y_indU, pos_z_indU).
   *
   * This is just like for_each, except that it passes the fraction of the cell-volume that is enclosed
   * by the stencil to ``f`` (instead of passing fraction of the stencil-volume enclosed by the cell).
   *
   * \note
   * This is primarily intended for testing purposes.
   */
  template<typename Function>
  static __device__ void for_each_enclosedCellVol(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f)
  {
    // along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    int leftmost_indx_x = int(pos_indU[0]) - CellsPerRadius;
    int leftmost_indx_y = int(pos_indU[1]) - CellsPerRadius;
    int leftmost_indx_z = int(pos_indU[2]) - CellsPerRadius;

    const SphereObj sphere{{pos_indU[0], pos_indU[1], pos_indU[2]}, CellsPerRadius*CellsPerRadius};

    const int stop = (2 * CellsPerRadius) + 1;
    for (int i = 0; i < stop; i++) {
      for (int j = 0; j < stop; j++) {
        for (int k = 0; k < stop; k++) {
          bool is_enclosed = sphere.encloses_point(leftmost_indx_x + i + 0.5,
                                                   leftmost_indx_y + j + 0.5,
                                                   leftmost_indx_z + k + 0.5);
          double enclosed_cell_vol = (is_enclosed) ? 1.0 : 0.0;  // could just cast is_enclosed
          const int ind3D = (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k));
          f(enclosed_cell_vol, ind3D);
        }
      }
    }
  }

  template<typename UnaryFunction>
  static __device__ void for_each_overlap_zone(Arr3<Real> pos_indU, int ng_x, int ng_y, UnaryFunction f)
  {
    // along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    int leftmost_indx_x = int(pos_indU[0]) - CellsPerRadius;
    int leftmost_indx_y = int(pos_indU[1]) - CellsPerRadius;
    int leftmost_indx_z = int(pos_indU[2]) - CellsPerRadius;
  
    const SphereObj sphere{/* center = */ {pos_indU[0], pos_indU[1], pos_indU[2]},
                           /* squared_radius = */ CellsPerRadius*CellsPerRadius}; 

    const int stop = (2 * CellsPerRadius) + 1;
    for (int i = 0; i < stop; i++) {
      for (int j = 0; j < stop; j++) {
        for (int k = 0; k < stop; k++) {
          const int indx_x = leftmost_indx_x + i;
          const int indx_y = leftmost_indx_y + j;
          const int indx_z = leftmost_indx_z + k;
          const int ind3D = indx_x + ng_x * (indx_y + ng_y * indx_z);
          bool is_enclosed = sphere.encloses_point(leftmost_indx_x + i + 0.5,
                                                   leftmost_indx_y + j + 0.5,
                                                   leftmost_indx_z + k + 0.5);
          if (is_enclosed) f(ind3D);
        }
      }
    }

  }

  /* returns the nearest location to (pos_x_indU, pos_y_indU, pos_z_indU) that the stencil's center
   * can be shifted to in order to avoid overlapping with the ghost zone.
   *
   * If the specified location already does not overlap with the ghost zone, that is the returned
   * value.
   */
  static __device__ Arr3<Real> nearest_noGhostOverlap_pos(Arr3<Real> pos_indU, int ng_x, int ng_y, int ng_z, int n_ghost)
  {
    // we actually provide an alternative more clever implementation. technically, we just can't overlap with the center 
    // of a cell in the ghost-zone. Any other amount of overlap is fair game!
    constexpr Real min_stencil_offset = CellsPerRadius;
    return nearest_noGhostOverlap_pos_(min_stencil_offset, pos_indU, ng_x, ng_y, ng_z, n_ghost);
  }

};


} // fb_stencil namespace
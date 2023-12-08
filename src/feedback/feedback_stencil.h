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

// maybe this should be called feedback_stencil
namespace fb_stencil {

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
  static __device__ void for_each(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                  int nx_g, int ny_g, Function f)
  {
    // Step 1: along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    //  - Consider the cell containing the stencil-center. If the stencil-center is at all to the left
    //    of that cell-center, then the stencil overlaps with the current cell and the one to the left
    //  - otherwise, the stencil covers the current cell and the one to the right
    int leftmost_indx_x = int(pos_x_indU - 0.5);
    int leftmost_indx_y = int(pos_y_indU - 0.5);
    int leftmost_indx_z = int(pos_z_indU - 0.5);

    // Step 2: along each axis, compute the distance between the stencil-center of the leftmost cell
    //  - Recall that an integer index, ``indx``, specifies the position of the left edge of a cell.
    //    In other words the reference point of the cell is on the left edge.
    //  - The center of the cell specified by ``indx`` is actually ``indx+0.5``
    Real delta_x = pos_x_indU - (leftmost_indx_x + 0.5);
    Real delta_y = pos_y_indU - (leftmost_indx_y + 0.5);
    Real delta_z = pos_z_indU - (leftmost_indx_z + 0.5);

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
  static __device__ void for_each_enclosedCellVol(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                                  int nx_g, int ny_g, Function f)
  {
    CIC::for_each(pos_x_indU, pos_y_indU, pos_z_indU, nx_g, ny_g, f);
  }

  ///* calls the unary function f at ever location where there probably is non-zero overlap with
  // * the stencil.
  // *
  // * \note
  // * This is intended to be conservative (it's okay for this to call the function on a cell with
  // * non-zero overlap). The reason this exacts (rather than just calling for_each), is that it
  // * it may be significantly cheaper for some stencils
  // */
  //template<typename UnaryFunction>
  //static __device__ void for_each_overlap_zone(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
  //                                             int ng_x, int ng_y, int n_ghost, UnaryFunction f)
  //{
  //  // this is a little crude!
  //  CIC::for_each(pos_x_indU, pos_y_indU, pos_z_indU, ng_x, ng_y, n_ghost,
  //                [f](double dummy_arg, int idx3D) {f(idx3D);});
  //}

  ///* returns the nearest location to (pos_x_indU, pos_y_indU, pos_z_indU) that the stencil's center
  // * can be shifted to in order to avoid overlapping with the ghost zone.
  // *
  // * If the specified location already does not overlap with the ghost zone, that is the returned
  // * value.
  // */
  //static __device__ Arr3<Real> nearest_noGhostOverlap_pos(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
  //                                                        int ng_x, int ng_y, int ng_z, int n_ghost)
  //{
  //  constexpr Real min_stencil_offset = 0.5;
  //  Real edge_offset = n_ghost + min_offset;
  //
  //  return { clamp(pos_x_indU, edge_offset, ng_x - edge_offset),
  //           clamp(pos_y_indU, edge_offset, ng_y - edge_offset),
  //           clamp(pos_z_indU, edge_offset, ng_z - edge_offset) };
  //}

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

  /* returns the count of the number of super-sampled points within a cell that correspond to integer indices of
   * (cell_idx_y, cell_idx_y, cell_idx_z).
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
};

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
  static __device__ void for_each(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                  int nx_g, int ny_g, Function f)
  {
    // Step 1: along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    const int leftmost_indx_x = int(pos_x_indU - 1);
    const int leftmost_indx_y = int(pos_y_indU - 1);
    const int leftmost_indx_z = int(pos_z_indU - 1);

    // Step 2: get the number of super-samples within each of the 27 possible cells
    const SphereObj sphere{{pos_x_indU, pos_y_indU, pos_z_indU}, 1*1};

    uint_least16_t counts[3][3][3]; // we want to keep the array-element size small to reduce memory
                                    // pressure on the stack (especially since every thread will be
                                    // allocating this much stack-space at the same time)
    unsigned long total_count = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          unsigned int count = sphere.Count_Super_Samples<Log2DivsionsPerAx_PerCell>(leftmost_indx_x + i, leftmost_indx_y + j, leftmost_indx_z + k);
          counts[i][j][k] = std::uint_least16_t(count);
          total_count += count;
        }
      }
    }

    //kernel_printf("ref: %g, %g, %g\n", pos_x_indU, pos_y_indU, pos_z_indU);

    // Step 3: actually invoke f at each cell-location that overlaps with the stencil location, passing both:
    //  1. fraction of the total stencil volume enclosed by the given cell
    //  2. the 1d index specifying cell-location (for a field with ghost zones)
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          const int ind3D = (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k));

          //kernel_printf("%d, %d, %d: %g\n", leftmost_indx_x + i, (leftmost_indx_y + j), (leftmost_indx_z + k),
          //              double(counts[i][j][k])/total_count);

          f(double(counts[i][j][k])/total_count, ind3D);
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
  static __device__ void for_each_enclosedCellVol(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                                  int nx_g, int ny_g, Function f)
  {
    // along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    const int leftmost_indx_x = int(pos_x_indU - 1);
    const int leftmost_indx_y = int(pos_y_indU - 1);
    const int leftmost_indx_z = int(pos_z_indU - 1);

    double inverse_max_counts_per_cell = 1.0 / double(std::pow(2,Log2DivsionsPerAx_PerCell*3));
    const SphereObj sphere{{pos_x_indU, pos_y_indU, pos_z_indU}, 1*1};

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          unsigned int count = sphere.Count_Super_Samples<Log2DivsionsPerAx_PerCell>(leftmost_indx_x + i, leftmost_indx_y + j, leftmost_indx_z + k);
          const int ind3D = (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k));
          f(count*inverse_max_counts_per_cell, ind3D);
        }
      }
    }

  }

  /* calls the unary function f at ever location where there probably is non-zero overlap with
   * the stencil.
   *
   * \note
   * This is is significantly cheaper than calling for_each.
   */
  template<typename UnaryFunction>
  static __device__ void for_each_overlap_zone(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                               int ng_x, int ng_y, int n_ghost, UnaryFunction f)
  {
    // along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    const int leftmost_indx_x = int(pos_x_indU) - 1;
    const int leftmost_indx_y = int(pos_y_indU) - 1;
    const int leftmost_indx_z = int(pos_z_indU) - 1;
  
    const SphereObj sphere{/* center = */ {pos_x_indU, pos_y_indU, pos_z_indU},
                           /* squared_radius = */ 1*1}; 

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {

          const Real indx_x = leftmost_indx_x + i;
          const Real indx_y = leftmost_indx_y + j;
          const Real indx_z = leftmost_indx_z + k;

          const int ind3D = indx_x + ng_x * (indx_y + ng_y * indx_z);

          if (sphere.encloses_point(clamp(pos_x_indU, indx_x, indx_x + 1),
                                    clamp(pos_y_indU, indx_y, indx_y + 1),
                                    clamp(pos_z_indU, indx_z, indx_z + 1))){
            f(ind3D);
          }

          // IN THE FUTURE: we can try to be clever and instead query whether the closes subgrid-point
          // is enclosed (care would need to be taken that you get consistent roundoff error with 
          // computing subgrid positions)

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
  /*
  static __device__ Arr3<Real> nearest_noGhostOverlap_pos(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                                          int ng_x, int ng_y, int ng_z, int n_ghost)
  {
    constexpr Real min_stencil_offset = 1.0;
    double edge_offset = n_ghost + min_stencil_offset;

    // we can get more clever. technically, we just can't overlap with nearest
    // super-sampled point inside of ghost zone. Anything else is fair game!
    return { clamp(pos_x_indU, edge_offset, ng_x - edge_offset),
             clamp(pos_y_indU, edge_offset, ng_y - edge_offset),
             clamp(pos_z_indU, edge_offset, ng_z - edge_offset) };
  }*/

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
  static __device__ void for_each(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                  int nx_g, int ny_g, Function f)
  {
    // Step 1: along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    int leftmost_indx_x = int(pos_x_indU) - CellsPerRadius;
    int leftmost_indx_y = int(pos_y_indU) - CellsPerRadius;
    int leftmost_indx_z = int(pos_z_indU) - CellsPerRadius;

    // Step 2: get the number of cells enclosed by the sphere
    const SphereObj sphere{{pos_x_indU, pos_y_indU, pos_z_indU}, CellsPerRadius*CellsPerRadius};
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
  static __device__ void for_each_enclosedCellVol(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                                  int nx_g, int ny_g, Function f)
  {
    // along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    int leftmost_indx_x = int(pos_x_indU) - CellsPerRadius;
    int leftmost_indx_y = int(pos_y_indU) - CellsPerRadius;
    int leftmost_indx_z = int(pos_z_indU) - CellsPerRadius;

    const SphereObj sphere{{pos_x_indU, pos_y_indU, pos_z_indU}, CellsPerRadius*CellsPerRadius};

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

  /* returns the nearest location to (pos_x_indU, pos_y_indU, pos_z_indU) that the stencil's center
   * can be shifted to in order to avoid overlapping with the ghost zone.
   *
   * If the specified location already does not overlap with the ghost zone, that is the returned
   * value.
   */
  /*
  static __device__ Arr3<Real> nearest_noGhostOverlap_pos(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                                          int ng_x, int ng_y, int ng_z, int n_ghost)
  {
    double edge_offset = n_ghost + CellsPerRadius;

    // I think we could be a little more clever

    return { clamp(pos_x_indU, edge_offset, ng_x - edge_offset),
             clamp(pos_y_indU, edge_offset, ng_y - edge_offset),
             clamp(pos_z_indU, edge_offset, ng_z - edge_offset) };
  }*/

};


} // fb_stencil namespace
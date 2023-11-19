/* This provides unit-tests for extracted easily-testable components of the feedback module.
 *
 * This mainly includes things like the deposition stencil
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
#include <utility>

// External Includes
#include <gtest/gtest.h> // Include GoogleTest and related libraries/headers

#include "../global/global.h"
#include "../utils/gpu.hpp" // gpuFor
#include "../utils/DeviceVector.h" // gpuFor

#include "../feedback/feedback_model.h"

namespace  // Anonymous namespace
{

struct Extent3D {
  int nx;
  int ny;
  int nz;
};

template<typename T>
std::string array_to_string(T* arr, Extent3D extent, unsigned int indent_size = 0)
{
  const std::string common_line_prefix = std::string(indent_size, ' ');

  std::string out = "";
  for (int iz = 0; iz < extent.nz; iz++){
    if (iz != 0) out += ",\n\n";  // add delimiter after last element

    for (int iy = 0; iy < extent.ny; iy++) {
      if (iy != 0)  out += ",\n";  // add delimiter after last element

      if ((iz == 0) and (iy == 0)) {
        // explicitly don't insert the indents on very first line
        out += "{{{";
      } else if (iy == 0) {
        out += common_line_prefix + " {{";
      } else {
        out += common_line_prefix + "  {";
      }

      for (int ix = 0; ix < extent.nx; ix++) {
        if (ix != 0)  out += ", ";  // put delimiter after last element

        out += std::to_string(arr[ix + extent.nx * (iy + extent.ny * iz)]);
      }
      out += '}';
    }
    out += '}';
  }
  return out + '}';
}

/* converts a 3 element mathematical vector to a string */
template<typename T>
std::string Vec3_to_String(T* arr3D) {
  return (std::string("(") + std::to_string(arr3D[0]) + ", " + std::to_string(arr3D[1]) + ", " + 
          std::to_string(arr3D[2]) + ")");
}

// this is a little overkill, for right now, but it could be nice to have
// based on signature of numpy's testing.assert_allclose function!
template<typename T>
void assert_allclose(T* actual, T* desired, Extent3D extent,
                     double rtol, double atol = 0.0, bool equal_nan = false,
                     const std::string& err_msg = "")
{

  auto is_close = [equal_nan, atol, rtol](double actual, double desired) -> bool {
    if (equal_nan and std::isnan(actual) and std::isnan(desired)) {
      return true;
    }
    double abs_diff = fabs(actual - desired);
    double max_allowed_abs_diff = (atol + rtol * fabs(desired));
    // need to use <= rather than <, to handle case where atol = actual = desired = 0
    return abs_diff <= max_allowed_abs_diff;
  };

  int count_notclose = 0;

  // on device code, we want to swap the iteration order
  for (int iz = 0; iz < extent.nz; iz++) {
    for (int iy = 0; iy < extent.ny; iy++) {
      for (int ix = 0; ix < extent.nx; ix++) {
        int ind3D = ix + extent.nx * (iy + extent.ny * iz);
        count_notclose += not is_close(actual[ind3D], desired[ind3D]);
      }
    }
  }

  if (count_notclose == 0) return;

  // make another pass through - this time gather information to provide an informative error message
  int first_bad_index[3] = {-1,0,0};
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;

  for (int iz = 0; iz < extent.nz; iz++) {
    for (int iy = 0; iy < extent.ny; iy++) {
      for (int ix = 0; ix < extent.nx; ix++) {
        int ind3D = ix + extent.nx * (iy + extent.ny * iz);
        max_abs_diff = std::fmax(max_abs_diff, std::fabs(actual[ind3D] - desired[ind3D]));

        if (desired[ind3D] != 0) {
          double cur_rel_diff = (actual[ind3D] - desired[ind3D])/double(desired[ind3D]);
          max_rel_diff = std::fmax(max_rel_diff, std::fabs(cur_rel_diff));
        }

        if (first_bad_index[0] == -1 and (not is_close(actual[ind3D], desired[ind3D]))) {
          first_bad_index[0] = ix;
          first_bad_index[1] = iy;
          first_bad_index[2] = iz;
        }
      }
    }
  }

    std::size_t total_size = std::size_t(extent.nz) * std::size_t(extent.ny) * std::size_t(extent.nx);
    int bad_ind3D = first_bad_index[0] + extent.nx * (first_bad_index[1] + extent.ny * first_bad_index[2]);

    FAIL() << "Not equal to tolerance rtol=" << rtol << ", atol=" << atol << '\n'
           << err_msg << '\n'
           << "Mismatched elements: " << count_notclose << " / " << total_size << '\n'
           << "Max absolute difference: " << max_abs_diff << '\n'
           << "Max relative difference: " << max_rel_diff << '\n'
           << "First bad index: " << Vec3_to_String(first_bad_index) << '\n'
           << "    actual: " << actual[bad_ind3D] << ", desired: " << desired[bad_ind3D] << "\n";
}


struct DomainSpatialProps {
  Real xMin, yMin, zMin;  /*!< Cell widths (in code units) along cur axis. */
  Real dx, dy, dz;  /*!< Cell widths (in code units) along cur axis. */
};

/* The old CIC stencil implementation (using the old interface). This logic was ripped almost
 * straight out of the original implementation of Apply_Resolved_SN
 */
struct OldCICStencil {

  template<typename Function>
  __device__ void for_each(Real pos_x, Real pos_y, Real pos_z, 
                           DomainSpatialProps spatial_props,
                           int nx_g, int ny_g, int n_ghost,
                           Function& f)
  {

    double xMin = spatial_props.xMin;
    double yMin = spatial_props.yMin;
    double zMin = spatial_props.zMin;
    double dx = spatial_props.dx;
    double dy = spatial_props.dy;
    double dz = spatial_props.dz;

    //kernel_printf("min_vals: %e, %e, %e\n", xMin, yMin, zMin);
    //kernel_printf("cell_widths: %e, %e, %e\n", dx, dy, dz);
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
    //kernel_printf("delta_x, delta_y, delta_z: %e, %e, %e\n", delta_x, delta_y, delta_z);

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          Real x_frac = i * (1 - delta_x) + (1 - i) * delta_x;
          Real y_frac = j * (1 - delta_y) + (1 - j) * delta_y;
          Real z_frac = k * (1 - delta_z) + (1 - k) * delta_z;

          int indx    = (indx_x + i + n_ghost) + (indx_y + j + n_ghost) * nx_g + (indx_z + k + n_ghost) * nx_g * ny_g;

          f(x_frac*y_frac*z_frac, indx);
        }  // k loop
      }    // j loop
    }      // i loop
  }

};



} // anonymous namespace

/* Struct that specifies 1D spatial properties. This is used to help parameterize tests */
struct AxProps {
  int num_cells;  /*!< number of cells along the given axis (excluding ghost zone)*/
  Real min;  /*!< the position of the left edge of left-most (non-ghost) cell, in code units */
  Real cell_width;  /*!< Cell width, in code units along cur axis. Must be positive */

  /* utility function! */
  static DomainSpatialProps construct_spatial_props(AxProps xax, AxProps yax, AxProps zax) {
    DomainSpatialProps out;
    out.xMin = xax.min;
    out.dx = xax.cell_width;
    out.yMin = yax.min;
    out.dy = yax.cell_width;
    out.zMin = zax.min;
    out.dz = zax.cell_width;
    return out;
  }
};


/* Updates elements in the ``out_data`` 3D-array with the fraction of the volume that is enclosed by
 * the specified ``stencil``, that is centered at the given position.
 *
 * \note
 * This handles the legacy function signature
 *
 * \note
 * Right now, this function should only be executed by a single thread-block with a single thread. This
 * choice reflects the fact that a single thread is historically assigned to a single particle.
 */
template<typename Stencil>
__global__ void Stencil_Overlap_Kernel_Legacy_Interface_(Real* out_data, double pos_x, double pos_y, double pos_z,
                                                         DomainSpatialProps domain_spatial_props, 
                                                         int nx_g, int ny_g, int n_ghost, Stencil stencil)
{
  // first, define the lambda function that actually updates out_data
  auto update_entry_fn = [out_data](double dV, int indx3D) -> void { out_data[indx3D] = dV; };

  // second, execute update_entry at each location where the stencil overlaps with the
  stencil.for_each(pos_x, pos_y, pos_z, domain_spatial_props, nx_g, ny_g, n_ghost, update_entry_fn);
}


/* Updates elements in the ``out_data`` 3D-array with the fraction of the volume that is enclosed by
 * the specified ``stencil``, that is centered at the given position.
 *
 * \note
 * Right now, this function should only be executed by a single thread-block with a single thread. This
 * choice reflects the fact that a single thread is historically assigned to a single particle.
 */
template<typename Stencil>
__global__ void Stencil_Overlap_Kernel_(Real* out_data, double pos_x_indU, double pos_y_indU, double pos_z_indU,
                                        int nx_g, int ny_g, Stencil stencil)
{
  // first, define the lambda function that actually updates out_data
  auto update_entry_fn = [out_data](double dV, int indx3D) -> void { out_data[indx3D] = dV; };

  // second, execute update_entry at each location where the stencil overlaps with the
  stencil.for_each(pos_x_indU, pos_y_indU, pos_z_indU, nx_g, ny_g, update_entry_fn);
}

// full_extent must include contributions from the ghost_depth
std::vector<double> eval_stencil_overlap_(const Real* pos_indxU, Extent3D full_extent,
                                          AxProps* prop_l, int n_ghost, bool legacy) {

  cuda_utilities::DeviceVector<Real> data(full_extent.nx*full_extent.ny*full_extent.nz, true);  // initialize to 0

  // launch the kernel
  const int num_blocks = 1;
  const int threads_per_block = 1;

  if (legacy) {
    // unpack prop_l into DomainSpatialProps
    DomainSpatialProps spatial_props = AxProps::construct_spatial_props(prop_l[0], prop_l[1], prop_l[1]);

    // convert pos_indxU to pos_codeU. The Legacy stencil computes the inverse of this, which will introduce some
    // minor differences. One of the main points of this function is to test these differences (and other
    // accumulated round-off differences).
    double pos_codeU[3];
    for (int i = 0; i < 3; i++) {
      // pos_indxU gives the position in index-units on the grid includeing ghost-zones

      // prop_l[i].min gives the position of the left edge of the leftmost cell NOT in the ghost zone
      pos_codeU[i] = (pos_indxU[i] - n_ghost) * prop_l[i].cell_width + prop_l[i].min;
    }

    OldCICStencil stencil;
    hipLaunchKernelGGL(Stencil_Overlap_Kernel_Legacy_Interface_, num_blocks, threads_per_block, 0, 0,
                       data.data(), pos_codeU[0], pos_codeU[1], pos_codeU[2], spatial_props,
                       full_extent.nx, full_extent.ny, n_ghost,
                       stencil);
  } else {
    feedback_model::CICDepositionStencil stencil;
    hipLaunchKernelGGL(Stencil_Overlap_Kernel_, num_blocks, threads_per_block, 0, 0,
                       data.data(), pos_indxU[0], pos_indxU[1], pos_indxU[2],
                       full_extent.nx, full_extent.ny,
                       stencil);
  }

  CHECK(cudaDeviceSynchronize());
  std::vector<double> out(full_extent.nx*full_extent.ny*full_extent.nz);
  data.cpyDeviceToHost(out);
  return out;
}

void compare_cic_stencil(AxProps* prop_l, int n_ghost) {


  std::vector<Real> sample_pos_indxU_minus_nghost = {
    1.0, 1.00001, 1.1, 1.5, 1.9, 1.9999999
  };

  std::vector<std::array<Real,3>> pos_indxU_l{};
  for (std::size_t i = 0; i < sample_pos_indxU_minus_nghost.size(); i++) {
    for (std::size_t j = 0; j < sample_pos_indxU_minus_nghost.size(); j++) {
      for (std::size_t k = 0; k < sample_pos_indxU_minus_nghost.size(); k++) {
        pos_indxU_l.push_back({sample_pos_indxU_minus_nghost[i] + n_ghost,
                               sample_pos_indxU_minus_nghost[j] + n_ghost,
                               sample_pos_indxU_minus_nghost[k] + n_ghost});
      }
    }
  }


  for (const std::array<Real, 3>& pos_indxU : pos_indxU_l) {
    const std::string pos_indxU_str = Vec3_to_String(pos_indxU.data());

    // include ghost cells within extent
    Extent3D extent = {prop_l[0].num_cells + 2*n_ghost,   // x-axis
                       prop_l[1].num_cells + 2*n_ghost,   // y-axis
                       prop_l[2].num_cells + 2*n_ghost};  // z-axis

    std::vector<double> overlap_legacy = eval_stencil_overlap_(pos_indxU.data(), extent, prop_l, n_ghost, true);
    std::vector<double> overlap_new = eval_stencil_overlap_(pos_indxU.data(), extent, prop_l, n_ghost, false);

    if (false) { // for debugging purposes!
      printf("considering: %s\n:", pos_indxU_str.c_str());
      int num_indents = 2;
      std::string tmp = array_to_string(overlap_legacy.data(), extent, num_indents);
      printf("legacy stencil overlap:\n  %s\n", tmp.c_str());
      tmp = array_to_string(overlap_new.data(), extent, num_indents);
      printf("modern stencil overlap:\n  %s\n", tmp.c_str());
    }

    const std::string err_msg = "error comparing stencils when it is centered at indices of " + pos_indxU_str;
    const double rtol = 0.0;//1e-15;
    const double atol = 0;
    assert_allclose(overlap_new.data(), overlap_legacy.data(), extent, rtol, atol, false, err_msg);
  }
}

/* This test compares the modern cloud-in-cell stencil against the legacy one! */
TEST(tALLFeedbackCiCStencil, ComparisonAgainstOld)
{
  Real dx = 1.0 / 256.0;
  std::vector<AxProps> prop_l = {{3, 0.0, dx}, {3,0.0, dx}, {3,0.0, dx}};

  for (int n_ghost = 0; n_ghost < 2; n_ghost++){
    compare_cic_stencil(prop_l.data(), n_ghost);
  }

};

// records the first and last index along an index where the stencil has non-zero overlap as well
// as the overlap values at those indices
struct OverlapRange{
  int first_indx, last_indx;
  double first_overlap, last_overlap;
};

// iterate over the x-axis (at fixed y_ind and z_ind) of an array that holds the overlap values computed
// with respect to the stencil. Return the OverlapRange object representing the range of cells with
// non-zero values
OverlapRange find_ovrange_(double* v, int y_ind, int z_ind, Extent3D full_extent) {

  for (int ix = 0; ix < full_extent.nx; ix++) {
    double overlap_frac = v[ix + full_extent.nx * (y_ind + full_extent.ny * z_ind)];
    if (overlap_frac > 0) {
      // launch inner-loop to search for the end of the overlap-range
      double prev_inner_overlap = overlap_frac;
      for (int inner_ix = (ix + 1); inner_ix < full_extent.nx; inner_ix++) {
        double inner_overlap = v[inner_ix + full_extent.nx * (y_ind + full_extent.ny * z_ind)];
        if (inner_overlap == 0.0)  return {ix, inner_ix - 1, overlap_frac, prev_inner_overlap};
        prev_inner_overlap = inner_overlap;
      }
      // if we got to this point, this means that the (full_extend.nx-1) overlaps with stencil
      return {ix, full_extent.nx-1, overlap_frac, prev_inner_overlap};
    }
  }

  return {-1, -1, 0.0, 0.0};  // there is no overlap
};

/* test some expected trends as we slowly move a stencil to the right */
void sliding_stencil_test(int n_ghost) {
    // the construction of prop_l and dx is somewhat unnecessary...
  // ToDo: drop them during refactoring
  Real dx = 1.0 / 256.0;
  std::vector<AxProps> prop_l = {{4, 0.0, dx}, {3,0.0, dx}, {3,0.0, dx}};
  Extent3D full_extent{2*n_ghost + prop_l[0].num_cells,
                       2*n_ghost + prop_l[1].num_cells,
                       2*n_ghost + prop_l[2].num_cells};

  // determine the centers of the stencil
  const Real dummy = 1.1 + n_ghost;
  const std::vector<Real> sliding_ax_indxU_vals = {
    1.50 + n_ghost, 1.75 + n_ghost, 2.00 + n_ghost,
    2.25 + n_ghost, 2.50 + n_ghost,
  };

  // evaluate the stencil at each location and store the fractional overlap grid in overlap_results
  std::vector<std::vector<double>> overlap_results{};
  for (const auto sliding_ax_indxU_val : sliding_ax_indxU_vals) {
    Real pos_indxU[3] = {sliding_ax_indxU_val, dummy, dummy};

    overlap_results.push_back(
      eval_stencil_overlap_(pos_indxU, full_extent, prop_l.data(), n_ghost, false)
    );


    // sanity check: ensure non-zero vals sum to 1
    std::string tmp = Vec3_to_String(pos_indxU);
    double total_overlap = 0.0;
    for (const double & overlap : overlap_results.back()) {
      total_overlap += overlap;
    }
    // in the future, we may nee
    EXPECT_NEAR(total_overlap, 1.0, 0.0) << "the total volume of the stencil (in index-units) is not "
                                         << "1.0, when the stencil is centerd at " << Vec3_to_String(pos_indxU);
  }

  // perform some checks based on the ranges of cells with overlap:

  const int y_ind = int(dummy);
  const int z_ind = int(dummy);

  OverlapRange prev_ovrange;
  for (std::size_t i = 0; i < overlap_results.size(); i++) {
    const OverlapRange cur_ovrange = find_ovrange_(overlap_results[i].data(), y_ind, z_ind, full_extent);

    //printf("%zu, first_overlap: (%d, %g) last_overlap: (%d, %g)\n",
    //       i, cur_ovrange.first_indx, cur_ovrange.first_overlap, 
    //       cur_ovrange.last_indx, cur_ovrange.last_overlap);

    if (i != 0) { // make comparisons to previous stencil position

      // as the stencil moves rightwards, we expect the first_overlap and last_overlap indices to generally increase
      ASSERT_GE(cur_ovrange.first_indx, prev_ovrange.first_indx);
      ASSERT_GE(cur_ovrange.last_indx, prev_ovrange.last_indx);

      // if first_ind is the same for both the current and previous stencil position, check that the overlap
      // fraction of that pixel has not increased
      if (cur_ovrange.first_indx == prev_ovrange.first_indx) {
        ASSERT_LE(cur_ovrange.first_overlap, prev_ovrange.first_overlap);
      }

      // if last_indx is the same for both the current and previous stencil position, confirm that the
      // overlap fraction of that pixel has not decreased
      if (cur_ovrange.last_indx == prev_ovrange.last_indx) {
        ASSERT_GE(cur_ovrange.last_overlap, prev_ovrange.last_overlap);
      }
    }
    prev_ovrange = cur_ovrange;
  }
}

TEST(tALLFeedbackCiCStencil, SlidingTest)
{
  sliding_stencil_test(0);
}

#include <cstdint>

/* Represents a sphere */
struct Sphere{
  double center_indU[3];
  int raidus2_indU;

  /* queries whether the sphere encloses a given point */
  __forceinline__ __device__ bool encloses_point(double pos_x_indU, double pos_y_indU, double pos_z_indU) const {
    double delta_x = pos_x_indU - center_indU[0];
    double delta_y = pos_y_indU - center_indU[1];
    double delta_z = pos_z_indU - center_indU[2];

    return (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) < raidus2_indU; 
  }

  /* returns the count of the number of super-sampled points */
  template<unsigned Log2DivsionsPerAx>
  __device__ unsigned int Count_Super_Samples(int cell_idx_x, int cell_idx_y, int cell_idx_z) const {
    static_assert(Log2DivsionsPerAx*3 < sizeof(unsigned int));
    int num_subdivisions_per_ax = std::pow(2,Log2DivsionsPerAx);

    double width = 1/num_subdivisions_per_ax;
    double offset = 0.5 * width;

    unsigned int count = 0;
    for (int ix = 0; ix < num_subdivisions_per_ax; ix++) {
      for (int iy = 0; iy < num_subdivisions_per_ax; iy++) {
        for (int iz = 0; iz < num_subdivisions_per_ax; iz++) {
          // since cell_idx_x, cell_idx_y, cell_idx_z are all integers, they specify
          // the position of the left edge of the cell
          double x = cell_idx_x + (offset + ix * width);
          double y = cell_idx_y + (offset + iy * width);
          double z = cell_idx_z + (offset + iz * width);

          count += encloses_point(x, y, z);
        }
      }
    }

    return count;
  }
};


#include<cstdint>

template<unsigned Log2DivsionsPerAx_PerCell = 2>
struct Sphere27DepositionStencil {

  static_assert(Log2DivsionsPerAx_PerCell <= 2);

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
    const int leftmost_indx_x = int(pos_x_indU) - 1;
    const int leftmost_indx_y = int(pos_y_indU) - 1;
    const int leftmost_indx_z = int(pos_z_indU) - 1;

    // Step 2: get the number of super-samples within each of the 27 possible cells
    const Sphere sphere{{pos_x_indU, pos_y_indU, pos_z_indU}, 1*1};

    uint_least8_t counts[3][3][3]; // we want to keep the array-element size small to reduce memory
                                   // pressure on the stack (especially since every thread will be
                                   // allocating this much stack-space at the same time)
    unsigned long total_count = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          unsigned int count = sphere.Count_Super_Samples<Log2DivsionsPerAx_PerCell>(leftmost_indx_x + i, leftmost_indx_y + j, leftmost_indx_z + k);
          counts[i][j][k] = std::uint_least8_t(counts);
          total_count += count;
        }
      }
    }

    // Step 3: actually invoke f at each cell-location that overlaps with the stencil location, passing both:
    //  1. fraction of the total stencil volume enclosed by the given cell
    //  2. the 1d index specifying cell-location (for a field with ghost zones)
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          const int ind3D = (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k));

          f(double(counts[i][j][k])/total_count, ind3D);
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

    const Sphere sphere{/* center = */ {pos_x_indU, pos_y_indU, pos_z_indU},
                        /* squared_radius = */ 1*1}; 

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {

          const Real indx_x = leftmost_indx_x + i;
          const Real indx_y = leftmost_indx_y + j;
          const Real indx_z = leftmost_indx_z + k;

          const int ind3D = indx_x + ng_x * (indx_y + ng_y * indx_z);

          // calculate the displacement vector for the nearest point on the
          // cube to the sphere
          // IN THE FUTURE: correct this to use the nearest super-sampled point
          if (sphere.encloses_point(clamp(pos_x_indU, indx_x, indx_x + 1),
                                    clamp(pos_x_indU, indx_x, indx_x + 1),
                                    clamp(pos_x_indU, indx_x, indx_x + 1))){
            f(ind3D);
          }

          
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
  }

};



/* Represents a spherical stencil with a radius of 3 cells, where the inclusion where inclusion of
 * cells in the sphere is a binary choice.
 *
 * Specifically, a cell is included if the cell-center lies within the sphere.
 */
/*
template<unsigned CellsPerRadius = 3>
struct SphereBinaryDepositionStencil {
  static_assert(CellsPerRadius > 0);

  template<typename Function>
  static __device__ void for_each(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                  int nx_g, int ny_g, Function f)
  {
    // Step 1: along each axis, identify the integer-index of the leftmost cell covered by the stencil.
    int leftmost_indx_x = int(pos_x_indU) - CellsPerRadius;
    int leftmost_indx_y = int(pos_y_indU) - CellsPerRadius;
    int leftmost_indx_z = int(pos_z_indU) - CellsPerRadius;

    // Step 2: get the number of cells enclosed by the sphere
    const Sphere sphere{{pos_x_indU, pos_y_indU, pos_z_indU}, CellsPerRadius*CellsPerRadius};
    int total_count = 0;

    const int stop = (2 * CellsPerRadius) + 1;
    for (int i = 0; i < stop; i++) {
      for (int j = 0; j < stop; j++) {
        for (int k = 0; k < stop; k++) {
          total_count += encloses_point(leftmost_indx_x + i + 0.5,
                                        leftmost_indx_y + j + 0.5,
                                        leftmost_indx_z + k + 0.5);
        }
      }
    }

    double enclosed_stencil_frac = 1.0/total_count;  // each enclosed cell, encloses this fraction of the sphere

    // Step 3: actually invoke f at each cell-location that overlaps with the stencil location, passing both:
    //  1. fraction of the total stencil volume enclosed by the given cell
    //  2. the 1d index specifying cell-location (for a field with ghost zones)
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        for (int k = 0; k < 7; k++) {
          
          if (sphere.encloses_point(leftmost_indx_x + i + 0.5, leftmost_indx_y + j + 0.5, leftmost_indx_z + k + 0.5)){
            const int ind3D = (leftmost_indx_x + i) + nx_g * ((leftmost_indx_y + j) + ny_g * (leftmost_indx_z + k));
            f(enclosed_stencil_frac, ind3D);
          }

        }
      }
    }

  }
};
*/

// some tests for the sphere stencil:
// -> try to back out the value of pi? This only works if we also calculate of the 
//    fraction of the cell that is enclosed by the stencil.
//    -> compute the stencil volume and divide that voume by the stencil-radius squared
//    -> As you increase the radius of SphereBinaryDepositionStencil or the number of
//       divisions in Sphere27DepositionStencil, the accuracy should improve
//    -> the answer will vary based on the center of the stencil
/* This provides unit-tests for extracted easily-testable components of the feedback module.
 *
 * This mainly includes things like the deposition stencil
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

// External Includes
#include <gtest/gtest.h> // Include GoogleTest and related libraries/headers

#include "../global/global.h"
#include "../utils/gpu.hpp" // gpuFor
#include "../utils/DeviceVector.h"
#include "../utils/error_handling.h"

#include "../feedback/feedback_model.h"
#include "../feedback/kernel.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define some general-purpose testing tools. These may be a little overkill for this particular file.
// It may make sense to move these to the testing_utils file and namespace. 
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace  // Anonymous namespace
{

template<typename T>
std::string array_to_string_1D_helper_(T* arr, int len)
{
  std::string out = "";
  for (int ix = 0; ix < len; ix++) {
    if (ix != 0)  out += ", ";  // put delimiter after last element
    out += std::to_string(arr[ix]);
  }
  return out;
}

template<typename T>
std::string array_to_string(T* arr, int len)
{
  return std::string("{") +  array_to_string_1D_helper_(arr, len) + std::string("}");
};

struct Extent3D {
  int nx;
  int ny;
  int nz;
};

/* Convert an array to a string.
 *
 * The indent_size arg is adopted from numpy's similar indent arg in array2string. As in numpy,
 * we don't apply indent on first line
 */
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

      // ToDo: replace following loop with out += array_to_string_1D_helper_(arr + extent.nx * (iy + extent.ny * iz), extent.nx);
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

template<typename T>
bool isclose(T actual, T desired, double rtol, double atol = 0.0, bool equal_nan = false) {
  if (equal_nan and std::isnan(actual) and std::isnan(desired))  return true;

  double abs_diff = fabs(actual - desired);
  double max_allowed_abs_diff = (atol + rtol * fabs(desired));
  // need to use <= rather than <, to handle case where atol = actual = desired = 0
  return abs_diff <= max_allowed_abs_diff;
}

// this is a little overkill, for right now, but it could be nice to have
// based on signature of numpy's testing.assert_allclose function!
template<typename T>
void assert_allclose(T* actual, T* desired, Extent3D extent,
                     double rtol, double atol = 0.0, bool equal_nan = false,
                     const std::string& err_msg = "")
{

  auto is_close = [equal_nan, atol, rtol](double actual, double desired) -> bool {
    return isclose(actual, desired, rtol, atol, equal_nan);
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

} // anonymous namespace


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define some tools for testing deposition-stencils
///////////////////////////////////////////////////////////////////////////////////////////////////////

/* Updates elements in the ``out_data`` 3D-array with the fraction of the volume that is enclosed by
 * the specified ``stencil``, that is centered at the given position.
 *
 * \note
 * Right now, this function should only be executed by a single thread-block with a single thread. This
 * choice reflects the fact that a single thread is historically assigned to a single particle.
 */
template<typename Stencil>
__global__ void Stencil_Overlap_Kernel_(Real* out_data, Arr3<Real> pos_indU,
                                        int nx_g, int ny_g, Stencil stencil, StencilEvalKind eval)
{
  // first, define the lambda function that actually updates out_data
  auto update_entry_fn = [out_data](double dV, int indx3D) -> void { out_data[indx3D] = dV; };

  // second, execute update_entry at each location where the stencil overlaps with the cells
  switch(eval) {
    case StencilEvalKind::enclosed_stencil_vol_frac:
      stencil.for_each(pos_indU, nx_g, ny_g, update_entry_fn);
      break;
    case StencilEvalKind::enclosed_cell_vol_frac:
      stencil.for_each_enclosedCellVol(pos_indU, nx_g, ny_g, update_entry_fn);
      break;
    case StencilEvalKind::for_each_overlap_zone:
      stencil.for_each_overlap_zone(pos_indU, nx_g, ny_g,
                                    [out_data](int indx3D) -> void { out_data[indx3D] = 1.0; });
      break;
  }
  
}

/* Utility function used in multiple tests that evaluates overlap values with a grid of cells (on the device)
 * and returns the results after copying them back to the host.
 *
 * \param pos_indxU A 3 element array specifying the postion of the (center of the) stencil
 * \param full_extent describes the extent of the array that the stencil will be evaluated on. (It MUST
 *     include contributions from the ghost_depth)
 * \param n_ghost the number of ghost-zones
 * \param stencil instance of the stencil that should be tested
 * \param eval Specifies the precise calculation that will be performed.
 */
template <typename Stencil>
std::vector<double> eval_stencil_overlap_(const Real* pos_indxU, Extent3D full_extent,
                                          int n_ghost, Stencil stencil,
                                          StencilEvalKind eval = StencilEvalKind::enclosed_stencil_vol_frac) {

  cuda_utilities::DeviceVector<Real> data(full_extent.nx*full_extent.ny*full_extent.nz, true);  // initialize to 0

  // launch the kernel
  const int num_blocks = 1;
  const int threads_per_block = 1;

  Arr3<Real> pos_indU{pos_indxU[0], pos_indxU[1], pos_indxU[2]};

  hipLaunchKernelGGL(Stencil_Overlap_Kernel_, num_blocks, threads_per_block, 0, 0,
                     data.data(), pos_indU, full_extent.nx, full_extent.ny, stencil,
                     eval);

  CHECK(cudaDeviceSynchronize());
  std::vector<double> out(full_extent.nx*full_extent.ny*full_extent.nz);
  data.cpyDeviceToHost(out);
  return out;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define a test comparing different versions of the CiC deposition kernel
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

// we can probably delete this struct
struct DomainSpatialProps {
  Real xMin, yMin, zMin;  /*!< Cell widths (in code units) along cur axis. */
  Real dx, dy, dz;  /*!< Cell widths (in code units) along cur axis. */
};

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

/* The old CIC stencil implementation (using the old interface). This logic was ripped almost
 * straight out of the original implementation of Apply_Resolved_SN */
struct OldCICStencil {
private:  // attributes
  DomainSpatialProps spatial_props;
  int n_ghost;

public:  // interface
  OldCICStencil(AxProps xax, AxProps yax, AxProps zax, int n_ghost)
   : spatial_props(AxProps::construct_spatial_props(xax,yax,zax)), n_ghost(n_ghost)
  {}

  /* adapts the arguments and passes the functions back to the legacy interface */
  template<typename Function>
  __device__ void for_each(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f) const
  {
    const Real pos_x_indU = pos_indU[0];
    const Real pos_y_indU = pos_indU[1];
    const Real pos_z_indU = pos_indU[2];

    // pos_indxU gives the position in index-units on the grid including ghost-zones
    Real pos_x = (pos_x_indU - this->n_ghost) * this->spatial_props.dx + this->spatial_props.xMin;
    Real pos_y = (pos_y_indU - this->n_ghost) * this->spatial_props.dy + this->spatial_props.yMin;
    Real pos_z = (pos_z_indU - this->n_ghost) * this->spatial_props.dz + this->spatial_props.zMin;

    for_each_legacy(pos_x, pos_y, pos_z, this->spatial_props, nx_g, ny_g, n_ghost, f);
  }

  template<typename Function>
  __device__ void for_each_legacy(Real pos_x, Real pos_y, Real pos_z, 
                                  DomainSpatialProps spatial_props,
                                  int nx_g, int ny_g, int n_ghost,
                                  Function& f) const
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

  /* identical to for_each (provided for compatability with interfaces of other stencils). */
  template<typename Function>
  __device__ void for_each_enclosedCellVol(Arr3<Real> pos_indU, int nx_g, int ny_g, Function f)
  {
    this->for_each(pos_indU, nx_g, ny_g, f);
  }

  /* provided for compatability with interfaces of other stencils */
  template<typename UnaryFunction>
  __device__ void for_each_overlap_zone(Arr3<Real> pos_indU, int ng_x, int ng_y, UnaryFunction f)
  {
    // this is a little crude!
    this->for_each(pos_indU, ng_x, ng_y,
                   [f](double stencil_enclosed_frac, int idx3D) { if (stencil_enclosed_frac > 0) f(idx3D);});
  }

};

} // anonymous namespace

/* this is used as a test comparing the old implementation of CiC deposition agains the new version. */
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

    std::vector<double> overlap_legacy = eval_stencil_overlap_(pos_indxU.data(), extent, n_ghost,
                                                               OldCICStencil(prop_l[0],prop_l[1],prop_l[2], n_ghost));
    std::vector<double> overlap_new = eval_stencil_overlap_(pos_indxU.data(), extent, n_ghost,
                                                            fb_stencil::CIC{});

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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define some tests that check some expected trends as we slowly move a stencil to the right along the
// x-axis. These tests could definitely be generalized (so that they are performed along other axes)
///////////////////////////////////////////////////////////////////////////////////////////////////////

// records the first and last index along an index where the stencil has non-zero overlap as well
// as the overlap values at those indices
struct OverlapRange{
  int first_indx, last_indx;
  double first_overlap, last_overlap;
};

// iterate over the x-axis (at fixed y_ind and z_ind) of an array that holds the overlap values computed
// with respect to the stencil. Return the OverlapRange object representing the range of cells with
// non-zero values
std::optional<OverlapRange> find_ovrange_(double* v, int y_ind, int z_ind, Extent3D full_extent) {

  for (int ix = 0; ix < full_extent.nx; ix++) {
    double overlap_frac = v[ix + full_extent.nx * (y_ind + full_extent.ny * z_ind)];
    if (overlap_frac > 0) {
      // launch inner-loop to search for the end of the overlap-range
      double prev_inner_overlap = overlap_frac;
      for (int inner_ix = (ix + 1); inner_ix < full_extent.nx; inner_ix++) {
        double inner_overlap = v[inner_ix + full_extent.nx * (y_ind + full_extent.ny * z_ind)];
        if (inner_overlap == 0.0)  return {{ix, inner_ix - 1, overlap_frac, prev_inner_overlap}};
        prev_inner_overlap = inner_overlap;
      }
      // if we got to this point, this means that the (full_extend.nx-1) overlaps with stencil
      return {{ix, full_extent.nx-1, overlap_frac, prev_inner_overlap}};
    }
  }

  return {};  // there is no overlap
};

/* test some expected trends as we slowly move a stencil to the right along the
 * x-axis.
 *
 * \param n_ghost the number of ghost-zones to use in the calculation
 * \param tot_vol_atol the acceptable absolute error bound between the empirical
 *                     total-overlap value and 1.0 (the expected value)tolerance for the 
 */
template <typename Stencil>
void sliding_stencil_test(int n_ghost, double tot_vol_atol = 0.0, bool ignore_monotonicity_comparisons = false) {
  
  Extent3D full_extent{2*n_ghost + 4,  // x-axis
                       2*n_ghost + 3,  // y-axis
                       2*n_ghost + 3}; // z-axis

  // determine the centers of the stencil
  Real dummy = 1.1 + n_ghost;
  std::vector<Real> sliding_ax_indxU_vals = {
    1.50 + n_ghost, 1.75 + n_ghost, 2.00 + n_ghost,
    2.25 + n_ghost, 2.50 + n_ghost,
  };

  if (Stencil::max_enclosed_neighbors > 1) {
    int min_width = 2*n_ghost + 1 + 2 * Stencil::max_enclosed_neighbors;
    full_extent = {min_width + 1,  // x-axis
                   min_width,  // y-axis
                   min_width}; // z-axis
    dummy = 0.1 + n_ghost + Stencil::max_enclosed_neighbors;
    sliding_ax_indxU_vals.clear();
    sliding_ax_indxU_vals = {n_ghost + Stencil::max_enclosed_neighbors + 0.5,
                             n_ghost + Stencil::max_enclosed_neighbors + 0.75,
                             n_ghost + Stencil::max_enclosed_neighbors + 1.0,
                             n_ghost + Stencil::max_enclosed_neighbors + 1.25,
                             n_ghost + Stencil::max_enclosed_neighbors + 1.5};
  }
  

  // evaluate the stencil at each location and store the fractional overlap grid in overlap_results
  std::vector<std::vector<double>> overlap_results{};
  for (const auto sliding_ax_indxU_val : sliding_ax_indxU_vals) {
    Real pos_indxU[3] = {sliding_ax_indxU_val, dummy, dummy};

    overlap_results.push_back(
      eval_stencil_overlap_(pos_indxU, full_extent, n_ghost, Stencil{},
                            StencilEvalKind::enclosed_cell_vol_frac)
    );

    std::vector<double> stencil_overlap_frac = eval_stencil_overlap_(
      pos_indxU, full_extent, n_ghost, Stencil{},
      StencilEvalKind::enclosed_stencil_vol_frac);


    //int num_indents = 2;
    //std::string tmp_arr = array_to_string(overlap_results.back().data(), full_extent, num_indents);
    //printf("\n  %s\n:", tmp_arr.c_str());

    // sanity check: ensure non-zero vals sum to 1
    std::string tmp = Vec3_to_String(pos_indxU);
    double total_overlap = 0.0;
    for (const double & overlap : stencil_overlap_frac) {
      total_overlap += overlap;
    }
    // in the future, we may nee
    ASSERT_NEAR(total_overlap, 1.0, tot_vol_atol) << "the sum of the stencil-vol-frac is not 1.0, when "
                                                  << "when the stencil is centered at " << Vec3_to_String(pos_indxU);
  }

  // perform some checks based on the ranges of cells with overlap:
  // - To make this test as generic as possible,
  //   -> need to handle cases (especially for super-sampled stencil) where a given y_ind/z_ind near
  //      the edge of the stencil won't have any overlap at all, except when the stencil is positioned
  //      at very particular x-values (basically it has to do with the distance of the subgrid to 
  //      the stencil-center)

  //const int y_ind = int(dummy);
  //const int z_ind = int(dummy);

  for (int y_ind = 0; y_ind < full_extent.ny; y_ind++) {
    for (int z_ind = 0; z_ind < full_extent.nz; z_ind++) {

      // basically, we setup a separate test each time we encounter this inner loop
      std::optional<OverlapRange> prev_ovrange;
      for (std::size_t i = 0; i < overlap_results.size(); i++) {
        std::optional<OverlapRange> cur_ovrange = find_ovrange_
          (overlap_results[i].data(), y_ind, z_ind, full_extent);

        //printf("%zu, first_overlap: (%d, %g) last_overlap: (%d, %g)\n",
        //       i, cur_ovrange.value().first_indx, cur_ovrange.value().first_overlap,
        //       cur_ovrange.value().last_indx, cur_ovrange.value().last_overlap);

        if (bool(prev_ovrange) and bool(cur_ovrange)) { // make comparisons to previous
                                                        //stencil position
          const OverlapRange& prev = *prev_ovrange;
          const OverlapRange& cur  = *cur_ovrange;

          // as the stencil moves rightwards, we expect the first_overlap and last_overlap
          // indices to generally increase
          ASSERT_GE(cur.first_indx, prev.first_indx);
          ASSERT_GE(cur.last_indx, prev.last_indx);

          if (not ignore_monotonicity_comparisons) {
            // these stencils should be ignored for the sphere-binary-stencil

            // if first_ind is the same for both the current and previous stencil position, check that the overlap
            // fraction of that pixel has not increased
            if (cur.first_indx == prev.first_indx) {
              ASSERT_LE(cur.first_overlap, prev.first_overlap);
            }

            // if last_indx is the same for both the current and previous stencil position, confirm that the
            // overlap fraction of that pixel has not decreased
            if (cur.last_indx == prev.last_indx) {
              ASSERT_GE(cur.last_overlap, prev.last_overlap);
            }

          }
        }
        prev_ovrange = cur_ovrange;
      }
    }
  }

}

TEST(tALLFeedbackCiCStencil, SlidingTest)
{
  sliding_stencil_test<fb_stencil::CIC>(0);
}


TEST(tALLFeedbackSphere27Stencil, SlidingTest)
{
  // primary stencil size we would use
  sliding_stencil_test<fb_stencil::Sphere27<2>>(0,2e-16);
  // just testing this case because we can
  sliding_stencil_test<fb_stencil::Sphere27<4>>(0,3e-16);
}

TEST(tALLFeedbackSphereBinaryStencil, SlidingTest)
{
  // we have to ignore the part of the test where we check that the enclosed stencil fraction monotonically
  // increases and decreases (we could refactor the test more a test a different version of that same 
  // behavior)
  sliding_stencil_test<fb_stencil::SphereBinary<3>>(0,2e-15,true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define some tests where we check our expectations about the total volume enclosed by the stencil
///////////////////////////////////////////////////////////////////////////////////////////////////////

// A helper-tool that comes up with the standard stencil setup that is necessary
// for evaluating a stencil (without overlapping beyond the edge of a grid)
template <typename Stencil>
struct StencilTestGridSetup{
  std::vector<Arr3<Real>> center_pos_indU_list;
  Extent3D full_extent;
};

template <typename Stencil>
static StencilTestGridSetup<Stencil> Build_Stencil_Test_Grid_Setup(const std::vector<Arr3<Real>>& center_offset_from_cellEdge_LIST,
                                                                   int n_ghost)
{
  StencilTestGridSetup<Stencil> out;

  // compute the centers of the stencil and the extent of the grid for evaluating the stencil
  for (const Arr3<Real>& center_offset_from_cellEdge : center_offset_from_cellEdge_LIST) {
    Arr3<Real> pos_indU{};
    for (std::size_t i = 0; i < 3; i++){
      // confirm the offset falls in the range 0 <= center_offset < 1
      CHOLLA_ASSERT(center_offset_from_cellEdge[i] >= 0.0, "Test parameter is flawed");
      CHOLLA_ASSERT(center_offset_from_cellEdge[i] < 1.0, "Test parameter is flawed");

      pos_indU[i] = Stencil::max_enclosed_neighbors + center_offset_from_cellEdge[i];
    }
    out.center_pos_indU_list.push_back(pos_indU);
  }

  // choose an extent value that we know will work (even when n_ghost is zero!)
  out.full_extent = Extent3D{1 + 2*(n_ghost + Stencil::max_enclosed_neighbors),
                             1 + 2*(n_ghost + Stencil::max_enclosed_neighbors),
                             1 + 2*(n_ghost + Stencil::max_enclosed_neighbors)};
  return out;
}


template<typename Stencil>
void stencil_volume_check(Arr3<Real> center_offset_from_cellEdge, int n_ghost, Stencil stencil,
                          double expected_vol, double vol_rtol = 0.0, double stencil_overlap_rtol = 0.0)
{
  // compute the center of the stencil and the extent of the grid for evaluating the stencil
  StencilTestGridSetup<Stencil> setup = Build_Stencil_Test_Grid_Setup<Stencil>({center_offset_from_cellEdge}, n_ghost);
  Arr3<Real> pos_indU = setup.center_pos_indU_list[0];
  Extent3D full_extent = setup.full_extent;

  // now gather the amount of cell-volume enclosed by the stencil
  std::vector<double> enclosed_cell_vol = eval_stencil_overlap_(pos_indU.data(), full_extent, n_ghost, stencil,
                                                                StencilEvalKind::enclosed_cell_vol_frac);
  // compute the total stencil volume
  double vtot = 0.0;
  for(double val : enclosed_cell_vol) { vtot += val; }

  // now perform the check on the total cell volume
  EXPECT_TRUE(isclose(vtot, expected_vol,
                      vol_rtol, 0.0, false)) << "stencil volume, " << vtot << ", does NOT match the expected "
                                             << "volume, " << expected_vol << ", to within the relative tolerance "
                                             << "of " << vol_rtol << ". The relative error is: "
                                             << (vtot - expected_vol)/expected_vol;

  ASSERT_GT(vtot, 0.0); // this is mostly a sanity check!

  // now let's confirm consistency with the calculation of the total stencil volume
  // (otherwise this test is meaningless)
  for(double& val : enclosed_cell_vol) { val /= vtot; }

  std::vector<double> enclosed_stencil_vol = eval_stencil_overlap_(pos_indU.data(), full_extent, n_ghost, stencil,
                                                                   StencilEvalKind::enclosed_stencil_vol_frac);
  assert_allclose(enclosed_cell_vol.data(), enclosed_stencil_vol.data(), full_extent, stencil_overlap_rtol, 0.0, false,
                  "the grid of stencil-overlap-vol-fracs computed from the grid on cellvol-fracs is "
                  "inconsistent with the direclty computed grid of stencil-overlap-vol-fracs");
}

TEST(tALLFeedbackCiCStencil, StencilVolumeTest)
{
  stencil_volume_check(Arr3<Real>{0.0,0.0,0.0}, 0, fb_stencil::CIC{},
                       /* expected_vol = */ 1.0, /* vol_rtol = */ 0.0, /*stencil_overlap_rtol =*/ 0.0);
  stencil_volume_check(Arr3<Real>{0.5,0.5,0.5}, 0, fb_stencil::CIC{},
                       /* expected_vol = */ 1.0, /* vol_rtol = */ 0.0, /*stencil_overlap_rtol =*/ 0.0);
}

TEST(tALLFeedbackSphere27Stencil, StencilVolumeTest)
{
  const double radius = 1; // in units of cell_widths
  const double expected_vol = 4 * 3.141592653589793 * (radius * radius) / 3;

  stencil_volume_check(Arr3<Real>{0.0,0.0,0.0}, 0, fb_stencil::Sphere27<2>{},
                       expected_vol, /* vol_rtol = */ 0.05, /*stencil_overlap_rtol =*/ 0.0);
  stencil_volume_check(Arr3<Real>{0.125,0.0,0.0}, 0, fb_stencil::Sphere27<2>{},
                       expected_vol, /* vol_rtol = */ 0.0004, /*stencil_overlap_rtol =*/ 0.0);
  stencil_volume_check(Arr3<Real>{0.5,0.5,0.5}, 0, fb_stencil::Sphere27<2>{},
                       expected_vol, /* vol_rtol = */ 0.05, /*stencil_overlap_rtol =*/ 0.0);
}

/* Something is funky! as you increase the radius, my intuition tells me that the relative error 
 * should improve, but that does not seem to be the case*/
//TEST(tALLFeedbackSphereBinaryStencil, StencilVolumeTest)
//{
//  const double radius = 3; // in units of cell_widths
//  const double expected_vol = 4 * 3.141592653589793 * (radius * radius) / 3.0;
//
//  stencil_volume_check(Arr3<Real>{0.0,0.0,0.0}, 0, fb_stencil::SphereBinary<3>{},
//                       expected_vol, /* vol_rtol = */ 0.0, /*stencil_overlap_rtol =*/ 0.0);
//  stencil_volume_check(Arr3<Real>{0.125,0.0,0.0}, 0, fb_stencil::SphereBinary<3>{},
//                       expected_vol, /* vol_rtol = */ 0.0, /*stencil_overlap_rtol =*/ 0.0);
//  stencil_volume_check(Arr3<Real>{0.5,0.5,0.5}, 0, fb_stencil::SphereBinary<3>{},
//                       expected_vol, /* vol_rtol = */ 0.0, /*stencil_overlap_rtol =*/ 0.0);
//}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define some tests where we check consistency between the different flavors of for_each
///////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class tALLFeedbackStencil : public testing::Test {
public:
  using StencilT=T;
};

using MyStencilTypes = ::testing::Types<fb_stencil::CIC, fb_stencil::LegacyCIC27, fb_stencil::Sphere27<2>, fb_stencil::SphereBinary<3>>;
TYPED_TEST_SUITE(tALLFeedbackStencil, MyStencilTypes);

TYPED_TEST(tALLFeedbackStencil, ForEachFlavorConsistency) {

  // determines the central positions of the stencils
  std::vector<Arr3<Real>> center_offset_from_cellEdge_List = {
    {0.00, 0.00, 0.00},
    {0.25, 0.00, 0.00}, {0.5, 0.0, 0.0},  {0.75, 0.00, 0.00},  {0.99, 0.00, 0.00},
    {0.00, 0.25, 0.00}, {0.0, 0.5, 0.0},  {0.00, 0.75, 0.00},  {0.00, 0.99, 0.00},
    {0.00, 0.00, 0.25}, {0.0, 0.0, 0.5},  {0.00, 0.00, 0.75},  {0.00, 0.00, 0.99},
    {0.25, 0.25, 0.25}, {0.5, 0.5, 0.5},  {0.75, 0.75, 0.75},  {0.99, 0.99, 0.99},
  };

  const int n_ghost = 0;

  using Stencil = typename TestFixture::StencilT;

  // compute the center of the stencil and the extent of the grid for evaluating the stencil
  StencilTestGridSetup<Stencil> setup = Build_Stencil_Test_Grid_Setup<Stencil>(center_offset_from_cellEdge_List, n_ghost);
  const Extent3D extent = setup.full_extent;

  // pair specifying the flavor-name and the actual flavor value
  std::vector<std::pair<std::string, StencilEvalKind>> flavor_pairs = {
    {"enclosed_stencil_vol_frac", StencilEvalKind::enclosed_stencil_vol_frac},
    {"enclosed_cell_vol_frac", StencilEvalKind::enclosed_cell_vol_frac},
    {"for_each_overlap_zone", StencilEvalKind::for_each_overlap_zone}
  };

  for (Arr3<Real> pos_indU : setup.center_pos_indU_list) {
    std::vector<std::vector<Real>> rslts{};

    std::size_t num_flavors = flavor_pairs.size();
    for (std::size_t i = 0; i < num_flavors; i++){
      // execute the current flavor of for_each
      rslts.push_back(eval_stencil_overlap_(pos_indU.data(), extent, n_ghost, Stencil{},
                                            flavor_pairs[i].second));

      if (i == 0) continue;

      // compare with the first flavor of for_each

      Real* ref = rslts[0].data();
      Real* cur = rslts[i].data();

      for (int iz = 0; iz < extent.nz; iz++) {
        for (int iy = 0; iy < extent.ny; iy++) {
          for (int ix = 0; ix < extent.nx; ix++) {
            int ind3D = ix + extent.nx * (iy + extent.ny * iz);

            bool ref_nonzero = ref[ind3D] > 0.0;
            bool cur_nonzero = cur[ind3D] > 0.0;

            if (ref_nonzero != cur_nonzero) {
              int index[3] = {ix,iy,iz};
              FAIL() << "Encountered an inconsistency when comparing the '"
                     << flavor_pairs[0].first << "' flavor of for_each against the '"
                     << flavor_pairs[i].first << "' flavor at (ix, iy, iz) = "
                     << Vec3_to_String(index) << ". Both should be zero or neither should be zero.";
            }

          }
        }
      }

    }

  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define some machinery to help with testing the full feedback functionality where we check our expectations about the total volume enclosed by the stencil
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

std::optional<Real> try_get_(const std::map<std::string, Real>& m, const std::string& key)
{
  if (auto rslt = m.find(key); rslt != m.end())  return {rslt->second};
  return {};
}

#ifdef DE
const bool idual = true;
#else
const bool idual = false;
#endif

void init_field_vals_(cuda_utilities::DeviceVector<Real>& data, std::size_t field_size,
                      const std::map<std::string, Real>& dflt_vals) 
{
  std::size_t total_size = data.size();

  Real* ptr = data.data();

  // default density should be 0.1 particles per cc
  // default thermal energy density should correspond to pressure of 1e3 K / cm**3 (for a gamma of 5/3)
  const Real density       = try_get_(dflt_vals, "density").value_or(1482737.17012665);
  const Real thermal_edens = try_get_(dflt_vals, "thermal_edens").value_or(0.00021335 *1.5);
  const Real vx            = try_get_(dflt_vals, "velocity_x").value_or(0.0);
  const Real vy            = try_get_(dflt_vals, "velocity_y").value_or(0.0);
  const Real vz            = try_get_(dflt_vals, "velocity_z").value_or(0.0);
  const Real tot_edens = thermal_edens + 0.5 * density * (vx * vx + vy * vy + vz * vz);

  auto loop_fn = [=] __device__ (int index) {
    ptr[index] = density;
    ptr[grid_enum::momentum_x * field_size + index] = density * vx;
    ptr[grid_enum::momentum_y * field_size + index] = density * vy;
    ptr[grid_enum::momentum_z * field_size + index] = density * vz;
    ptr[grid_enum::Energy * field_size + index]     = tot_edens;
# ifdef DE
    ptr[grid_enum::GasEnergy * field_size + index]   = thermal_edens;
# endif        
  };

  gpuFor(field_size, loop_fn);
}

// make sure this lives for the lifetime of the test where the data gets used!
struct TestFieldData {

private: // attributes
  // when cuda_utilities::DeviceVector is updated so that it can be moved in the future, it won't
  // be necessary to wrap particle_ids_ in a unique_ptr.
  std::unique_ptr<cuda_utilities::DeviceVector<Real>> data_;

  Extent3D single_field_extent_; // must include ghost zones

public:
  TestFieldData(Extent3D single_field_extent, std::map<std::string, Real> dflt_vals)
    : data_(nullptr),
      single_field_extent_(single_field_extent)
  {
    const std::size_t single_field_size = single_field_extent.nx*single_field_extent.ny*single_field_extent.nz;
    data_ = std::make_unique<cuda_utilities::DeviceVector<Real>>((5+idual) * single_field_size);
    init_field_vals_(*data_, single_field_size, dflt_vals);
  }

  TestFieldData(TestFieldData&&) = default;
  TestFieldData& operator=(TestFieldData&&) = default;

  Real* dev_ptr() { return data_->data(); }

  Extent3D single_field_extent() { return single_field_extent_; }

  std::vector<Real> host_copy()
  {
    std::vector<Real> out(this->data_->size());
    this->data_->cpyDeviceToHost(out);
    return out;
  }

  // this is inefficient! But should get the job done!
  void print_debug_info()
  {
    std::vector<Real> tmp = this->host_copy();
    Extent3D single_field_extent = this->single_field_extent_;

    auto print_fn = [single_field_extent, &tmp](int field_index, const std::string& name) {
      std::size_t field_offset = single_field_extent.nx*single_field_extent.ny*single_field_extent.nz;

      std::size_t output_indent_offset = name.size() + 2;
      std::string arr_str = array_to_string(tmp.data() + field_index*field_offset, single_field_extent,
                                            output_indent_offset);
      printf("%s: %s\n", name.c_str(), arr_str.c_str());
    };

    print_fn(grid_enum::density, "density");
    print_fn(grid_enum::momentum_x, "momentum_x");
    print_fn(grid_enum::momentum_y, "momentum_y");
    print_fn(grid_enum::momentum_z, "momentum_z");
    print_fn(grid_enum::Energy,     "etot_dens");
# ifdef DE
    print_fn(grid_enum::GasEnergy, "ethermal_dens");
# endif
  }

  /* copy TestFieldData into a new object and shift the reference frame */
  TestFieldData change_ref_frame(Arr3<Real> bulk_velocity) {

    const Extent3D& single_field_extent = this->single_field_extent_;

    TestFieldData out = TestFieldData(single_field_extent, {});

    const Real* in_ptr = this->data_->data();
    Real* out_ptr = out.data_->data();

    const std::size_t field_size = single_field_extent.nx*single_field_extent.ny*single_field_extent.nz;

    auto loop_fn = [in_ptr, out_ptr, field_size, bulk_velocity] __device__ (int index) {
      Real density   = in_ptr[index];
      Real old_mom_x = in_ptr[grid_enum::momentum_x * field_size + index];
      Real old_mom_y = in_ptr[grid_enum::momentum_y * field_size + index];
      Real old_mom_z = in_ptr[grid_enum::momentum_z * field_size + index];

      Real old_KE_dens = 0.5 * ((old_mom_x * old_mom_x) +
                                (old_mom_y * old_mom_y) +
                                (old_mom_z * old_mom_z)) / density;

      Real new_mom_x = old_mom_x - (bulk_velocity[0] * density);
      Real new_mom_y = old_mom_y - (bulk_velocity[1] * density);
      Real new_mom_z = old_mom_z - (bulk_velocity[2] * density);

      Real new_KE_dens = 0.5 * ((new_mom_x * new_mom_x) +
                                (new_mom_y * new_mom_y) +
                                (new_mom_z * new_mom_z)) / density;

      Real new_e = in_ptr[grid_enum::Energy * field_size + index] + (new_KE_dens - old_KE_dens);

      out_ptr[index] = density;
      out_ptr[grid_enum::momentum_x * field_size + index] = new_mom_x;
      out_ptr[grid_enum::momentum_y * field_size + index] = new_mom_y;
      out_ptr[grid_enum::momentum_z * field_size + index] = new_mom_z;
      out_ptr[grid_enum::Energy * field_size + index]     = new_e;
# ifdef DE
      out_ptr[grid_enum::GasEnergy* field_size + index]   = in_ptr[grid_enum::GasEnergy* field_size + index];
# endif
    };

    gpuFor(field_size, loop_fn);
    return out;
  }
};

void assert_fielddata_allclose(TestFieldData& actual_test_field_data, 
                               TestFieldData& ref_test_field_data, 
                               bool only_thermale_and_density = false,
                               double rtol = 0.0, double atol = 0.0)
{
  std::vector<Real> ref_data    = ref_test_field_data.host_copy();
  std::vector<Real> actual_data = actual_test_field_data.host_copy();

  // to do: we should really check consistency of extent between actual & desired
  Extent3D extent = ref_test_field_data.single_field_extent();

  const std::size_t single_field_size = extent.nx*extent.ny*extent.nz;

  const std::map<std::string,int> field_index_map = {
    {"density",    grid_enum::density},
    {"momentum_x", grid_enum::momentum_x},
    {"momentum_y", grid_enum::momentum_y},
    {"momentum_z", grid_enum::momentum_z},
  # ifdef DE
    {"ethermal_dens", grid_enum::GasEnergy},
  # endif
    {"etot_dens", grid_enum::Energy}
  };

  auto compare = [&](const std::string& name, Real* actual, Real* ref) {
    if (actual == nullptr) actual = actual_data.data() + single_field_size * field_index_map.at(name);
    if (ref == nullptr) ref = ref_data.data() + single_field_size * field_index_map.at(name);
    std::string err_msg = "problem comparing the field: " + name;
    assert_allclose(actual, ref, extent, rtol, atol, false, err_msg);
  };

  compare("density", nullptr, nullptr);
  if (not only_thermale_and_density) {
    compare("momentum_x", nullptr, nullptr);
    compare("momentum_y", nullptr, nullptr);
    compare("momentum_z", nullptr, nullptr);
    compare("etot_dens", nullptr, nullptr);
  } else {
    // compute thermal_energy density:
    std::vector<std::vector<Real>> ethermal_l{};
    for (Real* field_ptr : {actual_data.data(), ref_data.data()}) {
      Real* dens = field_ptr + single_field_size * grid_enum::density;
      Real* mom_x = field_ptr + single_field_size * grid_enum::momentum_x;
      Real* mom_y = field_ptr + single_field_size * grid_enum::momentum_y;
      Real* mom_z = field_ptr + single_field_size * grid_enum::momentum_z;
      Real* tot_edens = field_ptr + single_field_size * grid_enum::Energy;

      std::vector<Real> thermal_energy(single_field_size);
      for (std::size_t i = 0; i < single_field_size; i++) {
        Real ke_dens = 0.5 * (mom_x[i]*mom_x[i] + mom_y[i]*mom_y[i] + mom_z[i]*mom_z[i])/dens[i];
        thermal_energy[i] = tot_edens[i] - ke_dens;
      }
      ethermal_l.push_back(thermal_energy);
    }
    compare("(etot_dens - ke_dens)", ethermal_l[0].data(), ethermal_l[1].data());
  }
# ifdef DE
  compare("ethermal_dens", nullptr, nullptr);
# endif
}

// make sure this lives for the lifetime of the test where the data gets used!
struct TestParticleData {

private: // attributes

  // when cuda_utilities::DeviceVector is updated so that it can be moved in the future, it won't
  // be necessary to wrap particle_ids_ in a unique_ptr.
  std::unique_ptr<cuda_utilities::DeviceVector<part_int_t>> particle_ids_;
  std::map<std::string,cuda_utilities::DeviceVector<Real>> general_data_;

public:

  TestParticleData(const std::vector<Arr3<Real>>& pos_vec, const std::map<std::string, Real>& other_props)
    : particle_ids_(nullptr),
      general_data_()
  {
    const std::size_t count = pos_vec.size();

    std::vector<std::string> array_names = {"pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "mass", "age"};

    // initialize host copy of each vector
    std::vector<part_int_t> host_particle_ids(count);
    std::map<std::string, std::vector<Real>> host_data_{};
    for (const std::string& name : array_names) {
      host_data_.emplace(name, count);
    }

    // now fill in the local vectors
    for (std::size_t i = 0; i < count; i++) {

      host_particle_ids[i] = part_int_t(i);

      host_data_.at("pos_x")[i] = pos_vec[i][0];
      host_data_.at("pos_y")[i] = pos_vec[i][1];
      host_data_.at("pos_z")[i] = pos_vec[i][2];

      //printf("pos: %g, %g, %g\n", host_data_.at("pos_x")[i], host_data_.at("pos_y")[i], host_data_.at("pos_z")[i]);

      host_data_.at("vel_x")[i] = try_get_(other_props, "vel_x").value_or(0.0);
      host_data_.at("vel_y")[i] = try_get_(other_props, "vel_y").value_or(0.0);
      host_data_.at("vel_z")[i] = try_get_(other_props, "vel_z").value_or(0.0);
      host_data_.at("mass")[i]  = try_get_(other_props, "mass").value_or(1e3);  // defaults to 1e3 solar masses
      host_data_.at("age")[i]   = try_get_(other_props, "age").value_or(-1e4);  // defaults to -10 kyr (recall this is 
                                                                                 // really the formation time)
    }

    // now copy host vector contents to the device
    this->particle_ids_ = std::make_unique<cuda_utilities::DeviceVector<part_int_t>>(pos_vec.size(), false);
    this->particle_ids_->cpyHostToDevice(host_particle_ids);
    for (const std::string& name : array_names) {
      this->general_data_.emplace(name, count);
      this->general_data_.at(name).cpyHostToDevice(host_data_.at(name));
    }

    // now host-vectors are automatically deallocated
  }

  TestParticleData(TestParticleData&&) = default;
  TestParticleData& operator=(TestParticleData&&) = default;

  part_int_t num_particles() { return part_int_t(particle_ids_->size()); }

  feedback_details::ParticleProps particle_props() {
    return {
      num_particles(), // number of local particles
      particle_ids_->data(),
      general_data_.at("pos_x").data(), general_data_.at("pos_y").data(), general_data_.at("pos_z").data(),
      general_data_.at("vel_x").data(), general_data_.at("vel_y").data(), general_data_.at("vel_z").data(),
      general_data_.at("mass").data(),
      general_data_.at("age").data(),
    };
  }

  feedback_details::ParticleProps props_of_single_particle(int index)
  {
    CHOLLA_ASSERT((index >= 0) and (index < this->num_particles()), "Invalid Particle Index was specified!");
    return {
      1, // number of local particles
      particle_ids_->data() + index,
      general_data_.at("pos_x").data() + index,
      general_data_.at("pos_y").data() + index,
      general_data_.at("pos_z").data() + index,
      general_data_.at("vel_x").data() + index,
      general_data_.at("vel_y").data() + index,
      general_data_.at("vel_z").data() + index,
      general_data_.at("mass").data() + index,
      general_data_.at("age").data() + index,
    };
  }

  std::vector<part_int_t> host_copy_particle_ids()
  {
    std::vector<part_int_t> out(this->num_particles());
    this->particle_ids_->cpyDeviceToHost(out);
    return out;
  }

  std::map<std::string, std::vector<Real>> host_copy_general()
  {
    const std::size_t particle_count = this->num_particles();
    std::map<std::string, std::vector<Real>> out;
    for (auto& kv_pair : this->general_data_) {
      const std::string& key = kv_pair.first;
      cuda_utilities::DeviceVector<Real>& vec = kv_pair.second;

      out.emplace(key, particle_count);
      vec.cpyDeviceToHost(out.at(key));
    }
    return out;
  }

  // this is inefficient! But should get the job done!
  void print_debug_info()
  {
    auto print_fn = [](const std::string& name, auto& host_vec) {
      std::string arr_str = array_to_string(host_vec.data(), int(host_vec.size()));
      printf("%s: %s\n", name.c_str(), arr_str.c_str());
    };

    std::vector<part_int_t> particle_ids = this->host_copy_particle_ids();
    print_fn("ids", particle_ids);
    std::map<std::string, std::vector<Real>> general_data = this->host_copy_general();
    for (auto& kv_pair : general_data) {
      print_fn(kv_pair.first, kv_pair.second);
    }
  }

};

struct FeedbackResults {
  TestFieldData test_field_data;
  TestParticleData test_particle_data;
  std::vector<Real> info;
};

template <typename Prescription = feedback_model::CiCResolvedSNPrescription>
FeedbackResults run_full_feedback_(const int n_ghost, const std::vector<AxProps>& prop_l,
                                   const std::vector<Arr3<Real>>& particle_pos_vec,
                                   feedback_details::OverlapStrat ov_strat,
                                   bool separate_launch_per_particle,
                                   feedback_details::BoundaryStrategy bdry_strat = feedback_details::BoundaryStrategy::excludeGhostParticle_ignoreStencilIssues,
                                   const std::optional<Real> maybe_init_density = std::optional<Real>(),
                                   const std::optional<Real> maybe_init_internal_edens = std::optional<Real>(),
                                   const std::optional<Arr3<Real>> maybe_bulk_vel = std::optional<Arr3<Real>>())
{

  feedback_details::FieldSpatialProps spatial_props{
    // left-edges of active zone:
    prop_l[0].min, prop_l[1].min, prop_l[2].min,
    // right-edges of active zone:
    prop_l[0].min + prop_l[0].cell_width * prop_l[0].num_cells,
    prop_l[1].min + prop_l[1].cell_width * prop_l[1].num_cells,
    prop_l[2].min + prop_l[2].cell_width * prop_l[2].num_cells,
    // cell_widths
    prop_l[0].cell_width, prop_l[1].cell_width, prop_l[2].cell_width,
    // cells along each axis (including ghost zone)
    prop_l[0].num_cells + 2 * n_ghost,  // cells along x (with ghosts)
    prop_l[1].num_cells + 2 * n_ghost,  // cells along y (with ghosts)
    prop_l[2].num_cells + 2 * n_ghost,  // cells along z (with ghosts)
    // number of ghost zones:
    n_ghost,
  };

  // check for optional test-specific field/particle values::
  const Real init_density        = maybe_init_density.value_or(1482737.17012665); // should be 0.1 particles per cc
  const Real init_internal_edens = maybe_init_internal_edens.value_or(0.00021335 *1.5);

  const Arr3<Real> dflt_bulk_vel = {0.0,0.0,0.0};
  const Arr3<Real> bulk_vel = maybe_bulk_vel.value_or(dflt_bulk_vel);

  // allocate the temporary field data!
  const Extent3D full_extent{spatial_props.nx_g, spatial_props.ny_g, spatial_props.nz_g};
  TestFieldData test_field_data(full_extent, {{"density",       init_density},
                                              {"velocity_x",    bulk_vel[0]},
                                              {"velocity_y",    bulk_vel[1]},
                                              {"velocity_z",    bulk_vel[2]},
                                              {"thermal_edens", init_internal_edens}});

  const std::size_t num_particles = particle_pos_vec.size();
  // allocate the temporary particle data!
  TestParticleData test_particle_data(particle_pos_vec, {{"vel_x", bulk_vel[0]},
                                                         {"vel_y", bulk_vel[1]},
                                                         {"vel_z", bulk_vel[2]}});

  // Declare/allocate device buffer for holding the number of supernovae per particle in the current cycle
  cuda_utilities::DeviceVector<int> d_num_SN(particle_pos_vec.size(), true);  // initialized to 0

  // initialize vector so that there is one SN per particle
  {
    std::vector<int> tmp(num_particles, 1);
    d_num_SN.cpyHostToDevice(tmp);
  }

  // allocate a vector to hold summary-info. Make sure to initialize the counters to 0
  std::vector<Real> info(feedinfoLUT::LEN, 0.0);

  // give some dummy vals:
  const feedback_details::CycleProps cycle_props{0.0,  // current time
                                                 0.1,  // length of current timestep 
                                                 1};   // the current cycle-number

  feedback_details::OverlapScheduler ov_scheduler(ov_strat, spatial_props.nx_g, spatial_props.ny_g, spatial_props.nz_g);

  if (separate_launch_per_particle) {
    // actually execute feedback
    for (std::size_t i = 0; i < particle_pos_vec.size(); i++){
      std::array<Real, feedinfoLUT::LEN> info_tmp;
      for (int j = 0; j < feedinfoLUT::LEN; j++) { info_tmp[j] = 0.0; }

      feedback_details::Exec_Cluster_Feedback_Kernel<Prescription>(
        test_particle_data.props_of_single_particle(int(i)), spatial_props, cycle_props, 
        info_tmp.data(), test_field_data.dev_ptr(), d_num_SN.data() + i, ov_scheduler, bdry_strat);

      for (int j = 0; j < feedinfoLUT::LEN; j++) { info[j] += info_tmp[j]; }
    }
  } else {
    feedback_details::Exec_Cluster_Feedback_Kernel<Prescription>(
      test_particle_data.particle_props(), spatial_props, cycle_props,
      info.data(), test_field_data.dev_ptr(), d_num_SN.data(), ov_scheduler, bdry_strat);
  }

  FeedbackResults out{std::move(test_field_data), std::move(test_particle_data), std::move(info)};
  return out;
}

} // anonymous namespace

bool is_integer_(Real val) { return std::trunc(val) == val; }

void basic_infosummary_checks_(std::vector<Real> info) {
  ASSERT_EQ(info.size(), feedinfoLUT::LEN);

  // we may need to revisit the following if we ever add more summary-stats
  for (int i = 0; i < feedinfoLUT::LEN; i++) {
    ASSERT_GE(info[i], 0.0);
  }

  ASSERT_TRUE(is_integer_(info[feedinfoLUT::countSN]));
  ASSERT_TRUE(is_integer_(info[feedinfoLUT::countResolved]));
  ASSERT_TRUE(is_integer_(info[feedinfoLUT::countUnresolved]));
  ASSERT_EQ(info[feedinfoLUT::countSN],
            info[feedinfoLUT::countResolved] + info[feedinfoLUT::countUnresolved]);
}

// check the equality of all integers in actual and ref
void check_infosummary_int_equality_(std::vector<Real> actual, std::vector<Real> ref) {
  basic_infosummary_checks_(actual);
  basic_infosummary_checks_(ref);

  auto is_loseless_integer = [](Real val) { 
    if ((sizeof(Real) == 4) and (fabs(val) > 16777217)) {
      // in this case Real is a 32 bit float and may encode an integer that can't
      // be losslessly represented
      return false;
    } else if ((sizeof(Real) == 8) and (fabs(val) > 9007199254740993)) {
      // in this case Real is a 64 bit float and may encode an integer that can't
      // be losslessly represented
      return false;
    } else {
      return std::trunc(val) == val;
    }
  };

  for (int i = 0; i < feedinfoLUT::LEN; i++) {
    if (is_loseless_integer(actual[i]) and is_loseless_integer(ref[i])){
      ASSERT_EQ(actual[i], ref[i]);
    }
  }
}

void check_equality(FeedbackResults& rslt_actual, FeedbackResults& rslt_ref)
{
  int num_particles = int(rslt_ref.test_particle_data.num_particles());
  {
    std::vector<part_int_t> actual_particle_ids = rslt_actual.test_particle_data.host_copy_particle_ids();
    std::vector<part_int_t> ref_particle_ids = rslt_ref.test_particle_data.host_copy_particle_ids();
    assert_allclose(actual_particle_ids.data(), ref_particle_ids.data(),
                    {num_particles, 1, 1}, 0.0, 0.0, false,
                    "problem comparing the particle_ids");
  }

  {
    std::map<std::string, std::vector<Real>> actual_data = rslt_actual.test_particle_data.host_copy_general();
    std::map<std::string, std::vector<Real>> ref_data = rslt_ref.test_particle_data.host_copy_general();

    for (const auto& kv_pair : actual_data) {
      const std::string& key = kv_pair.first;

      std::vector<Real>& actual_vec = actual_data.at(key);
      std::vector<Real>& ref_vec     = ref_data.at(key);

      std::string err_msg = "problem comparing the particle property: " + key;

      assert_allclose(actual_vec.data(), ref_vec.data(),
                      {num_particles, 1, 1}, 0.0, 0.0, false, err_msg);
    }
  }

  assert_fielddata_allclose(rslt_actual.test_field_data, rslt_ref.test_field_data,
                            false, 0.0, 0.0);

  // perform a check of the info-summary-statistics
  check_infosummary_int_equality_(rslt_actual.info, rslt_ref.info);
}

template <typename T>
class tALLFeedbackFull : public testing::Test {
public:
  using PrescriptionT=T;
};

using MyPrescriptionTypes = ::testing::Types<feedback_model::CiCResolvedSNPrescription,
                                             feedback_model::CiCLegacyResolvedAndUnresolvedPrescription>;
TYPED_TEST_SUITE(tALLFeedbackFull, MyPrescriptionTypes);

// in this test we check that results are identical if we inject feedback for a bunch of supernovae
// that are directly on top of each other in 2 cases:
// - a case where we launch a separate kernel for each supernova
// - a case where we use the OverlapStrat functionallity to launch a single kernel, but within that
//   kernel, we sequentially launch the supernova feedback
TYPED_TEST(tALLFeedbackFull, CheckingOverlapStrat)
{
  using Prescription = typename TestFixture::PrescriptionT;

  const int n_ghost = 0;
  const Real dx = 1.0 / 256.0;
  const std::vector<AxProps> ax_prop_l = {{5, 0.0, dx}, {5,0.0, dx}, {5,0.0, dx}};

  // initialize 50 star particles directly atop each other
  const std::vector<Arr3<Real>> particle_pos_vec(50, {2.4 *dx, 2.4 *dx, 2.4 *dx});

  // Get the reference answer - here we sequentially launch one kernel after to handle feedback of 
  // each individual particle
  FeedbackResults rslt_ref = run_full_feedback_<Prescription>(
    n_ghost, ax_prop_l, particle_pos_vec, feedback_details::OverlapStrat::ignore, true,
    feedback_details::BoundaryStrategy::excludeGhostParticle_ignoreStencilIssues);
  FeedbackResults rslt_actual = run_full_feedback_<Prescription>(
    n_ghost, ax_prop_l, particle_pos_vec, feedback_details::OverlapStrat::sequential, false,
    feedback_details::BoundaryStrategy::excludeGhostParticle_ignoreStencilIssues);

  if (false) {
    printf("\nLooking at the OverlapStrat::ignore approach:\n");
    rslt_ref.test_field_data.print_debug_info();
    rslt_ref.test_particle_data.print_debug_info();
    printf("\nLooking at the OverlapStrat::sequential approach:\n");
    rslt_actual.test_field_data.print_debug_info();
    rslt_actual.test_particle_data.print_debug_info();
  }

  check_equality(rslt_actual, rslt_ref);

  ASSERT_EQ(rslt_actual.info[feedinfoLUT::countSN], particle_pos_vec.size());
}

struct InjectSummary {
  Real mass;
  Real net_mom_x;
  Real net_mom_y;
  Real net_mom_z;
  Real abs_mom_mag;
  Real thermal_energy;
};

// calculate the amount that is injected in each quantity
InjectSummary calc_inject_summary_(TestFieldData& field_data, Real init_density,
                                   Real init_internal_edens, Arr3<Real> bulk_vel,
                                   Real cell_vol)
{
  Extent3D extent = field_data.single_field_extent();
  const std::size_t single_field_size = extent.nx*extent.ny*extent.nz;

  std::vector<Real> vec = field_data.host_copy();

  InjectSummary out{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < single_field_size; i++) {
    const Real dens      = vec[single_field_size * grid_enum::density + i];
    const Real mom_x     = vec[single_field_size * grid_enum::momentum_x + i];
    const Real mom_y     = vec[single_field_size * grid_enum::momentum_y + i];
    const Real mom_z     = vec[single_field_size * grid_enum::momentum_z + i];
    const Real tot_edens = vec[single_field_size * grid_enum::Energy + i];

    Real ke_dens = 0.5 * (mom_x*mom_x + mom_y*mom_y + mom_z*mom_z)/dens;
    Real thermal_energy = tot_edens - ke_dens;

    // compute how different the momentum density is from the initial value
    const Real excess_mom_x = (mom_x - init_density * bulk_vel[0]);
    const Real excess_mom_y = (mom_y - init_density * bulk_vel[1]);
    const Real excess_mom_z = (mom_z - init_density * bulk_vel[2]);

    out.mass           += (dens - init_density);
    out.net_mom_x      += excess_mom_x;
    out.net_mom_y      += excess_mom_y;
    out.net_mom_z      += excess_mom_z;
    out.abs_mom_mag    += sqrt((excess_mom_x * excess_mom_x) +
                               (excess_mom_y * excess_mom_y) +
                               (excess_mom_z * excess_mom_z));
    out.thermal_energy += (thermal_energy - init_internal_edens);
  }

  out.mass           *= cell_vol;
  out.net_mom_x      *= cell_vol;
  out.net_mom_y      *= cell_vol;
  out.net_mom_z      *= cell_vol;
  out.abs_mom_mag    *= cell_vol;
  out.thermal_energy *= cell_vol;

  return out;
}

// in this test, we look into the actual injected amounts!
TYPED_TEST(tALLFeedbackFull, ComparingInjectionMagnitudes)
{
  using Prescription = typename TestFixture::PrescriptionT;

  const int n_ghost = 0;
  const Real dx = 1.0 / 256.0;
  const std::vector<AxProps> ax_prop_l = {{5, 0.0, dx}, {5,0.0, dx}, {5,0.0, dx}};

  // initialize 1 star particle
  const std::vector<Arr3<Real>> particle_pos_vec(1, {2.4 *dx, 2.4 *dx, 2.4 *dx});

  const Real density        = 1e9; // solar-masses per kpc**3
    // default thermal energy density should correspond to pressure of 1e4 K / cm**3 (for a gamma of 5/3)
  const Real internal_edens = 0.0021335 *1.5;
  const Arr3<Real> bulk_vel = {0.0, 0.0, 0.0};

  // launch the feedback
  FeedbackResults rslt = run_full_feedback_<Prescription>(
    n_ghost, ax_prop_l, particle_pos_vec, feedback_details::OverlapStrat::ignore, true,
    feedback_details::BoundaryStrategy::excludeGhostParticle_ignoreStencilIssues,
    {density}, {internal_edens}, {bulk_vel});

  ASSERT_EQ(rslt.info[feedinfoLUT::countSN], 1);

  InjectSummary summary = calc_inject_summary_(rslt.test_field_data, density, internal_edens, bulk_vel,
                                               dx * dx * dx);
  if (false) {
    
    printf("from_grid:  mass = %g, net_mom = {%g,%g,%g}, abs_mom_mag = %e, thermal_energy/erg = %e\n",
           summary.mass, summary.net_mom_x, summary.net_mom_y, summary.net_mom_z,
           summary.abs_mom_mag, summary.thermal_energy * FORCE_UNIT * LENGTH_UNIT);

    Real info_e = rslt.info[feedinfoLUT::totalEnergy] + rslt.info[feedinfoLUT::totalUnresEnergy];
    printf("from summary_stats:   abs_mom_mag = %e energy/erg = %e\n\n",
           rslt.info[feedinfoLUT::totalMomentum], 
           info_e * FORCE_UNIT * LENGTH_UNIT);
  }

  EXPECT_NEAR(feedback::MASS_PER_SN, summary.mass, 3.6e-15 * feedback::MASS_PER_SN);

  if (rslt.info[feedinfoLUT::countResolved] > 0) {
    Real rtol = 0.0;

    EXPECT_EQ(0.0, summary.net_mom_x);
    EXPECT_EQ(0.0, summary.net_mom_y);
    EXPECT_EQ(0.0, summary.net_mom_z);
    EXPECT_EQ(0.0, summary.abs_mom_mag);
    EXPECT_NEAR(feedback::ENERGY_PER_SN, summary.thermal_energy,
                rtol * feedback::ENERGY_PER_SN);

    // sanity check!
    EXPECT_NEAR(feedback::ENERGY_PER_SN, rslt.info[feedinfoLUT::totalEnergy],
                rtol * feedback::ENERGY_PER_SN);
  } else {
    // NOTE: feedback::FINAL_MOMENTUM does NOT directly specify the injected radial-momentum
    //       (the radial momentum also depends on the local conditions)

    Real rtol = 3e-16;
    Real atol = 9e-18;

    EXPECT_NEAR(0.0, summary.net_mom_x, atol);
    EXPECT_NEAR(0.0, summary.net_mom_y, atol);
    EXPECT_NEAR(0.0, summary.net_mom_z, atol);
    // for thermal energy, we theoretically need tolerance since we added and subtracted 
    // kinetic energy
    EXPECT_NEAR(0.0, summary.thermal_energy, atol);
    EXPECT_NEAR(rslt.info[feedinfoLUT::totalMomentum], summary.abs_mom_mag,
                rtol * rslt.info[feedinfoLUT::totalMomentum]);

    // sanity checks!
    EXPECT_EQ(rslt.info[feedinfoLUT::totalUnresEnergy], 0.0);
  }
}

// in this test, we compare a case with and without bulk velocity
TYPED_TEST(tALLFeedbackFull, ComparingNonThermalPropsDiffRefFrames)
{
  using Prescription = typename TestFixture::PrescriptionT;

  const int n_ghost = 0;
  const Real dx = 1.0 / 256.0;
  const std::vector<AxProps> ax_prop_l = {{5, 0.0, dx}, {5,0.0, dx}, {5,0.0, dx}};

  // initialize some star particles directly atop each other
  const std::vector<Arr3<Real>> particle_pos_vec(1, {2.4 *dx, 2.4 *dx, 2.4 *dx});

  [[maybe_unused]] const Real init_density = 1e8; // solar-masses per kpc**3

  // Get the reference answer (in the reference frame where there is no bulk velocity)
  [[maybe_unused]] Arr3<Real> bulk_vel_NULLCASE = {0.0, 0.0, 0.0};
  FeedbackResults rslt_ref = run_full_feedback_<Prescription>(
    n_ghost, ax_prop_l, particle_pos_vec, feedback_details::OverlapStrat::ignore, true,
    feedback_details::BoundaryStrategy::excludeGhostParticle_ignoreStencilIssues,
    {init_density}, {}, {bulk_vel_NULLCASE});

  [[maybe_unused]] Arr3<Real> bulk_vel_ALT = {0.000205, 0.0, 0.0}; // roughly 200 km/s
  FeedbackResults rslt_actual = run_full_feedback_<Prescription>(
    n_ghost, ax_prop_l, particle_pos_vec, feedback_details::OverlapStrat::ignore, true,
    feedback_details::BoundaryStrategy::excludeGhostParticle_ignoreStencilIssues,
    {init_density}, {}, {bulk_vel_ALT});

  // shift the reference-frame of the case with the bulk velocity
  TestFieldData actual_field_shifted = rslt_actual.test_field_data.change_ref_frame(
    Arr3<Real>{1 * bulk_vel_ALT[0], 1 * bulk_vel_ALT[1], 1 * bulk_vel_ALT[2]});

  if (true) {
    printf("\nLooking at the case without bulk velocity:\n");
    rslt_ref.test_field_data.print_debug_info();
    rslt_ref.test_particle_data.print_debug_info();
    printf("\nLooking at the case with bulk velocity:\n");
    rslt_actual.test_field_data.print_debug_info();
    rslt_actual.test_particle_data.print_debug_info();
    printf("\nLooking at the second case after shifting reference frame:\n");
    actual_field_shifted.print_debug_info();
  }

  double rtol = 2.0e-9; // it would be nice to specify tolerances on a per-field basis
  double atol = 5e-7;
  assert_fielddata_allclose(actual_field_shifted, rslt_ref.test_field_data,
                            false, rtol, atol);

  ASSERT_EQ(rslt_actual.info[feedinfoLUT::countSN], particle_pos_vec.size());

  // TODO: update so that we actually adjust the reference frame of the case with
  // bulk-velocity. This will let us more robustly check unresolved feedback!
}
/* This provides unit-tests for extracted easily-testable components of the feedback module.
 *
 * This mainly includes things like the deposition stencil
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
#include <utility>
#include <optional>

// External Includes
#include <gtest/gtest.h> // Include GoogleTest and related libraries/headers

#include "../global/global.h"
#include "../utils/gpu.hpp" // gpuFor
#include "../utils/DeviceVector.h" // gpuFor

#include "../feedback/feedback_model.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define some general-purpose testing tools. These may be a little overkill for this particular file.
// It may make sense to move these to the testing_utils file and namespace. 
///////////////////////////////////////////////////////////////////////////////////////////////////////

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

enum struct StencilEvalKind{
  enclosed_stencil_vol_frac, /*!< compute the fraction of the total stencil volume enclosed by each cell */
  enclosed_cell_vol_frac     /*!< compute the fraction of each cell's volume that is enclosed by the stencil */
};

/* Updates elements in the ``out_data`` 3D-array with the fraction of the volume that is enclosed by
 * the specified ``stencil``, that is centered at the given position.
 *
 * \note
 * Right now, this function should only be executed by a single thread-block with a single thread. This
 * choice reflects the fact that a single thread is historically assigned to a single particle.
 */
template<typename Stencil>
__global__ void Stencil_Overlap_Kernel_(Real* out_data, double pos_x_indU, double pos_y_indU, double pos_z_indU,
                                        int nx_g, int ny_g, Stencil stencil, StencilEvalKind eval)
{
  // first, define the lambda function that actually updates out_data
  auto update_entry_fn = [out_data](double dV, int indx3D) -> void { out_data[indx3D] = dV; };

  // second, execute update_entry at each location where the stencil overlaps with the cells
  switch(eval) {
    case StencilEvalKind::enclosed_stencil_vol_frac:
      stencil.for_each(pos_x_indU, pos_y_indU, pos_z_indU, nx_g, ny_g, update_entry_fn);
      break;
    case StencilEvalKind::enclosed_cell_vol_frac:
      stencil.for_each_enclosedCellVol(pos_x_indU, pos_y_indU, pos_z_indU, nx_g, ny_g, update_entry_fn);
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

  hipLaunchKernelGGL(Stencil_Overlap_Kernel_, num_blocks, threads_per_block, 0, 0,
                     data.data(), pos_indxU[0], pos_indxU[1], pos_indxU[2],
                     full_extent.nx, full_extent.ny, stencil, eval);

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
  __device__ void for_each(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                           int nx_g, int ny_g, Function f) const
  {
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
  __device__ void for_each_enclosedCellVol(Real pos_x_indU, Real pos_y_indU, Real pos_z_indU,
                                           int nx_g, int ny_g, Function f)
  {
    this->for_each(pos_x_indU, pos_y_indU, pos_z_indU, nx_g, ny_g, f);
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
                                                            feedback_model::CICDepositionStencil{});

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

/* test some expected trends as we slowly move a stencil to the right along the
 * x-axis.
 *
 * \param n_ghost the number of ghost-zones to use in the calculation
 * \param tot_vol_atol the acceptable absolute error bound between the empirical
 *                     total-overlap value and 1.0 (the expected value)tolerance for the 
 */
template <typename Stencil>
void sliding_stencil_test(int n_ghost, double tot_vol_atol = 0.0) {
  Extent3D full_extent{2*n_ghost + 4,  // x-axis
                       2*n_ghost + 3,  // y-axis
                       2*n_ghost + 3}; // z-axis

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
      eval_stencil_overlap_(pos_indxU, full_extent, n_ghost, Stencil{})
    );


    //int num_indents = 2;
    //std::string tmp_arr = array_to_string(overlap_results.back().data(), full_extent, num_indents);
    //printf("\n  %s\n:", tmp_arr.c_str());

    // sanity check: ensure non-zero vals sum to 1
    std::string tmp = Vec3_to_String(pos_indxU);
    double total_overlap = 0.0;
    for (const double & overlap : overlap_results.back()) {
      total_overlap += overlap;
    }
    // in the future, we may nee
    ASSERT_NEAR(total_overlap, 1.0, tot_vol_atol) << "the sum of the stencil-vol-frac is not 1.0, when "
                                                  << "when the stencil is centered at " << Vec3_to_String(pos_indxU);
  }

  // perform some checks based on the ranges of cells with overlap:
  // - currently we just pick a single y_ind, z_ind combination. 
  // - It might be nice to consider all y_ind, z_ind combinations. This could be a little messy:
  //   -> need to handle rows when there isn't ANY overlap
  //   -> need to handle cases (especially for super-sampled stencil) where a given y_ind/z_ind near
  //      the edge of the stencil won't have any overlap at all, except when the stencil is positioned
  //      at very particular x-values (basically it has to do with the distance of the subgrid to 
  //      the stencil-center)

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
  sliding_stencil_test<feedback_model::CICDepositionStencil>(0);
}


TEST(tALLFeedbackSphere27Stencil, SlidingTest)
{
  // primary stencil size we would use
  sliding_stencil_test<feedback_model::Sphere27DepositionStencil<2>>(0,2e-16);
  // just testing this case because we can
  sliding_stencil_test<feedback_model::Sphere27DepositionStencil<4>>(0,3e-16);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Define some tests where we check our expectations about the total volume enclosed by the stencil
///////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Stencil>
void stencil_volume_check(Arr3<Real> center_offset_from_cellEdge, int n_ghost, Stencil stencil,
                          double expected_vol, double vol_rtol = 0.0, double stencil_overlap_rtol = 0.0)
{
  // compute the center of the stencil and the extent of the grid for evaluating the stencil
  Arr3<Real> pos_indU{};
  for (std::size_t i = 0; i < 3; i++){
    // confirm the offset falls in the range 0 <= center_offset < 1
    ASSERT_GE(center_offset_from_cellEdge[i], 0.0) << "Test parameter is flawed";
    ASSERT_LT(center_offset_from_cellEdge[i], 1.0) << "Test parameter is flawed";

    pos_indU[i] = Stencil::max_enclosed_neighbors + center_offset_from_cellEdge[i];
  }
  // choose an extent value that we know will work (even when n_ghost is zero!)
  Extent3D full_extent{1 + 2*(n_ghost + Stencil::max_enclosed_neighbors),
                       1 + 2*(n_ghost + Stencil::max_enclosed_neighbors),
                       1 + 2*(n_ghost + Stencil::max_enclosed_neighbors)};

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
  stencil_volume_check(Arr3<Real>{0.0,0.0,0.0}, 0, feedback_model::CICDepositionStencil{},
                       /* expected_vol = */ 1.0, /* vol_rtol = */ 0.0, /*stencil_overlap_rtol =*/ 0.0);
  stencil_volume_check(Arr3<Real>{0.5,0.5,0.5}, 0, feedback_model::CICDepositionStencil{},
                       /* expected_vol = */ 1.0, /* vol_rtol = */ 0.0, /*stencil_overlap_rtol =*/ 0.0);
}

TEST(tALLFeedbackSphere27Stencil, StencilVolumeTest)
{
  const double radius = 1; // in units of cell_widths
  const double expected_vol = 4 * 3.141592653589793 * (radius * radius) / 3;

  stencil_volume_check(Arr3<Real>{0.0,0.0,0.0}, 0, feedback_model::Sphere27DepositionStencil<2>{},
                       expected_vol, /* vol_rtol = */ 0.05, /*stencil_overlap_rtol =*/ 0.0);
  stencil_volume_check(Arr3<Real>{0.125,0.0,0.0}, 0, feedback_model::Sphere27DepositionStencil<2>{},
                       expected_vol, /* vol_rtol = */ 0.0004, /*stencil_overlap_rtol =*/ 0.0);
  stencil_volume_check(Arr3<Real>{0.5,0.5,0.5}, 0, feedback_model::Sphere27DepositionStencil<2>{},
                       expected_vol, /* vol_rtol = */ 0.05, /*stencil_overlap_rtol =*/ 0.0);
}

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
#pragma once

#include <mpi.h>

#include "../../utils/gpu.hpp"


/*!
 * \brief Encapsulates core logic for using Discrete Sine Transform (implemented in terms of FFTs)
 * to solve Poisson's equation with isolated boundaries.
 *
 * \par Restrictions
 * This solver makes 2 large assumptions about the resulting potential:
 *   1. the laplacian of the potential is zero at the boundaries.
 *      - equivalently, the input density field must have a value of 0 at the boundaries.
 *      - if this is not satisfied, the density field can't be represented by a Discrete Sine
 *        Transform
 *   2. The gradient of the potential is 0 at the boundaries
 *      - equivalently, the gravitation field (aka accelereation due to gravity) is 0 at the
 *        boundaries
 * It is the caller's responsiblity to ensure that the density field passed into the solver
 * corresponds to a gravitational potential that largely satisfies these constraints. If the caller
 * does not satisfy these requirements, the solver will still execute. It might just provide a highly
 * inaccurate result
 */
class PoissonZero3DBlockedGPU
{
 public:
  PoissonZero3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);
  ~PoissonZero3DBlockedGPU();

  /* The minimum size of each buffer that is passed to `this->solve`*/
  long bytes() const { return bytes_; }

  /*!
   * \brief Solve Possion's equation. To be memory efficient, this will reuse
   * the density array as a scratch buffer
   *
   * This solver will NOT provide robust solutions for an arbitrary density
   * field. To get a robust solution, the density field *MUST* correspond to a
   * potential that satisfies the assumptions outlined in the documentation
   * for this class as a whole
   *
   * \param[in]  bytes The nominal sizes of the density and potential buffers.
   * This must be at least as large as `this->bytes()`.
   * \param[in]  density A device pointer to a buffer holding the values at each
   * grid point corresponding to the RHS of Poisson's equation. After the input
   * values are read from this buffer, it will be reused as scratch space. The
   * contents of this buffer are unspecified once the calculation concludes.
   * \param[out] potential A device pointer to a buffer holding the computed
   * gravitational potential.
   *
   * \note
   * The `bytes` argument is only used to ensure that the buffers are large
   * enough to hold intermediate results. The logic for reading the input values
   * from `density` and writing the final values of `potential` depend entirely
   * on the values used to configure `this` and entirely ignore the `bytes`
   * argument.
   */
  void solve(long bytes, double *density, double *potential) const;

 private:
  double ddi_, ddj_, ddk_;
  int idi_, idj_, idk_;
  int mi_, mj_, mk_;
  int ni_, nj_, nk_;
  int mp_, mq_;
  int idp_, idq_;
  MPI_Comm commI_, commJ_, commK_;
  int di_, dj_, dk_;
  int dip_, djp_, djq_, dkq_;
  int ni2_, nj2_, nk2_;
  long bytes_;
  cufftHandle d2zi_, d2zj_, d2zk_;
#ifndef MPI_GPU
  double *ha_, *hb_;
#endif
};

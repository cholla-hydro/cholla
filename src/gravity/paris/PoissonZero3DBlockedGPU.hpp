#pragma once

#include <mpi.h>

#include "../../utils/gpu.hpp"


/*!
 * \brief This class contains all of the core logic for actually solving Poisson's
 * equation (using FFTs). An instance of this object lies at the core of various
 * Paris Solvers.
 */
class PoissonZero3DBlockedGPU
{
 public:
  PoissonZero3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);
  ~PoissonZero3DBlockedGPU();

  /* The minimum size of each buffer that is passed to `this->solve`*/
  long bytes() const { return bytes_; }

  /*!
   * \brief Solve Possion's equation. To be memory efficient
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

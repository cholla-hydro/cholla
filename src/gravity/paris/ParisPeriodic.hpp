#pragma once

#include "HenryPeriodic.hpp"

/**
 * @brief Periodic Poisson solver using @ref Henry FFT filter.
 */
class ParisPeriodic
{
 public:
  /**
   * @param[in] n[3] { Global number of cells in each dimension, without ghost
   * cells. }
   * @param[in] lo[3] { Physical location of the global lower bound of each
   * dimension. }
   * @param[in] hi[3] { Physical location of the global upper bound of each
   * dimension, minus one grid cell. The one-cell difference is because of the
   * periodic domain. See @ref Potential_Paris_3D::Initialize for an example
   * computation of these arguments. }
   * @param[in] m[3] { Number of MPI tasks in each dimension. }
   * @param[in] id[3] { Coordinates of this MPI task, starting at `{0,0,0}`. }
   */
  ParisPeriodic(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3], double dx);

  /**
   * @return { Number of bytes needed for array arguments for @ref solve. }
   */
  size_t bytes() const { return henry.bytes(); }

  /**
   * @detail { Solves the Poisson equation for the potential derived from the
   * provided density. Assumes periodic boundary conditions. Assumes fields have
   * no ghost cells. Uses a 3D FFT provided by the @ref Henry class. }
   * @param[in] bytes { Number of bytes allocated for arguments @ref density and
   * @ref potential. Used to ensure that the arrays have enough extra work
   * space. }
   * @param[in,out] density { Input density field. Modified as a work array.
   *                          Must be at least @ref bytes() bytes, likely larger
   * than the original field. }
   * @param[out] potential { Output potential. Modified as a work array.
   *                         Must be at least @ref bytes() bytes, likely larger
   * than the actual output field. }
   */
  void solvePotential(size_t bytes, double *density, double *potential) const;
  void solveEddingtonTensor(size_t bytes, double *source, double *tensor, int component) const;

 private:
  int ni_, nj_;  //!< Number of elements in X and Y dimensions
#if defined(PARIS_3PT) || defined(PARIS_5PT)
  int nk_;  //!< Number of elements in Z dimension
#endif
  double dx_, ddi_, ddj_, ddk_;  //!< Frequency-independent terms in Poisson solve
  HenryPeriodic henry;      //!< FFT filter object
};

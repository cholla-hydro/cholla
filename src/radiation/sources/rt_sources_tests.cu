/*! \file rt_sources_tests.cu
 *  \brief Unit tests for the RT source model: kernel normalization, the ghost fold,
 *         lifetimes, and periodic migration on a single rank. Built only with TEST=true. */

#ifdef RT

  #include <gtest/gtest.h>

  #include <cmath>
  #include <vector>

  #include "../../global/global.h"
  #include "../../utils/gpu.hpp"
  #include "rt_sources.h"

using namespace rt_sources;

namespace
{

/*! A periodic, single-rank grid of n^3 active cells with n_ghost = 3, on [0, n) so dx = 1. */
RT_State Make_State(int n_active = 8, int n_ghost = 3)
{
  RT_State s;
  s.nghost = n_ghost;
  s.nx = s.ny = s.nz = n_active + 2 * n_ghost;
  s.ncells          = s.nx * s.ny * s.nz;
  s.dx = s.dy = s.dz = 1.0;
  s.xbound = s.ybound = s.zbound = 0;
  s.xdglobal = s.ydglobal = s.zdglobal = n_active;
  s.xblocal = s.yblocal = s.zblocal = 0;
  s.xblocal_max = s.yblocal_max = s.zblocal_max = n_active;
  for (int i = 0; i < 6; i++) s.flags[i] = 1;
  s.t_mid_myr = 0;
  return s;
}

/*! One packed source: position, zero velocity, lum, birth, life, id. */
std::vector<Real> One_Source(Real x, Real y, Real z, Real lum = 1, Real t_birth = -1, Real t_life = 0)
{
  return {x, y, z, 0, 0, 0, lum, t_birth, t_life, 0};
}

/*! Deposit one population and fold; return the field on the host. */
std::vector<Real> Deposit_And_Fold(SourceStore &store, KernelType kernel, Real radius, const RT_State &s)
{
  RTPopulation pop;
  pop.kernel        = kernel;
  pop.kernel_radius = radius;

  Real *dev_rs = nullptr;
  GPU_Error_Check(cudaMalloc((void **)&dev_rs, s.ncells * sizeof(Real)));
  GPU_Error_Check(cudaMemset(dev_rs, 0, s.ncells * sizeof(Real)));

  FoldBuffers buf;
  buf.Allocate(s);
  Deposit_Sources(store, pop, s, dev_rs);
  Fold_Source_Ghosts(dev_rs, s, buf);
  buf.Free();

  std::vector<Real> rs(s.ncells);
  GPU_Error_Check(cudaMemcpy(rs.data(), dev_rs, s.ncells * sizeof(Real), cudaMemcpyDeviceToHost));
  cudaFree(dev_rs);
  return rs;
}

bool Is_Ghost(const RT_State &s, int i, int j, int k)
{
  const int g = s.nghost;
  return i < g || j < g || k < g || i >= s.nx - g || j >= s.ny - g || k >= s.nz - g;
}

/*! Sum over active cells times the cell volume, and the largest |value| in a ghost cell. */
void Integrate(const RT_State &s, const std::vector<Real> &rs, Real &total, Real &max_ghost)
{
  total     = 0;
  max_ghost = 0;
  for (int k = 0; k < s.nz; k++)
    for (int j = 0; j < s.ny; j++)
      for (int i = 0; i < s.nx; i++) {
        const Real v = rs[i + s.nx * (j + s.ny * k)];
        if (Is_Ghost(s, i, j, k))
          max_ghost = std::fmax(max_ghost, std::fabs(v));
        else
          total += v * s.dx * s.dy * s.dz;
      }
}

}  // namespace

// A lum = 1 CIC source on a cell corner reproduces the 8-cell source of the Iliev /
// Shapiro-Giroux initial conditions: 0.125 / dx^3 in each of the 8 cells.
TEST(tRTSources, CICAtCellCornerMatchesICSource)
{
  RT_State s = Make_State();
  SourceStore store;
  auto src = One_Source(4.0, 4.0, 4.0);
  store.Append_Host(src.data(), 1);

  auto rs     = Deposit_And_Fold(store, KernelType::cic, 1, s);
  const int g = s.nghost;
  for (int k = 3; k <= 4; k++)
    for (int j = 3; j <= 4; j++)
      for (int i = 3; i <= 4; i++) EXPECT_DOUBLE_EQ(rs[(i + g) + s.nx * ((j + g) + s.ny * (k + g))], 0.125);

  Real total, max_ghost;
  Integrate(s, rs, total, max_ghost);
  EXPECT_NEAR(total, 1.0, 1e-13);
}

// Every kernel conserves the luminosity exactly, including when its footprint crosses the
// periodic boundary in all three directions (this exercises the fold, corners included).
TEST(tRTSources, AllKernelsConserveAcrossPeriodicCorner)
{
  const KernelType kernels[4] = {KernelType::ngp, KernelType::cic, KernelType::tsc, KernelType::tophat};
  for (KernelType kernel : kernels) {
    RT_State s = Make_State();
    SourceStore store;
    auto src = One_Source(0.1, 7.95, 0.3, 2.5);
    store.Append_Host(src.data(), 1);

    auto rs = Deposit_And_Fold(store, kernel, 2.0, s);
    Real total, max_ghost;
    Integrate(s, rs, total, max_ghost);
    EXPECT_NEAR(total, 2.5, 1e-12) << "kernel " << Kernel_Type_Name(kernel);
    EXPECT_EQ(max_ghost, 0.0) << "kernel " << Kernel_Type_Name(kernel);
  }
}

// Sources before their birth or after their death contribute nothing.
TEST(tRTSources, LifetimeWindow)
{
  RT_State s = Make_State();
  SourceStore store;
  auto a = One_Source(2.5, 2.5, 2.5, 1.0, 10.0, 5.0);  // lives 10 <= t < 15 Myr
  store.Append_Host(a.data(), 1);

  const Real t_mids[3]   = {9.0, 12.0, 15.0};
  const Real expected[3] = {0.0, 1.0, 0.0};
  for (int n = 0; n < 3; n++) {
    s.t_mid_myr = t_mids[n];
    auto rs     = Deposit_And_Fold(store, KernelType::ngp, 1, s);
    Real total, max_ghost;
    Integrate(s, rs, total, max_ghost);
    EXPECT_NEAR(total, expected[n], 1e-14) << "t_mid = " << t_mids[n];
  }
}

// A source drifting out through the upper x face comes back in at the lower face.
TEST(tRTSources, PeriodicMigrationOnOneRank)
{
  RT_State s = Make_State();
  SourceStore store;
  std::vector<Real> src = {7.9, 3.0, 3.0, 0.5, 0, 0, 1, -1, 0, 42};
  store.Append_Host(src.data(), 1);

  store.Drift(1.0);  // x -> 8.4, outside [0, 8)
  store.Migrate(s);

  std::vector<Real> out;
  store.Copy_To_Host(out);
  ASSERT_EQ(store.n_local, 1);
  EXPECT_NEAR(out[SRC_X], 0.4, 1e-12);
  EXPECT_EQ(out[SRC_ID], 42);
}

#endif  // RT
/*! \file RT_functions.cu
 #  \brief Definitions of functions for the RT solver */

#ifdef CUDA
  #ifdef RT

    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>

    #include "../global/global.h"
    #include "../global/global_cuda.h"
    #include "../grid/grid3D.h"
    #include "../utils/gpu.hpp"
    #include "RT_functions.h"
    #include "alt/atomic_data.h"
    #include "alt/constant.h"
    #include "radiation.h"

void Rad3D::Initialize_GPU()
{
  // copy over data from CPU fields
  CudaSafeCall(
      cudaMemcpy(rtFields.dev_rf, rtFields.rf, (1 + 2 * n_freq) * grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));

  // initialize values for the other fields:
  //   if these fields exist on CPU, just copy them
  //   if not, set to 0
  if (rtFields.et != nullptr) {
    CudaSafeCall(cudaMemcpy(rtFields.dev_et, rtFields.et, 6 * grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
  } else {
    CudaSafeCall(cudaMemset(rtFields.dev_et, 0, 6 * grid.n_cells * sizeof(Real)));
  }
  if (rtFields.rs != nullptr) {
    CudaSafeCall(cudaMemcpy(rtFields.dev_rs, rtFields.rs, grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
  } else {
    CudaSafeCall(cudaMemset(rtFields.dev_rs, 0, grid.n_cells * sizeof(Real)));
  }
}

void Rad3D::Copy_RT_Fields(void)
{
  // copy data back from GPU to CPU
  CudaSafeCall(
      cudaMemcpy(rtFields.rf, rtFields.dev_rf, (1 + 2 * n_freq) * grid.n_cells * sizeof(Real), cudaMemcpyDeviceToHost));
}

void __global__ Set_RT_Boundaries_Periodic_Kernel(int direction, int side, int n_i, int n_j, int nx, int ny, int nz,
                                                  int n_ghost, int n_freq, struct Rad3D::RT_Fields rtFields);
void Set_RT_Boundaries_Periodic(int direction, int side, int nx, int ny, int nz, int n_ghost, int n_freq,
                                struct Rad3D::RT_Fields& rtFields)
{
  int n_i, n_j, size;
  int nx_g, ny_g, nz_g;
  nx_g = nx;
  ny_g = ny;
  nz_g = nz;

  if (direction == 0) {
    n_i = ny_g;
    n_j = nz_g;
  }
  if (direction == 1) {
    n_i = nx_g;
    n_j = nz_g;
  }
  if (direction == 2) {
    n_i = nx_g;
    n_j = ny_g;
  }

  size = n_ghost * n_i * n_j;

  // set values for GPU kernels
  int ngrid = (size - 1) / TPB_RT;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_RT, 1, 1);

  // Copy the kernel to set the boundary cells (non MPI)
  hipLaunchKernelGGL(Set_RT_Boundaries_Periodic_Kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, n_i, n_j, nx_g,
                     ny_g, nz_g, n_ghost, n_freq, rtFields);
}

// Set boundary cells for radiation fields (non MPI)
void Rad3D::rtBoundaries(void)
{
  Set_RT_Boundaries_Periodic(0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  Set_RT_Boundaries_Periodic(0, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  Set_RT_Boundaries_Periodic(1, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  Set_RT_Boundaries_Periodic(1, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  Set_RT_Boundaries_Periodic(2, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
  Set_RT_Boundaries_Periodic(2, 1, grid.nx, grid.ny, grid.nz, grid.n_ghost, n_freq, rtFields);
}

// Function to launch the kernel to calculate absorption coefficients
void __global__ Calc_Absorption_Kernel(int nx, int ny, int nz, Real dx, CrossSectionInCU cs,
                                       const Real* __restrict__ dens, Real* __restrict__ abc);
void Rad3D::Calc_Absorption(Real* dev_scalar)
{
  int ngrid = (grid.n_cells + TPB_RT - 1) / TPB_RT;

  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_RT, 1, 1);

  auto ufac =
      1.0e-24 / Constant::mb * DENSITY_UNIT * LENGTH_UNIT;  // ufac is per length, hence multiplied by Units::Length.
    #ifdef COSMOLOGY
      #error "Not implemented.\n"
    #endif
  CrossSectionInCU xs;
  xs.HIatHI     = Physics::AtomicData::CrossSections()->csHIatHI * ufac;
  xs.HIatHeI    = Physics::AtomicData::CrossSections()->csHIatHeI * ufac;
  xs.HIatHeII   = Physics::AtomicData::CrossSections()->csHIatHeII * ufac;
  xs.HeIatHeI   = Physics::AtomicData::CrossSections()->csHeIatHeI * ufac;
  xs.HeIatHeII  = Physics::AtomicData::CrossSections()->csHeIatHeII * ufac;
  xs.HeIIatHeII = Physics::AtomicData::CrossSections()->csHeIIatHeII * ufac;

  // Launch the kernel
  hipLaunchKernelGGL(Calc_Absorption_Kernel, dim1dGrid, dim1dBlock, 0, 0, grid.nx, grid.ny, grid.nz, grid.dx, xs,
                     dev_scalar, rtFields.dev_abc);
}

// Function to launch the OTVETIteration kernel
// should function the way "LAUNCH" does on slack
void __global__ OTVETIteration_Kernel(int nx, int ny, int nz, int n_ghost, Real dx, bool lastIteration,
                                      const Real rsFarFactor, const Real* __restrict__ rs, const Real* __restrict__ et,
                                      const Real* __restrict__ rfOT, const Real* __restrict__ rfNear,
                                      const Real* __restrict__ rfFar, const Real* __restrict__ abc,
                                      Real* __restrict__ rfNearNew, Real* __restrict__ rfFarNew, int deb);
void Rad3D::OTVETIteration(void)
{
  const int numThreadsPerBlock = 256;
  int ngrid                    = (grid.n_cells + numThreadsPerBlock - 1) / numThreadsPerBlock;

  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(numThreadsPerBlock, 1, 1);

  // Launch the kernel for one frequency at a time
  for (int freq = 0; freq < n_freq; freq++) {
    auto rfOT      = rtFields.dev_rf;
    auto rfNearOld = rtFields.dev_rf + grid.n_cells * (1 + freq);
    auto rfFarOld  = rtFields.dev_rf + grid.n_cells * (1 + n_freq + freq);
    auto rfNearNew = rtFields.dev_rfNew + grid.n_cells * 0;
    auto rfFarNew  = rtFields.dev_rfNew + grid.n_cells * 1;

    hipLaunchKernelGGL(OTVETIteration_Kernel, dim1dGrid, dim1dBlock, 0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost,
                       grid.dx, lastIteration, rsFarFactor, rtFields.dev_rs, rtFields.dev_et, rfOT, rfNearOld, rfFarOld,
                       rtFields.dev_abc + freq * grid.n_cells, rfNearNew, rfFarNew, (freq == 0 ? 1 : 0));

    CudaSafeCall(cudaMemcpyAsync(rfNearOld, rfNearNew, grid.n_cells * sizeof(Real), cudaMemcpyDeviceToDevice));
    CudaSafeCall(cudaMemcpyAsync(rfFarOld, rfFarNew, grid.n_cells * sizeof(Real), cudaMemcpyDeviceToDevice));
  }
}

// CPU function that calls the GPU-based RT functions
void Rad3D::rtSolve(Real* dev_scalar)
{
  auto dt = grid.dt;

  // first call absorption coefficient kernel
  Calc_Absorption(dev_scalar);

  int niters                   = this->num_iterations;
  Real speedOfLightInCodeUnits = 3e10 / VELOCITY_UNIT;
  int niters2                  = (dt > 0 ? static_cast<int>(1 + speedOfLightInCodeUnits * dt / grid.dx) : niters);
  if (niters > niters2) niters = niters2;

  for (int iter = 0; iter < niters; iter++) {
    this->lastIteration = (iter == niters - 1);

    // then call OTVET iteration kernel
    OTVETIteration();

    // then call boundaries kernel
    rtBoundaries();
  }
  /*

  INTRO:

    Radiation field at each frequency is represented with 2 fields:
    "near" field g (from soures inside the box) and "far" field f
    (from sources outside the box). They are combined as

    J = \bar{J} f + L (g - \bar{g} f),  \bar{f} = 1

    where \bar{J} is cosmic background and L is the frequency dependence of
    sources (c.f. stellar spectrum).

    This is done so that one can account for cosmological effects in \bar{J},
    while having radiation near sources being shaped by the source spectrum.

    One can also consider an approximation f=0 and only track one field per
    frequency, although the only use case for that limit is modeling SF inside
    GMCs.

    Reference: Gnedin_2014_ApJ_793_29

  GIVEN:

    1) abundance fields as mass density \rhoHI, \rhoHeI, \rhoHeII
    2) radiation source field \rs
    3) optically thin near radiation field \ot ("0-frequency field", 0-frequency far field is just identically 1)

       \ot = \frac{1}{4\pi} int d^3 x_1 \frac{\rs(x_1)}{(x-x_1)^2}

    4) 6 components of the near Eddington tensor \et^{ij} (far Eddington tensor is a unit matrix over 3).

       \ot \et^{ij} = \frac{1}{4\pi} int d^3 x_1 \frac{rs(x_1)(x^i-x_1^i)(x^j-x_1^j)}{(x-x_1)^4}

    5) near and far radiation fields per frequency, \rfn, \rff

    6) 7 temporary storage fields \temp[1:7] (can be reduced to 4 with extra
    calculations)

    7) boundary data for all fields onan 18-point stencil (cube minus vertices)

  ALGORITHM:

    loop over iterations: at each iteration

    loop over frequencies: for each frequency \f:

      1) compute the dimensionless absorption coeffcient\abc:

         \abc = \csFact*(\csHI*\rhoHI+\csHeI*\rhoHeI+\csHeII*\rhoHeII)

      where \cs... are cross sections for 3 species at frequency \f (some
      could be zero) and

         \csFact = <unit-conversion-factor>*<cell-size>/<baryon-mass>

      ** uses \temp[0] extra storage for \abc
      ** runs on a separate CUDA kernel

      2) compute edge absortion coefficients \abch and fluxes \flux:

         \abchX[i+1/2,j,k] = epsNum + 0.5*(\abc[i,j,k]+abc[i+1,j,k])
         ...

         \fluxX[i+1/2,j,k) = (ux+uy+uz)/\abch[i+1/2,j,k];
         ux = \rf[i+1,j,k]*\et^{xx}[i+1,j,k] - \rf[i,j,k]*\et^{xx}[i,j,k]
         uy = 0.25*(\rf[i+1,j+1,k]*\et^{xy}[i+1,j+1,k] +
                    \rf[i,j+1,k]*\et^{xy}[i,j+1,k] -
                    \rf[i+1,j-1,k]*\et^{xy}[i+1,j-1,k] -
                    \rf[i,j-1,k]*\et^{xy}[i,j-1,k])
         ...

      where epsNum=1e-6 is to avoid division for zero when \abc=0.

      ** uses \temp[1:4] for \flux
      ** uses \temp[4:7] for \abch, or \abch needs to be recomputed in the
      next step
      ** runs on a separate CUDA kernel

      3) update the radiation field

         minus_wII = \et^{xx}[i,j,k]*(\abchX[i-1/2,j,k]+\abchX[i+1/2,j,k]) +
  \et^{yy}[i,j,k]*(\abchY[i,j-1/2,k]+\abchY[i,j+1/2,k]) + ...


         A = gamma/(1+gamma*(\abc[i,j,k]+minus_wII))
         d = dx*\rs[i,j,k] - \abc[i,j,k]*\rf[i,j,k] + \fluxX[i+1/2,j,k] - \fluxX[i-1/2,j,k] + ...

         rfNew = \rf[i,j,k] + alpha*A*d

         if(pars.lastIteration && rfNew>facOverOT*\ot[i,j,k]) rfNew = facOverOT*\ot[i,j,k];
         rf[i,j,k] = (rfNew<0 ? 0 : rfNew);

      where

         dx is the cell size

         alpha = 0.8  (a CFL-like number)
         gamma = 1
         facOverOT = 1.5 (a new parameter I am still exploring)

      ** runs on a separate CUDA kernel

      4) repeat for the far field

    end loop over frequencies

  pass boundaries

    end loop over iterations


    ** number of iterations is a simulation parameter.
    ** to limit the signal propagation spped to c, it should be capped at
    each step to

      unsigned int numIterationsAtC = (unsigned int)(1+mSpeedOfLightInCodeUnits*dt/dx);


  */
}

  #endif  // RT
#endif    // CUDA

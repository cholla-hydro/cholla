/*! \file RT_functions.cu
 #  \brief Definitions of functions for the RT solver */

//#ifdef CUDA
  #ifdef RT

    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>

    #include "../global/global.h"
    #include "../global/global_cuda.h"
    #include "../grid/grid3D.h"
    #include "../utils/gpu.hpp"
    #include "RT_functions.h"
    #include "RT_functors.h"
    #include "alt/atomic_data.h"
    #include "alt/constant.h"
    #include "radiation.h"
    #ifdef MPI_CHOLLA
      #include "../mpi/mpi_routines.h"
    #endif

void Rad3D::Initialize_GPU()
{

#ifdef OTVET
  // copy over data from CPU fields
  GPU_Error_Check(
      cudaMemcpy(rtFields.dev_rf, rtFields.rf, (1 + n_fpfreq * n_freq) * grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
#endif // OTVET
#ifdef M1
  // copy over data from CPU fields
  GPU_Error_Check(
      cudaMemcpy(rtFields.dev_rf, rtFields.rf, n_fpfreq * n_freq * grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
#endif // M1

  // initialize values for the other fields:
  //   if these fields exist on CPU, just copy them
  //   if not, set to 0

  // eddington tensor for OTVET
#ifdef OTVET
  if (rtFields.et != nullptr) {
    GPU_Error_Check(cudaMemcpy(rtFields.dev_et, rtFields.et, 6 * grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
  } else {
    GPU_Error_Check(cudaMemset(rtFields.dev_et, 0, 6 * grid.n_cells * sizeof(Real)));
  }
#endif //OTVET


  // source radiation field
  if (rtFields.rs != nullptr) {
    GPU_Error_Check(cudaMemcpy(rtFields.dev_rs, rtFields.rs, grid.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
  } else {
    GPU_Error_Check(cudaMemset(rtFields.dev_rs, 0, grid.n_cells * sizeof(Real)));
  }
}

void Rad3D::Copy_RT_Fields(void)
{

#ifdef OTVET
  // copy data back from GPU to CPU
  GPU_Error_Check(
      cudaMemcpy(rtFields.rf, rtFields.dev_rf, (1 + n_fpfreq * n_freq) * grid.n_cells * sizeof(Real), cudaMemcpyDeviceToHost));

  GPU_Error_Check(cudaMemcpy(rtFields.et, rtFields.dev_et, 6 * grid.n_cells * sizeof(Real), cudaMemcpyDeviceToHost));
#endif //OTVET

#ifdef M1
  // copy data back from GPU to CPU
  GPU_Error_Check(
      cudaMemcpy(rtFields.rf, rtFields.dev_rf, n_fpfreq * n_freq * grid.n_cells * sizeof(Real), cudaMemcpyDeviceToHost));
#endif
}

int Load_RT_Fields_To_Buffer(int direction, int side, int nx, int ny, int nz, int n_ghost, int n_fpfreq, int n_freq,
                             struct Rad3D::RT_Fields& rtFields, Real* buffer)
{
  // printf( "Loading RT Fields Buffer: Dir %d  side: %d \n", direction, side );
  int nx_rt, ny_rt, nz_rt, size_buffer, n_ghost_rt, n_ghost_transfer, n_i, n_j, ngrid;

  // for now assume RT grid has the same dimensions as hydro grid
  n_ghost_rt       = n_ghost;
  n_ghost_transfer = n_ghost;
  nx_rt            = nx;
  ny_rt            = ny;
  nz_rt            = nz;

  if (direction == 0) {
    n_i = ny_rt;
    n_j = nz_rt;
  }
  if (direction == 1) {
    n_i = nx_rt;
    n_j = nz_rt;
  }
  if (direction == 2) {
    n_i = nx_rt;
    n_j = ny_rt;
  }

  // buffer size for 1 field
  size_buffer = n_ghost_transfer * n_i * n_j;

  // set values for GPU kernels
  ngrid = (size_buffer - 1) / TPB_RT + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_RT, 1, 1);

  hipLaunchKernelGGL(Load_RT_Buffer_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i, n_j, nx_rt,
                     ny_rt, nz_rt, n_ghost_transfer, n_ghost_rt, n_fpfreq, n_freq, rtFields, buffer);
  GPU_Error_Check(cudaDeviceSynchronize());  // Loading Buffer needs to synchronize so it is complete before MPI sends are called

  // printf( "Loaded RT Fields Buffer: Dir %d  side: %d \n", direction, side );
  return size_buffer * n_fpfreq * n_freq;
}

void Unload_RT_Fields_From_Buffer(int direction, int side, int nx, int ny, int nz, int n_ghost, int n_fpfreq, int n_freq,
                                  struct Rad3D::RT_Fields& rtFields, Real* buffer)
{
  // printf( "Unloading RT Fields Buffer: Dir %d  side: %d \n", direction, side );
  int nx_rt, ny_rt, nz_rt, size_buffer, n_ghost_rt, n_ghost_transfer, n_i, n_j, ngrid;

  // for now assume RT grid has the same dimensions as hydro grid
  n_ghost_rt       = n_ghost;
  n_ghost_transfer = n_ghost;
  nx_rt            = nx;
  ny_rt            = ny;
  nz_rt            = nz;

  if (direction == 0) {
    n_i = ny_rt;
    n_j = nz_rt;
  }
  if (direction == 1) {
    n_i = nx_rt;
    n_j = nz_rt;
  }
  if (direction == 2) {
    n_i = nx_rt;
    n_j = ny_rt;
  }

  // buffer size for 1 field
  size_buffer = n_ghost_transfer * n_i * n_j;

  // set values for GPU kernels
  ngrid = (size_buffer - 1) / TPB_RT + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_RT, 1, 1);

  hipLaunchKernelGGL(Unload_RT_Buffer_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i, n_j,
                     nx_rt, ny_rt, nz_rt, n_ghost_transfer, n_ghost_rt, n_fpfreq,  n_freq, rtFields, buffer);
  // synchronize not needed here because the next MPI call will be preceded by a load kernel which will synchronize
}

__global__ void Set_RT_Boundaries_Periodic_Kernel(int direction, int side, int n_i, int n_j, int nx, int ny, int nz,
                                                  int n_ghost, int n_fpfreq,  int n_freq, struct Rad3D::RT_Fields rtFields);
void Set_RT_Boundaries_Periodic(int direction, int side, int nx, int ny, int nz, int n_ghost, int n_fpfreq,  int n_freq,
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
  int ngrid = (size - 1) / TPB_RT + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_RT, 1, 1);

  // Copy the kernel to set the boundary cells (non MPI)
  hipLaunchKernelGGL(Set_RT_Boundaries_Periodic_Kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, n_i, n_j, nx_g,
                     ny_g, nz_g, n_ghost, n_fpfreq,  n_freq, rtFields);
  // synchronize not needed here because the next MPI call will be preceded by a load kernel which will synchronize
}

// Function to launch the kernel to calculate absorption coefficients
void __global__ Calc_Absorption_Kernel(int nx, int ny, int nz, Real dx, CrossSectionInCU cs,
                                       const Real* __restrict__ dens, Real* __restrict__ abc);
void Rad3D::Calc_Absorption(Real* dev_scalar)
{
  int ngrid = (grid.n_cells - 1) / TPB_RT + 1;

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


#ifdef OTVET
// Function to launch the OTVETIteration kernel
// should function the way "LAUNCH" does on slack
void __global__ OTVETIteration_Kernel(int nx, int ny, int nz, int n_ghost, Real dx, bool lastIteration,
                                      const Real rsFarFactor, const Real* __restrict__ rs, const Real* __restrict__ et,
                                      const Real* __restrict__ rfOT, const Real* __restrict__ rfNear,
                                      const Real* __restrict__ rfFar, const Real* __restrict__ abc,
                                      Real* __restrict__ rfNearNew, Real* __restrict__ rfFarNew, int deb);
#endif //OTVET

// Function to launch the StepRFiIteration kernel
// should function the way "LAUNCH" does on slack
void __global__ StepRFiIteration_Kernel(int nx, int ny, int nz, int n_ghost, 
                                        Real dx, Real cdt2dxRSL, Real gamma,
                                        const Real* __restrict__ rs,
                                        const Real* __restrict__ rfi,
                                        const Real* __restrict__ abc,
                                        const Real* __restrict__ pij_,
                                        Real* __restrict__ rfiNew, int deb);

// Function to limit the RF fields after the iteration kernel
//void __global__ ClipRFi_Kernel(int nx, int ny, int nz, int n_ghost, const Real* __restrict__ rfi, int nout, Real* __restrict__ rfiOut, int deb);
void __global__ ClipRFi_Kernel(int nx, int ny, int nz, int n_ghost, 
                               const Real* __restrict__ rfi, Real* __restrict__ rfiOut, int deb);

// Functor to make the pressure tensor
//void __global__ GLFMakeP_Kernel(int nx, int ny, int nz, int n_ghost, Real dx, const Real* rfi, Real* pij, PijFunctorM1 pf, int deb);
 

#ifdef OTVET
// This function performs the OTVET iteration on the GPU
// and copies the newly updated fields ("New") back onto
// the old ones
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
    auto rfFarOld  = rtFields.dev_rf + grid.n_cells * (1 + n_freq + freq); // Suspicious -- BRANT, n_fpfreq hiding
    auto rfNearNew = rtFields.dev_rfNew + grid.n_cells * 0; // Suspicious -- BRANT, n_fpfreq hiding -- OK for OTVET?
    auto rfFarNew  = rtFields.dev_rfNew + grid.n_cells * 1; // Suspicious -- BRANT, n_fpfreq hiding -- OK for OTVET?

    hipLaunchKernelGGL(OTVETIteration_Kernel, dim1dGrid, dim1dBlock, 0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost,
                       grid.dx, lastIteration, rsFarFactor, rtFields.dev_rs, rtFields.dev_et, rfOT, rfNearOld, rfFarOld,
                       rtFields.dev_abc + freq * grid.n_cells, rfNearNew, rfFarNew, (freq == 0 ? 1 : 0));
    GPU_Error_Check(cudaMemcpyAsync(rfNearOld, rfNearNew, grid.n_cells * sizeof(Real), cudaMemcpyDeviceToDevice));
    GPU_Error_Check(cudaMemcpyAsync(rfFarOld, rfFarNew, grid.n_cells * sizeof(Real), cudaMemcpyDeviceToDevice));
  }
}
#endif //OTVET


#ifdef M1

struct DEVICE_ALIGN_DECL PijFunctorM1
{
    //__global__ void operator()(int offset, int nx, int ny, int nz, 
    __device__ void operator()(int offset, int nx, int ny, int nz, 
                    int ic, int jc, int kc, const Real* rfi, Real* pij, int deb)
    {
        if(ic<offset || jc<offset || kc<offset || ic>=nx-offset || jc>=ny-offset || kc>=nz-offset) return;

        const int nw3 = nx*ny*nz;
        const int idx = ic + nx*(jc+ny*kc);
        const float r =  rfi[idx];
        const float fx = rfi[idx+1*nw3];
        const float fy = rfi[idx+2*nw3];
        const float fz = rfi[idx+3*nw3];

        if(r > 0)
        {
            float flux2 = fx*fx + fy*fy + fz*fz;
            if(flux2 > 0)
            {
                float f2 = min(flux2/(r*r),1.0f);

                float alpha = (3+4*f2)/(5+2*sqrt(4-3*f2));
                float wd = (1-alpha)/2*r;
                float wn = (3*alpha-1)/2*r/flux2;

                pij[idx+0*nw3] = wn*fx*fx + wd;
                pij[idx+1*nw3] = wn*fy*fx;
                pij[idx+2*nw3] = wn*fy*fy + wd;
                pij[idx+3*nw3] = wn*fz*fx;
                pij[idx+4*nw3] = wn*fz*fy;
                pij[idx+5*nw3] = wn*fz*fz + wd;
            }
            else
            {
                pij[idx+0*nw3] = r/3;
                pij[idx+1*nw3] = 0;
                pij[idx+2*nw3] = r/3;
                pij[idx+3*nw3] = 0;
                pij[idx+4*nw3] = 0;
                pij[idx+5*nw3] = r/3;
            }
        }
        else
        {
            pij[idx+0*nw3] = 0;
            pij[idx+1*nw3] = 0;
            pij[idx+2*nw3] = 0;
            pij[idx+3*nw3] = 0;
            pij[idx+4*nw3] = 0;
            pij[idx+5*nw3] = 0;
        }
    }
};

// Perform the M1 iteration
void Rad3D::StepRFiIteration(void)
{
  const int numThreadsPerBlock = 256;
  int ngrid                    = (grid.n_cells + numThreadsPerBlock - 1) / numThreadsPerBlock;

// cdt2dxRSL is cbar*dt/dx, cbar is the effective speed of light which can be less than c 
// in the reduced speed of light approximation used.
//
// gamma is the parameter for the semi-implicit scheme:
// v_1 = v_0 + J*(gamma*v_1+(1-gamma)*v_0), J is the Jacobian
//
//  gamma=0 is the Aubert & Teyssier 2008 scheme.

  int scheme_ = 0;
  Real gamma_sis = 0.5; // semi-implicit scheme parameter
  Real cdt2dxRSL = 0.5; // default case 1
  //Real cdt2dxRSL = (3e10/VELOCITY_UNIT) * grid.dt / grid.dx; // look at moments.cpp
  //cdt2dxRSL /= (1 + gamma_sis); // NEEDS EDITING DEBUG

  switch(scheme_)
  {
    case 0: {
      cdt2dxRSL = 0.25;
      gamma_sis = 0;
      break;
    }
    case 1: {
      cdt2dxRSL = 0.5;
      gamma_sis = 0.5;
      break;   
    }
    case 2: {
      cdt2dxRSL = 0.75;
      gamma_sis = 1.0;
      break;   
    }
    default: {
      cdt2dxRSL = 0.5;
      gamma_sis = 0.5;
      break;   
    }
  }

  //PijFunctorM1& pf;
  PijFunctorM1 pf;

  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(numThreadsPerBlock, 1, 1);

  // Create the pressure tensor by allocating memory on the device
  GPU_Error_Check(cudaMalloc((void**)&rtFields.dev_pij, 6 * grid.n_cells * sizeof(Real)));

  // Launch the StepRFiIteration kernel for one frequency at a time
  for (int freq = 0; freq < n_freq; freq++) { /// hold -- 4 fields per freq * number of freq in M1
    auto rfOld  = rtFields.dev_rf    + grid.n_cells * (n_fpfreq * freq); // old radiation fields at this frequency
    auto rfNew  = rtFields.dev_rfNew;                                    // updated radiation fields at this frequency
    auto abc    = rtFields.dev_abc + freq * grid.n_cells;                // absorption coefficients at this frequency
    auto pij    = rtFields.dev_pij;                                      // reuse pressure tensor each frequency


    // Populate the pressure tensor for this frequency
    //GLFMakeP(nx,ny,nz,n_ghost,dx,rfOld,pij,pf,deb); 
    hipLaunchKernelGGL(GLFMakeP_Kernel, dim1dGrid, dim1dBlock, 0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, 
                       grid.dx,rfOld,pij,pf,(freq == 0 ? 1 : 0));

// DEBUG
    // Step the radiation fields at this frequency
    hipLaunchKernelGGL(StepRFiIteration_Kernel, dim1dGrid, dim1dBlock, 0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost,
                       grid.dx, cdt2dxRSL, gamma_sis, rtFields.dev_rs, rfOld, abc, pij, rfNew, (freq == 0 ? 1 : 0));
    GPU_Error_Check(cudaMemcpyAsync(rfOld, rfNew, n_fpfreq * grid.n_cells * sizeof(Real), cudaMemcpyDeviceToDevice));
  }
  GPU_Error_Check(cudaDeviceSynchronize());

  // Destroy the pressure tensor by freeing memory on the device
  cudaFree(rtFields.dev_pij);
}

// Apply the limiter on the RFi after the M1 iteration
void Rad3D::ClipRFiIteration(void)
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
    auto rfOld  = rtFields.dev_rf    + grid.n_cells * (n_fpfreq * freq); // old radiation fields at this frequency
    auto rfNew  = rtFields.dev_rfNew;                                    // clipped radiation fields at this frequency

    // Clip the radiation fields at this frequency
//    hipLaunchKernelGGL(ClipRFi_Kernel, dim1dGrid, dim1dBlock, 0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, rfOld, nout, rfNew, (freq == 0 ? 1 : 0));
    hipLaunchKernelGGL(ClipRFi_Kernel, dim1dGrid, dim1dBlock, 0, 0, grid.nx, grid.ny, grid.nz, grid.n_ghost, 
                       rfOld, rfNew, (freq == 0 ? 1 : 0));

    // Copy back the clipped result
    GPU_Error_Check(cudaMemcpyAsync(rfOld, rfNew, n_fpfreq * grid.n_cells * sizeof(Real), cudaMemcpyDeviceToDevice));
  }  
  GPU_Error_Check(cudaDeviceSynchronize());
}
#endif //M1



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


  chprintf("Number of RT iterations in rtSolve: %d\n",niters);

  for (int iter = 0; iter < niters; iter++) {
    this->lastIteration = (iter == niters - 1);


#ifdef OTVET
    // then call OTVET iteration kernel
    OTVETIteration();
#endif

#ifdef M1
    // Call the StepRFi Iteration kernel
    // This must create and destroy the pressure fields
    StepRFiIteration();

    // Clip the RFi fields
    ClipRFiIteration();
#endif



    // then call boundaries functions
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


#ifdef GRAVITY
void Rad3D::ComputeEddingtonTensor(const Parameters& P, Grav3D& G)
{
  // Compute the eddington tensor
  Real *rs, *ot, *et[6];

  #ifdef GRAVITY_GPU
  rs = rtFields.dev_rs;
  ot = rtFields.dev_rf;
  for(int j=0; j<6; j++) et[j] = rtFields.dev_et + j*grid.n_cells;
  #else
  rs = rtFields.rs;
  ot = rtFields.rf;
  for(int j=0; j<6; j++) et[j] = rtFields.et + j*grid.n_cells;
  #endif

  G.Poisson_solver.Get_EddingtonTensor(grid.n_ghost,rs,et,ot);
}
#endif // GRAVITY
#endif  // RT
//#endif    // CUDA

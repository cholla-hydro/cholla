/*! \file RT_functions.cu
 #  \brief Definitions of functions for the RT solver */


#ifdef CUDA
#ifdef RT

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"../utils/gpu.hpp"
#include"../global/global.h"
#include"../global/global_cuda.h"
#include"radiation.h"
#include"RT_functions.h"

void Rad3D::Initialize_RT_Fields_GPU(void) {

  // copy over data from CPU fields
  CudaSafeCall( cudaMemcpy(rtFields.dev_rfn, rtFields.rfn, n_freq*n_cells*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(rtFields.dev_rff, rtFields.rff, n_freq*n_cells*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(rtFields.dev_ot, rtFields.ot, n_cells*sizeof(Real), cudaMemcpyHostToDevice) );

  // initialize values for the other fields
  // (set to 0 for now, call a kernel to set different values)
  cudaMemset(rtFields.dev_et, 0, 6*n_cells*sizeof(Real));  
  cudaMemset(rtFields.dev_rs, 0, n_cells*sizeof(Real));  

}

void Rad3D::Copy_RT_Fields(void) {

  // copy data back from GPU to CPU
  CudaSafeCall( cudaMemcpy(rtFields.rfn, rtFields.dev_rfn, n_freq*n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );
  CudaSafeCall( cudaMemcpy(rtFields.rff, rtFields.dev_rff, n_freq*n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );
  CudaSafeCall( cudaMemcpy(rtFields.ot, rtFields.dev_ot, n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );  

}

void __global__ Set_RT_Boundaries_Periodic_Kernel(int direction, int side, int n_i, int n_j, int nx, int ny, int nz, int n_ghost, int n_freq, struct Rad3D::RT_Fields &rtFields){

  int n_cells = nx*ny*nz;
  
  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_src, tid_dst;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;
  
  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost ) return;
  
  if ( direction == 0 ){
    if ( side == 0 ) tid_src = ( nx - 2*n_ghost + tid_k )  + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 0 ) tid_dst = ( tid_k )                   + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 1 ) tid_src = ( n_ghost + tid_k  )        + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dst = ( nx - n_ghost + tid_k )    + (tid_i)*nx  + (tid_j)*nx*ny;
  }
  if ( direction == 1 ){
    if ( side == 0 ) tid_src = (tid_i) + ( ny - 2*n_ghost + tid_k  )*nx  + (tid_j)*nx*ny;
    if ( side == 0 ) tid_dst = (tid_i) + ( tid_k )*nx                    + (tid_j)*nx*ny;
    if ( side == 1 ) tid_src = (tid_i) + ( n_ghost + tid_k  )*nx         + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dst = (tid_i) + ( ny - n_ghost + tid_k )*nx     + (tid_j)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_src = (tid_i) + (tid_j)*nx + ( nz - 2*n_ghost + tid_k  )*nx*ny;
    if ( side == 0 ) tid_dst = (tid_i) + (tid_j)*nx + ( tid_k  )*nx*ny;
    if ( side == 1 ) tid_src = (tid_i) + (tid_j)*nx + ( n_ghost + tid_k  )*nx*ny;
    if ( side == 1 ) tid_dst = (tid_i) + (tid_j)*nx + ( nz - n_ghost + tid_k  )*nx*ny;
  }
  
  for (int i=0; i<n_freq; i++) {
    rtFields.dev_rfn[tid_dst+i*n_cells] = rtFields.dev_rfn[tid_src+i*n_cells];
    rtFields.dev_rff[tid_dst+i*n_cells] = rtFields.dev_rff[tid_src+i*n_cells];
  }
  
}

void Set_RT_Boundaries_Periodic( int direction, int side, int nx, int ny, int nz, int n_ghost, int n_freq, struct Rad3D::RT_Fields &rtFields){
  
  int n_i, n_j, size;
  int nx_g, ny_g, nz_g;
  nx_g = nx;
  ny_g = ny;
  nz_g = nz;

  if ( direction == 0 ){
    n_i = ny_g;
    n_j = nz_g;
  }
  if ( direction == 1 ){
    n_i = nx_g;
    n_j = nz_g;
  }
  if ( direction == 2 ){
    n_i = nx_g;
    n_j = ny_g;
  }

  size = n_ghost * n_i * n_j;

  // set values for GPU kernels
  int ngrid = ( size - 1 ) / TPB_RT;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_RT, 1, 1);

  // Copy the kernel to set the boundary cells (non MPI)
  hipLaunchKernelGGL( Set_RT_Boundaries_Periodic_Kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, n_i, n_j, nx_g, ny_g, nz_g, n_ghost, n_freq, rtFields);


}

// Set boundary cells for radiation fields (non MPI)
void Rad3D::rtBoundaries(void)
{
  Set_RT_Boundaries_Periodic(0,0, nx, ny, nz, n_ghost, n_freq, rtFields); 
  Set_RT_Boundaries_Periodic(0,1, nx, ny, nz, n_ghost, n_freq, rtFields); 
  Set_RT_Boundaries_Periodic(1,0, nx, ny, nz, n_ghost, n_freq, rtFields); 
  Set_RT_Boundaries_Periodic(1,1, nx, ny, nz, n_ghost, n_freq, rtFields); 
  Set_RT_Boundaries_Periodic(2,0, nx, ny, nz, n_ghost, n_freq, rtFields); 
  Set_RT_Boundaries_Periodic(2,1, nx, ny, nz, n_ghost, n_freq, rtFields); 

}

// Function to launch the kernel to calculate absorption coefficients
void Rad3D::Calc_Absorption(Real *dev_scalar)
{

}

// Function to launch the OTVETIteration kernel
// should function the way "LAUNCH" does on slack
void Rad3D::OTVETIteration(void)
{


}

// CPU function that calls the GPU-based RT functions
void Rad3D::rtSolve(Real *dev_scalar)
{
   // first call absorption coefficient kernel
   Calc_Absorption(dev_scalar);

   // then call OTVET iteration kernel
   OTVETIteration();


   // then call boundaries kernel
   rtBoundaries();
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
  3) optically thin near radiation field \ot ("0-frequency field", 0-frequency far field is just idenstically 1)

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

       minus_wII = \et^{xx}[i,j,k]*(\abchX[i-1/2,j,k]+\abchX[i+1/2,j,k]) + \et^{yy}[i,j,k]*(\abchY[i,j-1/2,k]+\abchY[i,j+1/2,k]) + ...


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

#endif // RT
#endif // CUDA

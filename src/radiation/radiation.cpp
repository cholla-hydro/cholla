/*! \file radiation.cpp
 *  \brief Definitions for the radiative transfer wrapper */

#ifdef RT

  #include "radiation.h"

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "RT_functions.h"
  #include "alt/atomic_data.h"
  #include "alt/photo_rates_csi_gpu.h"

Rad3D::Rad3D(const Header& grid_) : grid(grid_)
{
  Physics::AtomicData::Create();
  /// photoRates = new PhotoRatesCSI::TableWrapperGPU(2,6);
  photoRates = new PhotoRatesCSI::TableWrapperGPU(1, 6);
}

Rad3D::~Rad3D()
{
  delete photoRates;
  Physics::AtomicData::Delete();
}

void Rad3D::Initialize_Start(const parameters& params)
{
  num_iterations = params.num_iterations;

  // allocate memory on the host
  rtFields.rf = (Real*)malloc((1 + 2 * n_freq) * grid.n_cells * sizeof(Real));
}

// function to do various initialization tasks, i.e. allocating memory, etc.
void Rad3D::Initialize_Finish()
{
  chprintf("Initializing Radiative Transfer...\n");

  // Allocate memory for abundances (passive scalars added to the hydro grid)
  // This is done in grid3D::Allocate_Memory
  chprintf(" N scalar fields: %d \n", NSCALARS);

  // Allocate memory for radiation fields (non-advecting, 2 per frequency plus 1 optically thin field)
  chprintf("Allocating memory for radiation fields. \n");
  // allocate memory on the device
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_rf, (1 + 2 * n_freq) * grid.n_cells * sizeof(Real)));

  // Allocate memory for Eddington tensor only on device
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_et, 6 * grid.n_cells * sizeof(Real)));

  // Allocate memory for radiation source field only on device
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_rs, grid.n_cells * sizeof(Real)));

  // Allocate temporary fields on device
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_abc, n_freq * grid.n_cells * sizeof(Real)));
  CudaSafeCall(cudaMalloc((void**)&rtFields.dev_rfNew, 2 * grid.n_cells * sizeof(Real)));

  // Initialize Field values (for now)
  this->Initialize_GPU();
}

// function to call the radiation solver from main
void Grid3D::Update_RT()
{
  // passes d_scalar as that is the pointer to the first abundance array, rho_HI
  Rad.rtSolve(C.d_scalar);
}

void Rad3D::Free_Memory(void)
{
  cudaFree(rtFields.dev_rf);
  cudaFree(rtFields.dev_et);
  cudaFree(rtFields.dev_rs);
  cudaFree(rtFields.dev_abc);
  cudaFree(rtFields.dev_rfNew);

  if (rtFields.et != nullptr) free(rtFields.et);
  if (rtFields.rs != nullptr) free(rtFields.rs);
  free(rtFields.rf);
}

#endif  // RT

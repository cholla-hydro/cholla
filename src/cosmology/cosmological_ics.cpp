#ifdef COSMOLOGY
  #include <fstream>
	#include <cstdio>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../utils/cuda_utilities.h"
  #include "../io/io.h"
	#include "../fft/fft_3D.h"
	#include "rng.h"
	#include "field_operations.h"


/*! \fn void Generate_Cosmo_Phi_Init(void)
 *  \brief Create the potentials for cosmological ICs */
void Grid3D::Generate_Cosmo_Phi_Init(struct Parameters *P)
{
	// This function generates the initial potentials required
	// for constructing cosmological initial conditions. Since
	// these potentials may require a substantial memory footprint
	// to compute, they are generated before the main grid and
	// particle memory banks are allocated.

	// First, if we are not setting cosmological ICs, just
	// return
	if (strcmp(P->init, "Cosmological_ICs") ) {
		return;
	}

  int i, j, k, id;
  int istart, jstart, kstart, iend, jend, kend;
  istart = H.n_ghost;
  iend   = H.nx - H.n_ghost;
  if (H.ny > 1) {
    jstart = H.n_ghost;
    jend   = H.ny - H.n_ghost;
  } else {
    jstart = 0;
    jend   = H.ny;
  }
  if (H.nz > 1) {
    kstart = H.n_ghost;
    kend   = H.nz - H.n_ghost;
  } else {
    kstart = 0;
    kend   = H.nz;
  }

	// OK, let's proceed
	chprintf("Generating potentials for cosmological ICs.");

	// set the number of fields
	CP.n_fields = 1;	

	// initialize the RNG properties
	Initialize_Cosmo_Potential_RNG(P);

	// Initialize the FFT as well
	chprintf("Initializing the FFT system");
	fft.Initialize( H.xdglobal, H.ydglobal, H.zdglobal, H.xblocal, H.yblocal, H.zblocal,
		              P->nx, P->ny, P->nz, H.nx_real, H.ny_real, H.nz_real, H.dx, H.dy, H.dz );

	// Allocate the memory needed for the potentials
	Allocate_Cosmo_Potential_Memory();

  // We have allocated the potential arrays, and are ready to proceed

	// step 1) sample xi(m) by generating independent
	//         zero-mean normal deviates with variance N**d at 
	//         each spatial point
	Generate_Normal_Random_Field(CP.d_phi_1,&CP.rng_state);
	Rescale_Field(CP.d_phi_1, H.nx*H.ny*H.nz);

	// copy memory
	cudaMemcpy(CP.phi_1, CP.d_phi_1, CP.n_fields * H.n_cells * sizeof(Real), cudaMemcpyDeviceToHost);

  // reduce the grid values
	Real phi_sum = 0;
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        // get cell index
        id = i + j * H.nx + k * H.nx * H.ny;

        phi_sum += CP.phi_1[id]; // perform a local reduction
      }
    }
  }
  printf("Before FFT procID %d phi_sum %e",procID,phi_sum);


	// step 2) Take the fourier transform
	//         xi(k) = N**-d \sum_m exp( -(2 pi i / M) * kappa \dot m) * xi(m)
	//Rescale_FFT_Field(d_xi_k, 1./(H.nx*H.ny*H.nz));

	fft.Filter_identity(CP.phi_1,CP.phi_1,true);

	// copy memory
	cudaMemcpy(CP.phi_1, CP.d_phi_1, CP.n_fields * H.n_cells * sizeof(Real), cudaMemcpyDeviceToHost);

	phi_sum = 0;
  // reduce the grid values
  for (k = H.n_ghost; k < H.nz - H.n_ghost; k++) {
    for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
      for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        // get cell index
        id = i + j * H.nx + k * H.nx * H.ny;

        phi_sum += CP.phi_1[id]; // perform a local reduction
      }
    }
  }
  printf("After FFT procID %d phi_sum %e",procID,phi_sum);

	// step 2.5) create k vectors
	//Populate_Wavevectors(d_kx, d_ky, d_kz, d_kk);

	// step 3) Multiply xi(k) by the transfer function 
	//         T(k) \equiv [(2 \pi / L)**3 P(k)]^{1/2}
	//         note T(k) is computed at z=0

	// step 3.1) divide by k^2 to take inverse laplacian
	//FFT_Field_Inverse_Laplacian(d_xi_k, d_kk);

	// step 3.5) rescale by growth factor(a)/a
	//Rescale_FFT_Field(d_xi_k, Daa);

	// step 4) Reset the FFT system, free memory
	fft.Reset();
}

/*! \fn void Initialize_Cosmo_Potential_RNG(void)
 *  \brief Initialize the RNG for cosmological ICs potentials */
void Grid3D::Initialize_Cosmo_Potential_RNG(struct Parameters *P)
{
	// Initialize the parameters for the Philox RNG

	// Record the RNG seed from the parameter file
	CP.rng_seed = P->seed;

	// Set the RNG subsequence to be the global MPI Rank + 1
	CP.rng_subsequence = (unsigned long long) (procID);

	// Initialize the RNG offset to be zero
	CP.rng_offset = 0;

	// Call the RNG initialization function on the GPUs
	RNG_Init_GPU(CP.rng_seed, CP.rng_subsequence, CP.rng_offset, &CP.rng_state);
}


/*! \fn void Allocate_Cosmo_Potential_Memory(void)
 *  \brief Allocate the memory allocated for cosmological ICs potentials */
void Grid3D::Allocate_Cosmo_Potential_Memory()
{
  // allocate memory for the phi arrays
  // allocate all the memory to phi_1, to insure contiguous memory
  GPU_Error_Check(cudaHostAlloc((void **)&CP.host, CP.n_fields * H.n_cells * sizeof(Real), cudaHostAllocDefault));

  // point potential variables to the appropriate locations on host
  CP.phi_1    = CP.host;

  // allocate memory for the conserved variable arrays on the device
  GPU_Error_Check(cudaMalloc((void **)&CP.device, CP.n_fields * H.n_cells * sizeof(Real)));
  cuda_utilities::initGpuMemory(C.device, H.n_fields * H.n_cells * sizeof(Real));

  // point potential variables to the appropriate locations on the device
  CP.d_phi_1  = CP.device;

  // initialize host array
  for (int i = 0; i < CP.n_fields * H.n_cells; i++) {
    CP.host[i] = 0.0;
  }
}

/*! \fn void Free_Cosmo_Phi_Init(void)
 *  \brief Free the memory allocated for cosmological ICs potentials */
void Grid3D::Free_Cosmo_Potential_Memory(void)
{
  // free the host phi arrays
  GPU_Error_Check(cudaFreeHost(CP.host));

  // free the device phi arrays
  GPU_Error_Check(cudaFree(CP.device));
}


/*! \fn void Generate_Normal_Random_Field(Real *d_field)
 *  \brief Create a Gaussian random field on a grid */
void Grid3D::Generate_Normal_Random_Field(Real *d_field, rng_parallel_state_t *state)
{
	// Here, d_field has been pre-allocated on the device
	RNG_Normal_Field_GPU(d_field, H.n_cells, H.n_ghost, state);
}

/*! \fn void Rescale_Field(Real *d_x, Real A)
 *  \brief Rescale a field by a constant multiplicative factor. */
void Grid3D::Rescale_Field(Real *d_x, Real A)
{
	// Here, d_x has been pre-allocated on the device
	// Rescale the field by a multiplicative factor.
	Rescale_Field_GPU(d_x, A, H.n_cells, H.n_ghost);
}


/*! \fn void Field_Elementwise_Product(Real *d_x, Real *d_y)
 *  \brief Multiply one field elementwise by another. */
/*
void Grid3D::Field_Elementwise_Product(Real *d_x, Real *d_y)
{
	// Here, d_x and d_y
	// Rescale the field by a multiplicative factor.
	Field_Elementwise_Product_GPU(d_x, d_y, H.n_cells, H.n_ghost);
}*/

/*! \fn void FFT_Field_Inverse_Laplacian(Real *d_x, Real A)
 *  \brief Multiply one field elementwise by another. */
/*void Grid3D::FFT_Field_Inverse_Laplacian(Real *d_x_k, Real *d_kk)
{
	// Here, d_x and d_y
	// Rescale the field by a multiplicative factor.
	FFT_Field_Elementwise_Ratio_Power_GPU(d_x, d_y, H.n_cells, H.n_ghost);
}
*/

/*! \fn void FFT_Populate_Wavevectors(Real *d_kx, Real *d_ky, Real *d_kz, Real *d_kk)
 *  \brief Initialize the wavevector arrays for an FFT grid */
/*void Grid3D::FFT_Populate_Wavevectors(Real *d_kx, Real *d_ky, Real *d_kz, Real *d_kk)
{
	// Populate wavevectors for an FFT grid
	FFT_Populate_Wavevectors_GPU(d_kx, d_ky, d_kz, d_kk, H.n_cells, H.n_ghost);
}*/

#endif
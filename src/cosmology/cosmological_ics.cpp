#ifdef COSMOLOGY
  #include <fstream>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../io/io.h"
	#include "rng.h"


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

	// OK, let's proceed
	chprintf("Generating potentials for cosmological ICs.");

	// set the number of fields
	CP.n_fields = 1;	

	// initialize the RNG properties
	Initialize_Cosmo_Potential_RNG();

	// Allocate the memory needed for the potentials
	Allocate_Cosmo_Potential_Memory();

  // We have allocated the potential arrays, and are ready to proceed

	// step 1) sample xi(m) by generating independent
	//         zero-mean normal deviates with variance N**d at 
	//         each spatial point
	//
	// step 2) Take the fourier transform
	//         xi(k) = N**-d \sum_m exp( -(2 pi i / M) * kappa \dot m) * xi(m)
	//
	// step 2.5) create k vectors

	// step 3) Multiply xi(k) by the transfer function 
	//         T(k) \equiv [(2 \pi / L)**3 P(k)]^{1/2}
	//         note T(k) is computed at z=0

	// step 3.1) divide by k^2 to take inverse laplacian

	// step 3.5) rescale by growth factor(a)/a

}

/*! \fn void Initialize_Cosmo_Potential_RNG(void)
 *  \brief Initialize the RNG for cosmological ICs potentials */
void Grid3D::Initialize_Cosmo_Potential_RNG(struct Parameters *P)
{
	// Initialize the parameters for the Philox RNG

	// Record the RNG seed from the parameter file
	CP.rng_seed = (unsigned long long) P->seed;

	// Set the RNG subsequence to be the global MPI Rank + 1
	CP.rng_subsequence = (unsigned long long) ();

	// Initialize the RNG offset to be zero
	CP.rng_offset = 0;

	// Call the RNG initialization function on the GPUs
	RNG_Init(CP.rng_seed, CP.rng_subsequence, CP.rng_offset, rng_parallel_state_t)
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
void Grid3D::Generate_Normal_Random_Field(Real *d_field)
{
	// Here, d_field has been pre-allocated on the device
	RNG_Normal_Field_GPU(d_field, H.n_cells, H.n_ghost, rng_parallel_state_t);
}

/*! \fn void Rescale_Field(Real *d_x, Real A)
 *  \brief Rescale a field by a constant multiplicative factor. */
void Grid3D::Rescale_Field(Real *d_x, Real A)
{
	// Here, d_x has been pre-allocated on the device
	// Rescale the field by a multiplicative factor.
	Rescale_Field_GPU(d_x, A, H.n_cells, H.n_ghost);
}
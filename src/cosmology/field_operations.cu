
#include "field_operations.h"


/*! \fn void void Rescale_Field_GPU(Real *d_x, Real A, )
 *  \brief Multiply one field a multiplicative constant */
inline __device__ void Rescale_Field_GPU(Real *d_x, Real A, )
{
	// Rescale a field by a multiplicative constant

	// determine the cell location
	int id;

	// get a global thread ID
	id = threadIdx.x + blockIdx.x * blockDim.x;

	// only real cells participate
	if (id > n_ghost - 1 && id < n_cells - n_ghost) {

		// rescale the field
		d_x[id] *= A;
	}
}


/*! \fn void Field_Elemetwise_Product_GPU(Real *d_x, Real *d_y, int H.n_cells, int H.n_ghost)
 *  \brief Multiply one field elementwise by another */
inline __device__ void Field_Elemetwise_Product_GPU(Real *d_x, Real *d_y, int H.n_cells, int H.n_ghost)
{
	// determine the cell location
	int id;

	// get a global thread ID
	id = threadIdx.x + blockIdx.x * blockDim.x;

	// only real cells participate
	if (id > n_ghost - 1 && id < n_cells - n_ghost) {

		// rescale x by y
		d_x[id] *= d_y[id];
	}
}


/*! \fn void FFT_Populate_Wavevectors_GPU(Real *d_kx, Real *d_ky, Real *d_kz, Real *d_kk, int n_cells, int n_ghost)
 *  \brief Multiply one field elementwise by another */
inline __device__ void FFT_Populate_Wavevectors_GPU(Real *d_kx, Real *d_ky, Real *d_kz, Real *d_kk, int n_cells, int n_ghost)
{
	// determine the cell location
	int id;

	// get a global thread ID
	id = threadIdx.x + blockIdx.x * blockDim.x;

	// only real cells participate
	if (id > n_ghost - 1 && id < n_cells - n_ghost) {

		d_kx[id] = 0;
		d_ky[id] = 0;
		d_kz[id] = 0;
	}
}



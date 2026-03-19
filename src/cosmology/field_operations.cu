
#include "field_operations.h"

inline __device__ void Rescale_Field_GPU(Real *d_x, Real A, )
{
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



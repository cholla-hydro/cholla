# pragma once
#include "../global/global.h"
inline __device__ void Rescale_Field_GPU(Real *d_x, Real A, int n_cells, int n_ghost);
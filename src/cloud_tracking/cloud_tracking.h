#ifdef CLOUD_TRACKING
  #ifndef CLOUD_TRACKING_CUDA_H
    #define CLOUD_TRACKING_CUDA_H

    #include "../global/global.h"
    #include "../utils/gpu.hpp"

void Cloud_Velocity_Reduction(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost,
                              int n_fields, Real density_cloud_init, Real density_wind_init, Real *mass_cloud,
                              Real *integrand_cloud);

__global__ void Cloud_Reduction_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                       int n_ghost, int n_fields, Real density_cloud_init, Real density_wind_init,
                                       Real *mass_cloud, Real *integrand_cloud);

void Update_Grid_Frame(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                       Real velocity_x_cloud_avg);

__global__ void Frame_Shift_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                                   Real velocity_x_cloud_avg);

  #endif  // CLOUD_TRACKING_H
#endif    // CLOUD_TRACKING
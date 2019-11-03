#if defined(PARTICLES) && defined(PARTICLES_GPU)

#ifndef PARTICLES_GPU_FUNCTIONS_CIC_H
#define PARTICLES_GPU_FUNCTIONS_CIC_H


void Get_Density_CIC_GPU_function(part_int_t n_local, Real particle_mass,  Real xMin, Real xMax, Real yMin, Real yMax, Real zMin, Real zMax, Real dx, Real dy, Real dz, int nx_local, int ny_local, int nz_local, int n_ghost_particles_grid, int n_cells, Real *density_h, Real *density_dev, Real *pos_x_dev, Real *pos_y_dev , Real *pos_z_dev);

#endif
#endif
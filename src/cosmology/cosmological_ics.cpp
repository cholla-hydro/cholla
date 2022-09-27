#ifdef COSMOLOGY


#include <random>
#include <stdio.h>
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "power_spectrum.h"

void Grid3D::Generate_Cosmological_Initial_Conditions( struct parameters *P  ){
  chprintf( "Generating cosmological initial conditions... \n" );
  
  Cosmo.ICs.Power_Spectrum.Load_Power_Spectum_From_File( P );
  
  
  int n_local = Cosmo.ICs.nx_local * Cosmo.ICs.ny_local * Cosmo.ICs.nz_local;
  
  Cosmo.ICs.random_fluctiations = (Real *) malloc(n_local*sizeof(Real));
  Cosmo.ICs.rescaled_random_fluctiations_dm  = (Real *) malloc(n_local*sizeof(Real));
  Cosmo.ICs.rescaled_random_fluctiations_gas = (Real *) malloc(n_local*sizeof(Real));
  
  long unsigned int random_seed;
  if ( P->cosmo_ics_random_seed == -1 ){
    random_seed = static_cast<long unsigned int>( time(0) + procID );
    chprintf( " Using time based random seed: %ld\n", random_seed);
  } else{
    random_seed = static_cast<long unsigned int>( P->cosmo_ics_random_seed + procID );
    chprintf( " Using user specified random seed: %ld \n", random_seed);
  }
  
  // This uses a system seed
  // std::random_device rd{};
  // std::mt19937 gen{rd()};
  
  std::mt19937 gen{ random_seed };
  std::normal_distribution<Real> d( 0, 1 );
  
  Real mean, sigma;
  mean = 0;
  sigma = 0;
  for ( int i=0; i<n_local; i++ ){
    Cosmo.ICs.random_fluctiations[i] = d(gen);
    mean += Cosmo.ICs.random_fluctiations[i];
    Cosmo.ICs.rescaled_random_fluctiations_dm[i]  = 0.0;
    Cosmo.ICs.rescaled_random_fluctiations_gas[i] = 0.0; 
  }
  mean /= n_local;
  mean = ReduceRealAvg( mean );
  for ( int i=0; i<n_local; i++ ){
    sigma += ( Cosmo.ICs.random_fluctiations[i] - mean ) * ( Cosmo.ICs.random_fluctiations[i] - mean );
  }
  sigma /= n_local;
  sigma = ReduceRealAvg( sigma );
  sigma = sqrt( sigma );
  chprintf(" Random gauissian fluctuations.  mean: %f  sigma: %f \n", mean, sigma );
  
  Real D, D_dot;
  D = Cosmo.Get_Linear_Growth_Factor( Cosmo.current_a );
  D_dot = Cosmo.Get_Linear_Growth_Factor_Deriv( Cosmo.current_a );
  chprintf( " Current_a: %f   D: %e  D_dot: %e\n", Cosmo.current_a, D, D_dot );
  
  chprintf( " Initializing 3D FFT.. \n" );
  Cosmo.ICs.FFT.Initialize( H.xdglobal, H.ydglobal, H.zdglobal, H.xblocal, H.yblocal, H.zblocal,
                            P->nx, P->ny, P->nz, H.nx_real, H.ny_real, H.nz_real, H.dx, H.dy, H.dz );
  
  
  
  chprintf( " Generating particles initial conditions... \n");
  // Call the FFT with filter to multuply by the sqrt of the DM Power Spectrum
  Real dx3 = H.dx * H.dy * H.dz;
  Cosmo.ICs.FFT.Filter_rescale_by_power_spectrum( Cosmo.ICs.random_fluctiations, Cosmo.ICs.rescaled_random_fluctiations_dm, false,
                         Cosmo.ICs.Power_Spectrum.host_size, Cosmo.ICs.Power_Spectrum.dev_k, Cosmo.ICs.Power_Spectrum.dev_pk_dm, dx3 );
  
  
  Real *displacements_x, *displacements_y, *displacements_z;
  Real *positions_x, *positions_y, *positions_z;
  Real *velocities_x, *velocities_y, *velocities_z;
  displacements_x = (Real *) malloc(n_local*sizeof(Real));
  displacements_y = (Real *) malloc(n_local*sizeof(Real));
  displacements_z = (Real *) malloc(n_local*sizeof(Real));
  positions_x = (Real *) malloc(n_local*sizeof(Real));
  positions_y = (Real *) malloc(n_local*sizeof(Real));
  positions_z = (Real *) malloc(n_local*sizeof(Real));
  velocities_x = (Real *) malloc(n_local*sizeof(Real));
  velocities_y = (Real *) malloc(n_local*sizeof(Real));
  velocities_z = (Real *) malloc(n_local*sizeof(Real));
  
  // Call the FFT with filter to multiply the DM fluctuations by ik/k^2/D  
  Cosmo.ICs.FFT.Filter_rescale_by_k_k2( Cosmo.ICs.rescaled_random_fluctiations_dm, displacements_z, false, 0, D );
  Cosmo.ICs.FFT.Filter_rescale_by_k_k2( Cosmo.ICs.rescaled_random_fluctiations_dm, displacements_y, false, 1, D );
  Cosmo.ICs.FFT.Filter_rescale_by_k_k2( Cosmo.ICs.rescaled_random_fluctiations_dm, displacements_x, false, 2, D );
  
  //Initialize min and max values for position and velocity to print initial Statistics
  Real dx_min, dy_min, dz_min;
  Real dx_max, dy_max, dz_max;
  Real vx_min, vy_min, vz_min;
  Real vx_max, vy_max, vz_max;
  dx_min = dy_min = dz_min = 1e64;
  dx_max = dy_max = dz_max = -1e64;
  vx_min = vy_min = vz_min = 1e64;
  vx_max = vy_max = vz_max = -1e64;
  
  
  int indx_i, indx_j, indx_k, indx;
  for ( indx_k=0; indx_k<Cosmo.ICs.nz_local; indx_k++ ){
    for ( indx_j=0; indx_j<Cosmo.ICs.ny_local; indx_j++ ){
      for ( indx_i=0; indx_i<Cosmo.ICs.nx_local; indx_i++ ){
        indx = indx_i + indx_j*Cosmo.ICs.nx_local + indx_k*Cosmo.ICs.nx_local*Cosmo.ICs.ny_local;
  
        // Particles can be displaced out of the local domain, a particles boundary transfer is excecuted at the end. 
        positions_x[indx] = H.xblocal + ( indx_i + 0.5 ) * H.dx + D * displacements_x[indx];
        positions_y[indx] = H.yblocal + ( indx_j + 0.5 ) * H.dy + D * displacements_y[indx];
        positions_z[indx] = H.zblocal + ( indx_k + 0.5 ) * H.dz + D * displacements_z[indx];
  
        velocities_x[indx] = Cosmo.current_a * D_dot * displacements_x[indx] * KPC / Cosmo.cosmo_h; // km /s
        velocities_y[indx] = Cosmo.current_a * D_dot * displacements_y[indx] * KPC / Cosmo.cosmo_h; // km /s
        velocities_z[indx] = Cosmo.current_a * D_dot * displacements_z[indx] * KPC / Cosmo.cosmo_h; // km /s
        
        dx_min = fmin( dx_min, D * displacements_x[indx] );
        dy_min = fmin( dy_min, D * displacements_y[indx] );
        dz_min = fmin( dz_min, D * displacements_z[indx] );  
        dx_max = fmax( dx_max, D * displacements_x[indx] );
        dy_max = fmax( dy_max, D * displacements_y[indx] );
        dz_max = fmax( dz_max, D * displacements_z[indx] );
        
        vx_min = fmin( vx_min, velocities_x[indx] );
        vy_min = fmin( vy_min, velocities_y[indx] );
        vz_min = fmin( vz_min, velocities_z[indx] );  
        vx_max = fmax( vx_max, velocities_x[indx] );
        vy_max = fmax( vy_max, velocities_y[indx] );
        vz_max = fmax( vz_max, velocities_z[indx] );  
  
      }    
    }
  }
  
  dx_min = ReduceRealMin( dx_min );
  dy_min = ReduceRealMin( dy_min );
  dz_min = ReduceRealMin( dz_min );
  dx_max = ReduceRealMax( dx_max );
  dy_max = ReduceRealMax( dy_max );
  dz_max = ReduceRealMax( dz_max );
  
  vx_min = ReduceRealMin( vx_min );
  vy_min = ReduceRealMin( vy_min );
  vz_min = ReduceRealMin( vz_min );
  vx_max = ReduceRealMax( vx_max );
  vy_max = ReduceRealMax( vy_max );
  vz_max = ReduceRealMax( vz_max );
    
  Particles.n_local = n_local;
  Particles.n_total_initial = ReducePartIntSum( n_local );
  chprintf( " N total particles: %ld \n", Particles.n_total_initial );
  
  Real dens_dm_mean;   
  #ifdef ONLY_PARTICLES
  dens_dm_mean = 3*Cosmo.H0*Cosmo.H0 / ( 8*M_PI*Cosmo.cosmo_G ) * Cosmo.Omega_M /Cosmo.cosmo_h/Cosmo.cosmo_h *;
  #else
  dens_dm_mean = 3*Cosmo.H0*Cosmo.H0 / ( 8*M_PI*Cosmo.cosmo_G ) * ( Cosmo.Omega_M - Cosmo.Omega_b ) /Cosmo.cosmo_h/Cosmo.cosmo_h;
  #endif
  Particles.particle_mass = dens_dm_mean * H.xdglobal * H.ydglobal * H.zdglobal / Particles.n_total_initial; 
  chprintf( " Particle mass: %e Msun/h \n", Particles.particle_mass );
  
  //Print initial Statistics
  #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY)
  chprintf( "  Displacements X   Min: %e   Max: %e   [ kpc/h ]\n", dx_min, dx_max );
  chprintf( "  Displacements Y   Min: %e   Max: %e   [ kpc/h ]\n", dy_min, dy_max );
  chprintf( "  Displacements Z   Min: %e   Max: %e   [ kpc/h ]\n", dz_min, dz_max );
  chprintf( "  Velocities X      Min: %e   Max: %e   [ km/s ]\n", vx_min, vx_max );
  chprintf( "  Velocities Y      Min: %e   Max: %e   [ km/s ]\n", vy_min, vy_max );
  chprintf( "  Velocities Z      Min: %e   Max: %e   [ km/s ]\n", vz_min, vz_max );
  #endif//PRINT_INITIAL_STATS
  
  #ifdef PARTICLES_CPU
  for ( indx=0; indx<n_local; indx++ ){
    //Add the particle data to the particles vectors
    Particles.pos_x.push_back( positions_x[indx] );
    Particles.pos_y.push_back( positions_y[indx] );
    Particles.pos_z.push_back( positions_z[indx] );
    Particles.vel_x.push_back( velocities_x[indx] );
    Particles.vel_y.push_back( velocities_y[indx] );
    Particles.vel_z.push_back( velocities_z[indx] );
    Particles.grav_x.push_back( 0.0 );
    Particles.grav_y.push_back( 0.0 );
    Particles.grav_z.push_back( 0.0 );
  #endif //PARTICLES_CPU
  
  #ifdef PARTICLES_GPU
  Particles.particles_array_size = Particles.Compute_Particles_GPU_Array_Size( Particles.n_local );
  chprintf( " Allocating GPU buffer size: %ld * %f = %ld \n", n_local, Particles.G.gpu_allocation_factor, Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.pos_x_dev,  Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.pos_y_dev,  Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.pos_z_dev,  Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.vel_x_dev,  Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.vel_y_dev,  Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.vel_z_dev,  Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.grav_x_dev, Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.grav_y_dev, Particles.particles_array_size);
  Particles.Allocate_Particles_GPU_Array_Real( &Particles.grav_z_dev, Particles.particles_array_size);
  Particles.Allocate_Memory_GPU_MPI();
  chprintf( " Allocated GPU memory for particle data\n");
  
  //Copyt the particle data to GPU memory 
  Particles.Copy_Particles_Array_Real_Host_to_Device( positions_x, Particles.pos_x_dev,  n_local);
  Particles.Copy_Particles_Array_Real_Host_to_Device( positions_y, Particles.pos_y_dev,  n_local);
  Particles.Copy_Particles_Array_Real_Host_to_Device( positions_z, Particles.pos_z_dev,  n_local);
  Particles.Copy_Particles_Array_Real_Host_to_Device( velocities_x, Particles.vel_x_dev, n_local);
  Particles.Copy_Particles_Array_Real_Host_to_Device( velocities_y, Particles.vel_y_dev, n_local);
  Particles.Copy_Particles_Array_Real_Host_to_Device( velocities_z, Particles.vel_z_dev, n_local);
  #endif //PARTICLES_GPU
  
  //Transfer the particles that are outside the local domain
  Transfer_Particles_Boundaries(*P);
  
  chprintf( " Particles intial conditions generated successfully. \n" );
  
  #if !defined(ONLY_PARTICLES)
  
  chprintf( " Generating gas initial conditions... \n");
  
  Real *density, *momentum_x, *momentum_y, *momentum_z, *Energy, *GasEnergy;
  density    = (Real *) malloc(n_local*sizeof(Real));
  momentum_x = (Real *) malloc(n_local*sizeof(Real));
  momentum_y = (Real *) malloc(n_local*sizeof(Real));
  momentum_z = (Real *) malloc(n_local*sizeof(Real));
  Energy     = (Real *) malloc(n_local*sizeof(Real));
  GasEnergy  = (Real *) malloc(n_local*sizeof(Real));
  
  Real initial_temp = P->Init_temperature;
  chprintf( "  Initial temperature: %f K\n", initial_temp );
  
  // Call the FFT with filter to multuply by the sqrt of the Gas Power Spectrum
  Cosmo.ICs.FFT.Filter_rescale_by_power_spectrum( Cosmo.ICs.random_fluctiations, Cosmo.ICs.rescaled_random_fluctiations_gas, false,
                         Cosmo.ICs.Power_Spectrum.host_size, Cosmo.ICs.Power_Spectrum.dev_k, Cosmo.ICs.Power_Spectrum.dev_pk_gas, dx3 );
  
  // Call the FFT with filter to multiply the GAS fluctuations by ik/k^2/D  
  Cosmo.ICs.FFT.Filter_rescale_by_k_k2( Cosmo.ICs.rescaled_random_fluctiations_gas, displacements_z, false, 0, D );
  Cosmo.ICs.FFT.Filter_rescale_by_k_k2( Cosmo.ICs.rescaled_random_fluctiations_gas, displacements_y, false, 1, D );
  Cosmo.ICs.FFT.Filter_rescale_by_k_k2( Cosmo.ICs.rescaled_random_fluctiations_gas, displacements_x, false, 2, D );
  
  
  Real dens_max, dens_min, dens_mean;
  Real momentum_x_max, momentum_x_min, momentum_x_mean;
  Real momentum_y_max, momentum_y_min, momentum_y_mean;
  Real momentum_z_max, momentum_z_min, momentum_z_mean;
  Real Energy_max, Energy_min, Energy_mean;
  Real GasEnergy_max, GasEnergy_min, GasEnergy_mean;
  
  dens_min  = momentum_x_min  = momentum_y_min  = momentum_z_min  = Energy_min  = GasEnergy_min  = 1e64;
  dens_max  = momentum_x_max  = momentum_y_max  = momentum_z_max  = Energy_max  = GasEnergy_max  = -1e64;
  dens_mean = momentum_x_mean = momentum_y_mean = momentum_z_mean = Energy_mean = GasEnergy_mean = 0.0;
  
  Real mmw = 1; 
  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  Real HI_frac = INITIAL_FRACTION_HI;
  Real HII_frac = INITIAL_FRACTION_HII;
  Real HeI_frac = INITIAL_FRACTION_HEI;
  Real HeII_frac = INITIAL_FRACTION_HEII;
  Real HeIII_frac = INITIAL_FRACTION_HEIII;
  Real e_frac = INITIAL_FRACTION_ELECTRON;
  Real metal_frac = INITIAL_FRACTION_METAL;
  mmw = ( HI_frac + HII_frac + HeI_frac + HeII_frac + HeIII_frac ) / ( HI_frac + HII_frac + ( HeI_frac + HeII_frac + HeIII_frac )/4 + e_frac );
  #endif
  
  
  for ( indx_k=0; indx_k<Cosmo.ICs.nz_local; indx_k++ ){
    for ( indx_j=0; indx_j<Cosmo.ICs.ny_local; indx_j++ ){
      for ( indx_i=0; indx_i<Cosmo.ICs.nx_local; indx_i++ ){
        indx = indx_i + indx_j*Cosmo.ICs.nx_local + indx_k*Cosmo.ICs.nx_local*Cosmo.ICs.ny_local;
  
        density[indx] = Cosmo.rho_mean_baryon * ( 1 + Cosmo.ICs.rescaled_random_fluctiations_gas[indx] );
        momentum_x[indx] = density[indx] * Cosmo.current_a * D_dot * displacements_x[indx] * KPC / Cosmo.cosmo_h;
        momentum_y[indx] = density[indx] * Cosmo.current_a * D_dot * displacements_y[indx] * KPC / Cosmo.cosmo_h;
        momentum_z[indx] = density[indx] * Cosmo.current_a * D_dot * displacements_z[indx] * KPC / Cosmo.cosmo_h;
        GasEnergy[indx] = density[indx] * initial_temp / ( gama - 1 ) / MP * KB / 1e10 / mmw;
        Energy[indx] = GasEnergy[indx] + 0.5 * ( momentum_x[indx]*momentum_x[indx] + momentum_y[indx]*momentum_y[indx] + momentum_z[indx]*momentum_z[indx]  ) / density[indx];
  
        dens_max = fmax( dens_max, density[indx] );
        dens_min = fmin( dens_min, density[indx] );
        dens_mean += density[indx];
  
        momentum_x_max = fmax( momentum_x_max, momentum_x[indx] );
        momentum_x_min = fmin( momentum_x_min, momentum_x[indx] );
        momentum_x_mean += momentum_x[indx];
  
        momentum_y_max = fmax( momentum_y_max, momentum_y[indx] );
        momentum_y_min = fmin( momentum_y_min, momentum_y[indx] );
        momentum_y_mean += momentum_y[indx];
  
        momentum_z_max = fmax( momentum_z_max, momentum_z[indx] );
        momentum_z_min = fmin( momentum_z_min, momentum_z[indx] );
        momentum_z_mean += momentum_z[indx];
  
        GasEnergy_max = fmax( GasEnergy_max, GasEnergy[indx] );
        GasEnergy_min = fmin( GasEnergy_min, GasEnergy[indx] );
        GasEnergy_mean += GasEnergy[indx];
  
        Energy_max = fmax( Energy_max, Energy[indx] );
        Energy_min = fmin( Energy_min, Energy[indx] );
        Energy_mean += Energy[indx];
      }
    }
  }
  
  dens_mean /= n_local;
  momentum_x_mean /= n_local;
  momentum_y_mean /= n_local;
  momentum_z_mean /= n_local;
  Energy_mean /= n_local;
  GasEnergy_mean /= n_local;
  dens_min  = ReduceRealMin( dens_min );
  dens_max  = ReduceRealMax( dens_max );
  dens_mean = ReduceRealAvg( dens_mean );
  chprintf( "  Density  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3] \n", dens_mean, dens_min, dens_max );
  momentum_x_min  = ReduceRealMin( momentum_x_min );
  momentum_x_max  = ReduceRealMax( momentum_x_max );
  momentum_x_mean = ReduceRealAvg( momentum_x_mean );
  chprintf( "  Momentum X  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", momentum_x_mean, momentum_x_min, momentum_x_max );
  momentum_y_min  = ReduceRealMin( momentum_y_min );
  momentum_y_max  = ReduceRealMax( momentum_y_max );
  momentum_y_mean = ReduceRealAvg( momentum_y_mean );
  chprintf( "  Momentum Y  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", momentum_y_mean, momentum_y_min, momentum_y_max );
  momentum_z_min  = ReduceRealMin( momentum_z_min );
  momentum_z_max  = ReduceRealMax( momentum_z_max );
  momentum_z_mean = ReduceRealAvg( momentum_z_mean );
  chprintf( "  Momentum Z  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", momentum_z_mean, momentum_z_min, momentum_z_max );
  Energy_min  = ReduceRealMin( Energy_min );
  Energy_max  = ReduceRealMax( Energy_max );
  Energy_mean = ReduceRealAvg( Energy_mean );
  chprintf( "  Energy  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km^2 s^-2 ]\n", Energy_mean, Energy_min, Energy_max );
  GasEnergy_min  = ReduceRealMin( GasEnergy_min );
  GasEnergy_max  = ReduceRealMax( GasEnergy_max );
  GasEnergy_mean = ReduceRealAvg( GasEnergy_mean );
  chprintf( "  GasEnergy  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km^2 s^-2 ]\n", GasEnergy_mean, GasEnergy_min, GasEnergy_max );
  
  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  chprintf( "  Initial HI Fraction:    %e \n", HI_frac);
  chprintf( "  Initial HII Fraction:   %e \n", HII_frac);
  chprintf( "  Initial HeI Fraction:   %e \n", HeI_frac);
  chprintf( "  Initial HeII Fraction:  %e \n", HeII_frac);
  chprintf( "  Initial HeIII Fraction: %e \n", HeIII_frac);
  chprintf( "  Initial elect Fraction: %e \n", e_frac);
  chprintf( "  Initial mmw: %f \n", mmw);
  #endif
  #ifdef GRACKLE_METALS
  chprintf( "  Initial metal Fraction: %e \n", metal_frac);
  #endif  //GRACKEL_METALS
  
  
  int nx_grid, ny_grid, nz_grid, indx_grid;
  nx_grid = Cosmo.ICs.nx_local + 2*H.n_ghost;
  ny_grid = Cosmo.ICs.ny_local + 2*H.n_ghost;
  nz_grid = Cosmo.ICs.nz_local + 2*H.n_ghost;
    
  for ( indx_k=0; indx_k<Cosmo.ICs.nz_local; indx_k++ ){
    for ( indx_j=0; indx_j<Cosmo.ICs.ny_local; indx_j++ ){
      for ( indx_i=0; indx_i<Cosmo.ICs.nx_local; indx_i++ ){
        indx = indx_i + indx_j*Cosmo.ICs.nx_local + indx_k*Cosmo.ICs.nx_local*Cosmo.ICs.ny_local;
        indx_grid = (indx_i+H.n_ghost) + (indx_j+H.n_ghost)*nx_grid + (indx_k+H.n_ghost)*nx_grid*ny_grid;
        C.density[indx_grid] = density[indx];
        C.momentum_x[indx_grid] = momentum_x[indx];
        C.momentum_y[indx_grid] = momentum_y[indx];
        C.momentum_z[indx_grid] = momentum_z[indx];
        C.Energy[indx_grid] = Energy[indx];
        #ifdef DE
        C.GasEnergy[indx_grid] = GasEnergy[indx]; 
        #endif
  
        #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
        C.scalar[0*H.n_cells + indx_grid] = HI_frac    * C.density[indx_grid];
        C.scalar[1*H.n_cells + indx_grid] = HII_frac   * C.density[indx_grid];
        C.scalar[2*H.n_cells + indx_grid] = HeI_frac   * C.density[indx_grid];
        C.scalar[3*H.n_cells + indx_grid] = HeII_frac  * C.density[indx_grid];
        C.scalar[4*H.n_cells + indx_grid] = HeIII_frac * C.density[indx_grid];
        C.scalar[5*H.n_cells + indx_grid] = e_frac     * C.density[indx_grid];
        #endif
        #ifdef GRACKLE_METALS
        C.scalar[6*H.n_cells + indx_grid] = metal_frac * C.density[indx_grid];
        #endif
      }
    }
  }
  
  // Copy the conserved fields to the device
  CudaSafeCall( cudaMemcpy(C.device, C.density, H.n_fields*H.n_cells*sizeof(Real),  cudaMemcpyHostToDevice) );
  
  #endif// NOT ONLY_PARTICLES
  
  // Change to comoving Cosmological System
  Change_Cosmological_Frame_Sytem( true );
  
  // Reset everything
  free( displacements_x );
  free( displacements_y );
  free( displacements_z );
  free( positions_x );
  free( positions_y );
  free( positions_z );
  free( velocities_x );
  free( velocities_y );
  free( velocities_z );
  free( density );
  free( momentum_x );
  free( momentum_y );
  free( momentum_z );
  free( Energy );
  free( GasEnergy ); 
  
  Cosmo.ICs.FFT.Reset();
  chprintf( "Cosmological initial conditions generated successfully. \n\n" );

}

#endif // COSMOLOGY
#ifdef COSMOLOGY


#include <random>
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
  
  // std::default_random_engine generator;
  // std::normal_distribution<double> distribution(5.0,2.0);
  
  std::random_device rd{};
  std::mt19937 gen{rd()};
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
  
  
  chprintf( " Initializing 3D FFT.. \n" );
  Cosmo.ICs.FFT.Initialize( H.xdglobal, H.ydglobal, H.zdglobal, H.xblocal, H.yblocal, H.zblocal,
                            P->nx, P->ny, P->nz, H.nx_real, H.ny_real, H.nz_real, H.dx, H.dy, H.dz );
  
  // Allocate device memory for FFT
  Real *dev_fft_input, *dev_fft_output;
  CudaSafeCall( cudaMalloc((void**)&dev_fft_input, n_local*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_fft_output, n_local*sizeof(Real)) );
  
  CudaSafeCall( cudaMemcpy( dev_fft_input, Cosmo.ICs.random_fluctiations, n_local*sizeof(Real), cudaMemcpyHostToDevice ) ); 
  
  // Call the FFT with filter
  Cosmo.ICs.FFT.Filter_inv_k2(  dev_fft_input, dev_fft_output, true ); 
  
  CudaSafeCall( cudaMemcpy( Cosmo.ICs.rescaled_random_fluctiations_dm, dev_fft_output, n_local*sizeof(Real), cudaMemcpyDeviceToHost ) ); 
  
  Real diff;
  for ( int i=0; i<n_local; i++ ){
    diff = Cosmo.ICs.rescaled_random_fluctiations_dm[i] / Cosmo.ICs.random_fluctiations[i] - 1 ;
    printf("Diff %f\n", diff );
  }


  // Free the device memory for FFT
  CudaSafeCall( cudaFreeHost( dev_fft_input) );
  CudaSafeCall( cudaFreeHost( dev_fft_output) );
  chprintf( "Cosmological initial conditions generated successfully. \n" );
  
}

#endif // COSMOLOGY
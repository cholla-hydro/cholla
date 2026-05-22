#ifdef COSMOLOGY
  #include <fstream>
	#include <cstdio>
	#include <hdf5.h>

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../utils/cuda_utilities.h"
  #include "../utils/gpu.hpp"
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
  int n_cells = nx_local*ny_local*nz_local;
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
	chprintf("Generating potentials for cosmological ICs.\n");

	// set the number of fields
	CP.n_fields = 2; //initial potential and overdensity field	
#ifndef ONLY_PARTICLES
	CP.n_fields += 1;	// add a baryon overdensity field
#endif 

	// initialize the RNG properties
	Initialize_Cosmo_Potential_RNG(P);

  // load the P(k)
  Load_Cosmo_Power_Spectrum(P);

	// Initialize the FFT as well
	chprintf("Initializing the FFT system\n");
  chprintf("xdglobal %f %f %f\n",H.xdglobal, H.ydglobal, H.zdglobal);
  chprintf("xblocal %f %f %f\n",H.xblocal, H.yblocal, H.zblocal);
  chprintf("nx_local %d %d %d\n",nx_local, ny_local, nz_local);
  chprintf("nx_global %d ny_global %d nz_global %d\n",nx_global,ny_global,nz_global);
	fft.Initialize( H.xdglobal, H.ydglobal, H.zdglobal, H.xblocal, H.yblocal, H.zblocal,
		              nx_global, ny_global, nz_global, nx_local, ny_local, nz_local, H.dx, H.dy, H.dz );

	// Allocate the memory needed for the potentials
	Allocate_Cosmo_Potential_Memory();

  // We have allocated the potential arrays, and are ready to proceed

	// step 1) sample xi(m) by generating independent
	//         zero-mean normal deviates with variance N**d at 
	//         each spatial point
	Generate_Normal_Random_Field(CP.d_delta_c,rng_states);


	Rescale_Field(CP.d_delta_c, nx_global*ny_global*nz_global);


	// copy memory
	cudaMemcpy(CP.delta_c, CP.d_delta_c, n_cells * sizeof(Real), cudaMemcpyDeviceToHost);


  // reduce the grid values
	Real delta_sum = 0;
	Real delta_ave = 0;
  for (k = 0; k < H.nz - 2*H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2*H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2*H.n_ghost; i++) {

        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        delta_sum += CP.delta_c[id]; // perform a local reduction
      }
    }
  }

  // get the total of the grid to compute the mean
  MPI_Allreduce(MPI_IN_PLACE, &delta_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // find the average
  delta_ave = delta_sum/(nx_global*ny_global*nz_global);
  chprintf("Average of random field %e\n",delta_ave);


  // reduce the grid values
  for (k = 0; k < H.nz - 2*H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2*H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2*H.n_ghost; i++) {

        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        CP.delta_c[id] -= delta_ave; // remove global average
      }
    }
  }


	// copy mean zero phi back to GPU
	cudaMemcpy(CP.delta_c, CP.d_delta_c, n_cells * sizeof(Real), cudaMemcpyHostToDevice);

  chprintf("Rescaling field...\n");

	// step 2) Take the fourier transform
	//         xi(k) = N**-d \sum_m exp( -(2 pi i / M) * kappa \dot m) * xi(m)
	fft.Filter_rescale(CP.d_delta_c,1./(nx_global*ny_global*nz_global),CP.d_delta_c,true);


#ifndef ONLY_PARTICLES
  // step 2.5) Copy random field to baryonic field
	cudaMemcpy(CP.d_delta_b, CP.d_delta_c, n_cells * sizeof(Real), cudaMemcpyDeviceToDevice);
#endif 

	// step 3) Multiply xi(k) by the transfer function 
	//         T(k) \equiv [(2 \pi / L)**3 P(k)]^{1/2}
	//         note T(k) is computed at z=0
  //         also note  the [(2 \pi / L)**3 ]^{1/2} factor is handled
  //         when the power spectrum is loaded, so this just applies sqrt(P(k))
  chprintf("Applying power spectrum...\n");

/*
  fft.Filter_rescale_by_power_spectrum(CP.d_delta_c,CP.d_delta_c,true,CP.n_pk,CP.d_k_array,CP.d_pk_dm_array);
#ifndef ONLY_PARTICLES
  chprintf("Applying baryonic power spectrum...\n");

  fft.Filter_rescale_by_power_spectrum(CP.d_delta_b,CP.d_delta_b,true,CP.n_pk,CP.d_k_array,CP.d_pk_gas_array);
#endif 
*/

	// copy memory back to host
	cudaMemcpy(CP.delta_c, CP.d_delta_c, n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
#ifndef ONLY_PARTICLES
	cudaMemcpy(CP.delta_b, CP.d_delta_b, n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
#endif 

  // free the P(k)
  Free_Cosmo_Power_Spectrum();

  // At this stage, the cosmological potential(s) phi_ini
  // has/have been computed. These can be used to set
  // the remaining initial conditions in the ICs generation
  // locations of the code

	// step 2.5) create k vectors
	//Populate_Wavevectors(d_kx, d_ky, d_kz, d_kk);

	// step 3.1) divide by k^2 to take inverse laplacian
	//FFT_Field_Inverse_Laplacian(d_xi_k, d_kk);

	// step 3.5) rescale by growth factor(a)/a
	//Rescale_FFT_Field(d_xi_k, Daa);

	// step 4) Reset the FFT system, free memory
	//fft.Reset();


  // exit
  chprintf("Saving potential...\n");

  Save_Cosmo_Potential(P);

  chprintf("Exiting...\n");

	chexit(0);
}

/*! \fn void Save_Cosmo_Potential(struct Parameters *P)
 *  \brief Write out the cosmological potential field to hdf5 files*/
void Grid3D::Save_Cosmo_Potential(struct Parameters const *P)
{
  char fname[200];
  hid_t f_id, d_id, a0_id, a1_id, a2_id, a3_id;
#ifndef ONLY_PARTICLES
  hid_t db_id;
#endif
  hid_t fs_id, fsa0_id, fsa1_id, fsa2_id, fsa3_id, ms_id;
  hsize_t dimsf[3];
  hsize_t dimsa = 3;
  int attr_global[3];
  int attr_start[3];
  int attr_size[3];
  int attr_ghost[3];
  herr_t status;
  Real *phi_out; // output cosmological potential

  //Create a file name for each hdf5 output
  sprintf(fname,"%s0/phi_ini.h5.%d",P->outdir,procID);

  // create a file
  f_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(f_id < 0) {
    printf("HDF5 file create error on process %d\n",procID);
  }

  // information on HDF5 subvolume properties
  // within the computational volume
  attr_start[0] = nx_local_start;
  attr_start[1] = ny_local_start; 
  attr_start[2] = nz_local_start; 
  attr_size[0] = nx_local; 
  attr_size[1] = ny_local; 
  attr_size[2] = nz_local; 
  attr_ghost[0] = H.n_ghost;
  attr_ghost[1] = H.n_ghost; 
  attr_ghost[2] = H.n_ghost; 
  attr_global[0] = nx_global;
  attr_global[1] = ny_global; 
  attr_global[2] = nz_global; 


  // dimensions of output initial potential
  dimsf[0] = nx_local;
  dimsf[1] = ny_local;
  dimsf[2] = nz_local;


  // create a dataset
  fs_id   = H5Screate_simple(3, dimsf, NULL);
  d_id = H5Dcreate2(f_id, "delta_c", H5T_NATIVE_DOUBLE, fs_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // attach attributes
  fsa0_id = H5Screate_simple(1, &dimsa, NULL);
  fsa1_id = H5Screate_simple(1, &dimsa, NULL);
  fsa2_id = H5Screate_simple(1, &dimsa, NULL);
  fsa3_id = H5Screate_simple(1, &dimsa, NULL);

  a0_id = H5Acreate2(d_id, "global", H5T_NATIVE_INT, fsa0_id, H5P_DEFAULT, H5P_DEFAULT);
  a1_id = H5Acreate2(d_id, "start",  H5T_NATIVE_INT, fsa1_id, H5P_DEFAULT, H5P_DEFAULT);
  a2_id = H5Acreate2(d_id, "size",   H5T_NATIVE_INT, fsa2_id, H5P_DEFAULT, H5P_DEFAULT);
  a3_id = H5Acreate2(d_id, "ghost",  H5T_NATIVE_INT, fsa3_id, H5P_DEFAULT, H5P_DEFAULT);
                
                           
  status = H5Awrite(a0_id, H5T_NATIVE_INT, attr_global);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
                           
  status = H5Awrite(a1_id, H5T_NATIVE_INT, attr_start);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
  status = H5Awrite(a2_id, H5T_NATIVE_INT, attr_size);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
  status = H5Awrite(a3_id, H5T_NATIVE_INT, attr_ghost);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
  GPU_Error_Check(cudaHostAlloc((void **)&phi_out, nx_local*ny_local*nz_local*sizeof(Real), cudaHostAllocDefault));

  // store the output potential in row major order
  int i, j, k, id;
  int ii,jj,kk, idx;
  for (k = 0; k < H.nz - 2*H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2*H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2*H.n_ghost; i++) {

        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        ii = i;
        jj = j;
        kk = k;
        idx = ii*(nz_local*ny_local) + jj*nz_local + kk; // row major

        phi_out[idx] = CP.delta_c[id]; // map to output CDM overdensity field
      }
    }
  }


  status = H5Dwrite(d_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, phi_out);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }

  H5Dclose(d_id);


#ifndef ONLY_PARTICLES

  db_id = H5Dcreate2(f_id, "delta_b", H5T_NATIVE_DOUBLE, fs_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // store the output potential in row major order
  for (k = 0; k < H.nz - 2*H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2*H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2*H.n_ghost; i++) {

        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        ii = i;
        jj = j;
        kk = k;
        idx = ii*(nz_local*ny_local) + jj*nz_local + kk; // row major

        phi_out[idx] = CP.delta_b[id]; // map to output baryon overdensity field
      }
    }
  }


  status = H5Dwrite(db_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, phi_out);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }

  H5Dclose(db_id);

#endif //ONLY_PARTICLES

  H5Aclose(a0_id);
  H5Aclose(a1_id);
  H5Aclose(a2_id);
  H5Aclose(a3_id);
  H5Sclose(fs_id);
  H5Sclose(fsa3_id);
  H5Sclose(fsa2_id);
  H5Sclose(fsa1_id);
  H5Sclose(fsa0_id);
  H5Fclose(f_id);
}

/*! \fn void Initialize_Cosmo_Potential_RNG(void)
 *  \brief Initialize the RNG for cosmological ICs potentials */
void Grid3D::Initialize_Cosmo_Potential_RNG(struct Parameters *P)
{
	// Initialize the parameters for the Philox RNG
  int n_cells = nx_local*ny_local*nz_local;

	// Record the RNG seed from the parameter file
	CP.rng_seed = P->seed;

	// Set the RNG subsequence to be the global MPI Rank + 1
	CP.rng_subsequence = (unsigned long long) (procID);

	// Initialize the RNG offset to be the process ID
	CP.rng_offset = procID;

	// Call the RNG initialization function on the GPUs
	GPU_Error_Check(cudaMalloc((void **)&rng_states, n_cells * sizeof(rng_parallel_state_t)));

  // initialze
  cuda_utilities::AutomaticLaunchParams static const launchParams(RNG_Init_GPU, n_cells);
  hipLaunchKernelGGL(RNG_Init_GPU, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                     nx_local,ny_local,nz_local,0,CP.rng_seed, CP.rng_subsequence, CP.rng_offset, rng_states);
  GPU_Error_Check();
                     
	chprintf("Initialized Cosmological ICs RNG states.");
}



/*! \fn void Free_Cosmo_Power_Spectrum()
 *  \brief Free memory for cosmological power spectrum*/
void Grid3D::Free_Cosmo_Power_Spectrum()
{
  // free host P(k) info
  free(CP.k_array);
  free(CP.pk_tot_array);
  free(CP.pk_dm_array);
  free(CP.pk_gas_array);

  // free device P(k) info
  GPU_Error_Check(cudaFree(CP.d_n_pk));
  GPU_Error_Check(cudaFree(CP.d_k_array));
  GPU_Error_Check(cudaFree(CP.d_pk_tot_array));
  GPU_Error_Check(cudaFree(CP.d_pk_dm_array));
  GPU_Error_Check(cudaFree(CP.d_pk_gas_array));
}

/*! \fn void Load_Cosmo_Power_Spectrum(struct Parameters *P)
 *  \brief Allocate memory and load cosmological power spectrum*/
void Grid3D::Load_Cosmo_Power_Spectrum(struct Parameters *P)
{
  
  char pk_filename[MAXLEN];
  strcpy(pk_filename, P->cosmo_ics_pk_file);
  chprintf( " Loading Power Spectrum File: %s \n", pk_filename );
  
  std::fstream in_file(pk_filename);
  std::string line;
  std::vector<std::vector<float>> v;
  int i = 0;
  int j = 0;
  if (in_file.is_open()){
    while (std::getline(in_file, line))
    {
      if ( line.find("#") == 0 ) continue;
      
      float value;
      std::stringstream ss(line);
      if (line.length() == 0) continue;
      v.push_back(std::vector<float>());
       
      while (ss >> value){
        v[i].push_back(value);
        if(i==0)
          j++;
      }
      i += 1;    
    }
    
    in_file.close();
  
  } else{
  
    chprintf(" Error: Unable to open the input power spectrum file: %s\n", pk_filename);
    exit(1);
  
  }
  
  int n_lines = i;
  chprintf( " Loaded %d lines in file. \n", n_lines  );
  
  // Allocate cpu and device memory for power spectrum
  CP.n_pk          = n_lines;
  CP.k_array       = (Real *)malloc( CP.n_pk*sizeof(Real) );
  CP.pk_tot_array  = (Real *)malloc( CP.n_pk*sizeof(Real) );
  CP.pk_dm_array   = (Real *)malloc( CP.n_pk*sizeof(Real) );
  CP.pk_gas_array  = (Real *)malloc( CP.n_pk*sizeof(Real) );
  
  chprintf( "Lbox = %f %f %f  n_grid = %d %d %d\n", P->xlen, P->ylen, P->zlen, P->nx, P->ny, P->nz );
  
  Real dx = P->xlen / P->nx; 
  Real pk_factor = (2.0*M_PI/(1.0e-3*P->xlen))*(2.0*M_PI/(1.0e-3*P->ylen))*(2.0*M_PI/(1.0e-3*P->zlen));
  // note the pk_factor is supposed to remove the volume element from 
  // the normalization of P(k), and needs to be in the units of the original P(k)
  // We are assuming P(k) has units of (Mpc/h)^3 and xlen, ylen, zlen are in kpc/h
  
  for (i=0; i<n_lines; i++ ){
    CP.k_array[i]      = v[i][0] * 1e-3;       //Convert from 1/(Mpc/h) to  1/(kpc/h)
    CP.pk_tot_array[i] = v[i][1] * pk_factor;  // moving P(k) rescaling here
    CP.pk_dm_array[i]  = v[i][2] * pk_factor;  // moving P(k) rescaling here
    if(j==4)
    {
      CP.pk_gas_array[i] = v[i][3] * pk_factor;  // moving P(k) rescaling here
    }else{
      CP.pk_gas_array[i] = v[i][2] * pk_factor;  // moving P(k) rescaling here
    }
  }

  // Allocate device P(k) arrays  
  GPU_Error_Check(cudaMalloc((void**)&CP.d_n_pk,         sizeof(int)) );
  GPU_Error_Check(cudaMalloc((void**)&CP.d_k_array,      CP.n_pk*sizeof(Real)) );
  GPU_Error_Check(cudaMalloc((void**)&CP.d_pk_tot_array, CP.n_pk*sizeof(Real)) );
  GPU_Error_Check(cudaMalloc((void**)&CP.d_pk_dm_array,  CP.n_pk*sizeof(Real)) );
  GPU_Error_Check(cudaMalloc((void**)&CP.d_pk_gas_array, CP.n_pk*sizeof(Real)) );

  // Copy host P(k) to device P(k)
  GPU_Error_Check(cudaMemcpy(CP.d_n_pk,        &CP.n_pk,                 sizeof(int),  cudaMemcpyHostToDevice) );
  GPU_Error_Check(cudaMemcpy(CP.d_k_array,      CP.k_array,      CP.n_pk*sizeof(Real), cudaMemcpyHostToDevice) );
  GPU_Error_Check(cudaMemcpy(CP.d_pk_tot_array, CP.pk_tot_array, CP.n_pk*sizeof(Real), cudaMemcpyHostToDevice) );
  GPU_Error_Check(cudaMemcpy(CP.d_pk_dm_array,  CP.pk_dm_array,  CP.n_pk*sizeof(Real), cudaMemcpyHostToDevice) );
  GPU_Error_Check(cudaMemcpy(CP.d_pk_gas_array, CP.pk_gas_array, CP.n_pk*sizeof(Real), cudaMemcpyHostToDevice) );
}


/*! \fn void Allocate_Cosmo_Potential_Memory(void)
 *  \brief Allocate the memory allocated for cosmological ICs potentials */
void Grid3D::Allocate_Cosmo_Potential_Memory()
{
  // allocate memory for the phi arrays
  // allocate all the memory to phi_1, to insure contiguous memory
  int n_cells = nx_local*ny_local*nz_local;
  int offset = n_cells;

  GPU_Error_Check(cudaHostAlloc((void **)&CP.host, CP.n_fields * n_cells * sizeof(Real), cudaHostAllocDefault));

  chprintf("Host memory allocated for cosmological ICs initial potential.\n");

  // point potential variables to the appropriate locations on host
  CP.delta_c    = CP.host;
  offset += n_cells;
#ifndef ONLY_PARTICLES
  CP.delta_b   = &(CP.host[offset]);
  offset += n_cells;
#endif
  CP.phi_1 = &(CP.host[offset]);


  // allocate memory for the conserved variable arrays on the device
  GPU_Error_Check(cudaMalloc((void **)&CP.device, CP.n_fields * n_cells * sizeof(Real)));
  cuda_utilities::initGpuMemory(CP.device, CP.n_fields * n_cells * sizeof(Real));

  chprintf("Device memory allocated for cosmological ICs initial potential.\n");

  // point potential variables to the appropriate locations on the device
  CP.d_delta_c   = CP.device;
  offset += n_cells;
#ifndef ONLY_PARTICLES
  CP.d_delta_b  = &(CP.device[offset]);
  offset += n_cells;
#endif
  CP.d_phi_1 = &(CP.device[offset]);

  // initialize host array
  for (int i = 0; i < CP.n_fields * n_cells; i++) {
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


/*! \fn void Generate_Normal_Random_Field(Real *d_field, rng_parallel_state_t *state)
 *  \brief Create a Gaussian random field on a grid */
void Grid3D::Generate_Normal_Random_Field(Real *d_field, rng_parallel_state_t *state)
{
	// Here, d_field has been pre-allocated on the device
  int n_cells = nx_local*ny_local*nz_local;
  cuda_utilities::AutomaticLaunchParams static const launchParams(RNG_Normal_Field_GPU, n_cells);
  hipLaunchKernelGGL(RNG_Normal_Field_GPU, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                     d_field, nx_local, ny_local, nz_local, 0, state);
}

/*! \fn void Rescale_Field(Real *d_x, Real A)
 *  \brief Rescale a field by a constant multiplicative factor. */
void Grid3D::Rescale_Field(Real *d_x, Real A)
{
	// Here, d_x has been pre-allocated on the device
	// Rescale the field by a multiplicative factor.
  int n_cells = nx_local*ny_local*nz_local;
  cuda_utilities::AutomaticLaunchParams static const launchParams(Rescale_Field_GPU, n_cells);
  hipLaunchKernelGGL(Rescale_Field_GPU, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                     d_x, A, nx_local, ny_local, nz_local, 0);
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

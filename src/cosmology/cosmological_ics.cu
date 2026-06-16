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
	chprintf("Cosmological ICs: Generating potentials....\n");

	// set the number of fields
	CP.n_fields = 2; //initial potential and overdensity field	
#ifndef ONLY_PARTICLES
	CP.n_fields += 1;	// add a baryon overdensity field
#endif 

  Real H0      = P->H0;
  Real h       = H0 / 100;
  Real Omega_M = P->Omega_M;
  Real G       = G_COSMO;

  chprintf("Cosmological ICs: h = %f \n", h);
  chprintf("Cosmological ICs: Omega_M = %f \n", Omega_M);

  H0 /= 1000;  //[km/s / kpc]
  Real rho_0  = 3 * H0 * H0 / (8 * M_PI * G) * Omega_M / h / h;

	// initialize the RNG properties
	Initialize_Cosmo_Potential_RNG(P);

  // load the P(k)
  Load_Cosmo_Power_Spectrum(P);

  // load the growth function
  Cosmo.Compute_Growth_Function(P);

  // save the growth function data to file
  Cosmo.Create_Growth_Function_File(P);

	// Initialize the FFT as well
	fft.Initialize( H.xdglobal, H.ydglobal, H.zdglobal, H.xblocal, H.yblocal, H.zblocal,
		              nx_global, ny_global, nz_global, nx_local, ny_local, nz_local, H.dx, H.dy, H.dz );

	// Allocate the memory needed for the potentials
	Allocate_Cosmo_Potential_Memory();

  // We have allocated the potential arrays, and are ready to proceed

	// step 1) sample xi(m) by generating independent
	//         zero-mean normal deviates with variance N**d at 
	//         each spatial point
	Generate_Normal_Random_Field(CP.d_delta_m,rng_states);

	// copy memory -- only real, local cells
	cudaMemcpy(CP.delta_m, CP.d_delta_m, n_cells * sizeof(Real), cudaMemcpyDeviceToHost);

  Real delta_rms = 0;
  Real delta_ave = 0;
  // reduce the grid values
  for (k = 0; k < H.nz - 2*H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2*H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2*H.n_ghost; i++) {

        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        delta_rms += pow(CP.delta_m[id],2);
        delta_ave += CP.delta_m[id];
      }
    }
  }

  // get the total of the grid to compute the rms
  MPI_Allreduce(MPI_IN_PLACE, &delta_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // get the total of the grid to compute the mean
  MPI_Allreduce(MPI_IN_PLACE, &delta_ave, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // find the rms and average
  delta_rms = delta_rms/(nx_global*ny_global*nz_global);
  delta_ave = delta_ave/(nx_global*ny_global*nz_global);
  delta_rms = sqrt(delta_rms);
  chprintf("Cosmological ICs: Mean of unit-variance field over entire grid = %e\n",delta_ave);
  chprintf("Cosmological ICs: RMS  of unit-variance field over entire grid = %e\n",delta_rms);

  // reduce the grid values
  for (k = 0; k < H.nz - 2*H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2*H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2*H.n_ghost; i++) {

        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        CP.delta_m[id] -= delta_ave; // remove global average
      }
    }
  }


	// copy mean zero phi back to GPU
	GPU_Error_Check(cudaMemcpy(CP.delta_m, CP.d_delta_m, n_cells * sizeof(Real), cudaMemcpyHostToDevice));

	// step 2) Take the fourier transform
	//         xi(k) = N**-d \sum_m exp( -(2 pi i / M) * kappa \dot m) * xi(m)

#ifndef ONLY_PARTICLES
  // step 2.5) Copy random field to baryonic field
	GPU_Error_Check(cudaMemcpy(CP.d_delta_bc, CP.d_delta_m, n_cells * sizeof(Real), cudaMemcpyDeviceToDevice));
#endif 

	// step 3) Multiply xi(k) by the transfer function 
	//         T(k) \equiv [(2 \pi / L)**3 P(k)]^{1/2}
	//         note T(k) is computed at z=0
  //         also note  the [(2 \pi / L)**3 ]^{1/2} factor is handled
  //         when the power spectrum is loaded, so this just applies sqrt(P(k))
  chprintf("Cosmological ICs: applying matter power spectrum...\n");
  fft.Filter_rescale_by_power_spectrum(CP.d_delta_m,CP.d_delta_m,true,CP.n_pk,CP.d_k_array,CP.d_pk_m_array);

#ifndef ONLY_PARTICLES
  chprintf("Cosmological ICs: applying baryonic - cdm power spectrum...\n");
  fft.Filter_rescale_by_power_spectrum(CP.d_delta_bc,CP.d_delta_bc,true,CP.n_pk,CP.d_k_array,CP.d_pk_bc_array);
#endif 

	// copy memory back to host
	cudaMemcpy(CP.delta_m,  CP.d_delta_m,  n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
#ifndef ONLY_PARTICLES 
	cudaMemcpy(CP.delta_bc, CP.d_delta_bc, n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
#endif 

  // free the P(k)
  Free_Cosmo_Power_Spectrum();

  delta_ave = 0;
  delta_rms = 0;
  for (k = 0; k < H.nz - 2*H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2*H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2*H.n_ghost; i++) {
        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        delta_ave += CP.delta_m[id]; // perform a local reduction
        delta_rms += pow(CP.delta_m[id],2);
      }
    }
  }

  // get the total of the grid to compute the mean
  MPI_Allreduce(MPI_IN_PLACE, &delta_ave, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // get the total of the grid to compute the rms
  MPI_Allreduce(MPI_IN_PLACE, &delta_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // find the average
  delta_ave = delta_ave/(nx_global*ny_global*nz_global);
  delta_rms = delta_rms/(nx_global*ny_global*nz_global);
  delta_rms = sqrt(delta_rms);
  chprintf("Cosmological ICs: Mean of overdensity field after P(k) %e\n",delta_ave);
  chprintf("Cosmological ICs: RMS  of overdensity field after P(k) %e\n",delta_rms);


  // note that delta_bc is about a ~0.1-1% effect,
  // depending on the k-scale

  // f_b = Omega_b / Omega_m
  // f_c = 1.0 - f_b

  // delta_m = D \nabla^2 \phi_ini
  // delta_b_ini =  f_c * delta_bc
  // delta_c_ini = -f_b * delta_bc
  // delta_c = \delta_m + delta_c_ini = delta_m - f_b * delta_bc
  // delta_b = \delta_m + delta_b_ini = delta_m + f_c * delta_bc
  // delta_b = delta_c + f_b delta_bc + f_c delta_bc = delta_c + delta_bc

  // We have verified that the delta_m and delta_bc 
  // reflect the expected Pm(k) and Pbc(k)

  // to first-order the lagrangian displacement is
  // xi_c_pert = D xi_m(1) - \nabla^-2 \grad \delta_c_init
  // xi_m(1) = - \grad \phi_ini
  // x_c = q - D(z_start) \grad \phi_ini - \nabla^-2 \grad \delta_c_init
  // x_c = q - D(z_start) \grad \phi_ini + f_b * \nabla^-2 \grad \delta_bc
  // v_c = dD/dt * \grad \phi_ini
  // v_b = dD/dt * \grad \phi_ini
  // \phi_ini = \nabla^-2 \delta_m (a_Ref)/D+(a_Ref) * D+(a)/a

  // the first method still requires the PT total mass displacements
  // for the dark matter

  // so we should compute
  // 1) phi_ini from delta_m
  // 2) \grad phi_ini
  // 3) v_c and v_b from \grad phi_ini
  // 4) delta_b from delta_m and delta_bc
  // 5) replace delta_bc with \nabla^-2 delta_bc
  // 6) x_c from phi_ini and gradient of delta_bc


  // At this stage, the cosmological overdensity field(s)
  // has/have been computed. These can be used to compute
  // the initial potential fields, which are then used to set
  // the remaining initial conditions.

  // let's calculate phi_1
  chprintf("Cosmological ICs: calculating initial potential from inverse lapacian...\n");

  // Perhaps compute phi_init here as advertised?
  // should return phi_1 = \nabla^-2 delta_m
  fft.Filter_inv_k2(CP.d_delta_m,CP.d_phi_1,true);

#ifndef ONLY_PARTICLES
  // compute \nabla^-2 \delta_bc
  fft.Filter_inv_k2(CP.d_delta_bc,CP.d_phi_2,true);
#endif //ONLY_PARTICLES

	// copy memory back to host
  // note we are only using the
  // first n_local**3 cells on the host
  // and need to remap before populating
  // the potential. We can re-use existing
  // density arrays for the interim.
	cudaMemcpy(CP.phi_1,  CP.d_phi_1,  n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
#ifndef ONLY_PARTICLES
  cudaMemcpy(CP.phi_2,  CP.d_phi_2,  n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
#endif //ONLY_PARTICLES

  delta_ave = 0;
  delta_rms = 0;
  for (k = 0; k < H.nz - 2*H.n_ghost; k++) {
    for (j = 0; j < H.ny - 2*H.n_ghost; j++) {
      for (i = 0; i < H.nx - 2*H.n_ghost; i++) {
        // get cell index
        id = i + j * nx_local + k * nx_local * ny_local;

        delta_ave += CP.phi_1[id]; // perform a local reduction
        delta_rms += pow(CP.phi_1[id],2);
      }
    }
  }

  // get the total of the grid to compute the mean
  MPI_Allreduce(MPI_IN_PLACE, &delta_ave, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // get the total of the grid to compute the rms
  MPI_Allreduce(MPI_IN_PLACE, &delta_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // find the average
  delta_ave = delta_ave/(nx_global*ny_global*nz_global);
  delta_rms = delta_rms/(nx_global*ny_global*nz_global);
  delta_rms = sqrt(delta_rms);
  chprintf("Cosmological ICs: Mean of phi field %e\n",delta_ave);
  chprintf("Cosmological ICs: RMS  of phi field %e\n",delta_rms);

  chprintf("Cosmological ICs: Proceeding to finish initialization...\n");
  chexit(0);
}

/*! \fn void Save_Cosmo_Potential(struct Parameters *P)
 *  \brief Write out the cosmological potential field to hdf5 files*/
void Grid3D::Save_Cosmo_Potential(struct Parameters const *P)
{
  char fname[200];
  hid_t f_id, d_id, a0_id, a1_id, a2_id, a3_id;
  hid_t fs_id, fsa0_id, fsa1_id, fsa2_id, fsa3_id, ms_id;
#ifndef ONLY_PARTICLES
  hid_t db_id;
  hid_t ba0_id, ba1_id, ba2_id, ba3_id;
  hid_t fbs_id, fbsa0_id, fbsa1_id, fbsa2_id, fbsa3_id;
#endif
  hsize_t dimsf[3];
  hsize_t dimsa = 3;
  int attr_global[3];
  int attr_start[3];
  int attr_size[3];
  int attr_ghost[3];
  herr_t status;
  Real *phi_out; // output cosmological potential

  //Create a file name for each hdf5 output
  sprintf(fname,"%s0/delta_ini.h5.%d",P->outdir,procID);

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
  d_id = H5Dcreate2(f_id, "delta_m", H5T_NATIVE_DOUBLE, fs_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

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
        id = i + j * nx_local + k * nx_local * ny_local; // We are only using real cells

        ii = i;
        jj = j;
        kk = k;
        idx = ii*(nz_local*ny_local) + jj*nz_local + kk; // row major

        phi_out[idx] = CP.delta_m[id]; // map to output CDM overdensity field
      }
    }
  }


  status = H5Dwrite(d_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, phi_out);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
  H5Aclose(a0_id);
  H5Aclose(a1_id);
  H5Aclose(a2_id);
  H5Aclose(a3_id);
  H5Dclose(d_id);
  H5Sclose(fs_id);
  H5Sclose(fsa3_id);
  H5Sclose(fsa2_id);
  H5Sclose(fsa1_id);
  H5Sclose(fsa0_id);


#ifndef ONLY_PARTICLES

  fbs_id   = H5Screate_simple(3, dimsf, NULL);
  db_id = H5Dcreate2(f_id, "delta_bc", H5T_NATIVE_DOUBLE, fbs_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // attach attributes
  fbsa0_id = H5Screate_simple(1, &dimsa, NULL);
  fbsa1_id = H5Screate_simple(1, &dimsa, NULL);
  fbsa2_id = H5Screate_simple(1, &dimsa, NULL);
  fbsa3_id = H5Screate_simple(1, &dimsa, NULL);

  ba0_id = H5Acreate2(db_id, "global", H5T_NATIVE_INT, fbsa0_id, H5P_DEFAULT, H5P_DEFAULT);
  ba1_id = H5Acreate2(db_id, "start",  H5T_NATIVE_INT, fbsa1_id, H5P_DEFAULT, H5P_DEFAULT);
  ba2_id = H5Acreate2(db_id, "size",   H5T_NATIVE_INT, fbsa2_id, H5P_DEFAULT, H5P_DEFAULT);
  ba3_id = H5Acreate2(db_id, "ghost",  H5T_NATIVE_INT, fbsa3_id, H5P_DEFAULT, H5P_DEFAULT);
                
                           
  status = H5Awrite(ba0_id, H5T_NATIVE_INT, attr_global);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
                           
  status = H5Awrite(ba1_id, H5T_NATIVE_INT, attr_start);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
  status = H5Awrite(ba2_id, H5T_NATIVE_INT, attr_size);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
  status = H5Awrite(ba3_id, H5T_NATIVE_INT, attr_ghost);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }


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

        phi_out[idx] = CP.delta_bc[id]; // map to output baryon overdensity field
      }
    }
  }


  status = H5Dwrite(db_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, phi_out);
  if(status < 0) {
    printf("Error writing data to HDF5 on process %d\n",procID);
  }
  H5Aclose(ba0_id);
  H5Aclose(ba1_id);
  H5Aclose(ba2_id);
  H5Aclose(ba3_id);
  H5Dclose(db_id);
  H5Sclose(fbs_id);
  H5Sclose(fbsa3_id);
  H5Sclose(fbsa2_id);
  H5Sclose(fbsa1_id);
  H5Sclose(fbsa0_id);

#endif //ONLY_PARTICLES



  H5Fclose(f_id);
}

/*! \fn void Initialize_Cosmo_Potential_RNG(void)
 *  \brief Initialize the RNG for cosmological ICs potentials */
void Grid3D::Initialize_Cosmo_Potential_RNG(struct Parameters *P)
{
	// Initialize the parameters for the Philox RNG
  int n_cells = nx_local*ny_local*nz_local;
//  int n_cells = H.nx * H.ny * H.nz;

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
//                     H.nx,H.ny,H.nz,0,CP.rng_seed, CP.rng_subsequence, CP.rng_offset, rng_states);

  GPU_Error_Check();
                     
	chprintf("Initialized Cosmological ICs RNG states.");
}



/*! \fn void Free_Cosmo_Power_Spectrum()
 *  \brief Free memory for cosmological power spectrum*/
void Grid3D::Free_Cosmo_Power_Spectrum()
{
  // free host P(k) info
  free(CP.k_array);
  free(CP.pk_m_array);
  free(CP.pk_bc_array);

  // free device P(k) info
  GPU_Error_Check(cudaFree(CP.d_n_pk));
  GPU_Error_Check(cudaFree(CP.d_k_array));
  GPU_Error_Check(cudaFree(CP.d_pk_m_array));
  GPU_Error_Check(cudaFree(CP.d_pk_bc_array));
}

/*! \fn void Load_Cosmo_Power_Spectrum(struct Parameters *P)
 *  \brief Allocate memory and load cosmological power spectrum*/
void Grid3D::Load_Cosmo_Power_Spectrum(struct Parameters *P)
{
  
  char pk_filename[MAXLEN];
  strcpy(pk_filename, P->cosmo_ics_pk_file);
  chprintf( "Cosmological ICs: Loading Power Spectrum File: %s \n", pk_filename );
  
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
  chprintf( "Cosmological ICs: Loaded %d lines in file. \n", n_lines  );
  
  // Allocate cpu and device memory for power spectrum
  CP.n_pk         = n_lines;
  CP.k_array      = (Real *)malloc( CP.n_pk*sizeof(Real) );
  CP.pk_m_array   = (Real *)malloc( CP.n_pk*sizeof(Real) );
  CP.pk_bc_array  = (Real *)malloc( CP.n_pk*sizeof(Real) );
  
  chprintf( "Cosmological ICs: Lbox = %f %f %f  n_grid = %d %d %d\n", P->xlen, P->ylen, P->zlen, P->nx, P->ny, P->nz );
  
  Real dx = P->xlen / P->nx; 

  // Rescale the cosmological power spectrum

  // The log-log interpolation of the power spectrum
  // correctly returns the amplitude expected with the
  // rescaled P(k) -- for instance, if forcing a sample
  // at k=1./195 h/kpc ~ 5.1 h/Mpc gives P(k) ~ 1.26 (times 1e9 if rescaled)
  // because the rms density fluctuations are 3.548910e+04

  // if instead we adopt a Mpc unit system, we get 
  // at k=1./195 h/kpc ~ 5.1 h/Mpc gives P(k) ~ 1.26

  // So, evaluated at the same location in h/kpc or h/Mpc
  // returns the same P(k)

  // If we keep the h/Mpc units, then when filtering
  // by the P(k) we get 
  // RMS of density field over entire grid = 4.583223e-01

  // If we keep the h/kpc units, then when filtering
  // by the P(k) we get 
  // RMS of density field over entire grid = 1.449342e+04

  // The factor of the power spectrum is leaking through

  // So the scaling factor should not have the Mpc->kpc conversion
  // unless we undo the scaling



  // Multiplying by the square root of the power spectrum
  // re HERE
  //Real pk_factor = 1; // RMS of density field over entire grid = 4.583223e-01
  // if we transfer forward, multiply by 1, and then backward, we get the same variance we input
  Real pk_factor = 1.0e9*(nx_global*ny_global*nz_global)/(P->xlen*P->ylen*P->zlen);
  //Real pk_factor = 1.0e9*(/pow(50000,3); // convert (Mpc/h)^3 to (kpc/h)^3 1.296331e-03

  //Real pk_factor = 1.0e9; // convert (Mpc/h)^3 to (kpc/h)^3 RMS of density field over entire grid = 1.449342e+04
  //Real pk_factor = 1.0e9/pow(50000,3); // convert (Mpc/h)^3 to (kpc/h)^3 1.296331e-03
  //Real pk_factor = 1.0e9/pow(50000/(2*M_PI),3); // convert (Mpc/h)^3 to (kpc/h)^3 2e-2
  //Real pk_factor = 1; // RMS of density field over entire grid = 4.583223e-01
  //Real pk_factor = 1/(2.0*M_PI*M_PI); // RMS of density field over entire grid = 4.583223e-01
  //Real pk_factor = pow(50,3)/(2.0); // RMS of density field over entire grid = 4.583223e-01
  // (50/256)**3 = 0.007450580596923828
  // (50000/256)**3 = 7450580.596923828



  // The value of the power spectrum at redshift 0 and a ~195 kpc/h ~ 5 h/Mpc cell is ~ 1

  chprintf("Cosmological ICs: Power spectrum rescaling factor: %e\n",pk_factor);

  // note the pk_factor is supposed to remove the volume element from 
  // the normalization of P(k), and needs to be in the units of the original P(k)
  // We are assuming P(k) has units of (Mpc/h)^3 and xlen, ylen, zlen are in kpc/h
  
  for (i=0; i<n_lines; i++ ){
    //CP.k_array[i]      = v[i][0];       //Convert from 1/(Mpc/h) to  1/(kpc/h)
    CP.k_array[i]      = v[i][0] * 1e-3;       //Convert from 1/(Mpc/h) to  1/(kpc/h)
    CP.pk_m_array[i]   = v[i][1] * pk_factor;  // P(k) rescaling
    if(j==3)
    {
      CP.pk_bc_array[i] = v[i][2] * pk_factor; //P(k) rescaling
    }else{
      CP.pk_bc_array[i] = 0;  // no baryon-cdm difference
    }
  }

  // Allocate device P(k) arrays  
  GPU_Error_Check(cudaMalloc((void**)&CP.d_n_pk,        sizeof(int)) );
  GPU_Error_Check(cudaMalloc((void**)&CP.d_k_array,     CP.n_pk*sizeof(Real)) );
  GPU_Error_Check(cudaMalloc((void**)&CP.d_pk_m_array,  CP.n_pk*sizeof(Real)) );
  GPU_Error_Check(cudaMalloc((void**)&CP.d_pk_bc_array, CP.n_pk*sizeof(Real)) );

  // Copy host P(k) to device P(k)
  GPU_Error_Check(cudaMemcpy(CP.d_n_pk,       &CP.n_pk,        sizeof(int),  cudaMemcpyHostToDevice) );
  GPU_Error_Check(cudaMemcpy(CP.d_k_array,     CP.k_array,     CP.n_pk*sizeof(Real), cudaMemcpyHostToDevice) );
  GPU_Error_Check(cudaMemcpy(CP.d_pk_m_array,  CP.pk_m_array,  CP.n_pk*sizeof(Real), cudaMemcpyHostToDevice) );
  GPU_Error_Check(cudaMemcpy(CP.d_pk_bc_array, CP.pk_bc_array, CP.n_pk*sizeof(Real), cudaMemcpyHostToDevice) );
}


/*! \fn void Allocate_Cosmo_Potential_Memory(void)
 *  \brief Allocate the memory allocated for cosmological ICs potentials */
void Grid3D::Allocate_Cosmo_Potential_Memory()
{
  // allocate memory for the phi arrays
  // allocate all the memory to phi_1, to ensure contiguous memory
  int n_cells = nx_local*ny_local*nz_local;
  //int n_cells = H.n_cells;
  int offset = n_cells;

  GPU_Error_Check(cudaHostAlloc((void **)&CP.host, CP.n_fields * n_cells * sizeof(Real), cudaHostAllocDefault));
  chprintf("Host memory allocated for %d fields in cosmological ICs initial deltas (n = %d).\n",CP.n_fields, n_cells);

  // point potential variables to the appropriate locations on host
  CP.delta_m    = CP.host;
#ifndef ONLY_PARTICLES
  CP.delta_bc   = &(CP.host[offset]);
  offset += n_cells;
#endif

  // allocate memory for the conserved variable arrays on the device
  GPU_Error_Check(cudaMalloc((void **)&CP.device, CP.n_fields * n_cells * sizeof(Real)));
  cuda_utilities::initGpuMemory(CP.device, CP.n_fields * n_cells * sizeof(Real));

  chprintf("Device memory allocated for %d fields in cosmological ICs initial deltas (n = %d).\n",CP.n_fields, n_cells);

  // point potential variables to the appropriate locations on the device
  CP.d_delta_m   = CP.device;
  offset = n_cells;
#ifndef ONLY_PARTICLES
  CP.d_delta_bc  = &(CP.device[offset]);
#endif

  // initialize host array
  for (int i = 0; i < CP.n_fields * n_cells; i++) {
    CP.host[i] = 0.0;
  }


  // repeat for potentials
  // which include ghost cells
  n_cells = H.n_cells;
  offset  = n_cells;

  GPU_Error_Check(cudaHostAlloc((void **)&CP.hostp, CP.n_fields * n_cells * sizeof(Real), cudaHostAllocDefault));
  chprintf("Host memory allocated for %d fields in cosmological ICs initial potentials (n = %d).\n",CP.n_fields, n_cells);

  // point potential variables to the appropriate locations on host
  CP.phi_1    = CP.hostp;
#ifndef ONLY_PARTICLES
  CP.phi_2    = &(CP.hostp[offset]);
#endif

  // allocate memory for the conserved variable arrays on the device
  GPU_Error_Check(cudaMalloc((void **)&CP.devicep, CP.n_fields * n_cells * sizeof(Real)));
  cuda_utilities::initGpuMemory(CP.devicep, CP.n_fields * n_cells * sizeof(Real));

  chprintf("Device memory allocated for %d fields in cosmological ICs initial potentials (n = %d).\n",CP.n_fields, n_cells);

  // point potential variables to the appropriate locations on the device
  CP.d_phi_1 = CP.devicep;
  offset = n_cells;
#ifndef ONLY_PARTICLES
  CP.d_phi_2  = &(CP.devicep[offset]);
#endif

  // initialize host array
  for (int i = 0; i < CP.n_fields * n_cells; i++) {
    CP.hostp[i] = 0.0;
  }
}

/*! \fn void Free_Cosmo_Phi_Init(void)
 *  \brief Free the memory allocated for cosmological ICs potentials */
void Grid3D::Free_Cosmo_Potential_Memory(void)
{
  // free the host delta arrays
  GPU_Error_Check(cudaFreeHost(CP.host));

  // free the device delta arrays
  GPU_Error_Check(cudaFree(CP.device));

  // free the host phi arrays
  GPU_Error_Check(cudaFreeHost(CP.hostp));

  // free the device phi arrays
  GPU_Error_Check(cudaFree(CP.devicep));
}


/*! \fn void Generate_Normal_Random_Field(Real *d_field, rng_parallel_state_t *state)
 *  \brief Create a Gaussian random field on a grid */
void Grid3D::Generate_Normal_Random_Field(Real *d_field, rng_parallel_state_t *state)
{
	// Here, d_field has been pre-allocated on the device
  int n_cells = nx_local*ny_local*nz_local;
  //chprintf("nx_local %d ny_local %d nz_local %d\n",nx_local,ny_local,nz_local);
  //chexit(0);
  //int n_cells = H.nx * H.ny * H.nz;
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
  //int n_cells = H.nx * H.ny * H.nz;
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


// Set_Boundary_Conditions_Cosmo_Potential()
// -- calls Set_Boundaries_Cosmo_Potential
//    -- calls Set_Cosmo_Potential_Boundaries_Periodic
// OR
// Set_Boundaries_MPI_Cosmo_Potential


/*! \fn void Set_Boundary_Conditions_Cosmo_Potential(Parameters P )
 *  \brief Set the boundary conditions for all components based on info in the
 * parameters structure. */
void Grid3D::Set_Boundary_Conditions_Field(Parameters P, Real *field)
{
#ifndef MPI_CHOLLA

  int flags[6] = {0, 0, 0, 0, 0, 0};

  // Check for custom boundary conditions and set boundary flags
  // can use generic boundary check
  if (Check_Custom_Boundary(&flags[0], P)) {
    chprintf("Error -- custom boundary not implemented for cosmo ics.")
    chexit(-1);
  }

  // set regular boundaries
  if (H.nx > 1) {
    Set_Boundaries_Field(0, flags, field);
    Set_Boundaries_Field(1, flags, field);
  }
  if (H.ny > 1) {
    Set_Boundaries_Field(2, flags, field);
    Set_Boundaries_Field(3, flags, field);
  }
  if (H.nz > 1) {
    Set_Boundaries_Field(4, flags, field);
    Set_Boundaries_Field(5, flags, field);
  }

#else /*MPI_CHOLLA*/

  /*Set boundaries, including MPI exchanges*/

  Set_Boundaries_MPI_Field(P, field);

#endif /*MPI_CHOLLA*/
}


/*! \fn void Set_Boundaries(int dir, int flags[])
 *  \brief Apply boundary conditions to the grid. */
void Grid3D::Set_Boundaries_Field(int dir, int flags[], Real *field)
{
  int i, j, k;
  int imin[3] = {0, 0, 0};
  int imax[3] = {H.nx, H.ny, H.nz};
  Real a[3]   = {1, 1, 1};  // sign of momenta
  int idx;                  // index of a real cell
  int gidx;                 // index of a ghost cell

  int nPB, nBoundaries;
  int *iaBoundary, *iaCell;

  /*if the cell face is an custom boundary, exit */
  if (flags[dir] == 4) {
    return;
  }

#ifdef MPI_CHOLLA
  /*if the cell face is an mpi boundary, exit */
  if (flags[dir] == 5) {
    return;
  }
#endif /*MPI_CHOLLA*/
  if(true) {
    if (flags[dir] == 1) {
  // Set Periodic Boundaries for the ghost cells.
 
      if (dir == 0) {
        Set_Field_Boundaries_Periodic(0, 0, flags, field);
      }
      if (dir == 1) {
        Set_Field_Boundaries_Periodic(0, 1, flags, field);
      }
      if (dir == 2) {
        Set_Field_Boundaries_Periodic(1, 0, flags, field);
      }
      if (dir == 3) {
        Set_Field_Boundaries_Periodic(1, 1, flags, field);
      }
      if (dir == 4) {
        Set_Field_Boundaries_Periodic(2, 0, flags, field);
      }
      if (dir == 5) {
        Set_Field_Boundaries_Periodic(2, 1, flags, field);
      }
    }
    return;
  }
  /*
  // get the extents of the ghost region we are initializing
  Set_Boundary_Extents(dir, &imin[0], &imax[0]);

  // from grid/cuda_boundaries.cu
  SetGhostCells(C.device, H.nx, H.ny, H.nz, H.n_fields, H.n_cells, H.n_ghost, flags, imax[0] - imin[0],
                imax[1] - imin[1], imax[2] - imin[2], imin[0], imin[1], imin[2], dir);
  */
}


void Grid3D::Set_Field_Boundaries_Periodic(int direction, int side, int *flags, Real *field)
{
  // Flags: 1 (periodic), 2 (reflective), 3 (transmissive), 4 (custom), 5 (mpi)

  int i, j, k, indx_src, indx_dst;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;
  nx_g  = nx_local + 2 * nGHST;
  ny_g  = ny_local + 2 * nGHST;
  nz_g  = nz_local + 2 * nGHST;

  // Copy X boundaries
  if (direction == 0) {
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nGHST; i++) {
          if (side == 0) {
            indx_src = (nx_g - 2 * nGHST + i) + (j)*nx_g + (k)*nx_g * ny_g;  // Periodic
            indx_dst = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx_src = (i + nGHST) + (j)*nx_g + (k)*nx_g * ny_g;  // Periodic
            indx_dst = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          field[indx_dst] = field[indx_src];
        }
      }
    }
  }

  // Copy Y boundaries
  if (direction == 1) {
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < nGHST; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx_src = (i) + (ny_g - 2 * nGHST + j) * nx_g + (k)*nx_g * ny_g;  // Periodic
            indx_dst = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx_src = (i) + (j + nGHST) * nx_g + (k)*nx_g * ny_g;  // Periodic
            indx_dst = (i) + (ny_g - nGHST + j) * nx_g + (k)*nx_g * ny_g;
          }
          field[indx_dst] = field[indx_src];
        }
      }
    }
  }

  // Copy Z boundaries
  if (direction == 2) {
    for (k = 0; k < nGHST; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx_src = (i) + (j)*nx_g + (nz_g - 2 * nGHST + k) * nx_g * ny_g;  // Periodic
            indx_dst = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx_src = (i) + (j)*nx_g + (k + nGHST) * nx_g * ny_g;  // Periodic
            indx_dst = (i) + (j)*nx_g + (nz_g - nGHST + k) * nx_g * ny_g;
          }
          field[indx_dst] = field[indx_src];
        }
      }
    }
  }
}





void Grid3D::Set_Boundaries_MPI_Field(struct Parameters P, Real *field)
{
  int flags[6] = {0, 0, 0, 0, 0, 0};

  if (Check_Custom_Boundary(&flags[0], P)) { //returns zero of bcnd!=4
    chprintf("Error -- custom boundary not implemented for cosmo ics.");
    chexit(-1);
  }

  Set_Boundaries_MPI_BLOCK_Field(flags, P, field);
}

void Grid3D::Set_Boundaries_MPI_BLOCK_Field(int *flags, struct Parameters P, Real *field)
{

  if (H.nx > 1) {
    /* Step 1 - Send MPI x-boundaries */
    if (flags[0] == 5 || flags[1] == 5) {
      Load_and_Send_MPI_Comm_Buffers_Field(0, flags, field);
    }

    /* Step 2 - Set non-MPI x-boundaries */
    // This does both phi_1 and phi_2 if needed
    Set_Boundaries_Field(0, flags, field);
    Set_Boundaries_Field(1, flags, field);

    /* Step 3 - Receive MPI x-boundaries */

    if (flags[0] == 5 || flags[1] == 5) {
      Wait_and_Unload_MPI_Comm_Buffers_Field(0, flags, field);
    }
  }
  MPI_Barrier(world);
  if (H.ny > 1) {
    /* Step 4 - Send MPI y-boundaries */
    if (flags[2] == 5 || flags[3] == 5) {
      Load_and_Send_MPI_Comm_Buffers_Field(1, flags, field);
    }

    /* Step 5 - Set non-MPI y-boundaries */
    // This does both phi_1 and phi_2 if needed
    Set_Boundaries_Field(2, flags, field);
    Set_Boundaries_Field(3, flags, field);

    /* Step 6 - Receive MPI y-boundaries */
    if (flags[2] == 5 || flags[3] == 5) {
      Wait_and_Unload_MPI_Comm_Buffers_Field(1, flags, field);
    }
  }
  MPI_Barrier(world);
  if (H.nz > 1) {
    /* Step 7 - Send MPI z-boundaries */
    if (flags[4] == 5 || flags[5] == 5) {
      Load_and_Send_MPI_Comm_Buffers_Field(2, flags, field);
    }

    /* Step 8 - Set non-MPI z-boundaries */
    // This does both phi_1 and phi_2 if needed
    Set_Boundaries_Field(4, flags, field);
    Set_Boundaries_Field(5, flags, field);

    /* Step 9 - Receive MPI z-boundaries */
    if (flags[4] == 5 || flags[5] == 5) {
      Wait_and_Unload_MPI_Comm_Buffers_Field(2, flags, field);
    }
  }
}


void Grid3D::Load_and_Send_MPI_Comm_Buffers_Field(int dir, int *flags, Real *field)
{
  int ireq;
  ireq = 0;

  int xbsize = x_buffer_length, ybsize = y_buffer_length, zbsize = z_buffer_length;

  int buffer_length;

  // Flag to omit the transfer of the main buffer when tranferring the particles
  // buffer
  bool transfer_main_buffer = true;

  /* x boundaries */
  if (dir == 0) {
    if (flags[0] == 5) {
      // load left x communication buffer

      buffer_length = Load_Field_To_Buffer(0, 0, h_send_buffer_x0, 0, field); //check

      if (transfer_main_buffer) {

        // post non-blocking receive left x communication buffer
        MPI_Irecv(h_recv_buffer_x0, buffer_length, MPI_CHREAL, source[0], 0, world, &recv_request[ireq]);

        // non-blocking send left x communication buffer
        MPI_Isend(h_send_buffer_x0, buffer_length, MPI_CHREAL, dest[0], 1, world, &send_request[0]);

        MPI_Request_free(send_request);

        // keep track of how many sends and receives are expected
        ireq++;
      }
    }

    if (flags[1] == 5) {
      // load right x communication buffer
      buffer_length = Load_Field_To_Buffer(0, 1, h_send_buffer_x1, 0, field);

      if (transfer_main_buffer) {

        // post non-blocking receive right x communication buffer
        MPI_Irecv(h_recv_buffer_x1, buffer_length, MPI_CHREAL, source[1], 1, world, &recv_request[ireq]);

        // non-blocking send right x communication buffer
        MPI_Isend(h_send_buffer_x1, buffer_length, MPI_CHREAL, dest[1], 0, world, &send_request[1]);

        MPI_Request_free(send_request + 1);

        // keep track of how many sends and receives are expected
        ireq++;
      }
    }
  }

  /* y boundaries */
  if (dir == 1) {
    if (flags[2] == 5) {
      // load left y communication buffer
      buffer_length = Load_Field_To_Buffer(1, 0, h_send_buffer_y0, 0, field);

      if (transfer_main_buffer) {

        // post non-blocking receive left y communication buffer
        MPI_Irecv(h_recv_buffer_y0, buffer_length, MPI_CHREAL, source[2], 2, world, &recv_request[ireq]);

        // non-blocking send left y communication buffer
        MPI_Isend(h_send_buffer_y0, buffer_length, MPI_CHREAL, dest[2], 3, world, &send_request[0]);

        MPI_Request_free(send_request);

        // keep track of how many sends and receives are expected
        ireq++;
      }
    }

    if (flags[3] == 5) {
      // load right y communication buffer
      buffer_length = Load_Field_To_Buffer(1, 1, h_send_buffer_y1, 0, field);

      if (transfer_main_buffer) {
        // post non-blocking receive right y communication buffer
        MPI_Irecv(h_recv_buffer_y1, buffer_length, MPI_CHREAL, source[3], 3, world, &recv_request[ireq]);

        // non-blocking send right y communication buffer
        MPI_Isend(h_send_buffer_y1, buffer_length, MPI_CHREAL, dest[3], 2, world, &send_request[1]);

        MPI_Request_free(send_request + 1);

        // keep track of how many sends and receives are expected
        ireq++;
      }
    }
  }

  /* z boundaries */
  if (dir == 2) {
    if (flags[4] == 5) {
      // left z communication buffer
      buffer_length = Load_Field_To_Buffer(2, 0, h_send_buffer_z0, 0, field);

      if (transfer_main_buffer) {

        // post non-blocking receive left z communication buffer
        MPI_Irecv(h_recv_buffer_z0, buffer_length, MPI_CHREAL, source[4], 4, world, &recv_request[ireq]);

        // non-blocking send left z communication buffer
        MPI_Isend(h_send_buffer_z0, buffer_length, MPI_CHREAL, dest[4], 5, world, &send_request[0]);

        MPI_Request_free(send_request);

        // keep track of how many sends and receives are expected
        ireq++;
      }
    }

    if (flags[5] == 5) {
      // load right z communication buffer
      buffer_length = Load_Field_To_Buffer(2, 1, h_send_buffer_z1, 0, field);

      if (transfer_main_buffer) {
        // post non-blocking receive right x communication buffer
        MPI_Irecv(h_recv_buffer_z1, buffer_length, MPI_CHREAL, source[5], 5, world, &recv_request[ireq]);

        // non-blocking send right x communication buffer
        MPI_Isend(h_send_buffer_z1, buffer_length, MPI_CHREAL, dest[5], 4, world, &send_request[1]);

        MPI_Request_free(send_request + 1);

        // keep track of how many sends and receives are expected
        ireq++;
      }
    }
  }
}

void Grid3D::Wait_and_Unload_MPI_Comm_Buffers_Field(int dir, int *flags, Real *field)
{
  int iwait;
  int index    = 0;
  int wait_max = 0;
  MPI_Status status;

  // find out how many recvs we need to wait for
  if (dir == 0) {
    if (flags[0] == 5) {  // there is communication on this face
      wait_max++;         // so we'll need to wait for its comm
    }
    if (flags[1] == 5) {  // there is communication on this face
      wait_max++;         // so we'll need to wait for its comm
    }
  }
  if (dir == 1) {
    if (flags[2] == 5) {  // there is communication on this face
      wait_max++;         // so we'll need to wait for its comm
    }
    if (flags[3] == 5) {  // there is communication on this face
      wait_max++;         // so we'll need to wait for its comm
    }
  }
  if (dir == 2) {
    if (flags[4] == 5) {  // there is communication on this face
      wait_max++;         // so we'll need to wait for its comm
    }
    if (flags[5] == 5) {  // there is communication on this face
      wait_max++;         // so we'll need to wait for its comm
    }
  }

  // wait for any receives to complete
  for (iwait = 0; iwait < wait_max; iwait++) {
    // wait for recv completion
    MPI_Waitany(wait_max, recv_request, &index, &status);

    // depending on which face arrived, load the buffer into the ghost grid
    Unload_MPI_Comm_Buffers_Field(status.MPI_TAG, field);
  }
}



void Grid3D::Unload_MPI_Comm_Buffers_Field(int index, Real *field)
{
  // local recv buffers
  Real *l_recv_buffer_x0, *l_recv_buffer_x1, *l_recv_buffer_y0, *l_recv_buffer_y1, *l_recv_buffer_z0, *l_recv_buffer_z1;

  Grid3D_PMF_UnloadField Fptr_Unload_Field;

  l_recv_buffer_x0 = h_recv_buffer_x0;
  l_recv_buffer_x1 = h_recv_buffer_x1;
  l_recv_buffer_y0 = h_recv_buffer_y0;
  l_recv_buffer_y1 = h_recv_buffer_y1;
  l_recv_buffer_z0 = h_recv_buffer_z0;
  l_recv_buffer_z1 = h_recv_buffer_z1;

  Fptr_Unload_Field = &Grid3D::Unload_Field_from_Buffer;

  if (index == 0) {
    (this->*Fptr_Unload_Field)(0, 0, l_recv_buffer_x0, 0, field);
  }
  if (index == 1) {
    (this->*Fptr_Unload_Field)(0, 1, l_recv_buffer_x1, 0, field);
  }
  if (index == 2) {
    (this->*Fptr_Unload_Field)(1, 0, l_recv_buffer_y0, 0, field);
  }
  if (index == 3) {
    (this->*Fptr_Unload_Field)(1, 1, l_recv_buffer_y1, 0, field);
  }
  if (index == 4) {
    (this->*Fptr_Unload_Field)(2, 0, l_recv_buffer_z0, 0, field);
  }
  if (index == 5) {
    (this->*Fptr_Unload_Field)(2, 1, l_recv_buffer_z1, 0, field);
  }
}



#ifdef MPI_CHOLLA
int Grid3D::Load_Field_To_Buffer(int direction, int side, Real *buffer, int buffer_start, Real *field)
{
  int i, j, k, indx, indx_buff, length;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;
  nx_g  = nx_local + 2 * nGHST;
  ny_g  = ny_local + 2 * nGHST;
  nz_g  = nz_local + 2 * nGHST;

  // Load X boundaries
  if (direction == 0) {
    length = nGHST * nz_g * ny_g;
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nGHST; i++) {
          if (side == 0) {
            indx = (i + nGHST) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (nx_g - 2 * nGHST + i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          indx_buff                        = (j) + (k)*ny_g + i * ny_g * nz_g;
          buffer[buffer_start + indx_buff] = field[indx];
        }
      }
    }
  }

  // Load Y boundaries
  if (direction == 1) {
    length = nGHST * nz_g * nx_g;
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < nGHST; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx = (i) + (j + nGHST) * nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (i) + (ny_g - 2 * nGHST + j) * nx_g + (k)*nx_g * ny_g;
          }
          indx_buff                        = (i) + (k)*nx_g + j * nx_g * nz_g;
          buffer[buffer_start + indx_buff] = field[indx];
        }
      }
    }
  }

  // Load Z boundaries
  if (direction == 2) {
    length = nGHST * nx_g * ny_g;
    for (k = 0; k < nGHST; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx = (i) + (j)*nx_g + (k + nGHST) * nx_g * ny_g;
          }
          if (side == 1) {
            indx = (i) + (j)*nx_g + (nz_g - 2 * nGHST + k) * nx_g * ny_g;
          }
          indx_buff                        = (i) + (j)*nx_g + k * nx_g * ny_g;
          buffer[buffer_start + indx_buff] = field[indx];
        }
      }
    }
  }
  return length;
}

void Grid3D::Unload_Field_from_Buffer(int direction, int side, Real *buffer, int buffer_start, Real *field)
{
  int i, j, k, indx, indx_buff;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;

  nx_g  = nx_local + 2 * nGHST;
  ny_g  = ny_local + 2 * nGHST;
  nz_g  = nz_local + 2 * nGHST;

  // Load X boundaries
  if (direction == 0) {
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nGHST; i++) {
          if (side == 0) {
            indx = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          indx_buff                = (j) + (k)*ny_g + i * ny_g * nz_g;
          field[indx] = buffer[buffer_start + indx_buff];
        }
      }
    }
  }

  // Load Y boundaries
  if (direction == 1) {
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < nGHST; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (i) + (ny_g - nGHST + j) * nx_g + (k)*nx_g * ny_g;
          }
          indx_buff                = (i) + (k)*nx_g + j * nx_g * nz_g;
          field[indx] = buffer[buffer_start + indx_buff];
        }
      }
    }
  }

  // Load Z boundaries
  if (direction == 2) {
    for (k = 0; k < nGHST; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (i) + (j)*nx_g + (nz_g - nGHST + k) * nx_g * ny_g;
          }
          indx_buff                = (i) + (j)*nx_g + k * nx_g * ny_g;
          field[indx] = buffer[buffer_start + indx_buff];
        }
      }
    }
  }
}
#endif    // MPI_CHOLLA
#endif //COSMOLOGY

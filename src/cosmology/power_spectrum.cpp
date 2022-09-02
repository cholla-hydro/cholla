#ifdef COSMOLOGY

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "power_spectrum.h"
#include "../io/io.h"


// Constructor
Cosmo_Power_Spectrum::Cosmo_Power_Spectrum(){}

void Cosmo_Power_Spectrum:: Load_Power_Spectum_From_File( struct parameters *P ){
  
  char pk_filename[MAXLEN];
  strcpy(pk_filename, P->cosmo_ics_pk_file);
  chprintf( " Loading Power Spectrum File: %s \n", pk_filename );
  
  std::fstream in_file(pk_filename);
  std::string line;
  std::vector<std::vector<float>> v;
  int i = 0;
  if (in_file.is_open()){
    while (std::getline(in_file, line))
    {
       if ( line.find("#") == 0 ) continue;
      
       float value;
       std::stringstream ss(line);
       if (line.length() == 0) continue;
       // chprintf( "%s \n", line.c_str() );
       v.push_back(std::vector<float>());
       
       while (ss >> value){
         // printf( " %d   %f\n", i, value );
         v[i].push_back(value);
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
  
  host_size = n_lines;
  host_k      = (Real *)malloc( host_size*sizeof(Real) );
  host_pk_dm  = (Real *)malloc( host_size*sizeof(Real) );
  host_pk_gas = (Real *)malloc( host_size*sizeof(Real) );
  
  chprintf( "Lbox = %f   n_grid = %d ", P->xlen, P->nx  );
  
  Real dx = P->xlen / P->nx; 
  Real pk_factor = 1 / ( dx * dx * dx );
  
  for (i=0; i<n_lines; i++ ){
    host_k[i]      = v[i][0] * 1e-3; //Convert from 1/(Mpc/h) to  1/(kpc/h)
    host_pk_dm[i]  = v[i][1] * pk_factor;  //IMPORTANT: The Power Spectrum has to be rescaled by the reolution volume!! Need to understand this! 
    host_pk_gas[i] = v[i][2] * pk_factor; 
  }
  
  CudaSafeCall( cudaMalloc((void**)&dev_size,   sizeof(int)) );
  CudaSafeCall( cudaMalloc((void**)&dev_k,      host_size*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_pk_dm,  host_size*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_pk_gas, host_size*sizeof(Real)) );
  
  CudaSafeCall( cudaMemcpy(dev_size, &host_size, sizeof(int), cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(dev_k, host_k, host_size*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(dev_pk_dm,  host_pk_dm,  host_size*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(dev_pk_gas, host_pk_gas, host_size*sizeof(Real), cudaMemcpyHostToDevice) );

}











#endif //COSMOLOGY
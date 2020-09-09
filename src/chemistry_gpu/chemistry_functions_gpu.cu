#ifdef CHEMISTRY_GPU

#include "chemistry_gpu.h"
#include "chemistry_functions_gpu.cuh"
#include "../hydro_cuda.h"
#include"../global_cuda.h"
#include"../io.h"

  
void Chem_GPU::Allocate_Array_GPU_float( float **array_dev, int size ){
  cudaMalloc( (void**)array_dev, size*sizeof(float));
  CudaCheckError();
}

void Chem_GPU::Copy_Float_Array_to_Device( int size, float *array_h, float *array_d ){
  CudaSafeCall( cudaMemcpy(array_d, array_h, size*sizeof(float), cudaMemcpyHostToDevice ) );
}

void Chem_GPU::Free_Array_GPU_float( float *array_dev ){
  cudaFree( array_dev );
  CudaCheckError();
}

class Thermal_State{
public: 
  
  Real T;
  Real n_HI;
  Real n_HII;
  Real n_HeI;
  Real n_HeII;
  Real n_HeIII;
  Real n_e;
  
  // Constructor
  __host__ __device__ Thermal_State( Real T_0=1, Real n_HI_0=1, Real n_HII_0=0, Real n_HeI_0=1, Real n_HeII_0=0, Real n_HeIII_0=1, Real n_e_0=0    ) : T(T_0), n_HI(n_HI_0), n_HII(n_HII_0), n_HeI(n_HeI_0), n_HeII(n_HeII_0), n_HeIII(n_HeIII_0), n_e(n_e_0) {}
  
  __host__ __device__ Thermal_State operator+( const Thermal_State &TS ){
    return Thermal_State( T+TS.T, n_HI+TS.n_HI, n_HII+TS.n_HII, n_HeI+TS.n_HeI, n_HeII+TS.n_HeII, n_HeIII+TS.n_HeIII, n_e+TS.n_e );
  }

  __host__ __device__ Thermal_State operator-( const Thermal_State &TS ){
    return Thermal_State( T-TS.T, n_HI-TS.n_HI, n_HII-TS.n_HII, n_HeI-TS.n_HeI, n_HeII-TS.n_HeII, n_HeIII-TS.n_HeIII, n_e-TS.n_e );
  }

  __host__ __device__ Thermal_State operator*( const Real &x ){
    return Thermal_State( T*x, n_HI*x,  n_HII*x, n_HeI*x,  n_HeII*x, n_HeIII*x,  n_e*x  );
  }
  
  __host__ __device__ Real get_MMW( ){
    Real n_mass = n_HI + n_HII + 4 * ( n_HeI + n_HeII + n_HeIII );
    Real n_tot =  n_HI + n_HII + n_HeI + n_HeII + n_HeIII + n_e;  
    return n_mass / n_tot;
  }

};

texture<float, 1, cudaReadModeElementType> tex_Heat_rate_HI;
texture<float, 1, cudaReadModeElementType> tex_Heat_rate_HeI;
texture<float, 1, cudaReadModeElementType> tex_Heat_rate_HeII;
texture<float, 1, cudaReadModeElementType> tex_Ion_rate_HI;
texture<float, 1, cudaReadModeElementType> tex_Ion_rate_HeI;
texture<float, 1, cudaReadModeElementType> tex_Ion_rate_HeII;

cudaArray* H_HI_arr;
cudaArray* H_HeI_arr;
cudaArray* H_HeII_arr;
cudaArray* I_HI_arr;
cudaArray* I_HeI_arr;
cudaArray* I_HeII_arr; 

void Chem_GPU::Bind_GPU_Textures( int size, float *H_HI_h, float *H_HeI_h, float *H_HeII_h , float *I_HI_h, float *I_HeI_h, float *I_HeII_h ){
  
  chprintf( " Binding GPU textures. \n");
  // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  
  cudaMallocArray ( &H_HI_arr, &tex_Heat_rate_HI.channelDesc, size, 1 );
  cudaMemcpyToArray( H_HI_arr, 0, 0, H_HI_h, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTextureToArray( tex_Heat_rate_HI, H_HI_arr );
  tex_Heat_rate_HI.normalized = false;
  tex_Heat_rate_HI.filterMode = cudaFilterModeLinear;
  tex_Heat_rate_HI.addressMode[0] = cudaAddressModeBorder;
  tex_Heat_rate_HI.addressMode[1] = cudaAddressModeBorder;

  cudaMallocArray ( &H_HeI_arr, &tex_Heat_rate_HeI.channelDesc, size, 1 );
  cudaMemcpyToArray( H_HeI_arr, 0, 0, H_HeI_h, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTextureToArray( tex_Heat_rate_HeI, H_HeI_arr );
  tex_Heat_rate_HeI.normalized = false;
  tex_Heat_rate_HeI.filterMode = cudaFilterModeLinear;
  tex_Heat_rate_HeI.addressMode[0] = cudaAddressModeBorder;
  tex_Heat_rate_HeI.addressMode[1] = cudaAddressModeBorder;

  cudaMallocArray ( &H_HeII_arr, &tex_Heat_rate_HeII.channelDesc, size, 1 );
  cudaMemcpyToArray( H_HeII_arr, 0, 0, H_HeII_h, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTextureToArray( tex_Heat_rate_HeII, H_HeII_arr );
  tex_Heat_rate_HeII.normalized = false;
  tex_Heat_rate_HeII.filterMode = cudaFilterModeLinear;
  tex_Heat_rate_HeII.addressMode[0] = cudaAddressModeBorder;
  tex_Heat_rate_HeII.addressMode[1] = cudaAddressModeBorder;

  cudaMallocArray ( &I_HI_arr, &tex_Ion_rate_HI.channelDesc, size, 1 );
  cudaMemcpyToArray( I_HI_arr, 0, 0, I_HI_h, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTextureToArray( tex_Ion_rate_HI, I_HI_arr );
  tex_Ion_rate_HI.normalized = false;
  tex_Ion_rate_HI.filterMode = cudaFilterModeLinear;
  tex_Ion_rate_HI.addressMode[0] = cudaAddressModeBorder;
  tex_Ion_rate_HI.addressMode[1] = cudaAddressModeBorder;

  cudaMallocArray ( &I_HeI_arr, &tex_Ion_rate_HeI.channelDesc, size, 1 );
  cudaMemcpyToArray( I_HeI_arr, 0, 0, I_HeI_h, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTextureToArray( tex_Ion_rate_HeI, I_HeI_arr );
  tex_Ion_rate_HeI.normalized = false;
  tex_Ion_rate_HeI.filterMode = cudaFilterModeLinear;
  tex_Ion_rate_HeI.addressMode[0] = cudaAddressModeBorder;
  tex_Ion_rate_HeI.addressMode[1] = cudaAddressModeBorder;

  cudaMallocArray ( &I_HeII_arr, &tex_Ion_rate_HeII.channelDesc, size, 1 );
  cudaMemcpyToArray( I_HeII_arr, 0, 0, I_HeII_h, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTextureToArray( tex_Ion_rate_HeII, I_HeII_arr );
  tex_Ion_rate_HeII.normalized = false;
  tex_Ion_rate_HeII.filterMode = cudaFilterModeLinear;
  tex_Ion_rate_HeII.addressMode[0] = cudaAddressModeBorder;
  tex_Ion_rate_HeII.addressMode[1] = cudaAddressModeBorder;


}

  
  
__device__ int Binary_Search( int N, Real val, float *data, int indx_l, int indx_r ){
  int n, indx;
  n = indx_r - indx_l;
  indx = indx_l + n/2;
  if ( val >= data[N-1] ) return indx_r;
  if ( val <= data[0]   ) return indx_l;
  if ( indx_r == indx_l + 1 ) return indx_l;
  if ( data[indx] <= val ) indx_l = indx;
  else indx_r = indx;
  return Binary_Search( N, val, data, indx_l, indx_r );
}




__device__ void get_current_uvb_rates( Real current_z, int n_uvb_rates_samples, float *rates_z, float &photo_i_HI, float &photo_h_HI, float &photo_i_HeI, float &photo_h_HeI, float &photo_i_HeII, float &photo_h_HeII ){
  
  
  if ( current_z > rates_z[n_uvb_rates_samples - 1]){
    photo_h_HI   = 0;  
    photo_h_HeI  = 0;  
    photo_h_HeII = 0;  
    photo_i_HI   = 0;  
    photo_i_HeI  = 0;  
    photo_i_HeII = 0;  
    return;
  }
  
  // Find closest value of z in rates_z such that z<=current_z
  int indx_l;
  indx_l = Binary_Search( n_uvb_rates_samples, current_z, rates_z, 0, n_uvb_rates_samples-1 );
  
  Real z_l, delta_z;
  z_l = rates_z[indx_l];
  if ( indx_l < n_uvb_rates_samples - 1 ){
    delta_z = rates_z[indx_l+1] - z_l;
  }
  else{
    delta_z = 1;
  }
  
  // printf( "%f  %f %f \n", current_z, rates_z[indx_l], delta_z );
  float x = (float)indx_l + ( current_z - z_l ) / delta_z; 
  x += + 0.5;
  photo_h_HI   = tex1D( tex_Heat_rate_HI, x );
  photo_h_HeI  = tex1D( tex_Heat_rate_HeI, x );
  photo_h_HeII = tex1D( tex_Heat_rate_HeII, x );
  photo_i_HI   = tex1D( tex_Ion_rate_HI, x );
  photo_i_HeI  = tex1D( tex_Ion_rate_HeI, x );
  photo_i_HeII = tex1D( tex_Ion_rate_HeII, x );
  // printf("%f  %f  %e  %e  %e  %e  %e  %e \n ", current_z, z_l, photo_i_HI, photo_h_HI, photo_i_HeI, photo_h_HeI, photo_i_HeII, photo_h_HeII );
    
                     
}
                                       

__host__ __device__ Real Get_Chemistry_dt( Thermal_State &TS, Thermal_State &TS_derivs ){
  Real dt_min = 1e100;
  
  dt_min = fmin( dt_min, TS.T / fabs(TS_derivs.T) );  
  dt_min = fmin( dt_min, TS.n_HI / fabs(TS_derivs.n_HI) );
  dt_min = fmin( dt_min, TS.n_HII / fabs(TS_derivs.n_HII) );  
  dt_min = fmin( dt_min, TS.n_HeI / fabs(TS_derivs.n_HeI) );
  dt_min = fmin( dt_min, TS.n_HeII / fabs(TS_derivs.n_HeII) );
  dt_min = fmin( dt_min, TS.n_HeIII / fabs(TS_derivs.n_HeIII) );
  dt_min = fmin( dt_min, TS.n_e / fabs(TS_derivs.n_e) );
  
  Real alpha = 0.1;
  return dt_min * alpha;  
}



__device__ void Add_Expantion_Derivatives( Thermal_State &TS, Thermal_State &TS_derivs,Real dt, Real current_a, Real H0, Real Omega_M, Real Omega_L ){
  
  Real current_z, H;
  current_z = 1/current_a - 1;
  H = H0 * sqrt( Omega_M/pow(current_a,3) + Omega_L ) / ( 1000 * KPC ) * dt;
  TS_derivs.T += -2 * H * TS.T;
  TS_derivs.n_HI += -3 * H * TS.n_HI;
  TS_derivs.n_HII +=  -3 * H * TS.n_HII;
  TS_derivs.n_HeI += -3 * H * TS.n_HeI;
  TS_derivs.n_HeII += -3 * H * TS.n_HeII;
  TS_derivs.n_HeIII += -3 * H * TS.n_HeIII;
  TS_derivs.n_e += -3 * H * TS.n_e;
  
} 



__device__ void Get_Chemistry_Derivatives( Thermal_State &TS, Thermal_State &TS_derivs, Real current_a, Real H0, Real Omega_M, Real Omega_L, int n_uvb_rates_samples, float *rates_z, bool print ){
  
  float photo_h_HI, photo_h_HeI, photo_h_HeII; 
  float photo_i_HI, photo_i_HeI, photo_i_HeII;  
  
  Real current_z, H, n_tot;
  Real Q_phot, Q_tot, dnHI, dnHII, dnHeI, dnHeII, dnHeIII, dne    ;
  
  current_z = 1/current_a - 1;
  H = H0 * sqrt( Omega_M/pow(current_a,3) + Omega_L ) / ( 1000 * KPC );
  // printf(" H= %e\n", H);
  
  get_current_uvb_rates( current_z, n_uvb_rates_samples, rates_z, photo_i_HI, photo_h_HI, photo_i_HeI, photo_h_HeI, photo_i_HeII, photo_h_HeII );
  // if (print) printf( "z=%f  %e  %e  %e  %e  %e  %e  \n", current_z, photo_i_HI, photo_h_HI, photo_i_HeI, photo_h_HeI, photo_i_HeII, photo_h_HeII  );
  // Q_phot = TS.n_HI * photo_h_HI + TS.n_HeI * photo_h_HeI + TS.n_HeII * photo_h_HeII;  
  Q_phot =  TS.n_HeII * photo_h_HeII ;  
  
  
  dnHI  = -TS.n_HI*photo_i_HI *0;
  dnHII =  TS.n_HI*photo_i_HI *0;
  
  dnHeI   = -TS.n_HeI*photo_i_HeI *0;
  dnHeII  =  TS.n_HeI*photo_i_HeI *0 - TS.n_HeII*photo_i_HeII *0 ;
  dnHeIII =  TS.n_HeII*photo_i_HeII *0  ;
   
  dne  = TS.n_HI*photo_i_HI + TS.n_HeI*photo_i_HeI + TS.n_HeII*photo_i_HeII *0;
   
   
  
  
  Q_tot = Q_phot * 0;
  
  n_tot = TS.n_HI + TS.n_HII + TS.n_HeI + TS.n_HeII + TS.n_HeIII + TS.n_e;  
  
  TS_derivs.T       = -2 * H * TS.T + 2/(3*KB*n_tot)*Q_tot;
  TS_derivs.n_HI    = -3 * H * TS.n_HI    + dnHI;
  TS_derivs.n_HII   = -3 * H * TS.n_HII   + dnHII;
  TS_derivs.n_HeI   = -3 * H * TS.n_HeI   + dnHeI;
  TS_derivs.n_HeII  = -3 * H * TS.n_HeII  + dnHeII;
  TS_derivs.n_HeIII = -3 * H * TS.n_HeIII + dnHeIII;
  TS_derivs.n_e     = -3 * H * TS.n_e     + dne;
  
} 




__global__ void Update_Chemistry( Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt_hydro, Real gamma,  Real density_conv, Real energy_conv,
                                  Real current_z, float* cosmo_params, int n_uvb_rates_samples, float *rates_z   ){
  
    
  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz;
  Real  P, E, E_kin, GE, dt_chem, t_chem;
  
  Real H0, Omega_M, Omega_L;
  
  Real current_a, delta_a, a3, a2;
  
  
  n_cells = nx*ny*nz;
  
  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;
  bool print;
  
  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    E_kin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );
    #ifdef DE
    GE = dev_conserved[(n_fields-1)*n_cells + id];
    P = Get_Pressure_From_DE( E, E - E_kin, GE, gamma );  
    #else 
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    #endif
    P  = fmax(P, (Real) TINY_NUMBER);  
  
    // Convert to cgs units
    current_a = 1 / ( current_z + 1);
    a2 = current_a * current_a;
    a3 = a2 * current_a;
    // energy_conv = energy_conv / ( current_a * current_a );
    // density_conv = density_conv / ( current_a * current_a * current_a);
    
    if ( xid == n_ghost && yid == n_ghost && zid == n_ghost ) print = true;
    else print = false;
    
    P *= energy_conv / a2;  
    d *= density_conv / a3;
    // density_conv = density_conv / MP;
    
    // Load Cosmological Parameters
    H0 = cosmo_params[0];
    Omega_M = cosmo_params[1];
    Omega_L = cosmo_params[2];
    
    dt_hydro = dt_hydro * current_a * current_a / H0 * 1000 * KPC;
    delta_a = H0 * sqrt( Omega_M/current_a + Omega_L*pow(current_a, 2) ) / ( 1000 * KPC ) * dt_hydro;
    
  
    Thermal_State TS;
    Thermal_State TS_derivs;
    Thermal_State TS_delta_exp( 0, 0, 0, 0, 0, 0, 0 );
    Real mu;
    
    TS.n_HI    = dev_conserved[ 5*n_cells + id] * density_conv / a3; 
    TS.n_HII   = dev_conserved[ 6*n_cells + id] * density_conv / a3; 
    TS.n_HeI   = dev_conserved[ 7*n_cells + id] * density_conv / a3 / 4; 
    TS.n_HeII  = dev_conserved[ 8*n_cells + id] * density_conv / a3 / 4; 
    TS.n_HeIII = dev_conserved[ 9*n_cells + id] * density_conv / a3 / 4; 
    TS.n_e     = dev_conserved[10*n_cells + id] * density_conv / a3; 
    mu = TS.get_MMW();
    TS.T = P * MP  * mu * d_inv / KB  ;
    
    
    //   printf(" current_z = %f\n", current_z );
    //   printf(" dt = %f \n", dt_hydro / MYR ); 
    //   printf(" delta_a = %f \n", delta_a ); 
    //   printf(" H0 = %f\n", H0 );
    //   printf(" Omega_M = %f\n", Omega_M );
    //   printf(" Omega_L = %f\n", Omega_L );
    //   printf(" rho = %e\n", d );
    //   printf(" mu = %f \n", mu );
    // }
  
    if (print) printf(" T0 = %e  rho:%e  nHI:%e    \  n", TS.T, d, TS.n_HI );
  
    t_chem = 0;
    int n_iter = 0;
    while ( t_chem < dt_hydro ){
      
      
      
      Get_Chemistry_Derivatives( TS, TS_derivs, current_a, H0, Omega_M, Omega_L, n_uvb_rates_samples, rates_z, print );
      dt_chem = Get_Chemistry_dt( TS, TS_derivs );
      if ( dt_chem > dt_hydro ) dt_chem = dt_hydro;
      
      Add_Expantion_Derivatives( TS, TS_delta_exp, dt_chem, current_a, H0, Omega_M, Omega_L );
      TS = TS + (TS_derivs * dt_chem);
      
      current_a = current_a + delta_a;
      delta_a = H0 * sqrt( Omega_M/current_a + Omega_L*pow(current_a, 2) ) / ( 1000 * KPC ) * dt_chem;
      
      
      t_chem += dt_chem;
      n_iter += 1;
    }
    
    // Substract the change due to expansion
    TS = TS - TS_delta_exp;
    
    
    // Write the Updated Thermal State
    dev_conserved[ 5*n_cells + id] = TS.n_HI / density_conv * a3; 
    dev_conserved[ 6*n_cells + id] = TS.n_HII / density_conv * a3; 
    dev_conserved[ 7*n_cells + id] = TS.n_HeI / density_conv * a3 * 4; 
    dev_conserved[ 8*n_cells + id] = TS.n_HeII / density_conv * a3 * 4; 
    dev_conserved[ 9*n_cells + id] = TS.n_HeIII / density_conv * a3 * 4; 
    dev_conserved[10*n_cells + id] = TS.n_e / density_conv * a3; 
    P = TS.T / MP / mu / d_inv * KB / energy_conv * a2 ;
    dev_conserved[4*n_cells + id]  = P/(gamma - 1.0) + 0.5*d*(vx*vx + vy*vy + vz*vz);  
    #ifdef DE
    dev_conserved[(n_fields-1)*n_cells + id] = P / ( gamma - 1);
    #endif
    
    
    
    
    // 
    // if ( xid == n_ghost && yid == n_ghost && zid == n_ghost ){
    //   printf(" T1 = %e\n", TS.T);
    //   // printf(" Tf = %e\n", TS.T *pow( current_a/a_start, 2));
    //   printf(" df = %e\n", TS.n_HI / density_conv * a3 );
    // }  
    // 
  
  
  
  
  
  
  
  
  
  }
  
}













#endif
#ifdef CHEMISTRY_GPU

#include "chemistry_gpu.h"
#include "chemistry_functions_gpu.cuh"
#include "../hydro_cuda.h"
#include "../global_cuda.h"
#include "../io.h"
#include "rates.cuh"
#include "rates_Katz95.cuh"
#include "rates_grackle.cuh"

#define eV_to_K 1.160451812e4
#define K_to_eV 8.617333263e-5
  
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
  
  Real gamma;
  Real U;
  Real n_HI;
  Real n_HII;
  Real n_HeI;
  Real n_HeII;
  Real n_HeIII;
  Real n_e;
  
  // Constructor
  __host__ __device__ Thermal_State( Real gamma_0=5/3, Real U_0=1,  Real n_HI_0=1, Real n_HII_0=0, Real n_HeI_0=1, Real n_HeII_0=0, Real n_HeIII_0=1, Real n_e_0=0    ) : gamma(gamma_0), U(U_0), n_HI(n_HI_0), n_HII(n_HII_0), n_HeI(n_HeI_0), n_HeII(n_HeII_0), n_HeIII(n_HeIII_0), n_e(n_e_0) {}
  
  __host__ __device__ Thermal_State operator+( const Thermal_State &TS ){
    return Thermal_State( U+TS.U, n_HI+TS.n_HI, n_HII+TS.n_HII, n_HeI+TS.n_HeI, n_HeII+TS.n_HeII, n_HeIII+TS.n_HeIII, n_e+TS.n_e );
  }

  __host__ __device__ Thermal_State operator-( const Thermal_State &TS ){
    return Thermal_State( U-TS.U,  n_HI-TS.n_HI, n_HII-TS.n_HII, n_HeI-TS.n_HeI, n_HeII-TS.n_HeII, n_HeIII-TS.n_HeIII, n_e-TS.n_e );
  }

  __host__ __device__ Thermal_State operator*( const Real &x ){
    return Thermal_State( U*x, n_HI*x,  n_HII*x, n_HeI*x,  n_HeII*x, n_HeIII*x,  n_e*x  );
  }
  
  __host__ __device__ Real get_MMW( ){
    Real n_mass = n_HI + n_HII + 4 * ( n_HeI + n_HeII + n_HeIII );
    Real n_tot =  n_HI + n_HII + n_HeI + n_HeII + n_HeIII + n_e;  
    return n_mass / n_tot;
  }
  
  __host__ __device__ Real get_temperature(){
    Real mu, temp;
    mu = get_MMW();
    temp = (gamma - 1) * mu * U * MP / KB;
    return temp;
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




__device__ void get_current_uvb_rates_textures( Real current_z, int n_uvb_rates_samples, float *rates_z, float &photo_i_HI, float &photo_h_HI, float &photo_i_HeI, float &photo_h_HeI, float &photo_i_HeII, float &photo_h_HeII ){
  
  
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
  
  Real z_l, delta_z, x;
  
  z_l = rates_z[indx_l];
  if ( indx_l < n_uvb_rates_samples - 1 ){
    delta_z = rates_z[indx_l+1] - z_l;
  }
  else{
    delta_z = 1;
  }
  
  // printf( "%f  %f %f  %f\n", current_z, rates_z[indx_l], rates_z[indx_l+1], delta_z );
  x = (float)indx_l + ( current_z - z_l ) / delta_z; 
  x += 0.5;
  photo_h_HI   = tex1D( tex_Heat_rate_HI, x );
  photo_h_HeI  = tex1D( tex_Heat_rate_HeI, x );
  photo_h_HeII = tex1D( tex_Heat_rate_HeII, x );
  photo_i_HI   = tex1D( tex_Ion_rate_HI, x );
  photo_i_HeI  = tex1D( tex_Ion_rate_HeI, x );
  photo_i_HeII = tex1D( tex_Ion_rate_HeII, x );
  // printf("%f  %f  %e  %e  %e  %e  %e  %e \n ", current_z, z_l, photo_i_HI, photo_h_HI, photo_i_HeI, photo_h_HeI, photo_i_HeII, photo_h_HeII );
    
                     
}


__device__ Real linear_interpolation( Real delta_x, int indx_l, int indx_r, float*array ){
  float v_l, v_r; 
  Real v;
  v_l = array[indx_l];
  v_r = array[indx_r];
  v = delta_x * ( v_r - v_l ) + v_l;
  return v; 
}


__device__ void get_current_uvb_rates_arrays( Real current_z, int n_uvb_rates_samples, float *rates_z, 
  float *arr_i_HI, float *arr_i_HeI, float *arr_i_HeII, float *arr_h_HI, float *arr_h_HeI, float *arr_h_HeII,
  float &photo_i_HI, float &photo_h_HI, float &photo_i_HeI, float &photo_h_HeI, float &photo_i_HeII, float &photo_h_HeII, bool print ){
  
  
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
  
  Real z_l, z_r, delta_x;
  
  z_l = rates_z[indx_l];
  z_r = rates_z[indx_l+1];
  delta_x = (current_z - z_l) / ( z_r - z_l );
  
  photo_i_HI   = linear_interpolation( delta_x, indx_l, indx_l+1, arr_i_HI );
  photo_i_HeI  = linear_interpolation( delta_x, indx_l, indx_l+1, arr_i_HeI );
  photo_i_HeII = linear_interpolation( delta_x, indx_l, indx_l+1, arr_i_HeII );
  photo_h_HI   = linear_interpolation( delta_x, indx_l, indx_l+1, arr_h_HI );
  photo_h_HeI  = linear_interpolation( delta_x, indx_l, indx_l+1, arr_h_HeI );
  photo_h_HeII = linear_interpolation( delta_x, indx_l, indx_l+1, arr_h_HeII );
  
  
  
  // if ( print ){
  //   printf( " z:%f  n_samples:%d  indx_l:%d  z_l:%f z_r:%f  delta_x:%f \n", current_z, n_uvb_rates_samples, indx_l, z_l, rates_z[indx_l+1], delta_x   );  
  // }
  // 

}
  
                                       
// 
// __host__ __device__ Real Get_Chemistry_dt( Thermal_State &TS, Thermal_State &TS_derivs ){
//   Real dt_min = 1e100;
// 
//   dt_min = fmin( dt_min, TS.T / fabs(TS_derivs.T) );  
//   dt_min = fmin( dt_min, TS.n_HI / fabs(TS_derivs.n_HI) );
//   dt_min = fmin( dt_min, TS.n_HII / fabs(TS_derivs.n_HII) );  
//   dt_min = fmin( dt_min, TS.n_HeI / fabs(TS_derivs.n_HeI) );
//   dt_min = fmin( dt_min, TS.n_HeII / fabs(TS_derivs.n_HeII) );
//   dt_min = fmin( dt_min, TS.n_HeIII / fabs(TS_derivs.n_HeIII) );
//   dt_min = fmin( dt_min, TS.n_e / fabs(TS_derivs.n_e) );
// 
//   Real alpha = 0.1;
//   return dt_min * alpha;  
// }
// 
// 

// __device__ void Add_Expantion_Derivatives( Thermal_State &TS, Thermal_State &TS_derivs,Real dt, Real current_a, Real H0, Real Omega_M, Real Omega_L ){
// 
//   Real H;
//   H = H0 * sqrt( Omega_M/pow(current_a,3) + Omega_L ) / ( 1000 * KPC ) * dt;
//   TS_derivs.T += -2 * H * TS.T;
//   TS_derivs.n_HI += -3 * H * TS.n_HI;
//   TS_derivs.n_HII +=  -3 * H * TS.n_HII;
//   TS_derivs.n_HeI += -3 * H * TS.n_HeI;
//   TS_derivs.n_HeII += -3 * H * TS.n_HeII;
//   TS_derivs.n_HeIII += -3 * H * TS.n_HeIII;
//   TS_derivs.n_e += -3 * H * TS.n_e;
// 
// } 
// 

// 
// __device__ void Get_Chemistry_Derivatives( Thermal_State &TS, Thermal_State &TS_derivs, Real dens_gas, Real current_a, Real H0, Real Omega_M, Real Omega_L, int n_uvb_rates_samples, float *rates_z, bool print ){
// 
//   float photo_h_HI, photo_h_HeI, photo_h_HeII; 
//   float photo_i_HI, photo_i_HeI, photo_i_HeII;  
// 
//   // Real H;
//   Real current_z, n_tot;
//   Real Q_cool_rec_HII, Q_cool_rec_HeII, Q_cool_rec_HeII_d, Q_cool_rec_HeIII, dQ_dt_cool_recomb;
//   Real Q_cool_collis_ext_HI, Q_cool_collis_ext_HeII;
//   Real Q_cool_collis_ion_HI, Q_cool_collis_ion_HeI, Q_cool_collis_ion_HeII, dQ_dt_cool_collisional; 
//   Real dn_dt_HI, dn_dt_HII, dn_dt_HeI, dn_dt_HeII, dn_dt_HeIII, dn_dt_e;
//   Real dQ_dt_phot, dQ_dt_brem, dQ_dt_CMB, dQ_dt;
//   Real dn_dt_photo_HI, dn_dt_photo_HeI, dn_dt_photo_HeII; 
//   Real dn_dt_recomb_HII, dn_dt_recomb_HeII, dn_dt_recomb_HeII_d, dn_dt_recomb_HeIII; 
//   Real dn_dt_coll_HI, dn_dt_coll_HeI, dn_dt_coll_HeII,  dn_dt_coll_HI_HI, dn_dt_coll_HII_HI,  dn_dt_coll_HeI_HI;
//   Real dn_dt_HI_p, dn_dt_HII_p, dn_dt_HeI_p, dn_dt_HeII_p, dn_dt_HeIII_p, dn_dt_e_p;
//   Real dn_dt_HI_m, dn_dt_HII_m, dn_dt_HeI_m, dn_dt_HeII_m, dn_dt_HeIII_m, dn_dt_e_m;  
//   Real dx_dt, x_sum;
// 
//   current_z = 1/current_a - 1;
//   // H = H0 * sqrt( Omega_M/pow(current_a,3) + Omega_L ) / ( 1000 * KPC );
// 
//   //Get Photoheating amd Photoionization rates as z=current_z
//   get_current_uvb_rates( current_z, n_uvb_rates_samples, rates_z, photo_i_HI, photo_h_HI, photo_i_HeI, photo_h_HeI, photo_i_HeII, photo_h_HeII );
// 
//   // Total Photoheating Rate
//   dQ_dt_phot = TS.n_HI * photo_h_HI + TS.n_HeI * photo_h_HeI + TS.n_HeII * photo_h_HeII;  
// 
//   // Recombination Cooling
//   Q_cool_rec_HII    = Cooling_Rate_Recombination_HII_Hui97( TS.n_e, TS.n_HII, TS.T );
//   Q_cool_rec_HeII   = Cooling_Rate_Recombination_HeII_Hui97( TS.n_e, TS.n_HeII, TS.T );
//   Q_cool_rec_HeII_d = Cooling_Rate_Recombination_dielectronic_HeII_Hui97( TS.n_e, TS.n_HeII, TS.T );
//   Q_cool_rec_HeIII  = Cooling_Rate_Recombination_HeIII_Hui97( TS.n_e, TS.n_HeIII, TS.T );
//   dQ_dt_cool_recomb = Q_cool_rec_HII + Q_cool_rec_HeII + Q_cool_rec_HeII_d + Q_cool_rec_HeIII;
// 
//   // Collisional Cooling
//   Q_cool_collis_ext_HI = Cooling_Rate_Collisional_Excitation_e_HI_Hui97( TS.n_e, TS.n_HI, TS.T ); 
//   Q_cool_collis_ext_HeII = Cooling_Rate_Collisional_Excitation_e_HeII_Hui97( TS.n_e, TS.n_HeII, TS.T );
//   Q_cool_collis_ion_HI = Cooling_Rate_Collisional_Ionization_e_HI_Katz95( TS.n_e, TS.n_HI, TS.T );
//   Q_cool_collis_ion_HeI = Cooling_Rate_Collisional_Ionization_e_HeI_Katz95( TS.n_e, TS.n_HeI, TS.T );
//   Q_cool_collis_ion_HeII = Cooling_Rate_Collisional_Ionization_e_HeII_Katz95( TS.n_e, TS.n_HeII, TS.T );
//   dQ_dt_cool_collisional = Q_cool_collis_ext_HI + Q_cool_collis_ext_HeII + Q_cool_collis_ion_HI + Q_cool_collis_ion_HeI + Q_cool_collis_ion_HeII;
// 
//   // Cooling Bremsstrahlung 
//   dQ_dt_brem = Cooling_Rate_Bremsstrahlung_Katz95( TS.n_e, TS.n_HII, TS.n_HeII, TS.n_HeIII, TS.T );
// 
//   // Compton cooling off the CMB 
//   dQ_dt_CMB = Cooling_Rate_Compton_CMB_MillesOstriker01( TS.n_e, TS.T, current_z );
//   // # dQ_dt_CMB = Cooling_Rate_Compton_CMB_Katz95( n_e, temp, current_z ) 
// 
//   // Net Heating and Cooling Rates 
//   dQ_dt  = dQ_dt_phot - dQ_dt_cool_recomb - dQ_dt_cool_collisional - dQ_dt_brem - dQ_dt_CMB;
//   // dQ_dt  = dQ_dt_phot;
//   // dQ_dt  = 0;  
// 
// 
//   // Photoionization Rates
//   dn_dt_photo_HI   = TS.n_HI  *photo_i_HI;
//   dn_dt_photo_HeI  = TS.n_HeI *photo_i_HeI;
//   dn_dt_photo_HeII = TS.n_HeII*photo_i_HeII;
// 
//   // Recombination Rates 
//   dn_dt_recomb_HII    = Recombination_Rate_HII_Hui97( TS.T ) * TS.n_HII * TS.n_e;  
//   dn_dt_recomb_HeII   = Recombination_Rate_HeII_Hui97( TS.T ) * TS.n_HeII * TS.n_e;
//   dn_dt_recomb_HeIII  = Recombination_Rate_HeIII_Katz95( TS.T ) * TS.n_HeIII * TS.n_e;
//   dn_dt_recomb_HeII_d = Recombination_Rate_dielectronic_HeII_Hui97( TS.T ) * TS.n_HeII * TS.n_e;
// 
// 
//   // Collisional Ionization  Rates
//   dn_dt_coll_HI     = Collisional_Ionization_Rate_e_HI_Hui97( TS.T )       * TS.n_HI   * TS.n_e; 
//   dn_dt_coll_HeI    = Collisional_Ionization_Rate_e_HeI_Abel97( TS.T )     * TS.n_HeI  * TS.n_e;
//   dn_dt_coll_HeII   = Collisional_Ionization_Rate_e_HeII_Abel97( TS.T )    * TS.n_HeII * TS.n_e;
//   dn_dt_coll_HI_HI  = Collisional_Ionization_Rate_HI_HI_Lenzuni91( TS.T )  * TS.n_HI   * TS.n_HI;
//   dn_dt_coll_HII_HI = Collisional_Ionization_Rate_HII_HI_Lenzuni91( TS.T ) * TS.n_HII  * TS.n_HI;
//   dn_dt_coll_HeI_HI = Collisional_Ionization_Rate_HeI_HI_Lenzuni91( TS.T ) * TS.n_HeI  * TS.n_HI;
// 
// 
//   dn_dt_HI_p = dn_dt_recomb_HII;
//   dn_dt_HI_m =  dn_dt_photo_HI + dn_dt_coll_HI + dn_dt_coll_HI_HI + dn_dt_coll_HII_HI + dn_dt_coll_HeI_HI;
// 
//   dn_dt_HII_p = dn_dt_photo_HI + dn_dt_coll_HI + dn_dt_coll_HI_HI + dn_dt_coll_HII_HI + dn_dt_coll_HeI_HI;
//   dn_dt_HII_m = dn_dt_recomb_HII; 
// 
//   dn_dt_HeI_p = dn_dt_recomb_HeII + dn_dt_recomb_HeII_d;
//   dn_dt_HeI_m = dn_dt_photo_HeI + dn_dt_coll_HeI;
// 
//   dn_dt_HeII_p = dn_dt_photo_HeI + dn_dt_coll_HeI + dn_dt_recomb_HeIII;
//   dn_dt_HeII_m = dn_dt_photo_HeII + dn_dt_coll_HeII + dn_dt_recomb_HeII + dn_dt_recomb_HeII_d;
// 
//   dn_dt_HeIII_p = dn_dt_photo_HeII + dn_dt_coll_HeII; 
//   dn_dt_HeIII_m =  dn_dt_recomb_HeIII; 
// 
//   dn_dt_e_p = dn_dt_photo_HI + dn_dt_photo_HeI + dn_dt_photo_HeII + dn_dt_coll_HI + dn_dt_coll_HeI + dn_dt_coll_HeII + dn_dt_coll_HI_HI + dn_dt_coll_HII_HI + dn_dt_coll_HeI_HI;
//   dn_dt_e_m = dn_dt_recomb_HII + dn_dt_recomb_HeII + dn_dt_recomb_HeII_d + dn_dt_recomb_HeIII;
// 
// 
//   dn_dt_HI    = dn_dt_HI_p    - dn_dt_HI_m;
//   dn_dt_HII   = dn_dt_HII_p   - dn_dt_HII_m;
//   dn_dt_HeI   = dn_dt_HeI_p   - dn_dt_HeI_m;
//   dn_dt_HeII  = dn_dt_HeII_p  - dn_dt_HeII_m;
//   dn_dt_HeIII = dn_dt_HeIII_p - dn_dt_HeIII_m;
//   dn_dt_e     = dn_dt_e_p     - dn_dt_e_m;
// 
//   n_tot = TS.n_HI + TS.n_HII + TS.n_HeI + TS.n_HeII + TS.n_HeIII + TS.n_e;  
// 
//   // Compute the change in number of particles 
//   dx_dt = ( dn_dt_HI + dn_dt_HII + dn_dt_HeI + dn_dt_HeII + dn_dt_HeIII + dn_dt_e  ) / dens_gas * MP;
// 
//   // The total number of particles:
//   x_sum =  n_tot / dens_gas * MP;
// 
//   if ( print ){
//     printf( "Rates HI:    %e   %e \n",  photo_i_HI, photo_h_HI );
//     printf( "Rates HeI:   %e   %e \n",  photo_i_HeI, photo_h_HeI );
//     printf( "Rates HeII:  %e   %e \n",  photo_i_HeII, photo_h_HeII );
//     printf( "Phot Heat:   %e  \n", dQ_dt_phot);
//     printf( "Net Heat:    %e  \n", dQ_dt );
//     printf( "N_tot:    %e  \n", n_tot);
//     printf( "dT_dt:    %e  \n", 2/(3*KB*n_tot)*dQ_dt);      
//   } 
// 
//   // TS_derivs.T       =  2/(3*KB*n_tot)*dQ_dt -  TS.T/x_sum*dx_dt;
//   TS_derivs.T       =  2/(3*KB*n_tot)*dQ_dt;
//   TS_derivs.n_HI    =  dn_dt_HI;
//   TS_derivs.n_HII   =  dn_dt_HII;
//   TS_derivs.n_HeI   =  dn_dt_HeI;
//   TS_derivs.n_HeII  =  dn_dt_HeII;
//   TS_derivs.n_HeIII =  dn_dt_HeIII;
//   TS_derivs.n_e     =  dn_dt_e;
// 
// } 

__device__ Real Update_BDF( Real n, Real C, Real D, Real dt ){
  return ( n + C * dt ) / ( 1 + D * dt );
}

__device__ Real Get_dt_min( Real n, Real C, Real D, Real alpha ){
  Real dt = n * ( alpha - 1 ) / ( C - alpha * n * D );
  return dt;
}
  

__device__ Real Step_Update_BDF( Thermal_State &TS, Real rho, Real current_a, Real H0, Real Omega_M, Real Omega_L, Real t_chem, Real dt_hydro, int n_uvb_rates_samples, float *rates_z,
                                 float *photo_h_HI_rates, float *photo_h_HeI_rates,float *photo_h_HeII_rates,
                                 float *photo_i_HI_rates, float *photo_i_HeI_rates,float *photo_i_HeII_rates, bool print ){
  
  float photo_h_HI, photo_h_HeI, photo_h_HeII; 
  float photo_i_HI, photo_i_HeI, photo_i_HeII;  
  Real current_z, temp, C, D, dt, delta_a;
  Real dQ_dt_phot, dQ_dt_brem, dQ_dt_CMB, dQ_dt;
  Real Q_cool_rec_HII, Q_cool_rec_HeII, Q_cool_rec_HeII_d, Q_cool_rec_HeIII, dQ_dt_cool_recomb;
  Real Q_cool_collis_ext_HI, Q_cool_collis_ext_HeII, Q_cool_collis_ext_HeI;
  Real Q_cool_collis_ion_HI, Q_cool_collis_ion_HeI, Q_cool_collis_ion_HeII, dQ_dt_cool_collisional, Q_cool_collis_ion_HeIS; 
  Real recomb_HII, recomb_HeII, recomb_HeIII, recomb_HeII_d;
  Real coll_HI, coll_HeI, coll_HeII, coll_HI_HI, coll_HII_HI, coll_HeI_HI;
  Real dt_min_U, dt_min_HI, dt_min_e, dt_min_HeI, alpha_dt, n_min;
  Real init_nHI, init_nHeI;
  
  bool use_case_B = false;
  double units = 1;
  n_min = 1e-20;
   
  // Compute the resdrift at current_a
  current_z = 1/(current_a) - 1;
  
    
  //Get Photoheating amd Photoionization rates as z=current_z
  get_current_uvb_rates_arrays( current_z, n_uvb_rates_samples, rates_z, 
    photo_i_HI_rates, photo_i_HeI_rates, photo_i_HeII_rates, photo_h_HI_rates, photo_h_HeI_rates, photo_h_HeII_rates,
    photo_i_HI, photo_h_HI, photo_i_HeI, photo_h_HeI, photo_i_HeII, photo_h_HeII,   print );
    
  // get_current_uvb_rates_textures( current_z, n_uvb_rates_samples, rates_z, photo_i_HI, photo_h_HI, photo_i_HeI, photo_h_HeI, photo_i_HeII, photo_h_HeII );
  
  
  temp = TS.get_temperature();
  
  // Total Photoheating Rate
  dQ_dt_phot = TS.n_HI * photo_h_HI + TS.n_HeI * photo_h_HeI + TS.n_HeII * photo_h_HeII;  
  
  
  // Recombination Cooling
  // Q_cool_rec_HII    = Cooling_Rate_Recombination_HII_Hui97( TS.n_e, TS.n_HII, temp );
  // Q_cool_rec_HeII   = Cooling_Rate_Recombination_HeII_Hui97( TS.n_e, TS.n_HeII, temp );
  // Q_cool_rec_HeII_d = Cooling_Rate_Recombination_dielectronic_HeII_Hui97( TS.n_e, TS.n_HeII, temp );
  // Q_cool_rec_HeIII  = Cooling_Rate_Recombination_HeIII_Hui97( TS.n_e, TS.n_HeIII, temp );
  Q_cool_rec_HII    = cool_reHII_rate( temp, units, use_case_B )   * TS.n_HII * TS.n_e; 
  Q_cool_rec_HeII   = cool_reHeII1_rate( temp, units, use_case_B ) * TS.n_HeII * TS.n_e;
  Q_cool_rec_HeII_d = cool_reHeII2_rate( temp, units )             * TS.n_HeII * TS.n_e;
  Q_cool_rec_HeIII  = cool_reHeIII_rate( temp, units, use_case_B ) * TS.n_HeIII * TS.n_e;
  dQ_dt_cool_recomb = Q_cool_rec_HII + Q_cool_rec_HeII + Q_cool_rec_HeII_d + Q_cool_rec_HeIII;
  
  // Collisional Cooling
  // Q_cool_collis_ext_HI   = Cooling_Rate_Collisional_Excitation_e_HI_Hui97( TS.n_e, TS.n_HI, temp ); 
  // Q_cool_collis_ext_HeII = Cooling_Rate_Collisional_Excitation_e_HeII_Hui97( TS.n_e, TS.n_HeII, temp );
  // Q_cool_collis_ion_HI   = Cooling_Rate_Collisional_Ionization_e_HI_Katz95( TS.n_e, TS.n_HI, temp );
  // Q_cool_collis_ion_HeI  = Cooling_Rate_Collisional_Ionization_e_HeI_Katz95( TS.n_e, TS.n_HeI, temp );
  // Q_cool_collis_ion_HeII = Cooling_Rate_Collisional_Ionization_e_HeII_Katz95( TS.n_e, TS.n_HeII, temp );
  Q_cool_collis_ext_HI   = cool_ceHI_rate( temp, units )    * TS.n_HI * TS.n_e;
  Q_cool_collis_ext_HeI  = cool_ceHeI_rate( temp, units )   * TS.n_HeI * TS.n_e;
  Q_cool_collis_ext_HeII = cool_ceHeII_rate( temp, units )  * TS.n_HeII * TS.n_e;
  Q_cool_collis_ion_HI   = cool_ciHI_rate( temp, units )    * TS.n_HI * TS.n_e;
  Q_cool_collis_ion_HeI  = cool_ciHeI_rate( temp, units )   * TS.n_HeI * TS.n_e;
  Q_cool_collis_ion_HeIS  = cool_ciHeIS_rate( temp, units ) * TS.n_HeI * TS.n_e;
  Q_cool_collis_ion_HeII = cool_ciHeII_rate( temp, units )  * TS.n_HeII * TS.n_e;
  dQ_dt_cool_collisional = Q_cool_collis_ext_HI + Q_cool_collis_ext_HeI + Q_cool_collis_ext_HeII 
                         + Q_cool_collis_ion_HI + Q_cool_collis_ion_HeI + Q_cool_collis_ion_HeIS + Q_cool_collis_ion_HeII;
 
  // Cooling Bremsstrahlung 
  // dQ_dt_brem = Cooling_Rate_Bremsstrahlung_Katz95( TS.n_e, TS.n_HII, TS.n_HeII, TS.n_HeIII, temp );
  dQ_dt_brem = cool_brem_rate( temp, units ) * TS.n_e* TS.n_HII *  TS.n_HeII * TS.n_HeIII;
  
  // Compton cooling off the CMB 
  dQ_dt_CMB = Cooling_Rate_Compton_CMB_MillesOstriker01( TS.n_e, temp, current_z );
  // dQ_dt_CMB = Cooling_Rate_Compton_CMB_Katz95( TS.n_e, temp, current_z ); 
  
  // Net Heating and Cooling Rates 
  dQ_dt  = dQ_dt_phot - dQ_dt_cool_recomb - dQ_dt_cool_collisional - dQ_dt_brem - dQ_dt_CMB;
  
  // Recombination Rates 
  // // recomb_HII    = Recombination_Rate_HII_Hui97( temp );
  // // recomb_HeII   = Recombination_Rate_HeII_Hui97( temp );
  // // recomb_HII    = Recombination_Rate_HII_Katz95( temp );
  // recomb_HII    = Recombination_Rate_HII_Abel97( temp );
  // recomb_HeII   = Recombination_Rate_HeII_Katz95( temp );  
  // recomb_HeIII  = Recombination_Rate_HeIII_Katz95( temp );
  // recomb_HeII_d = Recombination_Rate_dielectronic_HeII_Hui97( temp );
  recomb_HII    = recomb_HII_rate( temp, units, use_case_B );
  recomb_HeII   = recomb_HeII_rate( temp, units, use_case_B );
  recomb_HeIII  = recomb_HeIII_rate( temp, units, use_case_B );
  recomb_HeII_d = Recombination_Rate_dielectronic_HeII_Hui97( temp );;
    
  // Collisional Ionization  Rates
  // coll_HI     = Collisional_Ionization_Rate_e_HI_Hui97( temp ); 
  // coll_HeI    = Collisional_Ionization_Rate_e_HeI_Abel97( temp );   
  // coll_HeII   = Collisional_Ionization_Rate_e_HeII_Abel97( temp );  
  // coll_HI_HI  = Collisional_Ionization_Rate_HI_HI_Lenzuni91( temp ); 
  // coll_HII_HI = Collisional_Ionization_Rate_HII_HI_Lenzuni91( temp ); 
  // coll_HeI_HI = Collisional_Ionization_Rate_HeI_HI_Lenzuni91( temp );
  coll_HI     = coll_i_HI_rate( temp, units );
  coll_HeI    = coll_i_HeI_rate( temp, units );
  coll_HeII   = coll_i_HeII_rate( temp, units );
  coll_HI_HI  = coll_i_HI_HI_rate( temp, units );
  coll_HII_HI = 0;
  coll_HeI_HI = coll_i_HI_HeI_rate( temp, units );
  
  // Compute delta_t for the time step
  alpha_dt = 0.1;
  dt_min_U = TS.U / fabs(dQ_dt) * rho * alpha_dt;
  // Electron creation and destruction rates
  C = photo_i_HI*TS.n_HI + photo_i_HeI*TS.n_HeI + photo_i_HeII*TS.n_HeII +
      coll_HI*TS.n_HI*TS.n_e + coll_HeI*TS.n_HeI*TS.n_e + coll_HeII*TS.n_HeII*TS.n_e + coll_HI_HI*TS.n_HI*TS.n_HI * coll_HII_HI*TS.n_HII*TS.n_HI * coll_HeI_HI*TS.n_HeI*TS.n_HI;
  D = recomb_HII*TS.n_HII + recomb_HeII*TS.n_HeII + recomb_HeII_d*TS.n_HeII + recomb_HeIII*TS.n_HeIII;
  // dt_min_e = Get_dt_min( TS.n_e, C, D, alpha_dt );
  dt_min_e = alpha_dt * TS.n_e / fabs( C - D * TS.n_e );
  
  // HI creation and destruction rates
  C = recomb_HII*TS.n_HII*TS.n_e;
  D = photo_i_HI + coll_HI*TS.n_e + coll_HI_HI*TS.n_HI + coll_HII_HI*TS.n_HII + coll_HeI_HI*TS.n_HeI;
  // dt_min_HI = Get_dt_min( TS.n_HI, C, D, alpha_dt );    
  dt_min_HI = alpha_dt * TS.n_HI / fabs( C - D * TS.n_HI );
  
  // HeI creation and destruction rates
  C = recomb_HeII*TS.n_HeII*TS.n_e + recomb_HeII_d*TS.n_HeII*TS.n_e;
  D = photo_i_HeI + coll_HeI*TS.n_e;
  // dt_min_HeI = Get_dt_min( TS.n_HeI, C, D, alpha_dt );    
  dt_min_HeI = alpha_dt * TS.n_HeI / fabs( C - D * TS.n_HeI );
  
  // Get the minimum dt
  dt = fmin( dt_min_U, dt_min_HI );
  // dt = fmin( dt_min_HeI, dt );
  dt = fmin( dt_min_e, dt );
  // dt = fmax( dt, dt_hydro/10000 );
  if ( t_chem + dt > dt_hydro ) dt = dt_hydro - t_chem;
  if ( dt < 0 ) printf( "Chem_GPU: Negative dt: %e \n", dt );
  
  
  // // 0. Update internal energy
  // TS.U += dQ_dt * dt / rho ;
  // temp = TS.get_temperature();
  // 
  // recomb_HII    = Recombination_Rate_HII_Abel97( temp );
  // recomb_HeII   = Recombination_Rate_HeII_Katz95( temp );  
  // recomb_HeIII  = Recombination_Rate_HeIII_Katz95( temp );
  // recomb_HeII_d = Recombination_Rate_dielectronic_HeII_Hui97( temp );
  // 
  // // Collisional Ionization  Rates
  // coll_HI     = Collisional_Ionization_Rate_e_HI_Hui97( temp ); 
  // coll_HeI    = Collisional_Ionization_Rate_e_HeI_Abel97( temp );   
  // coll_HeII   = Collisional_Ionization_Rate_e_HeII_Abel97( temp );  
  // coll_HI_HI  = Collisional_Ionization_Rate_HI_HI_Lenzuni91( temp );  
  // coll_HII_HI = Collisional_Ionization_Rate_HII_HI_Lenzuni91( temp ); 
  // coll_HeI_HI = Collisional_Ionization_Rate_HeI_HI_Lenzuni91( temp );
  // 
  
  // // Compute the resdrift at the end of the step
  // delta_a = H0 * sqrt( Omega_M/current_a + Omega_L*pow(current_a, 2) ) / ( 1000 * KPC ) * dt;
  // current_z = 1/(current_a+delta_a) - 1;
  // 
  // //Get Photoheating amd Photoionization rates at z=current_z
  // get_current_uvb_rates( current_z, n_uvb_rates_samples, rates_z, photo_i_HI, photo_h_HI, photo_i_HeI, photo_h_HeI, photo_i_HeII, photo_h_HeII );
  

  // 1. Update HI
  C = recomb_HII*TS.n_HII*TS.n_e;
  D = photo_i_HI + coll_HI*TS.n_e + coll_HI_HI*TS.n_HI + coll_HII_HI*TS.n_HII + coll_HeI_HI*TS.n_HeI;
  init_nHI = TS.n_HI;    
  TS.n_HI = Update_BDF( TS.n_HI, C, D, dt );
  if ( TS.n_HI < n_min ) TS.n_HI = n_min;
  
  // 1. Update HII
  C = photo_i_HI*TS.n_HI + coll_HI*TS.n_HI*TS.n_e + coll_HI_HI*TS.n_HI*TS.n_HI + coll_HII_HI*TS.n_HII*TS.n_HI + coll_HeI_HI*TS.n_HeI*TS.n_HI;
  D = recomb_HII*TS.n_e;  
  TS.n_HII = Update_BDF( TS.n_HII, C, D, dt );
  // TS.n_HII += init_nHI - TS.n_HI;
  if ( TS.n_HII < n_min ) TS.n_HII = n_min;
  
  // 3. Update electron
  C = photo_i_HI*TS.n_HI + photo_i_HeI*TS.n_HeI + photo_i_HeII*TS.n_HeII +
      coll_HI*TS.n_HI*TS.n_e + coll_HeI*TS.n_HeI*TS.n_e + coll_HeII*TS.n_HeII*TS.n_e + coll_HI_HI*TS.n_HI*TS.n_HI * coll_HII_HI*TS.n_HII*TS.n_HI * coll_HeI_HI*TS.n_HeI*TS.n_HI;
  D= recomb_HII*TS.n_HII + recomb_HeII*TS.n_HeII + recomb_HeII_d*TS.n_HeII + recomb_HeIII*TS.n_HeIII;
  TS.n_e = Update_BDF( TS.n_e, C, D, dt );
  // TS.n_e = TS.n_HII + TS.n_HeII + TS.n_HeIII; 
  if ( TS.n_e < n_min ) TS.n_e = n_min;
  
  // 4. Update HeI
  C = recomb_HeII*TS.n_HeII*TS.n_e + recomb_HeII_d*TS.n_HeII*TS.n_e;
  D = photo_i_HeI + coll_HeI*TS.n_e;
  TS.n_HeI = Update_BDF( TS.n_HeI, C, D, dt );
  if ( TS.n_HeI < n_min ) TS.n_HeI = n_min;
  
  // 5. Update HeII
  C = photo_i_HeI*TS.n_HeI + coll_HeI*TS.n_HeI*TS.n_e + recomb_HeIII*TS.n_HeIII*TS.n_e;
  D = photo_i_HeII + coll_HeII*TS.n_e + recomb_HeII*TS.n_e + recomb_HeII_d*TS.n_e;
  TS.n_HeII = Update_BDF( TS.n_HeII, C, D, dt );
  if ( TS.n_HeII < n_min ) TS.n_HeII = n_min;
  
  // 6. Update HeIII
  C = photo_i_HeII*TS.n_HeII + coll_HeII*TS.n_HeII*TS.n_e;
  D = recomb_HeIII*TS.n_e;
  TS.n_HeIII = Update_BDF( TS.n_HeIII, C, D, dt );
  if ( TS.n_HeIII < n_min ) TS.n_HeIII = n_min;
    
  // 7. Update internal energy
  TS.U += dQ_dt * dt / rho ;
  
  // if ( TS.n_HI < 0 )    printf( "Chem_GPU: Negative n_HI: %e \n", TS.n_HI );
  // if ( TS.n_HII < 0 )   printf( "Chem_GPU: Negative n_HII: %e \n", TS.n_HII );
  // if ( TS.n_HeI < 0 )   printf( "Chem_GPU: Negative n_HeI: %e \n", TS.n_HeI );
  // if ( TS.n_HeII < 0 )  printf( "Chem_GPU: Negative n_HeII: %e \n", TS.n_HeII );
  // if ( TS.n_HeIII < 0 ) printf( "Chem_GPU: Negative n_HeIII: %e \n", TS.n_HeIII );
  // if ( TS.n_e < 0 )     printf( "Chem_GPU: Negative n_e: %e \n", TS.n_e );
  if ( rho < 0 )     printf( "Chem_GPU: Negative rho: %e \n", rho );
  if ( TS.U < 0 )     printf( "Chem_GPU: Negative U: %e \n", TS.U );
  
  return dt;

}


__global__ void Update_Chemistry( Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt_hydro, Real gamma,  Real density_conv, Real energy_conv,
                                  Real current_z, float* cosmo_params, int n_uvb_rates_samples, float *rates_z, 
                                  float *photo_h_HI_rates, float *photo_h_HeI_rates,float *photo_h_HeII_rates,
                                  float *photo_i_HI_rates, float *photo_i_HeI_rates,float *photo_i_HeII_rates   ){
  
    
  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz;
  Real GE, dt_chem, t_chem;
  Real H0, Omega_M, Omega_L;
  Real current_a, delta_a, a3, a2;
  
  n_cells = nx*ny*nz;
  
  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;
  bool print;
  
  // int MAX_ITER = 500;
  
  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    #ifdef DE
    GE = dev_conserved[(n_fields-1)*n_cells + id];
    #else 
    GE  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz));
    #endif
  
    if ( xid == n_ghost && yid == n_ghost && zid == n_ghost ) print = true;
    else print = false;
    
    // Convert to cgs units
    current_a = 1 / ( current_z + 1);
    a2 = current_a * current_a;
    a3 = a2 * current_a;  
    d  *= density_conv / a3;
    GE *= energy_conv / a2; 
    
    // Load Cosmological Parameters
    H0 = cosmo_params[0];
    Omega_M = cosmo_params[1];
    Omega_L = cosmo_params[2];
    
    dt_hydro = dt_hydro * current_a * current_a / H0 * 1000 * KPC;
    delta_a = H0 * sqrt( Omega_M/current_a + Omega_L*pow(current_a, 2) ) / ( 1000 * KPC ) * dt_hydro;
    
    Thermal_State TS;    
    TS.gamma   = gamma;
    TS.n_HI    = dev_conserved[ 5*n_cells + id] * density_conv / a3 / MP; 
    TS.n_HII   = dev_conserved[ 6*n_cells + id] * density_conv / a3 / MP; 
    TS.n_HeI   = dev_conserved[ 7*n_cells + id] * density_conv / a3 / MP / 4; 
    TS.n_HeII  = dev_conserved[ 8*n_cells + id] * density_conv / a3 / MP / 4; 
    TS.n_HeIII = dev_conserved[ 9*n_cells + id] * density_conv / a3 / MP / 4; 
    TS.n_e     = dev_conserved[10*n_cells + id] * density_conv / a3 / MP; 
    TS.U = GE * d_inv; 
    
    // if ( print ){    
    //   printf(" current_z = %e\n", current_z );
      // printf(" dt = %e \n", dt_hydro  ); 
      // printf(" delta_a = %e \n", delta_a ); 
    //   printf(" H0 = %f\n", H0 );
    //   printf(" Omega_M = %f\n", Omega_M );
    //   printf(" Omega_L = %f\n", Omega_L );
    //   printf(" rho = %e\n", d );
    //   printf(" mu = %f \n", mu );
    // }
    
    // if (print) printf(" T0 = %e  U:%e  mu: %e rho:%e  nHI:%e  nHII:%e   ne:%e    \n",  TS.U, mu,  d, TS.n_HI, TS.n_HII, TS.n_e );
    // Get_Chemistry_Derivatives( TS, TS_derivs, current_a, H0, Omega_M, Omega_L, n_uvb_rates_samples, rates_z, print );
    
      
    t_chem = 0;
    int n_iter = 0;
    while ( t_chem < dt_hydro ){
    
      dt_chem = Step_Update_BDF( TS, d, current_a, H0, Omega_M, Omega_L, t_chem, dt_hydro, n_uvb_rates_samples, rates_z,
                                 photo_i_HI_rates, photo_i_HeI_rates, photo_i_HeII_rates, 
                                 photo_h_HI_rates, photo_h_HeI_rates, photo_h_HeII_rates, print );
      delta_a = H0 * sqrt( Omega_M/current_a + Omega_L*pow(current_a, 2) ) / ( 1000 * KPC ) * dt_chem;
      current_a = current_a + delta_a;
      t_chem += dt_chem;
      n_iter += 1;
      
    }
      
    // while ( t_chem < dt_hydro ){
    // 
    //   Get_Chemistry_Derivatives( TS, TS_derivs, d, current_a, H0, Omega_M, Omega_L, n_uvb_rates_samples, rates_z, false );
    //   dt_chem = Get_Chemistry_dt( TS, TS_derivs );
    // 
    //   if ( dt_chem < dt_hydro/ MAX_ITER ) dt_chem = dt_hydro / MAX_ITER; 
    //   if ( t_chem + dt_chem > dt_hydro ) dt_chem = dt_hydro - t_chem;
    // 
    //   // Add_Expantion_Derivatives( TS, TS_delta_exp, dt_chem, current_a, H0, Omega_M, Omega_L );
    //   TS = TS + (TS_derivs * dt_chem);
    // 
    //   delta_a = H0 * sqrt( Omega_M/current_a + Omega_L*pow(current_a, 2) ) / ( 1000 * KPC ) * dt_chem;
    //   current_a = current_a + delta_a;
    // 
    //   t_chem += dt_chem;
    //   n_iter += 1;
    // }
    
    // // Substract the change due to expansion
    // TS = TS - TS_delta_exp;
    
    
    // if ( print ) printf(" Chemistry: n_iter: %d     \n", n_iter );
    
    // Write the Updated Thermal State
    dev_conserved[ 5*n_cells + id] = TS.n_HI    / density_conv * a3 * MP; 
    dev_conserved[ 6*n_cells + id] = TS.n_HII   / density_conv * a3 * MP; 
    dev_conserved[ 7*n_cells + id] = TS.n_HeI   / density_conv * a3 * MP * 4; 
    dev_conserved[ 8*n_cells + id] = TS.n_HeII  / density_conv * a3 * MP * 4; 
    dev_conserved[ 9*n_cells + id] = TS.n_HeIII / density_conv * a3 * MP * 4; 
    dev_conserved[10*n_cells + id] = TS.n_e     / density_conv * a3 * MP; 
    d = d / density_conv * a3;
    GE = TS.U * d / energy_conv * a2;
    dev_conserved[4*n_cells + id]  = GE + 0.5*d*(vx*vx + vy*vy + vz*vz);  
    #ifdef DE
    dev_conserved[(n_fields-1)*n_cells + id] = GE;
    #endif
    
  
  }
  
}


// Rates from Katz 95
//////////////////////////////////////////////////////////////////////////////////////////////////////


// Colisional excitation of neutral hydrogen (HI) and singly ionized helium (HeII)

__device__ Real Cooling_Rate_Collisional_Excitation_e_HI_Katz95( Real n_e, Real n_HI, Real temp ){
  Real temp_5, Lambda_e_HI;
  temp_5 = temp / 1e5;
  Lambda_e_HI = 7.50e-19 * exp(-118348.0 / temp) * n_e * n_HI/( 1 + pow(temp_5, 0.5) );
  return Lambda_e_HI;
}

__device__ Real Cooling_Rate_Collisional_Excitation_e_HeII_Katz95( Real n_e, Real n_HeII, Real temp ){
  Real temp_5, Lambda_e_HeII;
  temp_5 = temp / 1e5;
  Lambda_e_HeII = 5.54e-17 * pow(temp, -0.397) * exp(-473638.0 / temp) * n_e * n_HeII / ( 1 + pow(temp_5,0.5) );
  return Lambda_e_HeII;
}



// Colisional ionization  of HI, HeI and HeII
__device__ Real  Cooling_Rate_Collisional_Ionization_e_HI_Katz95( Real n_e, Real n_HI, Real temp ){
  Real temp_5, Lambda_e_HI;
  temp_5 = temp / 1e5;
  Lambda_e_HI = 1.27e-21 * pow(temp, 0.5) * exp(-157809.1 / temp) * n_e * n_HI / ( 1 + pow(temp_5, 0.5) );
  return Lambda_e_HI;
}


__device__ Real  Cooling_Rate_Collisional_Ionization_e_HeI_Katz95( Real n_e, Real n_HeI, Real temp ){
  Real temp_5, Lambda_e_HeI;
  temp_5 = temp / 1e5;
  Lambda_e_HeI = 9.38e-22 * pow(temp, 0.5) * exp(-285335.4 / temp) * n_e * n_HeI / ( 1 + pow(temp_5, 0.5) );
  return Lambda_e_HeI;
}

__device__ Real  Cooling_Rate_Collisional_Ionization_e_HeII_Katz95( Real n_e, Real n_HeII, Real temp ){
  Real temp_5, Lambda_e_HeII;
  temp_5 = temp / 1e5;
  Lambda_e_HeII = 4.95e-22 * pow(temp, 0.5) * exp(-631515.0/ temp) * n_e * n_HeII / ( 1 + pow(temp_5, 0.5) );
  return Lambda_e_HeII;
}

__device__ Real  Collisional_Ionization_Rate_e_HI_Katz95( Real temp ){
  Real temp_5, Gamma_e_HI;
  temp_5 = temp / 1e5;
  Gamma_e_HI = 5.85e-11 * pow(temp, 0.5) *  exp(-157809.1 / temp) / ( 1 + pow(temp_5, 0.5) );
  // if Gamma_e_HI < Gamma_min: Gamma_e_HI = Gamma_min
  return Gamma_e_HI;
}

__device__ Real  Collisional_Ionization_Rate_e_HeI_Katz95( Real temp ){
  Real temp_5, Gamma_e_HeI;
  temp_5 = temp / 1e5;
  Gamma_e_HeI = 2.38e-11 * pow(temp, 0.5) * exp(-285335.4 / temp) / ( 1 + pow(temp_5, 0.5) );
  // if Gamma_e_HeI < Gamma_min: Gamma_e_HeI = Gamma_min
  return Gamma_e_HeI;
}

__device__ Real  Collisional_Ionization_Rate_e_HeII_Katz95( Real temp ){
  Real temp_5, Gamma_e_HeII; 
  temp_5 = temp / 1e5;
  Gamma_e_HeII = 5.688e-12 * pow(temp, 0.5) * exp(-631515.0 / temp) / ( 1 + pow(temp_5, 0.5) );
  // if Gamma_e_HeII < Gamma_min: Gamma_e_HeII = Gamma_min
  return Gamma_e_HeII;
}

// Standard Recombination of HII, HeII and HeIII

__device__ Real  Cooling_Rate_Recombination_HII_Katz95( Real n_e, Real n_HII, Real temp ){
  Real temp_3, temp_6, Lambda_HII;
  temp_3 = temp / 1e3;
  temp_6 = temp / 1e6;
  Lambda_HII = 8.70e-27 * pow(temp, 0.5) * pow(temp_3, -0.2) * 1/( 1 + pow(temp_6, 0.7 )) * n_e * n_HII;
  return Lambda_HII;
}

__device__ Real  Cooling_Rate_Recombination_HeII_Katz95( Real n_e, Real n_HeII, Real temp ){
  Real Lambda_HeII;
  Lambda_HeII = 1.55e-26 * pow(temp, 0.3647) * n_e * n_HeII; 
  return Lambda_HeII;
}

__device__ Real  Cooling_Rate_Recombination_HeIII_Katz95( Real n_e, Real n_HeIII, Real temp ){
  Real temp_3, temp_6, Lambda_HeIII;
  temp_3 = temp / 1e3;
  temp_6 = temp / 1e6;
  Lambda_HeIII = 3.48e-26 * pow(temp, 0.5) * pow(temp_3, -0.2) * 1/( 1 + pow(temp_6, 0.7 )) * n_e * n_HeIII; 
  return Lambda_HeIII;
}    


__device__ Real  Recombination_Rate_HII_Katz95( Real temp ){
  Real temp_3, temp_6, alpha_HII;
  temp_3 = temp / 1e3;
  temp_6 = temp / 1e6;
  alpha_HII = 8.4e-11 * pow(temp,-0.5) * pow(temp_3,-0.2) * 1/( 1 + pow(temp_6, 0.7 ) );
  // if alpha_HII < alpha_min: alpha_HII = alpha_min
  return alpha_HII;
}

__device__ Real  Recombination_Rate_HeII_Katz95( Real temp ){
 Real alpha_HeII;
 alpha_HeII = 1.5e-10 * pow(temp, -0.6353);
 // if alpha_HeII < alpha_min: alpha_HeII = alpha_min
 return alpha_HeII;
}

__device__ Real  Recombination_Rate_HeIII_Katz95( Real temp ){
  Real temp_3, temp_6, alpha_HeIII;
  temp_3 = temp / 1e3;
  temp_6 = temp / 1e6;
  alpha_HeIII = 3.36e-10 * pow(temp, -0.5) * pow(temp_3, -0.2) * 1/( 1 + pow(temp_6, 0.7) );
  // if alpha_HeIII < alpha_min: alpha_HeIII = alpha_min
  return alpha_HeIII;
}  


// Dielectronic recombination of HeII
__device__ Real  Cooling_Rate_Recombination_dielectronic_HeII_Katz95( Real n_e, Real n_HeII, Real temp ){
 Real Lambda_d;
 Lambda_d = 1.24e-13 * pow(temp, -1.5) * exp( -470000.0/temp ) * ( 1 + 0.3 * exp( -94000.0/temp ) ) * n_e * n_HeII;
 return Lambda_d;
}

__device__ Real  Recombination_Rate_dielectronic_HeII_Katz95( Real temp ){
 Real alpha_d;
 alpha_d = 1.9e-3 * pow(temp,-1.5) * exp( -470000.0/temp ) * ( 1 + 0.3 * exp( -94000.0/temp ) );
 // if alpha_d < alpha_min: alpha_d = alpha_min;
 return alpha_d;
}

// Free-Free emission (Bremsstrahlung) 
__device__ Real  gaunt_factor( Real log10_T ){
  Real gff;
  gff = 1.1 + 0.34 * exp( -pow(5.5 - log10_T, 2) / 3.0 );
  return gff ;
}

__device__ Real  Cooling_Rate_Bremsstrahlung_Katz95( Real n_e, Real n_HII, Real n_HeII, Real n_HeIII, Real temp ){
  Real gff, Lambda_bmst;
  gff = gaunt_factor( log10(temp) );
  Lambda_bmst = 1.42e-27 * gff * pow(temp, 0.5) * ( n_HII + n_HeII + 4*n_HeIII ) * n_e;
  return Lambda_bmst;
}


// Compton cooling off the CMB 
__device__ Real  Cooling_Rate_Compton_CMB_Katz95( Real n_e, Real temp, Real z ){
  //  Ikeuchi & Ostriker 1986
  return 5.41e-36 * n_e * temp * pow( 1 + z , 4 );
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Colisional excitation of neutral hydrogen (HI) and singly ionized helium (HeII)
__device__ Real Collisional_Ionization_Rate_e_HI_Abel97( Real temp ){
  Real ln_T, k1;
  temp = temp * K_to_eV;
  ln_T = log(temp);
  k1 = exp( -32.71396786 + 13.536556 * ln_T - 5.73932875 * pow(ln_T, 2) + 1.56315498 * pow(ln_T, 3) -
       0.2877056     * pow(ln_T, 4) + 3.48255977e-2 * pow(ln_T, 5) - 2.63197617e-3 * pow(ln_T, 6) +
       1.11954395e-4 * pow(ln_T, 7) - 2.03914985e-6 * pow(ln_T, 8 ) ); // cm^3 s^-1
  return k1;
}

__device__ Real Recombination_Rate_HII_Abel97( Real temp ){
  Real ln_T, k2;
  temp = temp * K_to_eV;
  ln_T = log(temp);
  k2 = exp( -28.6130338 - 0.72411256 * ln_T - 2.02604473e-2 * pow(ln_T, 2) -
      2.38086188e-3 * pow(ln_T, 3) - 3.21260521e-4 * pow(ln_T, 4) - 1.42150291e-5 * pow(ln_T, 5) +
      4.98910892e-6 * pow(ln_T, 6) + 5.75561414e-7 * pow(ln_T, 7) - 1.85676704e-8 * pow(ln_T, 8) -
      3.07113524e-9 * pow(ln_T, 9)  ); // cm 3 s -1 .
  return k2; 
}

__device__ Real Collisional_Ionization_Rate_e_HeI_Abel97( Real temp ){
  Real ln_T, k3;
  temp = temp * K_to_eV;
  ln_T = log(temp);
  k3 = exp( -44.09864886 + 23.91596563 * ln_T - 10.7532302 * pow(ln_T, 2) + 3.05803875 * pow(ln_T, 3) -
      0.56851189 * pow(ln_T, 4) + 6.79539123e-2 * pow(ln_T, 5) - 5.00905610e-3 * pow(ln_T, 6) +
      2.06723616e-4 * pow(ln_T, 7) - 3.64916141e-6 * pow(ln_T, 8 ) ); // cm 3 s -1 .
  return k3;
}
  
__device__ Real Collisional_Ionization_Rate_e_HeII_Abel97( Real temp ){
  Real ln_T, k5;
  temp = temp * K_to_eV;
  ln_T = log(temp);    
  k5 = exp( -68.71040990 + 43.93347633 * ln_T - 18.4806699 * pow(ln_T, 2) + 4.70162649 * pow(ln_T, 3) -
      0.76924663    * pow(ln_T, 4) + 8.113042e-2   * pow(ln_T, 5) - 5.32402063e-3 * pow(ln_T, 6) +
      1.97570531e-4 * pow(ln_T, 7) - 3.16558106e-6 * pow(ln_T, 8) ); // cm 3 s -1 .  
  return k5;
}

__device__ Real Collisional_Ionization_Rate_HI_HI_Lenzuni91( Real temp ){
  Real k;
  k = 1.2e-17 * pow(temp, 1.2) * exp(-157800 / temp );
  return k;
} 

__device__ Real Collisional_Ionization_Rate_HII_HI_Lenzuni91( Real temp ){
  Real k;
  k = 9e-31 * pow(temp, 3);
  return k;
}

__device__ Real Collisional_Ionization_Rate_HeI_HI_Lenzuni91( Real temp ){
  Real k;
  k = 1.75e-17 * pow(temp, 1.3) * exp(-157800 / temp );
  return k;
}



__device__ Real Recombination_Rate_HII_Hui97( Real temp ){
  Real T_thr_HI, lamda_HI, alpha, pow_1;
  T_thr_HI = 157807;
  lamda_HI = 2 * T_thr_HI / temp;
  pow_1 = pow(lamda_HI/0.522, 0.470);
  alpha = 1.269e-13 * pow(lamda_HI, 1.503) / pow( 1 + pow_1, 1.923);
  return alpha;
}

__device__ Real Recombination_Rate_HeII_Hui97( Real temp ){
  Real T_thr_HeI, lamda_HeI, alpha;
  T_thr_HeI = 285335;
  lamda_HeI = 2 * T_thr_HeI / temp;
  alpha = 3e-14 * pow(lamda_HeI, 0.654);
  return alpha;
}

__device__ Real Recombination_Rate_HeIII_Hui97( Real temp ){
  Real T_thr_HeII, lamda_HeII, alpha, pow_1;
  T_thr_HeII = 631515;
  lamda_HeII = 2 * T_thr_HeII / temp;
  pow_1 = pow(lamda_HeII/0.522, 0.470);
  alpha = 2 * 1.269e-13 * pow(lamda_HeII, 1.503) / pow( 1 + pow_1, 1.923);
  return alpha;
}


__device__ Real Cooling_Rate_Recombination_HII_Hui97( Real n_e,  Real n_HII, Real temp ){
  Real T_thr_HI, lambda_HI, lambda_HII, pow_1; 
  T_thr_HI = 157807;
  lambda_HI = 2 * T_thr_HI / temp;
  pow_1 = pow(lambda_HI/0.541, 0.502);
  lambda_HII = 1.778e-29 * temp * pow(lambda_HI, 1.965) / pow( 1 + pow_1, 2.697) * n_e * n_HII; 
  return lambda_HII;
}

__device__ Real Cooling_Rate_Recombination_HeII_Hui97( Real n_e, Real n_HII, Real temp ){
  Real T_thr_HeI, lambda_HeI, lambda_HeII;
  T_thr_HeI = 285335;
  lambda_HeI = 2 * T_thr_HeI / temp;
  lambda_HeII = KB * temp * 3e-14 * pow(lambda_HeI, 0.654) * n_e * n_HII; 
  return lambda_HeII;
}

__device__ Real Cooling_Rate_Recombination_HeIII_Hui97( Real n_e, Real n_HII, Real temp ){
  Real T_thr_HeII, lambda_HeII, lambda, pow_1;
  T_thr_HeII = 631515;
  lambda_HeII = 2 * T_thr_HeII / temp;
  pow_1 = pow(lambda_HeII/0.541, 0.502);
  lambda = 8 * 1.778e-29 * temp * pow(lambda_HeII, 1.965) / pow( 1 + pow_1, 2.697) * n_e * n_HII; 
  return lambda;
}

__device__ Real Recombination_Rate_dielectronic_HeII_Hui97( Real temp ){
  Real T_thr_HeI, T_thr_HeII, lambda_HeI, alpha ;
  T_thr_HeI = 285335;
  T_thr_HeII = 631515;
  lambda_HeI = 2 * T_thr_HeI / temp;
  alpha = 1.9e-3 * pow(temp, -3./2) * exp( -0.75*lambda_HeI/2 ) * ( 1 + 0.3 * exp(-0.15 * lambda_HeI / 2 ) );
  return alpha;
}


__device__ Real Cooling_Rate_Recombination_dielectronic_HeII_Hui97( Real n_e, Real n_HeII, Real temp ){
  Real T_thr_HeI, lambda, lambda_HeI; 
  T_thr_HeI = 285335;
  lambda_HeI = 2 * T_thr_HeI / temp;
  lambda = 0.75 * KB * temp * 1.9e-3 * pow(temp, -3./2) * exp( -0.75*lambda_HeI/2 ) * ( 1 + 0.3 * exp(-0.15 * lambda_HeI / 2 ) ) * n_e * n_HeII;
  return lambda;
}


__device__ Real Collisional_Ionization_Rate_e_HI_Hui97( Real temp ){
  Real T_thr_HI, lambda_HI, k, pow_1;
  T_thr_HI = 157807;
  lambda_HI = 2 * T_thr_HI / temp;
  pow_1 = pow(lambda_HI/0.354, 0.874);
  k = 21.11 * pow(temp,-3/2) * exp(-lambda_HI/2) * pow(lambda_HI, -1.089) / pow( 1 + pow_1, 1.101 );
  return k;
}

__device__ Real Cooling_Rate_Collisional_Excitation_e_HI_Hui97( Real n_e, Real n_HI, Real temp ){
  Real T_thr_HI, lambda_HI, k;
  T_thr_HI = 157807;
  lambda_HI = 2 * T_thr_HI / temp;
  k = 7.5e-19 * exp( -0.75 * lambda_HI / 2 ) / ( 1 + pow(temp/1e5, 0.5) );
  return k * n_e * n_HI;
}

__device__ Real Cooling_Rate_Collisional_Excitation_e_HeII_Hui97( Real n_e, Real n_HeII,  Real temp ){
  Real T_thr_HeII, lambda_HeII, k;
  T_thr_HeII = 631515;
  lambda_HeII = 2 * T_thr_HeII / temp;
  k = 5.54e-17 * pow(1/temp, 0.397) * exp( -0.75 * lambda_HeII / 2 ) / ( 1 + pow(temp/1e5, 0.5 ) );
  return k * n_e * n_HeII;
}

__device__ Real Cooling_Rate_Compton_CMB_MillesOstriker01( Real n_e, Real temp, Real z ){
  // M. Coleman Miller and Eve C. Ostriker 2001 (https://iopscience.iop.org/article/10.1086/323321/fulltext/)
  Real T_3, T_cm_3;
  T_3 = temp / 1e3;
  T_cm_3 = 0; //# Don't know this value
  return 5.6e-33 * n_e * pow( 1 + z , 4) * ( T_3 - T_cm_3 ) ;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Reaction and cooling rates from Grackle

//Kelvin to eV conversion factor
#ifndef tevk
#define tevk 1.1605e4
#endif
//Comparison value
#ifndef dhuge
#define dhuge 1.0e30
#endif
//Small value
#ifndef tiny
#define tiny 1.0e-20
#endif
// Boltzmann's constant
#ifndef kboltz
#define kboltz 1.3806504e-16 //Boltzmann's constant [cm2gs-2K-1] or [ergK-1] 
#endif



// Calculation of k1 (HI + e --> HII + 2e)
// k1_rate
__device__ double coll_i_HI_rate(double T, double units )
{
    double T_ev = T / 11605.0;
    double logT_ev = log(T_ev);

    double k1 = exp( -32.71396786375
                        + 13.53655609057*logT_ev
                        - 5.739328757388*pow(logT_ev, 2)
                        + 1.563154982022*pow(logT_ev, 3)
                        - 0.2877056004391*pow(logT_ev, 4)
                        + 0.03482559773736999*pow(logT_ev, 5)
                        - 0.00263197617559*pow(logT_ev, 6)
                        + 0.0001119543953861*pow(logT_ev, 7)
                        - 2.039149852002e-6*pow(logT_ev, 8)) / units;
    if (T_ev <= 0.8){
        k1 = fmax(tiny, k1); 
    }
    return k1;
}

//Calculation of k3 (HeI + e --> HeII + 2e)
// k3_rate
__device__ double coll_i_HeI_rate(double T, double units )
{
    double T_ev = T / 11605.0;
    double logT_ev = log(T_ev);

    if (T_ev > 0.8){
        return exp( -44.09864886561001
                + 23.91596563469*logT_ev
                - 10.75323019821*pow(logT_ev, 2)
                + 3.058038757198*pow(logT_ev, 3)
                - 0.5685118909884001*pow(logT_ev, 4)
                + 0.06795391233790001*pow(logT_ev, 5)
                - 0.005009056101857001*pow(logT_ev, 6)
                + 0.0002067236157507*pow(logT_ev, 7)
                - 3.649161410833e-6*pow(logT_ev, 8)) / units;
    } else {
        return tiny;
    }
}   

//Calculation of k4 (HeII + e --> HeI + photon)
// k4_rate
__device__ double recomb_HeII_rate(double T, double units, bool use_case_B )
{
    double T_ev = T / 11605.0;
    double logT_ev = log(T_ev);

    double k4;
    //If case B recombination on.
    if (use_case_B){
        return 1.26e-14 * pow(5.7067e5/T, 0.75) / units;
    }

    //If case B recombination off.
    if (T_ev > 0.8){
        return (1.54e-9*(1.0 + 0.3 / exp(8.099328789667/T_ev))
             / (exp(40.49664394833662/T_ev)*pow(T_ev, 1.5))
             + 3.92e-13/pow(T_ev, 0.6353)) / units;
    } else {
        return 3.92e-13/pow(T_ev, 0.6353) / units;
    }
}

//Calculation of k2 (HII + e --> HI + photon)
// k2_rate
__device__ double recomb_HII_rate(double T, double units, bool use_case_B )
{
    if (use_case_B) {
        if (T < 1.0e9) {
            return 4.881357e-6*pow(T, -1.5) \
                * pow((1.0 + 1.14813e2*pow(T, -0.407)), -2.242) / units;
        } else {
            return tiny;
        }  
    } else {
        if (T > 5500) {
            //Convert temperature to appropriate form.
            double T_ev = T / tevk;
            double logT_ev = log(T_ev);

            return exp( -28.61303380689232 \
                - 0.7241125657826851*logT_ev \
                - 0.02026044731984691*pow(logT_ev, 2) \
                - 0.002380861877349834*pow(logT_ev, 3) \
                - 0.0003212605213188796*pow(logT_ev, 4) \
                - 0.00001421502914054107*pow(logT_ev, 5) \
                + 4.989108920299513e-6*pow(logT_ev, 6) \
                + 5.755614137575758e-7*pow(logT_ev, 7) \
                - 1.856767039775261e-8*pow(logT_ev, 8) \
                - 3.071135243196595e-9*pow(logT_ev, 9)) / units;
        } else {
            return recomb_HeII_rate(T, units, use_case_B);
        }
    }
}

//Calculation of k5 (HeII + e --> HeIII + 2e)
// k5_rate
__device__ double coll_i_HeII_rate(double T, double units )
{
    double T_ev = T / 11605.0;
    double logT_ev = log(T_ev);

    double k5;
    if (T_ev > 0.8){
        k5 = exp(-68.71040990212001
                + 43.93347632635*logT_ev
                - 18.48066993568*pow(logT_ev, 2)
                + 4.701626486759002*pow(logT_ev, 3)
                - 0.7692466334492*pow(logT_ev, 4)
                + 0.08113042097303*pow(logT_ev, 5)
                - 0.005324020628287001*pow(logT_ev, 6)
                + 0.0001975705312221*pow(logT_ev, 7)
                - 3.165581065665e-6*pow(logT_ev, 8)) / units;
    } else {
        k5 = tiny;
    }
    return k5;
}

//Calculation of k6 (HeIII + e --> HeII + photon)
// k6_rate
__device__ double recomb_HeIII_rate(double T, double units, bool use_case_B )
{
    double k6;
    //Has case B recombination setting.
    if (use_case_B) {
        if (T < 1.0e9) {
            k6 = 7.8155e-5*pow(T, -1.5)
                * pow((1.0 + 2.0189e2*pow(T, -0.407)), -2.242) / units;
        } else {
            k6 = tiny;
        }
    } else {
        k6 = 3.36e-10/sqrt(T)/pow(T/1.0e3, 0.2)
             / (1.0 + pow(T/1.0e6, 0.7)) / units;
    }
    return k6;
}

//Calculation of k57 (HI + HI --> HII + HI + e)
// k57_rate
__device__ double coll_i_HI_HI_rate(double T, double units )
{
    // These rate coefficients are from Lenzuni, Chernoff & Salpeter (1991).
    // k57 value based on experimental cross-sections from Gealy & van Zyl (1987).
    if (T > 3.0e3) {
        return 1.2e-17  * pow(T, 1.2) * exp(-1.578e5 / T) / units;
    } else {
        return tiny;
    }
}

//Calculation of k58 (HI + HeI --> HII + HeI + e)
// k58_rate
__device__ double coll_i_HI_HeI_rate(double T, double units )
{
    // These rate coefficients are from Lenzuni, Chernoff & Salpeter (1991).
    // k58 value based on cross-sections from van Zyl, Le & Amme (1981).
    if (T > 3.0e3) {
        return 1.75e-17 * pow(T, 1.3) * exp(-1.578e5 / T) / units;
    } else {
        return tiny;
    }
}

//Calculation of ceHI.
// Cooling collisional excitation HI
__device__ double cool_ceHI_rate(double T, double units )
{
    return 7.5e-19*exp( -fmin(log(dhuge), 118348.0 / T) )
            / ( 1.0 + sqrt(T / 1.0e5) ) / units;    
}

//Calculation of ceHeI.
// Cooling collisional ionization HeI
__device__ double cool_ceHeI_rate(double T, double units )
{
    return 9.1e-27*exp(-fmin(log(dhuge), 13179.0/T))
            * pow(T, -0.1687) / ( 1.0 + sqrt(T/1.0e5) ) / units;
}

//Calculation of ceHeII.
// Cooling collisional excitation HeII
__device__ double cool_ceHeII_rate(double T, double units )
{
    return 5.54e-17*exp(-fmin(log(dhuge), 473638.0/T))
            * pow(T, -0.3970) / ( 1.0 + sqrt(T/1.0e5) ) / units;
}

//Calculation of ciHeIS.
// Cooling collisional ionization HeIS
__device__ double cool_ciHeIS_rate(double T, double units )
{
    return 5.01e-27*pow(T, -0.1687) / ( 1.0 + sqrt(T/1.0e5) )
              * exp(-fmin(log(dhuge), 55338.0/T)) / units;
}

//Calculation of ciHI.
// Cooling collisional ionization HI
__device__ double cool_ciHI_rate(double T, double units )
{
    //Collisional ionization. Polynomial fit from Tom Abel.
    return 2.18e-11 * coll_i_HI_rate(T, 1) / units;    
}


//Calculation of ciHeI.
// Cooling collisional ionization HeI
__device__ double cool_ciHeI_rate(double T, double units )
{
    //Collisional ionization. Polynomial fit from Tom Abel.
    return 3.94e-11 * coll_i_HeI_rate(T, 1) / units;
}

//Calculation of ciHeII.
// Cooling collisional ionization HeII
__device__ double cool_ciHeII_rate(double T, double units )
{
    //Collisional ionization. Polynomial fit from Tom Abel.
    return 8.72e-11 * coll_i_HeII_rate(T, 1) / units; 
}


//Calculation of reHII.
// Cooling recombination HII
__device__ double cool_reHII_rate(double T, double units, bool use_case_B )
{
    double lambdaHI    = 2.0 * 157807.0 / T;
    if (use_case_B) {
        return 3.435e-30 * T * pow(lambdaHI, 1.970)
                / pow( 1.0 + pow(lambdaHI/2.25, 0.376), 3.720)
                / units;
    } else {
        return 1.778e-29 * T * pow(lambdaHI, 1.965)
                / pow(1.0 + pow(lambdaHI/0.541, 0.502), 2.697)
                / units; 
    }
}


//Calculation of reHII.
// Cooling recombination HeII
__device__ double cool_reHeII1_rate(double T, double units, bool use_case_B )
{
    double lambdaHeII  = 2.0 * 285335.0 / T;
    if ( use_case_B ) {
        return 1.26e-14 * kboltz * T * pow(lambdaHeII, 0.75)
                        / units;
    } else {
        return 3e-14 * kboltz * T * pow(lambdaHeII, 0.654)
                / units;
    }    
}

//Calculation of reHII2.
// Cooling recombination HeII Dielectronic
__device__ double cool_reHeII2_rate(double T, double units )
{
    //Dielectronic recombination (Cen, 1992).
    return 1.24e-13 * pow(T, -1.5)
            * exp( -fmin(log(dhuge), 470000.0 / T) )
            * ( 1.0 + 0.3 * exp( -fmin(log(dhuge), 94000.0 / T) ) ) 
            / units;
}

//Calculation of reHIII.
// Cooling recombination HeIII
__device__ double cool_reHeIII_rate(double T, double units, bool use_case_B )
{
    double lambdaHeIII = 2.0 * 631515.0 / T;
    if ( use_case_B ) {
        return 8.0 * 3.435e-30 * T * pow(lambdaHeIII, 1.970)
                / pow(1.0 + pow(lambdaHeIII / 2.25, 0.376), 3.720) 
                / units;
    } else {
        return 8.0 * 1.778e-29 * T * pow(lambdaHeIII, 1.965)
                / pow(1.0 + pow(lambdaHeIII / 0.541, 0.502), 2.697)
                / units;
    }
}

//Calculation of brem.
// Cooling Bremsstrahlung
__device__ double cool_brem_rate(double T, double units )
{
    return 1.43e-27 * sqrt(T)
            * ( 1.1 + 0.34 * exp( -pow(5.5 - log10(T), 2) / 3.0) )
            / units;    
}












#endif
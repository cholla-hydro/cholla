#ifdef CHEMISTRY_GPU

#include "chemistry_gpu.h"
#include "../hydro/hydro_cuda.h"
#include "../global/global_cuda.h"
#include "../io/io.h"
#include "rates.cuh"
#include "rates_Katz95.cuh"

#include "../radiation/alt/static_table_gpu.cuh"

#define eV_to_K 1.160451812e4
#define K_to_eV 8.617333263e-5
#define n_min 1e-20
#define tiny  1e-20

#define TPB_CHEM 256

void Chem_GPU::Allocate_Array_GPU_float(float **array_dev, int size)
{
  CudaSafeCall(cudaMalloc((void **)array_dev, size * sizeof(float)));
}

void Chem_GPU::Copy_Float_Array_to_Device(int size, float *array_h, float *array_d)
{
  CudaSafeCall(cudaMemcpy(array_d, array_h, size * sizeof(float), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
}

void Chem_GPU::Free_Array_GPU_float(float *array_dev) { CudaSafeCall(cudaFree(array_dev)); }

void Chem_GPU::Allocate_Array_GPU_Real(Real **array_dev, int size)
{
  CudaSafeCall(cudaMalloc((void **)array_dev, size * sizeof(Real)));
}

void Chem_GPU::Copy_Real_Array_to_Device(int size, Real *array_h, Real *array_d)
{
  CudaSafeCall(cudaMemcpy(array_d, array_h, size * sizeof(Real), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
}

void Chem_GPU::Free_Array_GPU_Real(Real *array_dev) { CudaSafeCall(cudaFree(array_dev)); }

class Thermal_State
{
 public:
  Real U;
  Real d;
  Real d_HI;
  Real d_HII;
  Real d_HeI;
  Real d_HeII;
  Real d_HeIII;
  Real d_e;

  // Constructor
  __host__ __device__ Thermal_State(Real U_0 = 1, Real d_0 = 1, Real d_HI_0 = 1, Real d_HII_0 = 0, Real d_HeI_0 = 1,
                                    Real d_HeII_0 = 0, Real d_HeIII_0 = 1, Real d_e_0 = 0)
      : U(U_0), d(d_0), d_HI(d_HI_0), d_HII(d_HII_0), d_HeI(d_HeI_0), d_HeII(d_HeII_0), d_HeIII(d_HeIII_0), d_e(d_e_0)
  {
  }

  __host__ __device__ Real get_MMW()
  {
    // Real m_tot = d_HI + d_HII + d_HeI + d_HeII + d_HeIII;
    Real n_tot = d_HI + d_HII + 0.25 * (d_HeI + d_HeII + d_HeIII) + d_e;
    return d / n_tot;
    // return m_tot / n_tot;
  }

  __host__ __device__ Real get_temperature(Real gamma)
  {
    Real mu, temp;
    mu   = get_MMW();
    temp = (gamma - 1) * mu * U * MP / KB * 1e10;
    return temp;
  }

  __host__ __device__ Real compute_U(Real temp, Real gamma)
  {
    Real mu, U_local;
    mu      = get_MMW();
    U_local = temp / (gamma - 1) / mu / MP * KB / 1e10;
    return U_local;
  }
};

__device__ void get_temperature_indx(Real T, Chemistry_Header &Chem_H, int &temp_indx, Real &delta_T, Real temp_old,
                                     int print)
{
  Real logT, logT_start, d_logT, logT_l, logT_r;
  logT       = log(0.5 * (T + temp_old));
  logT_start = log(Chem_H.Temp_start);
  logT       = fmax(logT_start, logT);
  logT       = fmin(log(Chem_H.Temp_end), logT);
  d_logT     = (log(Chem_H.Temp_end) - logT_start) / (Chem_H.N_Temp_bins - 1);
  temp_indx  = (int)floor((logT - logT_start) / d_logT);
  temp_indx  = max(0, temp_indx);
  temp_indx  = min(Chem_H.N_Temp_bins - 2, temp_indx);
  logT_l     = logT_start + temp_indx * d_logT;
  logT_r     = logT_start + (temp_indx + 1) * d_logT;
  delta_T    = (logT - logT_l) / (logT_r - logT_l);
  // if (print) printf(" logT_start: %f  logT_end: %f  d_logT: %f   \n", logT_start, log( Chem_H.Temp_end ), d_logT );
  // if (print) printf(" logT: %f  logT_l: %f  logT_r: %f   \n", logT, logT_l, logT_r );
}

__device__ Real interpolate_rate(Real *rate_table, int indx, Real delta)
{
  Real rate_val;
  rate_val = rate_table[indx];
  rate_val = rate_val + delta * (rate_table[indx + 1] - rate_val);
  return rate_val;
}

__device__ Real Get_Cooling_Rates(Thermal_State &TS, Chemistry_Header &Chem_H, Real dens_number_conv, Real current_z,
                                  Real temp_prev, float photo_h_HI, float photo_h_HeI, float photo_h_HeII, int print)
{
  int temp_indx;
  Real temp, delta_T, U_dot;
  temp = TS.get_temperature(Chem_H.gamma);
  get_temperature_indx(temp, Chem_H, temp_indx, delta_T, temp_prev, print);
  if (print > 1) printf("mu: %f  temp: %f  temp_indx: %d  delta_T: %f  \n", TS.get_MMW(), temp, temp_indx, delta_T);
  U_dot = 0.0;

  // Collisional excitation cooling
  Real cool_ceHI, cool_ceHeI, cool_ceHeII;
  cool_ceHI = interpolate_rate(Chem_H.cool_ceHI_d, temp_indx, delta_T) * TS.d_HI * TS.d_e;
  cool_ceHeI =
      interpolate_rate(Chem_H.cool_ceHeI_d, temp_indx, delta_T) * TS.d_HeII * TS.d_e * TS.d_e * dens_number_conv / 4.0;
  cool_ceHeII = interpolate_rate(Chem_H.cool_ceHeII_d, temp_indx, delta_T) * TS.d_HeII * TS.d_e / 4.0;
  U_dot -= cool_ceHI + cool_ceHeI + cool_ceHeII;

  // Collisional excitation cooling
  Real cool_ciHI, cool_ciHeI, cool_ciHeII, cool_ciHeIS;
  cool_ciHI   = interpolate_rate(Chem_H.cool_ciHI_d, temp_indx, delta_T) * TS.d_HI * TS.d_e;
  cool_ciHeI  = interpolate_rate(Chem_H.cool_ciHeI_d, temp_indx, delta_T) * TS.d_HeI * TS.d_e / 4.0;
  cool_ciHeII = interpolate_rate(Chem_H.cool_ciHeII_d, temp_indx, delta_T) * TS.d_HeII * TS.d_e / 4.0;
  cool_ciHeIS =
      interpolate_rate(Chem_H.cool_ciHeIS_d, temp_indx, delta_T) * TS.d_HeII * TS.d_e * TS.d_e * dens_number_conv / 4.0;
  U_dot -= cool_ciHI + cool_ciHeI + cool_ciHeII + cool_ciHeIS;

  // Recombination cooling
  Real cool_reHII, cool_reHeII1, cool_reHeII2, cool_reHeIII;
  cool_reHII   = interpolate_rate(Chem_H.cool_reHII_d, temp_indx, delta_T) * TS.d_HII * TS.d_e;
  cool_reHeII1 = interpolate_rate(Chem_H.cool_reHeII1_d, temp_indx, delta_T) * TS.d_HeII * TS.d_e / 4.0;
  cool_reHeII2 = interpolate_rate(Chem_H.cool_reHeII2_d, temp_indx, delta_T) * TS.d_HeII * TS.d_e / 4.0;
  cool_reHeIII = interpolate_rate(Chem_H.cool_reHeIII_d, temp_indx, delta_T) * TS.d_HeIII * TS.d_e / 4.0;
  U_dot -= cool_reHII + cool_reHeII1 + cool_reHeII2 + cool_reHeIII;

  // Bremsstrahlung cooling
  Real cool_brem;
  cool_brem =
      interpolate_rate(Chem_H.cool_brem_d, temp_indx, delta_T) * (TS.d_HII + TS.d_HeII / 4.0 + TS.d_HeIII) * TS.d_e;
  U_dot -= cool_brem;

  #ifdef COSMOLOGY
  // Compton cooling or heating
  Real cool_compton, temp_cmb;
  temp_cmb     = 2.73 * (1.0 + current_z);
  cool_compton = Chem_H.cool_compton * pow(1.0 + current_z, 4) * (temp - temp_cmb) * TS.d_e / dens_number_conv;
  U_dot -= cool_compton;
  #endif

  // Phothoheating
  Real photo_heat;
  photo_heat = (photo_h_HI * TS.d_HI + 0.25 * (photo_h_HeI * TS.d_HeI + photo_h_HeII * TS.d_HeII)) / dens_number_conv;
  U_dot += photo_heat;

  if (temp <= 1.01 * Chem_H.Temp_start && fabs(U_dot) < 0) U_dot = tiny;
  if (fabs(U_dot) < tiny) U_dot = tiny;

  if (print > 1) printf("HI: %e  \n", TS.d_HI);
  if (print > 1) printf("HII: %e  \n", TS.d_HII);
  if (print > 1) printf("HeI: %e  \n", TS.d_HeI);
  if (print > 1) printf("HeII: %e  \n", TS.d_HeII);
  if (print > 1) printf("HeIII: %e  \n", TS.d_HeIII);
  if (print > 1) printf("de: %e  \n", TS.d_e);
  if (print > 1) printf("Cooling ceHI: %e  \n", cool_ceHI);
  if (print > 1) printf("Cooling ceHeI: %e   \n", cool_ceHeI);
  if (print > 1) printf("Cooling ceHeII: %e   \n", cool_ceHeII);
  if (print > 1) printf("Cooling ciHI: %e  \n", cool_ciHI);
  if (print > 1) printf("Cooling ciHeI: %e  \n", cool_ciHeI);
  if (print > 1) printf("Cooling ciHeII: %e  \n", cool_ciHeII);
  if (print > 1) printf("Cooling ciHeIS: %e  \n", cool_ciHeIS);
  if (print > 1) printf("Cooling reHII: %e  \n", cool_reHII);
  if (print > 1) printf("Cooling reHeII1: %e  \n", cool_reHeII1);
  if (print > 1) printf("Cooling reHeII2: %e  \n", cool_reHeII2);
  if (print > 1) printf("Cooling reHeIII: %e  \n", cool_reHeIII);
  if (print > 1) printf("Cooling brem: %e  \n", cool_brem);
  if (print > 0) printf("Cooling piHI: %e   rate: %e \n", photo_h_HI, photo_h_HI * TS.d_HI / dens_number_conv);
  if (print > 1)
    printf("Cooling piHeI: %e  rate: %e \n", photo_h_HeI, photo_h_HeI * TS.d_HeI / dens_number_conv * 0.25);
  if (print > 1)
    printf("Cooling piHeII: %e rate: %e \n", photo_h_HeII, photo_h_HeII * TS.d_HeII / dens_number_conv * 0.25);
  if (print > 1) printf("Cooling DOM: %e  \n", dens_number_conv);
  #ifdef COSMOLOGY
  if (print > 1) printf("Cooling compton: %e  \n", cool_compton);
  #endif
  if (print > 0) printf("Cooling U_dot: %e  \n", U_dot);

  return U_dot;
}

__device__ void Get_Reaction_Rates(Thermal_State &TS, Chemistry_Header &Chem_H, Real &k_coll_i_HI, Real &k_coll_i_HeI,
                                   Real &k_coll_i_HeII, Real &k_coll_i_HI_HI, Real &k_coll_i_HI_HeI, Real &k_recomb_HII,
                                   Real &k_recomb_HeII, Real &k_recomb_HeIII, int print)
{
  int temp_indx;
  Real temp, delta_T;
  temp = TS.get_temperature(Chem_H.gamma);
  get_temperature_indx(temp, Chem_H, temp_indx, delta_T, temp, print);

  k_coll_i_HI   = interpolate_rate(Chem_H.k_coll_i_HI_d, temp_indx, delta_T);
  k_coll_i_HeI  = interpolate_rate(Chem_H.k_coll_i_HeI_d, temp_indx, delta_T);
  k_coll_i_HeII = interpolate_rate(Chem_H.k_coll_i_HeII_d, temp_indx, delta_T);

  k_coll_i_HI_HI  = interpolate_rate(Chem_H.k_coll_i_HI_HI_d, temp_indx, delta_T);
  k_coll_i_HI_HeI = interpolate_rate(Chem_H.k_coll_i_HI_HeI_d, temp_indx, delta_T);

  k_recomb_HII   = interpolate_rate(Chem_H.k_recomb_HII_d, temp_indx, delta_T);
  k_recomb_HeII  = interpolate_rate(Chem_H.k_recomb_HeII_d, temp_indx, delta_T);
  k_recomb_HeIII = interpolate_rate(Chem_H.k_recomb_HeIII_d, temp_indx, delta_T);

  if (print > 1) printf("logT: %f   temp_indx: %d\n", log(temp), temp_indx);
  if (print > 1) printf("k_coll_i_HI: %e \n", k_coll_i_HI);
  if (print > 1) printf("k_coll_i_HeI: %e \n", k_coll_i_HeI);
  if (print > 1) printf("k_coll_i_HeII: %e \n", k_coll_i_HeII);
  if (print > 1) printf("k_coll_i_HI_HI: %e \n", k_coll_i_HI_HI);
  if (print > 1) printf("k_coll_i_HI_HeI: %e \n", k_coll_i_HI_HeI);
  if (print > 1) printf("k_recomb_HII: %e \n", k_recomb_HII);
  if (print > 1) printf("k_recomb_HeII: %e \n", k_recomb_HeII);
  if (print > 1) printf("k_recomb_HeIII: %e \n", k_recomb_HeIII);
}

__device__ int Binary_Search(int N, Real val, float *data, int indx_l, int indx_r)
{
  int n, indx;
  n    = indx_r - indx_l;
  indx = indx_l + n / 2;
  if (val >= data[N - 1]) return indx_r;
  if (val <= data[0]) return indx_l;
  if (indx_r == indx_l + 1) return indx_l;
  if (data[indx] <= val)
    indx_l = indx;
  else
    indx_r = indx;
  return Binary_Search(N, val, data, indx_l, indx_r);
}

__device__ Real linear_interpolation(Real delta_x, int indx_l, int indx_r, float *array)
{
  float v_l, v_r;
  Real v;
  v_l = array[indx_l];
  v_r = array[indx_r];
  v   = delta_x * (v_r - v_l) + v_l;
  return v;
}

__device__ void Get_Current_Photo_Rates(Chemistry_Header &Chem_H, const Real *rf, int id, int ncells, float &photo_i_HI,
                                        float &photo_i_HeI, float &photo_i_HeII, float &photo_h_HI, float &photo_h_HeI,
                                        float &photo_h_HeII, int print)
{
  #ifdef RT
  if (rf != nullptr) {
    const float rfN0    = rf[id + 0 * ncells];
    const float rfNHI   = rf[id + 1 * ncells];
    const float rfNHeI  = rf[id + 2 * ncells];
    const float rfNHeII = rf[id + 3 * ncells];

    float tauHI   = (rfNHI > rfN0 ? 0 : (rfNHI > 0 ? -log(1.0e-35 + rfNHI / rfN0) : 1001));
    float tauHeI  = (rfNHeI > rfN0 ? 0 : (rfNHeI > 0 ? -log(1.0e-35 + rfNHeI / rfN0) : 1001));
    float tauHeII = (rfNHeII > rfN0 ? 0 : (rfNHeII > 0 ? -log(1.0e-35 + rfNHeII / rfN0) : 1001));

    float pRates[6];
    float x[3] = {Chem_H.dStretch->tau2x(tauHI), Chem_H.dStretch->tau2x(tauHeI), Chem_H.dStretch->tau2x(tauHeII)};
    Chem_H.dTables[0]->GetValues(x, pRates, 0, 6);

    for (unsigned int i = 0; i < 6; i++) {
      pRates[i] *= rfN0;
    }

    if (Chem_H.dTables[1] != nullptr) {
      const float rfFHI   = rf[id + 4 * ncells];
      const float rfFHeI  = rf[id + 5 * ncells];
      const float rfFHeII = rf[id + 6 * ncells];

      float tauHI   = (rfFHI > 1 ? 0 : (rfFHI > 0 ? -log(1.0e-35 + rfFHI) : 1001));
      float tauHeI  = (rfFHeI > 1 ? 0 : (rfFHeI > 0 ? -log(1.0e-35 + rfFHeI) : 1001));
      float tauHeII = (rfFHeII > 1 ? 0 : (rfFHeII > 0 ? -log(1.0e-35 + rfFHeII) : 1001));

      x[0] = Chem_H.dStretch->tau2x(tauHI);
      x[1] = Chem_H.dStretch->tau2x(tauHeI);
      x[2] = Chem_H.dStretch->tau2x(tauHeII);

      float pRates2[6];
      Chem_H.dTables[1]->GetValues(x, pRates2, 0, 6);

      for (unsigned int i = 0; i < 6; i++) {
        pRates[i] += pRates2[i];
      }
    }

    photo_i_HI   = pRates[0] * Chem_H.unitPhotoIonization;
    photo_h_HI   = pRates[1] * Chem_H.unitPhotoHeating;
    photo_i_HeI  = pRates[2] * Chem_H.unitPhotoIonization;
    photo_h_HeI  = pRates[3] * Chem_H.unitPhotoHeating;
    photo_i_HeII = pRates[4] * Chem_H.unitPhotoIonization;
    photo_h_HeII = pRates[5] * Chem_H.unitPhotoHeating;
  } else
  #endif
  {
    if (Chem_H.current_z > Chem_H.uvb_rates_redshift_d[Chem_H.n_uvb_rates_samples - 1]) {
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
    Real z_l, z_r, delta_x;
    indx_l  = Binary_Search(Chem_H.n_uvb_rates_samples, Chem_H.current_z, Chem_H.uvb_rates_redshift_d, 0,
                            Chem_H.n_uvb_rates_samples - 1);
    z_l     = Chem_H.uvb_rates_redshift_d[indx_l];
    z_r     = Chem_H.uvb_rates_redshift_d[indx_l + 1];
    delta_x = (Chem_H.current_z - z_l) / (z_r - z_l);

    photo_i_HI   = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_ion_HI_rate_d);
    photo_i_HeI  = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_ion_HeI_rate_d);
    photo_i_HeII = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_ion_HeII_rate_d);
    photo_h_HI   = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_heat_HI_rate_d);
    photo_h_HeI  = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_heat_HeI_rate_d);
    photo_h_HeII = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_heat_HeII_rate_d);
  }
}

__device__ Real Get_Chemistry_dt(Thermal_State &TS, Chemistry_Header &Chem_H, Real &HI_dot, Real &e_dot, Real U_dot,
                                 Real k_coll_i_HI, Real k_coll_i_HeI, Real k_coll_i_HeII, Real k_coll_i_HI_HI,
                                 Real k_coll_i_HI_HeI, Real k_recomb_HII, Real k_recomb_HeII, Real k_recomb_HeIII,
                                 float photo_i_HI, float photo_i_HeI, float photo_i_HeII, int n_iter, Real HI_dot_prev,
                                 Real e_dot_prev, Real t_chem, Real dt_hydro, int print)
{
  Real dt, energy;
  // Rate of change of HI
  HI_dot = k_recomb_HII * TS.d_HII * TS.d_e - k_coll_i_HI * TS.d_HI * TS.d_e - k_coll_i_HI_HI * TS.d_HI * TS.d_HI -
           k_coll_i_HI_HeI * TS.d_HI * TS.d_HeI / 4.0 - photo_i_HI * TS.d_HI;

  // Rate of change of electron
  e_dot = k_coll_i_HI * TS.d_HI * TS.d_e + k_coll_i_HeI * TS.d_HeI / 4.0 * TS.d_e +
          k_coll_i_HeII * TS.d_HeII / 4.0 * TS.d_e + k_coll_i_HI_HI * TS.d_HI * TS.d_HI +
          +k_coll_i_HI_HeI * TS.d_HI * TS.d_HeI / 4.0 - k_recomb_HII * TS.d_HII * TS.d_e -
          k_recomb_HeII * TS.d_HeII / 4.0 * TS.d_e - k_recomb_HeIII * TS.d_HeIII / 4.0 * TS.d_e + photo_i_HI * TS.d_HI +
          photo_i_HeI * TS.d_HeI / 4.0 + photo_i_HeII * TS.d_HeII / 4.0;

  // Bound from below to prevent numerical errors
  if (fabs(HI_dot) < tiny) HI_dot = fmin(tiny, TS.d_HI);
  if (fabs(e_dot) < tiny) e_dot = fmin(tiny, TS.d_e);

  // If the net rate is almost perfectly balanced then set
  // it to zero (since it is zero to available precision)
  if (fmin(fabs(k_coll_i_HI * TS.d_HI * TS.d_e), fabs(k_recomb_HII * TS.d_HII * TS.d_e)) /
          fmax(fabs(HI_dot), fabs(e_dot)) >
      1e6) {
    HI_dot = tiny;
    e_dot  = tiny;
  }

  if (n_iter > 50) {
    HI_dot = fmin(fabs(HI_dot), fabs(HI_dot_prev));
    e_dot  = fmin(fabs(e_dot), fabs(e_dot_prev));
  }

  if (TS.d * Chem_H.dens_number_conv > 1e8 && U_dot > 0) {
    printf("#### Equlibrium  \n");
  }

  #ifdef TEMPERATURE_FLOOR
  if (TS.get_temperature(Chem_H.gamma) < TEMP_FLOOR) TS.U = TS.compute_U(TEMP_FLOOR, Chem_H.gamma);
  #endif
  
  energy = fmax( TS.U * TS.d, tiny );
  //dt = fmin( fabs( 0.1 * TS.d_HI / HI_dot ), fabs( 0.1 * TS.d_e / e_dot )  ); NG electrons should not set the time-step, they are a derived quantity
  dt = fabs( 0.1 * TS.d_HI / HI_dot );
  dt = fmin( fabs( 0.1 * energy / U_dot ), dt  );
  dt = fmin( 0.5 * dt_hydro, dt );
  dt = fmin( dt_hydro - t_chem, dt );
  
  if ( n_iter == Chem_H.max_iter-1 ){
    //printf("##### Chem_GPU: dt_hydro: %e   t_chem: %e   dens: %e   temp: %e  GE: %e  U_dot: %e   dt_HI: %e   dt_e: %e   dt_U: %e \n", dt_hydro,  t_chem, TS.d, TS.get_temperature(Chem_H.gamma), energy, U_dot, fabs( 0.1 * TS.d_HI / HI_dot ), fabs( 0.1 * TS.d_e / e_dot ), fabs( 0.1 * TS.U * TS.d / U_dot )   ) ;
  }

  if (print > 0) {
    printf("HIdot: %e (%g,%g)\n", HI_dot, k_recomb_HII * TS.d_HII * TS.d_e, photo_i_HI * TS.d_HI);
  }
  if (print > 1) printf("edot: %e\n", e_dot);
  if (print > 1) printf("energy: %e\n", TS.U * TS.d);
  if (print > 1) printf("Udot: %e\n", U_dot);
  if (print > 1) printf("dt_hydro: %e\n", dt_hydro);
  if (print > 1) printf("dt: %e\n", dt);

  return dt;
}

__device__ void Update_Step(Thermal_State &TS, Chemistry_Header &Chem_H, Real dt, Real U_dot, Real k_coll_i_HI,
                            Real k_coll_i_HeI, Real k_coll_i_HeII, Real k_coll_i_HI_HI, Real k_coll_i_HI_HeI,
                            Real k_recomb_HII, Real k_recomb_HeII, Real k_recomb_HeIII, float photo_i_HI,
                            float photo_i_HeI, float photo_i_HeII, Real &HI_dot_prev, Real &e_dot_prev, Real &temp_prev,
                            int print)
{
  Real d_HI_p, d_HII_p, d_HeI_p, d_HeII_p, d_HeIII_p, d_e_p;
  Real s_coef, a_coef;

  // Update HI
  s_coef = k_recomb_HII * TS.d_HII * TS.d_e;
  a_coef = k_coll_i_HI * TS.d_e + k_coll_i_HI_HI * TS.d_HI + k_coll_i_HI_HeI * TS.d_HeI / 4.0 + photo_i_HI;
  d_HI_p = (dt * s_coef + TS.d_HI) / (1.0 + dt * a_coef);
  if (print > 1) printf("Update HI  s_coef: %e    a_coef: %e   HIp: %e \n", s_coef, a_coef, d_HI_p);

  // Update HII
  s_coef = k_coll_i_HI * d_HI_p * TS.d_e + k_coll_i_HI_HI * d_HI_p * d_HI_p +
           k_coll_i_HI_HeI * d_HI_p * TS.d_HeI / 4.0 + photo_i_HI * d_HI_p;
  a_coef  = k_recomb_HII * TS.d_e;
  d_HII_p = (dt * s_coef + TS.d_HII) / (1.0 + dt * a_coef);
  if (print > 1) printf("Update HII  s_coef: %e    a_coef: %e   HIIp: %e \n", s_coef, a_coef, d_HII_p);

  // Update electron
  s_coef = k_coll_i_HI_HI * d_HI_p * d_HI_p + k_coll_i_HI_HeI * d_HI_p * TS.d_HeI / 4.0 + photo_i_HI * TS.d_HI +
           photo_i_HeI * TS.d_HeI / 4.0 + photo_i_HeII * TS.d_HeII / 4.0;
  a_coef = -k_coll_i_HI * TS.d_HI + k_recomb_HII * TS.d_HII - k_coll_i_HeI * TS.d_HeI / 4.0 +
           k_recomb_HeII * TS.d_HeII / 4.0 - k_coll_i_HeII * TS.d_HeII / 4.0 + k_recomb_HeIII * TS.d_HeIII / 4.0;
  d_e_p = (dt * s_coef + TS.d_e) / (1.0 + dt * a_coef);
  if (print > 1) printf("Update e  s_coef: %e    a_coef: %e   ep: %e \n", s_coef, a_coef, d_e_p);

  // Update HeI
  s_coef  = k_recomb_HeII * TS.d_HeII * TS.d_e;
  a_coef  = k_coll_i_HeI * TS.d_e + photo_i_HeI;
  d_HeI_p = (dt * s_coef + TS.d_HeI) / (1.0 + dt * a_coef);
  if (print > 1) printf("Update HeI  s_coef: %e    a_coef: %e   HeIp: %e \n", s_coef, a_coef, d_HeI_p);

  // Update HeII
  s_coef   = k_coll_i_HeI * d_HeI_p * TS.d_e + k_recomb_HeIII * TS.d_HeIII * TS.d_e + photo_i_HeI * d_HeI_p;
  a_coef   = k_recomb_HeII * TS.d_e + k_coll_i_HeII * TS.d_e + photo_i_HeII;
  d_HeII_p = (dt * s_coef + TS.d_HeII) / (1.0 + dt * a_coef);
  if (print > 1) printf("Update HeII  s_coef: %e    a_coef: %e   HeIIp: %e \n", s_coef, a_coef, d_HeII_p);

  // Update HeIII
  s_coef    = k_coll_i_HeII * d_HeII_p * TS.d_e + photo_i_HeII * d_HeII_p;
  a_coef    = k_recomb_HeIII * TS.d_e;
  d_HeIII_p = (dt * s_coef + TS.d_HeIII) / (1.0 + dt * a_coef);
  if (print > 1) printf("Update HeIII  s_coef: %e    a_coef: %e   HeIIIp: %e \n", s_coef, a_coef, d_HeIII_p);

  // Record the temperature for the next step
  temp_prev = TS.get_temperature(Chem_H.gamma);

  HI_dot_prev = fabs(TS.d_HI - d_HI_p) / fmax(dt, tiny);
  TS.d_HI     = fmax(d_HI_p, tiny);
  TS.d_HII    = fmax(d_HII_p, tiny);
  TS.d_HeI    = fmax(d_HeI_p, tiny);
  TS.d_HeII   = fmax(d_HeII_p, tiny);
  TS.d_HeIII  = fmax(d_HeIII_p, 1e-5 * tiny);

  // Use charge conservation to determine electron fraction
  e_dot_prev = TS.d_e;
  TS.d_e     = TS.d_HII + TS.d_HeII / 4.0 + TS.d_HeIII / 2.0;
  e_dot_prev = fabs(TS.d_e - e_dot_prev) / fmax(dt, tiny);

  // Update internal energy
  TS.U += U_dot / TS.d * dt;
  #ifdef TEMPERATURE_FLOOR
  if (TS.get_temperature(Chem_H.gamma) < TEMP_FLOOR) TS.U = TS.compute_U(TEMP_FLOOR, Chem_H.gamma);
  #endif
  if (print > 1) printf("Updated U: %e \n", TS.U);
}

__global__ void Update_Chemistry_kernel(Real *dev_conserved, const Real *dev_rf, int nx, int ny, int nz, int n_ghost,
                                        int n_fields, Real dt_hydro, Chemistry_Header Chem_H)
{
  int id, xid, yid, zid, n_cells, n_iter;
  Real d, d_inv, vx, vy, vz;
  Real GE, E_kin, dt_chem, t_chem;
  Real current_a, a3, a2;

  Real density_conv, energy_conv;
  density_conv = Chem_H.density_conversion;
  energy_conv  = Chem_H.energy_conversion;

  Real U_dot, HI_dot, e_dot, HI_dot_prev, e_dot_prev, temp_prev;
  Real k_coll_i_HI, k_coll_i_HeI, k_coll_i_HeII, k_coll_i_HI_HI, k_coll_i_HI_HeI;
  Real k_recomb_HII, k_recomb_HeII, k_recomb_HeIII;
  float photo_i_HI, photo_i_HeI, photo_i_HeII;
  float photo_h_HI, photo_h_HeI, photo_h_HeII;
  Real correct_H, correct_He;

  n_cells = nx * ny * nz;

  // get a global thread ID
  id        = threadIdx.x + blockIdx.x * blockDim.x;
  zid       = id / (nx * ny);
  yid       = (id - zid * nx * ny) / nx;
  xid       = id - zid * nx * ny - yid * nx;
  int print = 0;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    E_kin = 0.5 * d * (vx * vx + vy * vy + vz * vz);
  #ifdef DE
    GE = dev_conserved[(n_fields - 1) * n_cells + id];
  #else
    GE = dev_conserved[4 * n_cells + id] - E_kin;
  #endif

    if (xid == n_ghost && yid == n_ghost && zid == n_ghost) {
      ///print = 2;
    }

    // Convert to cgs units
    current_a = 1 / ( Chem_H.current_z + 1);
    a2 = current_a * current_a;
    a3 = a2 * current_a;  
    d  *= density_conv / a3;
    GE *= energy_conv  / a2; 
    //dt_hydro = dt_hydro / Chem_H.time_units; NG 221126: this is a bug, integration is in code units

  #ifdef COSMOLOGY
    dt_hydro *= current_a * current_a / Chem_H.H0 * 1000 *
                KPC
  #endif  // COSMOLOGY
          // dt_hydro = dt_hydro * current_a * current_a / Chem_H.H0 * 1000 * KPC / Chem_H.time_units;
          //  delta_a = Chem_H.H0 * sqrt( Chem_H.Omega_M/current_a + Chem_H.Omega_L*pow(current_a, 2) ) / (
          //  1000 * KPC ) * dt_hydro * Chem_H.time_units;

                    // Initialize the thermal state
                    Thermal_State TS;
    TS.d       = dev_conserved[id] / a3;
    TS.d_HI    = dev_conserved[5 * n_cells + id] / a3;
    TS.d_HII   = dev_conserved[6 * n_cells + id] / a3;
    TS.d_HeI   = dev_conserved[7 * n_cells + id] / a3;
    TS.d_HeII  = dev_conserved[8 * n_cells + id] / a3;
    TS.d_HeIII = dev_conserved[9 * n_cells + id] / a3;
    // TS.d_e     = dev_conserved[10*n_cells + id] / a3; NG 221127: removed, no need to advect electrons, they are a
    // derived field
    TS.U = GE * d_inv * 1e-10;

    // Ceiling species
    TS.d_HI    = fmax(TS.d_HI, tiny);
    TS.d_HII   = fmax(TS.d_HII, tiny);
    TS.d_HeI   = fmax(TS.d_HeI, tiny);
    TS.d_HeII  = fmax(TS.d_HeII, tiny);
    TS.d_HeIII = fmax(TS.d_HeIII, 1e-5 * tiny);
    // TS.d_e     = fmax( TS.d_e,     tiny );  NG 221127: removed, no need to advect electrons, they are a derived field

    // Use charge conservation to determine electron fractioan
    TS.d_e = TS.d_HII + TS.d_HeII / 4.0 + TS.d_HeIII / 2.0;

    // Compute temperature at first iteration
    temp_prev = TS.get_temperature(Chem_H.gamma);

    // if (print){
    //   printf("current_z: %f\n", current_z );
    //   printf("density_units: %e\n", Chem_H.density_units );
    //   printf("lenght_units: %e\n", Chem_H.length_units );
    //   printf("velocity_units: %e\n", Chem_H.velocity_units );
    //   printf("time_units: %e\n", Chem_H.time_units );
    //   printf("dom: %e \n", dens_number_conv );
    //   printf("density: %e \n",         TS.d );
    //   printf("HI_density: %e \n",      TS.d_HI );
    //   printf("HII_density: %e \n",     TS.d_HII );
    //   printf("HeI_density: %e \n",     TS.d_HeI );
    //   printf("HeII_density: %e \n",    TS.d_HeII );
    //   printf("HeIII_density: %e \n",   TS.d_HeIII );
    //   printf("e_density: %e \n",       TS.d_e );
    //   printf("internal_energy: %e \n", TS.U );
    //   printf("energy: %e \n", TS.U*TS.d );
    //   printf("dt_hydro: %e \n", dt_hydro / Chem_H.time_units );
    // }

    // Get the photoheating and photoionization rates at z=current_z
    Get_Current_Photo_Rates(Chem_H, dev_rf, id, n_cells, photo_i_HI, photo_i_HeI, photo_i_HeII, photo_h_HI, photo_h_HeI,
                            photo_h_HeII, print);

    HI_dot_prev = 0;
    e_dot_prev  = 0;
    n_iter      = 0;
    t_chem      = 0;
    while (t_chem < dt_hydro) {
      if (print != 0) printf("########################################## Iter %d \n", n_iter);

      U_dot = Get_Cooling_Rates(TS, Chem_H, Chem_H.dens_number_conv, Chem_H.current_z, temp_prev, photo_h_HI,
                                photo_h_HeI, photo_h_HeII, print);

      Get_Reaction_Rates(TS, Chem_H, k_coll_i_HI, k_coll_i_HeI, k_coll_i_HeII, k_coll_i_HI_HI, k_coll_i_HI_HeI,
                         k_recomb_HII, k_recomb_HeII, k_recomb_HeIII, print);
      if (print > 1) {
        printf("k_photo_ion_HI: %e \n", photo_i_HI);
        printf("k_photo_ion_HeI: %e \n", photo_i_HeI);
        printf("k_photo_ion_HeII: %e \n", photo_i_HeII);
      }

      dt_chem =
          Get_Chemistry_dt(TS, Chem_H, HI_dot, e_dot, U_dot, k_coll_i_HI, k_coll_i_HeI, k_coll_i_HeII, k_coll_i_HI_HI,
                           k_coll_i_HI_HeI, k_recomb_HII, k_recomb_HeII, k_recomb_HeIII, photo_i_HI, photo_i_HeI,
                           photo_i_HeII, n_iter, HI_dot_prev, e_dot_prev, t_chem, dt_hydro, print);

      Update_Step(TS, Chem_H, dt_chem, U_dot, k_coll_i_HI, k_coll_i_HeI, k_coll_i_HeII, k_coll_i_HI_HI, k_coll_i_HI_HeI,
                  k_recomb_HII, k_recomb_HeII, k_recomb_HeIII, photo_i_HI, photo_i_HeI, photo_i_HeII, HI_dot_prev,
                  e_dot_prev, temp_prev, print);

      t_chem += dt_chem;
      n_iter += 1;
      if (n_iter == Chem_H.max_iter) break;
    }
    if (print > 1) printf("Chem_GPU: N Iter:  %d\n", n_iter);

    // Make consistent abundances with the H and He density
    correct_H  = Chem_H.H_fraction * TS.d / (TS.d_HI + TS.d_HII);
    correct_He = (1.0 - Chem_H.H_fraction) * TS.d / (TS.d_HeI + TS.d_HeII + TS.d_HeIII);
    TS.d_HI *= correct_H;
    TS.d_HII *= correct_H;
    TS.d_HeI *= correct_He;
    TS.d_HeII *= correct_He;
    TS.d_HeIII *= correct_He;

    // Write the Updated Thermal State
    dev_conserved[5 * n_cells + id] = TS.d_HI * a3;
    dev_conserved[6 * n_cells + id] = TS.d_HII * a3;
    dev_conserved[7 * n_cells + id] = TS.d_HeI * a3;
    dev_conserved[8 * n_cells + id] = TS.d_HeII * a3;
    dev_conserved[9 * n_cells + id] = TS.d_HeIII * a3;
    // dev_conserved[10*n_cells + id] = TS.d_e     * a3; NG 221127: removed, no need to advect electrons, they are a
    // derived field
    d                               = d / density_conv * a3;
    GE                              = TS.U / d_inv / energy_conv * a2 / 1e-10;
    dev_conserved[4 * n_cells + id] = GE + E_kin;
  #ifdef DE
    dev_conserved[(n_fields - 1) * n_cells + id] = GE;
  #endif

    if (print != 0) printf("###########################################\n");
    if (print != 0) printf("Updated HI:  %e\n", TS.d_HI / TS.d);
    if (print != 0) printf("Updated HII:  %e\n", TS.d_HII / TS.d);
    if (print != 0) printf("Updated HeI:  %e\n", TS.d_HeI / TS.d);
    if (print != 0) printf("Updated HeII:  %e\n", TS.d_HeII / TS.d);
    if (print != 0) printf("Updated HeIII:  %e\n", TS.d_HeIII / TS.d);
    if (print != 0) printf("Updated GE:  %e\n", GE);
    if (print != 0) printf("Updated E:   %e\n", dev_conserved[4 * n_cells + id]);
  }
}

void Do_Chemistry_Update(Real *dev_conserved, const Real *dev_rf, int nx, int ny, int nz, int n_ghost, int n_fields,
                         Real dt, Chemistry_Header &Chem_H)
{
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int ngrid = (nx * ny * nz - 1) / TPB_CHEM + 1;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB_CHEM, 1, 1);
  hipLaunchKernelGGL(Update_Chemistry_kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, dev_rf, nx, ny, nz, n_ghost,
                     n_fields, dt, Chem_H);

  CudaCheckError();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  Chem_H.runtime_chemistry_step = (Real)time / 1000;  // (Convert ms to secs )
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Reaction and cooling rates from Grackle

  // Kelvin to eV conversion factor
  #ifndef tevk
    #define tevk 1.1605e4
  #endif
  // Comparison value
  #ifndef dhuge
    #define dhuge 1.0e30
  #endif
  // Small value
  #ifndef tiny
    #define tiny 1.0e-20
  #endif
  // Boltzmann's constant
  #ifndef kboltz
    #define kboltz 1.3806504e-16  // Boltzmann's constant [cm2gs-2K-1] or [ergK-1]
  #endif

// Calculation of k1 (HI + e --> HII + 2e)
// k1_rate
__device__ Real coll_i_HI_rate(Real T, Real units)
{
  Real T_ev    = T / 11605.0;
  Real logT_ev = log(T_ev);

  Real k1 = exp(-32.71396786375 + 13.53655609057 * logT_ev - 5.739328757388 * pow(logT_ev, 2) +
                1.563154982022 * pow(logT_ev, 3) - 0.2877056004391 * pow(logT_ev, 4) +
                0.03482559773736999 * pow(logT_ev, 5) - 0.00263197617559 * pow(logT_ev, 6) +
                0.0001119543953861 * pow(logT_ev, 7) - 2.039149852002e-6 * pow(logT_ev, 8)) /
            units;
  if (T_ev <= 0.8) {
    k1 = fmax(tiny, k1);
  }
  return k1;
}

// Calculation of k3 (HeI + e --> HeII + 2e)
//  k3_rate
__device__ Real coll_i_HeI_rate(Real T, Real units)
{
  Real T_ev    = T / 11605.0;
  Real logT_ev = log(T_ev);

  if (T_ev > 0.8) {
    return exp(-44.09864886561001 + 23.91596563469 * logT_ev - 10.75323019821 * pow(logT_ev, 2) +
               3.058038757198 * pow(logT_ev, 3) - 0.5685118909884001 * pow(logT_ev, 4) +
               0.06795391233790001 * pow(logT_ev, 5) - 0.005009056101857001 * pow(logT_ev, 6) +
               0.0002067236157507 * pow(logT_ev, 7) - 3.649161410833e-6 * pow(logT_ev, 8)) /
           units;
  } else {
    return tiny;
  }
}

// Calculation of k4 (HeII + e --> HeI + photon)
//  k4_rate
__device__ Real recomb_HeII_rate(Real T, Real units, bool use_case_B)
{
  Real T_ev    = T / 11605.0;
  Real logT_ev = log(T_ev);
  // If case B recombination on.
  if (use_case_B) {
    return 1.26e-14 * pow(5.7067e5 / T, 0.75) / units;
  }

  // If case B recombination off.
  if (T_ev > 0.8) {
    return (1.54e-9 * (1.0 + 0.3 / exp(8.099328789667 / T_ev)) / (exp(40.49664394833662 / T_ev) * pow(T_ev, 1.5)) +
            3.92e-13 / pow(T_ev, 0.6353)) /
           units;
  } else {
    return 3.92e-13 / pow(T_ev, 0.6353) / units;
  }
}
// k4_rate Case A
__device__ Real recomb_HeII_rate_case_A(Real T, Real units)
{
  Real T_ev    = T / 11605.0;
  Real logT_ev = log(T_ev);
  if (T_ev > 0.8) {
    return (1.54e-9 * (1.0 + 0.3 / exp(8.099328789667 / T_ev)) / (exp(40.49664394833662 / T_ev) * pow(T_ev, 1.5)) +
            3.92e-13 / pow(T_ev, 0.6353)) /
           units;
  } else {
    return 3.92e-13 / pow(T_ev, 0.6353) / units;
  }
}
// k4_rate Case B
__device__ Real recomb_HeII_rate_case_B(Real T, Real units)
{
  // If case B recombination on.
  return 1.26e-14 * pow(5.7067e5 / T, 0.75) / units;
}

// Calculation of k2 (HII + e --> HI + photon)
//  k2_rate
__device__ Real recomb_HII_rate(Real T, Real units, bool use_case_B)
{
  if (use_case_B) {
    if (T < 1.0e9) {
      return 4.881357e-6 * pow(T, -1.5) * pow((1.0 + 1.14813e2 * pow(T, -0.407)), -2.242) / units;
    } else {
      return tiny;
    }
  } else {
    if (T > 5500) {
      // Convert temperature to appropriate form.
      Real T_ev    = T / tevk;
      Real logT_ev = log(T_ev);

      return exp(-28.61303380689232 - 0.7241125657826851 * logT_ev - 0.02026044731984691 * pow(logT_ev, 2) -
                 0.002380861877349834 * pow(logT_ev, 3) - 0.0003212605213188796 * pow(logT_ev, 4) -
                 0.00001421502914054107 * pow(logT_ev, 5) + 4.989108920299513e-6 * pow(logT_ev, 6) +
                 5.755614137575758e-7 * pow(logT_ev, 7) - 1.856767039775261e-8 * pow(logT_ev, 8) -
                 3.071135243196595e-9 * pow(logT_ev, 9)) /
             units;
    } else {
      return recomb_HeII_rate(T, units, use_case_B);
    }
  }
}
// k2_rate Case A
__device__ Real recomb_HII_rate_case_A(Real T, Real units)
{
  if (T > 5500) {
    // Convert temperature to appropriate form.
    Real T_ev    = T / tevk;
    Real logT_ev = log(T_ev);

    return exp(-28.61303380689232 - 0.7241125657826851 * logT_ev - 0.02026044731984691 * pow(logT_ev, 2) -
               0.002380861877349834 * pow(logT_ev, 3) - 0.0003212605213188796 * pow(logT_ev, 4) -
               0.00001421502914054107 * pow(logT_ev, 5) + 4.989108920299513e-6 * pow(logT_ev, 6) +
               5.755614137575758e-7 * pow(logT_ev, 7) - 1.856767039775261e-8 * pow(logT_ev, 8) -
               3.071135243196595e-9 * pow(logT_ev, 9)) /
           units;
  } else {
    return recomb_HeII_rate_case_A(T, units);
  }
}

// k2_rate Case B
__device__ Real recomb_HII_rate_case_B(Real T, Real units)
{
    if (T < 1.0e9) {
        auto ret = 4.881357e-6*pow(T, -1.5) \
            * pow((1.0 + 1.14813e2*pow(T, -0.407)), -2.242);
        return ret/units;
    } else {
        return tiny;
    }  
}


__device__ Real recomb_HII_rate_case_Iliev1( Real T, Real units )
{
    return 2.59e-13 / units;
}


// Calculation of k5 (HeII + e --> HeIII + 2e)
//  k5_rate
__device__ Real coll_i_HeII_rate(Real T, Real units)
{
  Real T_ev    = T / 11605.0;
  Real logT_ev = log(T_ev);

  Real k5;
  if (T_ev > 0.8) {
    k5 = exp(-68.71040990212001 + 43.93347632635 * logT_ev - 18.48066993568 * pow(logT_ev, 2) +
             4.701626486759002 * pow(logT_ev, 3) - 0.7692466334492 * pow(logT_ev, 4) +
             0.08113042097303 * pow(logT_ev, 5) - 0.005324020628287001 * pow(logT_ev, 6) +
             0.0001975705312221 * pow(logT_ev, 7) - 3.165581065665e-6 * pow(logT_ev, 8)) /
         units;
  } else {
    k5 = tiny;
  }
  return k5;
}

// Calculation of k6 (HeIII + e --> HeII + photon)
//  k6_rate
__device__ Real recomb_HeIII_rate(Real T, Real units, bool use_case_B)
{
  Real k6;
  // Has case B recombination setting.
  if (use_case_B) {
    if (T < 1.0e9) {
      k6 = 7.8155e-5 * pow(T, -1.5) * pow((1.0 + 2.0189e2 * pow(T, -0.407)), -2.242) / units;
    } else {
      k6 = tiny;
    }
  } else {
    k6 = 3.36e-10 / sqrt(T) / pow(T / 1.0e3, 0.2) / (1.0 + pow(T / 1.0e6, 0.7)) / units;
  }
  return k6;
}
// k6_rate Case A
__device__ Real recomb_HeIII_rate_case_A(Real T, Real units)
{
  Real k6;
  // Has case B recombination setting.
  k6 = 3.36e-10 / sqrt(T) / pow(T / 1.0e3, 0.2) / (1.0 + pow(T / 1.0e6, 0.7)) / units;
  return k6;
}
// k6_rate Case B
__device__ Real recomb_HeIII_rate_case_B(Real T, Real units)
{
  Real k6;
  // Has case B recombination setting.
  if (T < 1.0e9) {
    k6 = 7.8155e-5 * pow(T, -1.5) * pow((1.0 + 2.0189e2 * pow(T, -0.407)), -2.242) / units;
  } else {
    k6 = tiny;
  }
  return k6;
}

// Calculation of k57 (HI + HI --> HII + HI + e)
//  k57_rate
__device__ Real coll_i_HI_HI_rate(Real T, Real units)
{
  // These rate coefficients are from Lenzuni, Chernoff & Salpeter (1991).
  // k57 value based on experimental cross-sections from Gealy & van Zyl (1987).
  if (T > 3.0e3) {
    return 1.2e-17 * pow(T, 1.2) * exp(-1.578e5 / T) / units;
  } else {
    return tiny;
  }
}

// Calculation of k58 (HI + HeI --> HII + HeI + e)
//  k58_rate
__device__ Real coll_i_HI_HeI_rate(Real T, Real units)
{
  // These rate coefficients are from Lenzuni, Chernoff & Salpeter (1991).
  // k58 value based on cross-sections from van Zyl, Le & Amme (1981).
  if (T > 3.0e3) {
    return 1.75e-17 * pow(T, 1.3) * exp(-1.578e5 / T) / units;
  } else {
    return tiny;
  }
}

// Calculation of ceHI.
//  Cooling collisional excitation HI
__host__ __device__ Real cool_ceHI_rate(Real T, Real units)
{
  return 7.5e-19 * exp(-fmin(log(dhuge), 118348.0 / T)) / (1.0 + sqrt(T / 1.0e5)) / units;
}

// Calculation of ceHeI.
//  Cooling collisional ionization HeI
__host__ __device__ Real cool_ceHeI_rate(Real T, Real units)
{
  return 9.1e-27 * exp(-fmin(log(dhuge), 13179.0 / T)) * pow(T, -0.1687) / (1.0 + sqrt(T / 1.0e5)) / units;
}

// Calculation of ceHeII.
//  Cooling collisional excitation HeII
__host__ __device__ Real cool_ceHeII_rate(Real T, Real units)
{
  return 5.54e-17 * exp(-fmin(log(dhuge), 473638.0 / T)) * pow(T, -0.3970) / (1.0 + sqrt(T / 1.0e5)) / units;
}

// Calculation of ciHeIS.
//  Cooling collisional ionization HeIS
__host__ __device__ Real cool_ciHeIS_rate(Real T, Real units)
{
  return 5.01e-27 * pow(T, -0.1687) / (1.0 + sqrt(T / 1.0e5)) * exp(-fmin(log(dhuge), 55338.0 / T)) / units;
}

// Calculation of ciHI.
//  Cooling collisional ionization HI
__host__ __device__ Real cool_ciHI_rate(Real T, Real units)
{
  // Collisional ionization. Polynomial fit from Tom Abel.
  return 2.18e-11 * coll_i_HI_rate(T, 1) / units;
}

// Calculation of ciHeI.
//  Cooling collisional ionization HeI
__host__ __device__ Real cool_ciHeI_rate(Real T, Real units)
{
  // Collisional ionization. Polynomial fit from Tom Abel.
  return 3.94e-11 * coll_i_HeI_rate(T, 1) / units;
}

// Calculation of ciHeII.
//  Cooling collisional ionization HeII
__host__ __device__ Real cool_ciHeII_rate(Real T, Real units)
{
  // Collisional ionization. Polynomial fit from Tom Abel.
  return 8.72e-11 * coll_i_HeII_rate(T, 1) / units;
}

// Calculation of reHII.
//  Cooling recombination HII
__host__ __device__ Real cool_reHII_rate(Real T, Real units, bool use_case_B)
{
  Real lambdaHI = 2.0 * 157807.0 / T;
  if (use_case_B) {
    return 3.435e-30 * T * pow(lambdaHI, 1.970) / pow(1.0 + pow(lambdaHI / 2.25, 0.376), 3.720) / units;
  } else {
    return 1.778e-29 * T * pow(lambdaHI, 1.965) / pow(1.0 + pow(lambdaHI / 0.541, 0.502), 2.697) / units;
  }
}

// Calculation of reHII.
//  Cooling recombination HII Case A
__host__ __device__ Real cool_reHII_rate_case_A(Real T, Real units)
{
  Real lambdaHI = 2.0 * 157807.0 / T;
  return 1.778e-29 * T * pow(lambdaHI, 1.965) / pow(1.0 + pow(lambdaHI / 0.541, 0.502), 2.697) / units;
}

// Calculation of reHII.
//  Cooling recombination HII Case B
__host__ __device__ Real cool_reHII_rate_case_B(Real T, Real units)
{
  Real lambdaHI = 2.0 * 157807.0 / T;
  return 3.435e-30 * T * pow(lambdaHI, 1.970) / pow(1.0 + pow(lambdaHI / 2.25, 0.376), 3.720) / units;
}

// Calculation of reHII.
//  Cooling recombination HeII
__host__ __device__ Real cool_reHeII1_rate(Real T, Real units, bool use_case_B)
{
  Real lambdaHeII = 2.0 * 285335.0 / T;
  if (use_case_B) {
    return 1.26e-14 * kboltz * T * pow(lambdaHeII, 0.75) / units;
  } else {
    return 3e-14 * kboltz * T * pow(lambdaHeII, 0.654) / units;
  }
}

// Calculation of reHII.
//  Cooling recombination HeII Case A
__host__ __device__ Real cool_reHeII1_rate_case_A(Real T, Real units)
{
  Real lambdaHeII = 2.0 * 285335.0 / T;
  return 3e-14 * kboltz * T * pow(lambdaHeII, 0.654) / units;
}

// Calculation of reHII.
//  Cooling recombination HeII Case B
__host__ __device__ Real cool_reHeII1_rate_case_B(Real T, Real units)
{
  Real lambdaHeII = 2.0 * 285335.0 / T;
  return 1.26e-14 * kboltz * T * pow(lambdaHeII, 0.75) / units;
}

// Calculation of reHII2.
//  Cooling recombination HeII Dielectronic
__host__ __device__ Real cool_reHeII2_rate(Real T, Real units)
{
  // Dielectronic recombination (Cen, 1992).
  return 1.24e-13 * pow(T, -1.5) * exp(-fmin(log(dhuge), 470000.0 / T)) *
         (1.0 + 0.3 * exp(-fmin(log(dhuge), 94000.0 / T))) / units;
}

// Calculation of reHIII.
//  Cooling recombination HeIII
__host__ __device__ Real cool_reHeIII_rate(Real T, Real units, bool use_case_B)
{
  Real lambdaHeIII = 2.0 * 631515.0 / T;
  if (use_case_B) {
    return 8.0 * 3.435e-30 * T * pow(lambdaHeIII, 1.970) / pow(1.0 + pow(lambdaHeIII / 2.25, 0.376), 3.720) / units;
  } else {
    return 8.0 * 1.778e-29 * T * pow(lambdaHeIII, 1.965) / pow(1.0 + pow(lambdaHeIII / 0.541, 0.502), 2.697) / units;
  }
}

// Calculation of reHIII.
//  Cooling recombination HeIII Case A
__host__ __device__ Real cool_reHeIII_rate_case_A(Real T, Real units)
{
  Real lambdaHeIII = 2.0 * 631515.0 / T;
  return 8.0 * 1.778e-29 * T * pow(lambdaHeIII, 1.965) / pow(1.0 + pow(lambdaHeIII / 0.541, 0.502), 2.697) / units;
}

// Calculation of reHIII.
//  Cooling recombination HeIII Case B
__host__ __device__ Real cool_reHeIII_rate_case_B(Real T, Real units)
{
  Real lambdaHeIII = 2.0 * 631515.0 / T;
  return 8.0 * 3.435e-30 * T * pow(lambdaHeIII, 1.970) / pow(1.0 + pow(lambdaHeIII / 2.25, 0.376), 3.720) / units;
}
// Calculation of brem.
//  Cooling Bremsstrahlung
__host__ __device__ Real cool_brem_rate(Real T, Real units)
{
  return 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-pow(5.5 - log10(T), 2) / 3.0)) / units;
}

#endif

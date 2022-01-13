#ifdef CHEMISTRY_GPU

#include "chemistry_gpu.h"
#include"../global_cuda.h"



// Calculation of k1 (HI + e --> HII + 2e)
// k1_rate
__device__ double coll_i_HI_rate(double T, double units );

//Calculation of k3 (HeI + e --> HeII + 2e)
// k3_rate
__device__ double coll_i_HeI_rate(double T, double units );

//Calculation of k4 (HeII + e --> HeI + photon)
// k4_rate
__device__ double recomb_HeII_rate(double T, double units, bool use_case_B );

//Calculation of k2 (HII + e --> HI + photon)
// k2_rate
__device__ double recomb_HII_rate(double T, double units, bool use_case_B );

//Calculation of k5 (HeII + e --> HeIII + 2e)
// k5_rate
__device__ double coll_i_HeII_rate(double T, double units );

//Calculation of k6 (HeIII + e --> HeII + photon)
// k6_rate
__device__ double recomb_HeIII_rate(double T, double units, bool use_case_B );

//Calculation of k57 (HI + HI --> HII + HI + e)
// k57_rate
__device__ double coll_i_HI_HI_rate(double T, double units );

//Calculation of k58 (HI + HeI --> HII + HeI + e)
// k58_rate
__device__ double coll_i_HI_HeI_rate(double T, double units );

//Calculation of ceHI.
// Cooling collisional excitation HI
__device__ double cool_ceHI_rate(double T, double units );

//Calculation of ceHeI.
// Cooling collisional ionization HeI
__device__ double cool_ceHeI_rate(double T, double units );

//Calculation of ceHeII.
// Cooling collisional excitation HeII
__device__ double cool_ceHeII_rate(double T, double units );

//Calculation of ciHeIS.
// Cooling collisional ionization HeIS
__device__ double cool_ciHeIS_rate(double T, double units );

//Calculation of ciHI.
// Cooling collisional ionization HI
__device__ double cool_ciHI_rate(double T, double units );


//Calculation of ciHeI.
// Cooling collisional ionization HeI
__device__ double cool_ciHeI_rate(double T, double units );

//Calculation of ciHeII.
// Cooling collisional ionization HeII
__device__ double cool_ciHeII_rate(double T, double units );


//Calculation of reHII.
// Cooling recombination HII
__device__ double cool_reHII_rate(double T, double units, bool use_case_B );


//Calculation of reHII.
// Cooling recombination HeII
__device__ double cool_reHeII1_rate(double T, double units, bool use_case_B );

//Calculation of reHII2.
// Cooling recombination HeII Dielectronic
__device__ double cool_reHeII2_rate(double T, double units );

//Calculation of reHIII.
// Cooling recombination HeIII
__device__ double cool_reHeIII_rate(double T, double units, bool use_case_B );

//Calculation of brem.
// Cooling Bremsstrahlung
__device__ double cool_brem_rate(double T, double units );


// Colisional excitation of neutral hydrogen (HI) and singly ionized helium (HeII)
Real __device__ Collisional_Ionization_Rate_e_HI_Abel97( Real temp );

Real __device__ Recombination_Rate_HII_Abel97( Real temp );

Real __device__ Collisional_Ionization_Rate_e_HeI_Abel97( Real temp );
  
Real __device__ Collisional_Ionization_Rate_e_HeII_Abel97( Real temp );

Real __device__ Collisional_Ionization_Rate_HI_HI_Lenzuni91( Real temp );

Real __device__ Collisional_Ionization_Rate_HII_HI_Lenzuni91( Real temp );

Real __device__ Collisional_Ionization_Rate_HeI_HI_Lenzuni91( Real temp );

Real __device__ Recombination_Rate_HII_Hui97( Real temp );

Real __device__ Recombination_Rate_HeII_Hui97( Real temp );

Real __device__ Recombination_Rate_HeIII_Hui97( Real temp );


Real __device__ Cooling_Rate_Recombination_HII_Hui97( Real n_e,  Real n_HII, Real temp );

Real __device__ Cooling_Rate_Recombination_HeII_Hui97( Real n_e, Real n_HII, Real temp );

Real __device__ Cooling_Rate_Recombination_HeIII_Hui97( Real n_e, Real n_HII, Real temp );

Real __device__ Recombination_Rate_dielectronic_HeII_Hui97( Real temp );

Real __device__ Cooling_Rate_Recombination_dielectronic_HeII_Hui97( Real n_e, Real n_HeII, Real temp );

Real __device__ Collisional_Ionization_Rate_e_HI_Hui97( Real temp );

Real __device__ Cooling_Rate_Collisional_Excitation_e_HI_Hui97( Real n_e, Real n_HI, Real temp );

Real __device__ Cooling_Rate_Collisional_Excitation_e_HeII_Hui97( Real n_e, Real n_HeII,  Real temp );

// Compton cooling off the CMB 
Real __device__ Cooling_Rate_Compton_CMB_MillesOstriker01( Real n_e, Real temp, Real z );

// Real __device__ Cooling_Rate_Compton_CMB_Peebles93( Real n_e, Real temp, Real current_z, cosmo );




#endif
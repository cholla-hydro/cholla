#ifdef CHEMISTRY_GPU

#include "chemistry_gpu.h"
#include"../global_cuda.h"

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
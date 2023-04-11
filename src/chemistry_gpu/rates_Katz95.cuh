#ifdef CHEMISTRY_GPU

  #include "../global/global_cuda.h"
  #include "chemistry_gpu.h"

// Colisional excitation of neutral hydrogen (HI) and singly ionized helium
// (HeII)

Real __device__ Cooling_Rate_Collisional_Excitation_e_HI_Katz95(Real n_e, Real n_HI, Real temp);

Real __device__ Cooling_Rate_Collisional_Excitation_e_HeII_Katz95(Real n_e, Real n_HeII, Real temp);

// Colisional ionization  of HI, HeI and HeII
Real __device__ Cooling_Rate_Collisional_Ionization_e_HI_Katz95(Real n_e, Real n_HI, Real temp);

Real __device__ Cooling_Rate_Collisional_Ionization_e_HeI_Katz95(Real n_e, Real n_HeI, Real temp);

Real __device__ Cooling_Rate_Collisional_Ionization_e_HeII_Katz95(Real n_e, Real n_HeII, Real temp);

Real __device__ Collisional_Ionization_Rate_e_HI_Katz95(Real temp);

Real __device__ Collisional_Ionization_Rate_e_HeI_Katz95(Real temp);

Real __device__ Collisional_Ionization_Rate_e_HeII_Katz95(Real temp);

// Standard Recombination of HII, HeII and HeIII

Real __device__ Cooling_Rate_Recombination_HII_Katz95(Real n_e, Real n_HII, Real temp);

Real __device__ Cooling_Rate_Recombination_HeII_Katz95(Real n_e, Real n_HeII, Real temp);

Real __device__ Cooling_Rate_Recombination_HeIII_Katz95(Real n_e, Real n_HeIII, Real temp);

Real __device__ Recombination_Rate_HII_Katz95(Real temp);

Real __device__ Recombination_Rate_HeII_Katz95(Real temp);

Real __device__ Recombination_Rate_HeIII_Katz95(Real temp);

// Dielectronic recombination of HeII
Real __device__ Cooling_Rate_Recombination_dielectronic_HeII_Katz95(Real n_e, Real n_HeII, Real temp);

Real __device__ Recombination_Rate_dielectronic_HeII_Katz95(Real temp);

// Free-Free emission (Bremsstrahlung)
Real __device__ gaunt_factor(Real log10_T);

Real __device__ Cooling_Rate_Bremsstrahlung_Katz95(Real n_e, Real n_HII, Real n_HeII, Real n_HeIII, Real temp);

// Compton cooling off the CMB
Real __device__ Cooling_Rate_Compton_CMB_Katz95(Real n_e, Real temp, Real z);

#endif
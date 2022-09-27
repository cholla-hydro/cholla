#ifdef CHEMISTRY_GPU

#include "chemistry_gpu.h"
#include"../global/global_cuda.h"



// Calculation of k1 (HI + e --> HII + 2e)
// k1_rate
__host__ __device__ Real coll_i_HI_rate(Real T, Real units );

//Calculation of k3 (HeI + e --> HeII + 2e)
// k3_rate
__host__ __device__ Real coll_i_HeI_rate(Real T, Real units );

//Calculation of k4 (HeII + e --> HeI + photon)
// k4_rate
__host__ __device__ Real recomb_HeII_rate(Real T, Real units, bool use_case_B );
// k4_rate Case A
__host__ __device__ Real recomb_HeII_rate_case_A(Real T, Real units );
// k4_rate Case B
__host__ __device__ Real recomb_HeII_rate_case_B(Real T, Real units );

//Calculation of k2 (HII + e --> HI + photon)
// k2_rate
__host__ __device__ Real recomb_HII_rate(Real T, Real units, bool use_case_B );
// k2_rate Case A
__host__ __device__ Real recomb_HII_rate_case_A(Real T, Real units );
// k2_rate Case B
__host__ __device__ Real recomb_HII_rate_case_B(Real T, Real units );

//Calculation of k5 (HeII + e --> HeIII + 2e)
// k5_rate
__host__ __device__ Real coll_i_HeII_rate(Real T, Real units );

//Calculation of k6 (HeIII + e --> HeII + photon)
// k6_rate
__host__ __device__ Real recomb_HeIII_rate(Real T, Real units, bool use_case_B );
// k6_rate Case A
__host__ __device__ Real recomb_HeIII_rate_case_A(Real T, Real units );
// k6_rate Case B
__host__ __device__ Real recomb_HeIII_rate_case_B(Real T, Real units );

//Calculation of k57 (HI + HI --> HII + HI + e)
// k57_rate
__host__ __device__ Real coll_i_HI_HI_rate(Real T, Real units );

//Calculation of k58 (HI + HeI --> HII + HeI + e)
// k58_rate
__host__ __device__ Real coll_i_HI_HeI_rate(Real T, Real units );

//Calculation of ceHI.
// Cooling collisional excitation HI
__host__ __device__ Real cool_ceHI_rate(Real T, Real units );

//Calculation of ceHeI.
// Cooling collisional ionization HeI
__host__ __device__ Real cool_ceHeI_rate(Real T, Real units );

//Calculation of ceHeII.
// Cooling collisional excitation HeII
__host__ __device__ Real cool_ceHeII_rate(Real T, Real units );

//Calculation of ciHeIS.
// Cooling collisional ionization HeIS
__host__ __device__ Real cool_ciHeIS_rate(Real T, Real units );

//Calculation of ciHI.
// Cooling collisional ionization HI
__host__ __device__ Real cool_ciHI_rate(Real T, Real units );


//Calculation of ciHeI.
// Cooling collisional ionization HeI
__host__ __device__ Real cool_ciHeI_rate(Real T, Real units );

//Calculation of ciHeII.
// Cooling collisional ionization HeII
__host__ __device__ Real cool_ciHeII_rate(Real T, Real units );


//Calculation of reHII.
// Cooling recombination HII
__host__ __device__ Real cool_reHII_rate(Real T, Real units, bool use_case_B );
// Cooling recombination HII Case A
__host__ __device__ Real cool_reHII_rate_case_A(Real T, Real units );
// Cooling recombination HII Case B
__host__ __device__ Real cool_reHII_rate_case_B(Real T, Real units );

//Calculation of reHII.
// Cooling recombination HeII
__host__ __device__ Real cool_reHeII1_rate(Real T, Real units, bool use_case_B );
// Cooling recombination HeII Case A
__host__ __device__ Real cool_reHeII1_rate_case_A(Real T, Real units );
// Cooling recombination HeII Case B
__host__ __device__ Real cool_reHeII1_rate_case_B(Real T, Real units );

//Calculation of reHII2.
// Cooling recombination HeII Dielectronic
__host__ __device__ Real cool_reHeII2_rate(Real T, Real units );

//Calculation of reHIII.
// Cooling recombination HeIII
__host__ __device__ Real cool_reHeIII_rate(Real T, Real units, bool use_case_B );
// Cooling recombination HeIII Case A
__host__ __device__ Real cool_reHeIII_rate_case_A(Real T, Real units );
// Cooling recombination HeIII Case B
__host__ __device__ Real cool_reHeIII_rate_case_B(Real T, Real units );

//Calculation of brem.
// Cooling Bremsstrahlung
__host__ __device__ Real cool_brem_rate(Real T, Real units );

//Calculation of comp.
// Compton cooling
__host__ __device__ Real comp_rate(Real n_e, Real T, Real zr, Real units);
__host__ __device__ Real cool_compton_rate( Real T, Real units );


// X-ray compton heating
__host__ __device__ Real xray_heat_rate( Real n_e, Real T,  Real Redshift, Real units );


#endif
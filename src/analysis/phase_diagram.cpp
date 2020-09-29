#ifdef ANALYSIS

#include"analysis.h"
#include"../io.h"



void Analysis_Module::Initialize_Phase_Diagram( struct parameters *P ){
  
  n_dens = 1000;
  n_temp = 1000;
  
  phase_diagram = (float *) malloc(n_dens*n_temp*sizeof(float));
  
  chprintf(" Phase Diagram Initialized.\n");
  
  
}













#endif
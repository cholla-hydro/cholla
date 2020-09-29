#ifdef ANALYSIS

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include"../global.h"

class Analysis_Module{
public:
  
  #ifdef PHASE_DIAGRAM
  int n_dens;
  int n_temp;
  float *phase_diagram;
  #endif
  
  
  #ifdef LYA_STATISTICS
  
  #endif
  
  
  Analysis_Module( void );
  void Initialize( struct parameters *P );
  void Reset(void);
  
  #ifdef PHASE_DIAGRAM
  void Initialize_Phase_Diagram( struct parameters *P );
  #endif
  
  #ifdef LYA_STATISTICS
  void Initialize_Lya_Statistics( struct parameters *P );
  #endif
};



#endif
#endif
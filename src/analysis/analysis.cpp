#ifdef ANALYSIS

#include"analysis.h"
#include"../io.h"

Analysis_Module::Analysis_Module( void ){}

void Analysis_Module::Initialize( struct parameters *P ){
  
  chprintf( "\nInitializng Analysis Module...\n");
  
  #ifdef PHASE_DIAGRAM
  Initialize_Phase_Diagram(P);
  #endif
  
  #ifdef LYA_STATISTICS
  Initialize_Lya_Statistics(P);
  #endif
  
  
  
  chprintf( "Analysis Module Sucessfully Initialized.\n\n");
  
}

void Analysis_Module::Reset(){
  
  #ifdef PHASE_DIAGRAM
  free(phase_diagram);
  #endif
  
}



#endif
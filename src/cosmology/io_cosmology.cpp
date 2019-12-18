#ifdef COSMOLOGY

#include <iostream>
#include <fstream>
#include "cosmology.h"
#include "../io.h"

using namespace std;


void Cosmology::Load_Scale_Outputs( struct parameters *P ) {

  char filename_1[100];
  // create the filename to read from
  strcpy(filename_1, P->scale_outputs_file);
  chprintf( " Loading Scale_Factor Outpus: %s\n", filename_1);
  
  ifstream file_out ( filename_1 );
  string line;
  Real a_value;
  if (file_out.is_open()){
    while ( getline (file_out,line) ){
      a_value = atof( line.c_str() );
      scale_outputs.push_back( a_value );
      n_outputs += 1;
      // chprintf("%f\n", a_value);
    }
    file_out.close();
    n_outputs = scale_outputs.size();
    next_output_indx = 0;
    chprintf("  Loaded %d scale outputs \n", n_outputs);
  }
  else{
    chprintf("  Error: Unable to open cosmology outputs file\n");
    exit(1);
  }
  
  chprintf(" Setting next snapshot output\n");
  
  int scale_indx = next_output_indx;
  a_value = scale_outputs[scale_indx];
  
  while ( (current_a - a_value) > 1e-3  ){
    // chprintf( "%f   %f\n", a_value, current_a);
    scale_indx += 1;
    a_value = scale_outputs[scale_indx];
  }
  next_output_indx = scale_indx;
  next_output = a_value;
  chprintf("  Next output scale index: %d  \n", next_output_indx );
  chprintf("  Next output scale value: %f  \n", next_output);
}

void Cosmology::Set_Scale_Outputs( struct parameters *P ){
  
  if ( P->scale_outputs_file[0] == '\0' ){
    chprintf( " Output every %d timesteps.\n", P->n_steps_output );
    Real scale_end = 1 / ( P->End_redshift + 1);
    scale_outputs.push_back( current_a );
    scale_outputs.push_back( scale_end );  
    n_outputs = scale_outputs.size();
    next_output_indx = 0;
    next_output = current_a;
    chprintf("  Next output scale index: %d  \n", next_output_indx );
    chprintf("  Next output scale value: %f  \n", next_output);
  }
  else  Load_Scale_Outputs( P );
  

  
}


void Cosmology::Set_Next_Scale_Output(  ){

  int scale_indx = next_output_indx;
  Real a_value = scale_outputs[scale_indx];
  if  ( ( scale_indx == 0 ) && ( abs(a_value - current_a )<1e-5 ) )scale_indx = 1;
  else scale_indx += 1;
  a_value = scale_outputs[scale_indx];
  // while ( (current_a - a_value) > 1e-2  ){
  //   // chprintf( "%f   %f\n", a_value, current_a);
  //   scale_indx += 1;
  //   a_value = scale_outputs[scale_indx];
  // }
  next_output_indx = scale_indx;
  next_output = a_value;
  // chprintf( " Next output scale_factor: %f\n", next_output);
}


#endif
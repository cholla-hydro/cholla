#include "random_functions.h"


Real Rand_Real( Real min, Real max ){
  Real r = ((double) rand() / (RAND_MAX));
  r = r*( max - min ) + min;
  return r;
}

#ifdef CPU_TIME
#ifndef TIMING_FUNCTIONS_H
#define TIMING_FUNCTIONS_H

#include <vector>
#include "../global/global.h"

// Each instance of this class represents a single timer, timing a single section of code. 
// All instances have their own n_steps, time_start, etc. so that all timers can run independently
class OneTime
{
 public:
  const char* name;
  int n_steps = 0;
  Real time_start;
  Real t_min;
  Real t_max;
  Real t_avg;
  Real t_all=0;
  bool inactive=true;
  OneTime(void){
  }
  OneTime(const char* input_name){
    name = input_name;
    inactive=false;
  }
  void Start();
  void Subtract(Real time_to_subtract);
  void End();
  void PrintStep();
  void PrintAverage();
  void PrintAll();
};

// Time loops through instances of OneTime. onetimes is initialized with pointers to each timer. 
//
class Time
{
public:

  int n_steps;

  OneTime Calc_dt;
  OneTime Hydro;
  OneTime Boundaries;
  OneTime Grav_Potential;
  OneTime Pot_Boundaries;
  OneTime Part_Density;
  OneTime Part_Boundaries;
  OneTime Part_Dens_Transf;
  OneTime Advance_Part_1;
  OneTime Advance_Part_2;
  OneTime Cooling;
  OneTime Chemistry;
    
  std::vector<OneTime*> onetimes;
  
  Time();
  void Initialize();
  void Print_Times();
  void Print_Average_Times( struct parameters P );
  
};


#endif
#endif //CPU_TIME

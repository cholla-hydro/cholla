
#ifdef CPU_TIME

#ifndef TIMING_FUNCTIONS_H
#define TIMING_FUNCTIONS_H

#include <vector>
#include "../global/global.h"

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
  void End();
  void PrintStep();
  void PrintAverage();
  void PrintAll();
};


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
    
  Real time_start;
  Real time_end;
  Real time;

  std::vector<OneTime> onetimes;

  Time();
  void Initialize();
  void Print_Times();
  void Get_Average_Times();
  void Print_Average_Times( struct parameters P );

};


#endif
#endif

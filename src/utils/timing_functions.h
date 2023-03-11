#ifndef TIMING_FUNCTIONS_H
#define TIMING_FUNCTIONS_H


#include <vector>

#include "../global/global.h" // Provides Real, get_time

//#ifdef CPU_TIME
// Each instance of this class represents a single timer, timing a single
// section of code. All instances have their own n_steps, time_start, etc. so
// that all timers can run independently
class OneTime
{
 public:
  const char* name;
  int n_steps     = 0;
  Real time_start = 0;
  Real t_min      = 0;
  Real t_max      = 0;
  Real t_avg      = 0;
  Real t_all      = 0;
  bool inactive   = true;
  OneTime(void) {}
  OneTime(const char* input_name)
  {
    name     = input_name;
    inactive = false;
  }
  void Start();
  void Subtract(Real time_to_subtract);
  void End();
  void PrintStep();
  void PrintAverage();
  void PrintAll();
  void RecordTime(Real time);
};

// Time loops through instances of OneTime. onetimes is initialized with
// pointers to each timer.
//
class Time
{
 public:
  int n_steps;

  OneTime Total;
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
  OneTime Feedback;
  OneTime FeedbackAnalysis;

  std::vector<OneTime*> onetimes;

  Time();
  void Initialize();
  void Print_Times();
  void Print_Average_Times(struct parameters P);
};
//#endif  // CPU_TIME


// ScopedTimer does nothing if CPU_TIME is disabled
/* \brief ScopedTimer helps time a scope. Initialize as first variable and C++ guarantees it is destroyed last */
class ScopedTimer
{
 public:
  const char* name;
  double time_start = 0;
  
  /* \brief ScopedTimer Constructor initializes name and time */
  ScopedTimer(const char* input_name);
  
  /* \brief ScopedTimer Destructor computes dt and prints */
  ~ScopedTimer(void);

  
};





#endif // TIMING_FUNCTIONS_H

#ifdef RT

  #include <fstream>
  #include <iostream>

  #include "../radiation/radiation.h"
  #include "../io/io.h"


void Rad3D::Load_Outputs(struct Parameters *P, Real current_time)
{
  char filename_1[100];
  // create the filename to read from
  strcpy(filename_1, P->rt_outputs_file);
  chprintf(" Loading RT Outputs: %s %s\n", filename_1, P->rt_outputs_file);

  std::ifstream file_out(filename_1);
  std::string line;
  Real value;
  if (file_out.is_open()) {
    while (getline(file_out, line)) {
      value = atof(line.c_str());
      rt_outputs.push_back(value);
    }
    file_out.close();
    n_outputs        = rt_outputs.size();
    next_output_indx = 0;
    chprintf("  Loaded %d rt outputs \n", n_outputs);
  } else {
    chprintf("  Error: Unable to open RT outputs file\n");
    exit(1);
  }

  chprintf(" Setting next RT output\n");

  int scale_indx = next_output_indx;
  value        = rt_outputs[scale_indx];

  while ((current_time - value) > 1e-3) {
    // chprintf( "%f   %f\n", a_value, current_a);
    scale_indx += 1;
    value = rt_outputs[scale_indx];
  }
  next_output_indx = scale_indx;
  next_output      = value;
  chprintf("  Next RT output index: %d  \n", next_output_indx);
  #ifdef COSMOLOGY
  chprintf("  Next RT output z value: %f  \n", 1. / next_output - 1);
  #else
  chprintf("  Next RT output time: %f  \n", next_output);
  #endif // COSMOLOGY

  exit_now = false;
}

void Rad3D::Set_Outputs(struct Parameters *P, Real current_time)
{
  if (P->rt_outputs_file[0] == '\0') {
    chprintf(" Output every %d timesteps.\n", P->n_steps_output);
    Real scale_end = 1 / (P->End_redshift + 1);
    rt_outputs.push_back(current_a);
    scale_outputs.push_back(scale_end);
    n_outputs        = scale_outputs.size();
    next_output_indx = 0;
    next_output      = current_a;
    chprintf("  Next output index: %d  \n", next_output_indx);
    chprintf("  Next output z value: %f  \n", 1. / next_output - 1);
        // update to the next output time, for logarithmic stepping
        if (P.outstep_dexinc != 0) outstep *= pow(10.0, P.outstep_dexinc);

        outtime += outstep;                                // update to the next output time
        outtime = std::fmin(outtime + P.outstep, P.tout);  // get the next output time

    Real outtime = 0;   // current output time
    Real outstep = 1.0; // time between output times
    Real tout = 1.0;    // final output time
    if(P->outstep > 0) {
      outstep = P->outstep;
    }
    if(P->tout > 0) {
      tout = P->tout;
    }
    while(outtime < tout) {
      if (P.outstep_dexinc != 0) outstep *= pow(10.0, P.outstep_dexinc);
        outtime += outstep;                                // update to the next output time
      outtime = std::fmin(outtime + outstep, tout);  // get the next output time
      rt_outputs.push_back(outtime);
      n_outputs = rt_outputs.size(); // record the number of output times
    }
  } else {
    Load_Outputs(P, current_time);
  }
}

void Rad3D::Set_Next_Output()
{
  int scale_indx = next_output_indx;
  Real value   = rt_outputs[scale_indx];

  scale_indx += 1;

  if (scale_indx < n_outputs) {
    value          = rt_outputs[scale_indx];
    next_output_indx = scale_indx;
    next_output      = value;
  } else {
    exit_now = true;
  }
}

#endif
